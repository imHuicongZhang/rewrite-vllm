"""Bucketed two-pass document shuffle, ported VERBATIM from 10_postprocess/pp_io.py.

source: pp_io.py:184-281.

`choose_buckets` and `bucketed_shuffle` below are transcriptions. Their internals are
Arrow/parquet on purpose: the per-bucket ParquetWriter temp files ARE the memory-safety
mechanism, so rewriting them for JSONL would throw away the property we are trying to
preserve. The only adaptation is the `load_fn` seam the source already provided -- here
it reads a JSONL shard into a pa.Table.

Scope: within (arm, prompt) ONLY, never across arms.

Determinism caveat, inherited from the source: pass-1 bucket ids come from ONE RNG stream
advanced over `sorted(specs)`, so identical output requires the same input file set, the
same sort order, and the same numpy version. The source pinned an environment for exactly
this reason (run_shuffle_10B_base.sh:17-18).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import data as D
from .config import Config, JobSpec, stop

GIB = 1024 ** 3
MEM_CAP_BYTES = 256 * GIB          # hard cpu-partition memory cap  (pp_io.py:30)


def choose_buckets(specs, mem_bytes=None, inflate=4.0, log=print):
    """B = max(16, ceil(2*text_bytes_est / (0.55*mem))). text_bytes_est = on-disk parquet
    bytes * inflate (zstd->Arrow in-memory). mem capped at the 256 GB cpu limit.

    source: pp_io.py:184-193, verbatim.
    """
    disk = sum(os.path.getsize(p) for p, _ in specs)
    est = disk * inflate
    mem = min(mem_bytes or MEM_CAP_BYTES, MEM_CAP_BYTES)
    B = max(16, math.ceil(2.0 * est / (0.55 * mem)))
    log(f'  shuffle sizing: on-disk={disk/GIB:.1f}GiB est_in_mem={est/GIB:.1f}GiB '
        f'(x{inflate}) mem_cap={mem/GIB:.0f}GiB -> B={B} (peak ~{2*est/B/GIB:.1f}GiB)')
    return B, disk, est


def bucketed_shuffle(specs, load_fn, out_dir, tmp_dir, seed=42, rows_per_shard=500_000,
                     mem_bytes=None, inflate=4.0, log=print):
    """Memory-bounded two-pass document-level shuffle (deterministic, seed-based).

    specs: list of (path, aux) ; load_fn(spec) -> unified pa.Table (same schema for all).
    Pass 1 scatters each input shard's rows into B on-disk bucket files by a random bucket
    id (one rng over inputs in sorted order). Pass 2 reads each bucket, shuffles within it
    (rng seeded per bucket), and writes ~rows_per_shard part_NNNNN.parquet shards with a
    global counter; a carry buffer keeps shard sizes uniform across bucket boundaries.
    Returns (total_rows, n_shards, B).

    source: pp_io.py:196-281, verbatim.
    """
    specs = sorted(specs, key=lambda s: str(s[0]))
    if not specs:
        raise RuntimeError('bucketed_shuffle: no inputs')
    out_dir = Path(out_dir); tmp_dir = Path(tmp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    B, _, _ = choose_buckets(specs, mem_bytes=mem_bytes, inflate=inflate, log=log)

    # ---- Pass 1: scatter into B bucket files ----
    rng = np.random.default_rng(seed)
    schema = None
    writers = {}
    bucket_path = {b: str(tmp_dir / f'bucket_{b:05d}.parquet') for b in range(B)}
    total_rows = 0
    try:
        for si, spec in enumerate(specs):
            t = load_fn(spec)
            if schema is None:
                schema = t.schema
            n = t.num_rows
            total_rows += n
            bk = rng.integers(0, B, size=n)
            for b in np.unique(bk):
                sub = t.filter(pa.array(bk == b))
                w = writers.get(int(b))
                if w is None:
                    w = pq.ParquetWriter(bucket_path[int(b)], schema, compression='zstd')
                    writers[int(b)] = w
                w.write_table(sub)
            del t
            if (si + 1) % 50 == 0:
                log(f'  shuffle pass1: {si+1}/{len(specs)} shards scattered')
    finally:
        for w in writers.values():
            w.close()
    log(f'  shuffle pass1 done: {total_rows:,} rows scattered into {len(writers)}/{B} buckets')

    # ---- Pass 2: gather + within-bucket shuffle -> uniform 500k shards ----
    shard_idx = 0
    leftover = None
    written = 0
    for b in range(B):
        bp = bucket_path[b]
        if not os.path.exists(bp):
            continue
        t = pq.read_table(bp)
        if t.num_rows:
            perm = np.random.default_rng([seed, b]).permutation(t.num_rows)
            t = t.take(pa.array(perm))
            if leftover is not None and leftover.num_rows:
                t = pa.concat_tables([leftover, t])
            leftover = None
            off = 0
            n = t.num_rows
            while n - off >= rows_per_shard:
                D.atomic_write_table(t.slice(off, rows_per_shard),
                                     out_dir / f'part_{shard_idx:05d}.parquet')
                written += rows_per_shard
                shard_idx += 1
                off += rows_per_shard
            leftover = t.slice(off) if off < n else None
        os.unlink(bp)
        if (b + 1) % 50 == 0:
            log(f'  shuffle pass2: {b+1}/{B} buckets gathered, {shard_idx} shards written')
    if leftover is not None and leftover.num_rows:
        D.atomic_write_table(leftover, out_dir / f'part_{shard_idx:05d}.parquet')
        written += leftover.num_rows
        shard_idx += 1
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    if written != total_rows:
        raise RuntimeError(f'bucketed_shuffle: wrote {written} != scattered {total_rows}')
    log(f'  shuffle pass2 done: {shard_idx} shards, {written:,} rows')
    return total_rows, shard_idx, B


# --------------------------------------------------------------------------- driver
def shuffle_job(cfg: Config, job: JobSpec, log=print):
    """Shuffle one (arm, prompt) job. All-or-nothing: bucket files are unlinked as they
    are consumed, so an interrupted shuffle must restart for that job. That is why
    tmp_root should be fast local scratch.
    """
    sh = cfg.data["shuffle"]
    if sh.get("scope") != "arm_prompt":
        stop("configs/data.yaml shuffle.scope must be 'arm_prompt' -- shuffling is "
             "within (arm, prompt) only, never across arms")

    in_place = bool(cfg.data["postprocess"]["in_place"])
    src_dir = job.output_dir if in_place else cfg.trimmed_dir(job.arm, job.prompt.id)
    out_dir = cfg.shuffled_dir(job.arm, job.prompt.id)
    tmp_dir = cfg.paths["tmp_root"] / "_shuffle" / job.arm / job.prompt.id

    marker = out_dir / "_shuffle.done"
    if marker.exists():
        log(f"[shuffle] {job.job_id}: already shuffled -> skip")
        return None

    keys = list(cfg.data["output"]["keys"])
    specs = [(str(p), None) for p in sorted(src_dir.glob(f"part_*{cfg.shard_suffix}"))]
    if not specs:
        stop(f"{job.job_id}: no input shards for shuffle in {src_dir}")

    log(f"[shuffle] {job.job_id}: {len(specs)} shards -> {out_dir} (seed={sh['seed']})")
    total_rows, n_shards, B = bucketed_shuffle(
        specs, lambda s: D.jsonl_to_table(s[0], keys),
        out_dir, tmp_dir,
        seed=int(sh["seed"]), rows_per_shard=int(sh["rows_per_shard"]),
        mem_bytes=int(cfg.cluster["compute"]["shuffle_mem_bytes"]),
        inflate=float(sh["inflate"]), log=log)

    expected = D.input_rows(cfg, job.arm)
    if total_rows != expected:
        raise RuntimeError(
            f"{job.job_id}: shuffled {total_rows:,} rows but the arm has {expected:,} "
            "input rows -- every prompt must rewrite the ENTIRE dataset for its arm")

    summary = {"job_id": job.job_id, "total_rows": total_rows, "n_shards": n_shards,
               "buckets": B, "seed": int(sh["seed"]),
               "rows_per_shard": int(sh["rows_per_shard"])}
    D.atomic_write_text(__import__("json").dumps(summary, indent=2), marker)
    return summary
