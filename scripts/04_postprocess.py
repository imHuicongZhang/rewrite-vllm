#!/usr/bin/env python3
"""Trim then shuffle every finished (arm, prompt) job.

Two stages, both resumable, both scoped to ONE (arm, prompt) job at a time:

  1. trim     per-arm/per-prompt rule, ported verbatim from the source's Step 1.
              Only status==2 rows are touched; Llama-2 counts are recomputed only for
              rows the trim actually changed; the row count must not change.
  2. shuffle  the source's bucketed two-pass shuffle, seed 42, 500k-row output shards.
              WITHIN (arm, prompt) ONLY -- never across arms.

MULTI-NODE. Run the same command on every node that has finished generating; the work
distributes itself. Each job is guarded by a per-job lock, so two nodes never take the
SAME job -- which would be real corruption, because they would share an output directory
and a bucket temp directory and the shuffle unlinks buckets as it consumes them.
Different jobs do not collide at all: separate output directories, and tmp_root is
node-local so bucket directories cannot overlap. There are only 10 jobs, so parallelism
caps at 10 nodes.

    python scripts/04_postprocess.py                       # sweep every unclaimed job
    python scripts/04_postprocess.py --arm wrap-inspired --prompt-id p3
    python scripts/04_postprocess.py --stage trim
    python scripts/04_postprocess.py --dry-run --sample 100000   # strip rates only
    python scripts/04_postprocess.py --no-lock              # single node, skip locking
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import os                                                       # noqa: E402
import socket                                                   # noqa: E402
import time                                                     # noqa: E402
from rewrite import data as D                                   # noqa: E402
from rewrite import postprocess as PP                           # noqa: E402
from rewrite import shuffle as SH                               # noqa: E402
from rewrite.config import enumerate_jobs, get_job, load_config  # noqa: E402

HOSTNAME = socket.gethostname().split(".")[0]


def dry_run(cfg, job, sample: int) -> dict:
    """Report what the trim WOULD strip, without writing anything.

    Worth doing on one job before committing to an in-place pass over terabytes: compare
    the strip rate against the source's own figures, quoted in
    docs/SOURCE_INVENTORY.md section 9 (wiki 0.0007-0.17%, distill 0.0066-0.56%).
    """
    rule = PP.TRIM_RULES[job.prompt.trim]
    n = n_s2 = n_strip = 0
    examples = []
    for p in sorted(job.output_dir.glob(f"part_*{cfg.shard_suffix}")):
        for row in D.iter_jsonl(p):
            n += 1
            if row.get("status") == 2:
                n_s2 += 1
                new, did = rule(row.get("rewritten_text"))
                if did:
                    n_strip += 1
                    if len(examples) < 3:
                        examples.append((row.get("rewritten_text") or "", new))
            if n >= sample:
                break
        if n >= sample:
            break
    pct = 100.0 * n_strip / n_s2 if n_s2 else 0.0
    print(f"[dry-run] {job.job_id} rule={job.prompt.trim}: sampled {n:,} rows, "
          f"{n_s2:,} status==2, would strip {n_strip:,} ({pct:.4f}%)")
    for before, after in examples:
        print(f"  -- before: {before[:160]!r}")
        print(f"     after : {after[:160]!r}")
    return {"sampled": n, "status2": n_s2, "would_strip": n_strip, "pct": pct}


def job_done(cfg, job, stage: str) -> bool:
    """Has the requested work for this job already finished?

    Reuses the markers the stages already write, rather than inventing a third one.
    """
    if stage in ("shuffle", "both"):
        return (cfg.shuffled_dir(job.arm, job.prompt.id) / "_shuffle.done").exists()
    in_place = bool(cfg.data["postprocess"]["in_place"])
    dst = job.output_dir if in_place else cfg.trimmed_dir(job.arm, job.prompt.id)
    return (dst / "_trimmed.done").exists()


def job_lock(cfg, job) -> Path:
    """Per-job lock, on the SHARED filesystem so every node can see it.

    out_root, not tmp_root: tmp_root is node-local, and a lock no other node can observe
    is not a lock.
    """
    return cfg.shuffled_dir(job.arm, job.prompt.id) / ".postprocess.lock"


def write_job_summary(cfg, job, result: dict) -> None:
    """One file per job, then rebuild the merged summary from whatever is on disk.

    The old code wrote the whole of manifests/postprocess_summary.json at the end of each
    invocation, containing only the jobs THAT invocation ran. With several nodes -- or
    even one `--arm`-scoped run -- the last writer erased everyone else's entries. Per-job
    files converge no matter who finishes last or in what order.
    """
    d = cfg.repo_root / "manifests" / "postprocess"
    d.mkdir(parents=True, exist_ok=True)
    D.atomic_write_text(json.dumps(result, indent=2, default=str),
                        d / f"{job.arm}__{job.prompt.id}.json")

    merged = {}
    for f in sorted(d.glob("*.json")):
        try:
            merged[f.stem.replace("__", "/")] = json.loads(f.read_text())
        except (OSError, ValueError):
            continue

    # NOT D.atomic_write_text here. That helper writes through a fixed `<dest>.tmp`, which
    # is exactly right for the per-shard and per-arm paths it was built for -- one writer
    # per destination -- but two nodes rebuilding this one shared file collide on that
    # single temp name and the loser's os.replace fails with ENOENT. Unique temp name per
    # writer instead, and a failure here is swallowed: the per-job file above is the
    # durable record, this is a convenience rollup, and the next node to finish rebuilds
    # it correctly anyway. Losing a job's result because the rollup raced would be absurd.
    dest = cfg.repo_root / "manifests" / "postprocess_summary.json"
    tmp = dest.with_name(f"{dest.name}.{HOSTNAME}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, dest)
    except OSError as e:
        print(f"[postprocess] note: could not rebuild {dest.name} ({e}); "
              f"per-job results are safe in {d}")
        try:
            tmp.unlink()
        except OSError:
            pass


def run_one(cfg, job, args) -> dict:
    """Trim and/or shuffle one job. Assumes the caller holds its lock."""
    result = {}
    if args.stage in ("trim", "both"):
        result["trim"] = PP.trim_job(cfg, job, workers=args.workers)
    if args.stage in ("shuffle", "both"):
        result["shuffle"] = SH.shuffle_job(cfg, job)
    return result


def sweep(cfg, jobs, args) -> dict:
    """Take whatever jobs are free, repeatedly, until a whole pass takes nothing.

    Not a single pass: a node that finishes job 1 after two hours should pick up whatever
    became free in the meantime rather than exiting with eight jobs outstanding. Not an
    unbounded wait either -- 25 nodes all blocking on 10 jobs would leave most of them
    asleep for the duration. So: keep sweeping while we are still winning jobs, and stop
    when everything remaining is held by a live holder.

    If a holder then dies, its lock goes stale and re-running this command picks the job
    up -- the same recovery action as everywhere else in this repo.
    """
    stale_after = float(cfg.cluster["compute"].get("claim_stale_after_s", 1800))
    out = {}
    while True:
        took_one, held = False, []
        for job in jobs:
            if job_done(cfg, job, args.stage):
                continue

            if args.no_lock:
                print(f"\n=== postprocess {job.job_id} (unlocked) ===")
                out[job.job_id] = run_one(cfg, job, args)
                write_job_summary(cfg, job, out[job.job_id])
                took_one = True
                continue

            lock = job_lock(cfg, job)
            owner = {"host": HOSTNAME, "pid": os.getpid(), "job": job.job_id,
                     "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if not D.try_dir_lock(lock, owner, stale_after_s=stale_after,
                                  label=f"{job.job_id} postprocess"):
                who = D.read_lock_owner(lock)
                print(f"[postprocess] {job.job_id}: held by "
                      f"{who.get('host', '?')}/pid {who.get('pid', '?')} -> next job")
                held.append(job.job_id)
                continue

            print(f"\n=== postprocess {job.job_id} ===")
            try:
                with D.LockHeartbeat(lock, interval=min(60.0, stale_after / 5)):
                    out[job.job_id] = run_one(cfg, job, args)
            finally:
                # Released on the exception path too. shuffle_job raises on a row-count
                # mismatch, and a lock left behind by that would make every other node
                # wait out the full stale timeout before retrying the job.
                D.break_lock(lock)
            write_job_summary(cfg, job, out[job.job_id])
            took_one = True

        if not held:
            break
        if not took_one:
            print(f"\n[postprocess] nothing left for this node: {len(held)} job(s) are "
                  f"held by other nodes ({', '.join(held)}).")
            print("  That is the normal end of a multi-node postprocess. If one of those "
                  "nodes dies,")
            print(f"  its lock goes stale after {stale_after / 60:.0f} min -- re-run this "
                  "command and it will be taken over.")
            break

    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config-root", default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--prompt-id", default=None)
    ap.add_argument("--stage", choices=["trim", "shuffle", "both"], default="both")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report strip rates without writing")
    ap.add_argument("--sample", type=int, default=100000)
    ap.add_argument("--no-lock", action="store_true",
                    help="skip the per-job lock (single node only -- two unlocked runs "
                         "on the same job corrupt its output)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config_root)
    if args.arm and args.prompt_id:
        jobs = [get_job(cfg, args.arm, args.prompt_id)]
    elif args.arm:
        jobs = [j for j in enumerate_jobs(cfg) if j.arm == args.arm]
    else:
        jobs = enumerate_jobs(cfg)

    # A dry run writes nothing and takes no locks, so it is safe from anywhere at any
    # time -- including while another node is genuinely postprocessing that same job.
    if args.dry_run:
        for job in jobs:
            print(f"\n=== postprocess {job.job_id} ===")
            dry_run(cfg, job, args.sample)
        return 0

    out = sweep(cfg, jobs, args)

    if out:
        print(f"\n[postprocess] this node completed {len(out)} job(s): "
              f"{', '.join(sorted(out))}")
        print(f"[postprocess] summary: "
              f"{cfg.repo_root / 'manifests' / 'postprocess_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
