"""HuggingFace download, re-sharding, manifests, shard IO, and job verification.

Deliberately free of torch/vllm imports so it works on a CPU-only shell.

Why we re-shard at all: the source consumed a pre-existing 200-shard parquet layout on
JHU scratch. Here the inputs arrive from the Hub in whatever layout their authors chose,
so we impose our own. The shard is simultaneously
  * the unit of work,
  * the unit of resume, and
  * the unit of the output-rows == input-rows proof,
which is why sharding happens ONCE PER ARM and is reused by all of that arm's prompts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .config import Config, JobSpec, stop

SHARD_RE = re.compile(r"part_(\d+)\.parquet$")
MANIFEST_NAME = "manifest.json"
DOC_ID_POLICY = "v1:sorted-files,row-order,0-based-int64"


# --------------------------------------------------------------------------- atomic IO
def atomic_write_bytes(data: bytes, dest: Path) -> None:
    """Write bytes atomically: .tmp in the SAME dir, then os.replace.

    On any exception the partial .tmp is removed so it can never be mistaken for output.
    source: 10_postprocess/pp_io.py:34-47
    """
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_text(text: str, dest: Path) -> None:
    atomic_write_bytes(text.encode("utf-8"), dest)


def atomic_write_table(table, dest: Path, compression: str = "zstd") -> None:
    """source: 10_postprocess/pp_io.py:34-47, ported verbatim."""
    dest = str(dest)
    tmp = dest + ".tmp"
    try:
        pq.write_table(table, tmp, compression=compression)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def sha1_text(s) -> str:
    """SHA-1 of the source document, UTF-8. `None` is treated as '' -- matching the
    source's `doc_text = doc_text or ""` (rewrite_worker.py:47)."""
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- manifest
@dataclass
class Manifest:
    arm: str
    repo_id: str        # the single gated repo
    subdir: str         # this arm's folder inside it
    revision: str
    fingerprint: str
    total_rows: int
    total_text_bytes: int
    content_sha1: str
    n_shards: int
    shards: list
    doc_id_policy: str = DOC_ID_POLICY
    total_tokens_llama2: int | None = None
    doc_id_source: str = "unknown"      # "dataset" | "synthesized"

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @staticmethod
    def from_json(s: str) -> "Manifest":
        return Manifest(**json.loads(s))

    def shard_rows(self, index: int) -> int:
        for s in self.shards:
            if s["index"] == index:
                return s["n_rows"]
        stop(f"{self.arm}: manifest has no shard {index}")


def manifest_path(cfg: Config, arm: str) -> Path:
    return cfg.shards_dir(arm) / MANIFEST_NAME


def load_manifest(cfg: Config, arm: str) -> Manifest:
    p = manifest_path(cfg, arm)
    if not p.exists():
        stop(f"{arm}: no manifest at {p}. Run scripts/02_download_data.py first.")
    return Manifest.from_json(p.read_text())


def input_rows(cfg: Config, arm: str) -> int:
    """O(1) -- the number every (arm, prompt) output must match."""
    return load_manifest(cfg, arm).total_rows


def compute_fingerprint(cfg: Config, content_sha1: str, doc_id_source: str) -> str:
    """Interlock against re-sharding under a finished run.

    Re-sharding renumbers doc_id and silently invalidates every .done marker downstream,
    so the fingerprint folds in the sharding parameters as well as the content.

    `doc_id_source` is folded in as of round 4. It had been left out, which was a hole:
    flipping sharding.require_doc_id switches doc_id between the dataset's own column and
    a synthesized row index -- i.e. it renumbers every row -- yet produced an identical
    fingerprint, so a resume across that flip would have silently matched .done markers
    written against different doc_ids. The rule is that the fingerprint covers everything
    that would renumber rows.
    """
    sh = cfg.data["sharding"]
    blob = "|".join([content_sha1, str(sh["shard_target_rows"]),
                     str(sh["shard_target_bytes"]), DOC_ID_POLICY, doc_id_source])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- shards
def shard_paths(cfg: Config, arm: str) -> list:
    out = []
    for p in sorted(cfg.shards_dir(arm).glob("part_*.parquet")):
        m = SHARD_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return out


def owned_shards(shards: list, worker_id: int, num_workers: int) -> list:
    """Modulo assignment, exactly as the source's SLURM array did it.
    source: 07_rewrite/rewrite_worker.py:197-203
    """
    return [(si, p) for si, p in shards if si % num_workers == worker_id]


def read_shard(path: Path):
    return pq.read_table(str(path), use_threads=False)


# --------------------------------------------------------------------------- sidecars
def sidecar_path(out_path: Path) -> Path:
    """`part_00012.jsonl.zst` -> `part_00012.done` (suffixes stripped, not replaced)."""
    name = Path(out_path).name
    for suf in (".jsonl.zst", ".jsonl", ".parquet"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return Path(out_path).with_name(name + ".done")


def write_sidecar(out_path: Path, payload: dict) -> None:
    """MUST be called AFTER the data file's os.replace.

    Ordering is the whole safety argument: a crash between the two leaves a complete data
    file with no marker, which is treated as NOT done and is regenerated. Wasteful by at
    most one shard per worker, never wrong. The reverse ordering would be unsafe.
    """
    atomic_write_text(json.dumps(payload, indent=2), sidecar_path(out_path))


def read_sidecar(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def done_shards(job_output_dir: Path) -> dict:
    out = {}
    for p in sorted(Path(job_output_dir).glob("part_*.done")):
        d = read_sidecar(p)
        if d is not None and "shard_index" in d:
            out[int(d["shard_index"])] = d
    return out


def clean_stale_tmp(job_output_dir: Path, owned: list, suffix: str) -> int:
    """Remove this worker's own leftover .tmp files before starting."""
    n = 0
    d = Path(job_output_dir)
    for si, _ in owned:
        for cand in (d / f"part_{si:05d}{suffix}.tmp", d / f"part_{si:05d}.done.tmp"):
            if cand.exists():
                try:
                    cand.unlink()
                    n += 1
                except OSError:
                    pass
    return n


# --------------------------------------------------------------------------- claims
# Dynamic shard claiming, for heterogeneous GPU fleets.
#
# The source assigned shards statically: shard_index % num_workers == worker_id. That is
# correct and perfectly balanced when every worker owns an identical GPU, which was true
# on its homogeneous H100 cluster. It is NOT true on a mixed H200 / B200 / B300 fleet: a
# Blackwell can be several times a Hopper on this workload, so equal shard counts leave the
# fast cards idling while the slow ones finish. Wall clock is the slowest worker.
#
# Claiming instead lets each worker take the next unclaimed shard when it is free, so the
# fleet self-balances whatever the mix. Which worker does which shard does not change what
# is generated for a document -- generation is per-document and independent -- but WHICH
# GPU generated it does affect the text slightly (see docs/HANDOFF_REVIEW.md), so every
# sidecar records the GPU that produced it.
#
# A claim is an atomically created DIRECTORY. os.mkdir is atomic on every POSIX filesystem
# including NFS, where O_CREAT|O_EXCL is not reliably so.

def claim_path(job_output_dir: Path, si: int) -> Path:
    return Path(job_output_dir) / f"part_{si:05d}.claim"


def fs_now(directory: Path) -> float:
    """'Now', as the FILESYSTEM sees it.

    Claim ages are compared across nodes, and mtimes are stamped by whichever node wrote
    them. Comparing those against a local time.time() would be at the mercy of clock skew
    between nodes -- which on a two-week run is exactly the sort of thing that silently
    causes a live claim to look stale. So we stamp a probe file on the same filesystem and
    read its mtime back: both timestamps then come from the same clock.
    """
    d = Path(directory)
    probe = d / f".now.{os.getpid()}"
    try:
        probe.touch()
        return probe.stat().st_mtime
    except OSError:
        return time.time()          # last resort; better than refusing to reap at all
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


def claim_age_s(job_output_dir: Path, si: int, now: float | None = None) -> float:
    """Seconds since this claim was last heartbeated. inf if there is no claim."""
    cp = claim_path(job_output_dir, si)
    try:
        mt = cp.stat().st_mtime
    except OSError:
        return float("inf")
    if now is None:
        now = fs_now(job_output_dir)
    return max(0.0, now - mt)


def try_claim(job_output_dir: Path, si: int, owner: dict) -> bool:
    """Atomically claim shard `si`. True if this worker got it.

    os.mkdir is atomic on every POSIX filesystem INCLUDING NFS, where O_CREAT|O_EXCL is
    not reliably so. That is what makes claiming safe across nodes as well as within one.
    """
    cp = claim_path(job_output_dir, si)
    try:
        cp.mkdir()                      # atomic: exactly one worker wins
    except FileExistsError:
        return False
    try:
        atomic_write_text(json.dumps(owner, indent=2), cp / "owner.json")
    except OSError:
        pass
    return True


def heartbeat_claim(job_output_dir: Path, si: int) -> None:
    """Refresh a claim's mtime so other nodes can see the owner is still alive."""
    try:
        os.utime(claim_path(job_output_dir, si), None)
    except OSError:
        pass


class ClaimHeartbeat:
    """Touch a claim while its shard is being generated.

    Without this there is no way for another node to tell a live claim from one orphaned
    by a crash: the worker is blocked inside llm.generate() for the whole shard and cannot
    update anything from the main thread. A daemon thread ticking every `interval` seconds
    gives every claim a liveness signal that is visible on the shared filesystem.
    """

    def __init__(self, job_output_dir: Path, si: int, interval: float = 60.0):
        self.dir, self.si, self.interval = Path(job_output_dir), si, interval
        self._stop = threading.Event()
        self._th = None

    def __enter__(self):
        def tick():
            while not self._stop.wait(self.interval):
                heartbeat_claim(self.dir, self.si)
        self._th = threading.Thread(target=tick, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)
        return False


def release_claim(job_output_dir: Path, si: int) -> None:
    """Drop a claim we could not complete, so another worker can pick the shard up."""
    cp = claim_path(job_output_dir, si)
    try:
        (cp / "owner.json").unlink()
    except OSError:
        pass
    try:
        cp.rmdir()
    except OSError:
        pass


def reap_stale_claims(job_output_dir: Path, stale_after_s: float | None = 1800.0,
                      force: bool = False, log=print) -> int:
    """Remove claims that no live worker can still be holding.

    MULTI-NODE SAFETY. An earlier version removed EVERY claim it found, on the reasoning
    that the launcher holds the job lock and no worker of "this run" has started yet. That
    reasoning holds on one node and is false on twelve: node 2's launcher would wipe node
    1's live claims, two workers would generate the same shard, and because each shard is
    written atomically with last-writer-wins, row conservation would still pass. The waste
    and the nondeterministic choice of surviving output would both be silent.

    So reaping is now safe by construction rather than by instruction:
      * a claim whose shard already has a .done is litter and always goes;
      * any other claim goes only if it has not been heartbeated for `stale_after_s`,
        which a live worker refreshes every 60s (see ClaimHeartbeat).

    force=True restores the old remove-everything behaviour. It is for an operator who
    KNOWS nothing is running, and it is never used by the launcher.
    """
    d = Path(job_output_dir)
    if not d.exists():
        return 0
    now = fs_now(d)
    n = kept = 0
    for cp in sorted(d.glob("part_*.claim")):
        try:
            si = int(cp.name.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        finished = any(d.glob(f"part_{si:05d}.done"))
        if not (force or finished):
            age = max(0.0, now - cp.stat().st_mtime) if cp.exists() else float("inf")
            if stale_after_s is not None and age < stale_after_s:
                kept += 1
                continue                      # a live worker is holding this
        try:
            (cp / "owner.json").unlink()
        except OSError:
            pass
        try:
            cp.rmdir()
            n += 1
        except OSError:
            pass
    if n or kept:
        log(f"[claims] reaped {n} stale/finished claim(s); left {kept} live claim(s) "
            f"alone in {d}")
    return n


# --------------------------------------------------------------------------- download
def _hf_kwargs(cfg: Config, write: bool = False):
    tok = cfg.env.get("HF_TOKEN_WRITE" if write else "HF_TOKEN") or None
    return {"token": tok} if tok else {}


def download_arm(cfg: Config, arm_name: str, log=print) -> None:
    """Snapshot ONE arm's folder out of the single gated repo. Idempotent and resumable.

    allow_patterns restricts the transfer to this arm's subdirectory. Without it every arm
    would pull all 662 GB of the repo -- including shared-core and the four other arms --
    five times over.
    """
    from huggingface_hub import snapshot_download
    a = cfg.arm(arm_name)
    pattern = f"{a.subdir}/*"
    log(f"[data] {a.name}: snapshot_download({cfg.hf_repo_id}, "
        f"revision={cfg.hf_revision[:12]}, allow_patterns={pattern!r})")
    snapshot_download(repo_id=cfg.hf_repo_id, repo_type="dataset",
                      revision=cfg.hf_revision, allow_patterns=[pattern],
                      cache_dir=str(cfg.paths["hf_cache"]), **_hf_kwargs(cfg))
    log(f"[data] {a.name}: download complete")


def _open_dataset(cfg: Config, a, streaming: bool = False):
    """Open ONE arm by explicit data_files glob.

    NOT by HF config name: the Hub card declares no `configs:` block, so the auto-converter
    exposes a single config `default`/`train` globbing every folder in the repo. Passing
    the bare repo id here would silently concatenate shared-core and all five arms, and the
    row-conservation proof would then be checked against the wrong denominator.
    See docs/DESIGN_DELTA.md section 4.
    """
    from datasets import load_dataset
    return load_dataset("parquet",
                        data_files={"train": f"hf://datasets/{cfg.hf_repo_id}@"
                                             f"{cfg.hf_revision}/{cfg.hf_data_files(a.name)}"},
                        split="train", streaming=streaming,
                        cache_dir=str(cfg.paths["hf_cache"]), **_hf_kwargs(cfg))


def probe_arm(cfg: Config, arm_name: str, log=print) -> dict:
    """Assert the `text` column exists and is a non-empty string. Cheap: streams 1 row."""
    a = cfg.arm(arm_name)
    col = cfg.text_column
    ds = _open_dataset(cfg, a, streaming=True)
    row = next(iter(ds), None)
    if row is None:
        stop(f"{a.name}: {cfg.hf_repo_id}/{a.subdir} appears to be empty")
    if col not in row:
        stop(f"{a.name}: {cfg.hf_repo_id}/{a.subdir} has no {col!r} column; "
             f"columns are {sorted(row)}")
    if not isinstance(row[col], str):
        stop(f"{a.name}: column {col!r} is {type(row[col]).__name__}, expected str")
    if not row[col].strip():
        stop(f"{a.name}: first row's {col!r} is empty")
    log(f"[data] {a.name}: '{col}' column OK (columns: {sorted(row)})")
    return {"columns": sorted(row)}


# --------------------------------------------------------------------------- sharding
def shard_arm(cfg: Config, arm_name: str, log=print, count_tokens: bool = False) -> Manifest:
    """Re-shard one arm into fixed-size part_%05d.parquet chunks and write its manifest.

    A shard closes when EITHER shard_target_rows or shard_target_bytes is reached, so one
    pathological run of very long documents cannot become a multi-hour work unit.

    Control arms (rewrite: false) are verified and counted but never sharded.
    """
    a = cfg.arm(arm_name)
    sh = cfg.data["sharding"]
    col = cfg.text_column
    target_rows = int(sh["shard_target_rows"])
    target_bytes = int(sh["shard_target_bytes"])
    out_dir = cfg.shards_dir(a.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = manifest_path(cfg, a.name)
    if existing.exists():
        man = Manifest.from_json(existing.read_text())
        log(f"[data] {a.name}: manifest exists ({man.total_rows:,} rows, "
            f"{man.n_shards} shards) -> skip")
        return man

    # Multi-node: several nodes may reach this at once on a fresh data_root. Exactly one
    # should shard; the rest wait for the manifest rather than racing to write the same
    # files. Atomic mkdir again, for the same reason claims use it.
    lock = out_dir / ".sharding.lock"
    try:
        lock.mkdir()
        holder = True
    except FileExistsError:
        holder = False
    if not holder:
        log(f"[data] {a.name}: another process is sharding this arm; waiting for its "
            f"manifest ...")
        waited = 0
        while not existing.exists():
            time.sleep(10)
            waited += 10
            if waited % 600 == 0:
                log(f"[data] {a.name}: still waiting ({waited // 60} min). If no other "
                    f"process is running, remove {lock} and re-run.")
        log(f"[data] {a.name}: manifest appeared -> skip")
        return Manifest.from_json(existing.read_text())

    try:
        ds = _open_dataset(cfg, a, streaming=False)
    except BaseException:
        try:
            lock.rmdir()
        except OSError:
            pass
        raise
    if col not in ds.column_names:
        stop(f"{a.name}: no {col!r} column; columns are {ds.column_names}")

    # §3: doc_id is the join key back to the input corpus. If the dataset supplies one it
    # is durable and independent of how this pipeline shards. If it does not, we synthesize
    # a row index -- reproducible, but only meaningful relative to OUR sharding, which makes
    # data_root/shards/ load-bearing forever. That difference is consequential enough to be
    # a decision rather than a silent fallback.
    has_doc_id = "doc_id" in ds.column_names
    require = bool(sh.get("require_doc_id", True))
    if not has_doc_id:
        msg = (f"{a.name}: {cfg.hf_repo_id}/{a.subdir} has NO 'doc_id' column "
               f"(columns: {ds.column_names}).\n"
               f"  doc_id is the key that joins rewritten output back to the input corpus "
               f"-- topic labels, quality scores, anything not carried in the 11-key output "
               f"schema.\n"
               f"  Without it this pipeline synthesizes a row index, which is deterministic "
               f"but only meaningful relative to this sharding: data_root/shards/ then has "
               f"to survive for the join to remain possible.\n"
               f"  FIX (cheap, and permanent): add an explicit doc_id column to the dataset "
               f"and re-upload.\n"
               f"  OVERRIDE: set sharding.require_doc_id: false in configs/data.yaml to "
               f"accept the synthesized index. That choice is recorded in the manifest.")
        if require:
            stop(msg)
        log("[data] WARNING " + "-"*58)
        for line in msg.splitlines():
            log("[data] " + line)
        log("[data] " + "-"*65)
    ltok = None
    if count_tokens:
        from .engine import load_llama2_tokenizer
        ltok = load_llama2_tokenizer(cfg)

    running = hashlib.sha1()
    total_rows = total_bytes = total_tok = 0
    shards = []
    shard_idx = 0
    buf_ids, buf_txt, buf_sha = [], [], []
    buf_bytes = 0

    def flush():
        nonlocal shard_idx, buf_ids, buf_txt, buf_sha, buf_bytes
        if not buf_txt:
            return
        tbl = pa.table({
            "doc_id": pa.array(buf_ids, type=pa.int64()),
            col: pa.array(buf_txt, type=pa.large_string()),
            "source_text_sha1": pa.array(buf_sha, type=pa.string()),
        })
        atomic_write_table(tbl, out_dir / f"part_{shard_idx:05d}.parquet")
        shards.append({"index": shard_idx, "path": f"part_{shard_idx:05d}.parquet",
                       "n_rows": len(buf_txt), "text_bytes": buf_bytes})
        shard_idx += 1
        buf_ids, buf_txt, buf_sha, buf_bytes = [], [], [], 0

    log(f"[data] {a.name}: sharding "
        f"(target {target_rows:,} rows / {target_bytes/2**20:.0f} MiB per shard)")

    for batch in ds.select_columns(
            [c for c in (col, "doc_id") if c in ds.column_names]).iter(batch_size=1000):
        texts = batch[col]
        ids = batch["doc_id"] if has_doc_id else None
        if count_tokens:
            total_tok += sum(len(x) for x in ltok(
                [t or "" for t in texts], add_special_tokens=False).input_ids)
        for k, t in enumerate(texts):
            t = t or ""
            h = sha1_text(t)
            running.update(h.encode("ascii"))
            nb = len(t.encode("utf-8"))
            buf_ids.append(int(ids[k]) if ids is not None else total_rows)
            buf_txt.append(t)
            buf_sha.append(h)
            buf_bytes += nb
            total_rows += 1
            total_bytes += nb
            if len(buf_txt) >= target_rows or buf_bytes >= target_bytes:
                flush()
        if total_rows % 1_000_000 < 1000:
            log(f"[data] {a.name}: {total_rows:,} rows, {shard_idx} shards")
    flush()

    content_sha1 = running.hexdigest()
    doc_id_source = "dataset" if has_doc_id else "synthesized"
    man = Manifest(
        arm=a.name, repo_id=cfg.hf_repo_id, subdir=a.subdir, revision=cfg.hf_revision,
        fingerprint=compute_fingerprint(cfg, content_sha1, doc_id_source),
        total_rows=total_rows, total_text_bytes=total_bytes, content_sha1=content_sha1,
        n_shards=len(shards), shards=shards,
        total_tokens_llama2=(total_tok if count_tokens else None),
        doc_id_source=doc_id_source,
    )

    # Cross-check against the count data.yaml declares. A mismatch means the pinned
    # revision is not the data these estimates were computed from -- which would make every
    # disk and wall-clock number in the run wrong, silently.
    if a.docs and total_rows != a.docs:
        stop(f"{a.name}: downloaded {total_rows:,} rows but configs/data.yaml declares "
             f"docs: {a.docs:,}.\n"
             f"  Either hf.revision ({cfg.hf_revision[:12]}) is not the revision those "
             f"numbers came from, or the arm's `docs` is stale.\n"
             f"  Every token, disk and wall-clock estimate in this run derives from "
             f"`docs` and `source_tokens_llama2`, so this is not a warning.")

    if True:
        min_ratio = int(sh["min_shards_per_gpu"])
        need = min_ratio * cfg.num_gpus
        if man.n_shards < need:
            # Do the arithmetic here rather than leaving it to be discovered: the useful
            # output is the number to use, not the fact that the current one is wrong.
            # At the shipped 5,000 rows/shard the smallest arm gives 6,677 shards, 67:1 at
            # 100 GPUs, so this should not fire -- if it does, either num_gpus is far
            # larger than planned or an arm is much smaller than data.yaml declares.
            suggested = max(200, (total_rows // need) // 100 * 100)
            stop(
                f"{a.name}: only {man.n_shards} shards for {cfg.num_gpus} GPUs "
                f"({man.n_shards / cfg.num_gpus:.1f}:1), below the required "
                f"{min_ratio}:1.\n"
                f"  A shard is the unit of work, so when shards barely outnumber workers "
                f"the tail of every job leaves most GPUs idle.\n"
                f"  This arm has {total_rows:,} rows and you have {cfg.num_gpus} GPUs, so "
                f"it needs >= {need:,} shards.\n"
                f"  SET configs/data.yaml sharding.shard_target_rows: {suggested}   "
                f"(currently {target_rows})\n"
                f"  Then delete {out_dir} and re-run 02_download_data.py.\n"
                f"  DECIDE THIS BEFORE JOB 1: shard size feeds the manifest fingerprint, so "
                f"changing it after generation starts invalidates every .done marker."
            )

    atomic_write_text(man.to_json(), manifest_path(cfg, a.name))
    try:
        lock.rmdir()
    except OSError:
        pass
    log(f"[data] {a.name}: {total_rows:,} rows, {len(shards)} shards, "
        f"{total_bytes/2**30:.1f} GiB text, doc_id={man.doc_id_source}, "
        f"content_sha1={content_sha1[:12]}")
    return man


def write_data_manifest(cfg: Config, log=print) -> Path:
    """Roll every arm's manifest into manifests/data_manifest.json."""
    out = {"arms": {}}
    for a in cfg.arms:
        man = load_manifest(cfg, a.name)
        out["arms"][a.name] = {
            "repo_id": man.repo_id, "subdir": man.subdir, "revision": man.revision,
            "total_rows": man.total_rows, "total_text_bytes": man.total_text_bytes,
            "content_sha1": man.content_sha1, "fingerprint": man.fingerprint,
            "n_shards": man.n_shards, "doc_id_source": man.doc_id_source,
            "total_tokens_llama2": man.total_tokens_llama2,
            "declared_docs": a.docs,
            "source_tokens_llama2": a.source_tokens_llama2,
            "n_prompts": len(a.prompts),
            "n_rewrite_jobs": len(a.prompts),
            "est_output_tokens": sum(pr.est_output_tokens for pr in a.prompts),
        }
    out["total_rewrite_jobs"] = sum(v["n_rewrite_jobs"] for v in out["arms"].values())
    # The raw half of the corpus, recorded so the token accounting in this file is complete
    # even though neither block is downloaded or rewritten. See docs/DESIGN_DELTA.md s3.
    out["raw_not_rewritten"] = {
        "shared-core": {"docs": 17909083, "tokens_llama2": 20000010702,
                        "note": "raw half carried into training by all five arms; "
                                "on the Hub at <repo>/shared-core, joined back on doc_id"},
        "quality-base": {"docs": 37298288, "tokens_llama2": 50000002028,
                         "note": "raw-text control; never rewritten and never uploaded"},
    }
    out["est_output_tokens_total"] = cfg.est_output_tokens_total
    out["source_tokens_llama2_total"] = sum(a.source_tokens_llama2 for a in cfg.arms)
    p = cfg.repo_root / "manifests" / "data_manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(json.dumps(out, indent=2), p)
    log(f"[data] wrote {p}")
    return p


# --------------------------------------------------------------------------- verify
@dataclass
class JobStatus:
    job_id: str
    n_shards: int
    done: int
    rows_out: int
    out_tokens: int
    state: str        # PENDING | PARTIAL | DONE
    problems: list


def verify_job(cfg: Config, job: JobSpec, deep: bool = False) -> JobStatus:
    """The output-rows == input-rows guarantee, audited cheaply.

    The guarantee itself is *created* per shard inside run_rewrite (status-0 documents
    are emitted, not dropped, so a shard always yields exactly its input row count).
    This function is the audit, and it costs only n_shards small JSON reads.

    Checks:
      1. shard-set coverage is exactly the manifest's shard set, in BOTH directions --
         this is what catches a prompt that only covered part of the corpus. Missing
         shards are named, not merely counted.
      2. per shard, n_rows_in == manifest rows == n_rows_out
      3. every sidecar's input_fingerprint matches the manifest (catches resume against
         a re-sharded input, which would have renumbered doc_id)
      4. sum(n_rows_out) == manifest.total_rows      <- the required assertion
      5. deep (opt-in): actually count the lines in every shard
    """
    man = load_manifest(cfg, job.arm)
    got = done_shards(job.output_dir)
    problems = []

    expect = {s["index"] for s in man.shards}
    missing = expect - set(got)
    extra = set(got) - expect
    if extra:
        problems.append(f"{len(extra)} output shard(s) with no matching input shard: "
                        f"{sorted(extra)[:5]}")
    if missing:
        shown = sorted(missing)[:10]
        problems.append(
            f"{len(missing)} of {len(expect)} shard(s) have no .done marker and were "
            f"never completed: {shown}"
            + (f" ... (+{len(missing) - len(shown)} more)" if len(missing) > len(shown) else "")
            + f"  -- re-run this job to generate them")

    rows_out = out_tok = 0
    for si, d in sorted(got.items()):
        rows_out += int(d.get("n_rows_out", 0))
        out_tok += int(d.get("n_output_tokens", 0))
        if d.get("input_fingerprint") != man.fingerprint:
            problems.append(f"shard {si}: sidecar fingerprint does not match the "
                            f"manifest -- the input was re-sharded after this shard ran")
        exp_rows = man.shard_rows(si)
        if int(d.get("n_rows_in", -1)) != exp_rows:
            problems.append(f"shard {si}: n_rows_in={d.get('n_rows_in')} != "
                            f"manifest {exp_rows}")
        if int(d.get("n_rows_out", -1)) != exp_rows:
            problems.append(f"shard {si}: n_rows_out={d.get('n_rows_out')} != "
                            f"input rows {exp_rows}  <-- ROW CONSERVATION VIOLATED")

    if deep:
        for si, d in sorted(got.items()):
            p = job.output_dir / f"part_{si:05d}{cfg.shard_suffix}"
            n = count_jsonl_rows(p)
            if n != int(d.get("n_rows_out", -1)):
                problems.append(f"shard {si}: file has {n} lines, sidecar claims "
                                f"{d.get('n_rows_out')}")

    # Check #4, the required assertion. This used to be guarded by `not missing`, which
    # made it dead code in precisely the case it exists for -- a short row count is exactly
    # what missing shards cause. Missing shards are now reported above with their indices,
    # and the total is checked whenever the shard set IS complete, which is the only
    # situation where a discrepancy would otherwise go unexplained.
    if not missing and rows_out != man.total_rows:
        problems.append(
            f"TOTAL row count {rows_out:,} != input row count {man.total_rows:,} for arm "
            f"{job.arm}. Every prompt must rewrite the ENTIRE dataset for its arm."
        )

    state = "DONE" if (not missing and not problems) else (
        "PENDING" if not got else "PARTIAL")
    return JobStatus(job_id=job.job_id, n_shards=man.n_shards, done=len(got),
                     rows_out=rows_out, out_tokens=out_tok, state=state,
                     problems=problems)


def count_jsonl_rows(path: Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    for _ in iter_jsonl(p):
        n += 1
    return n


# --------------------------------------------------------------------------- jsonl io
def open_jsonl_write(path: Path, compress: bool | None = None):
    """Return (file object, closer). zstd when the FINAL name says so, else plain.

    `compress=None` infers from the filename, stripping a trailing ".tmp" first --
    every caller writes to "<final>.tmp" and then os.replace()s it into place, so
    inferring from the literal name would silently write PLAIN text into a file that
    ends up named ".jsonl.zst" and is unreadable afterwards. Pass `compress` explicitly
    to be certain.
    """
    path = Path(path)
    name = path.name[:-4] if path.name.endswith(".tmp") else path.name
    if compress is None:
        compress = name.endswith(".zst")
    if compress:
        import zstandard as zstd
        raw = open(path, "wb")
        cctx = zstd.ZstdCompressor(level=3)
        w = cctx.stream_writer(raw)
        return w, (lambda: (w.close(), raw.close()))
    raw = open(path, "wb")
    return raw, (lambda: raw.close())


def iter_jsonl(path: Path):
    """Stream a shard's rows as dicts, transparently handling zstd."""
    path = Path(path)
    name = path.name[:-4] if path.name.endswith(".tmp") else path.name
    if name.endswith(".zst"):
        import zstandard as zstd
        with open(path, "rb") as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as r:
                import io
                for line in io.TextIOWrapper(r, encoding="utf-8"):
                    line = line.strip()
                    if line:
                        yield json.loads(line)
    else:
        with open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def jsonl_to_table(path: Path, keys: list):
    """Read a JSONL shard into a pa.Table -- the `load_fn` seam for the shuffle."""
    cols = {k: [] for k in keys}
    for row in iter_jsonl(path):
        for k in keys:
            cols[k].append(row.get(k))
    types = {
        "doc_id": pa.int64(), "arm": pa.large_string(), "prompt_id": pa.large_string(),
        "source_text_sha1": pa.string(), "rewritten_text": pa.large_string(),
        "finish_reason": pa.large_string(), "n_prompt_tokens": pa.int32(),
        "n_output_tokens": pa.int32(), "status": pa.int8(),
        "n_output_tokens_llama2": pa.int32(),
        # Present in every row of every job; the empty string outside wrap-inspired's
        # styled pass. Source had it in wrap mode only, but a column that exists for one
        # job and not the others makes the ten output sets non-uniform.
        "wrap_style": pa.large_string(),
    }
    return pa.table({k: pa.array(v, type=types.get(k)) for k, v in cols.items()})
