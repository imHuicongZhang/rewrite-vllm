#!/usr/bin/env python3
"""Push each finished (arm, prompt) output to its own HuggingFace dataset repo.

Repo names come from configs/data.yaml upload.repo_template, e.g.
    your-org/rewrite-{arm}-{prompt_id}

Retries on 429 and 5xx with exponential backoff and jitter. Uploads are resumable:
huggingface_hub uploads by content hash, so re-running skips files already on the Hub, and
a per-job marker skips finished jobs entirely.

    python scripts/05_upload_to_hf.py [--arm NAME] [--prompt-id pN] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rewrite import data as D                                   # noqa: E402
from rewrite.config import enumerate_jobs, get_job, load_config, stop  # noqa: E402

MAX_ATTEMPTS = 6
BASE_SLEEP = 5.0


def _retryable(exc) -> bool:
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if code is None:
        code = getattr(exc, "status_code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    txt = f"{type(exc).__name__}: {exc}".lower()
    return any(s in txt for s in ("429", "too many requests", "rate limit", "timeout",
                                  "timed out", "connection reset", "temporarily",
                                  "bad gateway", "service unavailable"))


def with_retry(fn, what: str, log=print):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:                                   # noqa: BLE001
            if attempt == MAX_ATTEMPTS or not _retryable(e):
                raise
            sleep = BASE_SLEEP * (2 ** (attempt - 1)) * (0.5 + random.random())
            log(f"    {what}: attempt {attempt}/{MAX_ATTEMPTS} failed ({e}); "
                f"retrying in {sleep:.0f}s")
            time.sleep(sleep)


def repo_name(cfg, job) -> str:
    tpl = cfg.data["upload"]["repo_template"]
    return tpl.format(arm=job.arm, prompt_id=job.prompt.id)


def stage_dir(cfg, job) -> Path:
    stage = cfg.data["upload"]["stage"]
    if stage == "raw":
        return job.output_dir
    if stage == "trimmed":
        return (job.output_dir if cfg.data["postprocess"]["in_place"]
                else cfg.trimmed_dir(job.arm, job.prompt.id))
    if stage == "shuffled":
        return cfg.shuffled_dir(job.arm, job.prompt.id)
    stop(f"configs/data.yaml upload.stage must be raw|trimmed|shuffled, got {stage!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config-root", default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--prompt-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config_root)
    token = cfg.env.get("HF_TOKEN_WRITE") or cfg.env.get("HF_TOKEN")
    if not token and not args.dry_run:
        stop("no HF_TOKEN_WRITE (or HF_TOKEN) in .env -- upload needs write scope")

    if args.arm and args.prompt_id:
        jobs = [get_job(cfg, args.arm, args.prompt_id)]
    elif args.arm:
        jobs = [j for j in enumerate_jobs(cfg) if j.arm == args.arm]
    else:
        jobs = enumerate_jobs(cfg)

    from huggingface_hub import HfApi
    api = HfApi(token=token) if not args.dry_run else None
    private = bool(cfg.data["upload"]["private"])

    # What actually gets uploaded. Bookkeeping must never reach the Hub, and this list has
    # to stay in step with allow/ignore_patterns below or the reported file count and byte
    # total are lies.
    #
    # p.is_file() is load-bearing, not defensive: a shard CLAIM is a *directory*
    # (part_NNNNN.claim/, created by data.try_claim), and Path.glob("part_*") yields
    # directories. Without the check it survived this filter, .stat() reported the dirent
    # size instead of a shard size, and -- worse -- huggingface_hub would have uploaded the
    # file inside it, because fnmatch's "*" crosses "/", so "part_*" matches
    # "part_99999.claim/owner.json". Verified against huggingface_hub.utils
    # .filter_repo_objects. Claims only survive a hard-killed worker, but if one does, it
    # silently ships.
    def payload(src: Path):
        return sorted(p for p in src.glob("part_*")
                      if p.is_file() and not p.name.endswith((".tmp", ".done")))

    for job in jobs:
        name = repo_name(cfg, job)
        src = stage_dir(cfg, job)
        marker = src / "_uploaded.done"
        files = payload(src)
        nbytes = sum(p.stat().st_size for p in files)
        print(f"\n=== upload {job.job_id} -> {name} "
              f"({len(files)} files, {nbytes/2**30:.1f} GiB, stage="
              f"{cfg.data['upload']['stage']}) ===")

        if args.dry_run:
            print("    dry-run: nothing sent")
            continue
        if marker.exists():
            print("    already uploaded -> skip")
            continue
        if not files:
            stop(f"{job.job_id}: nothing to upload in {src}")

        with_retry(lambda: api.create_repo(repo_id=name, repo_type="dataset",
                                           private=private, exist_ok=True),
                   f"create_repo {name}")
        with_retry(lambda: api.upload_folder(
            repo_id=name, repo_type="dataset", folder_path=str(src),
            path_in_repo="data",
            allow_patterns=["part_*"],
            # kept in step with payload() above; "*.claim/*" is what actually stops a
            # leaked claim directory, since fnmatch "*" crosses "/"
            ignore_patterns=["*.tmp", "*.done", "*.claim", "*.claim/*", ".joblock"],
            commit_message=f"{job.job_id}: {len(files)} shards"),
            f"upload_folder {name}")

        D.atomic_write_text(json.dumps(
            {"repo": name, "files": len(files), "bytes": nbytes,
             "stage": cfg.data["upload"]["stage"],
             "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2), marker)
        print(f"    uploaded -> https://huggingface.co/datasets/{name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
