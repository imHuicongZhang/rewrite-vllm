#!/usr/bin/env python3
"""Trim then shuffle every finished (arm, prompt) job.

Two stages, both resumable, both scoped to ONE (arm, prompt) job at a time:

  1. trim     per-arm/per-prompt rule, ported verbatim from the source's Step 1.
              Only status==2 rows are touched; Llama-2 counts are recomputed only for
              rows the trim actually changed; the row count must not change.
  2. shuffle  the source's bucketed two-pass shuffle, seed 42, 500k-row output shards.
              WITHIN (arm, prompt) ONLY -- never across arms.

    python scripts/04_postprocess.py                       # every job
    python scripts/04_postprocess.py --arm wrap-inspired --prompt-id p3
    python scripts/04_postprocess.py --stage trim
    python scripts/04_postprocess.py --dry-run --sample 100000   # strip rates only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rewrite import data as D                                   # noqa: E402
from rewrite import postprocess as PP                           # noqa: E402
from rewrite import shuffle as SH                               # noqa: E402
from rewrite.config import enumerate_jobs, get_job, load_config  # noqa: E402


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
    args = ap.parse_args(argv)

    cfg = load_config(args.config_root)
    if args.arm and args.prompt_id:
        jobs = [get_job(cfg, args.arm, args.prompt_id)]
    elif args.arm:
        jobs = [j for j in enumerate_jobs(cfg) if j.arm == args.arm]
    else:
        jobs = enumerate_jobs(cfg)

    out = {}
    for job in jobs:
        print(f"\n=== postprocess {job.job_id} ===")
        if args.dry_run:
            out[job.job_id] = dry_run(cfg, job, args.sample)
            continue
        if args.stage in ("trim", "both"):
            out.setdefault(job.job_id, {})["trim"] = PP.trim_job(
                cfg, job, workers=args.workers)
        if args.stage in ("shuffle", "both"):
            out.setdefault(job.job_id, {})["shuffle"] = SH.shuffle_job(cfg, job)

    if not args.dry_run:
        p = cfg.repo_root / "manifests" / "postprocess_summary.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        D.atomic_write_text(json.dumps(out, indent=2, default=str), p)
        print(f"\n[postprocess] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
