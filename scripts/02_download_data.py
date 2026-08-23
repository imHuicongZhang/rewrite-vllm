#!/usr/bin/env python3
"""Download all six arms, verify the `text` column, re-shard, and write the manifest.

quality-base is the CONTROL: it is downloaded and verified, its rows and content hash are
recorded so the token accounting is complete, and it is flagged rewrite: false. It is
never rewritten and produces no shards to generate from.

    python scripts/02_download_data.py [--arm NAME] [--count-tokens]

--count-tokens runs the Llama-2 tokenizer over every arm to record exact token totals.
That is a full extra pass over the corpus; it is off by default and is mainly useful for
the control arm, whose tokens are never counted anywhere else.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rewrite import data as D                            # noqa: E402
from rewrite.config import load_config                   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config-root", default=None)
    ap.add_argument("--arm", default=None, help="only this arm (default: all six)")
    ap.add_argument("--count-tokens", action="store_true",
                    help="also count Llama-2 tokens (an extra full pass)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config_root)
    arms = [cfg.arm(args.arm)] if args.arm else list(cfg.arms)

    for a in arms:
        tag = "CONTROL, never rewritten" if not a.rewrite else f"{len(a.prompts)} prompts"
        print(f"\n=== {a.name}  ({tag}) ===")
        D.download_arm(cfg, a.name)
        D.probe_arm(cfg, a.name)
        D.shard_arm(cfg, a.name, count_tokens=args.count_tokens)

    if not args.arm:
        D.write_data_manifest(cfg)
        print("\n--- summary ---")
        total_jobs = 0
        for a in cfg.arms:
            m = D.load_manifest(cfg, a.name)
            n = len(a.prompts) if a.rewrite else 0
            total_jobs += n
            print(f"  {a.name:22s} rows={m.total_rows:>14,}  shards={m.n_shards:>7,}  "
                  f"text={m.total_text_bytes/2**30:>8.1f} GiB  jobs={n}"
                  + ("   <- control, rewrite: false" if not a.rewrite else ""))
        print(f"  {'TOTAL REWRITE JOBS':22s} {total_jobs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
