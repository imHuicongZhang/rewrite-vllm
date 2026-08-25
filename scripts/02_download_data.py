#!/usr/bin/env python3
"""Download all five arms, verify the `text` column, re-shard, and write the manifest.

The input is ONE gated HuggingFace repo with one folder per arm; each arm pulls only its
own subdirectory. The raw half of the corpus -- the shared 20B core and the 50B
quality-base control -- is deliberately NOT downloaded: neither is rewritten, so pulling
20B raw tokens onto this machine would cost bandwidth and disk for nothing. Both are
recorded in the header comment of configs/data.yaml and in manifests/data_manifest.json
under "raw_not_rewritten" so the token accounting stays complete.
See docs/DESIGN_DELTA.md section 3.

    python scripts/02_download_data.py [--arm NAME] [--count-tokens]

--count-tokens runs the Llama-2 tokenizer over every arm to record exact token totals.
That is a full extra pass over the corpus and is off by default: configs/data.yaml already
declares each arm's source_tokens_llama2 from the upstream selection stage, and the row
count is cross-checked against `docs` on every download regardless.
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
    ap.add_argument("--arm", default=None, help="only this arm (default: all five)")
    ap.add_argument("--count-tokens", action="store_true",
                    help="also count Llama-2 tokens (an extra full pass)")
    ap.add_argument("--wait-only", action="store_true",
                    help="do not download or shard; just wait until every arm's manifest "
                         "exists. For worker nodes in a multi-node run, where one node "
                         "prepares the data and the rest must not race it.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config_root)
    arms = [cfg.arm(args.arm)] if args.arm else list(cfg.arms)

    if args.wait_only:
        import time as _t
        pending = list(arms)
        waited = 0
        while True:
            missing = [a.name for a in pending if not D.manifest_path(cfg, a.name).exists()]
            if not missing:
                break
            if waited % 300 == 0:
                print(f"[data] waiting for manifests: {missing} ({waited // 60} min)",
                      flush=True)
            _t.sleep(15)
            waited += 15
        for a in arms:
            m = D.load_manifest(cfg, a.name)
            print(f"[data] {a.name}: ready ({m.total_rows:,} rows, {m.n_shards} shards, "
                  f"doc_id={m.doc_id_source})")
        return 0

    for a in arms:
        print(f"\n=== {a.name}  ({len(a.prompts)} prompts, "
              f"{a.source_tokens_llama2/1e9:.1f}B source tokens) ===")
        D.download_arm(cfg, a.name)
        D.probe_arm(cfg, a.name)
        D.shard_arm(cfg, a.name, count_tokens=args.count_tokens)

    if not args.arm:
        D.write_data_manifest(cfg)
        print("\n--- summary ---")
        total_jobs = 0
        for a in cfg.arms:
            m = D.load_manifest(cfg, a.name)
            n = len(a.prompts)
            total_jobs += n
            print(f"  {a.name:22s} rows={m.total_rows:>14,}  shards={m.n_shards:>7,}  "
                  f"text={m.total_text_bytes/2**30:>8.1f} GiB  jobs={n}")
        print(f"  {'TOTAL REWRITE JOBS':22s} {total_jobs}")
        print()
        print("  NOT downloaded and NOT rewritten (the raw half of the corpus):")
        print(f"  {'shared-core':22s} 17,909,083 docs   20.0B tokens   "
              f"raw, carried into training by all five arms")
        print(f"  {'quality-base':22s} 37,298,288 docs   50.0B tokens   "
              f"raw control, never uploaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
