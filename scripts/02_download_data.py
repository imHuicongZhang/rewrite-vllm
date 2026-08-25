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
    ap.add_argument("--wait-timeout-s", type=float, default=24 * 3600,
                    help="give up waiting for manifests after this long (--wait-only). "
                         "Bounded on purpose: an unbounded wait cannot be told apart from "
                         "a hung run.")
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
        # Bounded, for the same reason the sharding lock is: a worker node that waits
        # forever on a preparing node that died looks identical to one that is working.
        # The preparing node holds a heartbeated lock per arm, so "no lock and no manifest"
        # means nobody is preparing this arm -- report that rather than sleeping on it.
        limit = int(args.wait_timeout_s)
        while True:
            missing = [a.name for a in pending if not D.manifest_path(cfg, a.name).exists()]
            if not missing:
                break
            if waited >= limit:
                idle = [n for n in missing
                        if not (cfg.shards_dir(n) / ".sharding.lock").exists()]
                print(f"\n*** STOP: waited {waited / 3600:.1f} h for manifests and "
                      f"{len(missing)} arm(s) are still missing: {missing}",
                      file=sys.stderr)
                if idle:
                    print(f"  {idle} have no .sharding.lock either, so NO process is "
                          f"preparing them. The node that was meant to run "
                          f"02_download_data.py has probably died.", file=sys.stderr)
                print(f"  Run 02_download_data.py (without --wait-only) on one node, or "
                      f"raise --wait-timeout-s if data prep genuinely takes longer.",
                      file=sys.stderr)
                return 2
            if waited % 300 == 0:
                held = {n: (cfg.shards_dir(n) / ".sharding.lock").exists() for n in missing}
                print(f"[data] waiting for manifests: {missing} ({waited // 60} min); "
                      f"being prepared now: {[n for n, v in held.items() if v]}",
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
