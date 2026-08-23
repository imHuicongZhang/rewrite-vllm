#!/usr/bin/env python3
"""Measure real generation throughput and project wall clock for all 12 jobs.

WHY, BEFORE JOB 1
-----------------
At ~100 GPUs the difference between 5,000 and 15,000 output tok/s per GPU is the
difference between a four-day run and a thirteen-day one, and nobody knows which it is
until real generation happens on real hardware. preflight's smoke test proves correctness,
not speed. A wrong number should be noticed on day zero, not on day six.

Two ways to get the rate, and the second is better once it is available:

  --measure         one shard-sized batch on ONE GPU, real model, real prompt, real
                    documents. Costs a model load plus a minute or two.
  --from-sidecars   derive the rate from shards already generated. Free, and far more
                    representative than a single batch because it is the actual run.

Default: use sidecars when at least --min-shards of them exist, otherwise measure.

    python scripts/06_calibrate.py
    python scripts/06_calibrate.py --measure --docs 400
    python scripts/06_calibrate.py --from-sidecars

Exit code is 0 even when it cannot measure: this is advisory, not a gate.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rewrite import data as D                                    # noqa: E402
from rewrite.config import enumerate_jobs, load_config, resolve_drop_threshold  # noqa: E402


def human_time(hours: float) -> str:
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def rate_from_sidecars(cfg, jobs, min_shards: int):
    """Aggregate measured tok/s and tokens/doc from completed shards."""
    tok = rows = 0
    gpu_secs = 0.0
    n = 0
    hosts, cards = set(), set()
    for job in jobs:
        for sp in job.output_dir.glob("part_*.done"):
            sc = D.read_sidecar(sp)
            if not sc or not sc.get("elapsed_s"):
                continue
            tok += int(sc.get("n_output_tokens", 0))
            rows += int(sc.get("n_rows_out", 0))
            gpu_secs += float(sc["elapsed_s"])
            hosts.add(sc.get("host", "?"))
            cards.add(sc.get("gpu_name", "?"))
            n += 1
    if n < min_shards or gpu_secs <= 0:
        return None
    return {"tok_s_per_gpu": tok / gpu_secs, "tok_per_doc": tok / max(1, rows),
            "shards": n, "hosts": len(hosts), "cards": sorted(c for c in cards if c != "?")}


def rate_by_measuring(cfg, jobs, n_docs: int):
    """One real batch on one GPU."""
    from rewrite import engine as E
    job = jobs[0]
    shards = D.shard_paths(cfg, job.arm)
    if not shards:
        print("  cannot measure: no input shards yet (run 02_download_data.py first)")
        return None

    E.set_source_env(cfg)
    qtok = E.load_qwen_tokenizer(cfg)
    ltok = E.load_llama2_tokenizer(cfg)
    tbl = D.read_shard(shards[0][1]).slice(0, n_docs)
    texts = tbl.column(cfg.text_column).to_pylist()
    drop, _ = resolve_drop_threshold(job.prompt, cfg.max_model_len, cfg.max_tokens)

    print(f"  loading the model (this dominates the wall clock of this step) ...")
    t_load = time.perf_counter()
    llm = E.build_llm(cfg)
    print(f"  model loaded in {time.perf_counter() - t_load:.0f}s; "
          f"generating {len(texts)} real documents with {job.job_id}'s prompt ...")

    prep = E.prepare_batch(qtok, cfg, texts, job.prompt, drop)
    t0 = time.perf_counter()
    res = E.run_batch(llm, prep)
    dt = time.perf_counter() - t0

    out_tok = sum(res.n_output_tokens)
    gpu = "?"
    try:
        import torch
        gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    print(f"  generated {out_tok:,} output tokens in {dt:.1f}s on {gpu}")
    return {"tok_s_per_gpu": out_tok / dt, "tok_per_doc": out_tok / max(1, len(texts)),
            "shards": 0, "hosts": 1, "cards": [gpu]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config-root", default=None)
    ap.add_argument("--measure", action="store_true", help="force a fresh measurement")
    ap.add_argument("--from-sidecars", action="store_true",
                    help="force using already-generated shards")
    ap.add_argument("--docs", type=int, default=256, help="documents to generate (--measure)")
    ap.add_argument("--min-shards", type=int, default=5,
                    help="sidecars needed before they are preferred over measuring")
    args = ap.parse_args(argv)

    cfg = load_config(args.config_root)
    jobs = enumerate_jobs(cfg)

    print("=" * 74)
    print("THROUGHPUT CALIBRATION")
    print("=" * 74)

    r = None
    if not args.measure:
        r = rate_from_sidecars(cfg, jobs, 0 if args.from_sidecars else args.min_shards)
        if r:
            print(f"  source: {r['shards']} already-generated shard(s) across "
                  f"{r['hosts']} node(s)")
    if r is None and not args.from_sidecars:
        try:
            r = rate_by_measuring(cfg, jobs, args.docs)
        except SystemExit:
            raise
        except Exception as e:                                   # noqa: BLE001
            print(f"  measurement failed: {type(e).__name__}: {e}")
    if r is None:
        print("\n  No throughput estimate available. This step is advisory; the "
              "\n  per-shard log line prints tok/s once job 1 starts.")
        return 0

    # ---- projection from the ACTUAL manifest row counts ----
    tps, tpd = r["tok_s_per_gpu"], r["tok_per_doc"]
    ngpu = cfg.num_gpus
    print()
    print(f"  measured : {tps:,.0f} output tok/s per GPU")
    print(f"             {tpd:,.0f} output tokens per document")
    if r["cards"]:
        print(f"  on       : {', '.join(r['cards'])}")
    print(f"  fleet    : {ngpu} GPU(s) -> {tps * ngpu / 1e6:,.1f} M tok/s aggregate")

    missing = []
    total_rows = total_tok = 0
    per_job = []
    for job in jobs:
        try:
            rows = D.input_rows(cfg, job.arm)
        except SystemExit:
            missing.append(job.arm)
            continue
        tok = rows * tpd
        total_rows += rows
        total_tok += tok
        per_job.append((job.job_id, rows, tok, tok / (tps * ngpu) / 3600))

    if missing:
        print(f"\n  NOTE: no manifest yet for {sorted(set(missing))}; those jobs are not "
              f"in the projection.")
    if not per_job:
        print("\n  Nothing to project until 02_download_data.py has run.")
        return 0

    print()
    print(f"  {'JOB':32s} {'ROWS':>14s} {'OUT TOKENS':>14s} {'WALL':>10s}")
    for jid, rows, tok, hrs in per_job:
        print(f"  {jid:32s} {rows:14,d} {tok / 1e9:13.1f}B {human_time(hrs):>10s}")
    total_h = total_tok / (tps * ngpu) / 3600

    print()
    print("=" * 74)
    print(f"  PROJECTED TOTAL: {total_tok / 1e9:,.0f}B output tokens over "
          f"{total_rows:,} rows")
    print(f"  AT {tps:,.0f} tok/s/GPU x {ngpu} GPUs  ->  {human_time(total_h)}")
    print("=" * 74)
    print(f"  Sensitivity, since the measured rate is the whole projection:")
    for mult, label in ((0.5, "half"), (1.0, "measured"), (2.0, "double")):
        print(f"    {tps * mult:>9,.0f} tok/s/GPU  ->  {human_time(total_h / mult)}"
              + ("   <-- measured" if mult == 1.0 else ""))
    print()
    print("  This ignores model load, data prep, postprocess and upload, and assumes every")
    print("  GPU stays busy. Treat it as a floor. If it is wildly longer than you expected,")
    print("  find out why NOW -- not on day six.")
    if r["shards"] == 0:
        print("  Based on ONE batch on ONE GPU. Re-run after a few shards have completed")
        print("  (`--from-sidecars`) for a number drawn from the real run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
