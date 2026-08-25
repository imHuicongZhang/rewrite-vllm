#!/usr/bin/env python3
"""Measure real generation throughput and project wall clock for all 10 jobs.

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
from rewrite.wrap_styles import assign_wrap_styles                # noqa: E402


def human_time(hours: float) -> str:
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def rate_from_sidecars(cfg, jobs, min_shards: int):
    """Aggregate measured tok/s and tokens/doc from completed shards."""
    tok = rows = in_tok = 0
    gpu_secs = 0.0
    n = 0
    hosts, cards = set(), set()
    for job in jobs:
        for sp in job.output_dir.glob("part_*.done"):
            sc = D.read_sidecar(sp)
            if not sc or not sc.get("elapsed_s"):
                continue
            tok += int(sc.get("n_output_tokens", 0))
            in_tok += int(sc.get("n_prompt_tokens", 0))
            rows += int(sc.get("n_rows_out", 0))
            gpu_secs += float(sc["elapsed_s"])
            hosts.add(sc.get("host", "?"))
            cards.add(sc.get("gpu_name", "?"))
            n += 1
    if n < min_shards or gpu_secs <= 0:
        return None
    return {"tok_s_per_gpu": tok / gpu_secs, "tok_per_doc": tok / max(1, rows),
            "in_tok_s_per_gpu": in_tok / gpu_secs, "in_tok_per_doc": in_tok / max(1, rows),
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

    styles = (assign_wrap_styles(0, len(texts))
              if job.prompt.mode == "wrap_multi" else None)
    prep = E.prepare_batch(qtok, cfg, texts, job.prompt, drop, styles=styles)
    t0 = time.perf_counter()
    res = E.run_batch(llm, prep)
    dt = time.perf_counter() - t0

    out_tok = sum(res.n_output_tokens)
    in_tok = sum(prep.n_in_list)
    gpu = "?"
    try:
        import torch
        gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    print(f"  generated {out_tok:,} output tokens from {in_tok:,} prompt tokens "
          f"in {dt:.1f}s on {gpu}")
    return {"tok_s_per_gpu": out_tok / dt, "tok_per_doc": out_tok / max(1, len(texts)),
            "in_tok_s_per_gpu": in_tok / dt, "in_tok_per_doc": in_tok / max(1, len(texts)),
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

    # ---- PREFILL vs DECODE --------------------------------------------------------
    # This workload is prefill-heavy in a way most rewriting workloads are not: the
    # configured budgets imply 720B input tokens against ~260B output, a 2.75:1 ratio.
    # A single blended tok/s hides that, and prefill is the side most likely to be
    # mistuned, so report the split explicitly.
    itps = r.get("in_tok_s_per_gpu") or 0.0
    itpd = r.get("in_tok_per_doc") or 0.0
    if itps > 0:
        ratio = itpd / tpd if tpd else 0.0
        # Both phases ran in the same wall clock, so their token rates are throughputs
        # over the SAME interval, not independent speeds. The useful decomposition is
        # therefore the token MIX, which is what determines where the time goes once you
        # know the per-phase cost. Prefill is compute-bound and roughly linear in tokens;
        # decode is memory-bound and roughly linear in tokens x steps.
        print()
        print(f"  prefill  : {itps:,.0f} prompt tok/s per GPU, {itpd:,.0f} prompt tok/doc")
        print(f"  decode   : {tps:,.0f} output tok/s per GPU, {tpd:,.0f} output tok/doc")
        print(f"  mix      : {ratio:.2f} prompt tokens per output token "
              f"({100*ratio/(1+ratio):.0f}% of all tokens processed are prefill)")
        cfg_ratio = 720.0 / 261.5
        flag = "" if abs(ratio - cfg_ratio) / cfg_ratio < 0.25 else \
               "   <-- differs >25% from the configured 2.75:1; check the estimates"
        print(f"             configured expectation is {cfg_ratio:.2f}:1{flag}")
        print(f"  NOTE     : at this ratio prefill is a large share of GPU time. If you "
              f"are tuning,\n             look at max_num_batched_tokens and chunked "
              f"prefill BEFORE decode-side knobs.\n             Do not change engine args "
              f"without asking Wytro -- source parity governs.")

    # Token VOLUME comes from configs/data.yaml's per-(arm, prompt) est_output_tokens,
    # which is source_tokens_llama2 x a MEASURED ratio -- not from the measured tok/doc of
    # a single sampled batch. The arms differ far too much for one tok/doc to stand in for
    # all of them: source documents run 949 tok/doc in rewire-inspired and 1,600 in
    # quality-first, and the distill prompt condenses harder than the wiki prompt. The
    # measured rate is used for SPEED only, which is what it actually measures.
    missing = []
    total_rows = total_tok = 0
    per_job = []
    for job in jobs:
        try:
            rows = D.input_rows(cfg, job.arm)
        except SystemExit:
            missing.append(job.arm)
            continue
        est = job.prompt.est_output_tokens
        tok = float(est) if est else rows * tpd
        total_rows += rows
        total_tok += tok
        per_job.append((job.job_id, rows, tok, tok / (tps * ngpu) / 3600,
                        (tok / rows if rows else 0.0), bool(est)))

    # Sanity: does the batch we just measured agree with the config's predicted tok/doc for
    # that same job? A large gap means either the measurement was unrepresentative or the
    # ratios in data.yaml no longer describe this model/prompt.
    ref = next((x for x in per_job if x[0] == jobs[0].job_id), None)
    if ref and ref[5] and tpd:
        pred = ref[4]
        ratio = tpd / pred if pred else 0.0
        note = "OK" if 0.5 <= ratio <= 2.0 else "<-- CHECK: measurement and config disagree"
        print(f"  cross-check: {jobs[0].job_id} predicted {pred:,.0f} tok/doc from "
              f"data.yaml, measured {tpd:,.0f} ({ratio:.2f}x)  {note}")

    if missing:
        print(f"\n  NOTE: no manifest yet for {sorted(set(missing))}; those jobs are not "
              f"in the projection.")
    if not per_job:
        print("\n  Nothing to project until 02_download_data.py has run.")
        return 0

    print()
    print(f"  {'JOB':30s} {'ROWS':>14s} {'OUT TOKENS':>13s} {'TOK/DOC':>9s} {'WALL':>10s}")
    for jid, rows, tok, hrs, tpd_job, measured in per_job:
        print(f"  {jid:30s} {rows:14,d} {tok / 1e9:12.2f}B {tpd_job:9,.0f} "
              f"{human_time(hrs):>10s}" + ("" if measured else "   [rate-derived]"))
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
