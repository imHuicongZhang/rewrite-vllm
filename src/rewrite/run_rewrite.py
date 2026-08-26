"""One rewrite job = one (arm, prompt) pair. One process = one GPU.

This is the merge of the source's two workers, 07_rewrite/rewrite_worker.py and
09_Distill/rewrite_worker.py. A diff of those two files shows they are identical apart
from the drop threshold and the output subdirectory, so both become configuration and
there is NO `if pass == "distill"` branch anywhere below.

THE SEMANTIC THAT IS EASY TO GET WRONG
--------------------------------------
Each prompt covers EVERY document of its arm exactly once. Prompts are not partitioned
across documents, not sampled, not round-robined. The shard list for a job is the arm's
complete shard list, with no prompt-dependent filtering anywhere, and the row-conservation
assertion at step 9 below fails the job if that ever stops being true.

`wrap-inspired`'s styled pass (p1) is ONE job that still covers every document exactly
once -- but the prompt it uses is chosen PER DOCUMENT from four styles, by
np.random.default_rng([42, shard_index]), exactly as the source did it. That is full
coverage without being a cross product. The chosen style is recorded per row in the
`wrap_style` output column, or it would be unrecoverable afterwards.

Rounds 1-3 modelled wrap as four complete passes and deleted the style RNG; round 4
restored it because that design was superseded. See docs/DESIGN_DELTA.md section 2.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

from . import data as D
from .wrap_styles import WRAP_STYLES, assign_wrap_styles
from .config import (Config, JobSpec, enumerate_jobs, get_job, load_config,
                     resolve_drop_threshold, stop)


HOST = socket.gethostname().split(".")[0]


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- progress
def write_progress(path: Path, prog: dict) -> None:
    """source: 07_rewrite/rewrite_worker.py:133-138 (atomic .tmp + os.replace)."""
    prog["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    D.atomic_write_text(json.dumps(prog, indent=2), path)


def _guards(cfg: Config, job: JobSpec) -> None:
    """The RAW-SOURCE guards from 09_Distill, generalised.

    source: 09_Distill/rewrite_worker.py:194-201 -- the distill pass must read the
    ORIGINAL text, never another pass's output. A later edit that silently switched the
    source would be invisible in the numbers, so it is asserted instead of documented.
    """
    if cfg.text_column == "rewritten_text":
        stop("text_column must be the raw source column, never a rewrite output column")
    leaf = job.input_dir.name
    if leaf in ("raw", "trimmed", "shuffled"):
        stop(f"--input dir points at an output directory ('{leaf}')")
    inp = job.input_dir.resolve()
    if inp.is_relative_to(cfg.paths["out_root"].resolve()):
        stop(f"input dir {inp} is inside out_root; a job must read the sharded input, "
             "never another job's output")


# --------------------------------------------------------------------------- one job
def run_worker(cfg: Config, job: JobSpec, worker_id: int, num_workers: int,
               limit_shards: int | None = None) -> int:
    _guards(cfg, job)
    from . import engine as E

    E.set_source_env(cfg)
    man = D.load_manifest(cfg, job.arm)
    drop_threshold, derived = resolve_drop_threshold(
        job.prompt, cfg.max_model_len, cfg.max_tokens)

    qtok = E.load_qwen_tokenizer(cfg)
    ltok = E.load_llama2_tokenizer(cfg)

    # ---- PARITY GATE ----------------------------------------------------------
    # One integer that proves prompt file + chat template + tokenizer are the source's.
    # source printed exactly this line: 09_Distill/logs/ds_diversity-first_1591082_0.out
    # wrap-inspired's styled pass emits four different prompt texts, so it has four
    # overheads and every one is checked. A silent mismatch on one style would corrupt a
    # quarter of that arm's corpus.
    measured = E.check_overheads(qtok, cfg, job.prompt)
    for label, got, want in measured:
        raw_budget = cfg.max_model_len - cfg.max_tokens - got
        log(f"[w{worker_id}] OVERHEAD(empty-doc templated tokens)[{label}]={got} "
            f"raw_text_budget={raw_budget} drop_threshold={drop_threshold} "
            f"(={'derived max_model_len-max_tokens' if derived else 'fixed'})")
    bad = [(lbl, got, want) for lbl, got, want in measured if got != want]
    if bad:
        detail = "\n".join(f"    {lbl}: measured {got}, expected {want}"
                            for lbl, got, want in bad)
        stop(
            f"{job.job_id}: empty-document templated overhead does not match for "
            f"{len(bad)} of {len(measured)} prompt text(s).\n{detail}\n"
            "  The prompt file, the chat template, or the tokenizer differs from the "
            "source pipeline. Every token count downstream would diverge.\n"
            "  Do NOT edit the expected value to make this pass -- find out what changed."
        )
    # For the sidecar: one number for a grounded job, or the four styles rendered.
    overhead = (measured[0][1] if len(measured) == 1
                else {lbl.split(":", 1)[-1]: got for lbl, got, _ in measured})

    all_shards = D.shard_paths(cfg, job.arm)
    if len(all_shards) != man.n_shards:
        stop(f"{job.arm}: found {len(all_shards)} shards on disk but the manifest says "
             f"{man.n_shards}. Re-run scripts/02_download_data.py.")
    assignment = cfg.cluster["compute"].get("shard_assignment", "dynamic")
    if assignment == "static":
        # Source parity: shard_index % num_workers == worker_id. Perfectly balanced only
        # when every GPU is identical. (07_rewrite/rewrite_worker.py:197-203)
        owned = D.owned_shards(all_shards, worker_id, num_workers)
    else:
        # Dynamic: this worker will claim shards as it frees up, so a mixed
        # H200/B200/B300 fleet self-balances instead of running at the slowest card's
        # pace. `owned` is the candidate list; claiming happens in the loop.
        owned = list(all_shards)
    if limit_shards:
        owned = owned[:limit_shards]

    job.output_dir.mkdir(parents=True, exist_ok=True)
    n_cleaned = D.clean_stale_tmp(job.output_dir, owned, cfg.shard_suffix)
    if n_cleaned:
        log(f"[w{worker_id}] removed {n_cleaned} stale .tmp file(s)")

    log(f"[w{worker_id}] {job.job_id} mode={job.prompt.mode} trim={job.prompt.trim} "
        + (f"owns {len(owned)}/{len(all_shards)} shards (static)"
           if assignment == "static"
           else f"claiming from {len(all_shards)} shards (dynamic)"))

    already = sum(1 for si, _ in owned
                  if D.sidecar_path(job.output_dir /
                                    f"part_{si:05d}{cfg.shard_suffix}").exists())
    if assignment != "static":
        # every worker sees the whole list, so "total" is the job, not a private slice
        already = sum(1 for si, _ in all_shards
                      if D.sidecar_path(job.output_dir /
                                        f"part_{si:05d}{cfg.shard_suffix}").exists())
    prog = {
        "job_id": job.job_id, "arm": job.arm, "prompt_id": job.prompt.id,
        "worker_id": worker_id, "shards_completed": already, "shards_total": len(owned),
        "docs_completed": 0, "docs_status_0": 0, "docs_status_1": 0, "docs_status_2": 0,
        "total_input_tokens": 0, "total_output_tokens": 0,
        "total_output_tokens_llama2": 0, "elapsed_seconds": 0.0, "last_updated": None,
    }
    # Node-qualified: on a multi-node run every node has a worker 0, and without the
    # hostname node 2's worker 0 silently overwrites node 1's. Progress reporting is how
    # anyone knows whether a two-week run is healthy, so it has to survive the fan-out.
    progress_path = (cfg.paths["log_root"] / job.arm / job.prompt.id /
                     f"progress_{HOST}_w{worker_id}.json")
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    owner = {"worker_id": worker_id, "pid": os.getpid(), "host": HOST,
             "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    stale_after = float(cfg.cluster["compute"].get("claim_stale_after_s", 1800))
    claimed = None

    llm = None
    gpu_name = "?"
    gpu_cc = "?"
    t_start = time.perf_counter()
    processed_docs = 0
    processed_out_tokens = 0
    every = int(cfg.cluster["runtime"]["progress_log_every_shards"])

    try:
      for si, shard_path in owned:
        out_path = job.output_dir / f"part_{si:05d}{cfg.shard_suffix}"
        done_path = D.sidecar_path(out_path)

        if done_path.exists():
            sc = D.read_sidecar(done_path)
            if sc and sc.get("input_fingerprint") == man.fingerprint:
                continue
            log(f"[w{worker_id}] shard {si:05d} has a stale .done "
                f"(input was re-sharded) -> redo")
            done_path.unlink()
        # A data file with no sidecar is INCOMPLETE by definition and is overwritten.

        if assignment != "static":
            if not D.try_claim(job.output_dir, si, owner):
                # Someone holds it. If that claim has not been heartbeated in
                # stale_after seconds its owner is gone -- a killed process, a lost node --
                # so take it over. Without this a node dying mid-run would strand its
                # in-flight shards until somebody noticed and restarted the job.
                if D.claim_age_s(job.output_dir, si) < stale_after:
                    continue                  # a live worker has it
                log(f"[w{worker_id}] shard {si:05d} claim is stale "
                    f"(>{stale_after:.0f}s since heartbeat) -> taking it over")
                D.release_claim(job.output_dir, si)
                if not D.try_claim(job.output_dir, si, owner):
                    continue                  # another worker beat us to the takeover
            # Re-check after winning the claim: a worker may have finished this shard
            # between our .done test above and the claim succeeding.
            if done_path.exists():
                D.release_claim(job.output_dir, si)
                continue
            claimed = si

        # Load the model only when there is real work -- a fully-resumed job must not pay
        # 12 x N model loads to discover it has nothing to do.
        if llm is None:
            # ARCHITECTURE GUARD. preflight checks this too, but preflight can be skipped
            # (run_all.sh --skip-preflight) and 03_run_job.sh can be invoked directly, so
            # the worker refuses on its own. Generating even one shard on a disallowed
            # architecture would silently contaminate the arm.
            try:
                import torch as _torch
                _pr = _torch.cuda.get_device_properties(0)
                _cc = f"sm_{_pr.major}{_pr.minor}"
                _okk, _lvl, _msg = cfg.check_gpu_arch(_cc, _torch.cuda.get_device_name(0))
                if not _okk:
                    stop(f"[w{worker_id}] refusing to run: {_msg}")
                if _lvl == "warn":
                    log(f"[w{worker_id}] WARNING: {_msg}")
            except SystemExit:
                raise
            except Exception as _e:
                log(f"[w{worker_id}] WARNING: could not check GPU architecture ({_e})")

            llm = E.build_llm(cfg)
            try:
                import torch
                gpu_name = torch.cuda.get_device_name(0)
                _p = torch.cuda.get_device_properties(0)
                gpu_cc = f"sm_{_p.major}{_p.minor}"
                owner["gpu_name"] = gpu_name
                owner["gpu_cc"] = gpu_cc
            except Exception:
                pass

        table = D.read_shard(shard_path)
        n_rows = table.num_rows
        texts = table.column(cfg.text_column).to_pylist()
        doc_ids = table.column("doc_id").to_pylist()
        shas = table.column("source_text_sha1").to_pylist()
        # Carried through from the input shard, which got it from the uploaded dataset.
        # Absent only for shards prepared before round 8 added it; those are invalidated by
        # the manifest fingerprint anyway, so this fallback should never be reached.
        rec_ids = (table.column("record_id").to_pylist()
                   if "record_id" in table.column_names else [""] * n_rows)

        t0 = time.perf_counter()
        # Hold a liveness signal on the claim for the whole shard. The main thread is
        # blocked inside llm.generate(), so a background tick is the only way another node
        # can distinguish "still working" from "died holding this".
        # ONE style per document for wrap-inspired's styled pass, keyed on (42, si) only,
        # so any worker that picks up shard si -- first time or after a crash -- assigns
        # the identical styles. See src/rewrite/wrap_styles.py.
        styles = (assign_wrap_styles(si, n_rows)
                  if job.prompt.mode == "wrap_multi" else None)

        with D.ClaimHeartbeat(job.output_dir, si, interval=min(60.0, stale_after / 5)):
            prep = E.prepare_batch(qtok, cfg, texts, job.prompt, drop_threshold,
                                   styles=styles)
            res = E.run_batch(llm, prep)
            n_llama = [0] * n_rows
            if prep.keep_idx:
                counts = E.count_llama2(ltok, [res.rewritten[j] for j in prep.keep_idx])
                for j, c in zip(prep.keep_idx, counts):
                    n_llama[j] = c
        dt = time.perf_counter() - t0

        lines = []
        for j in range(n_rows):
            lines.append(json.dumps({
                "doc_id": doc_ids[j],
                # WARC-Record-ID of the source document. A repair path for a doc_id shift,
                # NOT a unique key -- ~0.066% of documents share one. See data.yaml.
                "record_id": rec_ids[j],
                "arm": job.arm,
                "prompt_id": job.prompt.id,
                "source_text_sha1": shas[j],
                "rewritten_text": res.rewritten[j],   # RAW; trimming is a later step
                "finish_reason": res.finish_reason[j],
                "n_prompt_tokens": prep.n_in_list[j],
                "n_output_tokens": res.n_output_tokens[j],
                "status": res.status[j],
                "n_output_tokens_llama2": n_llama[j],
                # The style this document drew. Empty for the nine non-wrap jobs; without
                # it the assignment is unrecoverable after the fact.
                "wrap_style": styles[j] if styles is not None else "",
            }, ensure_ascii=False))

        # ---- ROW CONSERVATION: the guarantee is created here ----
        # status=0 documents are EMITTED (empty rewrite), never dropped, so a shard always
        # yields exactly its input row count. If this ever fails, the job fails.
        if len(lines) != n_rows:
            stop(f"{job.job_id} shard {si:05d}: built {len(lines)} output rows from "
                 f"{n_rows} input rows -- row conservation violated")

        fh, close = D.open_jsonl_write(Path(str(out_path) + ".tmp"),
                                       compress=(cfg.compression == "zstd"))
        try:
            fh.write(("\n".join(lines) + "\n").encode("utf-8"))
        finally:
            close()
        os.replace(str(out_path) + ".tmp", out_path)

        s0 = res.status.count(0); s1 = res.status.count(1); s2 = res.status.count(2)
        sum_in = sum(prep.n_in_list)
        sum_out = sum(res.n_output_tokens)
        sum_l2 = sum(n_llama)

        # Sidecar AFTER the data file's rename -- see data.write_sidecar for why.
        D.write_sidecar(out_path, {
            "shard_index": si, "job_id": job.job_id, "worker_id": worker_id,
            "host": HOST,
            "n_rows_in": n_rows, "n_rows_out": len(lines),
            "input_fingerprint": man.fingerprint,
            "status_0": s0, "status_1": s1, "status_2": s2,
            "n_prompt_tokens": sum_in, "n_output_tokens": sum_out,
            "n_output_tokens_llama2": sum_l2,
            "bytes": out_path.stat().st_size, "elapsed_s": round(dt, 2),
            "drop_threshold": drop_threshold, "prompt_overhead": overhead,
            # Per-style document counts for this shard. Lets the 25/25/25/25 balance be
            # audited without reading the data back, and makes an accidental re-seed
            # visible immediately.
            "wrap_style_counts": ({st: styles.count(st) for st in WRAP_STYLES}
                                  if styles is not None else None),
            # WHICH GPU produced these rows. On a mixed H200/B200/B300 fleet the attention
            # backend and reduction order differ by architecture, so greedy output can
            # differ slightly between cards. Recording it here is the only way to
            # reconstruct that after the fact -- see docs/HANDOFF_REVIEW.md.
            "gpu_name": gpu_name, "gpu_cc": gpu_cc,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        if claimed is not None:
            D.release_claim(job.output_dir, claimed)   # .done is now the marker
            claimed = None

        prog["shards_completed"] += 1
        prog["docs_completed"] += n_rows
        prog["docs_status_0"] += s0
        prog["docs_status_1"] += s1
        prog["docs_status_2"] += s2
        prog["total_input_tokens"] += sum_in
        prog["total_output_tokens"] += sum_out
        prog["total_output_tokens_llama2"] += sum_l2
        prog["elapsed_seconds"] = time.perf_counter() - t_start
        write_progress(progress_path, prog)

        processed_docs += n_rows
        processed_out_tokens += sum_out

        if prog["shards_completed"] % every == 0:
            elapsed = prog["elapsed_seconds"]
            dps = processed_docs / elapsed if elapsed > 0 else 0.0
            tps = processed_out_tokens / elapsed if elapsed > 0 else 0.0
            remaining = prog["shards_total"] - prog["shards_completed"]
            done_this_run = max(1, prog["shards_completed"] - already)
            avg_rows = processed_docs / done_this_run
            eta_s = (remaining * avg_rows / dps) if dps > 0 else 0.0
            log(f"[w{worker_id}] {job.job_id} shard {si:05d} done "
                f"({prog['shards_completed']}/{prog['shards_total']}) rows={n_rows} "
                f"s0={s0} s1={s1} s2={s2} wall={dt:.1f}s "
                f"out_tok={sum_out} tok/s={tps:.0f} docs/s={dps:.1f} "
                f"ETA={eta_s/3600:.1f}h GPU={gpu_name}")

    finally:
        # A crash mid-shard must not leave a claim nobody will ever complete.
        if claimed is not None:
            D.release_claim(job.output_dir, claimed)

    log(f"[w{worker_id}] {job.job_id} ALL OWNED SHARDS DONE "
        f"({prog['shards_completed']}/{prog['shards_total']}) GPU={gpu_name} {gpu_cc}")
    return 0


# --------------------------------------------------------------------------- status
def status_table(cfg: Config, deep: bool = False) -> str:
    rows = [f"{'JOB':32s} {'SHARDS':>13s} {'ROWS OUT':>16s} {'OUT TOKENS':>14s}  STATE"]
    for j in enumerate_jobs(cfg):
        # Check quietly first: before 02_download_data.py has run there is no manifest,
        # and that is an expected state for a status query, not an error worth shouting
        # about twelve times.
        if not D.manifest_path(cfg, j.arm).exists():
            rows.append(f"{j.job_id:32s} {'-':>13s} {'-':>16s} {'-':>14s}  NO-INPUT")
            continue
        try:
            st = D.verify_job(cfg, j, deep=deep)
        except SystemExit:
            rows.append(f"{j.job_id:32s} {'-':>13s} {'-':>16s} {'-':>14s}  NO-INPUT")
            continue
        rows.append(f"{j.job_id:32s} {st.done:6d}/{st.n_shards:<6d} "
                    f"{st.rows_out:16,d} {st.out_tokens:14,d}  {st.state}"
                    + (f"  ({len(st.problems)} problem(s))" if st.problems else ""))
    return "\n".join(rows)


# --------------------------------------------------------------------------- cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm")
    ap.add_argument("--prompt-id")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--config-root", default=None)
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="process at most N owned shards (smoke tests only)")
    ap.add_argument("--status", action="store_true", help="print the job table and exit")
    ap.add_argument("--verify", action="store_true",
                    help="verify one job's row conservation and exit")
    ap.add_argument("--deep-verify", action="store_true",
                    help="with --verify: also count lines in every output shard")
    ap.add_argument("--reap-claims", action="store_true",
                    help="clear shard claims whose owner is gone (no heartbeat for "
                         "compute.claim_stale_after_s). Safe to run while other nodes are "
                         "working -- live claims are left alone")
    ap.add_argument("--force", action="store_true",
                    help="with --reap-claims: clear EVERY claim regardless of liveness. "
                         "Only when you know nothing is running anywhere.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config_root)

    if args.status:
        print(status_table(cfg, deep=args.deep_verify))
        return 0

    if not args.arm or not args.prompt_id:
        ap.error("--arm and --prompt-id are required (or use --status)")
    job = get_job(cfg, args.arm, args.prompt_id)

    if args.reap_claims:
        D.reap_stale_claims(
            job.output_dir,
            stale_after_s=float(cfg.cluster["compute"].get("claim_stale_after_s", 1800)),
            force=args.force)
        return 0

    if args.verify:
        st = D.verify_job(cfg, job, deep=args.deep_verify)
        print(f"{st.job_id}: {st.state}  shards {st.done}/{st.n_shards}  "
              f"rows_out {st.rows_out:,}  input_rows {D.input_rows(cfg, job.arm):,}")
        for p in st.problems:
            print(f"  PROBLEM: {p}", file=sys.stderr)
        return 0 if (st.state == "DONE" and not st.problems) else 1

    return run_worker(cfg, job, args.worker_id, args.num_workers, args.limit_shards)


if __name__ == "__main__":
    raise SystemExit(main())
