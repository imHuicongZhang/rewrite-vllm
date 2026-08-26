#!/usr/bin/env python3
"""Everything that must be true before run_all.sh will spend a GPU-hour.

Checks run cheapest-first so failures are fast:

   1  placeholders          nothing unresolved in either class
   2  config                validates, engine args are exactly the source's, 10 jobs
   3  gpus                  visible, count matches config, enough VRAM
   4  imports               vllm/torch/transformers/... import, versions match the pins
   5  model                 Qwen + Llama-2 tokenizer present and loadable
   6  prompt parity         empty-doc templated overhead == expected, for all 13
                            (job, template) pairs -- 6 distinct texts, and wrap-inspired's
                            styled pass carries four of them in one job
   7  datasets              the one gated repo resolves AND is readable; each of the five
                            arm folders exists and has a non-empty `text` column
   8  hf token              valid, and the write token really has write scope
   9  disk                  free space vs estimated output bytes (assumption printed)
  10  smoke                 8 documents end-to-end: generate -> trim -> write -> verify,
                            printing before/after so a human can see the prompt working

    python scripts/preflight.py                 # everything
    python scripts/preflight.py --skip-smoke    # everything except step 10
    python scripts/preflight.py --only 1,2,3
    python scripts/preflight.py --emit-overheads
    python scripts/preflight.py --verify-tokenizer-batching
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PINS = {"torch": "2.11.0", "vllm": "0.22.0", "transformers": "5.9.0",
        "tokenizers": "0.22.2", "huggingface_hub": "1.17.0", "numpy": "2.3.5",
        "pyarrow": "24.0.0"}

_results = []


def check(name):
    def deco(fn):
        fn._check_name = name
        return fn
    return deco


def ok(msg):
    print(f"   \033[32mOK\033[0m   {msg}")


def warn(msg):
    print(f"   \033[33mWARN\033[0m {msg}")


def fail(msg):
    print(f"   \033[31mFAIL\033[0m {msg}")
    return False


# ------------------------------------------------------------------ 1 placeholders
def c1_placeholders(cfg=None, args=None) -> bool:
    rc = subprocess.call([sys.executable, str(REPO / "scripts" / "check_placeholders.py"),
                          "--root", str(REPO)])
    if rc != 0:
        return fail("unresolved placeholders (or missing .env) -- see the list above")
    ok("no unresolved placeholders, .env present")
    return True


# ------------------------------------------------------------------ 2 config
def c2_config(cfg, args) -> bool:
    from rewrite.config import describe_jobs, enumerate_jobs
    jobs = enumerate_jobs(cfg)
    n = len(jobs)
    if n != int(cfg.data["expected_jobs"]):
        return fail(f"{n} jobs enumerated, expected {cfg.data['expected_jobs']}")
    ok(f"{n} rewrite jobs across {len(cfg.arms)} arms "
       f"({sum(a.docs for a in cfg.arms):,} documents, "
       f"{sum(a.source_tokens_llama2 for a in cfg.arms)/1e9:.1f}B source tokens x "
       f"{n // max(1, len(cfg.arms))} passes)")
    ok("raw half NOT downloaded and NOT rewritten: shared-core 20.0B/17.9M docs, "
       "quality-base 50.0B/37.3M docs (see configs/data.yaml header)")
    print()
    print(describe_jobs(cfg))
    print()
    ok(f"engine args are exactly the source's: {sorted(cfg.vllm['engine'])}")
    ok(f"gpu_memory_utilization={cfg.vllm['engine']['gpu_memory_utilization']} "
       f"dtype={cfg.vllm['engine']['dtype']} max_model_len={cfg.max_model_len} "
       f"max_tokens={cfg.max_tokens}")
    return True


# ------------------------------------------------------------------ 3 gpus
def c3_gpus(cfg, args) -> bool:
    try:
        import torch
    except Exception as e:
        return fail(f"cannot import torch: {e}")
    if not torch.cuda.is_available():
        return fail("torch.cuda.is_available() is False -- no usable GPU")
    n = torch.cuda.device_count()
    if n != cfg.num_gpus:
        return fail(f"torch sees {n} GPU(s) but cluster.yaml compute.num_gpus is "
                    f"{cfg.num_gpus}. Fix the config, or set CUDA_VISIBLE_DEVICES.")
    good = True
    archs, models = {}, {}
    try:
        built = set(torch.cuda.get_arch_list())      # e.g. {'sm_90', 'sm_100', ...}
    except Exception:
        built = set()

    for i in range(n):
        pr = torch.cuda.get_device_properties(i)
        gib = pr.total_memory / 2**30
        cc = f"sm_{pr.major}{pr.minor}"
        archs.setdefault(cc, []).append(i)
        models.setdefault(pr.name, []).append(i)
        line = f"GPU {i}: {pr.name}  {gib:.1f} GiB  {cc}"
        if gib < 23.5:
            good = fail(line + "  -- too small for Qwen2.5-7B bf16 at "
                               "gpu_memory_utilization=0.85 / max_model_len=32768")
            continue

        # Does this torch build actually have kernels for this card?
        if not built:
            warn(line + "  -- could not read torch.cuda.get_arch_list()")
        elif cc in built:
            ok(line + "  (native kernels present)")
        else:
            same_major = [a for a in built
                          if a.startswith(f"sm_{pr.major}") or a.startswith(f"sm_{pr.major}0")]
            if same_major:
                warn(line + f"  -- no exact {cc} kernels in this torch build "
                            f"({sorted(built)}); it will PTX-JIT from {sorted(same_major)}. "
                            "That works but is slower on first use, and the kernels are not "
                            "the tuned ones. Watch the first job's tok/s.")
            else:
                good = fail(line + f"  -- this torch build has NO kernels for {cc} "
                                   f"(it has {sorted(built)}). vLLM will not run here. "
                                   "STOP and ask Wytro; do not swap in another torch build "
                                   "on your own.")

    kv = (0.85 * min(torch.cuda.get_device_properties(i).total_memory
                     for i in range(n)) / 2**30) - 14.2
    ok(f"estimated KV cache on the SMALLEST card after ~14.2 GiB of bf16 weights: "
       f"~{kv:.1f} GiB")

    # ---- ARCHITECTURE ALLOW-LIST: Blackwell only (B200 sm_100 / B300 sm_103) ----
    # This is a workload constraint from configs/data.yaml, not a machine setting. The
    # original data was generated on H100 (sm_90) with FlashAttention v3, which is
    # Hopper-only; vLLM picks a different attention backend on Blackwell, and greedy
    # decoding at temperature=0 is not bitwise identical across architectures. Restricting
    # every job to one architecture family removes that confound entirely.
    print()
    con = cfg.data["compute_constraints"]
    allowed_idx, rejected = [], []
    for i in range(n):
        pr = torch.cuda.get_device_properties(i)
        cc = f"sm_{pr.major}{pr.minor}"
        okk, level, msg = cfg.check_gpu_arch(cc, pr.name)
        if okk and level == "ok":
            ok(f"GPU {i}: {msg}")
            allowed_idx.append(i)
        elif okk:
            warn(f"GPU {i}: {msg}")
            allowed_idx.append(i)
        else:
            good = fail(f"GPU {i}: {msg}")
            rejected.append(i)

    if rejected:
        # CUDA_VISIBLE_DEVICES has already remapped indices, so tell him what to do in
        # terms he can act on rather than printing a list that may not match nvidia-smi.
        print()
        print("     This workload runs on Blackwell only. Restrict the run to the allowed "
              "cards:")
        print("       1. nvidia-smi --query-gpu=index,name,compute_cap --format=csv")
        print(f"       2. in configs/cluster.yaml set gpu_ids to just the "
              f"{sorted(con['allowed_gpu_arch'])} cards,")
        print(f"          and set num_gpus to how many that is")
        print("       3. re-run preflight")
        print("     Do NOT widen compute_constraints.allowed_gpu_arch in data.yaml to make "
              "this pass -- that is Wytro's experimental design, not a setting.")

    if len(archs) > 1:
        print()
        warn(f"more than one compute capability in use: {sorted(archs)}")
        print("     Allowed (B200 and B300 are both Blackwell and share an attention "
              "backend), but every shard's .done sidecar records gpu_name and gpu_cc so "
              "the mixture stays auditable. Do not delete the sidecars, and do not change "
              "compute.gpu_ids partway through the 10 jobs.")
    else:
        ok(f"single architecture across all {n} GPU(s): {list(archs)[0]}")

    return good


# ------------------------------------------------------------------ 4 imports
def c4_imports(cfg, args) -> bool:
    import importlib.metadata as md
    good = True
    for pkg, want in PINS.items():
        try:
            got = md.version(pkg)
        except Exception as e:
            good = fail(f"{pkg}: not installed ({e})")
            continue
        base = got.split("+")[0]
        if base != want:
            good = fail(f"{pkg} is {got}, pinned at {want}. A different build changes "
                        "generation behaviour -- do NOT proceed; re-run "
                        "scripts/00_setup_env.sh or ask Wytro.")
        else:
            ok(f"{pkg} {got}")
    for extra in ("datasets", "hf_transfer", "zstandard", "yaml"):
        try:
            __import__(extra)
            ok(f"{extra} importable")
        except Exception as e:
            good = fail(f"{extra}: {e}")
    return good


# ------------------------------------------------------------------ 5 model
def c5_model(cfg, args) -> bool:
    from rewrite import engine as E
    md_, ld = E.model_dir(cfg), E.llama2_dir(cfg)
    if not (md_ / "config.json").exists():
        return fail(f"model missing at {md_} -- run scripts/01_download_model.py")
    ok(f"model at {md_}")
    try:
        qtok = E.load_qwen_tokenizer(cfg)
        ok(f"Qwen tokenizer loads (vocab {getattr(qtok,'vocab_size','?')})")
    except Exception as e:
        return fail(f"Qwen tokenizer: {e}")
    try:
        ltok = E.load_llama2_tokenizer(cfg)
    except SystemExit:
        return False
    except Exception as e:
        return fail(f"Llama-2 tokenizer at {ld}: {e}")
    probe = "Hello world. This is a tokenizer test."
    n = len(ltok(probe, add_special_tokens=False).input_ids)
    if n != 10:
        return fail(f"Llama-2 tokenizer golden string gives {n} tokens, expected 10 -- "
                    "this is not the tokenizer the source used, so output token counts "
                    "would not be comparable")
    ok("Llama-2 tokenizer matches the source's golden string (10 tokens)")
    return True


# ------------------------------------------------------------------ 6 prompt parity
def c6_prompts(cfg, args) -> bool:
    from rewrite import engine as E
    from rewrite.config import enumerate_jobs, resolve_drop_threshold
    qtok = E.load_qwen_tokenizer(cfg)
    good = True
    seen = {}
    n_texts = 0
    for j in enumerate_jobs(cfg):
        drop, derived = resolve_drop_threshold(j.prompt, cfg.max_model_len, cfg.max_tokens)
        # A job can carry more than one prompt text: wrap-inspired's styled pass has four,
        # and all four are gated. Checking only the first would leave three quarters of
        # that arm unprotected.
        for label, got, exp in E.check_overheads(qtok, cfg, j.prompt):
            n_texts += 1
            tag = (f"{j.arm + '__' + label:34s} overhead={got:4d} (expected {exp:4d})  "
                   f"drop={drop}{'*' if derived else ' '}")
            if got != exp:
                good = fail(tag + "  <-- prompt/template/tokenizer differs from the source")
            else:
                ok(tag)
            seen[f"{j.arm}__{label}"] = got
    # Two different counts, and conflating them is how the header comment drifted:
    #   n_texts  = (job, template) pairs asserted -- 13, because wrap-inspired's styled
    #              pass carries four templates in one job
    #   distinct = distinct prompt TEXTS -- 6, since the four grounded arms share p1 and
    #              all five share the distill p2
    n_distinct = len({t for j in enumerate_jobs(cfg) for _, t, _ in j.prompt.overheads()})
    ok(f"{n_texts} (job, template) pair(s) asserted across "
       f"{len(enumerate_jobs(cfg))} jobs; {n_distinct} distinct prompt texts")
    if args.emit_overheads:
        print("\n   measured overheads (for configs/data.yaml prompt_defs):")
        print("   " + json.dumps(seen, indent=2).replace("\n", "\n   "))
    if not good:
        print("\n   Do NOT edit expected_overhead to make this pass. That integer is the "
              "\n   proof that the prompt text, the chat template and the tokenizer are "
              "\n   the ones that produced the original data.")
    return good


# ------------------------------------------------------------------ 7 datasets
def c7_datasets(cfg, args) -> bool:
    """The input is ONE gated repo. Resolve it, prove we can actually READ it, then probe
    each arm's folder.

    The gating check is the point of this whole check. The repo is
    `extra_gated_review: manual`, so an account that has not been granted access can list
    the repo's files perfectly happily and only fails when it tries to fetch bytes. That
    failure would otherwise surface partway through a multi-hour download on day one --
    or worse, on a worker node hours after the head node succeeded from a warm cache.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=cfg.env.get("HF_TOKEN") or None)
    good = True

    try:
        info = api.dataset_info(cfg.hf_repo_id, revision=cfg.hf_revision, files_metadata=False)
    except Exception as e:
        return fail(f"cannot resolve {cfg.hf_repo_id}@{cfg.hf_revision[:12]}: {e}\n"
                    f"   If this says 'gated' or '401/403', the account behind HF_TOKEN "
                    f"has not been granted access.\n"
                    f"   ASK WYTRO to approve it at "
                    f"https://huggingface.co/datasets/{cfg.hf_repo_id}/settings")
    ok(f"{cfg.hf_repo_id}  sha={str(info.sha)[:10]}  "
       f"gated={getattr(info, 'gated', None)}  private={getattr(info, 'private', None)}")
    if str(getattr(info, "sha", "")) != cfg.hf_revision:
        good = fail(f"resolved sha {str(info.sha)[:12]} != pinned hf.revision "
                    f"{cfg.hf_revision[:12]} -- the pin should name an immutable commit")

    # Prove READ access, not merely metadata access. hf_hub_download of the smallest
    # sidecar costs a few hundred KB and is the only thing that actually exercises gating.
    from huggingface_hub import hf_hub_download
    probe_file = f"{cfg.arms[0].subdir}/stats.json"
    try:
        hf_hub_download(repo_id=cfg.hf_repo_id, repo_type="dataset",
                        revision=cfg.hf_revision, filename=probe_file,
                        cache_dir=str(cfg.paths["hf_cache"]),
                        token=cfg.env.get("HF_TOKEN") or None)
        ok(f"{'':22s} read access confirmed (fetched {probe_file})")
    except Exception as e:
        return fail(f"metadata is readable but FILE CONTENT is not: {e}\n"
                    f"   This is exactly the gated-access failure. Listing a gated repo "
                    f"succeeds without access; downloading does not.\n"
                    f"   ASK WYTRO to grant the HF_TOKEN account access to "
                    f"{cfg.hf_repo_id} before starting the run.")

    repo_files = set(api.list_repo_files(cfg.hf_repo_id, repo_type="dataset",
                                         revision=cfg.hf_revision))
    for a in cfg.arms:
        n_parts = sum(1 for f in repo_files if f.startswith(f"{a.subdir}/data/")
                      and f.endswith(".parquet"))
        if not n_parts:
            good = fail(f"{a.name:22s} no parquet under {a.subdir}/data/ -- is `subdir` "
                        f"right? repo folders: "
                        f"{sorted({f.split('/')[0] for f in repo_files if '/' in f})}")
            continue
        ok(f"{a.name:22s} {a.subdir}/data/  {n_parts} parquet files  "
           f"{a.docs:,} docs declared  {len(a.prompts)} prompt(s)")
        try:
            from rewrite import data as D
            D.probe_arm(cfg, a.name, log=lambda m: None)
            ok(f"{'':22s} '{cfg.text_column}' column present and non-empty")
        except SystemExit:
            good = False
        except Exception as e:
            good = fail(f"{a.name}: text-column probe failed: {e}")
    return good


# ------------------------------------------------------------------ 8 hf token
def c8_token(cfg, args) -> bool:
    from huggingface_hub import HfApi
    good = True
    read_tok = cfg.env.get("HF_TOKEN")
    write_tok = cfg.env.get("HF_TOKEN_WRITE") or read_tok
    if not read_tok:
        return fail("HF_TOKEN missing from .env")
    try:
        who = HfApi(token=read_tok).whoami()
        ok(f"HF_TOKEN valid (user: {who.get('name')})")
    except Exception as e:
        return fail(f"HF_TOKEN invalid: {e}")

    # Upload is out of this pipeline's scope and ships disabled, so on the normal run
    # there is nothing to check a write token against. Probing it anyway produced a
    # warning about a token nobody needs -- noise in the one report that has to be read
    # carefully.
    if not cfg.data["upload"]["enabled"]:
        ok("upload is disabled (configs/data.yaml upload.enabled: false) -- no write "
           "token or output namespace needed")
        return good

    try:
        who = HfApi(token=write_tok).whoami()
    except Exception as e:
        return fail(f"HF_TOKEN_WRITE invalid: {e}")
    auth = (who.get("auth") or {}).get("accessToken") or {}
    role = auth.get("role") or auth.get("fineGrained")
    has_write = False
    if isinstance(role, str):
        has_write = role in ("write", "admin")
    elif isinstance(role, dict):
        blob = json.dumps(role)
        has_write = "write" in blob or "manage" in blob
    if has_write:
        ok(f"HF_TOKEN_WRITE has write scope (user: {who.get('name')})")
    else:
        warn(f"could not confirm write scope for HF_TOKEN_WRITE (role={role!r}). "
             "Upload will fail at the very end if it is read-only -- check it now.")
    tpl = cfg.data["upload"]["repo_template"]
    ns = tpl.split("/")[0] if "/" in tpl else ""
    orgs = {o.get("name") for o in (who.get("orgs") or [])}
    if ns and ns != who.get("name") and ns not in orgs:
        warn(f"upload namespace '{ns}' is neither your username ({who.get('name')}) nor "
             f"one of your orgs {sorted(orgs)}. Upload may fail.")
    else:
        ok(f"upload namespace '{ns}' is accessible")
    return good


# ------------------------------------------------------------------ 9 disk
def c9_disk(cfg, args) -> bool:
    from rewrite import data as D
    o = cfg.data["output"]
    bpt = float(o["bytes_per_output_token"])
    bpr = float(o["bytes_per_row_overhead"])

    rows_by_arm, known = {}, True
    for a in cfg.arms:
        try:
            rows_by_arm[a.name] = D.load_manifest(cfg, a.name).total_rows
        except SystemExit:
            known = False

    n_arms = len(cfg.arms)
    # Per-arm, measured. There is no est_output_tokens_per_arm scalar any more: the real
    # per-arm figures span 35.9B to 99.5B, so one number could not express them and the
    # old 100B x 5 = 500B overstated the total by 1.91x. Provenance for the ratios these
    # come from: docs/DESIGN_DELTA.md section 5.
    est_tokens = float(cfg.est_output_tokens_total)
    if known:
        est_rows = sum(rows_by_arm[a.name] * len(a.prompts) for a in cfg.arms)
    else:
        # Declared doc counts, which are the same numbers the estimates were built from.
        est_rows = float(cfg.est_output_rows_total)

    raw = est_tokens * bpt + est_rows * bpr
    comp = raw * 0.30 if cfg.compression == "zstd" else raw

    print(f"   sizing assumption: {bpt} bytes/output token "
          f"(~3.8 chars/token UTF-8 + ~10% JSON escaping) + {bpr:.0f} bytes/row of JSON "
          f"envelope")
    print(f"   estimate: {est_tokens/1e9:.1f}B output tokens over "
          f"{est_rows/1e6:.0f}M rows across {n_arms} arms"
          + ("" if known else "   [from data.yaml declared docs; no manifests yet]"))
    for a in cfg.arms:
        t = sum(pr.est_output_tokens for pr in a.prompts)
        print(f"     {a.name:22s} {a.source_tokens_llama2/1e9:6.1f}B src x "
              f"{len(a.prompts)} -> {t/1e9:6.2f}B out   "
              f"(r={', '.join(f'{pr.r:.4f}' for pr in a.prompts)})")
    print(f"   -> raw {raw/2**40:.2f} TiB; on disk with compression="
          f"{cfg.compression}: {comp/2**40:.2f} TiB per copy"
          + ("" if known else "   [row counts not known yet -- estimated]"))
    print(f"   out_root holds TWO copies at rest -- raw/ (the trim is in place, so it is "
          f"not consumed)")
    print(f"   and shuffled/ -- so budget {2*comp/2**40:.2f} TiB there, not "
          f"{comp/2**40:.2f} TiB.")

    good = True
    for key in ("out_root", "data_root", "tmp_root", "model_dir"):
        p = cfg.paths[key]
        p.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(p).free
        # data_root holds the downloaded remainders (662 GB of parquet across the five
        # arms, per the Hub) plus the re-sharded copy, not a fraction of the output.
        #
        # out_root is TWO full copies, not one. postprocess.in_place: true means the trim
        # rewrites raw/ in place rather than consuming it, and the shuffle then writes
        # shuffled/ as a second complete copy of the same rows. Both are still there when
        # the run ends. Gating on one copy passed a disk that then filled part-way through
        # a shuffle -- and the shuffle unlinks its bucket files as it consumes them, so
        # running out of space there loses that job's entire shuffle, days into the run.
        # That is the one stage where "free space and re-run" is not a clean recovery.
        need = {"out_root": 2 * comp,
                "data_root": 1.5 * 10**12,
                "tmp_root": comp / max(1, n_arms) * 1.2,
                "model_dir": 20 * 2**30}[key]
        line = f"{key:10s} {str(p):50s} free {free/2**40:.2f} TiB (needs ~{need/2**40:.2f} TiB)"
        if free < need:
            good = fail(line + "  <-- NOT ENOUGH")
        else:
            ok(line)
    if not good:
        print("\n   Disk is a hard blocker, but a mid-run disk-full is recoverable: every "
              "\n   shard is atomic and marked with a .done sidecar, so free space and "
              "\n   re-run the same command -- finished shards are skipped.")
    return good


# ------------------------------------------------------------------ 10 smoke
def c10_smoke(cfg, args) -> bool:
    """8 documents, end to end, through three different trim rules."""
    from rewrite import data as D
    from rewrite import engine as E
    from rewrite import postprocess as PP
    from rewrite.wrap_styles import assign_wrap_styles
    from rewrite.config import enumerate_jobs, resolve_drop_threshold

    jobs = enumerate_jobs(cfg)
    picks, seen_rules = [], set()
    for j in jobs:
        if j.prompt.trim not in seen_rules:
            picks.append(j)
            seen_rules.add(j.prompt.trim)
    print(f"   exercising {len(picks)} job(s), one per trim rule: "
          f"{[j.job_id for j in picks]}")

    smoke_root = cfg.paths["out_root"] / "_smoke"
    E.set_source_env(cfg)
    qtok = E.load_qwen_tokenizer(cfg)
    ltok = E.load_llama2_tokenizer(cfg)
    llm = E.build_llm(cfg)
    PP._init_worker(str(E.llama2_dir(cfg)))

    good = True
    for job in picks:
        shards = D.shard_paths(cfg, job.arm)
        if not shards:
            good = fail(f"{job.job_id}: no input shards -- run 02_download_data.py")
            continue
        tbl = D.read_shard(shards[0][1]).slice(0, 8)
        texts = tbl.column(cfg.text_column).to_pylist()
        doc_ids = tbl.column("doc_id").to_pylist()
        shas = tbl.column("source_text_sha1").to_pylist()
        rec_ids = (tbl.column("record_id").to_pylist()
                   if "record_id" in tbl.column_names else [""] * len(texts))

        drop, _ = resolve_drop_threshold(job.prompt, cfg.max_model_len, cfg.max_tokens)
        # Shard 0, so the smoke test exercises the same style draw the real run will make
        # for that shard -- not a fresh one.
        smoke_styles = (assign_wrap_styles(shards[0][0], len(texts))
                        if job.prompt.mode == "wrap_multi" else None)
        if smoke_styles:
            print(f"     styles for shard {shards[0][0]}: {smoke_styles}")
        prep = E.prepare_batch(qtok, cfg, texts, job.prompt, drop, styles=smoke_styles)
        res = E.run_batch(llm, prep)
        n_l2 = [0] * len(texts)
        if prep.keep_idx:
            for j2, c in zip(prep.keep_idx,
                             E.count_llama2(ltok, [res.rewritten[j2] for j2 in prep.keep_idx])):
                n_l2[j2] = c

        rows = [{
            "doc_id": doc_ids[i], "arm": job.arm, "prompt_id": job.prompt.id,
            "source_text_sha1": shas[i], "rewritten_text": res.rewritten[i],
            "finish_reason": res.finish_reason[i], "n_prompt_tokens": prep.n_in_list[i],
            "n_output_tokens": res.n_output_tokens[i], "status": res.status[i],
            "n_output_tokens_llama2": n_l2[i],
            "record_id": rec_ids[i],
            "wrap_style": smoke_styles[i] if smoke_styles else "",
        } for i in range(len(texts))]

        if len(rows) != len(texts):
            good = fail(f"{job.job_id}: {len(rows)} rows from {len(texts)} inputs")
            continue

        rule = PP.TRIM_RULES[job.prompt.trim]
        n_stripped = 0
        for r in rows:
            if r["status"] == 2:
                new, did = rule(r["rewritten_text"])
                if did:
                    r["rewritten_text"] = new
                    r["n_output_tokens_llama2"] = PP._n_llama(new)
                    n_stripped += 1

        d = smoke_root / job.arm / job.prompt.id
        d.mkdir(parents=True, exist_ok=True)
        out_p = d / f"part_00000{cfg.shard_suffix}"
        fh, close = D.open_jsonl_write(Path(str(out_p) + ".tmp"),
                                       compress=(cfg.compression == "zstd"))
        try:
            fh.write(("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
                      + "\n").encode("utf-8"))
        finally:
            close()
        os.replace(str(out_p) + ".tmp", out_p)
        n_back = D.count_jsonl_rows(out_p)

        print(f"\n   --- {job.job_id}  (trim rule: {job.prompt.trim}) ---")
        print(f"   rows in={len(texts)} out={n_back} "
              f"status0={sum(1 for r in rows if r['status']==0)} "
              f"status1={sum(1 for r in rows if r['status']==1)} "
              f"status2={sum(1 for r in rows if r['status']==2)} trimmed={n_stripped}")
        if n_back != len(texts):
            good = fail(f"{job.job_id}: wrote {n_back} rows, expected {len(texts)}")
        for r in rows[:2]:
            src = next(t for i, t in enumerate(texts) if doc_ids[i] == r["doc_id"])
            print(f"\n     BEFORE (source doc {r['doc_id']}, first 300 chars):")
            print("       " + repr((src or "")[:300]))
            print(f"     AFTER  (rewritten, first 300 chars, status={r['status']}):")
            print("       " + repr((r["rewritten_text"] or "")[:300]))
        print(f"\n   wrote {out_p}")
    print(f"\n   Eyeball the pairs above: the rewrite should look like the prompt asked "
          f"for.\n   Smoke output is in {smoke_root} and is safe to delete.")
    return good


CHECKS = [
    ("placeholders", c1_placeholders, False),
    ("config", c2_config, True),
    ("gpus", c3_gpus, True),
    ("imports", c4_imports, True),
    ("model", c5_model, True),
    ("prompt-parity", c6_prompts, True),
    ("datasets", c7_datasets, True),
    ("hf-token", c8_token, True),
    ("disk", c9_disk, True),
    ("smoke", c10_smoke, True),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config-root", default=None)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated check numbers, e.g. 1,2,6")
    ap.add_argument("--emit-overheads", action="store_true",
                    help="print measured prompt overheads as JSON")
    ap.add_argument("--verify-tokenizer-batching", action="store_true",
                    help="prove batched tokenizer calls equal the source's per-doc loop")
    args = ap.parse_args(argv)

    only = None
    if args.only:
        only = {int(x) for x in args.only.split(",") if x.strip()}

    cfg = None
    results = []
    t0 = time.time()
    for i, (name, fn, needs_cfg) in enumerate(CHECKS, 1):
        if only and i not in only:
            continue
        if name == "smoke" and args.skip_smoke:
            print(f"\n[{i}/10] {name}: SKIPPED (--skip-smoke)")
            continue
        print(f"\n[{i}/10] {name}")
        if needs_cfg and cfg is None:
            from rewrite.config import load_config
            cfg = load_config(args.config_root)
        try:
            passed = bool(fn(cfg, args))
        except SystemExit as e:
            passed = (e.code in (0, None))
        except Exception as e:                                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            passed = fail(f"{name} raised {type(e).__name__}: {e}")
        results.append((name, passed))

    if args.verify_tokenizer_batching and cfg is not None:
        print("\n[extra] tokenizer batching equivalence")
        from rewrite import data as D, engine as E
        ltok = E.load_llama2_tokenizer(cfg)
        a = next(x for x in cfg.arms if x.rewrite)
        sp = D.shard_paths(cfg, a.name)
        if sp:
            texts = D.read_shard(sp[0][1]).column(cfg.text_column).to_pylist()[:1000]
            batched = E.count_llama2(ltok, texts)
            per_doc = [len(ltok(t or "", add_special_tokens=False).input_ids) for t in texts]
            (ok if batched == per_doc else fail)(
                f"batched == per-document counts over {len(texts)} docs")
            results.append(("tokenizer-batching", batched == per_doc))
        else:
            warn("no shards yet; skipped")

    print("\n" + "=" * 72)
    bad = [n for n, p in results if not p]
    for n, p in results:
        print(f"  {'PASS' if p else 'FAIL'}  {n}")
    print(f"  ({time.time()-t0:.0f}s)")
    if bad:
        print(f"\nPREFLIGHT FAILED: {', '.join(bad)}")
        print("run_all.sh will not proceed. Fix the failures above and re-run.")
        return 1
    print("\nPREFLIGHT PASSED -- safe to run scripts/run_all.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
