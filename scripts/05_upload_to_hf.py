#!/usr/bin/env python3
"""Push each finished (arm, prompt) output to its own HuggingFace dataset repo.

DISABLED BY DEFAULT, AND THAT IS DELIBERATE. configs/data.yaml ships upload.enabled:
false because delivery of the finished data is arranged separately and is not this
pipeline's job. The run ends with postprocessed parquet on disk under out_root/shuffled/
-- see docs/GUIDE_FOR_TIANJIAN.md section 3.6. This script is kept, and works; it is
disabled, not deleted. To use it, set BOTH upload.enabled: true AND upload.repo_template.

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
    # load_config already refuses a blank or placeholder-shaped template when upload is
    # enabled, so reaching here with one means this function was called from somewhere
    # that bypassed the config gate. Say so rather than pushing to a repo named "".
    if not tpl:
        stop("upload.repo_template is empty -- refusing to derive a repo name from it. "
             "Set it in configs/data.yaml alongside upload.enabled: true.")
    return tpl.format(arm=job.arm, prompt_id=job.prompt.id)


WRAP_CAVEAT = """
## Style assignment — read before subsampling or stratifying this arm

Every document in this arm was rewritten **once**, with **one of four styles** chosen
uniformly at random per document: `easy`, `hard`, `wiki`, `qa`. The style that each row
actually received is in the **`wrap_style`** column. The choice is seeded on
`(42, shard_index)` and is reproducible.

This arm is therefore **not** four rewrites per document. It is one styled rewrite plus
the shared distill rewrite, exactly like every other arm's two passes.

### Documents are balanced across styles; tokens are not

Uniform *per-document* assignment does not give uniform *token* shares, because the four
styles expand very differently. Measured on the originating 1.5B run:

| style | doc share | token share | tokens/doc |
|---|---:|---:|---:|
| easy | 25.05% | 14.23% | 223 |
| hard | 24.91% | 33.55% | 530 |
| wiki | 25.03% | 23.16% | 364 |
| qa | 25.02% | 29.05% | 457 |

Documents land within 25.0% ± 0.1pp of uniform; token shares spread **2.37×**. The
originating pipeline applied no correction, and neither did this one.

The consequence for anyone cutting this arm to a token budget: **a uniform draw over
tokens is not a uniform draw over styles.** It will over-represent `hard` and
under-represent `easy` by roughly that 2.37× factor. If style balance matters for your
use, stratify on `wrap_style` rather than sampling rows uniformly.

The originating pipeline also drew this arm's distill supplement with a seeded *random*
draw rather than a quality sort — recorded in its own manifest as
`"distill_selection": "seeded_random_seed42_no_quality_sort"` — to keep the arm comparable
with the others.
"""


REWIRE_CAVEAT = """
## This arm is UNFILTERED — it must be cut to budget before training

`rewire-inspired` was given a doubled source budget (120B rather than 60B, κ=2)
specifically so that a post-rewrite filter would have room to work: score the **rewritten**
text with fastText, sort descending, and fill to the arm's **30B** training-token budget.

**That filter is not part of this pipeline.** What is published here is the full
~99.5B-token output. Training on it as-is would give this arm 3.3× the tokens every other
arm gets and would skip the quality selection the arm's design depends on.

### It is a fixed-budget fill, not "the top half"

From the originating 1.5B run (`10_postprocess/_step4_rewrite_summary.json`):

| | |
|---|---|
| pipeline | rewrite broadly → fastText-score the rewritten text → keep top 5B |
| selection | token-budget-matched top-5B (**not** the paper's top-10%) |
| pool | 16,221,811,013 tokens |
| kept | 5,000,000,351 tokens — **30.8% retained** |
| target / overshoot | 5,000,000,000 / 351 — `filled: true` |
| realized score cutoff | 0.1145634651184082 |

The cutoff is the **output** of filling to the target, not an input to the filter.

**Recommended:** reproduce that — `fill_to(order_by_score_desc, tokens, 30e9)` — rather than
porting `0.11456` as a constant. This corpus is a different document population from the
1.5B one, so its rewritten-text score distribution will differ; a budget fill self-corrects,
a fixed threshold does not. Record the realized cutoff and retained fraction either way.

**Headroom is comfortable.** Required retention here is `30B / 99.5B = 30.2%`, against
30.8% realized at 1.5B — κ=2 was sized so the pool-to-budget ratio carries over.
"""


def dataset_card(cfg, job, files, nbytes) -> str:
    """A README.md for the Hub repo.

    The point is that whoever later decides how to subsample these datasets may be neither
    Wytro nor Tianjian, and will meet the data long after this repository. Anything they
    need in order not to misuse it has to travel with the data, not sit in a review doc.
    """
    arm = cfg.arm(job.arm)
    drop, derived = __import__("rewrite.config", fromlist=["x"]).resolve_drop_threshold(
        job.prompt, cfg.max_model_len, cfg.max_tokens)
    eng, smp = cfg.vllm["engine"], cfg.vllm["sampling"]
    n_prompts = len(arm.prompts)
    card = f"""---
tags: [synthetic, rewritten, {job.arm}]
---

# {job.arm} — prompt {job.prompt.id}

One of **{n_prompts}** rewrites of the `{job.arm}` corpus. Each prompt rewrites the
**entire** corpus, so this dataset has exactly as many rows as the input
and `{job.arm}` has {n_prompts} datasets that differ only in the prompt used.

Generated with `{cfg.vllm['model']['repo_id']}` under vLLM.

## Generation settings

| | |
|---|---|
| prompt mode | `{job.prompt.mode}` |
| trim rule applied | `{job.prompt.trim}` |
| sampling | greedy: `temperature={smp['temperature']}`, `top_p={smp['top_p']}`, `max_tokens={smp['max_tokens']}` (per document: `min(max_tokens, max_model_len - n_prompt_tokens)`) |
| engine | `dtype={eng['dtype']}`, `max_model_len={eng['max_model_len']}`, `gpu_memory_utilization={eng['gpu_memory_utilization']}`, `tensor_parallel_size={eng['tensor_parallel_size']}` |
| over-length documents | **dropped, never truncated**, above {drop} templated tokens{' (derived as max_model_len - max_tokens)' if derived else ''} → `status=0` |
| shards | {files} files, {nbytes / 2**30:.1f} GiB |

## Columns

| column | meaning |
|---|---|
| `doc_id` | identity key; join back to the input corpus on this |
| `arm`, `prompt_id` | which arm and which of its prompts produced this row |
| `source_text_sha1` | SHA-1 of the source document (integrity, **not** a join key — duplicate documents share a hash) |
| `rewritten_text` | model output, with the arm's trim rule applied |
| `finish_reason` | raw from vLLM |
| `status` | `0` dropped (too long, never generated) · `1` truncated at the output cap · `2` clean stop |
| `n_prompt_tokens`, `n_output_tokens` | Qwen tokenizer |
| `n_output_tokens_llama2` | Llama-2 tokenizer — **use this one for token budgeting**, and add 1 per document for BOS |
| `record_id` | the WARC-Record-ID of the source document, e.g. `<urn:uuid:4038e004-...>`, verbatim from the input dataset. **Not a unique key** — 0.0662% of source documents share one (the same WARC record sampled twice), so a join on it alone fans out. Use `doc_id` to join; `record_id` exists so a `doc_id` shift is repairable against a rebuilt master, with `payload_digest` breaking ties |
| `wrap_style` | which of `easy`/`hard`/`wiki`/`qa` produced this row. Populated only for `wrap-inspired`'s styled pass; the empty string everywhere else |

Rows with `status != 2` are present but were not cleanly generated. Filter to `status == 2`
for training use; the others are retained so row counts match the input exactly.
"""
    if job.arm == "wrap-inspired":
        card += WRAP_CAVEAT
    if job.arm == "rewire-inspired":
        card += REWIRE_CAVEAT
    return card


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

    # The gate. Refuse clearly and early, including under --dry-run: a dry run that
    # cheerfully reports what it WOULD push is the wrong answer when nothing should be
    # pushed at all.
    if not cfg.data["upload"]["enabled"]:
        stop("upload is DISABLED in configs/data.yaml (upload.enabled: false).\n"
             "  That is the shipped default: delivery of the finished data is arranged\n"
             "  separately, and this pipeline's job ends at postprocess. The finished\n"
             "  data is already complete on disk at:\n"
             f"      {cfg.paths['out_root']}/shuffled/<arm>/<prompt_id>/part_NNNNN.parquet\n"
             "  If you really do want to upload, set BOTH in configs/data.yaml:\n"
             "      upload.enabled: true\n"
             "      upload.repo_template: your-org/rewrite-{arm}-{prompt_id}\n"
             "  and put a write-scoped token in HF_TOKEN_WRITE. See GUIDE section 3.6.")

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

        # Ship the card with the data. For wrap-inspired this is the only place the
        # style-assignment note reaches the person who will subsample it.
        card = dataset_card(cfg, job, len(files), nbytes)
        card_p = src / "README.md"
        D.atomic_write_text(card, card_p)
        with_retry(lambda: api.upload_file(
            path_or_fileobj=str(card_p), path_in_repo="README.md",
            repo_id=name, repo_type="dataset",
            commit_message="dataset card"), f"upload card {name}")
        _note = {"wrap-inspired": "  (includes the wrap style-assignment note)",
                 "rewire-inspired": "  (includes the UNFILTERED / cut-to-30B warning)"}
        print(f"    card uploaded" + _note.get(job.arm, ""))
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
