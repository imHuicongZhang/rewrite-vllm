# SOURCE_INVENTORY

Inventory of the source pipeline at `/scratch/bvandur1/zhuicon1/projects/rewrite`
(read-only; mounted on the authoring machine as
`/home/jhu/zhuicon1/scratch_bvandur1/zhuicon1/projects/rewrite`).

This file was written **before** any code in this repository, and is the record of what was
ported, what was adapted, and what was deliberately left behind. Every value quoted here was
read out of the source or out of its runtime logs — nothing is inferred.

---

## 1. Verdict table

Legend: **copy verbatim** = logic reproduced byte-for-byte · **adapt** = same behaviour, changed
to remove a cluster assumption · **drop** = out of scope for a rewrite-only handoff.

### 1.1 `07_rewrite/` — first rewrite pass ("wiki")

| source file | verdict | reason |
|---|---|---|
| `rewrite_worker.py` | adapt | Core worker. Logic reproduced exactly in `src/rewrite/run_rewrite.py`; only the two 07-vs-09 differences become config, and output becomes JSONL. |
| `sbatch_template.sh` | adapt | Sole authority for the runtime arg values (`--gpu-mem-util 0.85`, `--input-drop 30720`, `--max-model-len 32768`, `--max-tokens 4096`, `--num-workers 8`). Directives → `scripts/optional/slurm_job.sbatch`. |
| `launch_dataset.sh` | adapt | Mode routing + prompt-file selection → `configs/data.yaml`; submission → `scripts/03_run_job.sh`. |
| `launch_all.sh` | adapt | Prints the sequential dataset order → `scripts/run_all.sh` job loop. |
| `check_progress.py` | adapt | Per-worker progress aggregation → `run_rewrite.py --status` job table. |
| `generate_summary.py` | adapt | Final status/token statistics → `scripts/04_postprocess.py` summary output. |
| `README.md` | drop | Superseded by `docs/GUIDE_FOR_TIANJIAN.md`. Its "locked config" section is transcribed into §3 below, including its one documented error. |
| `progress/*.json` | drop | Runtime artifacts of the JHU run. Their aggregate figures are quoted in §9 as expected-range references. |
| `logs/*.out` | drop | Runtime artifacts — but they are the **only** authority for the effective vLLM defaults (§3.2) and the distill overhead (§5.2). |

### 1.2 `09_Distill/` — second rewrite pass ("distill")

| source file | verdict | reason |
|---|---|---|
| `rewrite_worker.py` | adapt | Self-contained fork of 07. `diff` shows it is identical except the derived drop threshold and `--output-subdir`; both become parameters. |
| `sbatch_template.sh` | adapt | Same as 07 plus the GLOO/NCCL `172.20.x` interface pin, which is a JHU-fabric workaround and is dropped. |
| `launch_dataset.sh` | adapt | Confirms **all five arms including `wrap` run `mode=grounded`** in this pass — this is what makes p2 uniform across arms. |
| `launch_all.sh`, `check_progress.py` | adapt | As 07. |
| `README.md` | drop | Its §"Two intentional divergences" is transcribed into §4.2 below — it is the only written record of the 28672 derivation. |

### 1.3 `10_postprocess/` — trim + shuffle

| source file | verdict | reason |
|---|---|---|
| `pp_io.py` | **copy verbatim** | `strip_instruction_leak`, `strip_distill_preamble`, `choose_buckets`, `bucketed_shuffle`, `atomic_write_table` → `src/rewrite/postprocess.py` and `src/rewrite/shuffle.py`. |
| `01_strip_prefix.py` | **copy verbatim** (rules) | The `wiki` and `distill` trim rules and the `status==2`-only / changed-rows-only-recount gating. Paths and the 200-shard assumption are adapted. |
| `01_strip_prefix_wrap.py` | **copy verbatim** (rules) | `strip_preamble` with `MAX_PREAMBLE_CHARS=300` and the 26/19/12-entry `OPENERS`/`SIGNAL_WORDS`/`STRICT_META` tuples. |
| `01_strip_prefix_diversity.py` | drop (redundant) | Its wiki rule is functionally identical to `01_strip_prefix.py` (prefix passed as a parameter instead of a module constant); its extra `scan_distill_prefix`/`_lcp` code is report-only. |
| `01_strip_prefix_rewrite.py` | drop (redundant) | Likewise identical; differs only in a larger diagnostic sample (`want=200`). |
| `02_assemble_5B*.py`, `02_build_pool_rewrite.py` | drop | Token-budget **selection**, not rewrite postprocessing. Depends on upstream scoring columns this package does not carry. |
| `03_fasttext_score_rewrite.py`, `04_filter_top5B_rewrite.py` | drop | Downstream data selection; needs a 2.4 GB FastText model and a 100M-doc reference distribution. |
| `03_mix_shared_top*.py`, `05_mix_shared_top_rewrite.py` | drop | Final dataset **mixing** with an un-rewritten `shared-top-5B` block; out of scope. |
| `04_assemble_base.py` | drop | Builds the control (`10B-base`/`15B-base`) by copying raw text. Here `quality-base` is download+verify only. |
| `06_shuffle_10B_base.py` | drop (caller only) | A caller of `bucketed_shuffle`, which *is* ported. Its own retrofit logic is baseline-specific. |
| `run_*.sh` (14 files) | drop | SLURM runners for the `cpu` partition, account `bvandur1`, `module load helpers/0.1.1 gcc/9.3.0 python/3.11.9`, venv `envs/data`. Every assumption is JHU-specific. |
| `_step*_summary.json` | drop (reference) | Runtime artifacts. Quoted in §9 as the expected strip-rate ranges. |
| `README.md`, `README_rewrite.md`, `DATASETS_SUMMARY.md`, `PREFLIGHT2_FINDINGS.md`, `postprocess_report.md` | drop (reference) | Documentation. Several statements in them are **stale** — flagged in §10. |

### 1.4 Prompt-bearing files

| source file | verdict | reason |
|---|---|---|
| `00_TMP/wikipedia_style_rephrasing_grounded.md` | **copy verbatim** | The production p1 prompt. Present on disk; copied byte-for-byte (597 bytes). |
| `data_rewrite/prompts/distill_prompt.txt` | **copy verbatim** (recovered) | **File is gone.** Text recovered from `00_Prompts/generations/nemotron_distill_7B.jsonl` and verified — see §5.2. |
| `data_rewrite/prompts/wrap_prompts.json` | **copy verbatim** (recovered) | **File is gone.** Four strings recovered from `06_vllm/rewritten_examples/WRAP/wrap.md` — see §5.3. |
| `00_Prompts/generations/*.jsonl` | drop (recovery source) | 28 files, ~57 MB. Each row's `templated_input` minus `doc_text` is the literal prompt. This is how the distill prompt was recovered. |
| `06_vllm/rewritten_examples/WRAP/wrap.md` | drop (recovery source) | Contains 20 full chat-templated inputs; four distinct wrap prompts recovered from them. |
| `00_TMP/prompts_registry.py` | drop (reference) | Names the 14 "finephrase" prompts and their paths under `00_TMP/finephrase/prompts/` — **a directory that no longer exists**. Establishes that the distill prompt is `nemotron/distill.md`. |
| `00_TMP/rewriting_monitor.md` (55 MB), `rewriting_monitor_distill.md` (44 MB) | drop (test fixture) | Monitoring artifacts holding **36,190** real model outputs (18,190 wiki + 18,000 distill). Used as the differential test corpus for trim parity; not shipped. |
| `06_vllm/**` (all other) | drop | vLLM bring-up, throughput benchmarks, and prompt-engineering ablations. Its `LLM(...)` calls confirm the engine args (§3.1). |

### 1.5 Whole directories dropped

| directory | reason |
|---|---|
| `01_explore/` | Upstream scorer pipeline (FastText / FineWeb-Edu / ModernBERT) over the DCLM-RefinedWeb sample. Produces the columns this package does not carry. |
| `02_WebOrganizer/` | Topic classifier. Carries a conflicting `transformers==4.46.3` pin. |
| `03_TokenCounts/` | Llama-2 token counting over the source corpus. |
| `04_select/`, `05_select_s5_variants/`, `06_lambda_grid/` | Arm **definition** and document selection. Determines what is *in* each arm; here the arms arrive pre-built from the Hub. |
| `08_evaluation/`, `11_analysis/` | Downstream evaluation and analysis. |
| `12_Signals/` | Empty. |
| `13_600M/` | A different (600M-doc) labelling pipeline. Its `slurm/_common.sh` is the only OLD→NEW cluster mapping in the repo; noted in §8 but not needed here. |
| `.claude/` (root + 8 subdirs) | Machine-local permission caches. No project conventions; there is no `CLAUDE.md` anywhere in the source. |

---

## 2. Entry points and how a run was launched

Two independent passes, each an 8-way SLURM array, submitted **one dataset at a time**.

```
07_rewrite/launch_all.sh          # prints 5 commands, submits NOTHING
  -> 07_rewrite/launch_dataset.sh <dataset>
       -> sbatch --array=0-7 07_rewrite/sbatch_template.sh
            -> srun $PY rewrite_worker.py --worker-id $SLURM_ARRAY_TASK_ID --num-workers 8 ...

09_Distill/launch_all.sh          # same shape, second prompt
  -> 09_Distill/launch_dataset.sh <dataset>
       -> sbatch --array=0-7 09_Distill/sbatch_template.sh
            -> srun $PY rewrite_worker.py ... --output-subdir distill
```

Order (`07_rewrite/launch_all.sh:12-16`, smallest first):

```
1) signal-disagreement-lambda05   # ~5.6M docs (smallest)
2) quality-first
3) diversity-first
4) wrap                           # 4-style WRAP
5) rewrite                        # largest, run last
```

Arguments flow launcher → sbatch via `--export=ALL,DATASET_NAME=...,DATASET_PATH=...,MODE=...,PROMPT_FILE=...,WRAP_PROMPTS=...`; the sbatch template then supplies the fixed numeric arguments listed in §3.3.

Mode routing (`07_rewrite/launch_dataset.sh:21-29`): every dataset is `MODE=grounded` **except** `wrap`, which is `MODE=wrap`. In `09_Distill/launch_dataset.sh:22-29` **every** dataset including `wrap` is `MODE=grounded` — this is why the distill prompt is uniform across all five arms.

Data parallelism was **process-level**: 8 SLURM array tasks × `tensor_parallel_size=1`, each owning
`shard_index % 8 == worker_id`. vLLM's own `data_parallel_size` was never used.

---

## 3. Exact vLLM engine args

### 3.1 What was actually passed

`07_rewrite/rewrite_worker.py:240-241` (and identically at `09_Distill/rewrite_worker.py`):

```python
llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
          gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len)
```

Five keyword arguments. Nothing else. Confirmed identical in shape across all nine `LLM(...)`
call sites in `06_vllm/` (`bench_worker.py:76-82`, `controlled_rewrite.py:77-78`,
`wrap_sample20.py:112-113`, `prompt_compare_gen.py:40-41`, `wrap_styles_sample.py:83-84`,
`sample_rewrite_examples.py:125-126`, `llama_think_compare.py:97-98`).

**Never passed anywhere in the source** (verified by exhaustive grep of both trees):
`data_parallel_size`, `pipeline_parallel_size`, `max_num_seqs`, `max_num_batched_tokens`,
`enable_prefix_caching`, `swap_space`, `enforce_eager`, `distributed_executor_backend`,
`seed`, `trust_remote_code`, `download_dir`, `quantization`, `kv_cache_dtype`, `load_format`,
`speculative_config`.

### 3.2 The effective vLLM 0.22.0 defaults these ran under

From `09_Distill/logs/ds_diversity-first_1591082_0.out`, the engine's own config dump:

```
non-default args: {'dtype': 'bfloat16', 'max_model_len': 32768, 'gpu_memory_utilization': 0.85,
                   'disable_log_stats': True, 'model': '.../Qwen2.5-7B-Instruct'}
Chunked prefill is enabled with max_num_batched_tokens=16384.
Initializing a V1 LLM engine (v0.22.0) with config: ... trust_remote_code=False,
  dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=auto,
  tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
  quantization=None, enforce_eager=False, kv_cache_dtype=auto, seed=0,
  enable_prefix_caching=True, enable_chunked_prefill=True,
  cudagraph_mode=FULL_AND_PIECEWISE
```

These are **recorded, never passed** — `configs/vllm.yaml` lists them under
`inherited_defaults_do_not_pass` so a reader can see them without a reader being able to set them.

### 3.3 `gpu_memory_utilization` — a real conflict in the source

| location | value |
|---|---|
| `07_rewrite/rewrite_worker.py:161` argparse default | `0.90` |
| `07_rewrite/README.md:21` | `0.90` |
| `09_Distill/README.md:26` | `0.90` |
| **`07_rewrite/sbatch_template.sh:62`** (what ran) | **`0.85`** |
| **`09_Distill/sbatch_template.sh`** (what ran) | **`0.85`** |
| `07_rewrite` runtime log | `'gpu_memory_utilization': 0.85` |

The argparse default was never used, because the sbatch template always passed the flag. **This
package uses 0.85** — the value that produced the data. (One `09_Distill` log shows `0.9`, from an
earlier submission; the 0.85 line is the one in the production template.)

Full fixed argument set from `sbatch_template.sh:50-65`: `--num-workers 8`,
`--max-model-len 32768`, `--max-tokens 4096`, `--input-drop 30720` (07 only; **absent** in 09),
`--gpu-mem-util 0.85`, `--monitor-every 10000`.

---

## 4. Exact sampling params

`07_rewrite/rewrite_worker.py:294-296`:

```python
# per-doc cap so prompt+output never exceeds max_model_len (no vLLM overflow)
max_new = min(args.max_tokens, args.max_model_len - n_in)
keep_sp.append(SamplingParams(temperature=0, top_p=1.0, max_tokens=max(1, max_new)))
```

A **list** of `SamplingParams`, one per kept document, is passed alongside the prompt list.

- `temperature=0`, `top_p=1.0`, `max_tokens` per-doc.
- **Never set anywhere**: `top_k`, `min_tokens`, `stop`, `stop_token_ids`, `seed`,
  `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `n`, `skip_special_tokens`,
  `include_stop_str_in_output`, `logprobs`, `best_of`.
- `07_rewrite/README.md:23` confirms: *"Greedy: `temperature=0`, no repetition penalty, no quant,
  no spec decode."*

Note: several `06_vllm/` **ablation** scripts do set `repetition_penalty` (1.0/1.1) and one sets
`temperature=0.7, top_p=0.9, seed=0`. Those are experiments, not production, and are **not** ported.

### 4.1 Seed handling

There is no sampling seed and no engine seed in production (`seed=0` is vLLM's default, visible in
the log). The only seeds in the source are `np.random.default_rng([42, shard_index])` for wrap-style
assignment (**restored in round 4** as `src/rewrite/wrap_styles.py`; it had been dropped in
rounds 1–3 — see `docs/DESIGN_DELTA.md` §2), `np.random.default_rng([7, worker_id])` for monitor sampling
(dropped), and `seed=42` in the bucketed shuffle (ported).

### 4.2 Drop thresholds — the per-prompt asymmetry

Documents are **dropped, never truncated**. The decision is made on the real chat-templated token
count `n_in`, not by additivity.

| pass / prompt | threshold | origin |
|---|---|---|
| 07 wiki + 07 wrap ×4 | **30720** | fixed `--input-drop 30720` in `07_rewrite/sbatch_template.sh:63` |
| 09 distill | **28672** | `--input-drop` unset ⇒ derived `max_model_len - max_tokens` |

`09_Distill/rewrite_worker.py:204-207`:

```python
# status=0 iff templated n_in exceeds this. Derived so every kept doc gets the full output
# budget: keep iff n_in <= max_model_len - max_tokens  =>  max_new = max_tokens for all kept.
drop_threshold = args.input_drop if args.input_drop is not None \
    else (args.max_model_len - args.max_tokens)
```

`09_Distill/README.md:36-40` explains why: *"07's fixed 30720 was tuned to its wikipedia prompt.
Here `--input-drop` is unset and the status=0 threshold is derived as 28672, so every kept doc gets
the full 4096-token output budget."*

Consequence preserved in this package: under the wiki prompt a document at exactly `n_in=30720`
receives `max_tokens=2048`, not 4096 — silently reduced output room that surfaces only as
`status=1`.

### 4.3 Status codes

`07_rewrite/rewrite_worker.py:16-19` and `:291,312`:

```
0 = templated input > drop threshold (NOT rewritten)
1 = finish_reason == 'length' (truncated)
2 = finish_reason == 'stop'   (completed)
```
```python
status[j] = 1 if g.finish_reason == "length" else 2
```

Status-0 rows are **emitted**, with `rewritten=""`, `rewritten_tokens=0`, `finish_reason=""`
(`:299-302`). This is why output row count always equals input row count.

---

## 5. Chat template and prompts

### 5.1 Chat template

`07_rewrite/rewrite_worker.py:285-287`:

```python
final = qtok.apply_chat_template([{"role": "user", "content": content}],
                                 tokenize=False, add_generation_prompt=True)
n_in = len(qtok(final, add_special_tokens=False).input_ids)
```

`tokenize=False` → the rendered **string** is what `llm.generate()` receives. `llm.chat()` is never
used. **No system message is authored anywhere** — a single user message is passed, and Qwen's own
template injects `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`
(visible in every saved templated input, e.g. `06_vllm/rewritten_examples/WRAP/wrap.md`).

Prompt assembly, `07_rewrite/rewrite_worker.py:45-51`:

```python
def build_content(mode, doc_text, prompt_template, wrap_prompts, style):
    doc_text = doc_text or ""
    if mode == "grounded":
        return prompt_template.replace("[TEXT]", doc_text)
    # wrap: instruction string already ends with "Passage:\n"
    return wrap_prompts[style] + doc_text
```

There is **no truncation of the document** at any point.

### 5.2 The prompt files — where they lived, and how the missing ones were recovered

| prompt | source path | status |
|---|---|---|
| wiki (p1) | `00_TMP/wikipedia_style_rephrasing_grounded.md` | present, copied verbatim |
| distill (p2) | `/scratch/bvandur1/zhuicon1/data_rewrite/prompts/distill_prompt.txt` | **gone** — recovered + verified |
| wrap ×4 | `/scratch/bvandur1/zhuicon1/data_rewrite/prompts/wrap_prompts.json` | **gone** — recovered |

The `data_rewrite/` tree does not exist on any reachable filesystem.

**Distill recovery and proof.** `00_TMP/prompts_registry.py:21` names the prompt
`nemotron/distill.md`. `00_Prompts/generations/nemotron_distill_7B.jsonl` stores each document's
full `templated_input` and its `doc_text`; subtracting one from the other yields the literal
prompt, with `[TEXT]` in the **middle** — matching `09_Distill/README.md:16-18` ("the `[TEXT]`
placeholder is in the middle").

The proof that this is the production prompt: `09_Distill/rewrite_worker.py:231-237` prints the
empty-document chat-templated token count, and the production log
`09_Distill/logs/ds_diversity-first_1591082_0.out:2` reads:

```
[w0] OVERHEAD(empty-doc templated tokens)=185 raw_text_budget=28487 drop_threshold=28672 (=derived max_model_len-max_tokens)
```

Re-tokenising the recovered text with the source's own Qwen2.5-7B-Instruct tokenizer gives
**exactly 185**. The wiki prompt gives **150**. Both are frozen as parity assertions in
`configs/data.yaml` and checked by `scripts/preflight.py` before any GPU time is spent.

### 5.3 Wrap prompts

`07_rewrite/rewrite_worker.py:39`: `WRAP_STYLES = ["easy", "hard", "wiki", "qa"]`.
The four strings were recovered verbatim from the 20 saved chat-templated inputs in
`06_vllm/rewritten_examples/WRAP/wrap.md` (all four styles appear; 5 documents each). Each ends
with `"\n\nPassage:\n"` and is **concatenated** with the document — there is no `[TEXT]`
placeholder in wrap mode.

Measured empty-document overheads (this package's frozen parity values):
`easy=72`, `hard=66`, `wiki=73`, `qa=83`.

### 5.4 Prompt → arm mapping in the source

All four grounded arms used the **identical** prompt pair. Arms differ by *which documents they
contain*, not by prompt.

| arm (source name) | pass 07 (p1) | pass 09 (p2) |
|---|---|---|
| `quality-first` | wiki-grounded | distill |
| `diversity-first` | wiki-grounded | distill |
| `signal-disagreement-lambda05` | wiki-grounded | distill |
| `rewrite` (ReWire) | wiki-grounded | distill |
| `wrap` | easy/hard/wiki/qa (one per doc) | distill |

`06_vllm/rewritten_examples/baseline/*.md` line 1 gives the display-name mapping:
`QualityFirst (quality-first)`, `DiversityFirst (diversity-first)`, `ReWire (rewrite)`,
`SignalDisagreement (signal-disagreement-lambda05)`, `WRAP (wrap)`.

---

## 6. Per-arm postprocessing and trim logic

Full rule text with source lines is in `docs/POSTPROCESSING.md`. Summary of the dispatch:

| rule | applied to | source |
|---|---|---|
| exact `WIKI_PREFIX` slice **then** `strip_instruction_leak` | wiki-prompt output | `01_strip_prefix.py:106-118` |
| `strip_distill_preamble` | distill-prompt output | `pp_io.py:93-128`, dispatched at `01_strip_prefix.py:119-123` |
| `strip_preamble` **then** `strip_instruction_leak` | wrap-prompt output | `01_strip_prefix_wrap.py:104-131`, dispatched at `:183-194` |

Three invariants, all preserved:

1. **Only `status == 2` rows are examined** (`s2 = np.flatnonzero(status == 2)`,
   `01_strip_prefix.py:102`). Status-0 rows are empty; status-1 rows are truncated and the source
   deliberately left them alone.
2. **Llama-2 token counts are recounted only for rows the trim actually changed**
   (`01_strip_prefix.py:123-125`).
3. **Row count must not change** — the source re-reads the file's footer and raises
   `RuntimeError(f'{subdir} shard {k}: rowcount changed on rewrite')` (`:132-133`).

Trimming is **in place**, via `atomic_write_table` (`.tmp` in the same directory + `os.replace`,
`pp_io.py:34-47`). Parallelism is a process pool with `pa.set_cpu_count(1)` per worker.

The four `01_strip_prefix*.py` variants collapse to three rules with no loss: the `diversity` and
`rewrite` variants apply a functionally identical wiki rule (the prefix is a parameter rather than
a module constant) and differ only in report-only diagnostics.

---

## 7. The bucketed two-pass shuffle

`pp_io.py:184-281`, ported byte-for-byte.

```python
def choose_buckets(specs, mem_bytes=None, inflate=4.0, log=print):
    disk = sum(os.path.getsize(p) for p, _ in specs)
    est  = disk * inflate                          # zstd -> Arrow in-memory
    mem  = min(mem_bytes or MEM_CAP_BYTES, MEM_CAP_BYTES)   # MEM_CAP_BYTES = 256 * GiB
    B    = max(16, math.ceil(2.0 * est / (0.55 * mem)))
```

- `specs = sorted(specs, key=lambda s: str(s[0]))` — determinism depends on this ordering.
- **Pass 1**: one `np.random.default_rng(seed)` advanced across all inputs;
  `bk = rng.integers(0, B, size=n)`; `t.filter(pa.array(bk == b))` written to a per-bucket
  `pq.ParquetWriter(..., compression='zstd')`; `del t` after each shard; writers closed in a
  `finally`.
- **Pass 2**: per bucket, `perm = np.random.default_rng([seed, b]).permutation(n)`, `t.take(perm)`,
  prepend the `leftover` carry, emit exact `rows_per_shard=500_000` slices via `atomic_write_table`,
  carry the remainder, `os.unlink(bucket_file)` immediately.
- Final `if written != total_rows: raise RuntimeError(...)`.
- `seed=42` in every caller.

Memory safety comes from four places: the `B` formula bounding the in-RAM working set to
~`2·corpus/B`; the `MEM_CAP_BYTES` clamp; `del t` after each scattered shard; and bucket files
deleted as they are consumed. There are no flush thresholds — one writer per bucket is held open
for all of pass 1.

In the source, shuffling was scoped **per arm**, over that arm's union of `shared-top-5B` +
`rewritten/` + `distill/`; never across arms. In this package it is scoped **per (arm, prompt)**,
which is narrower — see `docs/HANDOFF_REVIEW.md`.

`run_shuffle_10B_base.sh:17-18` records why the environment was pinned: *"Same venv as run_all.sh
(envs/data, numpy 1.26.4) so the seed-42 PCG64 stream matches the stream the other five settings
were shuffled with."*

---

## 8. Sharding, checkpointing, resume

- **Input**: pre-existing `part_%05d.parquet` shards; 200 per arm.
- **Ownership**: `si % num_workers == worker_id` (`07_rewrite/rewrite_worker.py:197-203`), with
  `--num-workers 8` and `--worker-id $SLURM_ARRAY_TASK_ID`. Observed: `owns 25/200 shards`.
- **Completion marker**: **the output parquet's own existence.** There are no `.done` files in the
  source (`:264-267`).
- **Atomic write**: `.tmp` + `os.replace` (`:324-326`). Progress JSON likewise (`:133-138`).
- **Granularity**: one shard. An interrupted shard is redone in full.
- **Resume**: re-run the identical `launch_dataset.sh <name>`.

---

## 9. Reference figures from the JHU run

Aggregated from `07_rewrite/progress/*.json` (8 workers × 25 shards each):

| dataset | docs | input tok (Qwen, templated) | output tok (Llama-2) |
|---|---:|---:|---:|
| `signal-disagreement-lambda05` | 5,602,476 | 9.44B | 3.65B |
| `quality-first` | 6,136,187 | 9.52B | 3.40B |
| `diversity-first` | 5,876,747 | 9.47B | 3.81B |
| `wrap` | 10,604,458 | 9.26B | 4.31B |
| `rewrite` | 21,214,299 | 20.15B | 9.26B |

Quality flags (`00_TMP/0612.md`): `status=2` is 99.6–99.7 % everywhere; `status=1` ≤ 0.33 %;
`status=0` ≤ 0.12 %. Output/input ratio 0.36–0.47 (two different tokenizers — see §11).

Trim strip rates from `_step1_*_summary.json`, useful as expected ranges when validating a port:
wiki 0.0007 %–0.17 %; distill 0.0066 %–0.56 %; wrap per style 1–6,755 rows out of millions.

---

## 10. Stale statements in the source documentation

Recorded so a future reader does not trust them:

1. `07_rewrite/README.md:21` and both `09_Distill/README.md:26` state
   `gpu_memory_utilization=0.90`; the sbatch templates passed **0.85**.
2. `10_postprocess/README.md:30-31` states *"Distill output is scanned and reported only — distill
   text is never modified"*. `01_strip_prefix.py:206-215` **does** strip distill.
3. `01_strip_prefix_diversity.py`'s `scan_distill_prefix` docstring and
   `PREFLIGHT2_FINDINGS.md` §FIX 1 both claim `main` no longer strips distill; `:231` does.
4. `README_rewrite.md:52` says `rewrite/distill/` is empty; `_step1_rewrite_summary.json` records
   200 shards.
5. `DATASETS_SUMMARY.md` marks the `rewrite` (ReWire) arm "not run"; four `_step*_rewrite_*.json`
   artifacts show it was.
6. `03_mix_shared_top.py:262` **overwrites** `postprocess_report.md` with `path.write_text` while
   the other three mix scripts append marker-delimited blocks — re-running one arm silently erased
   the others' sections. (Out of scope here, but a real data-loss bug worth knowing about.)

---

## 11. Every hard-coded path, cluster assumption, and env var

None of these survive into this package.

**Roots.** `/scratch/bvandur1/zhuicon1/` (equivalently `/weka/scratch/bvandur1/zhuicon1/`), with
`projects/rewrite/{07_rewrite,09_Distill,10_postprocess,00_TMP,00_Prompts,06_vllm}`,
`data_rewrite/experiments/train/{5B,10B}/<dataset>/`, `data_rewrite/prompts/`,
`data_rewrite/pretrain/`, `data_rewrite/6_merged_clean/`, `.cache/{pip,huggingface,vllm,torchinductor,tmp}`.

**Models / tokenizers.** `models/Qwen2.5-7B-Instruct`, `models/Qwen2.5-1.5B-Instruct`,
`models/Llama-3.1-8B-Instruct`, `models/5m/external/fasttext_oh_eli5.bin`,
`tokenizers/llama2-unsloth-tokenizer`.

The Llama-2 tokenizer is portable: its own `tokenizer_report.json` records
`"source_repo": "unsloth/llama-2-7b"`, `vocab_size 32000`, `LlamaTokenizer`, and confirms
`add_special_tokens=False` usage. This package downloads it from that repo ID.

**SLURM (rewrite passes).** `07_rewrite/sbatch_template.sh:2-13`:

```
#SBATCH --partition=nvl,h100      #SBATCH --qos=h200_4        #SBATCH --nodes=1
#SBATCH --exclude=n02,n03         #SBATCH --gres=gpu:h100:1   #SBATCH --cpus-per-task=8
#SBATCH --mem=64G                 #SBATCH --time=3-00:00:00   #SBATCH --array=0-7
```
`09_Distill` adds `--exclude=c001,n02,n03,n06`. **No `--account`** in either rewrite template.
The `10_postprocess/run_*.sh` runners use `--partition=cpu`, `--account=bvandur1`,
`--cpus-per-task=96`, `--mem=240G`.

**`module load`.** None in the rewrite passes. `10_postprocess` uses
`module load helpers/0.1.1 gcc/9.3.0 python/3.11.9` — a JHU module tree.

**Python environments.** Rewrite: the interpreter path directly,
`PY=/scratch/bvandur1/zhuicon1/basic/miniconda3/envs/vllm/bin/python` (Python 3.12); `06_vllm`
scripts `conda activate /scratch/bvandur1/zhuicon1/basic/miniconda3/envs/vllm`. Postprocess: a
venv, `source /scratch/bvandur1/zhuicon1/envs/data/bin/activate`.

**Environment variables** (`07_rewrite/sbatch_template.sh:30-39`):

```bash
export PIP_CACHE_DIR=/scratch/bvandur1/zhuicon1/.cache/pip
export HF_HOME=/scratch/bvandur1/zhuicon1/.cache/huggingface
export XDG_CACHE_HOME=/scratch/bvandur1/zhuicon1/.cache
export VLLM_CACHE_ROOT=/scratch/bvandur1/zhuicon1/.cache/vllm/${DATASET_NAME}/rank_${WID}
export TORCHINDUCTOR_CACHE_DIR=/scratch/bvandur1/zhuicon1/.cache/torchinductor/${DATASET_NAME}/rank_${WID}
export TMPDIR=/scratch/bvandur1/zhuicon1/.cache/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```
In-process (`rewrite_worker.py:29-32`): `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`TOKENIZERS_PARALLELISM=false`, all via `os.environ.setdefault`.

`09_Distill/sbatch_template.sh:55-68` additionally pins the fabric, a pure JHU workaround:

```bash
IFACE_2020=$(ip -o -4 addr show | awk '$4 ~ /^172\.20\./ {print $2; exit}')
export GLOO_SOCKET_IFNAME="$IFACE_2020"; export NCCL_SOCKET_IFNAME="$IFACE_2020"
```
(Intermittent engine-init crashes came from `torch.distributed` resolving the hostname to an
unbound `172.21.x` address.) No other `NCCL_*`, no `CUDA_*` beyond `CUDA_VISIBLE_DEVICES`,
no `VLLM_*` beyond `VLLM_CACHE_ROOT`.

**Dependency pins.** The source has **no** `requirements.txt`, `environment.yml`, `pyproject.toml`,
or lockfile. The stack was recovered from the archived environment tarball
(`env_packs/vllm.tar.gz`) and from runtime logs:

```
Python 3.12 · vllm 0.22.0 · torch 2.11.0+cu130 · transformers 5.9.0 · tokenizers 0.22.2
huggingface_hub 1.17.0 · numpy 2.3.5 · triton 3.6.0 · flashinfer-python 0.6.11.post2
pyarrow 24.0.0 · safetensors 0.7.0 · xgrammar 0.2.1 · hf_xet 1.5.0 · msgspec 0.21.1
nvidia-nccl-cu13 2.28.9 · torchvision 0.26.0 · torchaudio 2.11.0
```
Driver at runtime `580.159.04`; GPU `NVIDIA H100 NVL, 95830 MiB`.
`datasets` and `hf_transfer` were **not** present — the source read local parquet and never touched
the Hub at runtime. Both are added here for the download path only.

**HF repo IDs.** Exactly one appears in the whole source tree
(`06_vllm/wrap_styles_sample.py:27`, `"Qwen/Qwen2.5-1.5B-Instruct"`, used only in a markdown
header). There is no `push_to_hub`, no `HfApi`, and no dataset repo ID anywhere — every model and
dataset was a local absolute path under `HF_HUB_OFFLINE=1`. This is why the input and output repo
names started as `<<<WYTRO>>>` placeholders in `configs/data.yaml`.

> **Round 4 update.** The input side is now resolved: the corpus was uploaded to the single gated
> repo `wytro/Know-Your-Sources-7B` by `13_600M/02_select/upload_blocks.py:67`, and `data.yaml`
> pins it by commit sha. Only `upload.repo_template` (the *output* org) remains a placeholder,
> because nothing has been pushed there yet. See `docs/DESIGN_DELTA.md` §4.

---

## 12. Output record schema in the source

`07_rewrite/rewrite_worker.py:316-323` — parquet, preserving **all** original columns and appending:

| column | type | meaning |
|---|---|---|
| `rewritten` | `large_string` | raw model output, **not** stripped (trimming is a later step) |
| `rewritten_tokens` | `int32` | Llama-2 token count of the output |
| `status` | `int8` | 0 / 1 / 2 |
| `finish_reason` | `large_string` | raw from vLLM |
| `input_tokens_qwen` | `int32` | templated input token count |
| `wrap_style` | `large_string` | wrap pass only |

The mapping to this package's 10-key JSONL schema is given in `docs/HANDOFF_REVIEW.md`.
