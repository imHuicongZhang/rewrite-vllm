# HANDOFF_REVIEW

Written for Wytro, to review before this repo goes to Tianjian.

---

## 1. Headline

The 12 jobs were **not** invented to fit the spec — they fall straight out of the source's
own design. The source ran **two** rewrite passes over every arm (`07_rewrite` = "wiki",
`09_Distill` = "distill") plus four styles for `wrap`: `2+2+2+4+2 = 12`. Confirmed with
you before building.

All six prompt texts are recovered and shipped byte-exact. **No prompt text needed a
placeholder**, including the two whose files no longer exist on any reachable filesystem.

Both correctness-critical ports are verified against the source code itself, not by eye:

| port | verification | result |
|---|---|---|
| trim rules | differential test vs. the source functions on **36,190 real model outputs** + 21 adversarial cases | **72,443 comparisons, 0 mismatches** |
| bucketed shuffle | function-body line comparison **and** an execution test | identical bucket count, shard names, and **row ordering** across 63,000 rows |
| prompts | empty-doc chat-templated token count vs. the source's own logged value | wiki **150**, distill **185** (matches the source log line verbatim), wrap **72/66/73/83** |
| whole pipeline | end-to-end integration run with a stubbed engine | **90 checks, 0 failures** |

---

## 2. Copied verbatim vs. rewritten

### Verbatim (logic reproduced exactly)

| what | from | note |
|---|---|---|
| `strip_instruction_leak`, `strip_distill_preamble` | `pp_io.py:72-128` | including all constants |
| `WIKI_PREFIX` slice + leak strip | `01_strip_prefix.py:106-118` | including the `s is not None` guard |
| `strip_preamble` + 26/19/12-entry tuples | `01_strip_prefix_wrap.py:53-131` | including the `or ''` normalisation the wiki path lacks |
| `choose_buckets`, `bucketed_shuffle` | `pp_io.py:184-281` | line-for-line |
| `atomic_write_table` | `pp_io.py:34-47` | |
| prompt assembly (`build_content`) | `rewrite_worker.py:45-51` | |
| chat templating and `n_in` counting | `rewrite_worker.py:285-287` | |
| per-doc `SamplingParams` | `rewrite_worker.py:294-296` | |
| status-code assignment | `rewrite_worker.py:291,312` | |
| shard ownership (`si % N == worker_id`) | `rewrite_worker.py:197-203` | |
| the six prompt texts | see §4 | byte-exact |

### Rewritten, and why

| what | why |
|---|---|
| **Two workers → one** (`run_rewrite.py`) | `diff`ing `07_rewrite/rewrite_worker.py` against `09_Distill/rewrite_worker.py` shows they are identical apart from the drop threshold and the output subdirectory. Both became config fields. There is no `if pass == "distill"` anywhere — the asymmetry is data, which is the only way it survives a future edit. |
| **Config layer** (`config.py`) | The source hard-coded every path. Portability requires a config; validation requires a loader. |
| **HF download + self-sharding** (`data.py`) | The source consumed a pre-existing 200-shard parquet layout on JHU scratch. Inputs now come from the Hub. |
| **Launchers → `03_run_job.sh` / `run_all.sh`** | SLURM array → N background processes with `CUDA_VISIBLE_DEVICES`. Behaviourally identical: N independent single-GPU engines. |
| **Four trim scripts → three rule functions** | `01_strip_prefix_diversity.py` and `01_strip_prefix_rewrite.py` are byte-identical to `01_strip_prefix.py` in their rules; they differ only in report-only diagnostics. Verified before dropping. |
| **`preflight.py`, `check_placeholders.py`** | New; no source equivalent. |

### Dropped from scope

The source's downstream **selection and mixing** (`02_assemble_*`, `03_mix_shared_top*`,
`03_fasttext_score_rewrite`, `04_filter_top5B_rewrite`, `04_assemble_base`) is not ported.
Only trim + shuffle are. Those steps need upstream scoring columns
(`fasttext-ranking-v2`, `topic`, `tokens-llama2`) that the 10-key JSONL schema does not
carry, and a 2.4 GB FastText model. See open question 3.

---

## 3. Every behavioural difference from the source

Ordered by how much they matter.

0. **Target hardware is a mixed H200 / B200 / B300 fleet — read this first.**
   You told me this after the first build, and it is the single biggest threat to the
   experiment, so it goes above the rest.

   The original data was generated on **H100 (sm_90)** using **FlashAttention v3**, which
   is Hopper-only — I confirmed both from the source's own runtime logs (`Using
   FlashAttention version 3`, `NVIDIA H100 NVL`). On Blackwell (B200 = sm_100, B300 =
   sm_103) vLLM will select a **different attention backend**. This is not merely a
   different reduction order: it is a different kernel.

   Greedy decoding at `temperature=0` is **not** bitwise identical across architectures.
   An argmax can flip on a near-tie, and because generation is autoregressive, one flipped
   token diverges the remainder of a up-to-4096-token output. It is the same class of
   hazard the version pinning exists to prevent, except silent.

   **Why this is survivable, and what makes it survivable.** Jobs run sequentially and
   each uses every GPU, so if the GPU set is held constant across all 12 jobs, every arm
   sees the *same* hardware mixture. The architecture effect is then **balanced across
   arms** rather than segregated into one, which keeps the arm-vs-arm comparison honest.
   The failure mode to avoid is changing `gpu_ids` partway through — that would confound
   "which arm" with "which GPU". The guide says this in §2.5 and again in the do-not list.

   **What I added so this is auditable rather than invisible:** every shard's `.done`
   sidecar now records `gpu_name` and `gpu_cc`. After the run you can reconstruct exactly
   which architecture produced which rows, per shard, for all 12 jobs. Without it that
   information is gone forever.

   **What preflight now catches**, before a GPU-hour is spent: it compares each device's
   compute capability against `torch.cuda.get_arch_list()` and reports native kernels /
   PTX-JIT fallback / **no kernels at all** (hard fail); prints the fleet breakdown when it
   is heterogeneous; and repeats the comparability warning. `00_setup_env.sh` runs the same
   arch check right after installing, and its VRAM check now uses the **smallest** card
   rather than GPU 0.

   **My recommendation:** if you can spare it, run all 12 jobs on one architecture — most
   likely the B200/B300 pool, since it is faster and the H200s can do something else.
   Second best is the balanced mixture above. The worst outcome is an unplanned mixture
   that nobody recorded, and that one is now impossible.

   **Not verified here:** whether `torch 2.11.0+cu130` / `vllm 0.22.0` / `flashinfer
   0.6.11.post2` actually carry working sm_100 and sm_103 kernels. CUDA 13 supports both,
   and sm_103 should at worst PTX-JIT from sm_100, but I have no Blackwell card to test on.
   The preflight arch check and `00_setup_env.sh`'s real one-prompt generation are what
   will answer it, in minutes, on his machine.

1. **`wrap-inspired` semantics — the big one.** The source assigned **one of four styles
   per document** via `np.random.default_rng([42, shard_index])`: a single pass producing
   one output per document. This package does **four complete passes**, producing four
   outputs per document. Your explicit instruction. Consequences: 4× the wrap compute and
   4× the wrap output; the `wrap_style` column is gone (its information is now
   `prompt_id`); the per-row style dispatch in the trim step is gone (each style is its
   own job, so the rule is uniform within a shard). `assign_wrap_styles` and the seeded
   style RNG are **deleted, not disabled** — leaving them would invite a future reader to
   "restore" them.

2. **Output format: JSONL with 10 fixed keys, not parquet with every input column.**
   Field mapping: source `input_tokens_qwen` → `n_prompt_tokens`; source
   `rewritten_tokens` → `n_output_tokens_llama2`; `status` and `finish_reason` unchanged.
   New: `source_text_sha1`, and `n_output_tokens` (the Qwen-side count, free from vLLM,
   which the source never stored). **Dropped: every upstream column** — `tokens-llama2`,
   `fasttext-ranking-v2`, `topic`, `orig_doc_id`. That is what puts the source's selection
   steps out of scope. `source_text_sha1` at least lets any downstream join key on content
   rather than on our `doc_id` numbering.

3. **`gpu_memory_utilization = 0.85`, not `0.90`.** The source conflicts with itself:
   `rewrite_worker.py:161` and both READMEs say `0.90`, but both sbatch templates pass
   `--gpu-mem-util 0.85` and every runtime log records 0.85. Resolved in favour of what
   actually ran, per your decision.

4. **Self-sharding changes `doc_id`s and shard boundaries.** The source's `doc_id` came
   from an upstream column; here it is the dataset's own `doc_id` if present, otherwise a
   deterministic 0-based row index over sorted input files. Shard boundaries will not match
   the source's. Irrelevant to greedy per-document generation; it does change shuffle
   output ordering. A manifest `fingerprint` refuses to re-shard under a finished run,
   because that would renumber `doc_id` and silently invalidate every `.done` marker.

4b. **Shard assignment is dynamic by default, not static modulo.** The source used
   `shard_index % num_workers == worker_id` — correct and perfectly balanced on its
   homogeneous H100 cluster, and wrong for a mixed fleet, where equal shard counts mean
   wall-clock equals the slowest card. Workers now claim the next free shard via an
   atomically created directory (`os.mkdir`, which is atomic even on NFS, unlike
   `O_CREAT|O_EXCL`). Same shards processed, same output; only *which worker* does *which
   shard* changes. `shard_assignment: static` restores exact source behaviour. Stale claims
   from a killed run are reaped once by the launcher, under the job lock, before any worker
   starts — never by a worker, which would let two workers take the same shard. Measured in
   the integration test: a worker 8× slower than its peers took 28 shards to their 61,
   where static would have forced 50/50/50.

5. **`.done` sidecars added.** The source used the output parquet's own existence as the
   completion marker. Parquet is self-describing (its footer carries `num_rows`); JSONL is
   not, so the sidecar restores that property and makes the per-job row audit cost
   `n_shards` small JSON reads instead of re-reading terabytes. Written **after** the data
   file's rename, so a crash between the two leaves a complete file with no marker (redone,
   costing at most one shard) and never a half-written file that looks finished.

6. **Shuffle output is parquet; rewrite output is JSONL.** The shuffle is ported verbatim
   and its internals are Arrow/parquet — the per-bucket `ParquetWriter` temp files *are*
   the memory-safety mechanism. Only the injected `load_fn` changed.

7. **Batched tokenizer calls** replace the source's per-document loop for both the Qwen
   `n_in` count and the Llama-2 output count. Value-identical for these fast tokenizers,
   and `preflight.py --verify-tokenizer-batching` proves it on 1,000 real documents. At
   ~500B tokens tokenized twice per pass, the per-document Python loop was plausibly the
   throughput ceiling.

8. **`datasets` and `hf_transfer` added** to a stack that had neither — the source read
   local parquet and never touched the Hub at runtime. Both are used only in the download
   path; neither is imported anywhere near generation.

9. **Shuffle determinism is machine-local, not bit-identical to the original run.** The
   bucket count `B` derives from on-disk bytes and available RAM, and the pass-1 RNG stream
   depends on the numpy version. The original pinned numpy 1.26.4 for exactly this reason;
   this stack pins 2.3.5, which the vLLM 0.22.0 wheels require. Deterministic for him,
   not reproducible against ours.

10. **Bitwise non-determinism vs. the original outputs.** Different GPU architecture,
    different batch composition and vLLM's non-deterministic reduction ordering mean greedy
    outputs will occasionally differ from what JHU produced, even at `temperature=0`. This
    is unavoidable across hardware. The package reproduces the **procedure** exactly; it
    does not promise identical strings.

11. **Monitoring not ported.** The source's `rewriting_monitor.md` flock-append sampler and
    its `flag_format_only` / `flag_repetition` / `flag_short` heuristics are gone; the
    per-shard tok/s + ETA logging and the `.done` statistics replace them. Those monitor
    files were, however, used as the test corpus for the trim differential test.

12. **NCCL/GLOO interface pinning dropped.** `09_Distill/sbatch_template.sh:55-68` forced
    `NCCL_SOCKET_IFNAME` onto the `172.20.x` fabric to dodge a JHU-specific bind failure.
    Cluster-specific; documented in the guide's troubleshooting table as something to
    re-apply via `env.extra_preamble` if he hits the same symptom.

13. **One file beyond the specified layout: none.** The layout is exactly as specified. The
    shell scripts read config in two stages (sed for two scalars, then the validated Python
    loader) rather than adding a helper module — see §6.

---

## 4. Prompts: provenance and proof

| prompt | source | status |
|---|---|---|
| p1 wiki (4 arms) | `00_TMP/wikipedia_style_rephrasing_grounded.md` | present on disk; **byte-identical copy** (597 B, `cmp` clean). Note the trailing space after `Text:` — preserved. |
| p2 distill (4 arms) | `data_rewrite/prompts/distill_prompt.txt` | **file is gone.** Recovered from `00_Prompts/generations/nemotron_distill_7B.jsonl` by subtracting `doc_text` from `templated_input`. |
| wrap p1–p4 | `data_rewrite/prompts/wrap_prompts.json` | **file is gone.** Recovered from the 20 saved chat-templated inputs in `06_vllm/rewritten_examples/WRAP/wrap.md`. |

**The distill proof is worth your attention**, since that file no longer exists anywhere.
`09_Distill/rewrite_worker.py:231-237` prints the empty-document templated token count, and
the production log `09_Distill/logs/ds_diversity-first_1591082_0.out:2` reads:

```
[w0] OVERHEAD(empty-doc templated tokens)=185 raw_text_budget=28487 drop_threshold=28672
```

Re-tokenising the recovered text with the source's own Qwen2.5-7B-Instruct tokenizer gives
**exactly 185**. It also has `[TEXT]` in the middle of the file, which is what
`09_Distill/README.md:16-18` independently says about the real one. Two independent facts
agreeing on a file neither of us can open is about as strong as recovery gets — but it is
recovery, not a copy, so it is flagged here rather than buried.

All six overheads are frozen in `configs/data.yaml` as `expected_overhead` and asserted by
`preflight.py` **and** by every worker at startup. If a prompt file, the chat template, or
the tokenizer ever changes, the job stops before spending a GPU-hour.

---

## 5. Every placeholder, and why it could not be resolved

**11 `TIANJIAN`** (all in `configs/cluster.yaml`) + **1 in `.env`** — his machine, which
neither of us can inspect: `repo_root`, `model_dir`, `data_root`, `out_root`, `tmp_root`,
`log_root`, `hf_cache`, `num_gpus`, `cpu_workers`, `shuffle_mem_bytes`, `activate_cmd`,
and `HF_TOKEN`. Each carries a one-line "how to find this on your machine" comment, and
the guide's §2 repeats them as a checklist.

**7 `WYTRO`** (yours, to fill before handoff):

| placeholder | why I could not resolve it |
|---|---|
| `arms[*].repo_id` × 6 | **The source contains no dataset repo IDs at all.** Every model and dataset was a local absolute path under `HF_HUB_OFFLINE=1`; there is no `push_to_hub`, no `HfApi`, and exactly one hub ID in the entire tree (`Qwen/Qwen2.5-1.5B-Instruct`, in a markdown header string). The corpora these arms correspond to lived at `data_rewrite/experiments/train/10B/<name>/`, a path that no longer exists. |
| `upload.repo_template` | Same: the source never uploaded anything. Takes `{arm}` and `{prompt_id}`. |
| `HF_TOKEN_WRITE` in `.env` | Yours, since the output org is yours. May be the same token as his. |

Values I deliberately did **not** turn into placeholders, having found them in the source:
the Llama-2 tokenizer repo (`unsloth/llama-2-7b`, from its own `tokenizer_report.json`),
the model repo, all six prompt texts, all six prompt overheads, both drop thresholds, and
the entire pinned dependency stack.

---

## 6. Assumptions I made

1. **Arm ↔ source-dataset mapping**: `quality-first` → `quality-first`,
   `diversity-oriented` → `diversity-first`, `disagreement-aware` →
   `signal-disagreement-lambda05`, `wrap-inspired` → `wrap`, `rewire-inspired` → `rewrite`
   (ReWire). Unambiguous from `06_vllm/rewritten_examples/baseline/*.md`, whose first line
   gives the display-name mapping. Affects only which repo ID you put where.

2. **`rewire-inspired` uses the same wiki+distill pair as the other arms** — your decision.
   Source-faithful: the `rewrite` arm ran `mode=grounded` with the identical two prompts,
   and its identity came from its corpus and its downstream FastText filter, not its
   prompt. The unused `rewire/guided_rewrite_improved` prompt was not adopted.

3. **Shard sizing**: `shard_target_rows: 10000` with a 256 MiB byte cap, whichever hits
   first. Rationale: the shard is the unit of resume, so at ~2k output tokens/doc and a
   plausible ~2.5k tok/s per GPU it is roughly 25–40 minutes of work — small enough that an
   interruption loses under an hour, large enough that inter-batch idle time is negligible.
   The byte cap exists so one run of very long documents cannot become a multi-hour shard.
   A hard check refuses to proceed below 20 shards per GPU (the source ran 25:1).

4. **Model revision left at `main`** rather than a pinned sha. The source loaded from a
   local directory and recorded no revision, so there is nothing to port, and inventing a
   sha would be fabrication. The overhead parity gate (150/185) is the real guard: it
   catches any chat-template change before generation starts. Pin it if you prefer.

5. **Disk sizing model**: 4.2 bytes per output token (~3.8 chars/token of UTF-8 English
   plus ~10% JSON string escaping) + 220 bytes/row of JSON envelope. Both are `data.yaml`
   keys so the estimate is auditable, and `preflight.py` prints the assumption alongside the
   number rather than asserting a bare figure.

6. **Trim runs in place by default** (source parity), with `postprocess.in_place: false`
   available at the cost of a second full copy.

7. **Upload granularity**: one dataset repo per (arm, prompt) = 12 repos, from the
   `shuffled` stage. See open question 2.

8. **Shell config bootstrap**: the two shell scripts read `activate_cmd` / `extra_preamble`
   with `sed`, then activate, then read everything else through the validated Python
   loader. This is not stylistic — PyYAML lives inside the environment that has not been
   activated yet, and his system `python3` may well not have it. I verified the failure is
   real on this machine and that the two-stage fix works.

---

## 7. Where the source is ambiguous

Recorded rather than silently resolved.

1. **`gpu_memory_utilization`**: the source contradicts itself (0.90 in code and docs, 0.85
   in both sbatch templates and every log). You chose 0.85. Also worth knowing: one
   `09_Distill` log shows `0.9`, from an earlier submission — so the original *distill*
   data may itself be a mix of the two.

2. **`quality-base`'s exact definition.** The source built it two ways — `04_select/
   select_10b.py` (5B block) and `select_quality_base_15B.py` (10B block) — and
   `11_analysis/README.md` describes it as `quality-first/shared-top-5B`, i.e. raw
   top-ranked text, not a rewrite. Here it is simply whatever repo ID you supply, verified
   and counted. Nothing depends on the ambiguity.

3. **Which λ `disagreement-aware` should be.** The source ran λ ∈ {0, 0.5, 1, 1.5, 2, 3};
   λ=0.5 was the production choice (`check_progress.py:13`), but λ15 and λ2 were the ones
   actually postprocessed in the most recent run (`postprocess_report.md:5`). Moot for the
   code; it decides which repo ID you fill in.

4. **Stale source documentation** — six statements in the source READMEs contradict the
   code (e.g. `10_postprocess/README.md:30-31` says distill text "is never modified"; the
   code strips it). All six are catalogued in `SOURCE_INVENTORY.md` §10 so a future reader
   does not trust them.

5. **A real bug in the source, out of scope but worth knowing**:
   `03_mix_shared_top.py:262` **overwrites** `postprocess_report.md` with `path.write_text`
   while the other three mix scripts append marker-delimited blocks — so re-running one
   arm silently erased the other arms' sections. Not ported, so not inherited.

---

## 8. Open questions for you

1. **The six input repo IDs and the output repo template.** The only thing blocking
   handoff.

2. **Upload granularity.** I default to 12 separate dataset repos from the `shuffled`
   stage. One repo with 12 configs would be friendlier for a downstream loader but is a
   different shape; say if you want it. Also: should `quality-base` travel with the
   rewrites? It is verified but never uploaded at present.

3. **Do you need the dropped scoring columns?** If the source's selection/mixing steps
   (`02_assemble_*`, `03_fasttext_score_rewrite`, …) will ever run on this output, the
   10-key schema is not enough — it drops `tokens-llama2`, `fasttext-ranking-v2` and
   `topic`. Adding a pass-through of input columns is a one-line change now and an
   expensive regeneration later.

4. **Compression.** Default is zstd, so files are `.jsonl.zst`, not `.jsonl` (~0.7 TB vs
   ~2.4 TB). Confirm downstream consumers can read that, or set
   `runtime.output_compression: none`.

5. **Shuffled output format.** Your spec fixes JSONL for the rewrite stage and is silent
   after the shuffle; I default to parquet (source parity, and better on the Hub).

6. **One architecture, or the balanced mixture?** See divergence 0. My recommendation is
   all 12 jobs on one architecture if you can spare the hardware; the balanced mixture is a
   sound second. Either way the GPU set must be fixed before job 1. Tianjian should not be
   the one deciding this.

7. **A length-distribution check before the full run?** The source dropped 0.04–0.12% of
   documents as over-length. If the Hub-hosted corpora have a different length profile, the
   status-0 rate will differ. A 10,000-document histogram of templated `n_in` per arm is
   cheap and would catch a corpus mismatch before 12 jobs run. Not built — say the word.

---

## 9. What was and was not executed

**Executed and passing on this machine:**

* trim differential test — 72,443 comparisons vs. the source functions, 0 mismatches
* shuffle parity — body comparison + execution test, identical ordering on 63,000 rows
* prompt overhead parity — all 6 prompts against the source's logged values
* full end-to-end integration with a stubbed engine — **100 checks, 0 failures**, covering
  config validation, sharding, all 12 jobs across 3 workers, row conservation, `.done`
  resume, stale-`.tmp` cleanup, the fingerprint interlock, deliberate corruption detection,
  deep verify, trim, and shuffle — plus, on a simulated heterogeneous fleet, concurrent
  shard claiming under contention, load rebalancing toward the fast workers, a live claim
  correctly blocking a second worker, stale-claim reaping, GPU provenance in every sidecar,
  and `shard_assignment: static` still working
* `py_compile` on all Python, `bash -n` on all shell
* `check_placeholders.py` on both a blank and a fully-filled repo

**Not executed — no GPU or CUDA on this login node:**

* `00_setup_env.sh` — driver/CUDA detection, the pinned install, and the one-prompt
  generation check. Syntax-checked only.
* `preflight.py` steps 3, 4, 5, 6, 10 (GPU, imports, model, prompt parity against a real
  Qwen tokenizer, smoke test). Steps 1, 2, 7, 8, 9 are exercisable without a GPU; the
  prompt-parity logic is the same code the standalone parity test ran and passed.
* Any real vLLM call, and any Hub download or upload.
* **Anything on Blackwell.** No H200, B200 or B300 was available here — this login node has
  no GPU at all. The sm_100/sm_103 kernel-availability question is answered by preflight
  and by `00_setup_env.sh`'s generation check on his machine, in minutes.

**Two real bugs the integration test caught**, both fixed and regression-tested:

1. `open_jsonl_write` chose compression from the filename, but every caller passes the
   `.tmp` name — so shards were written **uncompressed under a `.zst` name** and were
   unreadable afterwards. This would have destroyed a full run's output silently.
2. `05_upload_to_hf.py` swept `.done` sidecars into its upload file list.

---
---

# Addendum — 2026-08-23: review round 2 + Blackwell-only decision

Everything above this line is the original record and is unchanged.

Two inputs since: Wytro's decision that **all 12 jobs run on B200 + B300 only**, and an
independent review by another Claude with no access to the source pipeline. This addendum
answers the review item by item, delivers the C1 analysis, and records the Blackwell work.

One correction to the review's premise: `main` had **2** commits (`d1a8628`, `7895e5f`),
not 3. Its §C3 endorsement of dynamic shard claiming confirms it read `7895e5f`.

---

## A. Defects

I re-derived all four testable items by execution rather than accepting them. Two were
right but for the wrong reason, and in both cases the true mechanism is worse.

### A1 — `.claim` directories reach the upload payload — **FIXED, and it was worse than reported**

Verified, and then some. Beyond the payload filter, I checked what
`huggingface_hub.utils.filter_repo_objects` actually does with the real library:

```
filter_repo_objects(names, allow_patterns=["part_*"],
                    ignore_patterns=["*.tmp","*.done",".joblock"])
  -> ['part_00000.jsonl.zst', 'part_99999.claim/owner.json']
```

So the file inside a leaked claim directory **would have been published to the Hub** at
`data/part_99999.claim/owner.json`. The reason is that `fnmatch`'s `*` crosses `/`, so
`part_*` matches the nested path and no ignore pattern touched it. The review flagged the
payload count; the actual exposure was publication.

Two further defects in the same block that the review did not catch:

* `nbytes = sum(p.stat().st_size for p in files)` called `stat()` on a **directory**,
  returning the dirent size (typically 4096) — so the reported GiB total was wrong.
* `len(files)` counted the directory as one entry while HF would upload a file *inside* it,
  so both the printed file count and the commit message were wrong.

That block's own comment promises the count and byte total are honest. They were not.

Fixed three ways, so the failure needs all three to regress: `p.is_file()` in `payload()`
(excludes it at source and repairs the byte total), `"*.claim"` **and** `"*.claim/*"` in
`ignore_patterns` (the second is the one that actually matters, given fnmatch semantics),
and a regression test that asserts against the real `filter_repo_objects`.

### A2 — missing shards computed but never reported — **FIXED, and the real defect is deeper**

The review's framing suggests a job could look healthy while shards are missing. That part
is **not** right: `missing` still feeds `state`, so the job reports `PARTIAL`, and every
consumer keys off `state`. `postprocess.trim_job` refuses to trim, `--verify` exits 1, and
`run_all.sh` will not skip it. Nothing silently proceeds.

The genuine defect is worse in a different way. The final guard read:

```python
if not missing and not problems and rows_out != man.total_rows:
```

That `not missing` conjunct made **check #4 — documented in the docstring as "the required
assertion" — dead code in exactly the situation it exists for.** Missing shards are the
principal cause of a short row count, and the guard excluded that case by construction. The
"every prompt rewrites the ENTIRE dataset" invariant, which is the single most important
property in this repo, had no reachable assertion behind it at job level.

Fixed: missing shards now raise a problem naming the count and the first ten indices, and
the total check became `if not missing and rows_out != total` — reachable whenever the
shard set is complete, which is the only case where a discrepancy would otherwise be
unexplained. Docstring corrected.

Consequence for the guide: the troubleshooting row promising `--verify --deep-verify`
"prints exactly which shards disagree" is now true rather than aspirational. I tightened it
to "missing or disagree" for precision.

### A3 — stale module reference — **FIXED**

`configs/data.yaml:28` said `rewrite.verify.verify_job()`; there is no such module. Now
`rewrite.data.verify_job()`. Plain error on my part.

### A4 — `run_all.sh --from-job N` — **FIXED; the review's diagnosis was wrong and the real bug is worse**

"Never works" is **false**, and the stated mechanism is not the mechanism.

Measured, by extracting the parser and running every form:

| invocation | old FROM_JOB | new |
|---|---|---|
| `--from-job=5` | `5` | `5` |
| `--from-job 5` | **`5`** | `5` |
| `--skip-upload --from-job 5` | `--from-job` | `5` |
| `--from-job 5 --skip-upload` | `5` | `5` |
| `--status --from-job 5` | `--from-job` | `5` |

It worked **iff `--from-job` was the first argument**. The cause is not that "`shift` does
not advance the loop variable" — it is that `shift` always drops from the **front** of the
positional list, so after one shift `$1` is the original argument **#2**, whatever that
happens to be. With `k` flags before `--from-job`, you get argument `k+2`; with `k=0` that
is accidentally the right value.

The part neither of us flagged initially is the downstream behaviour. `FROM_JOB` is used in
`(( idx < FROM_JOB ))`, and bash arithmetic evaluates the bare word `--from-job` as
`-(-(from - job))` with both names unset — that is, **`0`**. So a misparse did not error;
`run_all.sh --status --from-job 5` silently ran **all 12 jobs from job 1**. Idempotent
thanks to the `.done` sidecars, so the cost is wasted wall-clock rather than bad data — but
at this scale that could be days.

Rewritten as a `while [[ $# -gt 0 ]]` loop, with `FROM_JOB` validated as a positive integer
and **unknown options rejected**. Silently ignoring unknown options is what let a
mis-consumed value pass as a flag in the first place. Regression test covers six valid
forms and four malformed ones.

### A5 — guide section order — **FIXED**

Confirmed 2.1, 2.2, **2.5**, 2.3, 2.4. Renumbered so the order is now 2.1 paths → 2.2
compute → **2.3 GPU architecture** → 2.4 env → 2.5 `.env`, with all cross-references
updated. The section stays adjacent to `compute` because it is about `gpu_ids`, which is
where a reader needs it.

---

## B. Answers

### B1 — commit the verification harnesses — **DONE, with one design tension resolved**

Committed under `tests/`, plus `tests/README.md`.

The tension: design constraint 1 forbids JHU-specific paths in shipped code, but the two
parity tests must reach the source pipeline. Resolved by making `--source-root` a
**required argument with no default**; both tests exit **77** with an explanation when it
is absent. So they are committed, runnable on the cluster where the source still exists,
and contain no path that pins this repo to one machine.

| file | needs | last run |
|---|---|---|
| `tests/test_trim_parity.py` | `--source-root` | 72,443 comparisons, 0 mismatches, 9/9 constants equal |
| `tests/test_shuffle_parity.py` | `--source-root` | identical text and identical row order over 63,000 rows |
| `tests/test_integration.py` | nothing | ~100 checks, 0 failures |
| `scripts/verify_prompt_parity.py` | a local model dir | 12/12 prompts |

All four re-run and pass against the current tree.

### B2 — standalone prompt-parity check — **DONE**

`scripts/verify_prompt_parity.py`. Needs only `transformers`, `PyYAML` and a local model
directory: no cluster config, no filled placeholders, no GPU, no vLLM, no network. Reads
expected values from `configs/data.yaml`, so it cannot drift from what the workers assert
at runtime, and it checks structure (`[TEXT]` count, the wrap `\n\nPassage:\n` suffix) as
well as the token count. Output on the real tokenizer:

```
quality-first/p1  p1_wiki     150  150  ok      wrap-inspired/p1  wrap_easy  72  72  ok
quality-first/p2  p2_distill  185  185  ok      wrap-inspired/p2  wrap_hard  66  66  ok
...                                             wrap-inspired/p3  wrap_wiki  73  73  ok
PASS -- all 12 job prompts match their expected templated overhead.
```

It also prints provenance, so a reader sees which numbers are copies and which are
reconstructions.

### B3 — do the dropped columns need a pass-through? — **No. Join later on `doc_id`.**

The review asked whether `source_text_sha1` is sufficient to rejoin. **It is the wrong key
and I should not have implied otherwise.** Identical documents produce identical hashes,
and web corpora contain exact duplicates — so a sha1 join is ambiguous precisely where it
matters. Use `doc_id`, which is already in every output row.

That is not a workaround; it is the source's own convention.
`10_postprocess/README.md:18` states *"`doc_id` is the canonical identity key"*, and the
selection scripts reference `doc_id` 41 times versus 14 for `tokens-llama2` and 5 for
`fasttext-ranking-v2`. Our output rows carry `arm`, `prompt_id` and `doc_id`; the sharded
input under `data_root/shards/<arm>/` carries `doc_id` alongside the text. The join is a
local key lookup.

What the change *would* have been, since you asked: carry extra columns through
`shard_arm()` into the shard parquet and echo them into each output row. Cost, using the
source's own figures (~1e9 output rows across the 12 jobs, ~110 bytes of JSON for the four
columns): roughly **110 GB uncompressed, ~30 GB after zstd — about 4%** of output volume.
So "expensive" in my original §8.3 was overstated; it is affordable. I still recommend
against it, for three reasons:

1. **It may be a no-op.** Those columns are derived artifacts of the JHU selection
   pipeline. Whether the HF-hosted datasets carry them at all is unknown — `text` is the
   only column this package requires. If they are absent upstream, a pass-through cannot
   invent them, and the join cannot be done afterwards either.
2. **It writes each value 2–4 times.** Storing them in the input shards instead costs one
   copy per arm rather than one per (arm, prompt) — strictly cheaper and equally joinable.
3. **It widens the output schema you specified**, days before handoff, to buy something a
   key lookup already provides.

**One consequence I added to the guide:** do not delete `data_root/shards/` when the run
finishes. It is the join table. If it is lost the join is still recoverable — re-download
and re-shard with the same parameters, and the manifest fingerprint will verify you got the
same `doc_id` assignment — but that is hours of work to avoid a `rm`.

If you want the belt-and-braces version, the cheap form is a `sharding.carry_columns` list
that copies any of the four into the input shards when present. Say the word; it is small
and I would rather add it deliberately than guess.

### B4 — tokenizer batching vs `TOKENIZERS_PARALLELISM=false` — **my §3.7 claim was wrong**

The review was right to be suspicious. Measured on real documents from the source's own
monitor log, single core, `TOKENIZERS_PARALLELISM=false`:

```
corpus: 4000 real docs, 10.1 M chars, 2.05 M tokens
  batched call      : 3.709s  -> 0.55 M tok/s
  per-document loop : 3.844s  -> 0.53 M tok/s
  batching speedup  : 1.04x
```

**1.04×, not "the difference between minutes and hours".** `HANDOFF_REVIEW` §3.7 overstated
this and is corrected here. With the Rust tokenizer single-threaded, HF's fast tokenizer
already amortises per-call Python overhead; batching buys almost nothing.

Is tokenization a throughput bottleneck? **No.** Projecting onto one arm-pass (9.5B
templated input + 3.6B output tokens, source's own figures):

| generation speed | tokenization share of wall clock |
|---|---|
| 2,000 tok/s/GPU | 1.3% |
| 4,000 tok/s/GPU | 2.6% |
| 8,000 tok/s/GPU | 5.0% |

Even on fast Blackwell cards it stays a rounding error, and the share is independent of
worker count since both scale together.

**Did the source set it false at tokenization time, or only around generation?** Settled
from the source: `rewrite_worker.py:32` sets it, the tokenizer loads at `:190`, and
tokenization happens at `:222` and `:285`. It was false during **all** tokenization. Source
parity confirmed — the value stays, exactly as the review instructed.

Batching stays too: identical values (preflight's `--verify-tokenizer-batching` proves it
on 1,000 real documents), marginally faster, already tested. What changes is the claim, not
the code. No measurement instruction added to the guide — at 1.3–5% there is nothing
actionable to measure.

### B5 — anything in §A wrong

Two, both above: **A2**'s severity framing (jobs do report `PARTIAL`; the real defect is
that the required assertion was unreachable) and **A4**'s "never works" plus its stated
mechanism (it worked when `--from-job` came first; `shift` drops from the front; and the
downstream consequence is a silent full re-run, not a failure). Both defects are real and
both are fixed — the corrections concern diagnosis, not whether to act.

---

## C1. Analysis only — nothing changed

Per instruction I did **not** restore `assign_wrap_styles`, add a mode switch, or touch
`data.yaml`. Analysis follows, from the source run's own per-arm counters.

### Arm-level output tokens

Source design, as it actually ran (pass 1 + distill, both complete):

| arm | pass 1 | distill | total | copies/doc |
|---|---:|---:|---:|:--:|
| quality-first | 3.399B | 2.581B | 5.981B | 2 |
| diversity-first | 3.812B | 2.845B | 6.657B | 2 |
| signal-disagreement-λ05 | 3.649B | 2.750B | 6.400B | 2 |
| **wrap** | 4.313B | 3.657B | **7.970B** | **2** |
| rewrite (ReWire) | 9.256B | 7.320B | 16.576B | 2 |

The source was **symmetric at 2 copies per document across every arm, wrap included.**

Under the current design, wrap-inspired is four full style passes and has no distill pass.
Per-style output per document, from the source's own status-2 assembly figures — easy 223,
hard 530, wiki 364, qa 457 tokens — sums to ~1,574 tokens/doc across all four styles. Over
wrap's 10,604,458 documents:

| arm | passes | pool tokens | copies/doc |
|---|:--:|---:|:--:|
| quality-first | 2 | 5.98B | 2 |
| diversity-oriented | 2 | 6.66B | 2 |
| disagreement-aware | 2 | 6.40B | 2 |
| **wrap-inspired** | **4** | **≈16.7B** | **4** |
| rewire-inspired | 2 | 16.58B | 2 |
| quality-base | 0 | — (raw) | 1 |

So wrap's pool is ~2.6× the other small arms' and ~4× what the source's wrap pass produced.
Note rewire-inspired reaches a similar token total at 2 copies/doc — its size comes from
having 21.2M documents, not from more passes.

### The repetition asymmetry — real at pool level, and it inverts at fixed budget

At full-pool level the review is right: 4 copies/doc versus 2.

But arms are not used as whole pools; they are cut to a token budget (the source cut every
arm to 5B rewritten tokens). Under **uniform** subsampling to a common budget `B`, expected
copies per document is `n_passes × B / P`, and since the pool `P ≈ n_passes × T × D`, that
reduces to `B / (T × D)` — **independent of pass count.** More passes give a larger pool,
so the same budget is met with a smaller fraction of it.

Concretely, at `B` = 5B tokens:

| arm | pool | fraction needed | expected copies/doc |
|---|---:|---:|:--:|
| quality-first | 5.98B | 84% | **1.67** |
| disagreement-aware | 6.40B | 78% | 1.56 |
| diversity-oriented | 6.66B | 75% | 1.50 |
| **wrap-inspired (4 passes)** | 16.7B | 30% | **1.20** |
| wrap under the source design | 7.97B | 63% | 1.25 |

Under uniform sampling the four-pass design gives wrap **less** duplication than
quality-first, not more. The direction of the review's concern is inverted here, and the
arm most constrained is actually quality-first, whose pool barely exceeds the budget and is
therefore nearly forced to its full 2 copies/doc.

**Where the concern does bite:** non-uniform selection. The source assembled by taking
*all* status-2 pass-1 output first and topping up from distill, sorted by quality — under
that rule a four-pass pool behaves differently and could concentrate. Notably the source
was already alert to this: for wrap specifically it used a **seeded random** supplement
rather than a quality sort, recorded in its manifest as
`"distill_selection": "seeded_random_seed42_no_quality_sort"`, explicitly to preserve the
arm-2-vs-arm-4 comparison.

### What "wrap-inspired" should mean, and my recommendation

The source rephrased **each document once**, with the style varied across the corpus by a
seeded RNG. Four full passes asks a different question: *every document rephrased four
ways.* Both are defensible; they are not the same experiment. I have not verified which the
WRAP paper does — the source cites arXiv:2401.16380 Appendix G for the prompt *text* only,
and I would not assert more than that without reading it.

**Recommendation: keep four passes.** The decisive argument is that it is a strict
**superset**. Having all four styles for every document, you can recover the source's
one-style-per-document design afterwards by selecting one pass per document — at zero
generation cost. The reverse requires regenerating three quarters of the arm. The costs are
2× wrap compute versus the source, and an obligation to decide duplication at assembly
time rather than inheriting it.

Two caveats if you take that route. Our shard indices differ from the source's, so a
post-hoc style assignment would be equivalent-but-not-identical to the original seeded one.
And whoever assembles the final datasets should **compute and report copies-per-document
per arm** rather than assuming symmetry — the table above shows the answer is not obvious
in either direction.

**Changed for C1: nothing.**

---

## The Blackwell-only decision

All 12 jobs now run on **B200 (`sm_100`) and B300 (`sm_103`) only**; H200 is excluded. This
supersedes divergence 0 above, which described a balanced H200+Blackwell mixture as the
mitigation. Restricting to one architecture family removes the confound rather than
balancing it, which is strictly better.

Enforced in three places, because the constraint is only as good as its weakest bypass:

* `configs/data.yaml` gains `compute_constraints.allowed_gpu_arch: [sm_100, sm_103]` with
  an `allowed_gpu_arch_major: [10]` fallback, so a Blackwell variant reporting an
  unanticipated capability is allowed with a warning while a Hopper (major 9) is not. It
  lives in `data.yaml` because it is an experimental-design decision, not a machine setting.
* `preflight.py` fails on any disallowed card and prints the steps to select only the
  Blackwell ones.
* **Every worker refuses on its own, before building the engine.** This is the one that
  matters: preflight can be skipped with `--skip-preflight`, and `03_run_job.sh` can be
  invoked directly. Generating even one shard on the wrong architecture would contaminate
  an arm silently.

`gpu_name`/`gpu_cc` stay in every sidecar — B200 and B300 are still two capabilities, so
the record remains worth having.

**Still unverified, and only answerable on his machine:** whether `torch 2.11.0+cu130` /
`vllm 0.22.0` / `flashinfer 0.6.11.post2` carry working `sm_100` and `sm_103` kernels.
CUDA 13 supports both and `sm_103` should at worst PTX-JIT from `sm_100`, but there is no
Blackwell card here. `preflight.py` compares each device against
`torch.cuda.get_arch_list()` and reports native / JIT / **no kernels** (hard fail), and
`00_setup_env.sh` runs a real one-prompt generation. Between them the question is settled
in minutes, and both refuse to let anyone "fix" a gap by installing a different build.

---

## What the review got wrong about the code

1. **A2 severity** — jobs do not silently report `DONE`; `state` becomes `PARTIAL` and
   every consumer gates on it. The real defect was the unreachable assertion.
2. **A4 mechanism and scope** — it worked when `--from-job` was first; the cause is
   front-shifting, not the loop variable; and the consequence was a silent full re-run
   rather than a no-op.
3. **A1 was understated** — the file inside a leaked claim directory would have been
   published to the Hub, not merely counted, and two adjacent reporting defects went
   unnoticed.
4. **`main` had 2 commits, not 3.**
5. **`source_text_sha1` was described as the potential rejoin key** (following my own
   framing in §8.3). `doc_id` is the key; sha1 cannot disambiguate duplicate documents.

Everything else the review asserted about the code was accurate, and the four items it
endorsed under §C3 are unchanged.

---
---

# Addendum — 2026-08-23 (later): round 3, scaling to ~100 GPUs

New information: Tianjian may have **on the order of 100 B300s** — roughly a dozen nodes,
where the repo was built for one. Everything above remains the record.

Headline: **§2 would have stopped the run before it started**, and **§1.1 was a real
silent-corruption hazard**. Both are fixed. §6 turned out to be a non-problem, and I say so
rather than inventing work.

---

## §1 Multi-node

### 1.1 The reap hazard — **CONFIRMED, and worse than reported**

The reading is right, and understated. `reap_stale_claims` did not merely reap
optimistically — it removed **every** claim it found, unconditionally. The function even
computed whether the shard was finished and then discarded the answer:

```python
finished = [p for p in d.glob(f"part_{si:05d}.done")]
if finished:
    pass                      # dead branch: computed, never used
try: cp.rmdir()               # ... removes it either way
```

So on twelve nodes: node 2's launcher wipes node 1's live claims, node 2's workers claim
shards node 1 is mid-generation on, both write, and because each shard is written
atomically with last-writer-wins, **row conservation still passes**. Duplicated GPU-hours
and a nondeterministic choice of surviving output, entirely silent. At ~100 GPUs and a
two-week run that is an expensive failure to not detect.

**What I built: heartbeat + age-based reaping, measured on the filesystem's own clock.**

* A worker holds a `ClaimHeartbeat` for the duration of each shard — a daemon thread
  touching the claim's mtime every 60s. The main thread is blocked inside `llm.generate()`
  for minutes at a time, so a background tick is the only way to expose liveness at all.
* `reap_stale_claims(stale_after_s=1800)` removes a claim only if it has gone that long
  without a heartbeat (a 30× margin over the tick), or if its shard already has a `.done`
  and the claim is simply litter — the dead branch above, now doing its job.
* Claim ages are compared against `fs_now()`, which stamps a probe file on the same
  filesystem and reads its mtime back, rather than against local `time.time()`. mtimes are
  written by whichever node owns the claim; comparing them to a local clock would put the
  whole scheme at the mercy of inter-node skew, which is exactly the sort of thing that
  silently makes a live claim look stale.
* Workers can now **take over** a demonstrably stale claim mid-run, so a node dying no
  longer strands its in-flight shards until someone notices. The takeover goes through the
  same atomic `try_claim`, so only one of many contenders wins.
* `--reap-claims --force` preserves the old semantics for an operator who knows nothing is
  running. The launcher never uses it.

The launcher's reap is now safe while other nodes are working, which is the property the
review asked for: safe **by construction**, not by instruction.

**What I rejected, and why.**

* *A one-time coordinator step separate from the per-node launcher.* Correct only if the
  operator gets the sequencing right every time, and the failure mode is silent — the same
  class of problem as the bug. It also does nothing for a node that dies mid-run.
* *Liveness by recorded host and pid.* Works only same-node: you cannot check a pid on
  another host without a login. It would have failed precisely in the multi-node case that
  motivated the fix.
* *No-op if any claim is younger than some interval.* A single fresh claim anywhere would
  block reaping every genuinely dead one. Per-claim age is the same idea done at the right
  granularity.

Verified: a live claim survives a reap launched from another node; an hour-old orphan is
reaped; a finished shard's claim is reaped as litter; a heartbeat rescues a claim that was
about to look stale; 12 simulated nodes race for one shard and exactly one wins.

### 1.2 Job lock — **fixed, moved off the shared filesystem**

The lock exists to stop two engines landing on one GPU, which is per-node. It was at
`$OUT_ROOT/raw/<arm>/<prompt>/.joblock`, so on a shared mount it would either block every
node after the first or behave according to whatever `flock` semantics that mount happens
to implement. Agreed that is not something to bet a two-week run on.

Now `${TMPDIR:-/tmp}/rewrite-vllm.<hostname>.<arm>.<prompt>.lock` — node-local storage for
reliable semantics, and the hostname in the filename so that a *shared* `TMPDIR` cannot
quietly re-create the problem. The failure message says explicitly that the lock is
per-node and that other nodes running the same job concurrently is how multi-node works,
so the loud case does not read as an error when it is correct behaviour.

### 1.3 Worker ID collisions — **fixed**

`progress_{worker_id}.json` → `progress_{hostname}_w{worker_id}.json`, and `host` is now
recorded in every `.done` sidecar alongside `gpu_name`/`gpu_cc`. Agreed on the reasoning:
at this scale progress reporting is how anyone knows whether a week-long run is healthy.

### 1.4 Orchestration roles — the part the review did not ask for but the fan-out needs

`run_all.sh` runs preflight → model → data → 12 jobs → postprocess → upload. Run
unmodified on twelve nodes, generation would fan out correctly (claiming handles it) but
**postprocess would not**: two nodes shuffling the same job share an output directory *and*
a bucket temp directory, and `bucketed_shuffle` unlinks buckets as it consumes them. That
is corruption, not duplicated work. Trim is idempotent and merely wasteful; shuffle is not.

Added three roles — `--prepare-only`, `--generate-only`, `--postprocess-only` — plus a
per-arm sharding lock so that nodes reaching a fresh `data_root` together serialise instead
of racing (losers wait for the manifest via `--wait-only`). Documented as §3.5 of the
guide. No scheduler, no new framework, no change to the claiming logic.

**On the SLURM question:** I do not think multi-node needs it. Claiming already provides
the coordination, and `scripts/optional/slurm_job.sbatch` stays optional and unsupported —
it now simply `exec`s the same bash path, so if he does submit it, each allocation behaves
as one node. Making it "real" would mean owning a second launch path for no capability the
bash one lacks.

---

## §2 Shard sizing — **the guard would have fired; the run could not have started**

Computed against the source run's real per-arm document counts, at the shipped default of
10,000 rows/shard and `min_shards_per_gpu: 20`:

| arm | docs | shards @10k | needs at 100 GPUs | |
|---|---:|---:|---:|---|
| quality-first | 6,136,187 | 614 | 2,000 | **REFUSES** |
| diversity-oriented | 5,876,747 | 588 | 2,000 | **REFUSES** |
| disagreement-aware | 5,602,476 | 561 | 2,000 | **REFUSES** |
| wrap-inspired | 10,604,458 | 1,061 | 2,000 | **REFUSES** |
| rewire-inspired | 21,214,299 | 2,122 | 2,000 | passes |

**Four of five arms refuse to start.** Not a tuning issue — a hard blocker on day one.

Re-derived for ~100 workers rather than scaled from the 8-worker rationale:

| rows/shard | shard @8k tok/s | tail idle over 12 jobs | shards, smallest arm | 20:1 at 100 GPUs |
|---:|---:|---:|---:|:--:|
| 10,000 | 11.5 min | 2.3 GPU-h | 561 | no |
| 5,000 | 5.7 min | 1.1 GPU-h | 1,121 | no |
| **2,000** | **2.3 min** | **0.5 GPU-h** | **2,802** | **yes, 28:1** |
| 1,000 | 1.1 min | 0.2 GPU-h | 5,603 | yes, but 2× the files |

**Changed the default to 2,000.** It clears the guard for every arm at 100 GPUs with the
same 28:1 headroom the source ran at, keeps the tail to a couple of minutes, and still
hands vLLM a batch well above its default `max_num_seqs`. Total output files across all 12
jobs: ~60k shards plus sidecars, which is unremarkable. At 8 GPUs it is equally fine, so
there is one default rather than two.

**Should shard size be derived from GPU count? No — and this is worth being explicit
about.** `num_gpus` lives in `cluster.yaml`, which is Tianjian's file, while shard size
feeds the manifest fingerprint. Deriving one from the other would let a machine setting
silently renumber `doc_id` and invalidate every `.done` marker. It stays an explicit value
in `data.yaml` with the sizing table in a comment, decided before job 1, and the guard's
error message now does the arithmetic and prints the value to set rather than leaving it to
be discovered. Both the config and the guide say in terms that it must be fixed before
job 1.

---

## §3 Requiring `doc_id` — **agreed, and made a hard failure by default**

**Recommendation to Wytro: add an explicit `doc_id` column to all six datasets before
pushing them.** It costs nothing at upload, and it makes the join key durable and
independent of this pipeline's sharding — which is strictly better than the guide warning
people not to delete a directory.

Argued both ways, and I came down on hard failure:

*For a warning:* the fallback is deterministic and reproducible, so a hard failure blocks a
run over something that is not a safety issue.

*For a failure, which is what I implemented:* the cost of complying is a single column at
upload time; the cost of not noticing is discovering in six months that a join you assumed
was possible depends on a scratch directory nobody was told to preserve. A warning at
download time, during a step that prints thousands of lines and runs for hours, is not a
control. And the escape hatch is one line.

So `sharding.require_doc_id: true` is the default and absence is a hard stop naming the
consequence and the fix; setting it `false` accepts the synthesized index. Either way the
choice is recorded per arm in the manifest as `doc_id_source: "dataset" | "synthesized"`,
surfaced in the summary line, and carried into `manifests/data_manifest.json` — so the
decision is auditable rather than remembered.

---

## §4 Throughput calibration — **added**

Agreed the per-shard `tok/s` logging is not sufficient, and specifically for the reason
that it reports a rate but never a **projection**. Nothing in the repo turned a rate into
"this run will take N days", which is the number that decides whether to keep going.

`scripts/06_calibrate.py`, run by `run_all.sh` after preflight and before job 1. It prints
per-job and total projected wall clock computed from the **actual manifest row counts**,
plus a sensitivity band at half and double the measured rate, because the projection is
only ever as good as one measured number.

Two sources, and the second is better once it exists:

* `--measure` — one shard-sized batch on one GPU with the real model, real prompt and real
  documents.
* `--from-sidecars` — derives the rate from shards already generated. Free, and far more
  representative than a single batch because it is the actual run across the actual fleet.
  It picks this automatically once ≥5 shards exist, so re-running it on day two gives a
  much better number than day zero.

It exits 0 even when it cannot measure: advisory, not a gate. I did not want a throughput
estimate to be able to block a correct run.

---

## §5 Wrap selection caveat — **moved to where the data goes**

Agreed the reasoning was stranded in a review document. It now lives in two places a future
reader will actually meet:

* **`configs/data.yaml`**, as a block immediately above the `wrap-inspired` arm, with the
  worked numbers and the recommendation.
* **The uploaded dataset cards.** `05_upload_to_hf.py` now writes a `README.md` into every
  Hub repo describing the arm, the prompt, the locked generation settings, the column
  semantics, and — for `wrap-inspired` only — the full selection caveat.

The card matters more than the config, because the person who decides how to subsample this
arm to 10B tokens may be neither of us and may never see this repository. The caveat has to
travel with the data.

Every card also states that `n_output_tokens_llama2` is the column to budget on and that
`status == 2` is the filter for training use — two things easy to get wrong from the
outside.

---

## §6 Is data prep a bottleneck? — **No. Measured, and documented rather than fixed.**

Benchmarked the actual `shard_arm` pipeline on real documents, single process:

```
utf-8 encode only                    5,790 MB/s
sha1 (encode + hash + hexdigest)     1,039 MB/s
build arrow table                    3,219 MB/s
parquet write                          263 MB/s
FULL shard pipeline                    190 MB/s   <- the number that matters
```

At ~4.92 chars/token (measured on real source documents), Wytro's stated ~500B input
tokens is ≈2,460 GB of text:

| | wall clock |
|---|---|
| serial sharding, 1 process | **~3.6 h** |
| across 8 processes | ~0.4 h |

**Hours, not days**, so I have not parallelised it. Parquet write dominates at 263 MB/s and
would be the thing to attack if it ever mattered, but adding multiprocessing to a
correctness-critical path that fingerprints its own output, to save three hours of a
multi-day run, is a bad trade.

Two honest caveats: this excludes the HF **download**, which is network-bound and may well
be the larger term, and it excludes `datasets` iteration overhead, so 4–8 h end to end is a
fairer expectation than 3.6. Both are in the guide so the wait is expected rather than
alarming, and `--prepare-only` exists so it can be done once, ahead of the fleet, rather
than with 100 B300s idling.

---

## What the review got wrong

Very little, and only in degree.

1. **§1.1 was understated.** Reaping was not merely unsafe under concurrency — it was
   unconditional, with a computed-then-discarded check that made it look safer than it was.
2. **§1.2's "either blocks every node or behaves inconsistently"** is right, and there is a
   third case worth naming: on a filesystem where `flock` silently no-ops, it would have
   appeared to work while protecting nothing.
3. **§2 understates the severity.** This was not "the guard would fire" as a tuning
   annoyance — four of five arms could not have started, and the message did not say what
   to set.
4. **§6's framing** assumed serial sharding might be days. It is hours; the review was right
   to ask for the number rather than assume.

Everything else — that the generation core is already multi-node safe, that `os.mkdir` is
the right primitive, that `worker_id` does not determine which shards get processed, that
shard size interacts with the fingerprint and must be fixed before job 1 — was accurate.

---

# Addendum — 2026-08-25: round 4, realignment to the changed design

Round 4 was not a review round. The experiment design changed after rounds 1–3 were built,
so parts of this package were wrong rather than merely improvable. The per-item record is
below; the standalone write-up, including everything I established from the three sources
and every place the round-4 brief was mistaken, is **`docs/DESIGN_DELTA.md`**. Read that
first. This section is only the disposition list.

Earlier sections of this file are unchanged and remain accurate as history — including
§C1, whose recommendation this round reverses on instruction.

## §1 wrap-inspired: one style per document — RESTORED

**Disposition: done. This reverses round 2 §C1, on explicit instruction.**

`assign_wrap_styles` is back, verbatim from `07_rewrite/rewrite_worker.py:54-62`, as
`src/rewrite/wrap_styles.py`. `wrap-inspired` is now **2 jobs**: one styled pass that picks
one of `easy`/`hard`/`wiki`/`qa` per document via
`np.random.default_rng([42, shard_index])`, plus the shared distill pass.

Four things about this that are worth stating plainly:

1. **The job arithmetic is not "4 − 2".** wrap loses three style passes *and gains a
   distill pass it never had*. This package's `wrap-inspired` had no `p2` at all, which
   §C1 above acknowledged and which the round-4 brief did not mention. 2+2+2+2+2 = **10**.
2. **The prompts were correct.** The brief flagged a worst case where
   `prompts/wrap-inspired/p1..p4` might be the abandoned paper-verbatim set — a
   corpus-wide silent corruption. They are not: all four are byte-identical (md5-verified)
   to the production `easy/hard/wiki/qa` files. The abandoned set is identifiable by its
   `medium` key and appears nowhere in production. The files are now named
   `style_{easy,hard,wiki,qa}.txt` so job identity and style identity cannot be confused.
3. **The style is recorded**, as an 11th output key `wrap_style`, present in every row of
   every job (empty outside the styled pass) so the ten output sets share one schema. The
   source added the column only in wrap mode; a ragged schema across jobs was not worth
   the ~15 bytes/row saved.
4. **Seed determinism survives resume, and this is tested rather than asserted.**
   `tests/test_wrap_styles.py` (33 checks) covers: a golden vector pinned against
   `np.random.default_rng([42, i])` so a numpy PCG64 change is caught; identical output
   from two separate interpreter processes; a short draw being a prefix of a long one; the
   signature taking no worker id at all; and 25.0% ± 0.1pp balance over 4M draws and over
   2,000 shards. `tests/test_integration.py` adds the end-to-end property: every wrap row's
   style equals `assign_wrap_styles(shard_index, n_rows)`, and **a deleted shard
   regenerates with an identical style vector**.

**One thing I did not preserve, and could not.** Shards here are re-cut at 5,000 rows, so
`shard_index` is ours, not the source's. The assignment is reproducible *within this
pipeline*; it is not the same draw the 1.5B run made. Different corpus, different sharding
— the mechanism is preserved, not the specific values. Stated in the module docstring and
in DESIGN_DELTA §2 rather than left for someone to discover.

## §2 Only the remainder is rewritten — CONFIRMED, and stronger than reported

**Disposition: done.**

Each block upstream is a shared raw 20B core plus a block-specific remainder, and only the
remainder is rewritten (`02_select/select_600m.py:213-233`). On the Hub the split is
physical: each block folder holds the remainder only
(`upload_blocks.py:222`, `keep = pc.equal(tbl['is_core'], want_core)`) and the core ships
once as `shared-core/`. The core is provably identical across the five carrying blocks
(sha256, `select_600m.py:631-637`), which is what makes "rewriting it per arm would be
5× redundant" a fact rather than a guess.

`configs/data.yaml` now states the GPU/no-GPU split in its header, and every arm carries
`docs` and `source_tokens_llama2`, so a reader can see which tokens cost GPU time without
opening a Python file.

**Per your decision:** `quality-base` and `shared-core` are dropped as *arms* — comment-only
accounting, not config entries — so preflight and the manifest ignore them and nothing
downloads 20B raw tokens for no reason. The unfillable `quality-base` `repo_id` is deleted.
`manifests/data_manifest.json` records both under `raw_not_rewritten` so the accounting is
still machine-readable.

**Correction to the brief:** the 80B and 140B per-block figures it quoted are block
*totals*, not source budgets. The real remainders are 60B (×4) and 120B for rewire —
κ=2 applies to the remainder, not the total (`select_600m.py:19-20,81`). Treating them as
rewrite budgets would have overstated the GPU workload by 33% and 17%.

## §3 Data access layer — one gated repo — DONE

**Disposition: done, with two Hub-side findings the brief did not contain.**

`hf.repo_id` + `hf.revision` (pinned to sha `6e18cda6…`) + a per-arm `subdir`. Downloads
use `allow_patterns` so each arm pulls only its own folder rather than all 662 GB.

1. **The repo is `gated: "manual"`.** Listing works unauthenticated; fetching bytes does
   not. That failure mode surfaces late and on worker nodes, so `preflight` check 7 now
   *downloads a real file* to prove read access and names Wytro as the person who must
   clear it. This is the most likely day-zero blocker.
2. **The card declares no `configs:` block**, so the Hub auto-converter exposes one
   `default`/`train` config globbing every folder. `load_dataset(repo_id)` would silently
   concatenate `shared-core` into all five arms. The loader therefore addresses folders by
   explicit `data_files` glob and never by config name. A ready-made `configs:` block
   exists at `data_reports/DATASET_CARD_DRAFT.md` and should be pushed, but nothing here
   depends on it.

**doc_id.** int64, globally unique across the 600M corpus, stable across blocks. Unique
*within* an arm (which is all the row-conservation proof needs); not unique across arms,
since remainders are drawn from one pool by different criteria. The output rows already
key on `(doc_id, arm, prompt_id)`, so nothing needed changing. **Per your instruction, no
cross-arm disjointness check was added here** — it belongs upstream at `02_select` where
the id arrays are local; a pairwise numpy intersection there is seconds, whereas doing it
here would mean re-downloading ~300M ids to learn something the selection stage already
knows.

**Pre-existing defect found and fixed:** `compute_fingerprint` folded `shard_target_rows`,
`shard_target_bytes` and `DOC_ID_POLICY` but **not `require_doc_id`** — yet flipping that
flag switches `doc_id` between the dataset column and a synthesized index, renumbering
every row, while leaving the fingerprint identical. A resume across that flip would have
matched `.done` markers written against different doc_ids. `doc_id_source` is now folded in.

## §4 Every derived number recomputed — DONE

**Disposition: done. Measured, not assumed.**

**Compression ratio.** The brief said the 1.5B outputs "have both input and output token
counts recorded". The row-level parquets that had both **are deleted** — `data_rewrite/`
no longer exists. Two survivors were used: the exact arm-level census in
`07_rewrite/progress/*.json` + `09_Distill/progress/*.json`, and the seeded ~36k-row
sampled monitors, which do carry per-row `status`.

Per-prompt `r` in llama-2 tokens, census-exact:

| arm | r (pass 1) | r (distill) | sum | independently quoted |
|---|---:|---:|---:|---:|
| quality-first | 0.3399 | 0.2581 | 0.5981 | 0.598 |
| diversity-oriented | 0.3812 | 0.2845 | 0.6657 | 0.666 |
| disagreement-aware | 0.3649 | 0.2750 | 0.6400 | 0.640 |
| wrap-inspired | 0.4313 | 0.3657 | 0.7970 | 0.797 |
| rewire-inspired | 0.4628 | 0.3660 | 0.8288 | 0.829 |

The right column is `select_600m.py:114-123`'s own recorded yields, computed by someone
else from the same run. All five match to three decimals, which is what convinced me the
denominator was right. On the `status == 2` restriction you asked for: it moves these by
<3% and, in this denominator, truncation biases *upward* not downward — status-1 is
0.02–0.33% of documents and contributes a maxed-out 4,096-token output. Both figures are
in DESIGN_DELTA §5.

**What moved:**

| number | was | is |
|---|---|---|
| output tokens | `est_output_tokens_per_arm: 100e9` × 5 = 500B | per-arm; **261.49B** total (old figure overstated by **1.91×**) |
| disk | hardcoded "~2.4 TB / ~0.7 TB" | derived from config; **1.125 TiB / 0.338 TiB** |
| `shard_target_rows` | 2000 | **5000** |
| row envelope | 220 B | 235 B (the `wrap_style` key) |

The per-arm spread is 35.9B–99.5B — 2.8× — so the scalar was not merely wrong in magnitude,
it was the wrong shape. The disk figure is now computed in one place and quoted from there.
`06_calibrate.py` takes token *volume* from the per-arm measured estimates and uses the
measured rate for *speed only*, which is what it actually measures; it also cross-checks
the two and flags a >2× disagreement.

**`shard_target_rows`.** Round 3's 2000 was derived against a smallest arm of 5.6M docs,
where 10,000 rows would have failed the 20:1 guard. The smallest remainder is now 33.4M
docs, so every candidate passes and the binding constraint became filesystem metadata load
instead. 2000 would create 147,955 input shards and ~592k output files (63,241 shards for
`rewire-inspired` alone); 5000 gives 59,184 and ~237k, still 67:1 at 100 GPUs with a ~6 min
tail. It stays an explicit constant, not derived from `num_gpus`, for the round-3 reason
that still holds.

## §5 One-button run — updated, nothing weakened

**Disposition: done.**

`GUIDE_FOR_TIANJIAN.md` §1 is rewritten: 10 jobs, the real per-arm budgets, the half+half
explanation, the new disk figures, and an explicit "you are only rewriting half the corpus"
note so a missing sixth arm reads as intentional. §2.5 now explains the gating.

**Placeholders.** Six WYTRO blanks became one. The five arm `repo_id`s collapse into a
single `hf.repo_id` + pinned `revision`, both resolved from `upload_blocks.py:67` and
confirmed against the Hub; `quality-base`'s is deleted. Still outstanding:
`upload.repo_template` and `HF_TOKEN_WRITE` (both Wytro, both upload-only), plus **one new
item: granting Tianjian's HF account access to the gated dataset.**

Preflight gained a check and lost none. The gated-repo check is a new hard gate, and I
deliberately made it fetch a file rather than read metadata, because metadata succeeds
without access.

## §6 A claim in this package's own docs that was wrong

`docs/POSTPROCESSING.md` said the source's wrap trim "dispatched per row on a `wrap_style`
column", and that claim had been used to justify dropping the column. **It is false.**
`01_strip_prefix_wrap.py:185-188` applies the identical rule to all four styles and
branches only on `distill` vs `rewritten` — which is job-level and already modelled here.
`wrap_style` grouped the *statistics*, not the logic.

Corrected in place, and the real behaviour is now reproduced: `trim_job` emits a
`by_wrap_style` block with per-style doc share, token share and tokens/doc. The trim rule
is untouched and still matches the source byte-for-byte over 72,443 comparisons.

I mention it because it is the same failure mode this whole round exists to fix — a
plausible-sounding claim about the source, written down once, then used as grounds to
delete something.

## §7 What was and was not executed

**Ran, green:**

| harness | result |
|---|---|
| `tests/test_wrap_styles.py` | 33/33 — golden vector, cross-process determinism, resume prefix, balance, config wiring |
| `tests/test_integration.py` | all checks pass — 10 jobs, wrap 2 jobs, 11-key rows, per-row style match, deleted-shard regeneration identical |
| `tests/test_trim_parity.py --source-root …` | 72,443 comparisons, **0 mismatches** (unchanged by the per-style stats) |
| `tests/test_shuffle_parity.py --source-root …` | 63,000 rows byte-identical, unchanged |
| `scripts/check_placeholders.py` | 11 TIANJIAN + 1 WYTRO remaining, as expected |

**Not run, and why:** `scripts/preflight.py` end to end and
`scripts/verify_prompt_parity.py` both need a real Qwen2.5-7B-Instruct tokenizer and filled
cluster paths; neither exists here. The overhead constants they check (150 / 185 /
72 / 66 / 73 / 83) are unchanged from round 3, which verified them, and the code paths that
consume them are exercised by the integration test against a stubbed tokenizer. **No GPU
work of any kind was run this round.**

**Untouched, per instruction:** engine args, sampling params, the trim rules themselves,
shuffle, claiming, reaping, the Blackwell-only constraint, `min_shards_per_gpu`.

## §8 Open questions

1. **The ReWire top-half filter is not implemented.** `rewire-inspired` gets κ=2 (120B
   rather than 60B) specifically to buy headroom for a post-rewrite fastText filter — at
   1.5B: 20.0B source → 16.22B rewritten → cutoff 0.11456 → 5.0B kept
   (`_step3`/`_step4_rewrite_summary.json`). This package has no such stage, so that arm
   ships **~99.5B tokens unfiltered**. Per your decision this is out of scope for round 4
   and flagged rather than built. *Owner: Wytro, downstream. Blocks the run: no. Blocks
   rewire's training mix: yes.*
2. **doc_id overlap between arms is unmeasured** — expected and harmless here, but nobody
   has quantified it. Upstream job, as agreed.
3. **`quality-base` has no home.** 50B tokens, 37.3M documents, local-only. Whoever builds
   the final training mixes needs it; nothing in `rewrite-vllm` produces or moves it.
4. **`gpu_memory_utilization` is genuinely ambiguous in the source** — `0.90` in
   `07_rewrite/README.md:21` and the argparse default, `0.85` in the sbatch that actually
   ran. Unchanged here; flagged only so nobody later "fixes" it by citing the README.

---

# Addendum — 2026-08-25 (later): round 5, closing the loose ends

Small round. Five items plus a prompt-provenance check, no scope expansion. `DESIGN_DELTA.md`
is updated in place; this is the disposition list.

**Two of the five items in the brief turned out to be wrong, and both inverted a conclusion**
(§2 and §4). A third correction landed in §6: an apparent prompt difference resolved in this
repo's favour, and a method error of mine on the way there.

## §1 The 50B invariant — ADDED, and the brief's 30B target is correct

**Disposition: done. The inferred number checks out.**

`DESIGN_DELTA` §6 now states the constraint the half+half structure exists to satisfy:
**every arm lands at 50B final training tokens.** `quality-base` is 50B raw; every other arm
is 20B raw core + 30B rewritten.

The brief flagged the 30B as inferred from structure rather than read from source. It is
correct, and the source does support it:

- `select_600m.py:78-79` — `CORE_TARGET = 20B`, `QBASE_TARGET = 50B`.
- `select_600m.py:358` — quality-base's 50B is explicitly a **`final_training`** budget:
  *"All 50B is used as-is; never rewritten. This is what the model trains on."* It is the
  only arm whose final budget is stated directly, so it sets the target for the others.
- `02_select/README.md` + `verify_materialize.py:116-126` — `quality-base` is the **top-50B
  prefix** of the same `ftq` order whose top-20B prefix is the core, so it already *contains*
  the core. It is `20B core + 30B next-best raw`, and the other arms must match at 50B.
- 1.5B analogue: `select_10b.py:48-51` has `SHARED_TARGET = 5B` / `TARGET = 10B`, and the
  postprocess mixed a 5B core with a 5B rewritten selection. 10B = 5B + 5B there,
  50B = 20B + 30B here. The core's share moved from 50% to 40% between scales; the invariant
  is the 50B total, not the split.

The produced-vs-needed table is in `DESIGN_DELTA` §6 and in the header of `configs/data.yaml`.
The brief's headroom figures were right to within rounding, except rewire — see §2.

## §2 rewire's "3% headroom" — WRONG. It is the roomiest arm, not the tightest.

**Disposition: the "top half" error is fixed everywhere. The conclusion drawn from it is
reversed, with evidence.**

The wording error was real and is corrected in `DESIGN_DELTA` §10.1, `configs/data.yaml`, and
the generated dataset card for that arm. But the error was deeper than a fraction:

`_step4_rewrite_summary.json` states the mechanism outright —

> `pipeline`: "rewrite broadly → fasttext score rewritten → **keep top 5B**"
> `selection_note`: "…token-budget-matched top-5B (**NOT the paper top-10%**)"
> `target: 5000000000`, `overshoot: 351`, `filled: true`

**It is a fixed-BUDGET fill, not a fixed threshold.** The cutoff `0.1145634651184082` is the
*output* of filling to a 5B target, not an input. The brief's stated risk — "a fixed threshold
retains a different fraction at this scale, and a few points down puts the arm short" — does
not apply, because a budget fill cannot come up short while the pool exceeds the budget. What
moves instead is the cutoff score, i.e. the *quality* of the kept set.

**And the headroom is not 3%.** Required retention is `30B / 99.46B = 30.2%`, against 30.8%
realized at 1.5B — near-identical, because κ=2 was sized so the pool-to-budget ratio carries
over (3.32× here, 3.24× there). Produced ÷ needed:

| arm | produced | needed | headroom |
|---|---:|---:|---:|
| quality-first | 35.88B | 30B | **+19.6%** ← tightest |
| diversity-oriented | 39.94B | 30B | +33.1% |
| disagreement-aware | 38.40B | 30B | +28.0% |
| wrap-inspired | 47.82B | 30B | +59.4% |
| rewire-inspired | 99.46B | 30B | **+231%** ← roomiest |

Each ratio is `r_arm × source / 30B`, so it reproduces the 1.5B headroom identically — budgets
scaled 6× and `r` was carried over. Every arm demonstrably filled at 1.5B
(`shortfall_vs_target: 0`; `filled: true`). **The tight arm is `quality-first` at +19.6%**, and
what threatens it is not the filter but the `r` transfer in §4.

Recommendation recorded, not decided: implement as `fill_to(order_by_score_desc, tokens, 30e9)`
rather than porting `0.11456`, since a budget fill self-corrects if the score distribution
shifts and the 600M rewritten text is a different population. Filter still not implemented;
still Wytro's.

## §3 The two overhead-verification paths — CHECKED, both still correct

**Disposition: verified, nothing was broken. Measured, not assumed.**

Against the real Qwen2.5-7B-Instruct tokenizer on disk, all six templates match:
`p1_wiki` 150, `p2_distill` 185, `wrap_easy` 72, `wrap_hard` 66, `wrap_wiki` 73, `wrap_qa` 83.

All four wrap values are asserted, in three independent places:

- **runtime** — `PromptSpec.overheads()` yields one triple per prompt *text* (four for the
  styled pass); `engine.check_overheads()` measures each; `run_worker` fails the job if any
  mismatches and names which. Expected values live per style in
  `prompt_defs.wrap_styled.styles[*].expected_overhead`. The sidecar records a dict of four
  for that job rather than a single integer.
- **preflight check 6** — same expansion, and it prints `n_texts` (6 across 10 jobs), so a
  silent collapse to one text per job shows up as a changed count.
- **`verify_prompt_parity.py`** — runs standalone from the YAML without `config.py`'s loader,
  so it needed its own `_units()` expansion; it prints every expected-vs-actual pair and
  independently re-checks the style ORDER against `["easy","hard","wiki","qa"]`.

Four values asserted where there were four before. Neither check narrowed to a single style.

## §4 `r` transferred across populations — ACKNOWLEDGED, and it is the real risk

**Disposition: added as open question 3; precision reduced where it was overstated.**

The brief is right that 261.49B claims precision the transfer does not support. Recorded in
`DESIGN_DELTA` §10.3: the 1.5B `quality-first` was the top of the *full* pool while the
round-4 remainder is **core-excluded**, and mean document length varies ~1,600 tok/doc
(`quality-first`) against ~949 (`wrap`, `rewire`), with `r` not length-invariant.

**Direction of bias, reasoned:** core-exclusion should push `r` slightly **up** for the four
60B arms — the distill prompt condenses proportionally more from dense text, and the core
skimmed exactly that off the top, leaving more repetitive material that compresses less under
a preserve-the-facts instruction. So the estimates are probably mildly conservative. An
argument, not a measurement; not strong enough to lean on.

Totals are now quoted as **~260B ± 10%** in the summary table and in `data.yaml`; the exact
figure is retained only where the arithmetic is shown.

**This is where §2's concern actually belongs.** `quality-first`'s +19.6% headroom sits inside
that band: if `r` lands ~16% below the transferred value, that arm goes short of its 30B and
needs regenerating. It is the single most consequential open number in the document — and it
settles empirically on day one, from `06_calibrate.py`'s cross-check against the first job.

## §5 Prefill split — ADDED. No mistuning found.

**Disposition: reporting added; engine args untouched; the brief's implied concern does not
hold.**

`06_calibrate.py` now carries prompt tokens through both rate paths (sidecars and the live
measurement) and reports prefill and decode separately: prompt tok/s, output tok/s, the
measured mix, and whether it matches the 2.75:1 the config implies — flagging >25% divergence,
which would mean the token estimates are wrong rather than the engine. Guidance for Tianjian
is in `GUIDE_FOR_TIANJIAN.md` §4.5.

**On whether the inherited values look mistuned: no, and the evidence is direct.** The source
ran the same shape — its census counters give 192.6B input against 69.5B output, **2.77:1**,
within 1% of ours — on the same engine version, to completion. It never passed
`max_num_batched_tokens` or `enable_chunked_prefill`; it inherited
`max_num_batched_tokens=16384` / `enable_chunked_prefill=True`, visible in its own logged
config dump (`07_rewrite/logs/rw_*.out:6,9`) and already recorded in `configs/vllm.yaml` under
`inherited_defaults_do_not_pass`. So the prefill path here is the one that produced the 1.5B
corpus at this exact ratio. Nothing changed; the guide frames it as measure-first and bring
findings to Wytro.

## §6 Prompt provenance vs the published originals — **RESOLVED, all six identical**

**Disposition: comparison complete. All six templates verified byte-identical. Nothing
changed, and nothing needed changing.** Full detail in `DESIGN_DELTA` §9.

`blab-jhu/KYS-Configs` is `gated: manual` and its `prompts/` files 401, so the comparison
used **git blob SHA-1 OIDs** from the Hub tree API — SHA-1 over exact file bytes, published
even when content is not. Byte-exact, trailing newlines included.

| template | bytes | verdict |
|---|---:|---|
| wiki-grounded | 597 | IDENTICAL (`802aff3d…`) |
| distill | 842 | IDENTICAL to `finephrase/nemotron/distill.md` (`200cd2c3…`) |
| wrap easy / hard / wiki / qa | 218 / 197 / 231 / 248 | IDENTICAL, all four |

**Method error worth recording.** My first pass compared the *JSON container*
`wrap_prompts.json` against our four `.txt` files and failed to match under 18,432
serialisations — which proved nothing, and I reported it as "inconclusive" when it was simply
the wrong comparison. The right one is per value: extract each JSON value, compare to the
corresponding `.txt`. Done against the source-local `prompts/wrap_prompts.json`, all four are
byte-identical, and its key order is `easy, hard, wiki, qa` — the order that feeds the style
seed.

### The distill discrepancy: the published file drifted, not ours

`KYS-Configs` ships distill **twice, with different bytes**: `prompts/distill_prompt.txt`
841 B `38da40ce…` versus `prompts/finephrase/nemotron/distill.md` 842 B `200cd2c3…`. The
second is byte-identical to ours. So does the source's own surviving copy at
`projects/rewrite/prompts/distill/distill_prompt.txt` — 842 B, `200cd2c3…`. Three artifacts
carry 842; only the one published file carries 841.

**Was the trailing newline a log artefact?** This was the live question, and it now has a
direct answer. Every one of the **14** bake-off templates was recovered from its own logs —
strip the chat wrapper from `templated_input`, substitute the logged `doc_text` back to
`[TEXT]` — and compared against its published `finephrase/**` file:

**14 / 14 reproduce the published file byte-for-byte (git OID match).**

A logger that appended a newline per record would make all 14 recoveries one byte long and
none would match. All 14 match. So the `\n` before `<|im_end|>` in the distill log is real
template content, and `nemotron_distill` recovers to exactly our 842-byte file. The
reconstruction did not pick up an artefact; the published `distill_prompt.txt` lost its final
newline in packaging.

**Honest limit:** `09_Distill/launch_dataset.sh:20` shows production loaded
`/scratch/.../data_rewrite/prompts/distill_prompt.txt`, and that tree is gone. So the evidence
is the bake-off run plus two surviving copies, not production's own bytes. All three agree and
nothing anywhere carries the 841-byte form except that one published file.

### Consequence for the overhead assertions

Kept as the cheap runtime guard, and their expected values are now known to derive from the
originals, because our files *are* the originals. All six re-measured against the real
Qwen2.5-7B-Instruct tokenizer: 150 / 185 / 72 / 66 / 73 / 83, all matching `configs/data.yaml`.

**One thing not to carry forward:** an unchanged overhead is not evidence a template is
unchanged. The 841/842 pair produces the *same* overhead (185) and the same total length,
differing at exactly one token — Qwen merges `.` + `\n` into one token (id 624) where the
alternative is `.` (id 13). This also corrects the round-5 brief's premise that a trailing
newline "still shifts the overhead": for this tokenizer it does not. The count is a smoke
test; the byte comparison is the proof.

**B6 and B7 are closed.** Residual non-blocking item: `KYS-Configs` should not ship two
different distill files.

## §7 What was and was not executed

**Ran, green:**

| harness | result |
|---|---|
| `tests/test_wrap_styles.py` | 33/33 — golden vector, cross-process determinism, resume prefix, balance, config wiring |
| `tests/test_integration.py` | all checks pass — 10 jobs, wrap 2 jobs, 11-key rows, per-row style match, deleted-shard regeneration identical |
| `tests/test_trim_parity.py --source-root …` | 72,443 comparisons, **0 mismatches** (unchanged by the per-style stats) |
| `tests/test_shuffle_parity.py --source-root …` | 63,000 rows byte-identical, unchanged |
| `scripts/check_placeholders.py` | 11 TIANJIAN + 1 WYTRO remaining, as expected |

**Not run, and why:** `scripts/preflight.py` end to end and
`scripts/verify_prompt_parity.py` both need a real Qwen2.5-7B-Instruct tokenizer and filled
cluster paths; neither exists here. The overhead constants they check (150 / 185 /
72 / 66 / 73 / 83) are unchanged from round 3, which verified them, and the code paths that
consume them are exercised by the integration test against a stubbed tokenizer. **No GPU
work of any kind was run this round.**

**Untouched, per instruction:** engine args, sampling params, the trim rules themselves,
shuffle, claiming, reaping, the Blackwell-only constraint, `min_shards_per_gpu`.

## §8 Open questions

1. **The ReWire top-half filter is not implemented.** `rewire-inspired` gets κ=2 (120B
   rather than 60B) specifically to buy headroom for a post-rewrite fastText filter — at
   1.5B: 20.0B source → 16.22B rewritten → cutoff 0.11456 → 5.0B kept
   (`_step3`/`_step4_rewrite_summary.json`). This package has no such stage, so that arm
   ships **~99.5B tokens unfiltered**. Per your decision this is out of scope for round 4
   and flagged rather than built. *Owner: Wytro, downstream. Blocks the run: no. Blocks
   rewire's training mix: yes.*
2. **doc_id overlap between arms is unmeasured** — expected and harmless here, but nobody
   has quantified it. Upstream job, as agreed.
3. **`quality-base` has no home.** 50B tokens, 37.3M documents, local-only. Whoever builds
   the final training mixes needs it; nothing in `rewrite-vllm` produces or moves it.
4. **`gpu_memory_utilization` is genuinely ambiguous in the source** — `0.90` in
   `07_rewrite/README.md:21` and the argparse default, `0.85` in the sbatch that actually
   ran. Unchanged here; flagged only so nobody later "fixes" it by citing the README.


---

# Addendum — 2026-08-25 (final): round 6, defects and handoff

Last round before handoff. Two defects, one documentation error, one ceiling to state, and a
short check that came back negative. **Nothing on the generation path was touched** —
`prompts/`, `engine.py`, `run_rewrite.py`, `postprocess.py`, `shuffle.py` and `vllm.yaml` are
byte-identical to round 5.

## §1 The sharding lock had no stale-lock recovery — FIXED

**Disposition: fixed to the round-4 claim standard, and the failure is now covered by tests.**

The report is exactly right, including the asymmetry argument. `data.py` claimed an arm with
`lock.mkdir()` and then waited `while not existing.exists(): time.sleep(10)` — no liveness
check, no bound. A process killed while holding it wedged every later run of the entire fleet,
and the only signal was the same "still waiting" line every ten minutes forever. Round 4 had
already established the correct treatment for precisely this failure mode on shard claims and
this path never received it.

**What was built**, reusing the claim discipline rather than inventing a second one:

- `lock_age_s` / `heartbeat_lock` / `LockHeartbeat` / `break_lock` / `read_lock_owner` —
  path-keyed analogues of the shard-claim helpers, sharing `fs_now()` so ages are read from
  the shared filesystem's clock rather than a local one. The claim functions are untouched.
- `acquire_dir_lock(lock, done_when, owner, stale_after_s, max_wait_s, label)` — returns
  True (we hold it, do the work), returns False (`done_when()` came true while waiting), or
  `stop()`s with the manual fix if the wait exceeds the bound.
- The holder heartbeats for the whole held region — the dataset open *and* the batch loop.
  Without that, sharding a 126M-document arm would go quiet past the threshold and another
  node would take the lock over while the first was still writing into the same directory.
- `stale_after_s` reuses `cluster.compute.claim_stale_after_s` (1800s), so the two lock
  mechanisms share one tunable. Heartbeat interval is `min(60, stale/5)` — the same 30×
  margin the claim path uses.
- On takeover the new holder clears orphaned `part_*.parquet`. Reaching that code means there
  is no manifest, so any shard files present are debris from a dead run: they cannot be
  reused (nothing records how many there should be or what fingerprint they carry) and
  leaving them risks mixing two runs' output.

**On the design question raised:** takeover *and* a bound, not one or the other. Staleness
handles the common case (holder died) by taking over, which is what the claim path does and
what keeps a fleet moving. The bound only fires when a holder is alive and heartbeating but
producing nothing — a case takeover cannot resolve — and it fails with instructions rather
than sleeping. `SHARD_LOCK_MAX_WAIT_S` is 24h, generous because a live holder refreshes the
lock and so never trips it.

**Also found, same shape:** `scripts/02_download_data.py --wait-only`, which worker nodes run
while one node prepares data, had its own `while True: sleep(15)` with no bound. A preparing
node that dies leaves every worker waiting forever. Now bounded by `--wait-timeout-s`
(default 24h), and on timeout it distinguishes the two cases — it reports which arms have no
`.sharding.lock` at all, i.e. which have nobody preparing them.

**Checked and sound:** `03_run_job.sh` uses `flock -n`, which is non-blocking and released by
the kernel when the process dies. No stale-lock deadlock is possible there. No other
blocking lock exists in the download or postprocess paths.

**Tests (`test_integration.py` §1.5, 11 checks):** a live lock is not stolen and the wait is
bounded; a stale lock is taken over and the new owner recorded; the waiter returns False when
the manifest appears; **`shard_arm` with a stale lock and no manifest completes instead of
hanging**; the lock is released on success; orphaned shard files are cleared.

## §2 `preflight.py` said 12 prompts — FIXED, and it exposed an error of mine

**Disposition: fixed, and a round-5 statement corrected.**

It is **13**: nine grounded jobs with one template each, plus the styled pass with four. The
check itself always iterated correctly.

Grepping for siblings turned up two more, one of them mine:

- `preflight.py:12` still said "all six repos resolvable" — stale since round 4, when six flat
  repos became one gated repo with five arm folders.
- `preflight.py:279` printed `"{n_texts} distinct prompt texts checked"`, and
  `DESIGN_DELTA` §2 repeated it as "6 distinct texts across 10 jobs". Both wrong: `n_texts`
  counts `(job, template)` **pairs**, which is 13. There are only **6 distinct texts** — the
  four grounded arms share `p1`, all five share `p2`, plus four wrap styles. I conflated the
  two counts in round 5, which is exactly how the header comment drifted in the first place.

Preflight now prints both numbers with their meanings, and `DESIGN_DELTA` states the
distinction.

## §3 The Blackwell rationale argued the wrong way round — REWRITTEN

**Disposition: rewritten in `configs/data.yaml` and `GUIDE_FOR_TIANJIAN.md` §2.3. Enforcement
unchanged.**

The report is correct and the error was mine. The old comment cited FlashAttention v3 and the
H100 origin of the source data, then concluded "therefore Blackwell" — which reads as though
Blackwell keeps this run close to the source. **H100 and H200 are both `sm_90` and share that
path, so H200 is the architecture closest to how the source data was produced.** Excluding it
moves this run further from the source's numerics, not nearer.

Both places now say what is true:

1. **One architecture family across all ten jobs** — non-negotiable, and sufficient on its
   own, because mixing confounds "which arm" with "which GPU".
2. **Blackwell chosen for throughput** — ~2–3× Hopper on this workload. A cost decision, not
   a fidelity one.
3. **The accepted cost** — a different attention backend from the H100 run, so this run is
   **not** numerically continuous with the 1.5B corpora, and never could have been across a
   GPU generation change (the vLLM version differs too). Cross-scale comparability rests on
   matching prompts, budgets and selection procedure, all of which are verified.
4. `gpu_cc` is recorded per shard either way.

The constraint, the code enforcement and the "do not widen this" warnings are unchanged.

## §4 The GPU ceiling — STATED, with the arithmetic

**Disposition: added to `GUIDE_FOR_TIANJIAN.md` §2.2 next to `num_gpus`, and to the
`shard_target_rows` comment in `data.yaml`. Pinned by a test.**

```
max_gpus = ceil(docs / shard_target_rows) / min_shards_per_gpu,  over the SMALLEST arm

  disagreement-aware   33,381,230 / 5,000 =  6,677 shards / 20 =   333   <- binding
  diversity-oriented   35,304,301 / 5,000 =  7,061 shards / 20 =   353
  quality-first        37,511,431 / 5,000 =  7,503 shards / 20 =   375
  wrap-inspired        63,226,477 / 5,000 = 12,646 shards / 20 =   632
  rewire-inspired     126,480,544 / 5,000 = 25,297 shards / 20 = 1,264
```

**~333 GPUs.** The guide's box says what to do in both cases: at or under, nothing; above,
stop and tell Wytro *before* running anything, because the only remedy is a smaller
`shard_target_rows`, which changes the fingerprint and means re-preparing all the data —
cheap before job 1, expensive after. The full per-arm table is shown in both places so a
different arm or a different `min_shards_per_gpu` can be re-derived rather than re-guessed.
`test_integration.py` §2.1 asserts the binding arm, the 333, and that both documents quote it.

## §5 Narrowing the distill limit — NOT POSSIBLE. The premise does not hold.

**Disposition: checked, short, negative. Reported rather than pursued.**

The condition in the brief is satisfied — round 3 *did* verify against production logs, not
the bake-off's: `test_trim_parity.py:145-146` reads `00_TMP/rewriting_monitor.md` and
`rewriting_monitor_distill.md`, which are the production monitor samples from `07_rewrite`
and `09_Distill` (18,190 + 18,000 rows).

**But the recovery method cannot be applied to them.** Those files contain only model
*outputs* — `harvest()` reads lines beginning `  - output: ` and nothing else — and they
contain **zero** occurrences of `im_start` or `im_end`. There is no templated input in them.
Round 3 verified the *trim rules* against production outputs; it never touched the prompt.

A search for the distill prompt text across the whole source tree returns three files: the
source's own local prompt file, and `00_Prompts/samples/nemotron_distill_{1.5B,7B}.md` —
which are bake-off samples. **No surviving production artifact carries a templated distill
input.** The round-5 limit stands as written and cannot be narrowed from what exists.

The balance of evidence is unchanged: three artifacts carry the 842-byte form against one
published file carrying 841, the difference is one token at the very end, and this run already
diverges from the 1.5B numerics through the architecture and vLLM changes recorded in §3.
Nothing changed.

## §6 What was and was not executed

**Ran, green:** `test_wrap_styles` 33/33; `test_integration` all checks including the 11 new
§1.5 lock checks and 4 new §2.1 ceiling checks; `test_trim_parity` 72,443 comparisons,
0 mismatches; `test_shuffle_parity` byte-identical; `compileall` over `src scripts tests`;
all three YAML configs parse; `bash -n` on every shell script; `02_download_data.py --help`.

**Not run:** `preflight.py` end to end and `verify_prompt_parity.py` — both need filled
cluster paths. No GPU work.

**Untouched, and verified untouched by diff:** `prompts/`, `engine.py`, `run_rewrite.py`,
`postprocess.py`, `shuffle.py`, `configs/vllm.yaml`. Engine args, sampling params, trim rules,
shuffle, claiming and reaping are all as round 5 left them. The ReWire budget fill and the
doc_id disjointness check remain unimplemented by decision.

## §7 Readiness list — final

**Wytro, in order:**

| # | item | blocks |
|---|---|---|
| 1 | **Grant Tianjian's HF account access to `wytro/Know-Your-Sources-7B`** (it is `gated: manual`) | everything — download is step 2 |
| 2 | Confirm Tianjian's GPU count is ≤ ~330, or decide a new `shard_target_rows` **before job 1** | everything, if exceeded |
| 3 | Fill `upload.repo_template` in `configs/data.yaml` and `HF_TOKEN_WRITE` in `.env` | upload only (step 6) |
| 4 | Optional: push the `configs:` block from `data_reports/DATASET_CARD_DRAFT.md` to the Hub card | nothing |
| 5 | Optional: fix `blab-jhu/KYS-Configs` shipping two different distill files | nothing here |

**Tianjian, in order, after Wytro's 1:**

| # | item | blocks |
|---|---|---|
| 1 | `bash scripts/00_setup_env.sh`, then paste the printed line into `env.activate_cmd` | everything |
| 2 | Fill the 11 blanks in `configs/cluster.yaml`; `cp .env.example .env` and fill `HF_TOKEN` | everything |
| 3 | `python scripts/check_placeholders.py` — must report zero unresolved | everything |
| 4 | `python scripts/preflight.py` — must pass; check 7 proves gated read access, check 9 prints the disk estimate (~0.34 TiB zstd) | everything |
| 5 | `bash scripts/run_all.sh` — model → data → calibrate → 10 jobs → postprocess → upload | — |

**After generation, before the corpus is trained on:**

| # | item | owner |
|---|---|---|
| 1 | Cut `rewire-inspired` to its 30B budget with the fastText fill — it ships ~99.5B **unfiltered** | Wytro |
| 2 | Trim every arm's output to 30B and assemble `20B core + 30B rewritten = 50B` per arm | Wytro |
| 3 | Read `06_calibrate.py`'s tok/doc cross-check on job 1: it settles the `r` transfer, and `quality-first`'s +19.6% headroom is the number at risk | Wytro |

---

# Addendum — 2026-08-26: round 7, scope reduction and fleet realism

The last round. Four items plus one string, and one theme running through three of them:
**the fleet is ~25 nodes × 4 GPUs, not one node with ~100 GPUs**, and several things in this
repo were written as if `num_gpus` were the fleet total. **Nothing on the generation path
was touched** — `prompts/`, `engine.py`, `run_rewrite.py`, `postprocess.py`, `shuffle.py`,
`wrap_styles.py`, `configs/vllm.yaml`, `03_run_job.sh` and the shard-claim machinery are
byte-identical to round 6, verified by diff.

## §1 Upload is out of scope — DISABLED, and the placeholder deleted

**Disposition: done. `upload.enabled: false` ships as the default, the `WYTRO` placeholder
is gone rather than replaced, and the flag/config pair is made non-contradictory by
construction. `05_upload_to_hf.py` stays in the repo, working and off.**

**A note on the two bullets, which could not both be taken literally.** "Delete the
placeholder" and "make `check_placeholders.py` exempt `upload.repo_template` when upload is
disabled" are in tension: that script is a pure `<<<CLASS: …>>>` text scanner, so once the
marker is deleted there is nothing left for an exemption to act on. Implemented as **delete
the marker, and replace the exemption with the rule that actually protects**:
`check_placeholders.py` now reads `upload.enabled` and `upload.repo_template` (stdlib regex
only — it still has no third-party imports, because it runs before the environment exists)
and fails **only** when `enabled: true` meets a blank template. Same rule stated positively,
and it also covers the "enabled with nothing to upload to" case at the earliest gate.

**The blocker was worse than the docs said, and that is worth recording.** `load_config`
runs `assert_no_placeholders(data, …)` on *every* entry point, and in `preflight.py` that
call sits **outside** the per-check `try`. So the unfilled `repo_template` did not block
"upload only (step 6)" as `DESIGN_DELTA.md:627` and this document's own round-6 readiness
list both claimed — it aborted preflight at check 2 with exit 2, no summary and no
remaining checks, and equally killed `01_download_model.py`, `02_download_data.py`,
`04_postprocess.py`, `06_calibrate.py` and `run_rewrite.py`. `--skip-upload` could not get
past it. A blank documented as affecting the last step in fact blocked the first. Both
claims are corrected.

**What was built:**

- `configs/data.yaml` — `enabled: false`, `repo_template: ""`. The comment says explicitly
  that the emptiness is deliberate and why a plausible-looking id would be worse.
- `config.py` — the single choke point. `enabled` must be a bool; `enabled: true` with a
  blank template is a named `stop()`; so is a template missing `{arm}`/`{prompt_id}`, which
  would have pushed all ten jobs to one repo, each overwriting the last.
- `run_all.sh` — exports `UPLOAD_ENABLED` from the validated loader. Upload runs iff
  `UPLOAD_ENABLED && !SKIP_UPLOAD`. **Both are veto-only: neither can switch upload on when
  the other says off**, which is the whole reconciliation — they cannot contradict each
  other because there is no combination in which they disagree about the outcome.
  `--skip-upload` is kept, accepted, and reported honestly ("disabled in configs/data.yaml"
  rather than claiming the flag did it).
- `05_upload_to_hf.py` — refuses with a named `stop()` when disabled, **including under
  `--dry-run`**: a dry run that cheerfully reports what it would push is the wrong answer
  when nothing should be pushed. It prints the real output path instead.
- `preflight.py` check 8 — skips the write-scope probe and namespace check when upload is
  disabled, removing a warning about a token nobody needs.
- `.env.example` — `HF_TOKEN_WRITE`'s marker removed, left blank. It mattered:
  `.env.example` is exempt from the scan but Tianjian's copied `.env` is not, so the marker
  would have failed his checker for a token with no use.

**Result: zero `WYTRO` blanks remain.** `check_placeholders.py` on a clean checkout now
reports only Tianjian's eleven.

## §2 `model_dir` — NODE-LOCAL, decided on the arithmetic

**Disposition: decided, documented, and made safe either way. Node-local, by more than an
order of magnitude.**

At 25 nodes × 4 GPUs:

| | shared `model_dir` | node-local |
|---|---|---|
| per job start | 100 processes × 14.2 GiB ≈ **1.4 TiB** off the shared filesystem | local disk; 4 processes share one page cache |
| over 10 jobs | ≈ **14 TiB** in ten bursts, each competing with `data_root` reads and `out_root` writes | 0 |
| one-time cost | 1 × 14.2 GiB from the Hub | 25 × 14.2 GiB ≈ 355 GB, once |
| disk | 20 GiB once | 20 GiB per node — already what `cluster.yaml` asks for |

The one-time re-download is cheap and the local disk was already provisioned. The scarce
resource is shared-filesystem bandwidth during a run that also streams input shards and
writes output, and sharing the model spends it ten times for nothing. `hf_cache` follows
`model_dir` for the same reason; its suggested default is now `${model_dir}/hf_cache`.

**The role sequence needed no new step.** `run_all.sh` already runs `01_download_model.py`
in *every* role including `--generate-only` (it sits outside the `DO_PREPARE` guard), so
each node fetches its own copy at the start of step 2. That was already correct for
node-local; what was missing was anyone saying so. The guide now does, and explains that
step 1's download covers only the node it ran on.

**`01_download_model.py` is now safe to run concurrently on many nodes**, which it was not.
The download is wrapped in the round-6 heartbeated directory lock (`acquire_dir_lock` +
`LockHeartbeat` + `break_lock` on both exit paths). Uncontended and free when `model_dir` is
node-local; correct if someone points it at shared storage anyway. This also closes a real
TOCTOU: `_verify_model` checks only that shard *filenames* exist, not their size or hash, so
a second node could observe another node's in-flight file and declare the copy complete.

## §3 Per-job postprocess parallelism — IMPLEMENTED. The reasoning in the report is right.

**Disposition: built. The single-node rule was correct per job and too broad across jobs,
exactly as reported.**

Verified against the code rather than assumed. Per job: output dir is
`out_root/shuffled/<arm>/<prompt_id>`, bucket dir is `tmp_root/_shuffle/<arm>/<prompt_id>`
and `tmp_root` is node-local, trim's in-place target is `out_root/raw/<arm>/<prompt_id>`,
the arm manifest is read-only. Nothing shared across jobs — **with one exception the report
did not name**: `manifests/postprocess_summary.json` was rewritten whole at the end of every
invocation, containing only the jobs *that* invocation ran. Two nodes, or even one
`--arm`-scoped run, erased everyone else's entries.

**What was built:**

- `data.py` — `try_dir_lock(lock, owner, stale_after_s, label)`, the non-blocking sibling of
  `acquire_dir_lock`, reusing `lock_age_s` / `break_lock` / `read_lock_owner` unchanged. The
  distinction matters: sharding an arm *must* happen, so a loser there should wait;
  postprocess is ten independent jobs, so a loser should go do a different one. A node that
  blocked for up to 24 h on a job someone else was already shuffling would sit idle while
  eight other jobs went untouched.
- `04_postprocess.py` — a sweep that takes any free job, holds its lock under
  `LockHeartbeat` for trim+shuffle, releases on **both** exits (`shuffle_job` raises on a
  row-count mismatch, and a lock left behind there would make every other node wait out the
  full stale timeout), and repeats until a whole pass takes nothing. Not a single pass — a
  node that finishes a job after two hours should pick up whatever freed up meanwhile. Not
  an unbounded wait either — 25 nodes blocking on 10 jobs would leave most of them asleep.
- Per-job summary files under `manifests/postprocess/`, merged into the rollup. **The merge
  is deliberately not `atomic_write_text`**: that helper writes through a fixed `<dest>.tmp`,
  which is right for the one-writer-per-destination paths it was built for but makes two
  nodes collide on a single temp name, and the loser's `os.replace` fails with ENOENT. This
  was found by the test, not by reading. Unique temp name per writer, and a failure there is
  swallowed with a note — losing a completed job's result because the convenience rollup
  raced would be absurd.
- `run_all.sh --postprocess-only` accepts and forwards `--arm` / `--prompt-id`, and refuses
  them without `--postprocess-only` (on a full run they would silently mean "generate all
  ten jobs, then postprocess one", which nobody wants). **That refusal was also found by a
  test** — the first implementation keyed off `DO_POST`, which is 1 on a normal run.

**Stated honestly in the guide: there are only 10 jobs, so this caps at 10 nodes** — about
10× rather than 25×, and the floor on wall clock is the single largest job,
`rewire-inspired/p1` at 126.5 M rows, which cannot be split further.

Demonstrated at 2, 4 and 10 concurrent processes: no job run twice, all ten covered, every
node contributing; at 10 nodes it is a clean 1:1 partition.

## §4 The GPU ceiling cannot fire — DOCUMENTED TRUTHFULLY, and a real check added

**Disposition: the report is exactly right. `min_shards_per_gpu × cfg.num_gpus` with
`num_gpus` per node is 6,677 vs 4 → 1,669:1, and the fleet-wide ratio was examined by
nothing. Documentation corrected in four places; an optional real check added.**

`configs/data.yaml`'s box, `data.py`'s comment and error message, and the guide's §2.2 row,
warning box and troubleshooting row all now say the same true thing: **the ceiling is a
constraint on the fleet total, the automatic check sees one node, and with
`compute.fleet_gpus` unset nobody but a human is checking.** The arithmetic is given in the
form Tianjian can apply to whatever count he ends up with:

```
shards(smallest arm) / min_shards_per_gpu  =  6,677 / 20  =  333 GPUs   (fleet total)
```

— and, in his terms, ≈83 nodes at 4 GPUs each.

**The cheap check, shipped:** optional `compute.fleet_gpus`, null by default. When set,
`shard_arm` additionally requires `n_shards >= min_shards_per_gpu × fleet_gpus` and logs the
fleet-wide ratio; when null it emits a NOTE saying the fleet ceiling was *not* checked.
`config.py` validates it as a positive int or null, and rejects a `fleet_gpus` smaller than
this node's `num_gpus` — a total can never be less than one node's share. No behaviour
change for anyone who leaves it unset, which is the single-node case where `num_gpus`
already *is* the fleet.

*Rejected:* deriving the count from distinct claim owners. It only reads after generation
starts, undercounts idle nodes, and needs a scan of every claim directory — for a number
Tianjian already knows.

## §5 `wrap-inspired p4` — FIXED

**Disposition: one word.** `docs/POSTPROCESSING.md:196`, `p4` → `p1`. The styled four-style
pass is p1 (`configs/data.yaml`); `p4` was four-pass-design naming. The other `p1..p4`
mentions in `DESIGN_DELTA.md` and this file are historical references to the abandoned
design and the source's own filenames, and are correct as they stand.

## §6 The `num_gpus` audit — a pattern, and it was five sites, not three

**Disposition: asked for during planning, and it found more than expected. All five
corrected.**

`06_calibrate.py` was four of the five, and its whole projection block derives from one
multiply:

| site | what it did |
|---|---|
| `data.py` ceiling guard | §4 above |
| `06_calibrate.py` `fleet    :` line | **printed the literal word "fleet" for one node's 4 GPUs** |
| `06_calibrate.py` per-job WALL column | every row ~25× too long |
| `06_calibrate.py` `PROJECTED TOTAL` | the day-zero projection ~25× too pessimistic |
| `06_calibrate.py` sensitivity block | inherits `total_h`, so all three rows too |

Calibration runs on one node and single-node throughput is the honest thing for it to
report, so the fix is labelling, not arithmetic: `this node:` instead of `fleet    :`,
`WALL/NODE`, `ON THIS NODE` on the total, and a short table showing what 1 / 5 / 10 / 25
nodes would mean. Nothing that could change a measured number was touched.

**Verified per-node and left alone:** `config.py` (`gpu_ids: auto`, validation),
`03_run_job.sh` (spawns N workers on this box), `preflight.py` device-count check, KV-cache
estimate and Blackwell remedy text, and the test fixture.

**Reported, not fixed:** `run_all.sh` exports `NGPU` and nothing in that file reads it
(`03_run_job.sh` computes its own). Dead, harmless, and not worth touching the config
bootstrap for.

## §7 `out_root` was gated below its real requirement

**Disposition: fixed, both halves, on instruction. Not cosmetic.**

Three statements disagreed: the guide said `out_root` needs ~0.34 TiB, `cluster.yaml` said
~0.7 TB, and `preflight.py` gated on the 0.34 figure. `cluster.yaml` was right — the trim
runs in place, so `raw/` is never consumed and `shuffled/` is a second full copy. Both are
present at rest.

The gate mattered more than the prose. It passed a disk that then fills part-way through a
shuffle, and `bucketed_shuffle` unlinks bucket files as it consumes them, so a disk-full
there loses that job's entire shuffle — days into the run, at the worst possible moment.
The guide's own "Disk full mid-run" row promised "recovery is clean by construction", which
is true for generation and **false for the shuffle**; that row is corrected too.
`preflight.py` now requires `2 * comp` in `out_root` and prints both figures.

## §8 What was and was not executed

**Ran, green:** `test_integration.py` — all pre-existing checks plus **28 new ones**
(§4 upload ×11, §5 locking ×15 including a two-node sweep partition, §6 per-node and
`fleet_gpus` enforcement ×13); `test_wrap_styles` 33/33; `compileall` over `src scripts
tests`; all three YAML configs parse; `bash -n` on every shell script and the sbatch;
`run_all.sh --help`; the `run_all.sh` argparse block round-tripped through the marker
extraction across 12 invocation forms; `check_placeholders.py` in all four upload states;
`05_upload_to_hf.py --dry-run` refusing cleanly. Separately, `04_postprocess.py`'s real
`sweep()` driven from 2, 4 and 10 concurrent OS processes, and the summary merge from 5
concurrent writers over 3 trials.

**The pre-existing round-6 ceiling test (`test_integration.py` §2.1) passes unmodified.**
It asserts both `docs/GUIDE_FOR_TIANJIAN.md` and `configs/data.yaml` still contain `330`
and `333` and that the 6,677 arithmetic matches the config — the §4 rewrite changes what
the number *means*, not the number.

**Not run:** `test_trim_parity` and `test_shuffle_parity` — both require `--source-root`,
the original pipeline tree, which is not present on this machine. `preflight.py` end to end
and `verify_prompt_parity.py` — both need filled cluster paths. No GPU work. Note the
round-6 entry claimed the two parity suites green; they were not re-run this round, and
nothing they cover was touched (`postprocess.py` and `shuffle.py` have zero diff).

**Untouched, and verified untouched by `git diff --stat`:** `prompts/`, `engine.py`,
`run_rewrite.py`, `postprocess.py`, `shuffle.py`, `wrap_styles.py`, `configs/vllm.yaml`,
`02_download_data.py`, `03_run_job.sh`, `00_setup_env.sh`, `verify_prompt_parity.py`.
`data.py`'s diff is exactly two hunks — the new `try_dir_lock`, and the ceiling guard inside
`shard_arm`; the shard-claim functions and `acquire_dir_lock` are unchanged. Engine args,
sampling params, prompts, trim rules, shuffle, claiming and reaping are as round 6 left
them. The ReWire budget fill and the doc_id disjointness check remain unimplemented by
decision.

## §9 Readiness list — final handoff form

**Wytro, in order:**

| # | item | blocks |
|---|---|---|
| 1 | **Grant Tianjian's HF account access to `wytro/Know-Your-Sources-7B`** (it is `gated: manual`) | everything — it is the input |
| 2 | Confirm Tianjian's **fleet-total** GPU count is ≤ ~330, or decide a new `shard_target_rows` **before job 1** | everything, if exceeded |
| 3 | Optional: push the `configs:` block from `data_reports/DATASET_CARD_DRAFT.md` to the Hub card | nothing |
| 4 | Optional: fix `blab-jhu/KYS-Configs` shipping two different distill files | nothing here |

*The output repo template and `HF_TOKEN_WRITE` are gone from this list. Upload is out of
scope; there is nothing left for Wytro to fill before Tianjian can start.*

**Tianjian, in order, after Wytro's 1:**

| # | item | blocks |
|---|---|---|
| 1 | `bash scripts/00_setup_env.sh`, then paste the printed line into `env.activate_cmd` | everything |
| 2 | Fill the 11 blanks in `configs/cluster.yaml`; `cp .env.example .env` and fill `HF_TOKEN` only | everything |
| 3 | **If on more than one node:** set `compute.fleet_gpus` to the fleet total, and put `model_dir`, `hf_cache` and `tmp_root` on node-local disk (GUIDE §3.5) | correctness of the GPU-ceiling check; shared-filesystem load |
| 4 | `python scripts/check_placeholders.py` — must report zero unresolved | everything |
| 5 | `python scripts/preflight.py` — must pass; check 7 proves gated read access, check 9 prints the disk estimate (~0.65 TiB in `out_root`) | everything |
| 6 | `bash scripts/run_all.sh` — model → data → calibrate → 10 jobs → postprocess. On several nodes, GUIDE §3.5 instead | — |

**The run ends there.** Finished data is at
`out_root/shuffled/<arm>/<prompt_id>/part_NNNNN.parquet` — 10 directories, ~1,184 zstd
parquet files, ~592 M rows, ≈0.3 TiB. Nothing is uploaded and there is no further step.
GUIDE §3.6.

**After generation, before the corpus is trained on:**

| # | item | owner |
|---|---|---|
| 1 | Cut `rewire-inspired` to its 30B budget with the fastText fill — it ships ~99.5B **unfiltered** | Wytro |
| 2 | Trim every arm's output to 30B and assemble `20B core + 30B rewritten = 50B` per arm | Wytro |
| 3 | Read `06_calibrate.py`'s tok/doc cross-check on job 1: it settles the `r` transfer, and `quality-first`'s +19.6% headroom is the number at risk | Wytro |
| 4 | Arrange delivery of the finished data — out of this pipeline's scope by design | Wytro |

---

# Addendum — 2026-08-26 (later): round 8, the document identifier — VERIFICATION ONLY

**Nothing was changed.** No code, no schema, no config. `configs/data.yaml`'s
`output.keys` is untouched and still the 11 keys round 7 left. This round establishes facts
and hands the decision back.

## §1 What identifier columns exist

**First, a count correction.** The master corpus has **32** columns, not 34. The uploaded
dataset has 34 = the 32 master columns + `is_core` + `text`, which is what
`configs/data.yaml:153` already says. Verified by reading the on-disk schema of
`6_merged_clean/part_00000.parquet` and against `config_600m.py:169-182`.

Of those 32, the ones that could conceivably identify a document:

| column | dtype | example (real, row 0 of shard 0) |
|---|---|---|
| `record_id` | `string` | `<urn:uuid:63e688d6-238c-43a2-89d2-dcbc9e4f71dc>` |
| `url` | `string` | `http://cccscholarships.blogspot.com/2008/06/scholarships-can-help-you-through.html` |
| `payload_digest` | `string` | `sha1:RWBEZ7DBLT5ERTA6VNOJ3PWCT7NGPXGI` |
| `crawl` | `string` | `CC-MAIN-2018-09` |
| `shard` | `string` | `global-shard_01_of_10/local-shard_8_of_10/shard_00001392_processed.jsonl.zstd` |
| `line_no` | `int64` | `15741` |
| `warc_date` | `timestamp[us]` | `2018-02-21 19:21:15` |
| `warc_ip` | `string` | `172.217.7.225` |
| `metadata` | `string` | JSON blob: `WARC-Block-Digest`, `WARC-Concurrent-To`, `WARC-Date`, `WARC-IP-Address`, … |
| `doc_id` / `orig_doc_id` | `int64` | `0` / `0` — the surrogate, identical to each other |
| `shard_id` / `row_in_shard` | `int16` / `int32` | `0` / `0` — position in the master |

`doc_id` is confirmed surrogate: `materialize_master.py:114`,
`doc_id = np.arange(lo, lo + n, dtype=np.int64)`.

## §2 Which are unique — measured, not reasoned

Measured **exactly over all 600,000,000 rows** for `record_id`, by packing each UUID to its
16 raw bytes (vectorised, validated against Python's `uuid` parser) and taking a true
distinct count. Not a sample.

```
record_id   EXACT distinct: 599,603,031 / 600,000,000     duplicate rows: 396,969  (0.0662%)
```

Per-shard, across 12 shards spread over the corpus (7.3M rows):

| column | distinct, typical shard | reading |
|---|---:|---|
| `record_id` | 100.0000% in 10 of 12 shards | near-unique; see the exception below |
| `payload_digest` | 99.998% | content hash — collides on duplicate text by design |
| `url` | 99.99% | not an identifier; the same page is recrawled |
| `crawl` | **0.0179%** — 89 distinct in 497,055 rows | **a snapshot label** |

**The `CC-MAIN-2026-...` recollection was `crawl`, and the concern about it was right.** It
is a Common Crawl snapshot label shared by billions of documents — 89 distinct values across
a half-million-row shard. Switching the join key to it would be a severe regression, and it
is not a candidate.

**`record_id` is near-unique but is NOT a primary key**, and this is the finding that shapes
the recommendation. 396,969 rows share a `record_id` with another row. Two of the twelve
sampled shards showed it (440 at 99.9319%, 990 at 99.9966%). Inspecting shard 440: 1,179
duplicated ids, **all at multiplicity exactly 2**, and each pair is the *same WARC record*
drawn twice from different DCLM source shards — identical `url`, `payload_digest`, `crawl`
and `text_len`, differing only in `shard` / `line_no` / `doc_id`:

```
<urn:uuid:fe5865bd-c8ac-4a45-bde2-28d0cef5a7b4>  appears 2x
  doc_id=236776167  shard_00000075  line 6508  url=http://betabeat.com/tag/period-tracker/
  doc_id=237006013  shard_00000267  line 5492  url=http://betabeat.com/tag/period-tracker/
```

These are genuine duplicate documents in the upstream sample, **not id collisions between
different documents** — which is the benign reading, but it still means `record_id` cannot
replace `doc_id`. `doc_id` is a true primary key by construction; `record_id` is not.

## §3 Is it already uploaded? YES — no re-upload needed

**This is the cheap case.** Neither stage of the upload path projects columns:

- `materialize_blocks.py:87` — `tbl = pq.read_table(C.master_path(i))`, full read; then only a
  row filter and `append_column('is_core', ...)`. Its docstring: *"the master has 32 columns
  and no `text` column … so 'all columns except text' is simply all master columns; nothing
  has to be dropped."*
- `upload_blocks.py:214` — `tbl = pq.read_table(src_part)`, full read; `:222` filters **rows**
  only; `:231` — `out_schema = pa.schema(list(ids.schema) + [pa.field('text', pa.large_string())])`.

There is no column list anywhere in either file. `record_id` is not merely present, it is
**mandatory**: `upload_blocks.py:215-220` aborts if it is missing, because every uploaded row
is positionally cross-checked against the source on it.

Confirmed against the actual upload run logs, first-hand:

```
verify restore OK: rows=126,480,544 nulls=0 empty=0 doc_id set identical,
                   record_id cross-checked on all 126,480,544 rows
verify upload OK:  143 files present, all sizes match (202.4G)
```

— and the same for all six directories, with row counts matching `data.yaml`'s declared
`docs` exactly (37,511,431 / 35,304,301 / 33,381,230 / 63,226,477 / 126,480,544 / 17,909,083).

`configs/data.yaml`'s `data_files_template: "{subdir}/data/*.parquet"` points at those
full-column tables. It is **not** the 5-column `ids/` mirror (`doc_id, orig_doc_id, shard_id,
row_in_shard, is_core`), which carries no `record_id` and which an earlier abandoned upload
script used (`_archive_upload_ids_version.py.bak`).

**Caveat on the evidence.** There is no HF token on this machine and the repo is gated, so
this rests on the upload code plus the upload run logs, not on reading the Hub schema
directly. The logs are strong — a per-row `record_id` cross-check is only possible on parts
that contain the column, and those are the parts that were pushed — but a one-line
confirmation against the Hub is worth doing before committing to the change. Note also that
the local `<block>/data/` directories no longer exist; the Hub copy is the only place the
full-column tables now live.

## §4 What it costs to carry

Measured on 497,055 real `record_id` values, not estimated:

| representation | B/row | over 592M rows | share of the ~371 GB deliverable |
|---|---:|---:|---:|
| JSONL, uncompressed (`out_root/raw/`) | 62.0 | 36.7 GB | — |
| JSONL + zstd (`out_root/raw/`) | 21.1 | 12.5 GB | 3.4% |
| **parquet + zstd, string as-is (`shuffled/`)** | **19.4** | **11.5 GB** | **3.1%** |
| parquet + zstd, `binary(16)` packed UUID | 16.3 | 9.7 GB | 2.6% |

Both copies together: ~24 GB on a ~742 GB `out_root` — **3.2%**. UUIDs are ~16 bytes of real
entropy, so zstd cannot do much; the floor is the packed form at 16.3 B/row, and stripping
the `<urn:uuid:` wrapper saves nothing (19.8 vs 19.4 — the wrapper is a constant that
compresses away, and removing it costs the offsets). Recommend keeping the string verbatim:
3.1% for a value that stays greppable and matches the source byte for byte.

## §5 Recommendation

**Agreed — carry it, additively, and decide before job 1.** Keep `doc_id` exactly as it is:
it is folded into the manifest fingerprint, it is a true primary key, it is cheap, and it
works today. Add `record_id` as a 12th output column.

Three reasons the case is stronger than "nice to have":

1. **The failure mode being insured against is silent.** A regenerated master shifts every
   `doc_id`, and nothing downstream would raise — joins would resolve to the wrong documents
   and produce plausible results. That is the class of bug worth 3.1% of disk.
2. **It is nearly free, and free *now*.** No re-upload, no re-selection, ~3% of the
   deliverable, and the pipeline already reads the parquet that contains it. After job 1 it
   means re-preparing everything.
3. **It costs nothing to ignore.** A downstream consumer that does not want it can drop the
   column.

**One correction to the framing, and it matters.** The proposal was described as adding a
"natural per-document identifier … so the join survives independently of the master".
`record_id` does not quite deliver that on its own: at 0.0662% duplication it identifies a
*web record*, not a *row*, so a join on `record_id` alone fans out 2:1 on ~397k documents.
What it actually buys is the ability to **detect and repair** a `doc_id` shift — re-derive
the mapping by joining `record_id` (plus `payload_digest` or `url` to break the rare ties)
against a rebuilt master. That is still the property worth having, and it is worth stating
accurately so nobody later treats `record_id` as a unique key and is surprised.

**If a strictly unique natural key is wanted**, the pair `(record_id, payload_digest)` or the
triple with `shard`/`line_no` would do it — but `shard`/`line_no` are positional in DCLM and
inherit the same regeneration fragility as `doc_id`, so they add little. My advice is
`record_id` alone, documented as near-unique with the measured rate.

**Not recommended:** `crawl` (a snapshot label — see §2), `url` alone, or replacing `doc_id`
with anything.

## §6 What was and was not executed

**Ran:** exact 128-bit distinct count over all 1,103 master shards / 600,000,000 rows
(~8 min read + sort); per-shard distinct counts for `record_id`, `payload_digest`, `url`,
`crawl` on 12 shards spread over the corpus; duplicate-group inspection on shard 440;
compression measurements on 497,055 real `record_id` values in four representations; read of
`materialize_master.py`, `config_600m.py`, `materialize_blocks.py`, `upload_blocks.py`,
`select_600m.py`, `_archive_upload_ids_version.py.bak`; the six upload verification logs.

**Not run:** any read of the Hub — no token on this machine and the repo is gated (see §3).

**Changed: nothing.** No code, no schema, no config, no test. `git diff` covers only
`docs/DESIGN_DELTA.md` (open question 7) and this file.

---

# Addendum — 2026-08-26 (final): round 9, record_id implemented

Wytro confirmed `record_id` on the Hub from the Data Studio view — the column is there with
`<urn:uuid:...>` values matching `WARC-Record-ID` in `metadata` — and approved the round-8
recommendation. This round implements it. **Additive: `doc_id` is untouched.**

**Nothing on the generation path changed in substance** — `prompts/`, `engine.py`,
`configs/vllm.yaml`, `wrap_styles.py`, `postprocess.py`'s trim rules and `shuffle.py`'s
shuffle internals are byte-identical to round 7. `run_rewrite.py` gains one key in the
output row; that is the whole of its diff.

## §1 Two identifiers rejected, and why they are the same mistake

**`WARC-Warcinfo-ID` is out.** It identifies a WARC *file*, so every record written into
that file shares it — the same class of error as `crawl`, which labels a crawl snapshot
shared by billions of documents. Neither is a document identifier. It is also unnecessary:
the `record_id` column already carries exactly the `WARC-Record-ID` value, so **nothing is
parsed out of the `metadata` blob**. `shard_arm` reads the column directly, and the error
it raises when the column is absent says so explicitly, because parsing `metadata` is the
obvious wrong move for whoever hits it next.

**`record_id` does not replace `doc_id`.** The round-8 measurement settles it: 396,969 of
600,000,000 rows (0.0662%) share a `record_id`, so a join on it fans out 2:1 on ~397k
documents — silently inflating row counts rather than erroring, which is worse than
failing. `doc_id` is unique by construction. Implemented exactly as recommended.

## §2 What was built

| layer | change |
|---|---|
| `configs/data.yaml` | `record_id` added to `output.keys` (12 keys), positioned next to `doc_id`; a long comment on its role, its measured non-uniqueness, and the two rejected candidates |
| `src/rewrite/data.py` — `shard_arm` | reads the `record_id` column from the uploaded parquet, buffers it, writes it into each input shard; **hard `stop()`** if `output.keys` declares it and the dataset lacks it |
| `src/rewrite/data.py` — `compute_fingerprint` | `output.keys` folded in — see §3 |
| `src/rewrite/data.py` — `jsonl_to_table` | `record_id` → `pa.large_string()` |
| `src/rewrite/run_rewrite.py` | carries it from the input shard into every output JSONL row |
| `src/rewrite/postprocess.py` | **no change needed** — `trim_shard` mutates row dicts in place, so it passes unknown keys through unchanged. Verified by test, not assumed |
| `src/rewrite/shuffle.py` | **no change needed** — it takes its key list from `cfg.data["output"]["keys"]`. Verified by test |
| `scripts/preflight.py` | the smoke-test row carries it, so the end-to-end check exercises the real schema |
| `scripts/05_upload_to_hf.py` | dataset-card column table documents it, including the not-a-key warning |

The four layers the brief asked to agree — config, JSONL rows, Arrow schema, trim/shuffle —
now do, and the test asserts the `(doc_id, record_id)` pairing survives *into the shuffled
parquet*, which is where a newly added key would otherwise go missing silently.

## §3 The fingerprint decision: folded in, and why

**`output.keys` is now part of the manifest fingerprint.**

The argument against is real and worth stating: the fingerprint's documented job is to catch
anything that **renumbers rows**, and adding `record_id` renumbers nothing — `doc_id` and the
text are untouched, and the generated text for a given shard would be byte-identical either
way. On that reading it does not belong.

It belongs on the broader reading, which is what a `.done` marker actually asserts: *shard N's
output is complete and correct*. After a schema change, the rows behind an old marker are
neither. Without the interlock, adding a key mid-run would leave every already-generated shard
matching its marker, silently skipped, and the job would report DONE with some rows carrying
`record_id` and some not. That surfaces as nulls in the shuffled parquet — after the GPU time
is spent, and after row-conservation checks have passed, because the row *count* is still
right. With the interlock it fails loudly at data prep instead.

Folding in the whole key list rather than a `record_id`-specific flag also covers the next
schema change, whatever it is. Reordering counts too, deliberately: the shuffled parquet's
column order follows `output.keys`, so a reorder changes the artifact.

**It cost nothing to add today** — this lands before job 1, with no data generated. After job 1
the same change would mean re-preparing the data and regenerating every shard, which is
precisely the asymmetry `GUIDE §6` item 12b now warns about.

## §4 Disk — every quoted estimate revised

`bytes_per_row_overhead` 235 → **297**. The +62 is measured, not assumed: 497,055 real
`record_id` values serialised as a JSON field average exactly 62.0 B/row uncompressed
(`"record_id":"<urn:uuid:...>",` = 12 + 49 + 1).

| figure | before | after |
|---|---|---|
| raw JSONL, one copy | 1.13 TiB | **1.16 TiB** |
| zstd, one copy | 0.338 TiB | **0.348 TiB** |
| `out_root` (raw/ + shuffled/) | 0.68 TiB | **0.70 TiB** |
| `cluster.yaml` header figure | ~2.4 TB / ~0.7 TB | **~2.6 TB / ~0.77 TB** |

Updated in `configs/data.yaml`, `configs/cluster.yaml`, and `GUIDE` §1, §2.1, §3.6 and §4.
`preflight.py` check 9 derives from `bytes_per_row_overhead`, so it follows automatically and
still gates `out_root` on two copies. **+3.0%**, against the 3.1% projected in round 8.

Compressed, the column is far cheaper than 62 B: 21.1 B/row as JSONL+zstd, 19.4 B/row as
parquet+zstd, because the key name and the `<urn:uuid:` wrapper are constants and only the
32 hex digits carry entropy — about 16 bytes of it.

## §5 What was and was not executed

**Ran, green:** `test_integration.py` — full suite, 0 failed, including **17 new checks**:
the 12-key schema now driven off `output.keys` rather than a hardcoded list, `record_id`
non-empty and byte-identical to the input shard for the same `doc_id`, the shuffled parquet's
schema matching `output.keys` in order with `record_id` as `large_string`, the
`(doc_id, record_id)` pairing intact after trim + shuffle, the fingerprint changing on key
add/remove/reorder and stable otherwise, and the missing-column guard rejecting rather than
blanking. `test_wrap_styles` 33/33. `compileall` over `src scripts tests`; all three YAML
configs parse; `bash -n` on every shell script.

**The test fixture now reproduces the real defect**: its synthetic `record_id` values repeat
every 500th row, so anything that assumes uniqueness fails in the test rather than in
production. A check asserts the duplication is present.

**Not run:** `test_trim_parity` / `test_shuffle_parity` (need `--source-root`); `preflight.py`
end to end and `verify_prompt_parity.py` (need filled cluster paths); no GPU work. Nothing in
the trim or shuffle *logic* changed — both files' diffs are zero for `postprocess.py` and
`shuffle.py`.

## §6 Readiness list — unchanged except the disk figure

Round 7's §9 list stands. The only edit: preflight check 9 now prints **~0.70 TiB** for
`out_root` rather than ~0.65 TiB, and the finished data at
`out_root/shuffled/<arm>/<prompt_id>/part_NNNNN.parquet` carries **12 columns**, not 11.

One thing for whoever consumes the corpus, worth repeating because it is the single most
likely misuse: **join on `doc_id`.** `record_id` is there to rebuild `doc_id` if the upstream
master is ever regenerated — not to join on. It is not unique.
