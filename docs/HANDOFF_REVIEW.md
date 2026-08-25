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
