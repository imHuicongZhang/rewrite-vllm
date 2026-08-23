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
