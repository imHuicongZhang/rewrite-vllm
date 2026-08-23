# Guide for Tianjian

Primary reader: **Tianjian's Claude Code agent**. Execute this top to bottom; nothing here
should need inferring. Secondary reader: Tianjian, skimming.

---

## 1. Thirty-second summary

This repo rewrites five web-text corpora with **Qwen2.5-7B-Instruct** under **vLLM**, then
trims and shuffles the results. A sixth corpus, `quality-base`, is a **control**: it is
downloaded and verified so the token accounting is complete, and is **never rewritten**.

**12 rewrite jobs.** One job = one (arm, prompt) pair:

| arm | prompts | jobs |
|---|---|---|
| `quality-base` | 0 | 0 — control, download + verify only |
| `quality-first` | 2 | 2 |
| `diversity-oriented` | 2 | 2 |
| `disagreement-aware` | 2 | 2 |
| `wrap-inspired` | 4 | 4 |
| `rewire-inspired` | 2 | 2 |
| | | **12** |

**The thing that is easiest to get wrong:** each prompt rewrites the **entire** dataset
for its arm. Prompts are not split across documents, not sampled per document, not
round-robined. `wrap-inspired` with 4 prompts means **4 complete passes over the whole
corpus**, producing 4 output shard sets. The code asserts this (output rows per
(arm, prompt) must equal the arm's input rows) and fails the job if it is ever violated.

**Scale.** ~500B input tokens; roughly 100B output tokens per arm, ~500–550B total. Order
of **2.4 TB** of JSONL uncompressed, ~0.7 TB with the default zstd. Wall-clock depends
entirely on your GPUs — `preflight.py` prints a per-GPU KV-cache estimate and every job
logs a live tok/s and ETA. Assume days-to-weeks, and assume **it will be interrupted many
times**. That is fine: everything resumes at shard granularity. Re-running the same
command is always the correct recovery action.

**What you must supply:** eleven values in `configs/cluster.yaml` and one token in `.env`.
That is all. Section 2 lists every one.

---

## 2. Fill these blanks

Run this at any time to see exactly what is left:

```bash
python3 scripts/check_placeholders.py
```

It prints `file:line` for every remaining blank, grouped into `TIANJIAN` (yours) and
`WYTRO` (Wytro's — those should already be filled when you receive this; if any remain,
stop and tell Wytro rather than guessing). **Both classes are hard errors.** Nothing runs
until it exits 0.

### 2.1 `configs/cluster.yaml` — `paths`

Replace the whole placeholder including its surrounding quotes. `${other_key}` references
are expanded automatically, so `"${repo_root}/logs"` is a valid value.

| key | what it is | how to find it |
|---|---|---|
| `repo_root` | absolute path to this clone | `cd` into the repo, run `pwd` |
| `model_dir` | where model weights land, ~20 GB | any disk with room: `df -h <dir>` |
| `data_root` | downloaded + re-sharded inputs | needs roughly 2× the raw dataset size; `preflight.py` prints the real number |
| `out_root` | rewritten output — **the big one** | ~0.7 TB with zstd, ~2.4 TB without |
| `tmp_root` | shuffle scratch | must be **fast local** disk, not NFS/Lustre; ~1.2× the largest single job's output |
| `log_root` | per-worker logs | small; `"${repo_root}/logs"` is fine |
| `hf_cache` | `HF_HOME` | **not** your home directory if it has a quota — downloads will fill it |

### 2.2 `configs/cluster.yaml` — `compute`

| key | how to determine it |
|---|---|
| `num_gpus` | `nvidia-smi --list-gpus \| wc -l`. Use every GPU you are entitled to — see §4 if the node is shared. |
| `gpu_ids` | leave as `auto` (means `0..num_gpus-1`). Only set an explicit list like `[0,1,2,3]` if you must avoid specific cards. Its length must equal `num_gpus`. |
| `cpu_workers` | `nproc`. Used only by the CPU trim stage; no GPU involved. |
| `shuffle_mem_bytes` | `free -b \| awk '/^Mem:/{print $2}'`, then take ~70%. Internally capped at 256 GiB regardless of what you put. |

To see VRAM per GPU (you need ≥ 24 GiB per card):

```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
```

### 2.3 `configs/cluster.yaml` — `env`

| key | how to determine it |
|---|---|
| `activate_cmd` | **`scripts/00_setup_env.sh` prints the exact line to paste** when it finishes. Do not invent it. |
| `extra_preamble` | anything else needed before `python` runs (module loads, proxy vars). Leave `""` if nothing. |

`scheduler.kind` stays `bash`. Only change it if you have read §4 and
`scripts/optional/slurm_job.sbatch`.

### 2.4 `.env`

```bash
cp .env.example .env
```

| variable | what |
|---|---|
| `HF_TOKEN` | your HuggingFace token with **read** access to the six input datasets. Create at <https://huggingface.co/settings/tokens>. |
| `HF_TOKEN_WRITE` | a token with **write** scope on the output org (Wytro's blank; may be the same token). |

`.env` is gitignored. **Never commit it, never paste a token into a config file or a
message.**

---

## 3. The happy path

Five commands.

```bash
# 1. Build the environment with the exact pinned stack. Prints your activate_cmd.
bash scripts/00_setup_env.sh

# 2. Fill configs/cluster.yaml (§2) and cp .env.example .env, then:
python3 scripts/check_placeholders.py     # must exit 0

# 3. Everything that must be true before a GPU-hour is spent.
#    Ends with an 8-document end-to-end smoke test; READ ITS BEFORE/AFTER OUTPUT.
python scripts/preflight.py

# 4. The whole run: model -> data -> 12 jobs sequentially -> postprocess -> upload.
bash scripts/run_all.sh

# 5. Check progress at any time, from any shell.
bash scripts/run_all.sh --status
```

If `run_all.sh` dies for any reason — power, preemption, full disk, a bad node — run it
again. Finished shards are skipped via their `.done` sidecars, and finished jobs are
skipped without even loading a model.

Useful variants:

```bash
bash scripts/03_run_job.sh wrap-inspired p3   # one job on its own
bash scripts/run_all.sh --skip-upload         # generate + postprocess, upload later
bash scripts/run_all.sh --from-job=5          # resume at job 5 in the status table
python scripts/04_postprocess.py --dry-run --sample 100000   # strip rates, no writes
```

---

## 4. Stop and ask Tianjian

Agent: **do not decide these yourself.** Stop, ask, and wait.

1. **Where the data goes.** `out_root` needs up to ~2.4 TB and `data_root` a comparable
   amount. Do not pick a filesystem, quota, or scratch area on his behalf, and do not
   "free up space" by deleting anything.
2. **How many GPUs to take.** `num_gpus` decides how much of the machine this consumes,
   for days. If `nvidia-smi` shows other processes, or the node is shared, ask before
   claiming all of them.
3. **Whether other users share the node.** A job holds every GPU it is given at
   `gpu_memory_utilization=0.85` for its entire duration. If someone else needs a card,
   that has to be agreed up front — it cannot be renegotiated mid-job without losing work.
4. **Any environment mismatch in setup.** If `00_setup_env.sh` stops on a driver/CUDA
   mismatch, or `preflight.py` reports a version that differs from the pins, **stop**.
   Do not install a different vLLM, torch, or transformers to get past it. See §6.
5. **Anything that would change a config value outside `cluster.yaml`.** If a fix seems to
   require editing `vllm.yaml`, `data.yaml`, or a prompt file, that is a sign something
   else is wrong. Ask.
6. **A row-count mismatch that does not clear on a re-run** (§5). That means real output
   is missing or wrong, not a transient failure.

---

## 5. Troubleshooting

| symptom | what it means | what to do, in order |
|---|---|---|
| **CUDA OOM at engine startup** | The model plus its KV cache does not fit. | 1. Confirm no other process holds the GPU: `nvidia-smi`. Kill strays. 2. Confirm `num_gpus` matches reality; a stale value can put two engines on one card. 3. Confirm the `.joblock` is doing its job — never run two jobs at once. 4. **Do not lower `gpu_memory_utilization`.** It is `0.85` for source parity; changing it makes your output non-comparable. If the model genuinely cannot fit on your cards, **ask Wytro** — the fix is a decision, not a knob. |
| **CUDA OOM mid-generation** | A very long prompt batch. | vLLM's chunked prefill (`max_num_batched_tokens=16384`) normally prevents this. Re-run the job; the shard is redone cleanly. If one specific shard fails repeatedly, capture the log and ask — do not skip it, a skipped shard breaks row conservation. |
| **`tensor_parallel_size must be 1`** | Someone edited `vllm.yaml`. | Revert it. Parallelism here is N independent single-GPU processes, exactly as the original pipeline did it. TP > 1 is a different computation. |
| **DP/TP does not divide the GPU count** | Not possible here by construction — TP is always 1 and DP is exactly `num_gpus`. | If you see a mismatch, it is `gpu_ids` disagreeing with `num_gpus`. `03_run_job.sh` refuses to start in that case; make the two agree. |
| **NCCL init failure** | Unexpected: with `tensor_parallel_size=1` there is no cross-GPU communication. It is a socket-bootstrap problem. | Try `export NCCL_P2P_DISABLE=1`, and if your node has several interfaces, pin one: `export NCCL_SOCKET_IFNAME=<iface>` / `GLOO_SOCKET_IFNAME=<iface>` (`ip -o -4 addr show` to list them). The original cluster needed exactly this. Put the working line in `env.extra_preamble`. |
| **HF 401/403 on download** | Token missing, wrong, or lacking access to a gated dataset. | `python scripts/preflight.py --only 7,8`. Check `HF_TOKEN` in `.env`; ask Wytro to grant access to the repo it names. |
| **HF 429 / 5xx on upload** | Rate limited. | Already handled: `05_upload_to_hf.py` retries 6 times with exponential backoff and jitter, and uploads are content-addressed so a re-run skips what already landed. If it still fails, wait and re-run — nothing is lost. |
| **Disk full mid-run** | Out of space in `out_root` or `tmp_root`. | Free space, then re-run the same command. Recovery is clean by construction: each shard is written to `.tmp` and renamed, and its `.done` sidecar is written only **after** the rename. A crash can leave a complete shard with no marker (it gets redone) but never a half-written shard that looks finished. Nothing is corrupted. |
| **A job finishes with row count ≠ input row count** | Serious. Every prompt must rewrite the entire arm. | `python -m rewrite.run_rewrite --arm A --prompt-id pN --verify --deep-verify` prints exactly which shards disagree. Delete those shards' `.done` sidecars and re-run the job. If it does not clear, **stop and ask** (§4.6) — do not upload. |
| **`sidecar fingerprint does not match the manifest`** | The input was re-sharded after some output was generated, so `doc_id`s were renumbered. | Do not delete anything yet. This normally means `sharding.*` in `data.yaml` was changed, or `data_root` was rebuilt. Ask Wytro. The safe fix is to restore the original sharding; the expensive fix is regenerating that job. |
| **`overhead is N, expected M`** | The prompt file, chat template, or tokenizer differs from the original pipeline. | **Do not edit `expected_overhead` to make it pass.** That integer is the proof the prompt is byte-correct. Check that no prompt file was modified (`git status`), and that the model revision resolved to the same chat template. Then ask. |
| **Model download keeps restarting** | Interrupted transfers. | It is idempotent — re-run `python scripts/01_download_model.py`. Once complete it verifies locally and never touches the network again (`--offline` forces that). |
| **Everything is slow** | Usually `tmp_root` on a network filesystem, or `cpu_workers` set too low for the trim stage. | Both are safe to change — they affect speed only, never output. |

---

## 6. Do not do this

A hard list. Each item exists because doing it silently invalidates the experiment.

1. **Do not edit any file in `prompts/`.** All twelve are byte-exact copies of the
   originals. `preflight.py` verifies each one's templated token count and will refuse to
   run if a single character changes.
2. **Do not change sampling params or engine args** in `configs/vllm.yaml`. Greedy
   decoding, `top_p=1.0`, `max_tokens=4096`, `max_model_len=32768`,
   `gpu_memory_utilization=0.85`, `dtype=bfloat16`, `tensor_parallel_size=1`. The loader
   rejects any engine key the original never passed — that check is a feature, not an
   obstacle.
3. **Do not modify the trim logic** in `src/rewrite/postprocess.py`. Every rule, constant,
   and tuple is verbatim from the original and verified against it on 36,190 real
   outputs. It looks quirky in places; that is the point.
4. **Do not shuffle across arms**, or across prompts within an arm. Shuffling is scoped to
   one (arm, prompt) and the code enforces it.
5. **Do not skip `quality-base`.** It is never rewritten, but it must be downloaded and
   verified so the token accounting is complete.
6. **Do not "fix" a version mismatch by installing a different vLLM/torch/transformers.**
   A different build produces different text at `temperature=0`, and afterwards there is
   no way to tell which rows came from which build. Stop and ask instead.
7. **Do not parallelise jobs to save time.** Each job already uses every GPU. Two at once
   means two engines each trying to reserve 85% of the same card. The `flock` in
   `03_run_job.sh` prevents it — do not remove it.
8. **Do not delete output to free space** without asking. A missing shard breaks row
   conservation, and at this scale regenerating one is hours of GPU time.
9. **Do not commit `.env`, a token, model weights, or data.** `.gitignore` covers all of
   them; keep it that way.
10. **Do not lower `gpu_memory_utilization` to dodge an OOM.** See §5, row 1.
