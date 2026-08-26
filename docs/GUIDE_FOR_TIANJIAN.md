# Guide for Tianjian

Primary reader: **Tianjian's Claude Code agent**. Execute this top to bottom; nothing here
should need inferring. Secondary reader: Tianjian, skimming.

---

## 1. Thirty-second summary

This repo rewrites five web-text corpora with **Qwen2.5-7B-Instruct** under **vLLM**, then
trims and shuffles the results.

**10 rewrite jobs.** One job = one (arm, prompt) pair. Every arm gets two passes:

| arm | source tokens | documents | pass 1 | pass 2 | jobs |
|---|---:|---:|---|---|---|
| `quality-first` | 60.0 B | 37,511,431 | wiki | distill | 2 |
| `diversity-oriented` | 60.0 B | 35,304,301 | wiki | distill | 2 |
| `disagreement-aware` | 60.0 B | 33,381,230 | wiki | distill | 2 |
| `wrap-inspired` | 60.0 B | 63,226,477 | **styled** | distill | 2 |
| `rewire-inspired` | 120.0 B | 126,480,544 | wiki | distill | 2 |
| | **360.0 B** | **295,903,983** | | | **10** |

**The thing that is easiest to get wrong:** each prompt covers **every** document of its
arm, exactly once. Prompts are not split across documents, not sampled, not round-robined.
The code asserts this — output rows per (arm, prompt) must equal the arm's input rows —
and fails the job if it is ever violated.

`wrap-inspired`'s pass 1 is the one special case, and it is still full coverage: it
rewrites every document once, but picks one of **four styles per document**
(`easy`/`hard`/`wiki`/`qa`) using a seeded RNG, and records which one in the `wrap_style`
output column. You do not have to do anything about this — it is automatic and
reproducible across restarts — but do not be surprised to see four different prompts in
one job's log.

**You are only rewriting half the corpus.** Each block upstream is a shared raw 20B "core"
plus a block-specific remainder, and only the remainder is rewritten. The core, and a
separate 50B `quality-base` control block, never touch a GPU and are **not downloaded by
this repo at all**. The 360 B above is the remainder total — the part that is actually
yours to process. Full explanation: `docs/DESIGN_DELTA.md`.

**Scale.** 360 B source tokens × 2 passes = **720 B input tokens** through the GPUs,
producing **~261 B output tokens** (measured, not assumed — see `docs/DESIGN_DELTA.md`
§5). That is roughly **1.1 TiB** of JSONL uncompressed, **~0.34 TiB** with the default
zstd. Wall-clock depends entirely on your GPUs — `preflight.py` prints a per-GPU KV-cache
estimate, `06_calibrate.py` projects the whole run from a measured rate, and every job logs
a live tok/s and ETA. Assume days-to-weeks, and assume **it will be interrupted many
times**. That is fine: everything resumes at shard granularity. Re-running the same
command is always the correct recovery action.

**This run is Blackwell only: B200 and B300.** H200 is excluded by design and the code
enforces it. Read **§2.3** before you start.

If you have more than one node — at ~4 GPUs per node, ~100 GPUs is about **25 nodes** —
read **§3.5**; the orchestration has roles, and one job must never be postprocessed by two
nodes at once.

**Where the run ends.** With finished data on disk. **Nothing is uploaded** — delivery is
arranged separately and is not this pipeline's job. See **§3.6** for the exact path.

**What you must supply:** eleven values in `configs/cluster.yaml` and one token in `.env`,
plus one optional value (`fleet_gpus`) if you are on more than one node. Section 2 lists
every one.

**What Wytro must supply before you can start:** access to the gated input dataset (see
`HF_TOKEN` in §2.5). That is the only thing. If `preflight.py` stops on it, it is not
something you can fix on your side.

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

The **shared / node-local** column matters only if you have more than one node, but it is
easier to get right the first time than to move later. See §3.5.

| key | what it is | shared or node-local | how to find it |
|---|---|---|---|
| `repo_root` | absolute path to this clone | either | `cd` into the repo, run `pwd` |
| `model_dir` | where model weights land, ~20 GB | **node-local** | any local disk with room: `df -h <dir>` |
| `data_root` | downloaded + re-sharded inputs | **shared** | needs roughly 2× the raw dataset size; `preflight.py` prints the real number |
| `out_root` | rewritten output — **the big one** | **shared** | ~0.65 TiB with zstd, ~2.2 TiB without — **two** copies, see below |
| `tmp_root` | shuffle scratch | **node-local** | must be **fast local** disk, not NFS/Lustre; ~1.2× the largest single job's output |
| `log_root` | per-worker logs | **shared** | small; `"${repo_root}/logs"` is fine |
| `hf_cache` | `HF_HOME` | **node-local** | put it with `model_dir`; **not** your home directory if it has a quota |

> **Why `out_root` is two copies.** The trim runs *in place*, so `raw/` is not consumed,
> and the shuffle then writes `shuffled/` as a second complete copy of the same rows. Both
> are still there when the run ends. `preflight.py` check 9 gates on the doubled figure.

### 2.2 `configs/cluster.yaml` — `compute`

| key | how to determine it |
|---|---|
| `num_gpus` | `nvidia-smi --list-gpus \| wc -l`. **This node's GPUs, not the fleet's** — every node reads its own copy of this file. Use every GPU you are entitled to; see §4 if the node is shared. |
| `fleet_gpus` | **Optional, null by default. Set it if you have more than one node.** The TOTAL across every node, e.g. 25 nodes × 4 = `100`. It is the only thing that can check the fleet-wide GPU ceiling — see the box below. |
| `gpu_ids` | leave as `auto` (means `0..num_gpus-1`). Only set an explicit list like `[0,1,2,3]` if you must avoid specific cards. Its length must equal `num_gpus`. |
| `cpu_workers` | `nproc`. Used only by the CPU trim stage; no GPU involved. |
| `shard_assignment` | leave as `dynamic` — B300 is faster than B200; see §2.3. |
| `shuffle_mem_bytes` | `free -b \| awk '/^Mem:/{print $2}'`, then take ~70%. Internally capped at 256 GiB regardless of what you put. |

To see VRAM per GPU (you need ≥ 24 GiB per card):

```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
```

> **⚠ The sharding caps this run at about 330 GPUs — across the WHOLE FLEET.**
>
> Every job's work is divided into shards, and `configs/data.yaml` requires at least
> `min_shards_per_gpu: 20` shards per GPU so the tail of each job does not leave most of the
> fleet idle. The smallest arm sets the limit:
>
> ```
> max_gpus = ceil(docs / shard_target_rows) / min_shards_per_gpu     over the SMALLEST arm
>
>   disagreement-aware   33,381,230 docs / 5,000 =  6,677 shards / 20 =  333   <- binding
>   diversity-oriented   35,304,301 docs / 5,000 =  7,061 shards / 20 =  353
>   quality-first        37,511,431 docs / 5,000 =  7,503 shards / 20 =  375
>   wrap-inspired        63,226,477 docs / 5,000 = 12,646 shards / 20 =  632
>   rewire-inspired     126,480,544 docs / 5,000 = 25,297 shards / 20 = 1,264
> ```
>
> **Read this part carefully — it used to say something that was not true.**
>
> There *is* an automatic check in the data-prep step, but it compares the shard count
> against `compute.num_gpus`, and **`num_gpus` is per node**. On one node that is the
> fleet and the check is real. On 25 nodes it compares 6,677 shards against that node's
> 4 GPUs, gets 1,669:1, and passes without ever looking at the fleet. **It cannot catch a
> fleet that exceeds the ceiling.**
>
> Two things follow:
>
> 1. **Set `compute.fleet_gpus`** to your total across all nodes. That turns the check into
>    a real one, and it is the *only* automatic enforcement of this ceiling that exists.
> 2. **If you leave it null, nobody but you is checking.** Here is the arithmetic, so you
>    can apply it to whatever count you end up with:
>
>    ```
>    shards(smallest arm) / min_shards_per_gpu  =  6,677 / 20  =  333 GPUs   (fleet total)
>    ```
>
>    At ~4 GPUs per node that is about **83 nodes**. Re-derive it if `shard_target_rows`
>    or `min_shards_per_gpu` ever changes; do not carry the 333 forward as a constant.
>
> **If your fleet total is at or under ~330, nothing to do.**
>
> **If you have more than that, do not just set the number and start.** The only remedy is a
> smaller `shard_target_rows`, and that changes the manifest fingerprint — which invalidates
> every completed shard and means re-preparing all the data. It is cheap before job 1 and
> expensive after. Tell Wytro your GPU count first; the value is his to change, not yours.

### 2.3 GPU architecture — this run is Blackwell only

**All 10 jobs run on B200 (`sm_100`) and B300 / Blackwell Ultra (`sm_103`). H200 is
excluded, even if it is present on the node.** This is Wytro's experimental design, and it
is enforced in code: `configs/data.yaml` carries `compute_constraints.allowed_gpu_arch`,
`preflight.py` refuses to start on a disallowed card, and **every worker refuses on its
own** — because preflight can be skipped and `03_run_job.sh` can be run directly.

*Why — one architecture family.* Greedy decoding at `temperature=0` is **not** bitwise
identical across architectures: vLLM selects a different attention backend, an argmax can
flip on a near-tie, and because generation is autoregressive the rest of that output
diverges. If some arms ran on Hopper and others on Blackwell, arm-vs-arm differences would
be confounded with GPU architecture — the exact comparison this experiment exists to make.
Restricting every job to one family removes the confound. **This part is non-negotiable and
is sufficient on its own.**

*Why Blackwell specifically — throughput.* Roughly 2–3× Hopper on this workload. That is
the whole of it: a cost decision, not a fidelity one.

*What that costs, stated plainly.* The 1.5B corpora were generated on H100 (`sm_90`) with
FlashAttention v3, which is Hopper-only. H100 and H200 are both `sm_90` and share that
path, so **H200 is actually the architecture closest to how the source data was produced** —
excluding it moves this run *further* from the source's numerics, not nearer. That is
accepted: this run is **not** numerically continuous with the 1.5B run, and it never could
have been across a GPU generation change (the vLLM version differs too). Cross-scale
comparability rests on matching prompts, budgets and selection procedure — all verified —
not on bitwise-identical generation. Either way, every shard records the card that produced
it (`gpu_cc` in its `.done` sidecar).

If you are ever weighing a hardware swap, weigh it against *those* reasons.

**What you have to do.** If the node has non-Blackwell cards, select only the Blackwell
ones:

```bash
nvidia-smi --query-gpu=index,name,compute_cap --format=csv
```

Put just those indices in `compute.gpu_ids` and set `compute.num_gpus` to how many that is.
`preflight.py` prints this instruction with your actual card list if it finds a stray one.

**Do not widen `compute_constraints.allowed_gpu_arch` in `configs/data.yaml`** to make a
failure go away. That file is the experiment definition, not a settings file. If a card is
rejected and you think it should not be, ask Wytro.

**B200 and B300 together are fine.** They are both Blackwell and share an attention
backend, so their numerics are close. The residual `sm_100` vs `sm_103` difference is
accepted, and every shard's `.done` sidecar records `gpu_name` and `gpu_cc` so the mixture
stays auditable. Two consequences:

* **Do not delete the `.done` sidecars.** They are the resume markers *and* the record of
  which card produced which rows.
* **Do not change `compute.gpu_ids` partway through the 10 jobs.** Jobs are sequential and
  each uses every GPU, so a fixed set means every arm sees the same hardware mixture.
  Changing it midway is what would actually bias the comparison.

Leave `compute.shard_assignment` as `dynamic`: B300 is faster than B200, and dynamic
claiming stops the quicker cards idling while the slower ones finish. In testing, a worker
8x slower than its peers took 28 shards while they took 61 each; an even split would have
forced 50/50/50.

### 2.4 `configs/cluster.yaml` — `env`

| key | how to determine it |
|---|---|
| `activate_cmd` | **`scripts/00_setup_env.sh` prints the exact line to paste** when it finishes. Do not invent it. |
| `extra_preamble` | anything else needed before `python` runs (module loads, proxy vars). Leave `""` if nothing. |

`scheduler.kind` stays `bash`. Only change it if you have read §4 and
`scripts/optional/slurm_job.sbatch`.

### 2.5 `.env`

```bash
cp .env.example .env
```

| variable | what |
|---|---|
| `HF_TOKEN` | your HuggingFace token with **read** access to `wytro/Know-Your-Sources-7B`. Create at <https://huggingface.co/settings/tokens>.<br>**That dataset is GATED.** Having a valid token is not enough — the account behind it must have been *granted access* by Wytro. Listing the repo works without access; downloading does not, so this fails late unless checked. `preflight.py` check 7 fetches a real file to prove it. If it reports a 401/403 or the word `gated`, stop and ask Wytro to approve your account; do not work around it. |
| `HF_TOKEN_WRITE` | **Leave it blank.** Upload is disabled (§3.6), so nothing needs write scope. Only fill it if you have deliberately set `upload.enabled: true` in `configs/data.yaml`. |

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

# 4. The whole run: model -> data -> calibrate -> 10 jobs -> postprocess. That is all;
#    there is no upload step (see 3.6 for where the finished data lands).
#    Calibration measures real throughput and projects wall clock BEFORE job 1. Read that
#    projection -- a wrong number is worth noticing on day zero, not day six -- but note
#    it is a PER-NODE projection; divide by your node count.
#    On several nodes, see section 3.5 instead of this line.
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
bash scripts/run_all.sh --from-job=5          # resume at job 5 in the status table
python scripts/04_postprocess.py --dry-run --sample 100000   # strip rates, no writes
```

`--skip-upload` still exists and is still accepted, but it is redundant: upload is already
off by default. The flag and `upload.enabled` are both *veto-only* — neither can switch
upload on when the other says off — so they can never contradict each other.

---

## 3.5 Running across several nodes

At ~4 GPUs per node, ~100 GPUs is about **25 nodes**, not one. The generation core is
already multi-node safe — workers claim shards through atomic `mkdir` on the shared output
directory, so which node does which shard sorts itself out — but the **orchestration has
roles**, and getting them wrong is the one way to corrupt output rather than just waste
time.

**Requirements.**

| | roots | why |
|---|---|---|
| **shared** — same filesystem, same path on every node | `out_root`, `data_root`, `log_root` | shard claims, `.done` sidecars and per-job locks are all coordinated through them |
| **node-local** — each node's own disk | `tmp_root`, `model_dir`, `hf_cache` | see below |

`tmp_root` holds shuffle buckets, and being node-local is *also* what lets two nodes
postprocess two different jobs without their bucket directories overlapping.

**`model_dir` must be node-local, and this one is worth understanding.** Every vLLM process
loads the full ~14.2 GiB of weights at every job start. On 25 nodes × 4 GPUs that is 100
processes reading ~1.4 TiB at each of the ten job transitions — ~14 TiB off the shared
filesystem, in ten bursts, each one competing with `data_root` reads and `out_root` writes.
A per-node copy costs one 14.2 GiB download per node, once, against local disk you have
already provisioned (~20 GiB). It is not close.

You do not have to do anything to make this happen: `run_all.sh` runs
`01_download_model.py` in *every* role including `--generate-only`, so each node fetches
its own copy at the start of step 2. It is idempotent and safe to run concurrently on all
25 nodes — the download is guarded by the same heartbeated lock the sharding path uses, so
even if you do point `model_dir` at shared storage the nodes serialise instead of
interleaving writes into one directory.

`compute.num_gpus` is **per node**. `compute.fleet_gpus` is the fleet total, and setting it
is what makes the GPU-ceiling check in §2.2 real.

**Three roles, in order.**

```bash
# 1. ONCE, on one node. Downloads the five arm folders, shards them, and stops.
#    (It downloads the model too, but only onto the node it runs on -- see step 2.)
bash scripts/run_all.sh --prepare-only

# 2. On EVERY node, at the same time. Generates; does not postprocess.
#    Each node fetches its own copy of the model, then works the 10 jobs in the same
#    order and claims whatever shards are free.
bash scripts/run_all.sh --generate-only

# 3. On EVERY node that has finished step 2. Takes whole jobs, one node per job.
bash scripts/run_all.sh --postprocess-only
```

Step 2 is safe to start on all nodes simultaneously — if you skip step 1 they serialise on
a per-arm sharding lock rather than racing — but doing step 1 first is cleaner and lets you
see the row counts before committing the fleet.

**Step 3 fans out, but only per job.** Run the same command everywhere; each node takes a
per-job lock, does that whole job, and moves to the next one that is free. Nodes with
nothing left to take say so and exit — that is the normal ending, not an error.

> **The rule is "never two nodes on the SAME job", not "only one node".** Two nodes
> shuffling one job would share both an output directory and a bucket temp directory, and
> the shuffle unlinks buckets as it consumes them — real corruption, not duplicated work.
> Two nodes on *different* jobs share neither: separate output directories, and `tmp_root`
> is node-local so the bucket directories cannot overlap. The per-job lock is what enforces
> the distinction, and it also stops two nodes picking the same job by accident.

**There are only 10 jobs, so postprocess parallelism caps at 10 nodes** — about 10× faster
than one node, not 25×. Beyond ten, the extra nodes exit immediately with nothing to take.
The floor on wall clock is the single largest job, `rewire-inspired/p1` at 126.5 M rows;
that one job cannot be split further.

To assign jobs by hand instead — deterministic, and useful if you want a specific node on
the big arm:

```bash
bash scripts/run_all.sh --postprocess-only --arm rewire-inspired --prompt-id p1
```

On a single node, nothing changes: run `--postprocess-only` with no arguments and it works
the whole list in order. The lock is uncontended. `--no-lock` on
`scripts/04_postprocess.py` skips it entirely, which is only ever safe when you are certain
nothing else is running anywhere.

**Monitoring.** `bash scripts/run_all.sh --status` works from any node at any time. Per
worker progress is at `log_root/<arm>/<prompt>/progress_<hostname>_w<N>.json` — the
hostname is in the filename precisely so 25 nodes' worker 0 do not overwrite each
other.

**If a node dies mid-run**, do nothing special. Its in-flight shards are claimed but no
longer heartbeated, and after `compute.claim_stale_after_s` (default 30 min) another node
takes them over automatically. To hurry it along, re-run step 2 on any node.

**One thing that is genuinely unsafe:** `--reap-claims --force`. It clears every claim
regardless of liveness, which on a live multi-node run means two nodes generating the same
shard. Use it only when nothing is running anywhere. Plain `--reap-claims` is safe at any
time; it only removes claims that have gone quiet.

---

## 3.6 Where the finished data is — the run ends here

**This pipeline does not upload anything.** Delivery of the finished data is arranged
separately. When `run_all.sh` prints its closing banner, the work is done and the result is
on disk. There is no step 6 to wait for and nothing further for you to run.

```
<out_root>/shuffled/<arm>/<prompt_id>/part_NNNNN.parquet
```

Ten directories, one per (arm, prompt):

```
<out_root>/shuffled/quality-first/p1/        <out_root>/shuffled/quality-first/p2/
<out_root>/shuffled/diversity-oriented/p1/   ... and so on for all five arms
```

| | |
|---|---|
| format | **Parquet**, zstd-compressed (the generation stage writes JSONL; the shuffle converts) |
| rows per file | 500,000 (the last file of each job is the remainder) |
| files | ~1,184 in total across the ten jobs |
| rows | ~592 M in total — 2 × 295,903,983, one row per document per prompt |
| columns | the 11 in `configs/data.yaml` `output.keys`, including `doc_id` and `wrap_style` |
| size | ≈0.3 TiB for `shuffled/` |

Each job's directory also holds `_shuffle.done`, a small JSON marker recording the row
count, shard count and seed. It is the proof the job completed and the reason a re-run
skips it — **do not delete it**.

**`<out_root>/raw/` is also still there**, and is roughly the same size again. The trim
runs in place, so `raw/` holds the same rows in JSONL form along with every `.done`
sidecar recording which GPU produced which shard. Budget ≈0.65 TiB for `out_root` overall,
and do not clear `raw/` without asking Wytro (§6 item 9).

If you ever do need to upload, everything is still in the repo: set `upload.enabled: true`
and `upload.repo_template` in `configs/data.yaml`, put a write-scoped token in
`HF_TOKEN_WRITE`, and run `scripts/05_upload_to_hf.py`. Both keys are required; setting one
without the other is a hard error rather than a late crash.

---

## 4. Stop and ask Tianjian

Agent: **do not decide these yourself.** Stop, ask, and wait.

1. **Where the data goes.** `out_root` needs up to ~1.1 TiB and `data_root` a comparable
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

## 4.5 If throughput looks wrong: this workload is prefill-heavy

Worth knowing before you reach for a tuning knob, because the usual instinct is the wrong
one here.

The configured budgets imply **720B input tokens against ~260B output — a 2.75:1 ratio**.
Most rewriting workloads sit near 1:1. Here roughly **three quarters of all tokens the GPUs
touch are prompt tokens**, so prefill, not decode, is where the wall clock mostly goes.

`scripts/06_calibrate.py` now reports the two separately — prompt tok/s and output tok/s,
the measured mix, and whether that mix matches the 2.75:1 the config expects. Read that
before concluding anything is slow. A blended tok/s number will look mediocre on this
workload even when the engine is behaving perfectly.

**This is not a new or untested regime.** The originating 1.5B run had the same shape: its
own census counters give 192.6B input against 69.5B output, **2.77:1** — within 1% of ours.
It ran to completion on the same engine version with the same settings.

**The settings that matter here are the ones vLLM chose by itself.** The source never passed
`max_num_batched_tokens` or `enable_chunked_prefill`; it inherited
`max_num_batched_tokens=16384` and `enable_chunked_prefill=True` from vLLM's defaults, and
those are recorded in `configs/vllm.yaml` under `inherited_defaults_do_not_pass` — copied
from the engine's own config dump in the source's logs. So the prefill path in this run is
already the exact path that produced the 1.5B corpus at 2.77:1.

**Do not change them.** Round 3 verified the engine args and source parity governs: any
engine arg the source did not pass changes generation behaviour at `temperature=0` and
destroys comparability across arms. `config.py` rejects additions outright. If your
measurements genuinely suggest the inherited prefill settings are mistuned for your
hardware, that is a finding to bring to Wytro with the numbers attached — a decision about
the experiment, not a local tuning change.

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
| **HF 429 / 5xx on upload** *(only if you have set `upload.enabled: true` — upload is off by default, §3.6)* | Rate limited. | Already handled: `05_upload_to_hf.py` retries 6 times with exponential backoff and jitter, and uploads are content-addressed so a re-run skips what already landed. If it still fails, wait and re-run — nothing is lost. |
| **Disk full mid-run** | Out of space in `out_root` or `tmp_root`. | Free space, then re-run the same command. **During generation** recovery is clean by construction: each shard is written to `.tmp` and renamed, and its `.done` sidecar is written only **after** the rename, so a crash can leave a complete shard with no marker (it gets redone) but never a half-written shard that looks finished. **During the shuffle it is not**: bucket files are unlinked as they are consumed, so running out of space there loses that job's whole shuffle and it restarts from its trimmed input. Nothing is *corrupted* either way, but this is the one stage where the cost is hours rather than minutes — which is why `preflight.py` check 9 now requires **two** full copies' worth of free space in `out_root`. |
| **A job finishes with row count ≠ input row count** | Serious. Every prompt must rewrite the entire arm. | `python -m rewrite.run_rewrite --arm A --prompt-id pN --verify --deep-verify` prints exactly which shards are missing or disagree. Delete those shards' `.done` sidecars and re-run the job. If it does not clear, **stop and ask** (§4.6) — do not upload. |
| **`sidecar fingerprint does not match the manifest`** | The input was re-sharded after some output was generated, so `doc_id`s were renumbered. | Do not delete anything yet. This normally means `sharding.*` in `data.yaml` was changed, or `data_root` was rebuilt. Ask Wytro. The safe fix is to restore the original sharding; the expensive fix is regenerating that job. |
| **`overhead is N, expected M`** | The prompt file, chat template, or tokenizer differs from the original pipeline. | **Do not edit `expected_overhead` to make it pass.** That integer is the proof the prompt is byte-correct. Check that no prompt file was modified (`git status`), and that the model revision resolved to the same chat template. Then ask. |
| **Model download keeps restarting** | Interrupted transfers. | It is idempotent — re-run `python scripts/01_download_model.py`. Once complete it verifies locally and never touches the network again (`--offline` forces that). |
| **`this torch build has NO kernels for sm_XXX`** | The pinned torch build cannot run one of your cards. | **Stop.** Do not install a different torch or vLLM — a different build changes generation behaviour and destroys comparability across the arms. Report the exact line to Wytro; re-pinning the whole stack so all 10 jobs share one build is the only correct fix. |
| **`no exact sm_XXX kernels ... will PTX-JIT`** | Your card runs, but through JIT-compiled PTX rather than tuned kernels. | Not an error. Expect a slow first few minutes while it compiles, then normal speed. Watch the first job's `tok/s`; if it stays far below the other cards, tell Wytro. |
| **One GPU finishes long before the others** | Heterogeneous fleet with `shard_assignment: static`. | Set `shard_assignment: dynamic` in `cluster.yaml` and re-run. Safe at any point — it only changes which worker takes which shard, never what is generated. Finished shards are skipped. |
| **A shard is stuck; no worker picks it up** | A `part_NNNNN.claim` directory left by a run that was killed. | The launcher clears these automatically on every start, so just re-run `03_run_job.sh`. To do it by hand: `python -m rewrite.run_rewrite --arm A --prompt-id pN --reap-claims`. **Never run that while workers are live** — it would let two workers take the same shard. |
| **`only N shards for M GPUs`** at data prep | `sharding.shard_target_rows` is too large for the GPU count it was measured against. The message says which count that was: *"on this node"* means `num_gpus`, *"across the fleet"* means `fleet_gpus`. At the shipped 5,000 rows the smallest arm gives 6,677 shards, which is 333:1 at a 20-GPU fleet and 67:1 at 100 — so this should only fire on a genuinely large fleet, or if an arm downloaded short. | The error computes the value to use. Set it in `configs/data.yaml`, delete that arm's shard directory, re-run. **Do this before job 1** — shard size feeds the manifest fingerprint, so changing it later invalidates every `.done` marker. |
| **`dataset has NO 'doc_id' column`** | An input dataset does not carry the join key. | Ask Wytro to add a `doc_id` column and re-upload; it costs nothing at upload time and makes the key durable. Only if that is impossible, set `sharding.require_doc_id: false` — the pipeline then synthesizes a row index, which works but ties the join to `data_root/shards/` surviving forever. |
| **A shard seems stuck, owned by a node that died** | Its claim is no longer heartbeated. | Wait: another node takes it over after `claim_stale_after_s` (30 min). To hurry it, re-run the job on any node. Do **not** use `--reap-claims --force` while other nodes are working. |
| **Two nodes both say "already running"** | The per-node lock doing its job — one launch per node per job. | Correct behaviour. Different nodes running the same job concurrently is how multi-node works; the same node twice is not. |
| **`postprocess: <job> held by <host>/pid N -> next job`** | Another node is already doing that job. | Correct behaviour, and how the work distributes itself (§3.5). Nothing to do. |
| **`nothing left for this node`** at postprocess | Every unfinished job is held by a live node. | The normal ending on a fleet with more nodes than jobs. Nothing to do. If one of those nodes later dies, its lock goes stale after `claim_stale_after_s` (30 min) — re-run the same command on any node and the job is taken over. |
| **A postprocess job is stuck, owned by a node that died** | Its `.postprocess.lock` is no longer heartbeated. | Wait for `claim_stale_after_s` (30 min) and re-run `--postprocess-only` anywhere; the lock is taken over automatically, exactly like a shard claim. The job restarts from its trimmed input — the shuffle is all-or-nothing per job. |
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
5. **Do not try to download `quality-base` or `shared-core`.** They are the raw half of
   the corpus: never rewritten, so this repo does not fetch them and there is no arm for
   them. `quality-base` is not on the Hub at all. Their token accounting is recorded in
   the header of `configs/data.yaml` and in `manifests/data_manifest.json` under
   `raw_not_rewritten`. If you think a sixth arm is missing, it is not — read
   `docs/DESIGN_DELTA.md` §3.
6. **Do not "fix" a version mismatch by installing a different vLLM/torch/transformers.**
   A different build produces different text at `temperature=0`, and afterwards there is
   no way to tell which rows came from which build. Stop and ask instead.
7. **Do not change `compute.gpu_ids` or `num_gpus` partway through the 10 jobs.** On a
   mixed-architecture fleet that confounds "which arm" with "which GPU" and quietly biases
   the comparison the whole experiment exists to make. Decide the GPU set once, before job
   1, and keep it.
8. **Do not delete `.done` sidecars to reclaim space.** They are tiny, they are the resume
   markers, and they carry the record of which GPU generated which shard.
9. **Do not delete `data_root/shards/` when the run finishes.** Output rows carry `doc_id`,
   which is the key for joining back to the input corpus (topic labels, quality scores, and
   so on) if that is ever needed downstream. Ask Wytro before reclaiming that space.
10. **Do not run two postprocess passes over the SAME job at once**, on one node or two.
    Two shuffles sharing a bucket directory corrupt each other's output. Running
    `--postprocess-only` on many nodes *is* supported and is how it is meant to be used —
    a per-job lock keeps each job to one node. The thing not to do is defeat that lock:
    do not pass `--no-lock` while anything else is running, and do not delete a
    `.postprocess.lock` belonging to a node that is still alive.
11. **Do not use `--reap-claims --force` while anything is running**, anywhere. It clears
    live claims and two nodes will then generate the same shards. Plain `--reap-claims` is
    safe at any time.
12. **Do not change `sharding.shard_target_rows` after job 1 has started.** It feeds the
    manifest fingerprint; changing it renumbers `doc_id` and invalidates every `.done`
    marker, i.e. throws the run away.
13. **Do not parallelise jobs to save time.** Each job already uses every GPU. Two at once
   means two engines each trying to reserve 85% of the same card. The `flock` in
   `03_run_job.sh` prevents it — do not remove it.
14. **Do not delete output to free space** without asking. A missing shard breaks row
   conservation, and at this scale regenerating one is hours of GPU time.
15. **Do not commit `.env`, a token, model weights, or data.** `.gitignore` covers all of
   them; keep it that way.
16. **Do not lower `gpu_memory_utilization` to dodge an OOM.** See §5, row 1.
