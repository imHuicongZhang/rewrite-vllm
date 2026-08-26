# DESIGN_DELTA — what changed between the repo as built and the experiment as designed

**Date:** 2026-08-25 · **Repo at start:** `728ac8f` · **Author:** Wytro Cheung

This document is standalone. It assumes no knowledge of `rewrite-vllm`.

`rewrite-vllm` is a portable pipeline that takes a pretraining corpus, rewrites it with
Qwen2.5-7B-Instruct on a GPU fleet, postprocesses the output and uploads it. It was built
over rounds 1–3 against a design that changed afterwards. This document records, item by
item, what the design was, what it is now, and how each new fact was established.

Round 4's brief was explicitly second-hand — recovered from a conversation summary rather
than read from code. **Section 8 lists the places it was wrong.** Those corrections are the
most important content here.

---

## 0. The three sources, and what each is authoritative for

| # | source | authoritative for |
|---|---|---|
| S1 | `13_600M/02_select/` | block composition, core/remainder split, `doc_id`, upload schema |
| S2 | `https://huggingface.co/datasets/wytro/Know-Your-Sources-7B` | what is actually on the Hub |
| S3 | `projects/rewrite/` (original pipeline) | prompts, engine args, trim rules, shuffle, style assignment |

Where they disagreed, the resolution is stated explicitly rather than silently picked.

---

## 1. Summary of the delta

| # | item | was | is now | §|
|---|---|---|---|---|
| 1 | wrap-inspired passes | 4 style passes, no distill | 1 styled pass (style per doc) + distill | 2 |
| 2 | total rewrite jobs | 12 | **10** | 2 |
| 3 | style assignment | one prompt per job | `np.random.default_rng([42, shard_index])`, one style per doc | 2 |
| 4 | style recorded as | `prompt_id` | new `wrap_style` output column | 2 |
| 5 | what gets rewritten | the whole arm dataset | **the remainder only**; a shared 20B core stays raw | 3 |
| 6 | arms in `data.yaml` | 6 (incl. `quality-base` control) | **5**, all rewritten | 3 |
| 7 | data source | 6 independent flat HF repos | **1 gated repo**, 5 folders | 4 |
| 8 | `est_output_tokens_per_arm` | `100e9` scalar × 5 = 500B | per-arm, measured; **~260B ± 10%** total | 5 |
| 9 | disk estimate | hardcoded "~2.4 TB / ~0.7 TB" | derived; **1.125 TiB / 0.338 TiB** | 5 |
| 10 | `shard_target_rows` | 2000 | **5000** | 5 |
| 11 | compression ratio | assumed | **measured per prompt** | 5 |
| 12 | the 50B/arm invariant | never stated | stated, with produced-vs-needed per arm | 6 |
| 13 | ReWire filter | "keep the top half" | **fixed-budget fill; 30.8% realized at 1.5B** | 10.1 |
| 14 | `r` transfer across populations | not acknowledged | open question; estimate now **~260B ± 10%** | 10.3 |
| 15 | prompt provenance | overhead fingerprint only | byte-compared vs originals; **all 6 identical** | 9 |

---

## 2. wrap-inspired: back to one style per document

### What the repo did

`configs/data.yaml` gave `wrap-inspired` four prompts (`p1..p4` = easy/hard/wiki/qa) and
declared, as an enforced invariant, that *every prompt rewrites the entire dataset for its
arm*. Four prompts therefore meant **four complete passes** over the wrap corpus, producing
four rewrites per document. `wrap-inspired` had **no distill pass**, unlike every other arm.
`2+2+2+4+2 = 12` jobs.

This was deliberate. Round 2 §C1 analysed it, recommended keeping it, and
`docs/HANDOFF_REVIEW.md:122` records that `assign_wrap_styles` and its seeded RNG were
*"deleted, not disabled — leaving them would invite a future reader to 'restore' them."*

### What the design actually is

One styled pass with the style chosen per document, **plus** the shared distill pass —
2 rewrites per document, the same rewrite cost as every other arm.

Established from S1 and S3 independently:

- **S3, `07_rewrite/rewrite_worker.py:39,54-62`** — the function itself:
  ```python
  WRAP_STYLES = ["easy", "hard", "wiki", "qa"]  # index order is part of the reproducible seed

  def assign_wrap_styles(shard_index, n_rows, base_seed=42):
      rng = np.random.default_rng([base_seed, shard_index])
      idx = rng.integers(0, len(WRAP_STYLES), size=n_rows)
      return [WRAP_STYLES[i] for i in idx]
  ```
- **S3, `07_rewrite/rewrite_worker.py:277,284`** — called once per shard, consumed by
  `build_content(mode, text, template, wrap_prompts, style)` which returns
  `wrap_prompts[style] + doc_text`.
- **S1, `02_select/README.md`** — *"`wrap-inspired` follows the 1.5B arm exactly: pass 1 is
  the 4-style assignment with one style per document … pass 2 is the shared distill prompt —
  2 rewrites per document, the same rewrite cost as every other setting."*
- **S3, `09_Distill/launch_dataset.sh:22-29`** — every arm including wrap runs
  `MODE=grounded` with the distill prompt, confirming wrap does get a distill pass.

**Job count: `2+2+2+2+2 = 10`.** Note the arithmetic is *not* simply "wrap loses 3 jobs":
wrap loses 3 style jobs **and gains a distill job it never had**.

### The styles are correct — no corruption

The brief warned that if `prompts/wrap-inspired/p1..p4` were the abandoned paper-verbatim
set, that would be a corpus-wide silent corruption outranking everything else.

**It is not.** All four files are byte-identical to the production set. Verified by md5
and `cmp` against S3 `prompts/wrap/{easy,hard,wiki,qa}.txt` (file names below are the
round-3 ones, i.e. what was checked; see the rename note underneath):

| round-3 file | → round-4 file | style | md5 | overhead |
|---|---|---|---|---|
| `wrap-inspired/p1.txt` | `wrap-inspired/style_easy.txt` | easy | `0735f53aca80cadaa8d67727680dbbfd` | 72 |
| `wrap-inspired/p2.txt` | `wrap-inspired/style_hard.txt` | hard | `e99a613bcd4146416428d576af6f200a` | 66 |
| `wrap-inspired/p3.txt` | `wrap-inspired/style_wiki.txt` | wiki | `cec46736de0229e6d7a0f022cd2e661a` | 73 |
| `wrap-inspired/p4.txt` | `wrap-inspired/style_qa.txt` | qa | `733fbeea43050cb4a4e27f9384b9014e` | 83 |

Note that `wrap-inspired/p2.txt` now means something entirely different: it is the arm's
new **distill** prompt (md5 `538700534e99d5e80b268fd9b2408b48`, byte-identical to every
other arm's `p2.txt`). That collision is exactly why the styles were renamed — under the
new design `p1`/`p2` identify *jobs*, and `easy`/`hard`/`wiki`/`qa` identify *styles within
one job*, and a file called `p2.txt` that is sometimes a style and sometimes a distill
prompt is a trap.

The abandoned set survives at S3 `06_vllm/wrap_styles_sample.py:35-50`. It is distinguishable
at a glance: its keys are `easy / **medium** / hard / qa`. `medium` appears nowhere in
`07_rewrite/`, `09_Distill/` or `10_postprocess/`. It was killed by a 100-document pilot
(`06_vllm/rewritten_examples/wrap_styles_summary.md`: easy failed on 32/100, hard on 22/100).

**After this round the repo's file names change** so a reader cannot confuse job identity with
style identity: the four style files move to `prompts/wrap-inspired/style_{easy,hard,wiki,qa}.txt`
and the arm's two *jobs* become `p1` (styled) and `p2` (distill).

### How the style is recorded

S3 `07_rewrite/rewrite_worker.py:316-323` appends a `wrap_style` `large_string` column, and
only in wrap mode. Without it the assignment is unrecoverable after the fact.

This repo writes a fixed-schema JSONL for every job, so a column that exists for one job and
not the others would make the ten output sets non-uniform. **`wrap_style` becomes an 11th key
present in every row of every job**, carrying the style for the wrap styled pass and the empty
string elsewhere. Cost is ~15 bytes/row. The Arrow schema in `jsonl_to_table` gains a matching
`large_string` field.

### Seed determinism across workers and resume

The requirement is that a worker restarting shard N assigns the same style to the same
document. This holds **structurally**, not by convention:

- the RNG is keyed only by `(42, shard_index)` — never by worker id, wall-clock, claim order,
  or how many shards this worker has already done;
- the whole shard is drawn in a single `size=n_rows` call, so there is no partial-consumption
  state to lose mid-shard;
- `shard_index` is fixed by the input manifest, which is fingerprinted, so it cannot drift
  under a resume.

A crash at row 3,000 of 5,000 therefore re-derives rows 0..4,999 identically on restart.
Tested explicitly (`tests/test_wrap_styles.py`), including a golden vector pinned against
`np.random.default_rng([42, i]).integers(0, 4, 16)` so a numpy upgrade that changes the PCG64
stream is caught rather than silently re-rolling the corpus.

> **Not glossed:** this pipeline re-cuts shards at 5,000 rows, so `shard_index` is *ours*, not
> the source's. The assignment is reproducible **within this pipeline**; it is **not** the same
> assignment the 1.5B run produced, and cannot be — different corpus, different sharding. What
> is preserved is the *mechanism* and its statistical properties, not the specific draw.

### The two overhead-verification paths — checked, both still correct

The restructure moved four templates from four jobs into one, so both checks that were
keyed to "one prompt per job" had to change. Both were changed, and all four wrap
overheads are still asserted. Measured against the real Qwen2.5-7B-Instruct tokenizer:

| template | measured | expected | |
|---|---:|---:|---|
| `p1_wiki` | 150 | 150 | ok |
| `p2_distill` | 185 | 185 | ok |
| `wrap_easy` | 72 | 72 | ok |
| `wrap_hard` | 66 | 66 | ok |
| `wrap_wiki` | 73 | 73 | ok |
| `wrap_qa` | 83 | 83 | ok |

**`expected_overhead` at runtime.** `PromptSpec.overheads()` returns one
`(label, text, expected)` triple per prompt *text* the job can emit — one for a grounded
job, **four** for the styled pass. `engine.check_overheads()` measures each;
`run_rewrite.run_worker` fails the job if *any* mismatches and names which. The expected
values live per style in `prompt_defs.wrap_styled.styles[*].expected_overhead`, so all four
are config-visible rather than one standing in for four. The sidecar records a dict of the
four rather than a single integer for that job.

**`scripts/preflight.py` check 6.** Loops the same `overheads()` expansion and reports the
count explicitly, so a silent collapse to one text per job would show up as a changed number.
Two counts are involved and they are easy to conflate — the header comment in `preflight.py`
drifted for exactly that reason and was corrected in round 6:

- **13** = `(job, template)` pairs asserted — 9 grounded jobs × 1 + the styled pass × 4.
- **6** = distinct prompt *texts* — the four grounded arms share `p1`, all five share `p2`,
  plus the four wrap styles.

Both are printed.

**`scripts/verify_prompt_parity.py`.** Runs standalone from the YAML without
`config.py`'s loader, so it needed its own expansion: a `_units()` helper yields four
entries for a `wrap_multi` def (reading `prompt_defs.<def>.styles`) and one otherwise. It
prints every expected-vs-actual pair, and it independently re-checks the **style order**
against `["easy","hard","wiki","qa"]`, since the order is part of the seed and this script
is the one that runs without the loader's guard.

**Net:** four values asserted where there were four before, in three independent places.
Neither check silently narrowed to a single style.

### The subsampling caveat was written for the four-pass design — removed

`data.yaml` and the dataset cards carried a long caveat about `wrap-inspired` holding four
copies per document and what that does to duplication under a token budget. **That premise is
gone**: there is now one styled rewrite per document, exactly as for every other arm. The
caveat is deleted rather than patched.

It is replaced by the real inherited asymmetry, which is about *tokens*, not copies. Uniform
per-document assignment balances documents but not tokens, because the four styles have very
different expansion factors. Measured at 1.5B
(S3 `10_postprocess/_step2_wrap_summary.json`):

| style | docs | doc share | tokens | token share | tok/doc |
|---|---:|---:|---:|---:|---:|
| easy | 2,646,480 | 25.05% | 591,400,168 | 14.23% | 223.5 |
| hard | 2,631,853 | 24.91% | 1,394,320,430 | 33.55% | 529.8 |
| wiki | 2,644,343 | 25.03% | 962,494,191 | 23.16% | 364.0 |
| qa | 2,643,215 | 25.02% | 1,207,123,069 | 29.05% | 456.7 |

Documents are balanced to 25.0% ± 0.1pp; tokens spread **2.37×** (`hard` 33.6% vs `easy`
14.2%). The source applied no correction and neither does this pipeline — but anyone cutting
this arm to a token budget should know that a uniform token draw is not a uniform style draw.

---

## 3. Only the remainder is rewritten

### What the repo did

Six arms, each a whole dataset, five of them rewritten end to end. `quality-base` was a
control with `rewrite: false`, downloaded and verified so the token accounting was complete.

### The real structure

`02_select/select_600m.py:86-95` composes six **blocks**. Five carry a **shared 20B raw core**;
each block then adds its own **remainder**:

```python
BLOCK_SPEC = {
    'quality-base':       (False, QBASE_TARGET,      'ftq-full-pool'),
    'quality-first':      (True,  REM_TARGET,        'ftq-core-excluded'),
    'diversity-oriented': (True,  REM_TARGET,        'topic-stratified-cq'),
    'disagreement-aware': (True,  REM_TARGET,        'udis-core-excluded'),
    'wrap-inspired':      (True,  REM_TARGET,        'uniform-child2'),
    'rewire-inspired':    (True,  REWIRE_REM_TARGET, 'uniform-child3'),
}
```

**The core stays raw; the remainder is rewritten.** `select_600m.py:213-233`:

> `is_core` matters downstream: `doc_ids.npy` is the SORTED UNION of core + remainder, so the
> split is otherwise unrecoverable — and **the rewrite stage rewrites ONLY the remainder,
> carrying the core through as raw text.**

The core is provably identical across the five carrying blocks — `select_600m.py:631-637`
compares sha256 over the sorted doc_ids and `stop()`s on mismatch. So rewriting it once per
arm would be **five times redundant**, which is exactly why it is not rewritten at all.

### Is the raw core physically separate on the Hub? Yes

`upload_blocks.py:71-80` uploads six directories, and the last field selects which half:

```python
UPLOADS = [
    ('shared-core',        'quality-first',      True),
    ('quality-first',      'quality-first',      False),
    ('diversity-oriented', 'diversity-oriented', False),
    ('disagreement-aware', 'disagreement-aware', False),
    ('wrap-inspired',      'wrap-inspired',      False),
    ('rewire-inspired',    'rewire-inspired',    False),
]
```

applied at `upload_blocks.py:222-223` as `keep = pc.equal(tbl['is_core'], want_core)`.

**So each block folder on the Hub contains the remainder only, and the core ships once as
`shared-core/`.** There is no `split=` distinction — it is folder, plus an `is_core` column,
plus a `remainder_doc_ids.npy` sidecar. Three independent markers of the same split.

### `quality-base` gets zero rewrite passes — and is not on the Hub at all

`upload_blocks.py:15-16`:

> quality-base is NOT uploaded and NOT deleted: it is the raw-text control, is never rewritten,
> and has no place in a repo whose purpose is to feed the rewriting stage.

Confirmed on the Hub: the repo has six top-level folders and **`quality-base/` is not one of
them**. Its `stats.json` carries `"is_rewritten": false`, `"rewrite_source_tokens": 0`.

### Consequence for this repo

`quality-base` and `shared-core` are both dropped **as arms**. This pipeline's scope is
rewriting; neither sees a GPU, and downloading 20B raw core tokens onto the run machine would
cost bandwidth and disk for nothing. Both are recorded in a comment block at the top of
`configs/data.yaml` so the token accounting stays complete while preflight and the manifest
ignore them. The unfillable `quality-base` `repo_id` placeholder is deleted.

**Five arms, ten jobs.**

### Which tokens go through the GPU

| block | docs | source tokens (llama-2) | GPU? |
|---|---:|---:|:--|
| shared-core | 17,909,083 | 20,000,010,702 | **no** — raw, carried by all 5 arms |
| quality-base | 37,298,288 | 50,000,002,028 | **no** — raw control, local only |
| quality-first | 37,511,431 | 60,000,000,654 | yes, ×2 passes |
| diversity-oriented | 35,304,301 | 60,000,039,601 | yes, ×2 |
| disagreement-aware | 33,381,230 | 60,000,014,390 | yes, ×2 |
| wrap-inspired | 63,226,477 | 60,000,001,229 | yes, ×2 |
| rewire-inspired | 126,480,544 | 120,000,000,593 | yes, ×2 |

**360.0B source tokens × 2 passes = 720B input tokens through the GPU.
70B tokens (20B core + 50B quality-base) never touch a GPU.**

Token convention throughout is `tokens-llama2 + 1` (one leading BOS), per
`select_600m.py:136,273`.

---

## 4. Data access layer

### What the repo did

`configs/data.yaml` held six independent HF repo ids, each assumed to be a flat dataset with a
`text` column. `scripts/02_download_data.py` looped over all six.

### What is actually there

One repo: **`wytro/Know-Your-Sources-7B`** (`upload_blocks.py:67`), pinned at
sha `6e18cda64fbd24fe46010b2aa578f14a4255076d`.

Observed on the Hub (2026-08-25, via the public metadata API — the file *listing* is readable
anonymously even though file *contents* are not):

```
shared-core/         data/part_w{00..11}_{seq:04d}.parquet, core_doc_ids.npy, core.json
<block>/             data/part_w{00..11}_{seq:04d}.parquet, remainder_doc_ids.npy,
                     stats.json, _manifest.json, SELECTION_REPORT.md
```

487 files, 12 writers (`w00..w11`) in every folder, 662.50 GB total.

| folder | parquet files | LFS bytes | docs |
|---|---:|---:|---:|
| shared-core | 60 | 35.29 GB | 17,909,083 |
| quality-first | 60 | 101.12 GB | 37,511,431 |
| diversity-oriented | 60 | 101.37 GB | 35,304,301 |
| disagreement-aware | 60 | 99.93 GB | 33,381,230 |
| wrap-inspired | 84 | 108.50 GB | 63,226,477 |
| rewire-inspired | 139 | 216.30 GB | 126,480,544 |

Document counts were read off the `.npy` sidecar file sizes — `(bytes − 128) / 8` is an exact
integer for all six, consistent with an int64 array and a 128-byte header — and then confirmed
against `stats.json` and the upload logs in S1. They agree exactly.

Uploaded schema is **34 columns**: the 32-column master schema, plus `is_core` (bool, added at
`materialize_blocks.py:98`) and `text` (`large_string`, added at `upload_blocks.py:231`),
written zstd level 3. The pipeline needs `doc_id`, `text`, and — for provenance — nothing else;
the other 31 columns are ignored and can be joined back later on `doc_id`.

### Two Hub-side traps

1. **The repo is `gated: "manual"`.** The listing is public; contents 403 without an approved
   token. This is a hard blocker for a one-button run and it belongs to Wytro, not Tianjian.
2. **The card declares no `configs:` block.** `README.md` on the Hub contains only
   `license: cc-by-4.0`. The Hub auto-converter therefore reports a single config `default`
   with split `train` globbing *every* folder — so a naive `load_dataset("wytro/Know-Your-Sources-7B")`
   would silently mix `shared-core` into all five arms and inflate every arm by 17.9M rows.

   **The access layer therefore addresses folders by explicit `data_files` glob
   (`"<subdir>/data/*.parquet"`), never by config name.** That works today and keeps working if
   the card is fixed later. A ready-made `configs:` block already exists at
   `data_reports/DATASET_CARD_DRAFT.md` and should be pushed, but nothing here depends on it.

### `doc_id` uniqueness

`doc_id` is **int64 and globally unique across the whole 600M-document corpus** — it is the row
index into that corpus, assigned upstream in `materialize_master.py` and never renumbered by
selection (`select_600m.py:132`: *"doc_id IS the index into each of these"*). The same document
therefore carries the same `doc_id` in every block it appears in, which is what makes the
half+half reassembly work.

Two consequences that matter here:

- **Within one arm's remainder, `doc_id` is unique.** Asserted upstream ("no duplicate doc_id
  within a block"). This is the only uniqueness this pipeline depends on: shards are cut within
  an arm, and the row-conservation proof is per (arm, prompt).
- **Across arms it is not a row key.** Remainders are drawn from the same 577M core-excluded
  pool by different criteria, so the same `doc_id` can legitimately appear in several arms. The
  output rows already carry `(doc_id, arm, prompt_id)`, so the downstream join key is the
  triple, not `doc_id` alone. Nothing needed changing.

`require_doc_id: true` and the `doc_id_source` override are retained unchanged; the column is
present in the upload, so `doc_id_source` will read `dataset` for all five arms.

**A doc_id disjointness / namespace check is deliberately NOT added here.** That verification
belongs upstream at `13_600M/02_select`, where the id arrays are local and cheap to compare.
See §9.

### Fingerprint gap found and closed

`data.py:129-138` folds `content_sha1`, `shard_target_rows`, `shard_target_bytes` and
`DOC_ID_POLICY` into the manifest fingerprint, but **not `require_doc_id`** — yet flipping that
flag switches `doc_id` between the dataset's own column and a synthesized row index, which
renumbers every row. The brief's requirement is that the fingerprint cover everything that
would renumber rows, so `require_doc_id` and the resolved `doc_id_source` are now folded in.
This was a pre-existing defect, not a consequence of the design change.

---

## 5. Recomputed numbers

### Compression ratio — measured, not assumed

Every token estimate in the repo rested on an assumed output/input ratio. The brief asked for it
to be measured from the 1.5B-era outputs, per prompt, over `status == 2` rows only.

**The census parquets are gone.** `07_rewrite` wrote
`<dataset>/rewritten/part_*.parquet` and `09_Distill` wrote `<dataset>/distill/part_*.parquet`
under `DATA_ROOT=/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/10B`. That tree no
longer exists, and a recursive search for `*.parquet` across the scratch space finds no
`rewritten/` or `distill/` directory. Those files were the only place
`input_tokens_qwen` + `rewritten_tokens` + `status` lived together per row.

Two survivors were used instead:

- **Census, exact but arm-level** — `07_rewrite/progress/*.json` and `09_Distill/progress/*.json`
  (146 worker files). Carry `docs_status_{0,1,2}` and `total_output_tokens` (llama-2) per arm.
- **Sampled, per-row with status** — `00_TMP/rewriting_monitor.md` (18,190 records) and
  `rewriting_monitor_distill.md` (18,000). Seeded 10-per-10,000 subsample; `in_tok` and
  `out_tok` are both llama-2, so their ratio is tokenizer-consistent.

Budgets are denominated in llama-2 tokens, so the quantity wanted is
**output llama-2 tokens per source llama-2 token**. Taking the census output totals over the
1.5B per-arm source budget (10B llama-2; 20B for rewire, κ=2):

| 1.5B arm → round-4 arm | r (styled/wiki pass) | r (distill) | sum | independently quoted |
|---|---:|---:|---:|---:|
| quality-first → quality-first | 0.3399 | 0.2581 | 0.5981 | 0.598 |
| diversity-first → diversity-oriented | 0.3812 | 0.2845 | 0.6657 | 0.666 |
| signal-disagreement-λ05 → disagreement-aware | 0.3649 | 0.2750 | 0.6400 | 0.640 |
| wrap → wrap-inspired | 0.4313 | 0.3657 | 0.7970 | 0.797 |
| rewrite → rewire-inspired | 0.4628 | 0.3660 | 0.8288 | 0.829 |

**Validation.** The right-hand column is `select_600m.py:114-123`'s independently recorded
*"Measured 1.5B 2-prompt yields were 0.598 / 0.640 / 0.666 / 0.797 / 0.829"*. The sums
reproduce all five to three decimals. That confirms both the denominator and the decomposition.

**On the `status == 2` restriction.** The census totals include status-1 (truncated) rows. The
brief expected truncation to bias `r` downward; in this denominator it biases *upward*, because
a truncated row contributes a maxed-out 4,096-token output. Either way the effect is negligible:
status-1 is 0.02–0.33% of documents. The sampled monitors, filtered to `status == 2` exactly as
asked, give 0.3496 / 0.3956 / 0.3934 / 0.4178 / 0.4682 for pass 1 and 0.2655 / 0.2952 / 0.2957 /
0.3677 / 0.3744 for distill — within ~3% of the census figures, the residual being sampling
noise on ~2,000 documents per arm. **The census figures are used**, being exact.

### Output tokens per arm

| arm | source (B) | docs | out pass 1 (B) | out distill (B) | out total (B) |
|---|---:|---:|---:|---:|---:|
| quality-first | 60.0 | 37,511,431 | 20.40 | 15.49 | 35.88 |
| diversity-oriented | 60.0 | 35,304,301 | 22.87 | 17.07 | 39.94 |
| disagreement-aware | 60.0 | 33,381,230 | 21.90 | 16.50 | 38.40 |
| wrap-inspired | 60.0 | 63,226,477 | 25.88 | 21.94 | 47.82 |
| rewire-inspired | 120.0 | 126,480,544 | 55.54 | 43.92 | 99.46 |
| **total** | **360.0** | **295,903,983** | | | **261.49** |

`est_output_tokens_per_arm: 100e9` × 5 arms = 500B **overstated the real total by 1.91×**, and
the real per-arm figures span 35.9B–99.5B — a 2.8× spread a single scalar cannot express. It
becomes a per-arm value, with the measured `r` recorded beside it.

### Disk

Using the repo's own sizing model (4.2 bytes/output token; the per-row JSON envelope goes
from 220 to **235 bytes** because of the new `wrap_style` key):

```
est output tokens = 261,492,037,430                      = 261.49 B
output rows       = 295,903,983 docs × 2 passes          = 591,807,966
raw   = 261.49e9 × 4.2 + 591,807,966 × 235               =   1.125 TiB  (1.237 TB)
zstd  = raw × 0.30                                       =   0.338 TiB  (0.371 TB)
```

The guide's "~2.4 TB uncompressed, ~0.7 TB with zstd" was a hardcoded figure that never tracked
the design. It is now computed from the config in one place and quoted from there.

### `shard_target_rows`: 2000 → 5000

Round 3 chose 2000 because the smallest arm then had 5,602,476 documents, and at 100 GPUs
*across the fleet* 10,000-row shards would have produced only 561 shards — failing the
`min_shards_per_gpu: 20`
guard. **The smallest remainder is now 33,381,230 documents, 6× larger, so that failure mode is
gone** and every candidate passes with room. The live tradeoff is tail-idle against filesystem
metadata load:

| rows | input shards | output files (10 jobs) | smallest arm | ratio at 100 GPUs *(fleet-wide)* | tail |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 147,955 | 591,820 | 16,691 | 167:1 | ~2 min |
| **5,000** | **59,184** | **236,736** | **6,677** | **67:1** | **~6 min** |
| 10,000 | 29,594 | 118,376 | 3,339 | 33:1 | ~11 min |
| 16,000 | 18,497 | 73,988 | 2,087 | 21:1 | ~18 min |

**5,000 chosen.** At 2,000 the run would create 147,955 input shards and ~592k output files —
63,241 shards for `rewire-inspired` alone — which is heavy metadata load on a shared filesystem
for a tail saving of four minutes per job.

It stays an explicit constant rather than being derived from `num_gpus`, for the round-3 reason
that still holds: `num_gpus` lives in Tianjian's `cluster.yaml`, while shard size feeds the
manifest fingerprint. Deriving one from the other would let a machine-local setting silently
renumber `doc_id`.

### Prefill is ~73% of the tokens, and the calibration now says so

720B input against ~260B output is **2.75:1**. Most rewriting workloads sit near 1:1, so
prefill — not decode — dominates GPU time here, and a single blended tok/s hides that.
`scripts/06_calibrate.py` now reports prompt tok/s and output tok/s separately, the measured
mix, and whether that mix matches the 2.75:1 the config implies (flagging a >25% divergence,
which would indicate the token estimates are wrong rather than the engine).

**The regime is not new.** The originating 1.5B run had the same shape — its census counters
give 192.6B input against 69.5B output, **2.77:1**, within 1% of ours — and it completed on
the same engine version. So the inherited prefill settings have already been exercised at
this ratio at scale.

**Engine args unchanged, and the relevant ones were never ours to begin with.** The source
never passed `max_num_batched_tokens` or `enable_chunked_prefill`; it inherited
`max_num_batched_tokens=16384` / `enable_chunked_prefill=True` from vLLM's defaults, visible
in its own logged config dump (`07_rewrite/logs/rw_*_*.out:6,9`) and already recorded in
`configs/vllm.yaml` under `inherited_defaults_do_not_pass`. There is therefore no evidence
of mistuning to report — the prefill path in this run is the one that produced the 1.5B
corpus at 2.77:1. Guidance for Tianjian is in `GUIDE_FOR_TIANJIAN.md` §4.5, framed as
"measure before concluding, and bring findings to Wytro", not as a config change.

### Calibration

`scripts/06_calibrate.py` reads shard and row counts from the manifests, so it follows the new
counts automatically. Its per-job projection now takes `tok_per_doc` from the per-arm measured
`r` rather than a global constant, and it enumerates 10 jobs.

---

## 6. The 50B invariant, and did we generate enough?

### Every arm lands at 50B training tokens

This is the constraint the whole half+half structure exists to satisfy, and earlier
revisions of this document never stated it.

```
quality-base   =  50B raw                          (top-50B prefix of the fastText order)
every other arm =  20B raw core  +  30B rewritten   =  50B
```

Established, not inferred:

- `select_600m.py:78-79` — `CORE_TARGET = 20_000_000_000`, `QBASE_TARGET = 50_000_000_000`.
- `select_600m.py:358` — *"`quality-base` | `final_training` | the raw-text control. All 50B
  is used as-is; **never rewritten**. This is what the model trains on."* So 50B is a
  **final training** budget, and it is the only arm whose final budget is stated directly.
- `02_select/README.md` — `quality-base` is the **top-50B prefix** of the same `ftq` order
  whose top-20B prefix is the core; `verify_materialize.py:116-126` asserts that
  `quality-base` therefore *contains* the core. So `quality-base` is already
  `20B core + 30B next-best raw`.
- For the arms to be comparable, every other arm must also reach 50B: its 20B core plus
  **30B drawn from its rewritten output**.

The 1.5B run confirms the pattern at half the ratio: `select_10b.py:48-51` has
`SHARED_TARGET = 5B`, `QBASE_TARGET = 5B`, `TARGET = 10B`, and the postprocess mixed a 5B
shared core with a 5B rewritten selection (`03_mix_shared_top*.py`;
`_step2_wrap_summary.json` records `target: 5000000000`, `shortfall_vs_target: 0`). So
10B = 5B + 5B there, and 50B = 20B + 30B here. The core's share dropped from 50% to 40%
between scales; the *invariant* is the 50B total, not the split.

The remainder budgets are **source** budgets and always over-produce; the surplus is
trimmed to 30B. `select_600m.py:122`: *"Expect ~36B out of 60B and trim to budget, as at
1.5B."*

### Did we generate enough? — output produced vs output needed

| arm | source | output produced | needed | ratio | headroom |
|---|---:|---:|---:|---:|---:|
| quality-first | 60B | 35.88B | 30B | 1.196× | **+19.6%** ← tightest |
| diversity-oriented | 60B | 39.94B | 30B | 1.331× | +33.1% |
| disagreement-aware | 60B | 38.40B | 30B | 1.280× | +28.0% |
| wrap-inspired | 60B | 47.82B | 30B | 1.594× | +59.4% |
| rewire-inspired | 120B | 99.46B | 30B | 3.315× | +231% |

Every arm clears its budget, and **`quality-first` is the tightest at +19.6%**, not
`rewire-inspired`.

Each ratio is exactly `r_arm × source / 30B`, so it reproduces the 1.5B run's headroom
*identically* — the budgets scaled 6× (10B→60B source, 5B→30B needed) and `r` was carried
over, so the ratios cannot differ. Every one of these arms demonstrably filled its budget
at 1.5B (`_step2_wrap_summary.json` `shortfall_vs_target: 0`;
`_step4_rewrite_summary.json` `filled: true`). That is the strongest available evidence
that they will fill again — and it is also exactly why §9's `r`-transfer caveat matters:
the headroom column is only as good as `r`, and +19.6% is not a large margin.

---

## 6b. Which tokens go through the GPU — the short answer

```
INPUT  to the GPU   720.0 B tokens  (360.0 B source × 2 passes)
OUTPUT from the GPU ~261   B tokens  (measured r, per prompt, per arm)
KEPT for training    150   B tokens  (5 arms × 30B, after trimming to budget)

NEVER touches a GPU  70.0 B tokens
    shared-core    20.0 B   raw, carried into training by all five arms
    quality-base   50.0 B   raw control, never uploaded, local only

FINAL TRAINING MIX   50B per arm × 6 arms (incl. quality-base) = 300B
```

`configs/data.yaml` now states this explicitly: each arm carries its `docs`,
`source_tokens_llama2`, measured `r` and `est_output_tokens`, and the raw side is written out in
the header comment. A reader can see the GPU/no-GPU split without opening a Python file.

---

## 7. What still blocks a one-button run

| # | item | owner | blocks |
|---|---|---|---|
| B1 | `wytro/Know-Your-Sources-7B` is `gated: manual`. Tianjian's HF account must be granted access, or the repo ungated. | **Wytro** | everything — download is step 2 |
| ~~B2~~ | ~~`upload.repo_template` — the output HF org/name pattern is still a `WYTRO` placeholder~~ | — | **CLOSED (round 7)** — upload is out of scope. `upload.enabled: false` ships as the default and the placeholder is deleted rather than replaced with a plausible value. See the correction below. |
| ~~B3~~ | ~~`HF_TOKEN_WRITE` — write-scoped token for the output org~~ | — | **CLOSED (round 7)** — not needed while upload is disabled |
| B4 | 11 blanks in `configs/cluster.yaml` + `HF_TOKEN` in `.env` | **Tianjian** | everything |
| B5 | Hub card has no `configs:` block. Not required — the access layer uses explicit globs — but it makes the dataset usable by anyone else. Draft ready at `data_reports/DATASET_CARD_DRAFT.md`. | Wytro | nothing |
| ~~B6~~ | ~~`blab-jhu/KYS-Configs` access needed to diff the templates~~ | — | **CLOSED** — resolved without it, via git blob OIDs (§9) |
| ~~B7~~ | ~~distill template may differ from the original~~ | — | **CLOSED** — all six templates verified byte-identical (§9) |


**Correction to what B2 and B3 said (round 7).** Both were described as blocking "upload
only (step 6)". That was wrong, and the error mattered: `load_config` runs
`assert_no_placeholders` over `configs/data.yaml` on *every* entry point, so the unfilled
`upload.repo_template` marker actually killed `01_download_model.py`,
`02_download_data.py`, `04_postprocess.py`, `06_calibrate.py` and `run_rewrite.py` as well
— and in `preflight.py` the `load_config` call sits outside the per-check `try`, so it
aborted the whole preflight at check 2 with exit 2, no summary and no remaining checks.
`--skip-upload` could not get past it either. A blank that was documented as affecting the
last step in fact blocked the first. Deleting it removes the blocker outright.

B1 remains the most likely day-zero failure, so **preflight checks it directly**: it fetches a
real file rather than reading metadata, because listing a gated repo succeeds without access. No
existing gate was weakened to smooth the one-button path.

**B6 and B7 were raised and closed in round 5.** All six prompt templates in `prompts/` are
byte-identical to the published originals; the one apparent discrepancy is a drifted copy on the
publishing side, not here (§9). Nothing was changed. The residual, non-blocking item is that
`KYS-Configs` ships the distill prompt twice with different bytes — worth fixing there.

### Placeholders resolved this round

The five arm `repo_id` placeholders collapse into a single `hf.repo_id` +
`revision` pinned to `6e18cda64fbd24fe46010b2aa578f14a4255076d`, both read from
`upload_blocks.py:67` and confirmed against the Hub. The sixth — `quality-base`'s — is deleted,
because that block was never uploaded and never will be.

---

## 8. Where the round-4 brief was wrong

The brief was second-hand and said so. These are its errors.

1. **"per-block source budgets of roughly 50B for quality-base, 80B for quality-first /
   diversity-oriented / disagreement-aware / wrap-inspired, and 140B for rewire-inspired"** —
   the 80B and 140B figures are **block totals, not source budgets**. `select_600m.py:77-83`:
   `REM_TARGET = 60_000_000_000` and `REWIRE_REM_TARGET = 120_000_000_000`. A block's total is
   `20B core + remainder`. Taking 80B and 140B as rewrite budgets would have overstated the GPU
   workload by 33% and 17%. The 20B core and 50B quality-base figures were correct.

2. **"is the raw core physically separate … or a column that marks it?"** — implied one or the
   other. It is **both, plus a third marker**: separate `shared-core/` folder on the Hub, an
   `is_core` bool column in every parquet, and a `remainder_doc_ids.npy` sidecar per block.

3. **"does `quality-base` still get zero rewrite passes?"** — yes, but the more useful fact is
   that **`quality-base` is not on the Hub at all** and never will be
   (`upload_blocks.py:15-16`). The repo's `quality-base` arm therefore had a `repo_id`
   placeholder that could never have been filled. Nothing in the brief hinted at this.

4. **"confirm `prompts/wrap-inspired/p1..p4` are the easy/hard/wiki/qa set … If they are the
   wrong set, that is a corpus-wide silent corruption"** — they are the correct set, verified by
   md5 against the source files. The feared corruption did not occur.

5. **"wrap-inspired becomes 2 passes, not 4, so the run is 10 jobs, not 12"** — the conclusion
   is right, the arithmetic implied is not. wrap does not simply drop two jobs: it drops **three**
   style passes and **gains a distill pass it never had**. The repo's `wrap-inspired` had no
   distill prompt at all, which is a divergence from the source the brief did not mention and
   which `HANDOFF_REVIEW.md` C1 had recorded as deliberate.

6. **"the 1.5B-era outputs have both input and output token counts recorded"** — the row-level
   outputs that had both **no longer exist**. What survives is arm-level census counters and a
   seeded ~36k-row sampled monitor. The ratio was still measurable, and cross-validated, but not
   from the source the brief assumed. See §5.

7. **"a shared core around 20B tokens"** — correct, but the brief did not mention that the core
   is *identical* across the five carrying blocks and provably so. That is what makes "rewriting
   it once per arm would be redundant" true rather than merely plausible, and it is asserted in
   code at `select_600m.py:631-637`.

8. **Not an error, but absent from the brief:** the dataset is **gated**, and its card has **no
   `configs:` block**. Both directly affect the one-button goal (§7 B1, B5). Neither is
   discoverable from the upload script alone; both required looking at the Hub.

### A claim in this repo's own docs that turned out to be wrong

Not from the brief, but found while acting on it, and worth recording because it had
already been used to justify deleting data:

`docs/POSTPROCESSING.md` stated that the source's wrap trim *"dispatched **per row** on a
`wrap_style` column"*, and used that as the reason the column could disappear once each
style became its own job. **The source does no such thing.** Every style goes through the
identical rule — `01_strip_prefix_wrap.py:185-188` branches only on `distill` vs
`rewritten`, which is job-level. `strip_preamble` is deliberately style-agnostic; its
`qa`-safety comes from `STRICT_META`, not from knowing the row's style.

What `wrap_style` actually drove in the source was the **statistics**: per-style openings,
`status==2` counts, stripped counts and token totals
(`01_strip_prefix_wrap.py:151-152,164-181`). That is what round 4 reproduces —
`trim_job` now emits a `by_wrap_style` block with per-style doc share, token share and
tokens/doc. The trim rule itself is untouched, and the parity harness still matches the
source byte-for-byte over 72,443 comparisons.

### One thing the brief was right about that looked wrong at first

The brief's byte-level intuition that block folders hold only part of the data was correct, but
it could not be confirmed from the Hub alone: docs-per-file, MB-per-file and KB-per-doc are all
non-constant across folders and are consistent with *both* "remainder only" and "core +
remainder concatenated". It was settled from `upload_blocks.py:222-223`, not from arithmetic.
Recorded here because the arithmetic route looked convincing and was not.

---

## 9. Prompt provenance: byte comparison against the published originals — **RESOLVED, all six verified**

The authoritative prompt files were published at
`https://huggingface.co/datasets/blab-jhu/KYS-Configs` (sha `d094c621…`). Four of this
repo's six templates — the distill prompt and the four wrap styles — were **reconstructed**
from logged chat-templated inputs rather than copied, so a direct byte comparison is
decisive where the token-overhead fingerprint was only suggestive.

**Outcome: all six templates in `prompts/` are byte-identical to the originals. Nothing was
changed.** One published file disagrees with the rest of the evidence, and the finding is
that *it* drifted, not this repo — see "the distill discrepancy" below.

`KYS-Configs` is `gated: manual` and its `prompts/` files 401 unauthenticated, so the
comparison used **git blob SHA-1 OIDs** from the Hub tree API — SHA-1 over exact file bytes
(`sha1("blob <len>\0" + content)`), published even when content is not. That is byte-exact,
not a proxy. Trailing whitespace and final newlines are included.

### Results

| template | bytes | ours (md5) | verdict |
|---|---:|---|---|
| wiki-grounded | 597 | `bca104fe6e298615e5ccb9c9c747073b` | **IDENTICAL** — git OID `802aff3d…` matches published |
| distill | 842 | `538700534e99d5e80b268fd9b2408b48` | **IDENTICAL** to `finephrase/nemotron/distill.md` (`200cd2c3…`) |
| wrap `easy` | 218 | `0735f53aca80cadaa8d67727680dbbfd` | **IDENTICAL** |
| wrap `hard` | 197 | `e99a613bcd4146416428d576af6f200a` | **IDENTICAL** |
| wrap `wiki` | 231 | `cec46736de0229e6d7a0f022cd2e661a` | **IDENTICAL** |
| wrap `qa` | 248 | `733fbeea43050cb4a4e27f9384b9014e` | **IDENTICAL** |

**Method note, recorded because the first attempt got it wrong.** The four wrap originals live
as *values inside* a single `wrap_prompts.json`, not as separate files. Comparing the JSON
*container* to our four `.txt` files cannot work — 18,432 candidate serialisations were tried
and none matched, which proved nothing. The correct comparison is per value: extract each JSON
value and compare it to the corresponding `.txt`. Done against the source-local
`prompts/wrap_prompts.json`, all four are byte-identical, and its key order is
`easy, hard, wiki, qa` — the order that is part of the style seed.

(The two `wrap_prompts.json` *files* do differ — source-local is 962 B `a9242e71…`, published
is 972 B `07930c8e…` — but that is JSON formatting only. The four values are identical, and
this repo stores the values, not the container.)

### The distill discrepancy: the published `distill_prompt.txt` is the outlier

`KYS-Configs` ships the distill prompt **twice, with different bytes**:

| path | bytes | git OID | |
|---|---:|---|---|
| `prompts/distill_prompt.txt` | 841 | `38da40ce870e392ee23b8f27d9c61b3a358d3047` | no trailing newline |
| `prompts/finephrase/nemotron/distill.md` | 842 | `200cd2c3f6782d9fcfada993054fbdf1a4091a57` | **= ours exactly** |

The difference is a single trailing newline. Three independent artifacts carry the 842-byte
form and one carries the 841-byte form:

1. **Our reconstruction** — 842 B, `200cd2c3…`
2. **The source's own surviving copy** — `projects/rewrite/prompts/distill/distill_prompt.txt`,
   842 B, `200cd2c3…`
3. **The published `finephrase/nemotron/distill.md`** — 842 B, `200cd2c3…`
4. …against `prompts/distill_prompt.txt` — 841 B, `38da40ce…`

**Was the trailing newline a log artefact?** This was the live question, since the
reconstruction came from logged `templated_input` fields and a logger that appends a newline
per record would manufacture exactly this. **It does not.** Every one of the 14 bake-off
prompt templates was recovered from its own logs — take the logged chat-templated string, strip
the chat wrapper, substitute the logged `doc_text` back to `[TEXT]` — and compared to its
published `finephrase/**` file:

```
14 / 14 recovered templates reproduce the published file byte-for-byte (git OID match)
    format/{article,commentary,discussion,explanation,faq,narrative,table,tutorial}
    nemotron/{distill,diverse_qa_pairs,extract_knowledge,knowledge_list,wikipedia_style_rephrasing}
    rewire/guided_rewrite_improved
```

If the logger appended a newline, all 14 recoveries would be one byte long and none would
match. All 14 match. So the `\n` before `<|im_end|>` in the distill log is real template
content, and `nemotron_distill` recovers to 842 B / `200cd2c3…` — our file.

**Conclusion: nothing in this repo changes.** The 842-byte form is what the bake-off executed,
what the source repo kept, and what the published `finephrase/` copy contains. The published
`prompts/distill_prompt.txt` appears to have lost its final newline somewhere in packaging;
worth correcting there, but it is not this pipeline's problem.

**Honest limit on that conclusion.** `09_Distill/launch_dataset.sh:20` shows production loaded
`/scratch/.../data_rewrite/prompts/distill_prompt.txt`, and that tree no longer exists — so
there is no *direct* production-side byte evidence, only the bake-off run plus the two
surviving copies. All three agree, and no artifact anywhere carries the 841-byte form except
the one published file, but the production file itself is gone and cannot be checked.

### What this means for the overhead assertions

They are kept as the cheap runtime guard, and their expected values are now known to be derived
from the originals, because **our files are the originals**. Measured against the real
Qwen2.5-7B-Instruct tokenizer: `p1_wiki` 150, `p2_distill` 185, `wrap_easy` 72, `wrap_hard` 66,
`wrap_wiki` 73, `wrap_qa` 83 — all six match `configs/data.yaml`.

**But do not read an unchanged overhead as proof a template is unchanged.** The 841/842 pair
was the counter-example: it produces the **same overhead (185) and the same total length**,
differing at exactly one token position — Qwen merges `.` + `\n` into a single token `'.\n'`
(id 624) where the alternative is `'.'` (id 13). The count is preserved while the identity is
not. A token count is a smoke test; the byte comparison is the proof.

---

## 10. Open questions

1. **The ReWire filter is not implemented — and it is a fixed-BUDGET cut, not "the top half".**

   `rewire-inspired` gets κ=2 (120B rather than 60B) to buy headroom for a post-rewrite
   filter: score the *rewritten* text with fastText, sort descending, and fill to the
   arm's token budget. **This repo has no such stage**, so it will ship ~99.5B output
   tokens unfiltered. *Owner: Wytro (downstream). Blocks the run: no. Blocks rewire's
   training mix: yes.* Out of scope by decision; recorded here so whoever runs it has the
   numbers.

   **Two corrections to how this was previously written down.**

   *(a) "Keep the top half" was wrong, in this document and in `configs/data.yaml`.* The
   1.5B run kept **30.8%**, not 50%: `_step4_rewrite_summary.json` records
   `pool_tokens 16,221,811,013 → kept_tokens 5,000,000,351`.

   *(b) More importantly, it is not a fraction at all, and not a fixed threshold either.*
   The same file states the mechanism outright:

   > `pipeline`: "rewire-inspired: rewrite broadly → fasttext score rewritten → **keep top 5B**"
   > `selection_note`: "…kept set is the **token-budget-matched top-5B (NOT the paper top-10%)**"
   > `target: 5000000000`, `overshoot: 351`, `filled: true`

   The score cutoff `0.1145634651184082` is the **output** of filling to a 5B target, not
   an input to the filter. That distinction matters, because it removes the failure mode
   the tightest reading of this suggested — that a fixed threshold might retain a smaller
   fraction at this scale and leave the arm short. A budget-fill cannot come up short
   while the pool exceeds the budget; what moves instead is the *cutoff score*, i.e. the
   **quality** of the kept set.

   **The headroom is not tight.** Required retention at 600M is `30B / 99.46B = 30.2%`,
   against `30.8%` realized at 1.5B — the same, because κ=2 was sized precisely so the
   pool-to-budget ratio carries over (3.32× here, 3.24× there). By §6's table
   `rewire-inspired` has **+231% headroom, the roomiest of the five arms**; the tightest
   is `quality-first` at +19.6%.

   **Recommendation, not a decision:** implement it as a fixed-budget fill
   (`fill_to(order_by_score_desc, tokens, 30e9)`), matching the 1.5B code path, rather
   than porting `0.11456` as a constant. A budget fill is self-correcting if the score
   distribution shifts; a fixed threshold is not, and the 600M rewritten text is a
   different population (see item 4). Whichever is chosen, record the realized cutoff and
   the retained fraction the way `_step4` did, so the next scale-up has the same evidence
   this one had.

2. **doc_id disjointness across arms is unverified.** Remainders are core-excluded but drawn
   from the same 577M pool by different criteria, so overlap between arms is expected and
   harmless for this pipeline (the join key is `(doc_id, arm, prompt_id)`). But nobody has
   measured how much. *Owner: Wytro, upstream at `13_600M/02_select`*, where the
   `remainder_doc_ids.npy` arrays are local and a pairwise intersection is a few seconds of
   numpy. Deliberately **not** added here: re-downloading 300M ids to check something the
   selection stage can check for free would be the wrong place for it.

3. **`r` is transferred across document populations, and 261.49B is quoted more precisely
   than that transfer supports.**

   `r` was measured on the 1.5B corpora and applied to the round-4 remainders. These are
   not the same population, in two ways that are easy to state and hard to quantify:

   - **Selection differs.** The 1.5B `quality-first` was the top of the *full* pool. The
     round-4 remainder is **core-excluded** — drawn from below the top 20B — so it is
     lower-quality text by the same fastText ranking that defined the core.
   - **Length differs, and `r` is not length-invariant.** Source documents run ~1,600
     tok/doc in `quality-first` against ~949 in `wrap-inspired` and `rewire-inspired`
     (`source_tokens_llama2 / docs`). Both rewrite prompts have a roughly fixed-cost
     framing and a length-dependent body, and the distill prompt explicitly *condenses*,
     so a shorter input does not simply scale its output down proportionally.

   **Direction of the bias, as far as it can be reasoned:** core-exclusion should push `r`
   **up** slightly for the four 60B arms. The distill prompt condenses proportionally more
   from dense, information-rich text, and the core skimmed exactly that text off the top;
   what remains is more repetitive and web-like, which tends to compress *less* per input
   token under a "preserve the facts" instruction. That argues the estimates are mildly
   conservative. This is an argument, not a measurement, and it is not strong enough to
   lean on.

   **Consequence for precision.** 261.49B is five significant figures on a quantity whose
   dominant error term is an unquantified population transfer. Treat it as **~260B ± 10%**.
   The per-arm figures inherit the same band. Nothing downstream breaks — the disk estimate
   has slack and every arm's headroom in §6 exceeds 19% — but the two should be read
   together: **`quality-first`'s +19.6% headroom sits inside this uncertainty band**, so if
   `r` comes in ~16% below the transferred value that arm goes short of its 30B and would
   need regenerating. That is the single most consequential open number in this document.

   **This settles empirically on day one.** `scripts/06_calibrate.py` already
   cross-checks measured tok/doc against the config's predicted value for the first job and
   flags a disagreement beyond 2×. Tighten that comparison once real shards exist: after
   the first few hundred shards of `quality-first__p2`, `r` is known to well within a
   percent and every estimate here can be replaced with a measurement.

4. **`gpu_memory_utilization` disagreement in the source.** `07_rewrite/README.md:21` and the
   argparse default say `0.90`; the sbatch that actually ran passes `0.85`
   (`07_rewrite/sbatch_template.sh:50-65`). This repo uses the value in `configs/vllm.yaml` and
   round 3 verified it; flagging only because the source is genuinely ambiguous and someone may
   later cite the README.

5. **Style assignment cannot reproduce the 1.5B draw**, because shard boundaries differ (§2).
   If bitwise reproduction of the original wrap assignment ever matters, it would require
   carrying the source's shard boundaries, which are not preserved in the uploaded data.
   Assumed not to matter: this is a new corpus at a new scale.

6. **`quality-base` has no home in this pipeline.** It is 50B raw tokens, 37.3M documents, and
   exists only at `01_before_rw/quality-base/`. Whoever assembles the final training mixes needs
   it; nothing in `rewrite-vllm` will produce or move it.
