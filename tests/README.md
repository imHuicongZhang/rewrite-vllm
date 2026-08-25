# tests

Four checks. Together they are the evidence that this repo faithfully reproduces a source
pipeline that neither Tianjian nor any reviewer can see — and which may not outlive the
project. That is why they are committed rather than run once and reported.

| test | proves | needs |
|---|---|---|
| `test_trim_parity.py` | the trim rules are byte-identical to the original's, on real model outputs | `--source-root` |
| `test_shuffle_parity.py` | the bucketed shuffle is the original one, in text *and* in row ordering | `--source-root` |
| `../scripts/verify_prompt_parity.py` | the twelve prompt files are the original prompts | a local Qwen2.5-7B-Instruct dir |
| `test_integration.py` | the whole pipeline works end to end and every invariant holds | nothing |

**No path to the source pipeline is hard-coded anywhere in this repository.** The two
parity tests take `--source-root` and exit **77** with an explanation when it is absent,
so they can be committed without pinning this repo to one machine.

## Running them

```bash
# needs the ORIGINAL pipeline on disk -- normally only the cluster it was written on
python tests/test_trim_parity.py    --source-root /path/to/rewrite
python tests/test_shuffle_parity.py --source-root /path/to/rewrite

# needs only a local model directory; no GPU, no vLLM, no config
python scripts/verify_prompt_parity.py --model /path/to/Qwen2.5-7B-Instruct

# needs nothing at all
python tests/test_integration.py
```

Exit codes: `0` pass, `1` fail, `77` skipped because the source tree is not present.

## What each one actually does

### `test_trim_parity.py` — the important one

Imports the original `pp_io.py` and `01_strip_prefix_wrap.py` **directly** and diffs their
output against ours, document by document. The corpus is real: model outputs harvested from
the original run's own monitor logs (`00_TMP/rewriting_monitor.md` and
`rewriting_monitor_distill.md`, ~36,000 generations), plus ~21 adversarial cases covering
branches real data may not reach — content headers that must *survive* the strip, a leak
with no blank line, `Q:` openings, an over-long first paragraph.

Last run: **72,443 comparisons, 0 mismatches**, all 9 constants equal.

It compares every constant too, because a single changed entry in `OPENERS`,
`SIGNAL_WORDS` or `STRICT_META` would alter what gets stripped without ever raising.

### `test_shuffle_parity.py`

Two independent checks. First, a source-text comparison of `choose_buckets` and
`bucketed_shuffle` against the originals with comments and whitespace normalised away.
Second, execution: both implementations run over the same synthetic input and must agree on
bucket count, output shard names, and **row order** — not merely row counts, since a
shuffle that preserved counts but reordered differently would still be wrong.

Last run: identical on all five checks over 63,000 rows.

### `scripts/verify_prompt_parity.py`

Four of the six distinct prompt texts were **reconstructed**, not copied — their source
files no longer exist. This renders each prompt through the chat template with an empty
document and counts tokens, which pins down the prompt text, the template and the tokenizer
in one integer. The distill prompt's expected value, **185**, is not a number anyone chose:
it is what the original pipeline printed in its own production log.

Expected values are read from `configs/data.yaml`, so this cannot drift from what the
workers assert at runtime.

### `test_integration.py`

Stubs vLLM and the tokenizers, builds a synthetic corpus, and drives all 10 jobs across
several workers. Around 100 assertions covering: config validation and the engine/sampling
whitelists; sharding and the manifest; row conservation per job, deep-verified by counting
lines on disk; **every prompt covering its arm's entire corpus**; status-0 rows emitted
rather than dropped; resume via `.done` sidecars; stale-`.tmp` cleanup; the manifest
fingerprint interlock; deliberate corruption being detected; trim touching only `status==2`
rows; shuffle scoped to one (arm, prompt); dynamic claiming under contention on a simulated
heterogeneous fleet; and a regression for each defect found in review round 2.

It needs no GPU and no source tree, so it is the one to run after any change here.

## If a parity test fails

Do not "fix" it by editing the expected value or relaxing the comparison. These tests exist
to detect exactly the kind of silent divergence that would otherwise be discovered months
later in the training results. Find out what changed, and if the answer is not obvious,
ask before proceeding.
