# POSTPROCESSING

Every trim rule in `src/rewrite/postprocess.py`, with the source line it came from.

All rules are ported **verbatim**. Where the original kept a rule arm-specific, it stays
arm-specific; nothing is unified, simplified, or "cleaned up". Two source scripts
(`01_strip_prefix_diversity.py`, `01_strip_prefix_rewrite.py`) were dropped as duplicates
only after confirming their rules are byte-identical to `01_strip_prefix.py` — they differ
solely in report-only diagnostics.

**Verification.** The ported rules were tested against the original functions, imported
directly from the source tree, on **36,190 real model outputs** harvested from
`00_TMP/rewriting_monitor.md` (18,190) and `00_TMP/rewriting_monitor_distill.md` (18,000),
plus 21 hand-built adversarial cases covering branches real data may not reach.
**72,443 comparisons, zero mismatches**, including every constant.

---

## Which rule applies to which job

The rule is a property of the **prompt**, not the arm — which is why `trim` is a field on
each prompt in `configs/data.yaml` rather than a branch in the code.

| arm | prompt | prompt text | rule | original script |
|---|---|---|---|---|
| quality-first | p1 | wiki-style rephrasing (grounded) | `wiki` | `01_strip_prefix.py` (kind=wiki) |
| quality-first | p2 | distill | `distill` | `01_strip_prefix.py` (kind=distill) |
| diversity-oriented | p1 / p2 | wiki / distill | `wiki` / `distill` | `01_strip_prefix_diversity.py` |
| disagreement-aware | p1 / p2 | wiki / distill | `wiki` / `distill` | `01_strip_prefix.py` |
| rewire-inspired | p1 / p2 | wiki / distill | `wiki` / `distill` | `01_strip_prefix_rewrite.py` |
| wrap-inspired | p1–p4 | easy / hard / wiki / qa | `wrap` (all four) | `01_strip_prefix_wrap.py` |

Eight `wiki`/`distill` jobs plus four `wrap` jobs = 12.

One structural simplification the new semantics allow: the original `wrap` trim dispatched
**per row** on a `wrap_style` column, because each document had been assigned one of four
styles. Here each style is its own job, so the rule is uniform within a shard and
`wrap_style` disappears — its information is now `prompt_id`. The rule function itself is
unchanged; `strip_preamble` was already style-agnostic.

---

## Three invariants that hold for every rule

Each is easy to lose in a port, and each is asserted or structurally enforced.

1. **Only `status == 2` rows are examined.**
   Source: `01_strip_prefix.py:102` — `s2 = np.flatnonzero(status == 2)`.
   Status-0 rows are empty (the document was too long and was never generated).
   Status-1 rows were truncated at the output cap and the original deliberately left them
   alone — so a truncated row **keeps** its artifact prefix. That is correct behaviour,
   not a bug, and the integration test asserts it explicitly.

2. **Llama-2 token counts are recomputed only for rows the trim actually changed.**
   Source: `01_strip_prefix.py:123-125`. Unchanged rows keep the worker's original count
   bit-for-bit, so postprocessing cannot perturb token accounting on the 99%+ of the
   corpus it does not touch.

3. **The row count must not change.**
   Source: `01_strip_prefix.py:132-133` — `RuntimeError(... 'rowcount changed on rewrite')`.
   Trimming rewrites the shard in place via `.tmp` + `os.replace`, then re-counts.

---

## Shared rules

Both live in the original's `pp_io.py` and are used by every arm.

### `strip_instruction_leak` — source `pp_io.py:80-90`

```python
INSTRUCTION_LEAK_ANCHOR = (
    "Important: Do not add any information, claims, or details that are not")

def strip_instruction_leak(text):
    if text and text.startswith(INSTRUCTION_LEAK_ANCHOR):
        cut = text.find('\n\n')
        return (text[cut + 2:], True) if cut >= 0 else ('', True)
    return text, False
```

Removes the grounded prompt's own instruction paragraph when the model echoed it back at
the very start of its output. **Start-anchored only** — a mid-text occurrence is genuine
content and is left alone. This is the only rule that can empty a document: if the whole
output is the leak with no `\n\n`, it returns `''`.

### `strip_distill_preamble` — source `pp_io.py:93-128`

```python
DISTILL_MAX_PREAMBLE_CHARS = 120
DISTILL_PREAMBLE_WORDS = ('paraphras', 'condensed', 'rewrit', 'summary', 'version')
DISTILL_OPENERS = ('here is', "here's", 'here are', 'certainly', 'sure', 'below is',
                   'of course', 'the following is')
```

Two branches, both start-anchored:

* **(a) first paragraph** (up to `\n\n`, within 120 chars) is a meta-preamble if it is a
  markdown header (`###`/`##`/`**`) **and** contains a preamble word, **or** starts with an
  opener **and** contains a preamble word, **or** starts with `paraphrased` and ends `:`.
  Remove it plus its `\n\n`.
* **(b) first line** (up to `\n`, within 120 chars) ending in `:`, containing a preamble
  word, and starting with a header / opener / `paraphrased`. Remove that line plus break.

Never `.replace` or `.lstrip` on the body. A document that opens directly with content is
untouched.

---

## `wiki` rule — source `01_strip_prefix.py:108-118`

```python
WIKI_PREFIX = "Here is a paraphrased version:\n\n"   # exact, case-sensitive, start-anchored only

def trim_wiki(text):
    s, did = text, False
    if s is not None and s.startswith(WIKI_PREFIX):
        s = s[len(WIKI_PREFIX):]; did = True
    s, did_leak = strip_instruction_leak(s)
    return s, (did or did_leak)
```

The wiki prompt literally instructs the model to *"Begin your answer on a separate line
with «Here is a paraphrased version:»"*, so essentially every output carries it. Removal
is an exact slice — never `.replace`, never `.lstrip`.

Note the `s is not None` guard: this path does **not** normalise `None` to `''`, whereas
the `wrap` path does. That asymmetry is in the original and is preserved.

## `distill` rule — source `01_strip_prefix.py:119-123`

`strip_distill_preamble` alone. No exact-prefix step: the distill prompt does not instruct
a fixed opening line, so there is nothing to slice — only free-form preambles to detect.
No instruction-leak strip either, matching the original.

## `wrap` rule — source `01_strip_prefix_wrap.py:104-131`, dispatched at `:183-194`

`MAX_PREAMBLE_CHARS = 300`, plus three tuples: 26 `OPENERS`, 19 `SIGNAL_WORDS`,
12 `STRICT_META`.

```python
def strip_preamble(text):
    if not text: return text, False
    nl = text.find('\n\n')
    if nl < 0 or nl > MAX_PREAMBLE_CHARS: return text, False
    head = text[:nl]; low = head.lower()
    # (a) case-sensitive opener + any signal word
    if head.startswith(OPENERS) and any(k in low for k in SIGNAL_WORDS):
        return text[nl + 2:], True
    # (b) markdown/bold header or bare label, gated on a STRICT_META phrase
    if (head.lstrip().startswith(('###', '##', '**'))
            or low.startswith('paraphrased') or low.startswith('rewritten passage')) \
            and any(s in low for s in STRICT_META):
        return text[nl + 2:], True
    return text, False
```

Then `strip_instruction_leak`. Dispatch normalises `None` to `''` first
(`s = rewritten[j] or ''`, `01_strip_prefix_wrap.py:180`) — unlike the wiki path.

The two-tuple design is the subtle part, and the reason `STRICT_META` exists separately
from `SIGNAL_WORDS`. Branch (a) needs a meta-opener *and* a signal word, so
`"Here is a list of ingredients\n\n..."` survives. Branch (b) is gated on the **stronger**
`STRICT_META` set, which deliberately excludes generic content words like *summary*,
*question*, *passage* and *style*, so genuine content headers survive:

* `### Frequently Asked Questions` — kept (no STRICT_META phrase)
* `### Case Summary` — kept
* `### Passage` — kept
* `### Paraphrased Text` — stripped
* `Rewritten Passage:` — stripped

`qa`-style outputs opening with `Q:` never match either branch, which is what keeps the
`wrap-inspired` p4 pass intact.

---

## Expected strip rates

From the original run's `_step1_*_summary.json`, useful as a sanity range when validating:

| rule | observed strip rate |
|---|---|
| `wiki` | 0.0007% – 0.17% of status-2 rows |
| `distill` | 0.0066% – 0.56% |
| `wrap` | 1 – 6,755 rows per style, out of millions |

These are low because the prompts are explicit about not adding preamble; the rules exist
for the tail that ignores the instruction. `wiki` looks anomalously low in that table only
because that particular run was re-run over already-stripped data — the rules are
idempotent by design.

Check yours before committing to an in-place pass over terabytes:

```bash
python scripts/04_postprocess.py --dry-run --sample 100000 --arm quality-first --prompt-id p1
```

---

## The shuffle

`src/rewrite/shuffle.py` is a byte-for-byte port of `pp_io.py:184-281`, verified two ways:
a line-by-line comparison of the function bodies, and an execution test confirming
identical bucket count, identical output shard names, and **identical row ordering** across
63,000 rows.

* `B = max(16, ceil(2·est / (0.55·mem)))`, `est` = on-disk bytes × 4.0, `mem` capped at
  256 GiB. Bucket count adapts to the machine; the floor is 16.
* **Pass 1** scatters rows into `B` on-disk parquet buckets using one RNG stream
  (`default_rng(42)`) advanced across inputs in sorted path order.
* **Pass 2** reads each bucket, permutes it with `default_rng([42, b])`, emits exact
  500,000-row shards with a carry buffer across bucket boundaries, and unlinks each bucket
  as it is consumed.
* Ends with `written != total_rows -> RuntimeError`.

Memory safety comes from the `B` formula, the 256 GiB clamp, `del t` after each scattered
shard, and deleting buckets as they are consumed. Those on-disk bucket files **are** the
mechanism — that is why the shuffle stays Arrow/parquet internally even though the rewrite
output is JSONL. Only the injected `load_fn` differs: it reads a JSONL shard into a
`pa.Table`.

**Scope: within (arm, prompt) only.** Never across arms, never across prompts within an
arm. `shuffle_job` refuses any other scope, and the integration test asserts every
shuffled shard contains exactly one `arm` and one `prompt_id`.

**Determinism caveat**, inherited from the original: pass-1 bucket ids come from a single
RNG stream advanced over `sorted(specs)`, so identical output requires the same input file
set, the same sort order, **and the same numpy version**. The original pinned an
environment for exactly this reason (`run_shuffle_10B_base.sh:17-18`). Your shuffle will be
deterministic for you, but not bit-identical to the original JHU run — which used numpy
1.26.4, while this stack pins 2.3.5.

**Not resumable mid-pass.** Bucket files are unlinked as consumed, so an interrupted
shuffle restarts that job's shuffle from scratch. That is why `tmp_root` should be fast
local disk. A `_shuffle.done` marker skips completed jobs.
