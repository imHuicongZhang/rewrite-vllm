#!/usr/bin/env python3
"""Wrap style assignment: determinism, resume-safety, and a golden vector.

The whole point of restoring assign_wrap_styles is that a worker restarting shard N must
give the same document the same style. If that ever silently stops being true, a resumed
run produces a corpus where some documents were rewritten under one style before the crash
and another after it -- and nothing downstream would notice, because every row still has a
plausible wrap_style value.

So this file tests the property directly rather than trusting the implementation to be
obviously correct.

    python tests/test_wrap_styles.py

Needs numpy only. Exits 0 on success, 1 on any failure.
"""
from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np                                              # noqa: E402

from rewrite.wrap_styles import (BASE_SEED, WRAP_STYLES,        # noqa: E402
                                 assign_wrap_styles)

FAILS = []


def expect(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def hdr(t):
    print(f"\n--- {t} " + "-" * max(0, 66 - len(t)))


# ============================================================ 1. the style set
hdr("1. the style set")
expect(WRAP_STYLES == ["easy", "hard", "wiki", "qa"],
       f"WRAP_STYLES is easy/hard/wiki/qa in seed order (got {WRAP_STYLES})")
expect("medium" not in WRAP_STYLES,
       "'medium' is absent -- that key belongs to the ABANDONED paper-verbatim set "
       "(06_vllm/wrap_styles_sample.py), which never ran in production")
expect(BASE_SEED == 42, f"base seed is 42 (got {BASE_SEED})")

# ============================================================ 2. golden vector
hdr("2. golden vector -- catches a numpy PCG64 stream change")
# Computed here from numpy directly, exactly as 07_rewrite/rewrite_worker.py:54-62 does.
# If a numpy upgrade changes the PCG64 stream, these stop matching and the whole corpus
# would be re-rolled silently on the next run. The source pins numpy 1.26.4 for the same
# reason in its shuffle (run_shuffle_10B_base.sh:17-18).
for si in (0, 1, 7, 12345):
    ref = [WRAP_STYLES[i] for i in
           np.random.default_rng([42, si]).integers(0, len(WRAP_STYLES), size=16)]
    expect(assign_wrap_styles(si, 16) == ref,
           f"shard {si}: matches np.random.default_rng([42, {si}]).integers(0, 4, 16)")

# Frozen literals, so this test still fails loudly if BOTH the module and the reference
# above are changed together.
FROZEN = {
    0: ['easy', 'qa', 'wiki', 'hard', 'hard', 'qa', 'easy', 'wiki',
        'easy', 'easy', 'wiki', 'qa', 'wiki', 'qa', 'wiki', 'qa'],
    7: ['hard', 'wiki', 'hard', 'easy', 'hard', 'wiki', 'wiki', 'qa',
        'hard', 'qa', 'easy', 'easy', 'easy', 'hard', 'wiki', 'easy'],
}
for si, want in FROZEN.items():
    expect(assign_wrap_styles(si, 16) == want,
           f"shard {si}: matches the frozen 16-element vector recorded in this test")

# ============================================================ 3. determinism
hdr("3. determinism within a process")
a = assign_wrap_styles(4242, 5000)
expect(a == assign_wrap_styles(4242, 5000), "same (shard, n_rows) -> identical result")
expect(assign_wrap_styles(4242, 5000) != assign_wrap_styles(4243, 5000),
       "different shards -> different assignments (the seed actually varies)")

# ============================================================ 4. resume safety
hdr("4. resume safety")
# A worker that dies partway through shard N re-runs the WHOLE shard on restart. The
# property that matters is that re-running from scratch reproduces the same vector -- and,
# as a stronger guarantee, that a shorter draw is a prefix of a longer one, so the result
# does not depend on how many rows the shard turned out to have.
full = assign_wrap_styles(999, 5000)
expect(assign_wrap_styles(999, 3000) == full[:3000],
       "a 3,000-row draw is a prefix of the 5,000-row draw (no length dependence)")
expect(assign_wrap_styles(999, 0) == [], "a 0-row shard yields an empty assignment")
expect(assign_wrap_styles(999, 1) == full[:1], "a 1-row shard matches the first element")

# independent of worker identity / claim order: there is no worker argument at all
import inspect                                                   # noqa: E402
params = list(inspect.signature(assign_wrap_styles).parameters)
expect(params == ["shard_index", "n_rows", "base_seed"],
       f"signature takes no worker id, host or claim order (got {params})")

# ============================================================ 5. across processes
hdr("5. across processes -- a real restart, not a simulated one")
code = ("import sys; sys.path.insert(0, %r);"
        "from rewrite.wrap_styles import assign_wrap_styles;"
        "print(','.join(assign_wrap_styles(31337, 64)))" % str(REPO / "src"))
outs = set()
for _ in range(2):
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
    outs.add(r.stdout.strip())
expect(len(outs) == 1 and outs != {""},
       "two separate interpreter processes produce the identical assignment")
expect(outs.pop() == ",".join(assign_wrap_styles(31337, 64)),
       "and it matches this process")

# ============================================================ 6. balance
hdr("6. uniformity -- documents balanced, per the source's 25.0% +/- 0.1pp")
N = 4_000_000
c = Counter(assign_wrap_styles(3, N))
expect(set(c) == set(WRAP_STYLES), "all four styles appear")
for st in WRAP_STYLES:
    pct = 100.0 * c[st] / N
    expect(abs(pct - 25.0) < 0.1, f"{st:5s} {pct:.3f}% is within 0.1pp of 25%")

# Across many small shards too -- the real run draws per shard, not once for the corpus.
c2 = Counter()
for si in range(2000):
    c2.update(assign_wrap_styles(si, 5000))
tot = sum(c2.values())
for st in WRAP_STYLES:
    pct = 100.0 * c2[st] / tot
    expect(abs(pct - 25.0) < 0.1,
           f"{st:5s} {pct:.3f}% across 2,000 shards of 5,000 rows")

# ============================================================ 7. config wiring
hdr("7. the shipped config uses this, in this order")
import yaml                                                      # noqa: E402
data = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())
wm = [d for d in data["prompt_defs"].values() if d.get("mode") == "wrap_multi"]
expect(len(wm) == 1, f"exactly one wrap_multi prompt_def (got {len(wm)})")
if wm:
    names = [st["style"] for st in wm[0]["styles"]]
    expect(names == WRAP_STYLES,
           f"configs/data.yaml style order {names} == WRAP_STYLES {WRAP_STYLES} "
           "(the order is part of the seed)")
    for st in wm[0]["styles"]:
        expect((REPO / st["file"]).exists(), f"{st['file']} exists")

wrap_arms = [a for a in data["arms"] if a["name"] == "wrap-inspired"]
expect(len(wrap_arms) == 1 and len(wrap_arms[0]["prompts"]) == 2,
       "wrap-inspired has exactly 2 prompts: the styled pass + the shared distill pass")

print(f"\nchecks failed: {len(FAILS)}")
for f in FAILS:
    print("  - " + f)
sys.exit(1 if FAILS else 0)
