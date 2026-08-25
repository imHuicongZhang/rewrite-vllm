"""Per-document wrap style assignment, restored verbatim from the source.

The source's wrap arm did NOT run four passes over the corpus. It ran ONE pass and chose
one of four styles per document, seeded so the choice is reproducible. Round 2 of this
package replaced that with four full passes and deleted this function; round 4 restored it
because the four-pass design was superseded. See docs/DESIGN_DELTA.md section 2.

WHY THE SEEDING IS SHAPED THIS WAY
----------------------------------
`np.random.default_rng([base_seed, shard_index])` keys the stream on the shard alone --
never on worker id, wall-clock, claim order, or how many shards this worker has already
processed. Combined with drawing the whole shard in a single `size=n_rows` call (so there
is no partial-consumption state to lose mid-shard), that makes the assignment:

  * worker-independent -- any worker that picks up shard N produces the same styles;
  * resume-safe        -- a crash at row 3,000 of 5,000 re-derives rows 0..4,999
                          identically on restart, because nothing about the first attempt
                          is carried forward;
  * order-independent  -- dynamic shard claiming cannot perturb it.

`shard_index` is fixed by the input manifest, which is fingerprinted, so it cannot drift
under a resume against re-sharded input either -- that case is caught by the fingerprint
check before it can reach this function.

WHAT THIS DOES *NOT* REPRODUCE
------------------------------
This pipeline re-cuts shards at `sharding.shard_target_rows`, so `shard_index` here is
OURS, not the source's. The assignment is reproducible within this pipeline; it is NOT the
same draw the 1.5B run produced, and cannot be -- different corpus, different sharding.
What is preserved is the mechanism and its statistical properties, not the specific draw.

source: 07_rewrite/rewrite_worker.py:39,54-62
"""
from __future__ import annotations

import numpy as np

# The four adapted styles. NOT the paper-verbatim set from arXiv:2401.16380 Appendix G --
# that one had keys easy/medium/hard/qa and was abandoned after a 100-document pilot
# (06_vllm/rewritten_examples/wrap_styles_summary.md). `medium` appears nowhere in
# production. If you ever see `medium` in a style column, something loaded the wrong set.
#
# THE ORDER IS PART OF THE REPRODUCIBLE SEED: the RNG draws an index into this list.
# Reordering it silently changes which document gets which style.
# source: 07_rewrite/rewrite_worker.py:39
WRAP_STYLES = ["easy", "hard", "wiki", "qa"]

BASE_SEED = 42


def assign_wrap_styles(shard_index: int, n_rows: int, base_seed: int = BASE_SEED) -> list:
    """Deterministic per-(shard, row) style assignment, robust to variable shard sizes.

    Seeded only by (base_seed, shard_index) so row i in a given shard ALWAYS maps to the
    same style regardless of which worker runs it or when (resume-safe).

    Reproduced verbatim from 07_rewrite/rewrite_worker.py:54-62. Do not "simplify" this:
    every element of it -- the list-form seed, the single vectorised draw, the index into
    WRAP_STYLES -- is load-bearing for reproducibility.
    """
    if n_rows < 0:
        raise ValueError(f"n_rows must be non-negative, got {n_rows}")
    rng = np.random.default_rng([base_seed, shard_index])
    idx = rng.integers(0, len(WRAP_STYLES), size=n_rows)
    return [WRAP_STYLES[i] for i in idx]
