"""Per-arm trim logic, ported VERBATIM from the source's Step 1.

Sources:
  10_postprocess/pp_io.py:72-128            shared rules (leak strip, distill preamble)
  10_postprocess/01_strip_prefix.py:106-123 wiki + distill dispatch
  10_postprocess/01_strip_prefix_wrap.py:53-131, 183-194   wrap rule + dispatch

Nothing here is unified or "cleaned up". Where the source kept a rule arm-specific, it
stays arm-specific; where two source scripts had byte-identical rules, one copy is used
(01_strip_prefix_diversity.py and 01_strip_prefix_rewrite.py differ from
01_strip_prefix.py only in report-only diagnostics -- see docs/SOURCE_INVENTORY.md 1.3).

Three invariants carried over from the source, each easy to lose in a port:
  1. ONLY status == 2 rows are examined.       (01_strip_prefix.py:102)
  2. Llama-2 token counts are recomputed ONLY for rows the trim actually changed.
                                                (01_strip_prefix.py:123-125)
  3. Row count must not change.                 (01_strip_prefix.py:132-133)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import data as D
from .config import Config, JobSpec, stop

# ===========================================================================
# SHARED RULES -- verbatim from 10_postprocess/pp_io.py:72-128
# ===========================================================================
INSTRUCTION_LEAK_ANCHOR = (
    "Important: Do not add any information, claims, or details that are not")
DISTILL_MAX_PREAMBLE_CHARS = 120
DISTILL_PREAMBLE_WORDS = ('paraphras', 'condensed', 'rewrit', 'summary', 'version')
DISTILL_OPENERS = ('here is', "here's", 'here are', 'certainly', 'sure', 'below is',
                   'of course', 'the following is')


def strip_instruction_leak(text):
    """Remove a leaked rephrasing-instruction block from the START of `text`.

    Some wiki rewrites begin with the (fixed) system-prompt instruction paragraph that
    opens with INSTRUCTION_LEAK_ANCHOR and ends at the first '\\n\\n', after which the
    real article follows. START-ANCHORED only (mid-text occurrences are genuine content
    and are left untouched). If the whole doc is the leak (no '\\n\\n'), returns
    ('', True). Returns (new_text, stripped?).

    source: pp_io.py:80-90, verbatim.
    """
    if text and text.startswith(INSTRUCTION_LEAK_ANCHOR):
        cut = text.find('\n\n')
        return (text[cut + 2:], True) if cut >= 0 else ('', True)
    return text, False


def strip_distill_preamble(text):
    """Remove a leading paraphrase/condense TEMPLATE preamble (start-anchored).

    (a) first-PARAGRAPH rule: the first paragraph (up to '\\n\\n', within
        DISTILL_MAX_PREAMBLE_CHARS) is a meta-preamble -- a markdown header (###/##/**)
        OR a DISTILL_OPENERS opener, in either case containing a DISTILL_PREAMBLE_WORDS
        word, OR a bare 'Paraphrased ...:' label. Remove the paragraph + its '\\n\\n'.
    (b) first-LINE rule (single-'\\n' forms): the first line (up to '\\n', within
        DISTILL_MAX_PREAMBLE_CHARS) ENDS WITH ':' AND contains a meta word AND starts
        with a header (###/##/**) or a DISTILL_OPENERS opener (or 'paraphrased').
        Remove that line + break.
    Never .replace/.lstrip on the body; never touches docs that open directly with
    content. Returns (new_text, stripped?).

    source: pp_io.py:93-128, verbatim.
    """
    if not text:
        return text, False
    # (a) first-paragraph meta preamble
    cut = text.find('\n\n')
    if 0 <= cut <= DISTILL_MAX_PREAMBLE_CHARS:
        head = text[:cut].strip()
        low = head.lower()
        has_word = any(w in low for w in DISTILL_PREAMBLE_WORDS)
        if ((head.startswith(('###', '##', '**')) and has_word)
                or (low.startswith(DISTILL_OPENERS) and has_word)
                or (low.startswith('paraphrased') and head.endswith(':'))):
            return text[cut + 2:], True
    # (b) first-line meta preamble ending in ':' (single-'\n' forms)
    nl = text.find('\n')
    if 0 <= nl <= DISTILL_MAX_PREAMBLE_CHARS:
        head = text[:nl].strip()
        low = head.lower()
        has_word = any(w in low for w in DISTILL_PREAMBLE_WORDS)
        if head.endswith(':') and has_word and (
                head.startswith(('###', '##', '**'))
                or low.startswith(DISTILL_OPENERS)
                or low.startswith('paraphrased')):
            return text[nl + 1:].lstrip('\n'), True
    return text, False


# ===========================================================================
# WIKI RULE -- verbatim from 10_postprocess/01_strip_prefix.py
# ===========================================================================
WIKI_PREFIX = "Here is a paraphrased version:\n\n"   # exact, case-sensitive, start-anchored only


def trim_wiki(text):
    """Exact start-anchored prefix slice, THEN the instruction-leak strip.

    source: 01_strip_prefix.py:108-118, verbatim (including the `s is not None` guard,
    which differs from the wrap path's `or ''` normalisation -- both are preserved
    exactly as written).
    """
    s = text
    did = False
    if s is not None and s.startswith(WIKI_PREFIX):
        s = s[len(WIKI_PREFIX):]
        did = True                                   # exact prefix, start-anchored slice ONLY
    s, did_leak = strip_instruction_leak(s)           # + leaked system-prompt block
    return s, (did or did_leak)


def trim_distill(text):
    """source: 01_strip_prefix.py:119-123 -- the shared template-preamble rule."""
    return strip_distill_preamble(text)


# ===========================================================================
# WRAP RULE -- verbatim from 10_postprocess/01_strip_prefix_wrap.py:53-131
# ===========================================================================
MAX_PREAMBLE_CHARS = 300

# meta-openers (case-sensitive, start-anchored). A match alone is NOT enough -- the first
# paragraph must ALSO contain a rewrite-signal word (below) to qualify as a preamble.
OPENERS = (
    "Here is", "Here's", "Here are", "Below is", "Below are",
    "Sure, here", "Sure! Here", "Sure, here's", "Sure thing, here",
    "Certainly! Here", "Certainly, here", "Of course! Here", "Of course, here",
    "I have rewritten", "I've rewritten", "I have reworded", "I've reworded",
    "The following is", "This is the rewritten", "This is a rewritten",
    "Rewritten passage", "Rewritten version", "Rewritten text",
    "Here is a paraphrased", "Here's the rewritten", "Here is the rewritten",
)
SIGNAL_WORDS = (
    'passage', 'rewritten', 'rewrite', 'reworded', 'rephrased', 'paraphrase',
    'version', 'simplified', 'simpler', 'plain language', 'neutral', 'factual',
    'summary', 'question', 'q&a', 'style', 'reading level', 'young child', 'requested',
)
# STRICT meta-phrases for markdown-header / bare-label preambles (e.g. "### Paraphrased
# Text", "### Simple Version for Young Children", "Rewritten Passage:"). Deliberately
# STRONGER than SIGNAL_WORDS -- excludes generic content words (summary / question /
# passage / style) so genuine content headers like "### Frequently Asked Questions",
# "### Case Summary", "### Passage" survive.
STRICT_META = (
    'paraphras', 'rewritten', 'rewrite', 'reworded', 'rephrased',
    'simple version', 'simpler version', 'simplified version', 'plain language',
    'condensed version', 'scholarly language', 'young child version',
)


def strip_preamble(text):
    """Return (new_text, stripped_bool). Conservative, START-ANCHORED, first-paragraph only.

    Strips the leading paragraph (up to the first '\\n\\n', within MAX_PREAMBLE_CHARS)
    iff it is:
      (a) a SENTENCE-OPENER preamble -- starts with an OPENER ("Here is/Sure/Certainly/
          ...") AND contains a SIGNAL_WORD; OR
      (b) a MARKDOWN/BOLD header (###/##/**) or a bare "Paraphrased ...:" /
          "Rewritten passage..." label whose text contains a STRICT_META phrase.
    qa "Q:" openings never match (a) or (b); content headers like "### Frequently Asked
    Questions" / "### Case Summary" / "### Passage" are NOT stripped (no STRICT_META
    phrase). Never .replace/.lstrip on the body; only the leading preamble paragraph +
    its break are removed.

    source: 01_strip_prefix_wrap.py:104-131, verbatim.
    """
    if not text:
        return text, False
    nl = text.find('\n\n')
    if nl < 0 or nl > MAX_PREAMBLE_CHARS:
        return text, False
    head = text[:nl]
    low = head.lower()
    # (a) sentence-opener preamble (case-sensitive opener + any signal word)
    if head.startswith(OPENERS) and any(k in low for k in SIGNAL_WORDS):
        return text[nl + 2:], True
    # (b) markdown/bold header, or bare 'Paraphrased ...'/'Rewritten passage ...' label,
    #     gated on a STRICT_META phrase so genuine content headers survive
    if (head.lstrip().startswith(('###', '##', '**'))
            or low.startswith('paraphrased') or low.startswith('rewritten passage')) \
            and any(s in low for s in STRICT_META):
        return text[nl + 2:], True
    return text, False                                # opens with content -> leave untouched


def trim_wrap(text):
    """Wrap rule then leak strip.

    source: 01_strip_prefix_wrap.py:186-190. Note the `or ''` normalisation at :180,
    which the wiki path does NOT do -- preserved exactly.
    """
    s = text or ''
    new, did = strip_preamble(s)
    new, did_leak = strip_instruction_leak(new)
    return new, (did or did_leak)


TRIM_RULES = {"wiki": trim_wiki, "distill": trim_distill, "wrap": trim_wrap}


# ===========================================================================
# shard / job drivers
# ===========================================================================
@dataclass
class ShardStats:
    shard_index: int = 0
    n_rows: int = 0
    n_status2: int = 0
    n_stripped: int = 0
    tok_before_s2: int = 0
    tok_after_s2: int = 0
    openings: dict = field(default_factory=dict)


_TOK = None


def _init_worker(llama2_path: str):
    """source: 01_strip_prefix_wrap.py -- one tokenizer per process, Arrow pinned to a
    single thread so N processes do not each spawn N Arrow threads."""
    global _TOK
    import pyarrow as pa
    pa.set_cpu_count(1)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(llama2_path)


def _n_llama(text) -> int:
    """Raw Llama-2 length, NO special tokens (no BOS). +1 is applied only at budget time.
    source: 01_strip_prefix.py:81-83."""
    return len(_TOK(text or '', add_special_tokens=False).input_ids)


def trim_shard(args) -> ShardStats:
    """Trim one shard in place, atomically. Returns per-shard tallies."""
    in_path, out_path, rule_name, suffix, sample_openings = args
    in_path, out_path = Path(in_path), Path(out_path)
    rule = TRIM_RULES[rule_name]

    m = D.SHARD_RE.search(in_path.name.replace(suffix, ".parquet"))
    st = ShardStats(shard_index=int(m.group(1)) if m else -1)

    rows = list(D.iter_jsonl(in_path))
    st.n_rows = len(rows)
    openings = {}

    for r in rows:
        if r.get("status") != 2:          # ONLY status==2 rows are examined
            continue
        st.n_status2 += 1
        st.tok_before_s2 += int(r.get("n_output_tokens_llama2") or 0)
        s = r.get("rewritten_text")
        if sample_openings:
            key = (s or "")[:80]
            openings[key] = openings.get(key, 0) + 1
        new, did = rule(s)
        if did:
            r["rewritten_text"] = new
            r["n_output_tokens_llama2"] = _n_llama(new)   # recount ONLY changed rows
            st.n_stripped += 1
        st.tok_after_s2 += int(r.get("n_output_tokens_llama2") or 0)

    if sample_openings:
        st.openings = dict(sorted(openings.items(), key=lambda kv: -kv[1])[:5])

    if st.n_stripped or in_path != out_path:
        tmp = Path(str(out_path) + ".tmp")
        fh, close = D.open_jsonl_write(tmp, compress=str(out_path).endswith(".zst"))
        try:
            fh.write(("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
                      + "\n").encode("utf-8"))
        finally:
            close()
        os.replace(tmp, out_path)
        n_after = D.count_jsonl_rows(out_path)
        if n_after != st.n_rows:
            raise RuntimeError(
                f"{out_path.name}: rowcount changed on rewrite "
                f"({st.n_rows} -> {n_after})")
    return st


def trim_job(cfg: Config, job: JobSpec, workers: int | None = None, log=print) -> dict:
    """Trim every shard of one (arm, prompt) job. Resumable via a .trimmed marker."""
    from concurrent.futures import ProcessPoolExecutor
    from . import engine as E

    rule_name = job.prompt.trim
    in_place = bool(cfg.data["postprocess"]["in_place"])
    src_dir = job.output_dir
    dst_dir = src_dir if in_place else cfg.trimmed_dir(job.arm, job.prompt.id)
    dst_dir.mkdir(parents=True, exist_ok=True)

    marker = dst_dir / "_trimmed.done"
    if marker.exists():
        log(f"[trim] {job.job_id}: already trimmed -> skip")
        return json.loads(marker.read_text())

    st_job = D.verify_job(cfg, job)
    if st_job.state != "DONE":
        stop(f"{job.job_id}: cannot trim, generation is {st_job.state} "
             f"({st_job.done}/{st_job.n_shards} shards)."
             + ("\n  " + "\n  ".join(st_job.problems) if st_job.problems else ""))

    shards = sorted(src_dir.glob(f"part_*{cfg.shard_suffix}"))
    workers = workers or int(cfg.cluster["compute"]["cpu_workers"])
    log(f"[trim] {job.job_id}: rule={rule_name} shards={len(shards)} workers={workers} "
        f"in_place={in_place}")

    tasks = [(str(p), str(dst_dir / p.name), rule_name, cfg.shard_suffix, i < 5)
             for i, p in enumerate(shards)]
    agg = ShardStats()
    t0 = time.time()
    openings = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(str(E.llama2_dir(cfg)),)) as ex:
        for i, st in enumerate(ex.map(trim_shard, tasks), 1):
            agg.n_rows += st.n_rows
            agg.n_status2 += st.n_status2
            agg.n_stripped += st.n_stripped
            agg.tok_before_s2 += st.tok_before_s2
            agg.tok_after_s2 += st.tok_after_s2
            for k, v in st.openings.items():
                openings[k] = openings.get(k, 0) + v
            if i % 50 == 0:
                log(f"[trim] {job.job_id}: {i}/{len(shards)} shards "
                    f"({time.time()-t0:.0f}s)")

    pct = 100.0 * agg.n_stripped / agg.n_status2 if agg.n_status2 else 0.0
    summary = {
        "job_id": job.job_id, "rule": rule_name, "shards": len(shards),
        "rows": agg.n_rows, "status2": agg.n_status2, "stripped": agg.n_stripped,
        "stripped_pct": round(pct, 4),
        "tok_before_s2": agg.tok_before_s2, "tok_after_s2": agg.tok_after_s2,
        "tok_saved": agg.tok_before_s2 - agg.tok_after_s2,
        "top_openings_pre_strip": dict(
            sorted(openings.items(), key=lambda kv: -kv[1])[:5]),
        "elapsed_s": round(time.time() - t0, 1),
    }
    log(f"[trim] {job.job_id}: status2={agg.n_status2:,} stripped={agg.n_stripped:,} "
        f"({pct:.4f}%) tok_saved={summary['tok_saved']:,}")
    D.atomic_write_text(json.dumps(summary, indent=2), marker)
    return summary
