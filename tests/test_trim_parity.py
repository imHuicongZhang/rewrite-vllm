#!/usr/bin/env python3
"""Differential test: our trim rules vs the ORIGINAL pipeline's, on real model outputs.

This is the most important test in the repo. The trim rules were transcribed by hand from
four scripts in a source tree that Tianjian cannot see and that may not outlive this
project. This test imports the ORIGINAL functions directly and diffs them against ours
over tens of thousands of real generations, so "ported verbatim" is a measurement rather
than a claim.

It needs the source pipeline on disk, so it can only run where that tree still exists --
normally the cluster it was written on. There is no hard-coded path anywhere in this repo;
you must pass --source-root.

    python tests/test_trim_parity.py --source-root /path/to/rewrite

What it needs under --source-root:
    10_postprocess/pp_io.py                  shared strip rules
    10_postprocess/01_strip_prefix.py        WIKI_PREFIX constant
    10_postprocess/01_strip_prefix_wrap.py   wrap preamble rule
    00_TMP/rewriting_monitor.md              real wiki-pass outputs   (optional)
    00_TMP/rewriting_monitor_distill.md      real distill-pass outputs (optional)

If the monitor logs are absent it still runs the constant comparison and the adversarial
cases, and says so. Requires numpy + pyarrow (imported by the source module).
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Branches real data may not reach: empty/None, the bare prefix, a leak with no blank line,
# content headers that must SURVIVE, and a qa opening that must never match.
ADVERSARIAL_TAIL = [
    "", None,
    "### Frequently Asked Questions\n\nQ: a\nA: b",
    "### Case Summary\n\nBody.",
    "### Passage\n\nBody.",
    "### Paraphrased Text\n\nBody.",
    "Here is a list of items\n\nBody.",
    "Here is the rewritten passage:\n\nBody.",
    "Sure, here's the rewritten version:\n\nBody.",
    "Paraphrased version:\nBody.",
    "**Condensed Version:**\n\nBody.",
    "Rewritten Passage:\n\nBody.",
    "The following is the paraphrase\n\nBody.",
    "Q: What?\nA: This.\n\nQ: Why?\nA: Because.",
    "x" * 400 + "\n\nBody.",
    "Here is\n\n",
    "\n\nBody.",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def harvest(path: Path, limit=None):
    """Pull real model outputs out of the source's monitor logs."""
    out = []
    if not path.exists():
        return out
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("  - output: "):
                try:
                    out.append(ast.literal_eval(line[len("  - output: "):].rstrip("\n")))
                except Exception:
                    pass
                if limit and len(out) >= limit:
                    break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-root", required=True,
                    help="path to the ORIGINAL rewrite pipeline tree")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap harvested documents (default: all)")
    args = ap.parse_args(argv)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    src = Path(args.source_root).expanduser().resolve()
    pp = src / "10_postprocess"
    need = [pp / "pp_io.py", pp / "01_strip_prefix.py", pp / "01_strip_prefix_wrap.py"]
    missing = [str(x) for x in need if not x.exists()]
    if missing:
        print("SKIP: the original pipeline is not at --source-root; cannot compare.")
        for m in missing:
            print(f"  missing: {m}")
        print("\nThis test is only runnable where the source tree still exists.")
        return 77

    sys.path.insert(0, str(pp))
    S_PP = load_module("src_pp_io", pp / "pp_io.py")
    S_WRAP = load_module("src_strip_wrap", pp / "01_strip_prefix_wrap.py")
    wiki_prefix = ast.literal_eval(
        re.search(r'WIKI_PREFIX = (".*?")\s', (pp / "01_strip_prefix.py").read_text()).group(1))

    from rewrite import postprocess as P

    print("=" * 68)
    print("1. constants")
    print("=" * 68)
    consts = [
        ("INSTRUCTION_LEAK_ANCHOR", S_PP.INSTRUCTION_LEAK_ANCHOR, P.INSTRUCTION_LEAK_ANCHOR),
        ("DISTILL_MAX_PREAMBLE_CHARS", S_PP.DISTILL_MAX_PREAMBLE_CHARS, P.DISTILL_MAX_PREAMBLE_CHARS),
        ("DISTILL_PREAMBLE_WORDS", S_PP.DISTILL_PREAMBLE_WORDS, P.DISTILL_PREAMBLE_WORDS),
        ("DISTILL_OPENERS", S_PP.DISTILL_OPENERS, P.DISTILL_OPENERS),
        ("MAX_PREAMBLE_CHARS", S_WRAP.MAX_PREAMBLE_CHARS, P.MAX_PREAMBLE_CHARS),
        ("OPENERS", S_WRAP.OPENERS, P.OPENERS),
        ("SIGNAL_WORDS", S_WRAP.SIGNAL_WORDS, P.SIGNAL_WORDS),
        ("STRICT_META", S_WRAP.STRICT_META, P.STRICT_META),
        ("WIKI_PREFIX", wiki_prefix, P.WIKI_PREFIX),
    ]
    bad = 0
    for name, a, b in consts:
        same = a == b
        bad += not same
        size = len(a) if hasattr(a, "__len__") else a
        print(f"  {'ok  ' if same else 'FAIL'} {name:28s} ({size})")
    if bad:
        print(f"\nFAIL: {bad} constant(s) differ from the source.")
        return 1

    print()
    print("=" * 68)
    print("2. behaviour over real model outputs")
    print("=" * 68)
    tmp = src / "00_TMP"
    wiki_out = harvest(tmp / "rewriting_monitor.md", args.limit)
    dist_out = harvest(tmp / "rewriting_monitor_distill.md", args.limit)
    if not wiki_out and not dist_out:
        print("  NOTE: monitor logs absent -- adversarial cases only.")
    print(f"  harvested: wiki={len(wiki_out):,}  distill={len(dist_out):,}")

    # reference implementations, transcribed from the source's own dispatch sites
    def src_wiki(s):
        did = False
        if s is not None and s.startswith(wiki_prefix):
            s = s[len(wiki_prefix):]; did = True
        s, leak = S_PP.strip_instruction_leak(s)
        return s, (did or leak)

    def src_distill(s):
        return S_PP.strip_distill_preamble(s)

    def src_wrap(s):
        s = s or ''
        new, did = S_WRAP.strip_preamble(s)
        new, leak = S_PP.strip_instruction_leak(new)
        return new, (did or leak)

    extra = list(ADVERSARIAL_TAIL) + [wiki_prefix, wiki_prefix + "Body here.",
        S_PP.INSTRUCTION_LEAK_ANCHOR + " explicitly stated.\n\nReal article.",
        S_PP.INSTRUCTION_LEAK_ANCHOR + " no double newline ever"]

    cases = [("wiki", src_wiki, P.trim_wiki, wiki_out),
             ("distill", src_distill, P.trim_distill, dist_out),
             ("wrap", src_wrap, P.trim_wrap, wiki_out + dist_out)]

    total = fails = 0
    for name, fsrc, fport, corpus in cases:
        n = mism = 0
        for s in list(corpus) + extra:
            try:
                a = fsrc(s)
            except Exception as e:
                a = ("EXC", type(e).__name__)
            try:
                b = fport(s)
            except Exception as e:
                b = ("EXC", type(e).__name__)
            n += 1
            if a != b:
                mism += 1
                if mism <= 3:
                    print(f"    MISMATCH [{name}] in={s!r:.100}")
                    print(f"       source={a!r:.140}")
                    print(f"       ported={b!r:.140}")
        total += n
        fails += mism
        print(f"  {'ok  ' if mism == 0 else 'FAIL'} {name:8s} {n:,} cases, {mism} mismatch(es)")

    print()
    if fails:
        print(f"FAIL: {fails} behavioural mismatch(es) over {total:,} comparisons.")
        return 1
    print(f"PASS: trim rules byte-identical to the source over {total:,} comparisons.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
