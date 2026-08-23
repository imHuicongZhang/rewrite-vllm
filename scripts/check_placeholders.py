#!/usr/bin/env python3
"""Find every unresolved placeholder in the repo, grouped by class, with file:line.

Two classes:
  TIANJIAN -- his environment: paths, GPU count, conda prefix, HF token, scheduler
  WYTRO    -- Wytro's content: HF dataset repo ids, prompt text, token budgets

BOTH are hard errors. Wytro fills the WYTRO blanks before handoff, so by the time this
repo reaches Tianjian only TIANJIAN blanks should remain; by the time he runs anything,
none should.

Exit code: 0 if the repo is clean, 1 if anything is left.

Run standalone -- no third-party imports, so it works before any environment is built:
    python3 scripts/check_placeholders.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Built by concatenation so this file never matches its own scan.
_O = "<" * 3
_C = ">" * 3
PATTERN = re.compile(_O + r"(TIANJIAN|WYTRO)\s*:\s*([^" + ">" + r"]*)" + _C)

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
             "data", "out", "logs", "models", "manifests", "_smoke", ".mypy_cache"}

# Files whose JOB is to show the convention. They keep their markers forever.
EXEMPT_FILES = {
    ".env.example",                     # the template; .env is what gets checked
    "README.md",
    "scripts/check_placeholders.py",    # this file
}
EXEMPT_DIRS = {"docs"}                  # the guide lists every blank verbatim

TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".txt", ".md", ".json", ".sbatch",
                 ".cfg", ".toml", ".env", ".example", ""}


def is_exempt(rel: Path) -> bool:
    if str(rel) in EXEMPT_FILES:
        return True
    return bool(rel.parts) and rel.parts[0] in EXEMPT_DIRS


def scan(root: Path):
    hits = {"TIANJIAN": [], "WYTRO": []}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if is_exempt(rel):
            continue
        if p.suffix not in TEXT_SUFFIXES and p.name != ".env":
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for m in PATTERN.finditer(line):
                hits[m.group(1)].append((str(rel), i, " ".join(m.group(2).split())))
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    hits = scan(root)

    # .env is required and is NOT exempt -- .env.example is the template, .env is real.
    env = root / ".env"
    missing_env = not env.exists()

    total = sum(len(v) for v in hits.values())
    if total == 0 and not missing_env:
        print("check_placeholders: OK -- no unresolved placeholders.")
        print(f"  exempt (templates/docs): {', '.join(sorted(EXEMPT_FILES))}, docs/")
        return 0

    for klass in ("TIANJIAN", "WYTRO"):
        rows = hits[klass]
        who = ("Tianjian -- your machine" if klass == "TIANJIAN"
               else "Wytro -- workload content")
        print(f"\n=== {klass} ({len(rows)} unresolved) -- {who} ===")
        if not rows:
            print("  (none)")
            continue
        if args.quiet:
            continue
        width = max(len(f"{f}:{n}") for f, n, _ in rows)
        for f, n, desc in rows:
            print(f"  {f}:{n}".ljust(width + 4) + f"  {desc}")

    if missing_env:
        print("\n=== .env ===")
        print("  MISSING. Run:  cp .env.example .env   then fill in the blanks.")

    print(f"\ncheck_placeholders: FAIL -- {total} placeholder(s) unresolved"
          + (" and .env is missing." if missing_env else "."))
    print("  Both classes are hard errors. Nothing will run until they are all filled.")
    print("  Tianjian: see docs/GUIDE_FOR_TIANJIAN.md section 2.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
