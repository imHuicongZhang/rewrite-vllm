#!/usr/bin/env python3
"""Find every unresolved placeholder in the repo, grouped by class, with file:line.

Two classes:
  TIANJIAN -- his environment: paths, GPU count, conda prefix, HF token, scheduler
  WYTRO    -- Wytro's content: HF dataset repo ids, prompt text, token budgets

BOTH are hard errors. Wytro fills the WYTRO blanks before handoff, so by the time this
repo reaches Tianjian only TIANJIAN blanks should remain; by the time he runs anything,
none should.

One conditional check beyond the scan: configs/data.yaml upload.repo_template is required
ONLY when upload.enabled is true. Upload is out of this pipeline's scope and ships
disabled, so the shipped template is empty and there is no marker to find -- what would
have been an exemption is instead the positive rule that the two keys must agree.

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


# --------------------------------------------------------------- upload, conditionally
# configs/data.yaml upload.repo_template used to be a hard-failing WYTRO blank. Upload is
# now out of scope -- delivery is arranged separately -- so the shipped default is
# `enabled: false` with an EMPTY template, and there is no marker left for the scan above
# to find. What replaces the marker is this: the template is required only when upload is
# actually switched on. Same rule, stated positively, and it also catches the reverse
# mistake (enabled with nothing to upload to) at the earliest possible gate.
#
# No PyYAML: this script runs before the environment exists, which is the whole reason it
# has no third-party imports. Two unique scalars inside one block, so the same targeted
# regex approach run_all.sh:83-86 uses for cluster.yaml is sufficient here.


def _upload_block(root: Path) -> dict:
    """The scalars under `upload:` in configs/data.yaml. {} if the file is unreadable."""
    try:
        text = (root / "configs" / "data.yaml").read_text(encoding="utf-8")
    except OSError:
        return {}
    out, inside = {}, False
    for line in text.splitlines():
        if re.match(r"^upload:\s*(#.*)?$", line):
            inside = True
            continue
        if inside:
            if line.strip() and not line.startswith((" ", "\t")):
                break                                   # dedented out of the block
            m = re.match(r"^\s+([a-z_]+):\s*(.*?)\s*(?:#.*)?$", line)
            if m:
                out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def upload_problem(root: Path) -> str | None:
    """The message to print, or None when nothing is wrong."""
    up = _upload_block(root)
    if up.get("enabled", "false").lower() != "true":
        return None
    tpl = up.get("repo_template", "")
    if not tpl or _O in tpl:
        return ("configs/data.yaml sets upload.enabled: true but upload.repo_template "
                "is empty.\n"
                "  Upload needs BOTH. Set a real template, e.g.\n"
                "      repo_template: your-org/rewrite-{arm}-{prompt_id}\n"
                "  or set  enabled: false  (the shipped default -- delivery of the "
                "finished data\n"
                "  is arranged separately, and the run's output is complete on disk "
                "without it).")
    return None


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

    upload_bad = upload_problem(root)

    total = sum(len(v) for v in hits.values())
    if total == 0 and not missing_env and not upload_bad:
        print("check_placeholders: OK -- no unresolved placeholders.")
        print(f"  exempt (templates/docs): {', '.join(sorted(EXEMPT_FILES))}, docs/")
        if (_upload_block(root).get("enabled", "false").lower() != "true"):
            print("  upload is disabled in configs/data.yaml, so upload.repo_template and "
                  "HF_TOKEN_WRITE")
            print("  are not required. The run ends with finished data on disk under "
                  "out_root/shuffled/.")
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

    if upload_bad:
        print("\n=== upload ===")
        print("  " + upload_bad.replace("\n", "\n  "))

    reasons = []
    if total:
        reasons.append(f"{total} placeholder(s) unresolved")
    if missing_env:
        reasons.append(".env is missing")
    if upload_bad:
        reasons.append("upload.enabled is true with no repo_template")
    print(f"\ncheck_placeholders: FAIL -- " + "; ".join(reasons) + ".")
    print("  Both classes are hard errors. Nothing will run until they are all filled.")
    print("  Tianjian: see docs/GUIDE_FOR_TIANJIAN.md section 2.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
