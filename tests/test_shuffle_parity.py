#!/usr/bin/env python3
"""Prove the bucketed shuffle is the original one, two independent ways.

1. Source-text comparison of the two ported functions against the originals, with
   comments and whitespace normalised away.
2. Execution equivalence: run both implementations over the same synthetic input and
   require the same bucket count, the same output shard names, and byte-identical ROW
   ORDER -- which is the property that actually matters, since a shuffle that merely
   preserves row counts could still order them differently.

Needs the original pipeline on disk. No hard-coded path lives in this repo:

    python tests/test_shuffle_parity.py --source-root /path/to/rewrite

Needs: 10_postprocess/pp_io.py under --source-root, plus numpy + pyarrow.
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import inspect
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def normalise(fn):
    """Statement lines only: drop the docstring, comments, blanks, and our namespacing."""
    text = inspect.getsource(fn)
    body = text.split('"""')[2] if text.count('"""') >= 2 else text
    out = []
    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s = s.replace("D.atomic_write_table", "atomic_write_table")
        out.append(s)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--rows-per-input", type=int, default=7000)
    ap.add_argument("--inputs", type=int, default=9)
    args = ap.parse_args(argv)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    src = Path(args.source_root).expanduser().resolve()
    pp_io = src / "10_postprocess" / "pp_io.py"
    if not pp_io.exists():
        print(f"SKIP: {pp_io} not found; this test only runs where the source tree exists.")
        return 77

    sys.path.insert(0, str(pp_io.parent))
    spec = importlib.util.spec_from_file_location("src_pp_io", str(pp_io))
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)

    import pyarrow as pa
    import pyarrow.parquet as pq
    from rewrite import shuffle as P

    print("=" * 68)
    print("1. source-text comparison")
    print("=" * 68)
    bad = 0
    for name, a, b in [("choose_buckets", S.choose_buckets, P.choose_buckets),
                       ("bucketed_shuffle", S.bucketed_shuffle, P.bucketed_shuffle)]:
        na, nb = normalise(a), normalise(b)
        same = na == nb
        bad += not same
        print(f"  {'ok  ' if same else 'FAIL'} {name}: {len(na)} source lines vs {len(nb)} ported")
        if not same:
            for d in list(difflib.unified_diff(na, nb, "source", "ported", lineterm=""))[:40]:
                print("      " + d)
    for name, a, b in [("MEM_CAP_BYTES", S.MEM_CAP_BYTES, P.MEM_CAP_BYTES),
                       ("GIB", S.GIB, P.GIB)]:
        same = a == b
        bad += not same
        print(f"  {'ok  ' if same else 'FAIL'} {name} = {a}")
    if bad:
        print("\nFAIL: the ported shuffle differs from the source.")
        return 1

    print()
    print("=" * 68)
    print("2. execution equivalence")
    print("=" * 68)
    tmp = Path(tempfile.mkdtemp(prefix="shufparity-"))
    try:
        n = args.rows_per_input
        specs = []
        for i in range(args.inputs):
            t = pa.table({
                "doc_id": pa.array(range(i * n, (i + 1) * n), type=pa.int64()),
                "txt": pa.array([f"doc-{i}-{j}" * 20 for j in range(n)],
                                type=pa.large_string()),
            })
            p = tmp / f"in_{i:03d}.parquet"
            pq.write_table(t, str(p), compression="zstd")
            specs.append((str(p), None))

        load = lambda s: pq.read_table(s[0])
        res = {}
        for tag, fn in (("source", S.bucketed_shuffle), ("ported", P.bucketed_shuffle)):
            od, td = tmp / f"out_{tag}", tmp / f"tmp_{tag}"
            r = fn(specs, load, od, td, seed=42, rows_per_shard=5000,
                   mem_bytes=8 * 1024 ** 3, inflate=4.0, log=lambda *a: None)
            files = sorted(x.name for x in od.iterdir())
            ids = []
            for f in files:
                ids += pq.read_table(str(od / f)).column("doc_id").to_pylist()
            res[tag] = (r, files, ids)

        (sr, sf, si), (pr, pf, pi) = res["source"], res["ported"]
        total = args.inputs * n
        checks = [
            ("return value (rows, shards, buckets)", sr == pr, f"{sr} vs {pr}"),
            ("output shard names", sf == pf, f"{len(sf)} shards"),
            ("ROW ORDER byte-identical", si == pi, f"{len(si):,} rows"),
            ("complete permutation (nothing lost or duplicated)",
             sorted(si) == list(range(total)), f"{total:,} rows"),
            ("rows actually reordered", si != sorted(si), ""),
        ]
        failed = 0
        for label, okk, detail in checks:
            failed += not okk
            print(f"  {'ok  ' if okk else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
        print()
        if failed:
            print("FAIL: the ported shuffle does not reproduce the source's output.")
            return 1
        print("PASS: shuffle is identical to the source in text and in behaviour.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
