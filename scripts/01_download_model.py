#!/usr/bin/env python3
"""Download Qwen2.5-7B-Instruct and the Llama-2 tokenizer to a local directory.

Idempotent, uses hf_transfer, verifies config.json and the safetensors shard count, and
works with HF_HUB_OFFLINE=1 on reruns (a complete local copy short-circuits before any
network call is attempted).

    python scripts/01_download_model.py [--force] [--offline]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rewrite.config import load_config, stop            # noqa: E402
from rewrite import engine as E                          # noqa: E402


def _verify_model(d: Path) -> tuple[bool, str]:
    cfg_p = d / "config.json"
    if not cfg_p.exists():
        return False, "config.json missing"
    try:
        conf = json.loads(cfg_p.read_text())
    except Exception as e:
        return False, f"config.json unreadable: {e}"
    if "architectures" not in conf:
        return False, "config.json has no 'architectures'"

    idx = d / "model.safetensors.index.json"
    shards = sorted(d.glob("*.safetensors"))
    if idx.exists():
        want = json.loads(idx.read_text()).get("weight_map", {})
        expected = sorted(set(want.values()))
        missing = [f for f in expected if not (d / f).exists()]
        if missing:
            return False, (f"{len(missing)}/{len(expected)} weight shard(s) missing, "
                           f"e.g. {missing[:3]}")
        return True, f"{len(expected)} shard(s), arch={conf['architectures'][0]}"
    if not shards:
        return False, "no .safetensors files and no index"
    return True, f"{len(shards)} shard(s) (single-file), arch={conf['architectures'][0]}"


def _verify_tokenizer(d: Path, expect_vocab: int) -> tuple[bool, str]:
    if not (d / "tokenizer.json").exists() and not (d / "tokenizer.model").exists():
        return False, "no tokenizer.json / tokenizer.model"
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(d))
    except Exception as e:
        return False, f"failed to load: {e}"
    got = getattr(tok, "vocab_size", None)
    if got is not None and int(got) != int(expect_vocab):
        return False, f"vocab_size {got} != expected {expect_vocab}"
    return True, f"vocab_size={got}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config-root", default=None)
    ap.add_argument("--force", action="store_true", help="re-download even if complete")
    ap.add_argument("--offline", action="store_true",
                    help="never touch the network; fail if anything is missing")
    args = ap.parse_args(argv)

    cfg = load_config(args.config_root)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER",
                          cfg.env.get("HF_HUB_ENABLE_HF_TRANSFER", "1"))
    os.environ.setdefault("HF_HOME", str(cfg.paths["hf_cache"]))
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    targets = [
        ("model", cfg.vllm["model"], E.model_dir(cfg), None),
        ("llama2_tokenizer", cfg.vllm["llama2_tokenizer"], E.llama2_dir(cfg),
         int(cfg.vllm["llama2_tokenizer"]["expected_vocab_size"])),
    ]

    rc = 0
    for label, spec, dest, expect_vocab in targets:
        dest.mkdir(parents=True, exist_ok=True)
        ok, why = (_verify_tokenizer(dest, expect_vocab) if expect_vocab
                   else _verify_model(dest))
        if ok and not args.force:
            print(f"[model] {label}: already complete at {dest} ({why}) -> skip")
            continue
        if args.offline:
            print(f"[model] {label}: INCOMPLETE at {dest} ({why}) and --offline was "
                  f"given", file=sys.stderr)
            rc = 1
            continue

        print(f"[model] {label}: downloading {spec['repo_id']}@{spec.get('revision','main')} "
              f"-> {dest}  ({why})")
        from huggingface_hub import snapshot_download
        kw = {}
        tok = cfg.env.get("HF_TOKEN")
        if tok:
            kw["token"] = tok
        if spec.get("allow_patterns"):
            kw["allow_patterns"] = list(spec["allow_patterns"])
        snapshot_download(
            repo_id=spec["repo_id"], revision=spec.get("revision", "main"),
            local_dir=str(dest), cache_dir=str(cfg.paths["hf_cache"]), **kw)

        ok, why = (_verify_tokenizer(dest, expect_vocab) if expect_vocab
                   else _verify_model(dest))
        if not ok:
            print(f"[model] {label}: verification FAILED after download: {why}",
                  file=sys.stderr)
            rc = 1
        else:
            print(f"[model] {label}: OK ({why})")

    if rc == 0:
        print("[model] all downloads verified.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
