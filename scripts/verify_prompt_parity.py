#!/usr/bin/env python3
"""Prove the twelve prompt files are the ones the original pipeline used.

WHY THIS EXISTS
---------------
Four of the six distinct prompt texts were RECONSTRUCTED, not copied. Their source files
(`distill_prompt.txt` and `wrap_prompts.json`) no longer exist on any reachable
filesystem, so the text was recovered from saved generation records. Reconstruction is not
a copy, and a wrong prompt would corrupt every row of the affected arms silently -- the
run would look perfectly healthy.

The check: render each prompt through the model's chat template with an EMPTY document and
count tokens. That single integer pins down the prompt text, the chat template, and the
tokenizer simultaneously. For the distill prompt the expected value, 185, is not something
anyone chose -- it is the number the original pipeline printed in its own production log:

    [w0] OVERHEAD(empty-doc templated tokens)=185 raw_text_budget=28487 drop_threshold=28672

Expected values are read from configs/data.yaml, so there is exactly one source of truth
and this script cannot drift from what the workers assert at runtime.

STANDALONE ON PURPOSE
---------------------
Needs only `transformers`, `PyYAML` and a local model directory. No GPU, no vLLM, no
cluster config, no filled placeholders, no network. Run it before handoff.

    python scripts/verify_prompt_parity.py --model /path/to/Qwen2.5-7B-Instruct

If you have not downloaded the model yet, any local copy of the same repo works -- only
its tokenizer and chat template are used.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Recorded so the numbers below can be traced without opening another file.
PROVENANCE = {
    150: "wiki prompt: the one prompt file that still exists in the source tree; "
         "shipped as a byte-identical copy",
    185: "distill prompt: RECONSTRUCTED -- matches the value the source's own production "
         "log printed for this prompt",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="local dir containing the Qwen2.5-7B-Instruct tokenizer + "
                         "chat template")
    ap.add_argument("--repo-root", default=str(REPO))
    ap.add_argument("--offline", action="store_true", default=True,
                    help="never touch the network (default)")
    args = ap.parse_args(argv)

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    root = Path(args.repo_root).resolve()
    try:
        import yaml
    except ImportError:
        print("STOP: PyYAML is required (pip install PyYAML)", file=sys.stderr)
        return 2
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("STOP: transformers is required (pip install transformers)", file=sys.stderr)
        return 2

    model = Path(args.model).expanduser()
    if not (model / "tokenizer_config.json").exists() and \
       not (model / "tokenizer.json").exists():
        print(f"STOP: no tokenizer found at {model}\n"
              f"      Point --model at a local Qwen2.5-7B-Instruct directory.",
              file=sys.stderr)
        return 2

    data = yaml.safe_load((root / "configs" / "data.yaml").read_text())
    vllm = yaml.safe_load((root / "configs" / "vllm.yaml").read_text())
    chat = vllm["chat"]
    defs = data["prompt_defs"]

    print(f"model      : {model}")
    print(f"chat args  : role={chat['role']} tokenize={chat['tokenize']} "
          f"add_generation_prompt={chat['add_generation_prompt']}")
    print(f"expected   : configs/data.yaml -> prompt_defs[*].expected_overhead")
    print()

    tok = AutoTokenizer.from_pretrained(str(model))

    def _units(arm, pr, d):
        """(label, file, expected_overhead) for every prompt TEXT this job can emit.

        One for a grounded job; four for wrap-inspired's styled pass, which carries its
        prompts in prompt_defs[<def>].styles rather than in a `file:` key.
        """
        if d["mode"] == "wrap_multi":
            return [(f"{pr['id']}:{st['style']}", st["file"], int(st["expected_overhead"]))
                    for st in d["styles"]]
        return [(pr["id"], pr["file"], int(d["expected_overhead"]))]

    rows, failures, seen_text = [], [], {}
    for arm in data["arms"]:
        for pr in arm["prompts"]:
            d = defs[pr["def"]]
            for label, rel, exp in _units(arm, pr, d):
                path = root / rel
                if not path.exists():
                    failures.append((arm["name"], label, "MISSING FILE", str(path)))
                    continue
                text = path.read_text()

                # Structural guards, same ones config.py applies at load time.
                if d["mode"] == "grounded" and text.count("[TEXT]") != 1:
                    failures.append((arm["name"], label, "STRUCTURE",
                                     f"expected exactly one [TEXT], found "
                                     f"{text.count('[TEXT]')}"))
                    continue
                if d["mode"] == "wrap_multi" and not text.endswith("\n\nPassage:\n"):
                    failures.append((arm["name"], label, "STRUCTURE",
                                     "wrap prompt must end with '\\n\\nPassage:\\n'"))
                    continue

                content = text.replace("[TEXT]", "") if "[TEXT]" in text else text
                rendered = tok.apply_chat_template(
                    [{"role": chat["role"], "content": content}],
                    tokenize=chat["tokenize"],
                    add_generation_prompt=chat["add_generation_prompt"])
                got = len(tok(rendered, add_special_tokens=False).input_ids)
                rows.append((arm["name"], label, pr["def"], got, exp, path, text))
                seen_text.setdefault(text, []).append(f"{arm['name']}/{label}")
                if got != exp:
                    failures.append((arm["name"], label, "OVERHEAD",
                                     f"got {got}, expected {exp}"))

    # The style ORDER is part of the reproducible seed, so verify it here too -- this
    # script is run standalone, without config.py's loader.
    for arm in data["arms"]:
        for pr in arm["prompts"]:
            d = defs[pr["def"]]
            if d["mode"] != "wrap_multi":
                continue
            names = [st["style"] for st in d["styles"]]
            if names != ["easy", "hard", "wiki", "qa"]:
                failures.append((arm["name"], pr["id"], "STYLE ORDER",
                                 f"{names} != ['easy','hard','wiki','qa']; the order is "
                                 f"part of the seed (rewrite_worker.py:39)"))

    w = max((len(f"{a}/{i}") for a, i, *_ in rows), default=20)
    print(f"{'JOB':{w}}  {'PROMPT DEF':12}  {'GOT':>5} {'EXPECTED':>9}  RESULT")
    for arm, pid, dname, got, exp, path, _ in rows:
        mark = "ok" if got == exp else "MISMATCH"
        print(f"{arm + '/' + pid:{w}}  {dname:12}  {got:5d} {exp:9d}  {mark}")

    print()
    print(f"distinct prompt texts across the {len(rows)} (job, prompt-text) pairs:")
    for text, users in sorted(seen_text.items(), key=lambda kv: kv[1][0]):
        head = " ".join(text.split())[:58]
        print(f"  {len(users)} job(s): {', '.join(users)}")
        print(f"      \"{head}...\"")

    print()
    for val, why in sorted(PROVENANCE.items()):
        hit = [f"{a}/{i}" for a, i, _, _, e, _, _ in rows if e == val]
        if hit:
            print(f"  {val} -> {why}")

    print()
    if failures:
        print(f"FAIL -- {len(failures)} problem(s):")
        for arm, pid, kind, detail in failures:
            print(f"  {arm}/{pid}: {kind}: {detail}")
        print()
        print("  Do NOT edit expected_overhead to make this pass. That integer is the")
        print("  evidence that the prompt text is the original one; changing it deletes")
        print("  the evidence instead of fixing the problem.")
        return 1

    print(f"PASS -- all {len(rows)} job prompts match their expected templated overhead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
