# rewrite-vllm

Rewrites five web-text corpora with Qwen2.5-7B-Instruct under vLLM, then trims and
shuffles the output. **10 jobs**, one per (arm, prompt): every arm gets a wiki-style or
styled pass plus the shared distill pass, and each prompt covers its arm's corpus in full.
Plain bash, one node, N GPUs, no scheduler. Resumable at shard level throughout.

Input is **one gated HuggingFace repo**, `wytro/Know-Your-Sources-7B`, with one folder per
arm. Only the *remainder* half of each block is rewritten: the shared 20B raw core and the
50B `quality-base` control never touch a GPU and are not downloaded here — see
[docs/DESIGN_DELTA.md](docs/DESIGN_DELTA.md).

`wrap-inspired` is the one arm whose first pass is not a single prompt: it assigns one of
four styles (`easy`/`hard`/`wiki`/`qa`) **per document**, seeded on `(42, shard_index)`,
and records the choice in the `wrap_style` column.

**Start here → [docs/GUIDE_FOR_TIANJIAN.md](docs/GUIDE_FOR_TIANJIAN.md)**

```bash
bash scripts/00_setup_env.sh           # pinned env; prints your activate_cmd
$EDITOR configs/cluster.yaml           # fill your blanks; cp .env.example .env
python3 scripts/check_placeholders.py  # must exit 0
python scripts/preflight.py            # must pass before anything runs
bash scripts/run_all.sh                # the whole run; re-run to resume
```

Also: [SOURCE_INVENTORY](docs/SOURCE_INVENTORY.md) · [POSTPROCESSING](docs/POSTPROCESSING.md) · [HANDOFF_REVIEW](docs/HANDOFF_REVIEW.md)
