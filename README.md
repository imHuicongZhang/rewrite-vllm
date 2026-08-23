# rewrite-vllm

Rewrites five web-text corpora with Qwen2.5-7B-Instruct under vLLM, then trims and
shuffles the output. A sixth, `quality-base`, is the control: verified, never rewritten.
**12 jobs**, one per (arm, prompt) — each prompt rewrites its arm's corpus in full, so
`wrap-inspired`'s 4 prompts mean 4 complete passes, not 4 slices. Plain bash, one node,
N GPUs, no scheduler. Resumable at shard level throughout.

**Start here → [docs/GUIDE_FOR_TIANJIAN.md](docs/GUIDE_FOR_TIANJIAN.md)**

```bash
bash scripts/00_setup_env.sh           # pinned env; prints your activate_cmd
$EDITOR configs/cluster.yaml           # fill your blanks; cp .env.example .env
python3 scripts/check_placeholders.py  # must exit 0
python scripts/preflight.py            # must pass before anything runs
bash scripts/run_all.sh                # the whole run; re-run to resume
```

Also: [SOURCE_INVENTORY](docs/SOURCE_INVENTORY.md) · [POSTPROCESSING](docs/POSTPROCESSING.md) · [HANDOFF_REVIEW](docs/HANDOFF_REVIEW.md)
