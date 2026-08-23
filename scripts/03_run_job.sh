#!/usr/bin/env bash
# Run ONE rewrite job = one (arm, prompt) pair, across all N GPUs.
#
#   bash scripts/03_run_job.sh <arm> <prompt_id>
#   bash scripts/03_run_job.sh wrap-inspired p3
#
# N single-GPU worker processes are launched, one per GPU, each with
# tensor_parallel_size=1 and owning the shards where (shard_index % N == worker_id).
# This is exactly the source's data-parallel scheme -- its 8-way SLURM array -- and NOT
# vLLM's own data_parallel_size, which the source never used.
#
# Resume is the default: a shard with a valid .done sidecar is skipped. Just re-run the
# same command after any interruption.
#
# A flock prevents two copies of the same job running at once. Do not remove it: two
# engines per GPU would each try to reserve gpu_memory_utilization=0.85 and OOM.
set -euo pipefail

ARM="${1:-}"; PROMPT_ID="${2:-}"
if [[ -z "$ARM" || -z "$PROMPT_ID" ]]; then
  echo "usage: $0 <arm> <prompt_id>" >&2
  echo "  e.g. $0 wrap-inspired p3" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Config bootstrap, in two stages.
#
# Stage 1 pulls ONLY env.activate_cmd / env.extra_preamble out of cluster.yaml with sed.
# It cannot use PyYAML: PyYAML lives inside the environment we have not activated yet,
# and the system python3 may well not have it. Two unique, simple scalars, so sed is
# sufficient and dependency-free.
#
# Stage 2 activates, then asks rewrite.config for everything else -- so every other value
# is read through the SAME validated code path the Python entry points use, with
# ${...} references already expanded. One source of truth, no second YAML parser.
# ---------------------------------------------------------------------------
yaml_scalar() {  # $1 = key name, unique in configs/cluster.yaml
  sed -n "s/^[[:space:]]*$1:[[:space:]]*//p" configs/cluster.yaml \
    | head -1 | sed -e 's/^"//' -e "s/^'//" -e 's/"$//' -e "s/'$//"
}

ACTIVATE="$(yaml_scalar activate_cmd)"
EXTRA="$(yaml_scalar extra_preamble)"
if [[ -z "$ACTIVATE" ]]; then
  echo "*** STOP: could not read env.activate_cmd from configs/cluster.yaml." >&2
  echo "    Run scripts/00_setup_env.sh first; it prints the exact line to paste." >&2
  exit 2
fi
case "$ACTIVATE" in *"<<"*) 
  echo "*** STOP: env.activate_cmd in configs/cluster.yaml is still a placeholder." >&2
  echo "    Run: python3 scripts/check_placeholders.py" >&2
  exit 2 ;;
esac

[[ -n "$EXTRA" ]] && eval "$EXTRA"
# shellcheck disable=SC1090
eval "$ACTIVATE"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Stage 2: everything else, through the validated config loader.
eval "$(python - "$REPO_ROOT" <<'PYCFG'
import shlex, sys
sys.path.insert(0, sys.argv[1] + "/src")
from rewrite.config import load_config
c = load_config(sys.argv[1])
print(f"NGPU={c.num_gpus}")
print("GPU_IDS=" + shlex.quote(" ".join(str(g) for g in c.gpu_ids)))
print("LOG_ROOT=" + shlex.quote(str(c.paths["log_root"])))
print("OUT_ROOT=" + shlex.quote(str(c.paths["out_root"])))
PYCFG
)"

# Fail on a bad arm/prompt now, not after N engines have spun up on N GPUs.
python - "$REPO_ROOT" "$ARM" "$PROMPT_ID" <<'PYJOB'
import sys
sys.path.insert(0, sys.argv[1] + "/src")
from rewrite.config import load_config, get_job
get_job(load_config(sys.argv[1]), sys.argv[2], sys.argv[3])
PYJOB

read -r -a GPU_IDS <<< "$GPU_IDS"
if [[ "${#GPU_IDS[@]}" -ne "$NGPU" ]]; then
  echo "*** STOP: gpu_ids has ${#GPU_IDS[@]} entries but num_gpus is $NGPU." >&2
  exit 2
fi

LOG_DIR="$LOG_ROOT/$ARM/$PROMPT_ID"
LOCK_DIR="$OUT_ROOT/raw/$ARM/$PROMPT_ID"
mkdir -p "$LOG_DIR" "$LOCK_DIR"
LOCK="$LOCK_DIR/.joblock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "*** STOP: another process is already running $ARM/$PROMPT_ID (lock: $LOCK)." >&2
  echo "    Jobs must run one at a time -- each one already uses every GPU." >&2
  exit 3
fi

echo "=== $ARM/$PROMPT_ID :: $NGPU worker(s) on GPU(s) ${GPU_IDS[*]} ==="
echo "    logs: $LOG_DIR/worker<i>.log"

pids=(); rcs=0
cleanup() { echo "  interrupted -- signalling workers"; kill -TERM "${pids[@]}" 2>/dev/null || true; }
trap cleanup INT TERM

for ((i=0; i<NGPU; i++)); do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" \
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -u -m rewrite.run_rewrite \
      --arm "$ARM" --prompt-id "$PROMPT_ID" \
      --worker-id "$i" --num-workers "$NGPU" \
      --config-root "$REPO_ROOT" \
      >>"$LOG_DIR/worker$i.log" 2>&1 &
  pids+=("$!")
  echo "    worker $i -> GPU ${GPU_IDS[$i]}  (pid ${pids[$i]})"
done

for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "*** worker $i FAILED -- tail of $LOG_DIR/worker$i.log:" >&2
    tail -30 "$LOG_DIR/worker$i.log" >&2 || true
    rcs=1
  fi
done
[[ $rcs -ne 0 ]] && { echo "*** STOP: at least one worker failed." >&2; exit 1; }

echo "--- verifying row conservation for $ARM/$PROMPT_ID ---"
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
python -u -m rewrite.run_rewrite --arm "$ARM" --prompt-id "$PROMPT_ID" \
    --config-root "$REPO_ROOT" --verify
echo "=== $ARM/$PROMPT_ID COMPLETE ==="
