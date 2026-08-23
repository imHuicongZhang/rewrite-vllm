#!/usr/bin/env bash
# The whole run, end to end, resumable:
#
#   preflight -> model -> data -> 12 jobs SEQUENTIALLY -> postprocess -> upload
#
# Safe to re-run after ANY interruption. Finished shards are skipped via their .done
# sidecars; finished jobs are skipped without even loading a model.
#
#   bash scripts/run_all.sh                 # everything
#   bash scripts/run_all.sh --status        # just print the job table and exit
#   bash scripts/run_all.sh --skip-upload
#   bash scripts/run_all.sh --from-job 5    # resume at job 5 (1-based, see the table)
#
# JOBS RUN ONE AT A TIME, ON PURPOSE. Each job already uses every GPU. Running two would
# make each vLLM engine try to reserve gpu_memory_utilization=0.85 of the same card and
# OOM -- and 0.85 is a locked source-parity value, not a knob. Do not parallelise this.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- ARGPARSE BEGIN (tests/test_integration.py extracts between these markers) ---
STATUS_ONLY=0; SKIP_UPLOAD=0; SKIP_PREFLIGHT=0; FROM_JOB=1

usage() {
  sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# A `while` loop over "$@", not `for a in "$@"`. The `for` form snapshots the positional
# list before the body runs, while `shift` always drops from the FRONT -- so after one
# shift, "$1" is the original argument #2, not the argument following the flag. That made
# `--from-job N` correct only when --from-job happened to be the first argument, and
# `run_all.sh --status --from-job 5` set FROM_JOB to the literal string "--from-job".
# Bash arithmetic then evaluated that bare word as 0 in `(( idx < FROM_JOB ))`, so instead
# of erroring it silently ran all 12 jobs from job 1.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --status)         STATUS_ONLY=1 ;;
    --skip-upload)    SKIP_UPLOAD=1 ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --from-job=*)     FROM_JOB="${1#*=}" ;;
    --from-job)
      shift
      [[ $# -gt 0 ]] || { echo "*** STOP: --from-job needs a job number." >&2; exit 2; }
      FROM_JOB="$1" ;;
    -h|--help)        usage; exit 0 ;;
    # Unknown options used to be ignored, which is how a misparsed value slipped through
    # unnoticed. Refuse instead.
    *) echo "*** STOP: unknown option: $1" >&2
       echo "    valid: --status --skip-upload --skip-preflight --from-job N|--from-job=N" >&2
       exit 2 ;;
  esac
  shift
done

if ! [[ "$FROM_JOB" =~ ^[1-9][0-9]*$ ]]; then
  echo "*** STOP: --from-job must be a positive integer, got '$FROM_JOB'." >&2
  exit 2
fi
# --- ARGPARSE END ---

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

table() {
  echo
  echo "------------------------------------------------------------------------------"
  python -u -m rewrite.run_rewrite --status --config-root "$REPO_ROOT" || true
  echo "------------------------------------------------------------------------------"
}

if [[ "$STATUS_ONLY" == "1" ]]; then table; exit 0; fi

echo "=============================================================================="
echo " rewrite-vllm :: full run"
echo " started $(date -Is)"
echo "=============================================================================="

# ---- 0. preflight gates everything -------------------------------------------------
if [[ "$SKIP_PREFLIGHT" == "0" ]]; then
  echo; echo "### preflight"
  python -u scripts/preflight.py --config-root "$REPO_ROOT"
else
  echo "### preflight SKIPPED (--skip-preflight) -- you are on your own"
fi

# ---- 1. model ----------------------------------------------------------------------
echo; echo "### model"
python -u scripts/01_download_model.py --config-root "$REPO_ROOT"

# ---- 2. data -----------------------------------------------------------------------
echo; echo "### data (all six arms; quality-base is verified but never rewritten)"
python -u scripts/02_download_data.py --config-root "$REPO_ROOT"

# ---- 3. the 12 jobs, strictly sequentially -----------------------------------------
echo; echo "### job table BEFORE"
table

mapfile -t JOBS < <(python - "$REPO_ROOT" <<'PYJ'
import sys
sys.path.insert(0, sys.argv[1] + "/src")
from rewrite.config import load_config, enumerate_jobs
for j in enumerate_jobs(load_config(sys.argv[1])):
    print(f"{j.arm} {j.prompt.id}")
PYJ
)
echo "### ${#JOBS[@]} jobs to run, one at a time"

idx=0
for spec in "${JOBS[@]}"; do
  idx=$((idx+1))
  read -r ARM PID <<< "$spec"
  if (( idx < FROM_JOB )); then
    echo "  [$idx/${#JOBS[@]}] $ARM/$PID  -- skipped (--from-job=$FROM_JOB)"
    continue
  fi
  if python -u -m rewrite.run_rewrite --arm "$ARM" --prompt-id "$PID" \
        --config-root "$REPO_ROOT" --verify >/dev/null 2>&1; then
    echo "  [$idx/${#JOBS[@]}] $ARM/$PID  -- already DONE, skipping (no model load)"
    continue
  fi
  echo
  echo "=== [$idx/${#JOBS[@]}] $ARM/$PID  $(date -Is) ==="
  bash scripts/03_run_job.sh "$ARM" "$PID"
done

# ---- 4. postprocess: trim then shuffle, per job -------------------------------------
echo; echo "### postprocess (trim -> shuffle, within (arm, prompt) only)"
python -u scripts/04_postprocess.py --config-root "$REPO_ROOT"

# ---- 5. upload ---------------------------------------------------------------------
if [[ "$SKIP_UPLOAD" == "0" ]]; then
  echo; echo "### upload"
  python -u scripts/05_upload_to_hf.py --config-root "$REPO_ROOT"
else
  echo; echo "### upload SKIPPED (--skip-upload)"
fi

echo; echo "### job table AFTER"
table
echo
echo "=============================================================================="
echo " finished $(date -Is)"
echo "=============================================================================="
