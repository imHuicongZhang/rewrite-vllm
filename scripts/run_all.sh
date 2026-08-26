#!/usr/bin/env bash
# The whole run, end to end, resumable:
#
#   preflight -> model -> data -> 10 jobs SEQUENTIALLY -> postprocess
#
# UPLOAD IS DISABLED BY DEFAULT (configs/data.yaml upload.enabled: false). Delivery of the
# finished data is arranged separately; the run ends with parquet under out_root/shuffled/.
# --skip-upload and upload.enabled are BOTH veto-only -- neither can switch upload ON when
# the other says off -- so the two can never contradict each other.
#
# Safe to re-run after ANY interruption. Finished shards are skipped via their .done
# sidecars; finished jobs are skipped without even loading a model.
#
#   bash scripts/run_all.sh                 # everything
#   bash scripts/run_all.sh --status        # just print the job table and exit
#   bash scripts/run_all.sh --skip-upload   # redundant while upload is disabled
#   bash scripts/run_all.sh --from-job 5    # resume at job 5 (1-based, see the table)
#   bash scripts/run_all.sh --postprocess-only [--arm A --prompt-id pN]
#
# JOBS RUN ONE AT A TIME, ON PURPOSE. Each job already uses every GPU. Running two would
# make each vLLM engine try to reserve gpu_memory_utilization=0.85 of the same card and
# OOM -- and 0.85 is a locked source-parity value, not a knob. Do not parallelise this.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- ARGPARSE BEGIN (tests/test_integration.py extracts between these markers) ---
STATUS_ONLY=0; SKIP_UPLOAD=0; SKIP_PREFLIGHT=0; FROM_JOB=1
DO_PREPARE=1; DO_GENERATE=1; DO_POST=1; SKIP_CALIBRATION=0
POST_ARM=""; POST_PROMPT=""; POST_ONLY=0

usage() {
  sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# A `while` loop over "$@", not `for a in "$@"`. The `for` form snapshots the positional
# list before the body runs, while `shift` always drops from the FRONT -- so after one
# shift, "$1" is the original argument #2, not the argument following the flag. That made
# `--from-job N` correct only when --from-job happened to be the first argument, and
# `run_all.sh --status --from-job 5` set FROM_JOB to the literal string "--from-job".
# Bash arithmetic then evaluated that bare word as 0 in `(( idx < FROM_JOB ))`, so instead
# of erroring it silently ran all 10 jobs from job 1.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --status)         STATUS_ONLY=1 ;;
    --skip-upload)    SKIP_UPLOAD=1 ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --from-job=*)     FROM_JOB="${1#*=}" ;;
    # --- multi-node roles. On one node: --prepare-only, then --postprocess-only at the
    # --- end. On every node in between: --generate-only. See GUIDE section 3.
    --prepare-only)     DO_GENERATE=0; DO_POST=0 ;;
    --generate-only)    DO_PREPARE=0; DO_POST=0 ;;
    --postprocess-only) DO_PREPARE=0; DO_GENERATE=0; SKIP_PREFLIGHT=1; POST_ONLY=1 ;;
    --skip-calibration) SKIP_CALIBRATION=1 ;;
    # --- postprocess job scoping. Different nodes may take DIFFERENT jobs; a per-job
    # --- lock makes that safe and stops two nodes picking the same one. Without these
    # --- flags every node sweeps the whole list and the locks sort out who takes what.
    --arm)
      shift
      [[ $# -gt 0 ]] || { echo "*** STOP: --arm needs an arm name." >&2; exit 2; }
      POST_ARM="$1" ;;
    --arm=*)          POST_ARM="${1#*=}" ;;
    --prompt-id)
      shift
      [[ $# -gt 0 ]] || { echo "*** STOP: --prompt-id needs a prompt id." >&2; exit 2; }
      POST_PROMPT="$1" ;;
    --prompt-id=*)    POST_PROMPT="${1#*=}" ;;
    --from-job)
      shift
      [[ $# -gt 0 ]] || { echo "*** STOP: --from-job needs a job number." >&2; exit 2; }
      FROM_JOB="$1" ;;
    -h|--help)        usage; exit 0 ;;
    # Unknown options used to be ignored, which is how a misparsed value slipped through
    # unnoticed. Refuse instead.
    *) echo "*** STOP: unknown option: $1" >&2
       echo "    valid: --status --skip-upload --skip-preflight --skip-calibration" >&2
       echo "           --from-job N|--from-job=N" >&2
       echo "           --prepare-only --generate-only --postprocess-only" >&2
       echo "           --arm NAME --prompt-id pN   (scope --postprocess-only to one job)" >&2
       exit 2 ;;
  esac
  shift
done

if ! [[ "$FROM_JOB" =~ ^[1-9][0-9]*$ ]]; then
  echo "*** STOP: --from-job must be a positive integer, got '$FROM_JOB'." >&2
  exit 2
fi
if [[ -n "$POST_PROMPT" && -z "$POST_ARM" ]]; then
  echo "*** STOP: --prompt-id needs --arm as well; a prompt id alone is ambiguous." >&2
  exit 2
fi
# Only meaningful with --postprocess-only. On a full run they would silently mean
# "generate all ten jobs, then postprocess one of them", which nobody wants and which
# reads like a scoping flag for the whole run.
if [[ -n "$POST_ARM" && "$POST_ONLY" != "1" ]]; then
  echo "*** STOP: --arm/--prompt-id scope the POSTPROCESS stage, so they need" >&2
  echo "    --postprocess-only. To run ONE generation job:  bash scripts/03_run_job.sh ARM pN" >&2
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
print("UPLOAD_ENABLED=" + ("1" if c.data["upload"]["enabled"] else "0"))
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
# Always: every node needs local weights, and it is idempotent.
echo; echo "### model"
python -u scripts/01_download_model.py --config-root "$REPO_ROOT"

# ---- 2. data -----------------------------------------------------------------------
# Sharding is per-arm lock-protected, so running this on several nodes at once is safe --
# the losers wait for the manifest rather than duplicating the work. On a multi-node run
# it is still cleaner to do it once with --prepare-only before fanning out.
if [[ "$DO_PREPARE" == "1" ]]; then
  echo; echo "### data (all five arms; the raw shared-core + quality-base halves are NOT downloaded)"
  python -u scripts/02_download_data.py --config-root "$REPO_ROOT"
else
  echo; echo "### data: skipped (not this node's role); waiting for manifests"
  python -u scripts/02_download_data.py --config-root "$REPO_ROOT" --wait-only
fi

# ---- 2b. calibration: measure real throughput before committing days to this --------
if [[ "$DO_GENERATE" == "1" && "$SKIP_CALIBRATION" == "0" ]]; then
  echo; echo "### calibration"
  python -u scripts/06_calibrate.py --config-root "$REPO_ROOT" || \
    echo "  (calibration failed; continuing -- it is advisory, not a gate)"
fi

if [[ "$DO_GENERATE" != "1" ]]; then
  echo; echo "### generation: skipped (not this node's role)"
fi

# ---- 3. the 10 jobs, strictly sequentially -----------------------------------------
if [[ "$DO_GENERATE" == "1" ]]; then
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
fi   # DO_GENERATE

# ---- 4. postprocess: trim then shuffle, per job -------------------------------------
# NEVER TWO NODES ON THE SAME JOB -- they would share an output dir AND a bucket temp dir,
# and bucketed_shuffle unlinks buckets as it consumes them, which is real corruption
# rather than duplicated work. But DIFFERENT jobs do not collide: separate output dirs,
# and tmp_root is node-local so bucket dirs cannot overlap either. So this fans out.
#
# 04_postprocess.py takes a per-job lock (the round-6 heartbeat machinery) and sweeps the
# job list, skipping whatever another node holds. Run the same command on every node that
# has finished generating and the work distributes itself. There are only 10 jobs, so
# parallelism caps at 10 nodes.
if [[ "$DO_POST" == "1" ]]; then
echo; echo "### postprocess (trim -> shuffle, within (arm, prompt) only)"
POST_ARGS=()
[[ -n "$POST_ARM" ]]    && POST_ARGS+=(--arm "$POST_ARM")
[[ -n "$POST_PROMPT" ]] && POST_ARGS+=(--prompt-id "$POST_PROMPT")
python -u scripts/04_postprocess.py --config-root "$REPO_ROOT" ${POST_ARGS[@]+"${POST_ARGS[@]}"}

# ---- 5. upload ---------------------------------------------------------------------
# Two independent vetoes, and that is the whole reconciliation: upload runs only if the
# config allows it AND the flag did not suppress it. Neither can switch it ON alone, so
# `--skip-upload` with upload.enabled: true means skip, and no flag at all with
# upload.enabled: false also means skip. They cannot contradict each other.
if [[ "$UPLOAD_ENABLED" == "0" ]]; then
  echo; echo "### upload SKIPPED (disabled in configs/data.yaml: upload.enabled: false)"
  echo "    The finished data is complete on disk at ${OUT_ROOT}/shuffled/ -- see"
  echo "    docs/GUIDE_FOR_TIANJIAN.md section 3.6. Nothing further is required."
elif [[ "$SKIP_UPLOAD" == "1" ]]; then
  echo; echo "### upload SKIPPED (--skip-upload)"
else
  echo; echo "### upload"
  python -u scripts/05_upload_to_hf.py --config-root "$REPO_ROOT"
fi
else
  echo; echo "### postprocess + upload: skipped (not this node's role)"
fi   # DO_POST

echo; echo "### job table AFTER"
table
echo
echo "=============================================================================="
echo " finished $(date -Is)"
echo "=============================================================================="
