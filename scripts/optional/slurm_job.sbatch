#!/bin/bash
# ============================================================================
#  OPTIONAL. NOT PART OF THE DEFAULT PATH. NOT SUPPORTED.
# ============================================================================
#  The default and tested way to run this pipeline is plain bash on one node:
#
#      bash scripts/run_all.sh
#
#  This file exists only in case you already have SLURM and would rather submit
#  than babysit a login shell. It is a thin wrapper: it does exactly what
#  run_all.sh does, inside one allocation that holds ALL the GPUs.
#
#  It is NOT an array job. The source pipeline used an 8-way array because its
#  cluster handed out one GPU at a time; this package instead launches N worker
#  processes inside one allocation, which is simpler and identical in behaviour.
#
#  Before using this:
#    1. bash scripts/run_all.sh must already work interactively.
#    2. Fill scheduler.slurm.* in configs/cluster.yaml and set scheduler.kind: slurm.
#    3. Replace the four directives below with values valid on YOUR cluster.
#       The ones commented out are from the ORIGINAL JHU cluster and will not
#       work anywhere else -- they are here only as a worked example.
#
#  Submit:   sbatch scripts/optional/slurm_job.sbatch
#  Resume:   submit the same file again. Finished shards are skipped.
# ============================================================================
#SBATCH --job-name=rewrite-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-rewrite-%j.out
#SBATCH --error=slurm-rewrite-%j.out
#
# ---- FILL THESE IN FOR YOUR CLUSTER ----------------------------------------
##SBATCH --partition=<your gpu partition>
##SBATCH --gres=gpu:<count>            # must equal compute.num_gpus
##SBATCH --cpus-per-task=<cores>       # >= compute.cpu_workers for postprocess
##SBATCH --mem=<memory>                # >= compute.shuffle_mem_bytes
##SBATCH --time=<walltime>             # resume is built in, so a short limit is fine
#
# (The original pipeline's own directives are recorded in docs/SOURCE_INVENTORY.md
#  section 11. They are specific to that site and will not work here -- do not copy them.)
##SBATCH --account=<your account>      # if your site requires one
#
# ----------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Everything else -- env activation, preflight, the 12 sequential jobs, postprocess,
# upload -- is exactly the bash path. Nothing scheduler-specific happens below.
exec bash scripts/run_all.sh
