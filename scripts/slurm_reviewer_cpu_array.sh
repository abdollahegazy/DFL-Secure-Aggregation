#!/bin/bash --login
#SBATCH --job-name=dfl_review_cpu
#SBATCH --array=0-879%1000
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --requeue
#SBATCH --qos=scavenger

set -euo pipefail

REPO=/mnt/scratch/hegazyab/DFL-Secure-Aggregation
CONDA_ENV=${DFL_CONDA_ENV:-base}
CONDA_BIN=/mnt/home/hegazyab/miniconda3/bin/conda
MANIFEST=${MANIFEST:-data/sweeps/reviewer_fixes_v1/manifest.csv}

cd "$REPO"
mkdir -p logs

CONFIG=$(
  awk -F, -v id="$SLURM_ARRAY_TASK_ID" 'NR > 1 && $1 == id {gsub(/\r/, "", $2); print $2}' "$MANIFEST"
)
STATUS=$(
  awk -F, -v id="$SLURM_ARRAY_TASK_ID" 'NR > 1 && $1 == id {gsub(/\r/, "", $13); print $13}' "$MANIFEST"
)

if [[ -z "${CONFIG}" ]]; then
  echo "No config found for task ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

if [[ "${STATUS}" != "active" ]]; then
  echo "Skipping task ${SLURM_ARRAY_TASK_ID}; status=${STATUS}"
  exit 0
fi

echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "Config: ${CONFIG}"
echo "Status: ${STATUS}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Conda env: ${CONDA_ENV}"

cd "$REPO/experiments"

export PYTHONPATH="$REPO/src"
export MPLCONFIGDIR="/tmp/dfl_mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
#   python -c 'import sys; print("Python:", sys.executable); import torch, torchvision; print("torch:", torch.__version__); print("torchvision:", torchvision.__version__)'

"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  python simulate.py --config "$CONFIG"
