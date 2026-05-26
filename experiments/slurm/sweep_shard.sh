#!/bin/bash --login
#SBATCH --job-name=dfl-shard
#SBATCH --constraint=NOAUTO:grace
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/gs21/scratch/hegazyab/DFL-Secure-Aggregation/experiments/slurm/logs/%x-%j.out
#SBATCH --error=/mnt/gs21/scratch/hegazyab/DFL-Secure-Aggregation/experiments/slurm/logs/%x-%j.err

set -eo pipefail

cd /mnt/gs21/scratch/hegazyab/DFL-Secure-Aggregation


{ conda deactivate || true; } 2>/dev/null
{ conda deactivate || true; } 2>/dev/null
{ module purge || true; } 2>/dev/null

source /mnt/scratch/hegazyab/arm/miniforge3/bin/activate


# Count remaining configs — resubmit only if work is left
DONE=$(ls experiments/data/results/v1/*.json 2>/dev/null | wc -l)



python experiments/run_sweep.py \
  --sweep-id v1 \
  --device cuda:0 \
  --shard "${SHARD}" \
  --log-metrics \
  --num-shards 3 


TOTAL=1680
if [ "$DONE" -lt "$TOTAL" ]; then
  sbatch --job-name=dfl-shard${SHARD} --export=ALL,SHARD=${SHARD} "${BASH_SOURCE[0]}"
fi

# run like sbatch --job-name=dfl-shard0 --nodelist=nch-000 --export=ALL,SHARD=0 run_shard.sb
