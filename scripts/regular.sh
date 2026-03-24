#!/bin/bash
#SBATCH --job-name=aniline-ccpvtz
#SBATCH --account=nvr_qualg_lmbm
#SBATCH --partition=batch_block1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-%j.txt
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-%j.txt
#SBATCH --open-mode=append

set -euo pipefail

export MASTER_PORT=$(expr 10000 + $(echo -n "$SLURM_JOBID" | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

PROJECT_DIR=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/ccsd_gpu
cd "$PROJECT_DIR"

uv sync
source scripts/setup_gpu_env.sh

echo "sbatch scripts/regular.sh $* # $SLURM_JOB_ID" >> slurmlog.txt

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $PROJECT_DIR"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "============================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

if [ "$#" -gt 0 ]; then
  CMD=("$@")
else
  CMD=(python examples/h2o_ccsd_grad.py --molecule aniline --basis cc-pvtz)
fi

echo "Running: uv run ${CMD[*]}"
echo "============================================"

srun uv run "${CMD[@]}"

echo "============================================"
echo "Complete!"
echo "End Time: $(date)"
echo "============================================"
