#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --output=logs/%A_%a.out
#SBATCH --array=0-7                   # 4 kernels × 2 strategies = 8 jobs
#SBATCH --ntasks=1                    # 1 process per job
#SBATCH --cpus-per-task=2             # 2 cores to avoid locks
#SBATCH --mem=4G                      # 4 GB RAM for each job
#SBATCH --time=01:00:00           
#SBATCH --partition=cpu

strategies=(
  bfgs
  lhs
)
kernels=(
  matern52
  squaredexponential
  matern32
  matern12
)

# Two strategies per kernel: bfgs (index 0) and lhs (index 1)
num_kernels=${#kernels[@]}          # 4
strategy_idx=$(( SLURM_ARRAY_TASK_ID % 2 ))
kernel_idx=$(( SLURM_ARRAY_TASK_ID / 2 ))

strategy="${strategies[$strategy_idx]}"
kernel="${kernels[$kernel_idx]}"

mkdir -p logs


echo "Running strategy=$strategy | kernel=$kernel"

case "$strategy" in
  bfgs)
    uv run run.py --kernel "$kernel" bfgs --multi_starts 16 --max_iterations 100
    ;;
  lhs)
    uv run run.py --kernel "$kernel" lhs --points 30
    ;;
esac