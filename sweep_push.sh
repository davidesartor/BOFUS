#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --output=logs/push/%A_%a.out
#SBATCH --array=0-7                   # 4 kernels × 2 strategies 
#SBATCH --ntasks=1                    # 1 process per job
#SBATCH --cpus-per-task=2             # 2 cores to avoid locks
#SBATCH --mem=8G                      # 8 GB RAM for each job
#SBATCH --time=04:00:00           
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

num_kernels=${#kernels[@]}        
num_strategies=${#strategies[@]}
total_combinations=$(( num_kernels * num_strategies ))

strategy_idx=$(( SLURM_ARRAY_TASK_ID % num_strategies ))
kernel_idx=$(( SLURM_ARRAY_TASK_ID / num_strategies ))

strategy="${strategies[$strategy_idx]}"
kernel="${kernels[$kernel_idx]}"

mkdir -p logs/push


echo "Running test_function=push | strategy=$strategy | kernel=$kernel"

case "$strategy" in
  bfgs)
    uv run run.py --test_function push --initial_acquisitions 14 --total_acquisitions 600 --kernel "$kernel" bfgs --multi_starts 14 --max_iterations 100
    ;;
  lhs)
    uv run run.py --test_function push --initial_acquisitions 14 --total_acquisitions 600 --kernel "$kernel" lhs --points 14
    ;;
esac