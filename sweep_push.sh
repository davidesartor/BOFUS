#!/usr/bin/env bash
#SBATCH --job-name=push
#SBATCH --output=logs/push/%A_%a.out
#SBATCH --array=0-63
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --partition=cpu

mkdir -p logs/push

echo "Running push | seed=$SLURM_ARRAY_TASK_ID"


# LHS
uv run run.py \
  --config configs/push.yaml \
  --seed=$SLURM_ARRAY_TASK_ID \
  --acquisition_strategy.class_path=LHS \
  --acquisition_strategy.init_args.multi_starts=1000 &

# Voronoi
uv run run.py \
  --config configs/push.yaml \
  --seed=$SLURM_ARRAY_TASK_ID \
  --acquisition_strategy.class_path=Voronoi \
  --acquisition_strategy.init_args.multi_starts=1000 &

# BFGS
uv run run.py \
  --config configs/push.yaml \
  --seed=$SLURM_ARRAY_TASK_ID \
  --acquisition_strategy.class_path=BFGS \
  --acquisition_strategy.init_args.multi_starts=100 &

wait
