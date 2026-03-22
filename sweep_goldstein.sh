#!/usr/bin/env bash
#SBATCH --job-name=goldstein
#SBATCH --output=logs/goldstein/%A_%a.out
#SBATCH --array=0-63
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu

mkdir -p logs/goldstein

echo "Running goldstein | seed=$SLURM_ARRAY_TASK_ID"

# LHS
uv run run.py \
  --config configs/goldstein.yaml \
  --seed=$SLURM_ARRAY_TASK_ID \
  --acquisition_strategy.class_path=LHS \
  --acquisition_strategy.init_args.multi_starts=100 &

# Voronoi
uv run run.py \
  --config configs/goldstein.yaml \
  --seed=$SLURM_ARRAY_TASK_ID \
  --acquisition_strategy.class_path=Voronoi \
  --acquisition_strategy.init_args.multi_starts=100 &

# BFGS
uv run run.py \
  --config configs/goldstein.yaml \
  --seed=$SLURM_ARRAY_TASK_ID \
  --acquisition_strategy.class_path=BFGS \
  --acquisition_strategy.init_args.multi_starts=10 &

wait


