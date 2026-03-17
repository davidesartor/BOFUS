#!/usr/bin/env bash
kernels=(
  matern52
  squaredexponential
  # matern32
  # matern12
)

# Run BFGS acquisition strategy
for k in "${kernels[@]}"; do
  uv run run.py --kernel "$k" bfgs --multi_starts 16 &
  uv run run.py --kernel "$k" lhs --points 30 &
done
wait