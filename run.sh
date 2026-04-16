#!/usr/bin/env bash

method=${1:?Usage: bash $0 <method> <target_fn> <lengthscale> <runs>}
target_fn=${2:?Usage: bash $0 <method> <target_fn> <lengthscale> <runs>}
lengthscale=${3:?Usage: bash $0 <method> <target_fn> <lengthscale> <runs>}
runs=${4:?Usage: bash $0 <method> <target_fn> <lengthscale> <runs>}

mkdir -p logs/

sbatch --job-name="${method}_${target_fn}_${lengthscale}" <<EOF
#!/usr/bin/env bash
#SBATCH --output=logs/%A_${method}_${target_fn}_${lengthscale}/%a.out
#SBATCH --array=0-$((runs - 1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --partition=cpu

uv run run.py \
    --method=$method \
    --target_fn=$target_fn \
    --lengthscale=$lengthscale \
    --seed=\$SLURM_ARRAY_TASK_ID \
    --initial_acquisitions=10 \
    --minimum_k=1 \
    --maximum_k=10 \
    --acquisitions_each_k=10 \
    --acquisition_raw_samples=1024 \
    --acquisition_max_restarts=16
EOF