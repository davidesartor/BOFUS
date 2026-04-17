#!/usr/bin/env bash

target_fn=${1:?Usage: bash $0 <target_fn> <seed_start-seed_end> <memory> <time>}
seed_range=${2:?Usage: bash $0 <target_fn> <seed_start-seed_end> <memory> <time>}
memory=${3:?Usage: bash $0 <target_fn> <seed_start-seed_end> <memory> <time>}
time=${4:?Usage: bash $0 <target_fn> <seed_start-seed_end> <memory> <time>}

seed_start=${seed_range%-*}
seed_end=${seed_range#*-}

methods=(wycoff wycoff_gp vellanky kundu vien shilton)
profiles=(rbf matern52 matern32)
lengthscales=(0.3 0.1 0.03)

combos=()
for profile in "${profiles[@]}"; do
for lengthscale in "${lengthscales[@]}"; do
for method in "${methods[@]}"; do
for seed in $(seq $seed_start $seed_end); do
    combos+=("$profile $lengthscale $method $seed")
done
done
done
done

n=${#combos[@]}
mkdir -p logs/

sbatch --job-name="sweep_${target_fn}" <<EOF
#!/usr/bin/env bash
#SBATCH --output=logs/${target_fn}_%A/%a.out
#SBATCH --array=0-$((n - 1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=${memory}
#SBATCH --time=${time}
#SBATCH --partition=cpu

combos=($(printf '"%s" ' "${combos[@]}"))

read -r profile lengthscale method seed <<< "\${combos[\$SLURM_ARRAY_TASK_ID]}"

PYTHONUNBUFFERED=1 uv run run.py \\
    --method=\$method \\
    --profile=\$profile \\
    --target_fn=${target_fn} \\
    --lengthscale=\$lengthscale \\
    --seed=\$seed \\
    --initial_acquisitions=10 \\
    --minimum_k=1 \\
    --maximum_k=10 \\
    --acquisitions_each_k=10 \\
    --acquisition_raw_samples=1024 \\
    --acquisition_max_restarts=16
EOF