#!/usr/bin/env bash

target_fn=${1:?Usage: bash $0 <target_fn> <runs>}
runs=${2:?Usage: bash $0 <target_fn> <runs>}

methods=(wycoff wycoff_gp vellanky kundu vien shilton)
profiles=(rbf matern52 matern32)
lengthscales=(0.3 0.1 0.03)

combos=()
for profile in "${profiles[@]}"; do
for lengthscale in "${lengthscales[@]}"; do
for seed in $(seq 0 $((runs - 1))); do
    combos+=("$profile $lengthscale $seed")
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
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu

combos=($(printf '"%s" ' "${combos[@]}"))
methods=(${methods[@]})

read -r profile lengthscale seed <<< "\${combos[\$SLURM_ARRAY_TASK_ID]}"

for method in "\${methods[@]}"; do
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
        --acquisition_max_restarts=16 &
done

wait
EOF