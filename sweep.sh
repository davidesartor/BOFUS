#!/usr/bin/env bash

target_fn=${1:?Usage: bash $0 <target_fn> <memory> <time>}
memory=${2:?Usage: bash $0 <target_fn> <memory> <time>}
time=${3:?Usage: bash $0 <target_fn> <memory> <time>}

combos=()
for profile in "rbf" "matern52" "matern32"; do
for lengthscale in 0.3 0.1 0.03; do
for seed in $(seq 0 9); do
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
#SBATCH --mem=${memory}
#SBATCH --time=${time}
#SBATCH --partition=cpu

combos=($(printf '"%s" ' "${combos[@]}"))

read -r profile lengthscale seed <<< "\${combos[\$SLURM_ARRAY_TASK_ID]}"

run() {
    PYTHONUNBUFFERED=1 uv run run.py --profile=\$profile --target_fn=${target_fn} --lengthscale=\$lengthscale --seed=\$seed \\
    --initial_acquisitions=10 --minimum_k=1 --maximum_k=10 --acquisitions_each_k=10 \\
    --acquisition_raw_samples=1024 --acquisition_max_restarts=16 "\$@" &
}

# RUN ALL METHODS
run --method=wycoff
run --method=vien
run --method=shilton
run --method=vellanky
run --method=kundu

# EXTRA ABLATIONS:
run --method=wycoff --disable_natural_gradient
run --method=vien   --disable_natural_gradient
run --method=wycoff --sample_candidates_from_gp

wait
EOF