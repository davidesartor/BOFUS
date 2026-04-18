#!/usr/bin/env bash

target_fn=${1:?Usage: bash $0 <target_fn> <memory> <time> [--skip_existing]}
memory=${2:?Usage: bash $0 <target_fn> <memory> <time> [--skip_existing]}
time=${3:?Usage: bash $0 <target_fn> <memory> <time> [--skip_existing]}
skip_existing=false
[[ "${4}" == "--skip_existing" ]] && skip_existing=true

profiles=(rbf matern52 matern32)
lengthscales=(0.3 0.1 0.03)
seeds=($(seq 0 9))

# "method [extra_flags...]"
variants=(
    "wycoff"
    "vien"
    "shilton"
    "vellanky"
    "kundu"
    "wycoff --disable_natural_gradient"
    "vien   --disable_natural_gradient"
    "wycoff --sample_candidates_from_gp"
)

# GENERATE A LIST OF COMBINATIONS TO RUN 
# if --rerun, only include combinations for which the result file is missing
combos=()
for profile in "${profiles[@]}"; do
for lengthscale in "${lengthscales[@]}"; do
for seed in "${seeds[@]}"; do
for variant in "${variants[@]}"; do
    read -r method extra_flags <<< "$variant"
    if [[ "$extra_flags" == *"--disable_natural_gradient"* ]]; then
        dir="${method}_no_natural_grad"
    elif [[ "$extra_flags" == *"--sample_candidates_from_gp"* ]]; then
        dir="${method}_sample_from_gp"
    else
        dir=$method
    fi
    result="results/${dir}/${profile}/${target_fn}/lengthscale_${lengthscale}/seed_${seed}.pkl"
    if ! $skip_existing || [[ ! -e "$result" ]]; then
        combos+=("$profile $lengthscale $seed $variant")
    fi
done
done
done
done

n=${#combos[@]}
if [[ $n -eq 0 ]]; then
    echo "All results already exist, nothing to do!"
    exit 0
fi
echo "Submitting $n jobs..."

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

# combo format: "profile lengthscale seed method [extra_flags...]"
read -r profile lengthscale seed method extra_flags <<< "\${combos[\$SLURM_ARRAY_TASK_ID]}"

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
    --acquisition_max_restarts=16 \\
    \$extra_flags
EOF