#!/usr/bin/env bash
# bash sweep.sh sinc 4G 2:00:00
# bash sweep.sh gramacylee 4G 2:00:00
# bash sweep.sh rosenbrock 4G 2:00:00
# bash sweep.sh ackley 4G 2:00:00
# bash sweep.sh hartmann 4G 2:00:00
# bash sweep.sh pendulum 4G 4:00:00
# bash sweep.sh pinwheel 4G 16:00:00
# bash sweep.sh mnist 12G 8:00:00

target_fn=${1:?Usage: bash $0 <target_fn> <memory> <time> [--force_rerun]}
memory=${2:?Usage: bash $0 <target_fn> <memory> <time> [--force_rerun]}
time=${3:?Usage: bash $0 <target_fn> <memory> <time> [--force_rerun]}
force_rerun=false
[[ "${4}" == "--force_rerun" ]] && force_rerun=true

profiles=(rbf matern52 matern32)
lengthscales=(0.3 0.1 0.03)
seeds=($(seq 0 15))

# "method [extra_flags...]"
variants=(
    "random"
    "wycoff"
    "vien"
    "shilton"
    "vellanky"
    "kundu"
    "wycoff --disable_natural_gradient"
    "vien   --disable_natural_gradient"
    "wycoff --sample_candidates_from_gp"
    "shilton --reduced_grid"
)

# GENERATE A LIST OF COMBINATIONS TO RUN 
# only include missing combinations unless --force_rerun is specified
combos=()
for profile in "${profiles[@]}"; do
for lengthscale in "${lengthscales[@]}"; do
for seed in "${seeds[@]}"; do
for variant in "${variants[@]}"; do
    read -r method extra_flags <<< "$variant"
    # skip invalid method-target_fn combinations
    if [[ "$method" == "vellanky" && "$target_fn" =~ ^(ackley|hartmann|pendulum)$ ]]; then
        continue
    fi
    # determine result path based on method and extra flags
    if [[ "$extra_flags" == *"--disable_natural_gradient"* ]]; then
        dir="${method}_no_natural_grad"
    elif [[ "$extra_flags" == *"--sample_candidates_from_gp"* ]]; then
        dir="${method}_sample_from_gp"
    elif [[ "$extra_flags" == *"--reduced_grid"* ]]; then
        dir="${method}_reduced_grid"
    else
        dir=$method
    fi
    result="results/${dir}/${profile}/${target_fn}/lengthscale_${lengthscale}/seed_${seed}.pkl"
    # only add to combos if result doesn't exist unless --force_rerun 
    if $force_rerun || [[ ! -e "$result" ]]; then
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