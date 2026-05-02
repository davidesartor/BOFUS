#!/usr/bin/env bash

declare -A SWEEP_RESOURCES=(
    [sinc]="4G 2:00:00"
    [gramacylee]="4G 2:00:00"
    [ackley]="4G 2:00:00"
    [hartmann]="4G 2:00:00"
    [rosenbrock]="4G 2:00:00"
    [pendulum]="8G 4:00:00"
    [pinwheel]="8G 8:00:00"
    [brachistochrone]="4G 4:00:00"
    [mnist]="30G 8:00:00"
)

deadline_hours=${1:-72}

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
multidim_fns="ackley|hartmann|rosenbrock|pendulum"
lengthscales=(0.4 0.2 0.1 0.05)
profiles=(rbf matern52 matern32 matern12)
seeds=($(seq 0 15))


LOG="sweep.log"
DEADLINE=$(( $(date +%s) + deadline_hours * 3600 ))
MAX_JOBS_PER_SUBMISSION=200

echo "[$(date)] Watcher started (PID $$), deadline in ${deadline_hours}h at $(date -d @$DEADLINE)" | tee -a "$LOG"

ALL_COMBOS=()
for target_fn in "${!SWEEP_RESOURCES[@]}"; do
for seed in "${seeds[@]}"; do
for profile in "${profiles[@]}"; do
for lengthscale in "${lengthscales[@]}"; do
for variant in "${variants[@]}"; do
    read -r method extra_flags <<< "$variant"
    if [[ "$method" == "vellanky" && "$target_fn" =~ ^($multidim_fns)$ ]]; then
        continue
    fi
    ALL_COMBOS+=("$target_fn $profile $lengthscale $seed $variant")
done
done
done
done
done

variant_to_dir() {
    local method=$1 extra_flags=${2:-}
    if [[ "$extra_flags" == *"--disable_natural_gradient"* ]]; then
        echo "${method}_no_natural_grad"
    elif [[ "$extra_flags" == *"--sample_candidates_from_gp"* ]]; then
        echo "${method}_sample_from_gp"
    elif [[ "$extra_flags" == *"--reduced_grid"* ]]; then
        echo "${method}_reduced_grid"
    else
        echo "$method"
    fi
}

submit_sweep() {
    local target_fn=$1 memory=$2 time=$3

    local combos=()
    for entry in "${ALL_COMBOS[@]}"; do
        [[ "${entry%% *}" != "$target_fn" ]] && continue
        read -r _ profile lengthscale seed method extra_flags <<< "$entry"
        local dir result
        dir=$(variant_to_dir "$method" "$extra_flags")
        result="results/${target_fn}/${dir}/${profile}_lengthscale_${lengthscale}/seed_${seed}.pkl"
        if [[ ! -e "$result" ]]; then
            combos+=("$profile $lengthscale $seed $method${extra_flags:+ $extra_flags}")
            [[ ${#combos[@]} -ge $MAX_JOBS_PER_SUBMISSION ]] && break 
        fi
    done

    local n=${#combos[@]}
    if [[ $n -eq 0 ]]; then
        echo "[$(date)] $target_fn: all results exist, skipping" >> "$LOG"
        return
    fi

    echo "[$(date)] $target_fn: submitting $n jobs" | tee -a "$LOG"
    mkdir -p logs/

    sbatch --job-name="sweep_${target_fn}" <<EOF >> "$LOG" 2>&1
#!/usr/bin/env bash
#SBATCH --output=logs/${target_fn}_%A/%a.out
#SBATCH --array=0-$((n - 1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=${memory}
#SBATCH --time=${time}
#SBATCH --partition=cpu

combos=($(printf '"%s" ' "${combos[@]}"))

read -r profile lengthscale seed method extra_flags <<< "\${combos[\$SLURM_ARRAY_TASK_ID]}"

uv run run.py \
    --method=\$method \
    --profile=\$profile \
    --target_fn=${target_fn} \
    --lengthscale=\$lengthscale \
    --seed=\$seed \
    --initial_acquisitions=10 \
    --minimum_k=1 \
    --maximum_k=10 \
    --acquisitions_each_k=10 \
    --acquisition_raw_samples=1024 \
    --acquisition_max_restarts=16 \
    \$extra_flags
EOF
}

while true; do
    if (( $(date +%s) >= DEADLINE )); then
        echo "[$(date)] ${deadline_hours}h elapsed, exiting." | tee -a "$LOG"
        exit 0
    fi

    ACTIVE=$(squeue --me --format="%j" --states=RUNNING,PENDING,COMPLETING 2>/dev/null)

    for target_fn in "${!SWEEP_RESOURCES[@]}"; do
        if echo "$ACTIVE" | grep -qF "sweep_${target_fn}"; then
            echo "[$(date)] $target_fn: active, skipping" >> "$LOG"
        else
            read -r memory time <<< "${SWEEP_RESOURCES[$target_fn]}"
            submit_sweep "$target_fn" "$memory" "$time"
        fi
    done

    sleep 600
done