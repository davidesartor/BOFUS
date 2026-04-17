#!/usr/bin/env bash
runs=${1:?Usage: bash $0 <runs> <target_fn>}
target_fn=${2:?Usage: bash $0 <runs> <target_fn>}

methods=(wycoff wycoff_gp vellanky kundu vien shilton)
profiles=(rbf matern32 matern52)
lenghtscales=(0.3 0.1 0.03)

    
for profile in "${profiles[@]}"; do
    for lenghtscale in "${lenghtscales[@]}"; do
        for method in "${methods[@]}"; do
            echo "Launching: $method $profile $target_fn l=$lenghtscale"
            bash run.sh $method $profile $target_fn $lenghtscale $runs &
            sleep 1
        done
    done
done