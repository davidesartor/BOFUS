#!/usr/bin/env bash
runs=${1:?Usage: bash $0 <runs> <target_fn>}
target_fn=${2:?Usage: bash $0 <runs> <target_fn>}

methods=(wycoff vellanky kundu vien shilton)
lenghtscales=(0.3 0.1 0.03)

for lenghtscale in "${lenghtscales[@]}"; do
    for method in "${methods[@]}"; do
        echo "Launching: $method $target_fn l=$lenghtscale"
        bash run.sh $method $target_fn $lenghtscale $runs &
        sleep 1
    done
done