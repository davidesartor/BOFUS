#!/usr/bin/env bash
runs=${1:?Usage: bash $0 <runs> <target_fn> <lenghtscale>}
target_fn=${2:?Usage: bash $0 <runs> <target_fn> <lenghtscale>}
lenghtscale=${3:?Usage: bash $0 <runs> <target_fn> <lenghtscale>}

methods=(wycoff vellanky kundu vien shilton)
for method in "${methods[@]}"; do
    echo "Launching: $method $target_fn l=$lenghtscale"
    bash run.sh $method $target_fn $lenghtscale $runs &
    sleep 1
done