#!/usr/bin/env bash
runs=${1:?Usage: bash $0 <runs> <lenghtscale>}
lenghtscale=${2:?Usage: bash $0 <runs> <lenghtscale>}

methods=(wycoff vellanky kundu vien shilton)
target_fns=(sinc mnist)

for target_fn in "${target_fns[@]}"; do
    for method in "${methods[@]}"; do
        echo "Launching: $method $target_fn l=$lenghtscale"
        bash run.sh $method $target_fn $lenghtscale $runs &
    done
done