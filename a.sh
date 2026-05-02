#!/usr/bin/env bash

profiles=(rbf matern52 matern32 matern12)
lengthscales=(0.4 0.2 0.1 0.05)
seeds=($(seq 0 15))

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

target_fns=(
    "sinc" 
    "gramacylee" 
    "ackley" 
    "hartmann" 
    "rosenbrock" 
    "pendulum" 
    "pinwheel"
    "brachistochrone"
    "mnist" 
)


for target_fn in "${target_fns[@]}"; do
    echo "$target_fn"
    total=0
    for variant in "${variants[@]}"; do
        missing=0
        for profile in "${profiles[@]}"; do
        for lengthscale in "${lengthscales[@]}"; do
        for seed in "${seeds[@]}"; do
            read -r method extra_flags <<< "$variant"
            if [[ "$method" == "vellanky" && "$target_fn" =~ ^(ackley|hartmann|rosenbrock|pendulum)$ ]]; then
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
            result="results/${target_fn}/${dir}/${profile}_lengthscale_${lengthscale}/seed_${seed}.pkl"
            [[ ! -e "$result" ]] && ((missing++))
        done
        done
        done
        if [[ $missing -gt 0 ]]; then
            echo "      $variant: $missing missing"
        fi
        total=$((total + missing))
    done
    echo "Total missing: $total"
    echo
    echo
done