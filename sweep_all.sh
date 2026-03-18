#!/bin/bash

# Submit all sweep_*.sh files using sbatch
for sweep_file in sweep_*.sh; do
    if [ -f "$sweep_file" ] && [ "$sweep_file" != "sweep_all.sh" ]; then
        echo "Submitting $sweep_file..."
        sbatch "$sweep_file"
    fi
done

echo "All sweep files submitted."