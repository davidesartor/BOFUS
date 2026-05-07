## Overview

Code for submission (NeurIPS 2026) for reproducibility.
We propose a method for Functional Bayesian Optimization (FBO) that searches a sparse sub-manifold of an RKHS by jointly optimizing over basis point locations and coefficients. The repository contains implementations of Ours and four baselines (Vien, Kundu, Shilton, Vellanky), along with a JAX-compatible library of over 70 BO benchmark functions extended to the functional domain.

## Repository Structure

```
.
├── run.py               # Main entry point: run a single experiment
├── sweep.sh             # SLURM sweep script: launches all experiments
├── summarize.py         # Aggregate results into CSVs
├── plot.py              # Generate all figures from CSVs
├── src/
│   ├── gp.py            # Gaussian process surrogate models
│   ├── rkhs.py          # RKHS function representations
│   ├── kernels.py       # Kernel functions (RBF, Matérn)
│   ├── acquisition.py   # Acquisition functions and optimization
│   └── targets/         # Benchmark functions and real-world tasks
├── results/             # Experiment outputs (see below)
└── plots/               # Generated figures
```

## Installation

```bash
# Requires Python 3.11+. We use uv for dependency management.
pip install uv
uv sync
```

## Pre-computed Results

We provide an archive of all pre-computed results — no need to rerun the sweep to reproduce the figures. Download and extract:

```bash
tar -xzf results.tar.gz
```

The archive contains:
- `results/{target_fn}/{method}/{profile}_lengthscale_{lengthscale}/seed_{seed}.pkl` — raw per-run results
- `results/summary_all.csv` — aggregated summary across all hyperparameter combinations
- `results/summary_filtered.csv` — summary filtered to the best profile and lengthscale per (method, target)
- `results/ys_all.csv` — per-acquisition observation values for all runs
- `results/ys_filtered.csv` — same, filtered to the best hyperparameter combination

To regenerate the CSVs from the raw `.pkl` files (e.g. after running new experiments):

```bash
uv run summarize.py
```

## Reproducing Figures

With the results archive extracted and CSVs in place:

```bash
uv run plot.py
```

This writes all figures to `plots/`, organized as:

| Directory | Contents |
|---|---|
| `plots/method_comparison/` | Figure 1: running best per target, all methods |
| `plots/natural_gradient_ablation_ours/` | Appendix: natural gradient ablation for ours |
| `plots/natural_gradient_ablation_vien/` | Appendix: natural gradient ablation for Vien |
| `plots/candidates_sampling_ablation/` | Appendix: initial candidate distribution |
| `plots/kernel_profile_ablation/` | Appendix: GP kernel profile sweep |
| `plots/tables/` | Tables: average and final regret, best hyperparameters |
| `plots/f_visualizations/` | MNIST learned activation, brachistochrone path |

## Running Individual Experiments to Spot Check

A single run takes a method, target function, and hyperparameters:

```bash
uv run run.py \
    --method ours \
    --target_fn brachistochrone \
    --profile matern52 \
    --lengthscale 0.2 \
    --seed 0
```

**Methods:** `ours`, `vien`, `kundu`, `shilton`, `vellanky`, `random`

**Target functions:** `sinc1d`, `sinc2d`, `sinc3d`, `sinc4d`, `gramacylee`, `ackley`, `hartmann`, `rosenbrock`, `brachistochrone`, `pendulum`, `pinwheel`, `mnist`

**Ablation flags:**
- `--disable_natural_gradient` — use Euclidean gradient instead of natural gradient
- `--sample_candidates_from_gp` — sample LHS candidates from the GP prior (ours only)
- `--reduced_grid` — use a reduced grid for Shilton's method

Results are saved to `results/{target_fn}/{method}/{profile}_lengthscale_{lengthscale}/seed_{seed}.pkl`.

## Running the Full Sweep (SLURM)

The sweep script replicates all experiments in the paper. It submits SLURM array jobs per target function and reruns any missing results every 10 minutes until a deadline:

```bash
bash sweep.sh 72   # run for up to 72 hours (default)
```

The full sweep covers 6 methods × 4 profiles × 4 lengthscales × 16 seeds across 12 target functions, plus ablation variants. Estimated wall time per job ranges from 2 hours (sinc projections) to 20 hours (pinwheel). The script skips jobs whose output file already exists.

After the sweep completes, regenerate the CSVs and plots:

```bash
uv run summarize.py
uv run plot.py
```

## Benchmark Functions

We provide a JAX-compatible implementation of the [virtual library of simulation experiments](http://www.sfu.ca/~ssurjano) (Surjanovic & Bingham 2013), extended to the functional domain via the ridge function construction described in Section 5 of the paper. All functions support `jit` and `grad`.

```python
from src.targets import Ridge, virtual_library

# Hartmann3 as a functional benchmark (d=3 dimensional input)
target = Ridge(virtual_library.Hartmann3(), d=3)
```