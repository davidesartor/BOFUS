import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float
from tqdm import tqdm
import scipy as sp
import numpy as np
import argparse
import os
import time
import random

from src import gp, test_functions, acquisition

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))
N_RUNS = 64


def run(
    # simualtion parameters
    seed: int,
    test_function: str,
    initial_acquisitions: int,
    total_acquisitions: int,
    # gaussian process parameters
    kernel: gp.Kernel,
    max_fit_iterations: int,
    warm_start_update: bool,
    # acquisition strategy parameters
    acquisition_strategy: str,
    *args,
    **kwargs,
) -> Float[Array, "k"]:
    # set random seeds
    np.random.seed(seed)
    random.seed(seed)

    # select test function
    if test_function == "goldstein":
        test_fn = test_functions.GoldsteinPrice()
    elif test_function == "rover":
        test_fn = test_functions.Rover()
    elif test_function == "push":
        test_fn = test_functions.Push()
    elif test_function == "lunar":
        test_fn = test_functions.LunarLander()
    else:
        raise ValueError(f"Unknown test function: {test_function}")
    
    # select acquisition strategy
    if acquisition_strategy == "lhs":
        acquisition_fn = acquisition.LHS()
    elif acquisition_strategy == "bfgs":
        acquisition_fn = acquisition.BFGS()
    else:
        raise ValueError(f"Unknown acquisition strategy: {acquisition_strategy}")
    
    # initial acquisitions
    lhs_sampler = sp.stats.qmc.LatinHypercube(d=test_fn.d, seed=np.random.mtrand._rand)
    x0 = lhs_sampler.random(n=initial_acquisitions)
    y0 = jnp.array([test_fn(xi) for xi in x0])
    model = gp.GaussianProcess.fit(
        x0, y0, kernel=kernel, max_iterations=max_fit_iterations
    )

    # experiment loop
    ymin = np.empty(total_acquisitions + 1)
    ymin[:initial_acquisitions] = np.nan
    ymin[initial_acquisitions] = model.y.min()
    for i in tqdm(range(initial_acquisitions, total_acquisitions)):
        x = acquisition_fn(model, *args, **kwargs)
        y = jnp.array([test_fn(xi) for xi in x])
        model = model.update(
            x, y, warmstart=warm_start_update, max_iterations=max_fit_iterations
        )
        ymin[i + 1] = model.y.min()
    return jnp.array(ymin)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ######################################################################
    # Simulation parameters
    parser.add_argument(
        "--test_function",
        choices=["goldstein", "rover", "push", "lunar"],
        required=True
    )
    parser.add_argument("--initial_acquisitions", type=int, default=4)
    parser.add_argument("--total_acquisitions", type=int, default=50)
    ######################################################################
    # Gaussian process parameters
    parser.add_argument("--max_fit_iterations", type=int, default=100)
    parser.add_argument("--warm_start_update", type=bool, default=True)
    parser.add_argument(
        "--kernel",
        choices=["squaredexponential", "matern52", "matern32", "matern12"],
        required=True,
    )
    ######################################################################
    # Acquisition strategy parameters
    subparser = parser.add_subparsers(dest="acquisition_strategy", required=True)
    # ---> Latin Hypercube Sampling
    parser_lhs = subparser.add_parser("lhs")
    parser_lhs.add_argument("--points", type=int, required=True)
    # ---> Optimize with L-BFGS-B
    parser_bfgs = subparser.add_parser("bfgs")
    parser_bfgs.add_argument("--multi_starts", type=int, required=True)
    parser_bfgs.add_argument("--max_iterations", type=int, default=100)
    args = parser.parse_args()

    ######################################################################
    # Run the experiments and save results
    save_dir = f"results/{args.test_function}/{args.acquisition_strategy}/{args.kernel}"
    os.makedirs(save_dir, exist_ok=True)
    for seed in range(N_RUNS):
        start_time = time.time()
        ymin = run(seed=seed, **vars(args))
        end_time = time.time()
        print(f"Run {seed} completed in {end_time - start_time:.2f} seconds.")

        filename = f"{save_dir}/seed={seed}.npz"
        np.savez_compressed(filename, ymin=ymin, time=end_time - start_time)
