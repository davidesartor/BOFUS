from jaxtyping import Array, Float
import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import scipy as sp
import numpy as np
import os
import time
import random

from src.gp import GaussianProcess
from src.kernels import Metric, Euclidean, Minkowski
from src.kernels import Profile, Matern, SquaredExponential


from src.acquisition import AcquisitionStrategy, LHS, BFGS, Voronoi
from src.test_functions import TestFunction, GoldsteinPrice, Rover, Push, LunarLander

from jsonargparse import ArgumentParser, ActionConfigFile

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


def run(
    seed: int,
    initial_acquisitions: int,
    total_acquisitions: int,
    acquisition_strategy: AcquisitionStrategy,
    test_function: TestFunction,
    model: GaussianProcess,
) -> tuple[Float[Array, "k"], float, float]:
    # set random seeds initialize timer
    np.random.seed(seed)
    random.seed(seed)
    acquisition_time = 0.0
    model_time = 0.0

    # initial acquisitions
    x0 = sp.stats.qmc.LatinHypercube(
        d=test_function.d, seed=np.random.mtrand._rand
    ).random(n=initial_acquisitions)
    y0 = jnp.array([test_function(xi) for xi in x0])
    model = model.fit(x0, y0)

    # experiment loop
    ymin = np.empty(total_acquisitions + 1)
    ymin[:initial_acquisitions] = np.nan
    ymin[initial_acquisitions] = model.observed_ys.min()
    for i in tqdm(range(initial_acquisitions, total_acquisitions)):
        # acquisition step
        start_time = time.time()
        x = acquisition_strategy(
            acqusition_fn=model.expected_improvement,
            observed_xs=model.observed_xs,
        )
        acquisition_time += time.time() - start_time

        # get the test function values
        y = jnp.array([test_function(xi) for xi in x])

        start_time = time.time()
        model = model.fit(
            x=jnp.concatenate([model.observed_xs, x], axis=0),
            y=jnp.concatenate([model.observed_ys, y], axis=0),
        )
        ymin[i + 1] = model.observed_ys.min()
        model_time += time.time() - start_time
    return jnp.array(ymin), acquisition_time, model_time


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial_acquisitions", type=int, default=4)
    parser.add_argument("--total_acquisitions", type=int, default=50)

    parser.add_argument(
        "--test_function", type=GoldsteinPrice | Rover | Push | LunarLander
    )
    parser.add_argument("--model", type=GaussianProcess)
    parser.add_argument("--acquisition_strategy", type=LHS | BFGS | Voronoi)

    args = parser.parse_args()
    cfg = parser.instantiate_classes(args)

    # create save directory
    save_dir = f"results/{type(cfg.test_function).__name__}/{type(cfg.acquisition_strategy).__name__}+{type(cfg.model.kernel_metric).__name__}+{type(cfg.model.kernel_profile).__name__}"
    os.makedirs(save_dir, exist_ok=True)

    # run experiment
    start_time = time.time()
    ymin, acquisition_time, model_time = run(
        seed=cfg.seed,
        initial_acquisitions=cfg.initial_acquisitions,
        total_acquisitions=cfg.total_acquisitions,
        acquisition_strategy=cfg.acquisition_strategy,
        test_function=cfg.test_function,
        model=cfg.model,
    )
    end_time = time.time()
    print(f"Run {cfg.seed} completed in {end_time - start_time:.2f} seconds.")

    # save results
    np.savez_compressed(
        f"{save_dir}/seed={cfg.seed}.npz",
        ymin=ymin,
        run_time=end_time - start_time,
        acquisition_time=acquisition_time,
        model_time=model_time,
    )
