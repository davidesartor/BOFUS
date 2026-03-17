import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float
from tqdm import tqdm
import scipy as sp
import numpy as np
import gp
import argparse
import os

N_RUNS = 32
EPS = float(jnp.sqrt(jnp.finfo(float).eps))
jax.config.update("jax_enable_x64", True)

######################################################################
# target functions
######################################################################


def goldstein_price(x: Float[Array, "n 2"]) -> Float[Array, "n"]:
    x1, x2 = 4 * x[:, 0] - 2, 4 * x[:, 1] - 2
    fact1a = (x1 + x2 + 1) ** 2
    fact1b = 19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
    fact1 = 1 + fact1a * fact1b

    fact2a = (2 * x1 - 3 * x2) ** 2
    fact2b = 18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
    fact2 = 30 + fact2a * fact2b
    y = fact1 * fact2
    y = (jnp.log(y) - 8.6928) / 2.4269
    return y


######################################################################
# acquisition strategies
######################################################################


def lhs_acquisition(
    model: gp.GaussianProcess,
    points: int,
) -> Float[Array, "#n d"]:
    n, d = model.x.shape
    lhs_sampler = sp.stats.qmc.LatinHypercube(d=d, seed=np.random.mtrand._rand)
    x = lhs_sampler.random(n=points)
    ei = model.expected_improvement(x)
    best = jnp.argmax(ei)
    return x[best : best + 1]


def bfgs_acquisition(
    model: gp.GaussianProcess,
    multi_starts: int,
    max_iterations: int,
) -> Float[Array, "#n d"]:
    @jax.jit
    @jax.value_and_grad
    def loss(x):
        return -model.expected_improvement(x[None]).squeeze()

    def safe_loss(x):
        v, g = loss(x)
        nan_in_v = jnp.isnan(v).any()
        nan_in_g = jnp.isnan(g).any()
        if nan_in_v or nan_in_g:
            mu, cov = model.predict(x[None])
            sigma = jnp.diag(cov) ** 0.5
            y_star = model.y.min()
            z = (y_star - mu) / sigma
            print()
            print(f"NaN encountered")
            print(f"NaN in loss: {nan_in_v}, NaN in gradient: {nan_in_g}")
            print(f"at z={z}, mu={mu}, sigma={sigma}")
            print()
        return v, g

    n, d = model.x.shape
    lhs_sampler = sp.stats.qmc.LatinHypercube(d=d, seed=np.random.mtrand._rand)
    results = [
        sp.optimize.minimize(
            fun=safe_loss,
            x0=x0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0, 1) for _ in range(d)],
            options=dict(maxiter=max_iterations, ftol=EPS, gtol=0),
        )
        for x0 in lhs_sampler.random(n=multi_starts)
    ]
    x = jnp.array([result.x for result in results])
    best = jnp.argmin(jnp.array([result.fun for result in results]))
    return x[best : best + 1]


######################################################################
# simulation loop
######################################################################


def run(
    # simualtion parameters
    seed: int,
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
    np.random.seed(seed)
    if acquisition_strategy == "lhs":
        acquisition_fn = lhs_acquisition
    elif acquisition_strategy == "bfgs":
        acquisition_fn = bfgs_acquisition
    else:
        raise ValueError(f"Unknown acquisition strategy: {acquisition_strategy}")

    # initial acquisitions
    lhs_sampler = sp.stats.qmc.LatinHypercube(d=2, seed=np.random.mtrand._rand)
    x0 = lhs_sampler.random(n=initial_acquisitions)
    y0 = goldstein_price(x0)
    model = gp.GaussianProcess.fit(
        x0, y0, kernel=kernel, max_iterations=max_fit_iterations
    )

    # experiment loop
    ymin = np.empty(total_acquisitions + 1)
    ymin[:initial_acquisitions] = np.nan
    ymin[initial_acquisitions] = model.y.min()
    for i in tqdm(range(initial_acquisitions, total_acquisitions)):
        x = acquisition_fn(model, *args, **kwargs)
        y = goldstein_price(x)
        model = model.update(
            x, y, warmstart=warm_start_update, max_iterations=max_fit_iterations
        )
        ymin[i + 1] = model.y.min()
    return jnp.array(ymin)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ######################################################################
    # Simulation parameters
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
    save_dir = f"results/{args.acquisition_strategy}/{args.kernel}"
    os.makedirs(save_dir, exist_ok=True)
    for seed in range(N_RUNS):
        ymin = run(seed=seed, **vars(args))
        filename = f"{save_dir}/seed={seed}.npz"
        np.savez_compressed(filename, ymin=ymin)
