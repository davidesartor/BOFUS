from typing import Callable
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import equinox as eqx
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from src import kernels, gp, acquisition


jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


def sample_from_gp_prior(
    kernel: gp.RKHS, xs_grid: Float[Array, "n d"], basis_size: int
) -> Float[Array, "n basis_size"]:
    mean = jnp.zeros(len(xs_grid))
    cov = kernel(xs_grid, xs_grid)
    ys = np.random.multivariate_normal(mean, cov, size=basis_size)
    return jnp.array(ys).T


def run(
    seed: int,
    target_fn: Callable,
    rkhs: gp.RKHS,
    surrogate_model: gp.FunctionalGaussianProcess,
    # simulation parameters
    initial_acquisitions: int,
    min_subspace_dim: int,
    max_subspace_dim: int,
    acquisitions_each: int,
    acquisition_raw_samples: int,
    acquisition_max_restarts: int,
):
    # set random seed
    rng = np.random.default_rng(seed=seed)

    # initial acquisition (work with flat, normalized parameters in [0, 1])
    d = 1  # assuming 1-dimensional input
    grid_sampler = sp.stats.qmc.LatinHypercube(d=d, rng=rng)

    # sample a some random funciton from the gp prior
    xs_grid = jnp.array(grid_sampler.random(n=min_subspace_dim))
    grid_cov = rkhs(xs_grid, xs_grid) + jnp.eye(len(xs_grid)) * EPS
    ys_grid = sample_from_gp_prior(rkhs, xs_grid, basis_size=initial_acquisitions)
    fs = [
        gp.RKHSFunction(kernel=rkhs, a=jnp.linalg.solve(grid_cov, y), x=xs_grid)
        for y in ys_grid.T
    ]

    # evaluate target function at initial points and fit surrogate model
    ys = jnp.array([target_fn(f) for f in fs])
    surrogate_model = surrogate_model.fit(fs, ys)

    # sequential acquisition loop
    for n in range(min_subspace_dim, max_subspace_dim + 1):
        sampler = sp.stats.qmc.LatinHypercube(d=n, rng=rng)
        for i in range(acquisitions_each):
            # sample a n-dim random subspace centered at the current best point
            xs_grid = jnp.array(grid_sampler.random(n=n))
            ys_grid = sample_from_gp_prior(rkhs, xs_grid, basis_size=n)
            b = jnp.array([fs[jnp.argmin(ys)](x) for x in xs_grid])
            grid_cov = rkhs(xs_grid, xs_grid) + jnp.eye(len(xs_grid)) * EPS

            # optimize acquisition function
            def acquisition_loss(c: Float[Array, "n"]) -> Scalar:
                a = jnp.linalg.solve(grid_cov, ys_grid @ c + b)
                f = gp.RKHSFunction(kernel=rkhs, a=a, x=xs_grid)
                mu, cov = surrogate_model.predict([f])
                return -acquisition.log_expected_improvement(
                    mu=mu.squeeze(),
                    sigma=cov.squeeze() ** 0.5,
                    y_best=surrogate_model.observed_ys.min(),
                )

            c = acquisition.optimize_lhs_candidates(
                acquisition_loss=acquisition_loss,
                candidates=sampler.random(n=acquisition_raw_samples),
                max_restarts=acquisition_max_restarts,
            )

            # evaluate target function at the new point
            a = jnp.linalg.solve(grid_cov, ys_grid @ c + b)
            f = gp.RKHSFunction(kernel=rkhs, a=a, x=xs_grid)
            y = target_fn(f)

            # fit surrogate model on the new data
            fs = fs + [f]
            ys = jnp.concatenate([ys, y[None]])
            surrogate_model = surrogate_model.fit(fs, ys)

            print(f"Iteration {i+1}: current= {y:.8f}, best = {ys.min():.8f}")
    return ys


if __name__ == "__main__":
    seed = 0
    initial_acquisitions = 10
    min_subspace_dim = 5
    max_subspace_dim = 10
    acquisitions_each = 10
    acquisition_raw_samples = 1000
    acquisition_max_restarts = 16

    rkhs = gp.RKHS(
        metric=kernels.Euclidean(),
        profile=kernels.SquaredExponential(),
        rho=1 / (4 * jnp.pi),  # type: ignore
    )

    def target_fn(f, n: int = 1000):
        x = np.linspace(0, 1, n).reshape(-1, 1)
        y = np.sinc(x * 2 * jnp.pi).squeeze()
        pred = np.array([f(xi) for xi in x])
        return np.mean(np.square(pred - y))

    ys = run(
        seed=seed,
        target_fn=target_fn,
        rkhs=rkhs,
        surrogate_model=gp.FunctionalGaussianProcess(),
        initial_acquisitions=initial_acquisitions,
        min_subspace_dim=min_subspace_dim,
        max_subspace_dim=max_subspace_dim,
        acquisitions_each=acquisitions_each,
        acquisition_raw_samples=acquisition_raw_samples,
        acquisition_max_restarts=acquisition_max_restarts,
    )

    plt.figure(figsize=(5, 3))
    n0, dn = initial_acquisitions, acquisitions_each
    plt.plot(range(0, n0), ys[:n0], "o", label="initial samples")
    for i, degree in enumerate(range(min_subspace_dim, max_subspace_dim + 1)):
        y_deg = ys[n0 + i * dn : n0 + (i + 1) * dn]
        plt.plot(
            range(n0 + i * dn, n0 + (i + 1) * dn),
            y_deg,
            "o",
            label=f"acquired samples (degree={degree})",
        )

    plt.yscale("log")
    plt.xlabel("Total Evaluations")
    plt.ylabel("Target fn")
    plt.title("Convergence of Bayesian Optimization")
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig("shilton.png", dpi=300, bbox_inches="tight")
    plt.close()
