from typing import Callable
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from src import kernels, gp, acquisition


jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


def f_from_array(
    rkhs: gp.RKHS,
    p: Float[Array, "k d+1"],
    x_range: tuple[float, float] = (0.0, 1.0),
    a_range: tuple[float, float] = (-1.0, 1.0),
) -> gp.RKHSFunction:
    x, a = p[:, :-1], p[:, -1]
    x = x * (x_range[1] - x_range[0]) + x_range[0]  # [0,1]->x_range
    a = a * (a_range[1] - a_range[0]) + a_range[0]  # [0,1]->a_range
    return gp.RKHSFunction(kernel=rkhs, a=a, x=x)


def linear_combination(
    kernel: gp.RKHS,
    basis_fs: list[gp.RKHSFunction],
    coefficients: Float[Array, "n"],
):
    x = [fi.x for fi in basis_fs]
    a = [fi.a * ci for fi, ci in zip(basis_fs, coefficients)]
    return gp.RKHSFunction(kernel=kernel, x=jnp.concat(x), a=jnp.concat(a))


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
    input_range: tuple[float, float] = (0.0, 1.0),
    coefficient_range: tuple[float, float] = (-1.0, 1.0),
):
    # set random seed
    rng = np.random.default_rng(seed=seed)
    as_function = lambda p: f_from_array(
        rkhs, p.reshape(-1, d + 1), x_range=input_range, a_range=coefficient_range
    )

    # initial acquisition (work with flat, normalized parameters in [0, 1])
    k = 3  # number of basis points for fs of the random subspace basis
    d = 1  # assuming 1-dimensional input
    basis_sampler = sp.stats.qmc.LatinHypercube(d=k * (d + 1), rng=rng)
    fs = [as_function(p) for p in basis_sampler.random(n=initial_acquisitions)]

    # evaluate target function at initial points and fit surrogate model
    ys = jnp.array([target_fn(f) for f in fs])
    surrogate_model = surrogate_model.fit(fs, ys)

    # sequential acquisition loop
    for n in range(min_subspace_dim, max_subspace_dim + 1):
        p_sampler = sp.stats.qmc.LatinHypercube(d=n, rng=rng)
        for i in range(acquisitions_each):
            # sample a n-dim random subspace
            basis_fs = [as_function(p) for p in basis_sampler.random(n=n)]

            # optimize acquisition function
            def acquisition_loss(p: Float[Array, "n"]) -> Scalar:
                f = linear_combination(rkhs, basis_fs, p)
                mu, cov = surrogate_model.predict([f])
                return -acquisition.log_expected_improvement(
                    mu=mu.squeeze(),
                    sigma=cov.squeeze() ** 0.5,
                    y_best=surrogate_model.observed_ys.min(),
                )

            p = acquisition.optimize_lhs_candidates(
                acquisition_loss=acquisition_loss,
                candidates=p_sampler.random(n=acquisition_raw_samples),
                max_restarts=acquisition_max_restarts,
            )

            # evaluate target function at the new point
            f = linear_combination(rkhs, basis_fs, p)
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
    plt.savefig("kundu.png", dpi=300, bbox_inches="tight")
    plt.close()
