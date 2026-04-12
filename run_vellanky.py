from typing import Callable
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from src import kernels, gp, acquisition


def berstein_polynomial(
    normalized_coefficients: Float[Array, "n+1"],
    coefficient_range: tuple[float, float],
):
    # rescale to the original range
    min_c, max_c = coefficient_range
    c = normalized_coefficients * (max_c - min_c) + min_c

    # define the berstein polynomial callable
    def f(x: Float[Array, "... 1"]) -> Float[Array, "..."]:
        g = jsp.stats.binom.pmf(jnp.arange(len(c)), len(c) - 1, x)
        return jnp.sum(c * g, axis=-1)

    return jax.jit(f)


@jax.jit
def bump_degree(
    normalized_coefficients: Float[Array, "... n+1"],
    coefficient_range: tuple[float, float],
) -> Float[Array, "... n+2"]:
    # rescale to the original range
    min_c, max_c = coefficient_range
    c = normalized_coefficients * (max_c - min_c) + min_c

    # express degree n berstein polynomial as degree n+1 berstein polynomial
    n = c.shape[-1] - 1
    i = jnp.arange(n + 2) / (n + 1)
    c_new = jnp.zeros((*c.shape[:-1], n + 2))
    c_new = c_new.at[..., 1:].add(i[1:] * c)
    c_new = c_new.at[..., :-1].add((1 - i[:-1]) * c)

    # rescale back to [0, 1]
    c_new = (c_new - min_c) / (max_c - min_c)
    return c_new


def run(
    seed: int,
    target_fn: Callable,
    surrogate_model: gp.GaussianProcess,
    # simulation parameters
    initial_acquisitions: int,
    min_polynomial_degree: int,
    max_polynomial_degree: int,
    acquisitions_each: int,
    acquisition_raw_samples: int,
    acquisition_max_restarts: int,
    coefficient_range: tuple[float, float] = (-1.0, 1.0),
):
    # set random seed
    rng = np.random.default_rng(seed=seed)

    # initial acquisition (work with normalized coefficients in [0, 1])
    sampler = sp.stats.qmc.LatinHypercube(d=min_polynomial_degree + 1, rng=rng)
    cs = jnp.array(sampler.random(n=initial_acquisitions))

    # evaluate target function at initial points and fit surrogate model
    fs = [berstein_polynomial(c, coefficient_range) for c in cs]
    ys = jnp.array([target_fn(f) for f in fs])
    surrogate_model = surrogate_model.fit(cs, ys)

    # sequential acquisition loop
    for degree in range(min_polynomial_degree, max_polynomial_degree + 1):
        # adjust the size of the representations
        while cs.shape[-1] < degree + 1:
            cs = bump_degree(cs, coefficient_range)
        surrogate_model = surrogate_model.fit(cs, ys)

        sampler = sp.stats.qmc.LatinHypercube(d=degree + 1, rng=rng)
        for i in range(acquisitions_each):
            # optimize acquisition function 
            def acquisition_loss(c: Float[Array, "n+1"]) -> Scalar:
                mu, cov = surrogate_model.predict(c[None, :])
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
            f = berstein_polynomial(c, coefficient_range)
            y = target_fn(f)

            # fit surrogate model on the new data
            cs = jnp.concatenate([cs, c[None]])
            ys = jnp.concatenate([ys, y[None]])
            surrogate_model = surrogate_model.fit(cs, ys)

            print(f"Iteration {i+1}: current= {y:.8f}, best = {ys.min():.8f}")
    return ys


if __name__ == "__main__":
    seed = 0
    initial_acquisitions = 10
    min_polynomial_degree = 5
    max_polynomial_degree = 10
    acquisitions_each = 30
    acquisition_raw_samples = 1000
    acquisition_max_restarts = 16

    def target_fn(f, n: int = 100000) -> Scalar:
        x = jnp.linspace(0, 1, n).reshape(-1, 1)
        y = jnp.sinc(x * 2 * jnp.pi).squeeze()
        return jnp.mean((f(x) - y) ** 2)

    ys = run(
        seed=seed,
        target_fn=target_fn,
        surrogate_model=gp.GaussianProcess(),
        initial_acquisitions=initial_acquisitions,
        min_polynomial_degree=min_polynomial_degree,
        max_polynomial_degree=max_polynomial_degree,
        acquisitions_each=acquisitions_each,
        acquisition_raw_samples=acquisition_raw_samples,
        acquisition_max_restarts=acquisition_max_restarts,
    )

    plt.figure(figsize=(5, 3))
    n0, dn = initial_acquisitions, acquisitions_each
    plt.plot(range(0, n0), ys[:n0], "o", label="initial samples")
    for i, degree in enumerate(range(min_polynomial_degree, max_polynomial_degree + 1)):
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
    plt.savefig("vellanky.png", dpi=300, bbox_inches="tight")
    plt.close()
