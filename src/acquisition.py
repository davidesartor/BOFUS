from typing import Callable, Literal, NamedTuple, Protocol, Self
from jaxtyping import Array, Float, Scalar


import jax
import jax.numpy as jnp
import jax.scipy as jsp
import scipy as sp
import numpy as np


jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


def optimize_lhs_candidates(
    acquisition_loss: Callable[[Float[Array, "d"]], Scalar],
    candidates: Float[Array, "n d"],
    max_restarts: int,
    optimizer_options: dict = dict(maxiter=100, ftol=EPS, gtol=0.0),
) -> Float[Array, "d"]:
    # only keep the best initial candidates
    loss_fn = jax.jit(jax.value_and_grad(acquisition_loss))
    losses = np.array([loss_fn(candidate)[0] for candidate in candidates])
    candidates = candidates[np.argsort(losses)[:max_restarts]]

    # optimize each initial guesses with L-BFGS-B
    results = [
        sp.optimize.minimize(
            fun=loss_fn,
            x0=candidate,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * len(candidate),
            options=optimizer_options,
        )
        for candidate in candidates
    ]

    # sort results and return the best one
    losses = jnp.array([result.fun for result in results])
    best_candidate = results[jnp.argmin(losses)].x
    return best_candidate


def log_expected_improvement(mu: Scalar, sigma: Scalar, y_best: Scalar) -> Scalar:
    # numerically stable version following https://arxiv.org/pdf/2310.20708:
    z = (y_best - mu) / sigma

    # use lax.cond to avoid propagating NaNs in the gradients
    branch1 = lambda: jnp.log(z * jsp.stats.norm.cdf(z) + jsp.stats.norm.pdf(z))
    branch2a = lambda: -2 * jnp.log(-z)
    branch2b = lambda: jax.nn.log1mexp(
        -jnp.log(-z) - jsp.stats.norm.logsf(-z) - z**2 / 2 - jnp.log(2 * jnp.pi) / 2.0
    )
    branch2 = lambda: (
        -(z**2) / 2
        - jnp.log(2 * jnp.pi) / 2
        + jax.lax.cond(z < -1 / jnp.sqrt(EPS), branch2a, branch2b)
    )
    ei = jnp.log(sigma) + jax.lax.cond(z > -1, branch1, branch2)
    return ei


def upper_confidence_bound(mu: Scalar, sigma: Scalar, beta: Scalar) -> Scalar:
    return -mu + jnp.sqrt(beta) * sigma


# endregion
################################################################################
