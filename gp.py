from typing import NamedTuple, Optional
from jaxtyping import Array, Float, Key
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from jax.numpy.linalg import norm
from einops import rearrange
import scipy
import numpy
import autobounds

EPS = float(jnp.sqrt(jnp.finfo(float).eps))
MAX_NUGGET = 1e-1


class Gaussian(NamedTuple):
    mean: Float[Array, "n"]
    cov: Float[Array, "n n"]


class Parameters(NamedTuple):
    theta: Float[Array, "o d"]
    g: Float[Array, "o"]
    b: Float[Array, "o"]
    nu: Float[Array, "o"]


def kernel(
    x1: Float[Array, "n d"],
    x2: Float[Array, "m d"],
    theta: Float[Array, "d"],
) -> Float[Array, "n m"]:
    def squared_distance(
        x1: Float[Array, "n d"],
        x2: Float[Array, "m d"],
    ) -> Float[Array, "n m"]:
        def dist(a, b):
            return jnp.sum((a - b) ** 2)

        dist = jax.vmap(jax.vmap(dist, (None, 0)), (0, None))
        return dist(x1, x2)

    # matern 5/2 kernel
    theta = jnp.sqrt(theta)
    d2 = squared_distance(x1 / theta, x2 / theta)
    d = jnp.sqrt(d2 + EPS)
    return (1 + jnp.sqrt(5) * d + 5 / 3 * d2) * jnp.exp(-jnp.sqrt(5) * d)

    # squared exponential kernel
    theta = jnp.sqrt(theta)
    d2 = squared_distance(x1 / theta, x2 / theta)
    return jnp.exp(-0.5 * d2)


@jax.jit
def likelihood(
    theta: Float[Array, "d"],
    g: Float[Array, ""],
    x: Float[Array, "n d"],
    y: Float[Array, "n"],
):
    n, d = x.shape

    # foward of kernel
    K = kernel(x, x, theta)
    K = K + jnp.eye(n) * (EPS + g)

    # cholesky of K and compute logdet
    K_sqrt, is_lower = jsp.linalg.cho_factor(K)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))

    # compute Ki_1=(K^-1 @ 1) and Ki_y=(K^-1 @ y)
    Ki_1, Ki_y = jsp.linalg.cho_solve(
        c_and_lower=(K_sqrt, is_lower),
        b=jnp.stack([jnp.ones_like(y), y], 1),
    ).T

    # compute optimal trend b
    b = (Ki_1 * y).sum() / Ki_1.sum()
    nu = jnp.dot((y - b) / n, (Ki_y - Ki_1 * b))

    # # likelihood when marginalizing over trend and variance
    loglik = -0.5 * (n * jnp.log(nu) + logdetK)
    return loglik, b, nu


@jax.jit
@jax.value_and_grad
def mle_loss(
    theta_g: Float[Array, "d+1"],
    x_train: Float[Array, "n d"],
    y_train: Float[Array, "n"],
):
    theta, g = theta_g[:-1], theta_g[-1]
    loglik, b, nu = likelihood(theta, g, x_train, y_train)
    return -loglik


class GaussianProcess(NamedTuple):
    x: Float[Array, "n d"]
    y: Float[Array, "n"]
    parameters: Parameters
    bounds: Float[Array, "2 o d"]

    @classmethod
    def fit(
        cls,
        x: Float[Array, "n d"],
        y: Float[Array, "n"],
        warmstart: Optional[Parameters] = None,
        max_iterations: int = 100,
        verbose: bool = False,
    ) -> "GaussianProcess":
        # initialization
        lower, upper = autobounds.hetgpy_auto_bounds(x)
        init_theta = 0.9 * upper + 0.1 * lower
        init_g = jnp.minimum(0.1, MAX_NUGGET)
        if warmstart is not None:
            init_theta = warmstart.theta
            init_g = warmstart.g
        if verbose:
            print(f"Initial theta: {init_theta}")
            print(f"Initial g: {init_g}")
            print()

        # bounds for optimization
        theta_bounds = jnp.array([lower, upper])
        g_bounds = jnp.array([EPS, MAX_NUGGET])
        bounds = jnp.concat([theta_bounds, g_bounds[:, None]], axis=-1)
        if verbose:
            print(f"Bounds for theta:")
            print(f"Min: {bounds[0, ..., :-1]}")
            print(f"Max: {bounds[1, ..., :-1]}")
            print(f"Bounds for g:")
            print(f"Min: {bounds[0, ..., -1]}")
            print(f"Max: {bounds[1, ..., -1]}")
            print()

        # run optimization
        result = scipy.optimize.minimize(
            fun=mle_loss,
            x0=jnp.concatenate([init_theta, init_g[None]]),
            args=(x, y),
            jac=True,
            method="L-BFGS-B",
            bounds=[(a, b) for a, b in zip(*bounds)],
            options=dict(maxiter=max_iterations, ftol=EPS, gtol=0),
        )

        # extract the optimal parameters and infer the rest
        theta, g = result.x[..., :-1], result.x[..., -1]
        llk, b, nu = likelihood(theta, g, x, y)

        if verbose:
            print(f"Optimal theta: {theta}")
            print(f"Optimal g: {g}")
            print(f"Optimal b: {b}")
            print(f"Optimal nu: {nu}")
            print(f"Log-likelihood at optimum: {llk}")
            print()
        return cls(x=x, y=y, parameters=Parameters(theta, g, b, nu), bounds=bounds)

    def update(self, x: Float[Array, "n d"], y: Float[Array, "n"]):
        x = jnp.concatenate([self.x, x], axis=0)
        y = jnp.concatenate([self.y, y], axis=0)
        return self.fit(x, y, warmstart=self.parameters)

    @jax.jit
    def predict(self, x: Float[Array, "n d"]) -> Gaussian:
        n, d = self.x.shape
        theta, g, b, nu = self.parameters

        Koo = nu * (kernel(self.x, self.x, theta) + jnp.eye(n) * (EPS + g))
        Kxo = nu * kernel(x, self.x, theta)
        Kxx = nu * kernel(x, x, theta)

        # posterior mean and covariance
        mean = b + Kxo @ jnp.linalg.solve(Koo, self.y - b)
        cov = Kxx - Kxo @ jnp.linalg.solve(Koo, Kxo.T)

        # Add correction based on the trend estimation correlation
        Kbx = jnp.ones((1, n)) @ jnp.linalg.solve(Koo, Kxo.T)
        cov = cov + (1 - Kbx).T @ (1 - Kbx) / jnp.linalg.inv(Koo).sum()
        return Gaussian(mean=mean, cov=cov)

    @jax.jit
    def expected_improvement(self, x: Float[Array, "n d"]) -> Float[Array, "n"]:
        # numerically stable version following https://arxiv.org/pdf/2310.20708:
        mu, cov = self.predict(x)
        sigma = jnp.diag(cov) ** 0.5
        y_star = self.y.min()
        z = (y_star - mu) / sigma

        def eval_single(z):
            branch1 = lambda: jnp.log(z * jsp.stats.norm.cdf(z) + jsp.stats.norm.pdf(z))
            branch2a = lambda: -2 * jnp.log(-z)
            branch2b = lambda: jax.nn.log1mexp(
                -jnp.log(-z)
                - jsp.stats.norm.logsf(-z)
                - z**2 / 2
                - jnp.log(2 * jnp.pi) / 2.0
            )
            branch2 = lambda: (
                -(z**2) / 2
                - jnp.log(2 * jnp.pi) / 2
                + jax.lax.cond(z < -1 / EPS, branch2a, branch2b)
            )
            return jax.lax.cond(z > -1, branch1, branch2)

        return jnp.log(sigma) + jax.vmap(eval_single)(z)

    def acquisition(
        self, multistart: int = 16, max_iterations=10, verbose: bool = False
    ) -> tuple[Float[Array, "n d"], Float[Array, "n"]]:
        @jax.jit
        @jax.value_and_grad
        def loss(x_flat):
            x = x_flat.reshape(-1, d)
            return -self.expected_improvement(x).sum()

        def verbose_loss(x_flat):
            v, g = loss(x_flat)
            if jnp.isnan(v):
                print(f"NaN loss at {x_flat}")
            elif jnp.any(jnp.isnan(g)):
                print(f"NaN in gradient at {x_flat}")
            return v, g

        n, d = self.x.shape
        x0 = scipy.stats.qmc.LatinHypercube(d).random(n=multistart).flatten()
        result = scipy.optimize.minimize(
            fun=verbose_loss,
            x0=x0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0, 1) for _ in range(len(x0))],
            options=dict(maxiter=max_iterations, ftol=EPS, gtol=0),
        )
        x = jnp.array(result.x.reshape(-1, d))
        ei = self.expected_improvement(x)
        best = x[jnp.argmax(ei)]
        if verbose:
            for i in range(len(x)):
                print(f"Acquisition point {i}: {x[i]}, EI: {ei[i]}")
            print(f"best: {best}, best EI: {jnp.max(ei)}")
        return best, x
