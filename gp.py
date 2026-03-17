from typing import NamedTuple, Literal, Optional, Self
from jaxtyping import Array, Float

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import scipy as sp
import numpy as np
import autobounds

EPS = float(jnp.sqrt(jnp.finfo(float).eps))
MAX_NUGGET = 1e-3

Kernel = Literal["squaredexponential", "matern52", "matern32", "matern12"]


class Gaussian(NamedTuple):
    mean: Float[Array, "n"]
    cov: Float[Array, "n n"]


@eqx.filter_jit
def cov_fn(
    kernel: Kernel,
    theta: Float[Array, "d"],
    x1: Float[Array, "n d"],
    x2: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    def eval_single(
        a: Float[Array, "d"],
        b: Float[Array, "d"],
    ):
        d2 = jnp.sum(jnp.square(a - b) / theta)
        d = jnp.sqrt(jnp.where(d2 > 0, d2, EPS))
        if kernel == "squaredexponential":
            k = jnp.exp(-0.5 * d2)
        elif kernel == "matern12":
            k = jnp.exp(-d)
        elif kernel == "matern32":
            k = (1 + jnp.sqrt(3) * d) * jnp.exp(-jnp.sqrt(3) * d)
        elif kernel == "matern52":
            k = (1 + jnp.sqrt(5) * d + 5 / 3 * d2) * jnp.exp(-jnp.sqrt(5) * d)
        k = jax.lax.cond(d2 == 0.0, lambda: 1.0, lambda: k)
        return k

    return jax.vmap(jax.vmap(eval_single, (None, 0)), (0, None))(x1, x2)


@eqx.filter_jit
def likelihood(
    kernel: Kernel,
    theta: Float[Array, "d"],
    g: Float[Array, ""],
    x: Float[Array, "n d"],
    y: Float[Array, "n"],
):
    n, d = x.shape

    # foward of kernel
    K = cov_fn(kernel, theta, x, x)
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

    # likelihood when marginalizing over trend and variance
    loglik = -0.5 * (n * jnp.log(nu) + logdetK)
    return loglik, b, nu


class GaussianProcess(NamedTuple):
    kernel: Kernel
    x: Float[Array, "n d"]
    y: Float[Array, "n"]
    theta: Float[Array, "d"]
    g: Float[Array, ""]
    b: Float[Array, ""]
    nu: Float[Array, ""]
    Koo: Float[Array, "n n"]

    @classmethod
    def fit(
        cls,
        x: Float[Array, "n d"],
        y: Float[Array, "n"],
        *,
        warmstart: Optional[Self] = None,
        kernel: Kernel = "squaredexponential",
        max_iterations: int = 100,
        verbose: bool = False,
    ):
        n, d = x.shape

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

        # define the loss function for optimization
        @jax.jit
        @jax.value_and_grad
        def mle_loss(theta_g: Float[Array, "d+1"]):
            theta, g = theta_g[:-1], theta_g[-1]
            loglik, b, nu = likelihood(kernel, theta, g, x, y)
            return -loglik

        # run optimization
        result = sp.optimize.minimize(
            fun=mle_loss,
            x0=jnp.concatenate([init_theta, init_g[None]]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(a, b) for a, b in zip(*bounds)],
            options=dict(maxiter=max_iterations, ftol=EPS, gtol=0),
        )

        # extract the optimal parameters and infer the rest
        theta, g = result.x[..., :-1], result.x[..., -1]
        llk, b, nu = likelihood(kernel, theta, g, x, y)
        Koo = nu * (cov_fn(kernel, theta, x, x) + jnp.eye(x.shape[0]) * (EPS + g))
        if verbose:
            print(f"Optimal theta: {theta}")
            print(f"Optimal g: {g}")
            print(f"Optimal b: {b}")
            print(f"Optimal nu: {nu}")
            print(f"Log-likelihood at optimum: {llk}")
            print()

        return GaussianProcess(
            kernel=kernel, x=x, y=y, theta=theta, g=g, b=b, nu=nu, Koo=Koo
        )

    def update(
        self,
        x: Float[Array, "n d"],
        y: Float[Array, "n"],
        *,
        warmstart: bool = True,
        max_iterations: int = 100,
        verbose: bool = False,
    ):
        x = jnp.concatenate([self.x, x], axis=0)
        y = jnp.concatenate([self.y, y], axis=0)
        return self.fit(
            x,
            y,
            kernel=self.kernel,
            warmstart=self if warmstart else None,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    @eqx.filter_jit
    def predict(self, x: Float[Array, "n d"]) -> Gaussian:
        theta, g, b, nu = self.theta, self.g, self.b, self.nu
        n, d = self.x.shape

        Kxo = nu * cov_fn(self.kernel, theta, x, self.x)
        Kxx = nu * cov_fn(self.kernel, theta, x, x)

        # posterior mean and covariance
        mean = self.b + Kxo @ jnp.linalg.solve(self.Koo, self.y - self.b)
        cov = Kxx - Kxo @ jnp.linalg.solve(self.Koo, Kxo.T)

        # Add correction based on the trend estimation correlation
        Kbx = jnp.ones((1, n)) @ jnp.linalg.solve(self.Koo, Kxo.T)
        cov = cov + (1 - Kbx).T @ (1 - Kbx) / jnp.linalg.inv(self.Koo).sum()
        return Gaussian(mean=mean, cov=cov)

    @eqx.filter_jit
    def expected_improvement(self, x: Float[Array, "n d"]) -> Float[Array, "n"]:
        # numerically stable version following https://arxiv.org/pdf/2310.20708:
        # NOTE: it might fail due to numerical precision if the argument of log1mexp is negative
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
                + jax.lax.cond(z < -1 / jnp.sqrt(EPS), branch2a, branch2b)
            )
            return jax.lax.cond(z > -1, branch1, branch2)

        ei = jnp.log(sigma) + jax.vmap(eval_single)(z)
        return ei
