from typing import Callable, NamedTuple, Self
from jaxtyping import Array, Float, Scalar, Int, PyTree

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import equinox as eqx
import optax
from . import kernels
from .utils import Module, EPS, lbfgs_minimize


class Gaussian(NamedTuple):
    mean: Float[Array, "n"]
    cov: Float[Array, "n n"]


class GaussianProcess[T: PyTree](Module):
    kernel_metric: kernels.Metric[T]
    kernel_profile: kernels.Profile
    nugget_range: tuple[float, float] = (EPS, 1e-3)
    max_fit_iterations: int = 100

    # gp parameters
    logit_nugget: Scalar = eqx.field(default=None)
    cov_scale: Scalar = eqx.field(default=None)
    trend: Scalar = eqx.field(default=None)

    # observed data
    observed_xs: list[T] = eqx.field(default=None)
    observed_ys: Float[Array, "n"] = eqx.field(default=None)

    @property
    def nugget(self):
        lb, ub = self.nugget_range
        return lb + (ub - lb) * jax.nn.sigmoid(self.logit_nugget)

    @eqx.filter_jit
    def kernel(self, x1: list[T], x2: list[T]) -> Float[Array, "n m"]:
        # stack list of pytrees into a single pytree of arrays
        d = jnp.array([[self.kernel_metric(xi, xj) for xj in x2] for xi in x1])
        k = self.kernel_profile(d)
        return k

    @eqx.filter_jit
    def predict(self, xs: list[T]) -> Gaussian:
        n = len(self.observed_ys)

        # compute covariance matrices
        Kxx = self.cov_scale * self.kernel(xs, xs)
        Kxo = self.cov_scale * self.kernel(xs, self.observed_xs)
        Koo = self.cov_scale * (
            self.kernel(self.observed_xs, self.observed_xs) + jnp.eye(n) * self.nugget
        )

        # posterior mean and covariance
        mean = self.trend + Kxo @ jnp.linalg.solve(Koo, self.observed_ys - self.trend)
        cov = Kxx - Kxo @ jnp.linalg.solve(Koo, Kxo.T)

        # Add correction based on the trend estimation correlation
        Kbx = jnp.ones((1, n)) @ jnp.linalg.solve(Koo, Kxo.T)
        cov = cov + (1 - Kbx).T @ (1 - Kbx) / jnp.linalg.inv(Koo).sum()
        return Gaussian(mean=mean, cov=cov)

    @eqx.filter_jit
    def fit(self, xs: list[T], ys: Float[Array, "n"]) -> Self:
        # initialize kernel and nugget, reset other fields
        model = self.replace(
            kernel_metric=self.kernel_metric.initialize(xs),
            logit_nugget=jnp.zeros(()),
            observed_xs=None,  # avoids computing gradients
            observed_ys=None,  # avoids computing gradients
            trend=None,  # avoids computing gradients
            cov_scale=None,  # avoids computing gradients
        )
        # initialize L-BFGS optimization
        params, static = eqx.partition(model, eqx.is_array)

        # define MLE loss function
        def mle_loss(params) -> Scalar:
            model = eqx.combine(params, static)
            loglik, b, nu = model.loglikelihood(xs, ys)
            return -loglik

        params, loss, iters = lbfgs_minimize(
            mle_loss, params, max_iterations=self.max_fit_iterations
        )

        # write optimized params back into self
        model: Self = eqx.combine(params, static)  
        llk, b, nu = model.loglikelihood(xs, ys)
        model = model.replace(trend=b, cov_scale=nu, observed_xs=xs, observed_ys=ys)
        return model

    def loglikelihood(
        self, xs: list[T], ys: Float[Array, "n"]
    ) -> tuple[Scalar, Scalar, Scalar]:
        # foward of kernel
        K = self.kernel(xs, xs)
        K = K + jnp.eye(len(ys)) * self.nugget

        # cholesky of K and compute logdet
        K_sqrt, is_lower = jsp.linalg.cho_factor(K)
        logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))

        # compute Ki_1=(K^-1 @ 1) and Ki_y=(K^-1 @ y)
        Ki_1, Ki_y = jsp.linalg.cho_solve(
            c_and_lower=(K_sqrt, is_lower),
            b=jnp.stack([jnp.ones_like(ys), ys], 1),
        ).T

        # compute optimal trend b and scale nu
        b = (Ki_1 * ys).sum() / Ki_1.sum()
        nu = jnp.dot((ys - b) / len(ys), (Ki_y - Ki_1 * b))

        # likelihood when marginalizing over trend and variance
        loglik = -0.5 * (len(ys) * jnp.log(nu) + logdetK)
        return loglik, b, nu

    @eqx.filter_jit
    def log_expected_improvement(self, x: T) -> Scalar:
        # numerically stable version following https://arxiv.org/pdf/2310.20708:
        mu, cov = self.predict([x])
        mu = mu.squeeze()
        sigma = cov.squeeze() ** 0.5
        z = (self.observed_ys.min() - mu) / sigma

        # use lax.cond to avoid propagating NaNs in the gradients
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
        ei = jnp.log(sigma) + jax.lax.cond(z > -1, branch1, branch2)
        return ei
