from typing import Callable, NamedTuple, Literal, Optional, Protocol, Self
from jaxtyping import Array, Float, Scalar, Int, Key, PyTree
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import equinox as eqx
import optax
from . import kernels

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


def lbfgs_minimize[T: PyTree](
    f: Callable[[T], Scalar],
    x0: T,
    max_iterations: int = 100,
    ftol: float = EPS,
    gtol: float = 0.0,  # TODO: add gtol stopping condition
) -> tuple[T, Scalar, Int]:
    # initialize L-BFGS optimization
    solver = optax.lbfgs()
    opt = solver.init(x0)
    initial_state = (x0, opt, jnp.inf, 0.0, 0)

    # early stopping condition (false -> stop)
    def while_cond_fn(state):
        x, opt, loss, prev_loss, i = state
        abs_change = jnp.abs(loss - prev_loss)
        return (i < max_iterations) & (abs_change > ftol)

    # optimization step
    def while_body_fn(state):
        x, opt, loss, prev_loss, i = state
        prev_loss = loss
        loss, grad = optax.value_and_grad_from_state(f)(x, state=opt)
        updates, opt = solver.update(grad, opt, x, value=loss, grad=grad, value_fn=f)
        x = optax.apply_updates(x, updates)
        return (x, opt, loss, prev_loss, i + 1)

    # runs the optimization loop (lowered to a single while op)
    x, _, loss, _, iters = jax.lax.while_loop(
        while_cond_fn, while_body_fn, initial_state
    )
    return x, loss, iters # type: ignore


class Gaussian(NamedTuple):
    mean: Float[Array, "n"]
    cov: Float[Array, "n n"]


class GaussianProcess(kernels.Module):
    kernel_metric: kernels.Metric
    kernel_profile: kernels.Profile
    nugget_range: tuple[float, float] = (EPS, 1e-3)
    max_fit_iterations: int = 100

    # gp parameters
    logit_nugget: Scalar = eqx.field(default=None)
    cov_scale: Scalar = eqx.field(default=None)
    trend: Scalar = eqx.field(default=None)

    # observed data
    observed_xs: Float[Array, "n d"] = eqx.field(default=None)
    observed_ys: Float[Array, "n"] = eqx.field(default=None)

    @property
    def nugget(self):
        lb, ub = self.nugget_range
        return lb + (ub - lb) * jax.nn.sigmoid(self.logit_nugget)

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, 0, None), out_axes=0)  # map over x1
    @partial(jax.vmap, in_axes=(None, None, 0), out_axes=0)  # map over x2
    def kernel(
        self, x1: Float[Array, "n d"], x2: Float[Array, "m d"]
    ) -> Float[Array, "n m"]:
        # this is defined for a single input pair
        d = self.kernel_metric(x1, x2)
        k = self.kernel_profile(d)
        return k

    @eqx.filter_jit
    def predict(self, x: Float[Array, "n d"]) -> Gaussian:
        n, d = self.observed_xs.shape

        # compute covariance matrices
        Kxx = self.cov_scale * self.kernel(x, x)
        Kxo = self.cov_scale * self.kernel(x, self.observed_xs)
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
    def fit(self, x: Float[Array, "n d"], y: Float[Array, "n"]) -> Self:
        # initialize kernel and nugget, reset other fields
        model = self.replace(
            kernel_metric=self.kernel_metric.initialize(x),
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
            loglik, b, nu = model.loglikelihood(x, y)
            return -loglik
        
        params, loss, iters = lbfgs_minimize(
            mle_loss, params, max_iterations=self.max_fit_iterations
        )

        # write optimized params back into self
        model: Self = eqx.combine(params, static)  
        llk, b, nu = model.loglikelihood(x, y)
        model = model.replace(trend=b, cov_scale=nu, observed_xs=x, observed_ys=y)
        return model

    def loglikelihood(self, xs: Float[Array, "n d"], ys: Float[Array, "n"]):
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
    def expected_improvement(self, x: Float[Array, "d"]) -> Scalar:
        # numerically stable version following https://arxiv.org/pdf/2310.20708:
        mu, cov = self.predict(x[None, :])
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
