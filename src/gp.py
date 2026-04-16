from typing import NamedTuple, Self
from jaxtyping import Array, Float, Scalar
import warnings

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import scipy as sp

from . import kernels, rkhs


jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


class Module(eqx.Module):
    def _replace(self, **kwargs) -> Self:
        where = lambda m: tuple(getattr(m, k) for k in kwargs.keys())
        return eqx.tree_at(where, self, kwargs.values(), is_leaf=lambda x: x is None)


class Gaussian(NamedTuple):
    mean: Float[Array, "n"]
    cov: Float[Array, "n n"]


@jax.jit
def gp_posterior(
    Kxx: Float[Array, "m m"],
    Kox: Float[Array, "n m"],
    Koo: Float[Array, "n n"],
    observed_ys: Float[Array, "n"],
    b: Scalar,
) -> Gaussian:
    # posterior mean and covariance
    gain = jnp.linalg.solve(Koo, Kox).T
    mean = b + gain @ (observed_ys - b)
    cov = Kxx - gain @ Kox

    # Add correction based on the trend estimation correlation
    Kbx = jnp.ones((1, len(observed_ys))) @ gain.T
    cov = cov + (1 - Kbx).T @ (1 - Kbx) / jnp.linalg.inv(Koo).sum()
    return Gaussian(mean=mean, cov=cov)


@jax.jit
def loglikelihood(
    Koo: Float[Array, "n n"],
    ys: Float[Array, "n"],
) -> tuple[Scalar, Scalar, Scalar]:
    # cholesky of K and compute logdet
    K_sqrt, is_lower = jsp.linalg.cho_factor(Koo)
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
    return (loglik, b, nu)


class GaussianProcess(Module):
    # kernel definition
    metric: kernels.Metric = kernels.Euclidean()
    profile: kernels.Profile = kernels.SquaredExponential()

    # model parameters
    rho: Float[Array, "d"] = eqx.field(default=None)
    g: Scalar = eqx.field(default=None)
    nu: Scalar = eqx.field(default=None)
    b: Scalar = eqx.field(default=None)

    # observed data
    observed_xs: Float[Array, "n d"] = eqx.field(default=None)
    observed_ys: Float[Array, "n"] = eqx.field(default=None)

    # cached covariance matrix of the observed ys
    Koo: Float[Array, "n n"] = eqx.field(default=None)

    @eqx.filter_jit
    def kernel(
        self,
        rho: Float[Array, "d"],
        xs1: Float[Array, "m d"],
        xs2: Float[Array, "n d"],
    ) -> Float[Array, "m n"]:
        return self.profile(self.metric(rho, xs1, xs2))

    @eqx.filter_jit
    def predict(self, xs: Float[Array, "m d"]) -> Gaussian:
        # compute covariance matrices
        Kxx = self.nu * self.kernel(self.rho, xs, xs)
        Kox = self.nu * self.kernel(self.rho, self.observed_xs, xs)
        Koo = self.nu * self.Koo
        return gp_posterior(Kxx, Kox, Koo, self.observed_ys, self.b)

    def fit(
        self,
        xs: Float[Array, "n d"],
        ys: Float[Array, "n"],
        *,
        warmstart: bool = False,
        lengthscale_range: tuple[float, float] = (EPS, 10.0),
        nugget_range: tuple[float, float] = (EPS, 1e-3),
        max_iterations: int = 100,
        ftol: float = EPS,
        gtol: float = 0.0,
    ) -> Self:
        @jax.jit
        @jax.value_and_grad
        def mle_loss(params: Float[Array, "d+1"]):
            rho, g = params[:-1], params[-1]
            Koo = self.kernel(rho, xs, xs) + g * jnp.eye(len(ys))
            loglik, b, nu = loglikelihood(Koo, ys)
            return -loglik

        def verbose_loss(params: Float[Array, "d+1"]):
            val, grad = mle_loss(params)
            if jnp.isnan(val) or jnp.isnan(grad).any():
                warnings.warn(f"NaN detected in loss or gradient: {params}")
            return val, grad

        # initialization
        n, d = xs.shape
        nugget = min(0.1, nugget_range[1])
        lengthscale = 0.9 * lengthscale_range[1] + 0.1 * lengthscale_range[0]
        if warmstart:
            nugget = self.g if self.g is not None else nugget
            lengthscale = self.rho if self.rho is not None else lengthscale
        init_params = jnp.array([lengthscale for _ in range(d)] + [nugget])

        # run optimization
        result = sp.optimize.minimize(
            fun=verbose_loss,
            x0=init_params,
            jac=True,
            method="L-BFGS-B",
            bounds=[lengthscale_range] * d + [nugget_range],
            options=dict(maxiter=max_iterations, ftol=ftol, gtol=gtol),
        )

        # extract the optimal parameters and infer the rest
        rho = jnp.array(result.x[:-1])
        g = jnp.array(result.x[-1])
        Koo = self.kernel(rho, xs, xs) + g * jnp.eye(len(ys))
        llk, b, nu = loglikelihood(Koo, ys)

        # return a new instance with the fitted parameters and observed data
        return self._replace(
            rho=rho, g=g, nu=nu, b=b, Koo=Koo, observed_xs=xs, observed_ys=ys
        )



class FunctionalGaussianProcess(Module):
    # kernel definition
    profile: kernels.Profile = kernels.SquaredExponential()

    # model parameters
    rho: Scalar = eqx.field(default=None)
    g: Scalar = eqx.field(default=None)
    nu: Scalar = eqx.field(default=None)
    b: Scalar = eqx.field(default=None)

    # observed data
    observed_fs: list[rkhs.Function] = eqx.field(default=None)
    observed_ys: Float[Array, "n"] = eqx.field(default=None)

    # cached covariance matrix of the observed ys
    Koo: Float[Array, "n n"] = eqx.field(default=None)

    @eqx.filter_jit
    def metric(self, f1: rkhs.Function, f2: rkhs.Function) -> Scalar:
        K11 = f1.kernel(f1.x, f1.x)
        K12 = f1.kernel(f1.x, f2.x)
        K22 = f2.kernel(f2.x, f2.x)
        d2 = +f1.a @ K11 @ f1.a + f2.a @ K22 @ f2.a - 2 * f1.a @ K12 @ f2.a
        return jax.lax.cond(d2 <= 0.0, lambda: 0.0, lambda: jnp.sqrt(d2))

    @eqx.filter_jit
    def kernel(
        self,
        rho: Scalar,
        fs1: list[rkhs.Function],
        fs2: list[rkhs.Function],
    ) -> Float[Array, "m n"]:
        d = jnp.array([[self.metric(f1, f2) for f2 in fs2] for f1 in fs1])
        return self.profile(d / jnp.sqrt(rho))

    @eqx.filter_jit
    def predict(self, fs: list[rkhs.Function]) -> Gaussian:
        # compute covariance matrices
        Kxx = self.nu * self.kernel(self.rho, fs, fs)
        Kox = self.nu * self.kernel(self.rho, self.observed_fs, fs)
        Koo = self.nu * self.Koo
        return gp_posterior(Kxx, Kox, Koo, self.observed_ys, self.b)

    def fit(
        self,
        fs: list[rkhs.Function],
        ys: Float[Array, "n"],
        *,
        warmstart: bool = False,
        lengthscale_range: tuple[float, float] = (EPS, 10.0),
        nugget_range: tuple[float, float] = (EPS, 1e-3),
        max_iterations: int = 100,
        ftol: float = EPS,
        gtol: float = 0.0,
    ) -> Self:
        # precalc the metric to speedup mle calls
        dists = jnp.array([[self.metric(f1, f2) for f2 in fs] for f1 in fs])

        @jax.jit
        @jax.value_and_grad
        def mle_loss(params: Float[Array, "2"]):
            rho, g = params[0], params[-1]
            Koo = self.profile(dists / jnp.sqrt(rho)) + g * jnp.eye(len(ys))
            loglik, b, nu = loglikelihood(Koo, ys)
            return -loglik

        def verbose_loss(params: Float[Array, "2"]):
            val, grad = mle_loss(params)
            if jnp.isnan(val) or jnp.isnan(grad).any():
                warnings.warn(f"NaN detected in loss or gradient: {params}")
            return val, grad

        # initialization
        nugget = min(0.1, nugget_range[1])
        lengthscale = 0.9 * lengthscale_range[1] + 0.1 * lengthscale_range[0]
        if warmstart:
            nugget = self.g if self.g is not None else nugget
            lengthscale = self.rho if self.rho is not None else lengthscale
        init_params = jnp.array([lengthscale, nugget])

        # run optimization
        result = sp.optimize.minimize(
            fun=verbose_loss,
            x0=init_params,
            jac=True,
            method="L-BFGS-B",
            bounds=[lengthscale_range, nugget_range],
            options=dict(maxiter=max_iterations, ftol=ftol, gtol=gtol),
        )

        # extract the optimal parameters and infer the rest
        rho = jnp.array(result.x[0])
        g = jnp.array(result.x[-1])
        Koo = self.profile(dists / jnp.sqrt(rho)) + g * jnp.eye(len(ys))
        llk, b, nu = loglikelihood(Koo, ys)

        # return a new instance with the fitted parameters and observed data
        return self._replace(
            rho=rho, g=g, nu=nu, b=b, Koo=Koo, observed_fs=fs, observed_ys=ys
        )

