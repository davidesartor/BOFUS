from typing import Literal, NamedTuple, Optional, Protocol, Self
from jaxtyping import Array, Float, Scalar, PyTree
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.numpy.linalg import norm
import equinox as eqx
from .utils import Module, EPS


class Profile(Module):
    def __call__(self, d: Float[Array, "..."]) -> Float[Array, "..."]: ...


class Metric[T: PyTree](Module):
    def initialize(self, xs: list[T]) -> Self: ...

    def __call__(self, x1: T, x2: T) -> Scalar: ...


################################################################################
# region Kernel Profiles


class SquaredExponential(Profile):
    def __call__(self, d: Float[Array, "..."]) -> Float[Array, "..."]:
        k = jnp.exp(-0.5 * d**2)
        return k


class Matern(Profile):
    nu: float

    def __call__(self, d: Float[Array, "..."]) -> Float[Array, "..."]:
        # TODO: add support for general nu using
        if self.nu == 1/2:
            k = jnp.exp(-d)
        elif self.nu == 3/2:
            k = (1 + jnp.sqrt(3) * d) * jnp.exp(-jnp.sqrt(3) * d)
        elif self.nu == 5/2:
            k = (1 + jnp.sqrt(5) * d + 5 / 3 * d**2) * jnp.exp(-jnp.sqrt(5) * d)
        else:
            raise ValueError(f"Unsupported nu={self.nu}")
        return k


# endregion
################################################################################


################################################################################
# region Metrics on R^d


def hetgpy_scale_init(
    x: Float[Array, "n d"], min_cor: float = 0.01, max_cor: float = 0.5
) -> Float[Array, "d"]:
    # rescale X to [0,1]^d
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    x = (x - x_min) @ jnp.diag(1 / (x_max - x_min))

    # compute pairwise squared distances only for proper pairs
    d_fn = lambda a, b: norm(a - b)
    d_fn = jax.vmap(jax.vmap(d_fn, (None, 0)), (0, None))
    dists = d_fn(x, x) ** 2  # squared distances
    dists = dists[*jnp.tril_indices(len(dists), k=-1)]
    
    # magic hetgpy initialization using inverse of kernel
    lower = -jnp.quantile(dists, q=0.05) / jnp.log(min_cor) * (x_max - x_min) ** 2
    upper = -jnp.quantile(dists, q=0.95) / jnp.log(max_cor) * (x_max - x_min) ** 2
    scale = 0.9 * upper + 0.1 * lower
    return scale


class Minkowski(Metric[Float[Array, "d"]]):
    p: int | Literal["inf", "-inf"]
    log_scale: Float[Array, "d"] | None = None

    def initialize(self, xs: list[Float[Array, "d"]]) -> Self:
        scale = hetgpy_scale_init(jnp.stack(xs))
        return self.replace(log_scale=jnp.log(scale))

    def __call__(self, x1: Float[Array, "d"], x2: Float[Array, "d"]) -> Scalar:
        if self.log_scale is not None:
            scale = jnp.exp(self.log_scale)
            v = (x1 - x2) / jnp.sqrt(scale)
        else:
            v = x1 - x2  # default to unscaled Minkowski metric

        # add EPS to avoid NaNs in gradients when d==0
        return norm(v, ord=self.p) + EPS


class Euclidean(Minkowski):
    def __init__(self, *args, **kwargs):
        super().__init__(p=2, *args, **kwargs)


class Manhattan(Minkowski):
    def __init__(self, *args, **kwargs):
        super().__init__(p=1, *args, **kwargs)


class Chebyshev(Minkowski):
    def __init__(self, *args, **kwargs):
        super().__init__(p="inf", *args, **kwargs)
        

class Mahalanobis(Metric[Float[Array, "d"]]):
    log_cov: Float[Array, "d d"] | None = None

    def initialize(self, xs: list[Float[Array, "d"]]) -> Self:
        scale = hetgpy_scale_init(jnp.stack(xs))
        log_cov = jnp.log(jnp.diag(scale**2))
        return self.replace(log_cov=log_cov)

    def __call__(self, x1: Float[Array, "d"], x2: Float[Array, "d"]) -> Scalar:
        if self.log_cov is not None:
            # TODO: better parametrization? no matrix exp, use sqrt directly instead?
            # d = || K^-0.5 @ (x1-x2) ||_2
            K = jsp.linalg.expm(self.log_cov)
            K_sqrt, is_lower = jsp.linalg.cho_factor(K)
            v = jsp.linalg.solve_triangular(K_sqrt, x1 - x2, lower=is_lower)
        else:
            v = x1 - x2  # default to Euclidean metric
        
        # add EPS to avoid NaNs in gradients when d==0
        return norm(v, ord=2) + EPS


# endregion
################################################################################


#################################################################################
# region Metrics on RKHS


class RKHSFunction(NamedTuple):
    # f = sum ai k(xi, .)
    x: Float[Array, "n d"]
    a: Float[Array, "n"]


class MaximumMeanDiscrepancy(Metric[RKHSFunction]):
    rkhs_metric: Metric[Float[Array, "d"]] = eqx.field(static=True)
    rkhs_profile: Profile = eqx.field(static=True)
    log_scale: Scalar | None = None

    def initialize(self, xs: list[RKHSFunction]) -> Self:
        return self.replace(log_scale=jnp.zeros(()))

    def __call__(self, x1: RKHSFunction, x2: RKHSFunction) -> Scalar:
        kernel = lambda a, b: self.rkhs_profile(self.rkhs_metric(a, b))
        kernel = jax.vmap(jax.vmap(kernel, in_axes=(None, 0)), in_axes=(0, None))
        K11 = kernel(x1.x, x1.x)
        K12 = kernel(x1.x, x2.x)
        K22 = kernel(x2.x, x2.x)

        if NAIVE_PARAMETRIZATION := True:
            a1 = x1.a
            a2 = x2.a
        else:
            a1 = jnp.linalg.solve(K11, x1.y)
            a2 = jnp.linalg.solve(K22, x2.y)

        d2 = +a1 @ K11 @ a1 + a2 @ K22 @ a2 - 2 * a1 @ K12 @ a2

        d = jnp.sqrt(d2 + EPS)
        if self.log_scale is not None:
            scale = jnp.exp(self.log_scale)
            d = d / scale
        return d


# endregion
