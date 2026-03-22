from typing import Literal, NamedTuple, Optional, Protocol, Self
from jaxtyping import Array, Float, Scalar, PyTree
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.numpy.linalg import norm
import equinox as eqx
from einops import rearrange

class Module(eqx.Module):
    def replace(self, **kwargs)-> Self:
        where = lambda m: tuple(getattr(m, k) for k in kwargs.keys())
        return eqx.tree_at(where, self, kwargs.values(), is_leaf=lambda x: x is None)
    

################################################################################
# region Kernel Profiles


class Profile(Module):
    def __call__(self, d: Float[Array, "..."]) -> Float[Array, "..."]: ...


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


class Metric[T: PyTree](Module):
    def initialize(self, xs: T) -> Self: ...

    def __call__(self, x1: T, x2: T) -> Scalar: ...


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
    dists = dists[*jnp.tril_indices(x.shape[-1], k=-1)]

    # magic hetgpy initialization using inverse of kernel
    lower = -jnp.quantile(dists, q=0.05) / jnp.log(min_cor) * (x_max - x_min) ** 2
    upper = -jnp.quantile(dists, q=0.95) / jnp.log(max_cor) * (x_max - x_min) ** 2
    scale = 0.9 * upper + 0.1 * lower
    return scale


class Minkowski(Metric[Float[Array, "d"]]):
    p: int | Literal["inf", "-inf"]
    log_scale: Float[Array, "d"] | None = None

    def initialize(self, xs: Float[Array, "n d"]) -> Self:
        scale = hetgpy_scale_init(xs)
        return self.replace(log_scale=jnp.log(scale))

    def __call__(self, x1: Float[Array, "d"], x2: Float[Array, "d"]) -> Scalar:
        if self.log_scale is not None:
            scale = jnp.exp(self.log_scale)
            v = (x1 - x2) / jnp.sqrt(scale)
        else:
            v = x1 - x2  # default to unscaled Minkowski metric

        # use lax.cond to avoid NaNs in grad when d==0
        return jax.lax.cond(
            (v==0).all(),
            lambda: 0.0,
            lambda: norm(v, self.p),
        )
    

class Euclidean(Minkowski):
    def __init__(self):
        super().__init__(p=2)


class Manhattan(Minkowski):
    def __init__(self):
        super().__init__(p=1)


class Chebyshev(Minkowski):
    def __init__(self):
        super().__init__(p="inf")
        

class Mahalanobis(Metric[Float[Array, "d"]]):
    log_cov: Float[Array, "d d"] | None = None

    def initialize(self, xs: Float[Array, "n d"]) -> Self:
        scale = hetgpy_scale_init(xs)
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
        
        # use lax.cond to avoid NaNs in grad when d==0
        return jax.lax.cond(
            (v==0).all(), 
            lambda: 0.0, 
            lambda: norm(v, ord=2),
        )


# endregion
################################################################################


#################################################################################
# region Metrics on RKHS


class RKHSFunction(NamedTuple):
    # f = sum ai k(xi, .)
    x: Float[Array, "n d"]
    a: Float[Array, "n"]


class MaximumMeanDiscrepancy(Metric[RKHSFunction]):
    rkhs_kernel_metric: Metric[Float[Array, "d"]] = eqx.field(static=True)
    rkhs_kernel_profile: Profile = eqx.field(static=True)
    log_scale: Scalar | None = None

    def initialize(self, xs: RKHSFunction) -> Self:
        # TODO: handle properly the initialization
        x = rearrange(xs.x, "k n d -> (k n) d")
        rkhs_kernel_metric = self.rkhs_kernel_metric.initialize(x)
        ...
        return self.replace(rkhs_kernel_metric=rkhs_kernel_metric)

    def __call__(self, x1: RKHSFunction, x2: RKHSFunction) -> Scalar:
        kernel = lambda a, b: self.rkhs_kernel_profile(self.rkhs_kernel_metric(a, b))
        K11 = kernel(x1.x, x1.x)
        K12 = kernel(x1.x, x2.x)
        K22 = kernel(x2.x, x2.x)

        d2 = x1.a @ K11 @ x1.a - 2 * x1.a @ K12 @ x2.a + x2.a @ K22 @ x2.a
        d = jnp.sqrt(d2)
        if self.log_scale is not None:
            scale = jnp.exp(self.log_scale)
            d = d / jnp.sqrt(scale)
        return d


# endregion
