from typing import NamedTuple, Self
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx

from . import kernels


jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


class RKHS(NamedTuple):
    metric: kernels.Metric
    profile: kernels.Profile
    rho: Float[Array, "d"]

    @property
    def d(self) -> int:
        return len(self.rho)

    @eqx.filter_jit
    def __call__(
        self,
        xs1: Float[Array, "n d"],
        xs2: Float[Array, "m d"],
    ) -> Float[Array, "n m"]:
        return self.profile(self.metric(self.rho, xs1, xs2))


class Function(NamedTuple):
    kernel: RKHS
    x: Float[Array, "k d"]  # basis points
    a: Float[Array, "k"]  # coefficients

    @eqx.filter_jit
    def __call__(self, t: Float[Array, "d"]) -> Scalar:
        Ktx = self.kernel(t[None, :], self.x)
        return (Ktx @ self.a).squeeze()

    @classmethod
    def from_array(
        cls,
        rkhs: RKHS,
        p: Float[Array, "k d+1"],
        x_range: tuple[float, float] = (0.0, 1.0),
        y_range: tuple[float, float] = (-1.0, 1.0),
        eps: float = 0.01,
    ) -> Self:
        x, y = p[:, :-1], p[:, -1]
        x = x * (x_range[1] - x_range[0]) + x_range[0]  # [0,1]->x_range
        y = y * (y_range[1] - y_range[0]) + y_range[0]  # [0,1]->y_range
        return cls.from_xy(rkhs, x, y, eps)

    @classmethod
    def from_xy(
        cls, rkhs: RKHS, x: Float[Array, "k d"], y: Float[Array, "k"], eps: float = 0.01
    ) -> Self:
        Kxx = rkhs(x, x) + eps * jnp.eye(len(x))
        a = jnp.linalg.solve(Kxx, y)
        return cls(kernel=rkhs, a=a, x=x)


class BernsteinPolynomial(NamedTuple):
    c: Float[Array, "n+1"]

    @property
    def degree(self) -> int:
        return self.c.shape[-1] - 1

    @classmethod
    def from_array(
        cls, p: Float[Array, "n+1"], c_range: tuple[float, float] = (-1.0, 1.0)
    ) -> Self:
        c = p * (c_range[1] - c_range[0]) + c_range[0]  # [0,1]->c_range
        return cls(c=jnp.asarray(c))

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... 1"]) -> Float[Array, "..."]:
        n = self.degree
        g = jsp.stats.binom.pmf(jnp.arange(n + 1), n, x)
        return jnp.sum(self.c * g, axis=-1)

    @eqx.filter_jit
    def as_degree(self, n: int) -> Self:
        assert n >= self.degree, "Can only bump degree up"
        while self.degree < n:
            # express degree n berstein polynomial as degree n+1 berstein polynomial
            i = jnp.arange(self.degree + 2) / (self.degree + 1)
            c_new = jnp.zeros((*self.c.shape[:-1], self.degree + 2))
            c_new = c_new.at[..., 1:].add(i[1:] * self.c)
            c_new = c_new.at[..., :-1].add((1 - i[:-1]) * self.c)
            self = self._replace(c=c_new)
        return self
