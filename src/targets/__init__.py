from typing import Protocol, Callable
from jaxtyping import Float, Array, Scalar

import jax
import jax.numpy as jnp
import jax.random as jr

from . import virtual_library


class TestFunction(Protocol):
    d: int

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar: ...


class Ridge(TestFunction):
    def __init__(
        self, profile: virtual_library.TestFunction, d: int, seed: int = 0
    ):
        self.d = d
        self.profile = profile
        k1, k2, k3 = jr.split(jr.key(seed), 3)
        # sample d directions g = sum a * k(x, .)
        self.a = jr.uniform(k1, (d, d), minval=-1.0, maxval=1.0)
        self.x = jr.uniform(k2, (d, d, d), minval=0.0, maxval=1.0)
        # sample d biases b
        self.b = jr.uniform(k3, (d,), minval=-1.0, maxval=1.0)

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar:
        f = jax.vmap(jax.vmap(f))  # vectorize so it can be evaluated on x in one go
        g = self.b + jnp.sum(self.a * f(self.x), axis=-1)
        g = jax.nn.sigmoid(g)  # squash to [0, 1]
        return self.profile(g)


class SincProjection:
    def __init__(self, d: int = 1, seed: int = 0, n: int = 10000):
        self.d = d
        self.grid = jr.uniform(jr.key(seed), (n, d))

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar:
        target = jnp.sinc(2 * jnp.pi * self.grid - jnp.pi).mean(axis=-1)
        pred = jax.vmap(f)(self.grid)
        return jnp.mean(jnp.square(pred - target))


from .gymnasium import Pendulum
from .pinwheel import PinWheel
from .neuralnetworks import MNIST
from .brachistochrone import Brachistochrone
