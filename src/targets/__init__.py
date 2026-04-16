from typing import Protocol, Callable
from jaxtyping import Float, Array, Scalar

import jax
import jax.numpy as jnp
import jax.random as jr


class TestFunction(Protocol):
    d: int

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar: ...


class SincProjection:
    def __init__(self, d: int = 1, seed: int = 0, n: int = 10000):
        self.d = d
        self.grid = jr.uniform(jr.key(seed), (n, d))

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar:
        target = jnp.sinc(2 * jnp.pi * self.grid - jnp.pi).prod(axis=-1)
        pred = jax.vmap(f)(self.grid)
        return jnp.mean(jnp.square(pred - target))


from .neuralnetworks import MNIST
from .gymnasium import Pendulum
