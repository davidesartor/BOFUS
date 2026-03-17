from typing import Protocol, Callable
from jaxtyping import Array, Float
import argparse
import os

import jax
import jax.numpy as jnp
import jax.random as jr
import scipy as sp
import numpy as np
import gp


class AcquisitionFunction(Protocol):
    def __call__(self, x: Float[Array, "n d"]) -> Float[Array, "n"]: ...


class AcquisitionStrategy(Protocol):
    def __call__(self, fn: AcquisitionFunction) -> Float[Array, "#n d"]: ...


class LHS(AcquisitionStrategy):
    d: int
    q: int
    lhs_points: int

    def __call__(self, fn: AcquisitionFunction) -> Float[Array, "q d"]:
        x = sp.stats.qmc.LatinHypercube(self.d).random(n=self.lhs_points)
        top_q = jnp.argsort(fn(x))[-self.q :]
        return x[top_q]


class BFGS(AcquisitionStrategy):
    d: int
    q: int
    multi_starts: int
    max_iterations: int

    def __call__(self, fn: AcquisitionFunction) -> Float[Array, "q d"]:
        @jax.jit
        @jax.value_and_grad
        def loss(x):
            return -fn(x[None]).squeeze()

        results = [
            sp.optimize.minimize(
                fun=loss,
                x0=x0,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0, 1) for _ in range(self.d)],
                options=dict(maxiter=self.max_iterations, ftol=gp.EPS, gtol=0),
            )
            for x0 in sp.stats.qmc.LatinHypercube(self.d).random(n=self.multi_starts)
        ]
        x = jnp.array([result.x for result in results])
        fn_vals = -jnp.array([result.fun for result in results])
        top_q = jnp.argsort(fn_vals)[: self.q]
        return x[top_q]
