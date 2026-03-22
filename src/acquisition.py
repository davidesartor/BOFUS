from typing import Literal, Protocol
from jaxtyping import Array, Float, Scalar, PyTree
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import scipy as sp
import numpy as np
from . import gp

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


class AcquisitionFunction[T: PyTree](Protocol):
    def __call__(self, x: T) -> Scalar: ...


class AcquisitionStrategy[T: PyTree](Protocol):
    def __call__(
        self, 
        acqusition_fn: AcquisitionFunction[T],
        observed_xs: T,
        q: int = 1
    ) -> T: ... 


@dataclass
class LHS(AcquisitionStrategy[Float[Array, "d"]]):
    multi_starts: int

    def __call__(
        self, 
        acqusition_fn: AcquisitionFunction, 
        observed_xs: Float[Array, "n d"],
        q: int = 1
    ) -> Float[Array, "q d"]:
        n, d = observed_xs.shape
        lhs_sampler = sp.stats.qmc.LatinHypercube(d=d, seed=np.random.mtrand._rand)
        x = lhs_sampler.random(n=self.multi_starts)
        values = np.array([acqusition_fn(xi) for xi in x])
        idx = jnp.argsort(values, descending=True)
        return x[idx[:q]]


@dataclass
class BFGS(AcquisitionStrategy[Float[Array, "d"]]):
    multi_starts: int
    max_iterations: int = 100

    def __call__(
        self, 
        acqusition_fn: AcquisitionFunction, 
        observed_xs: Float[Array, "n d"],
        q: int = 1
    ) -> Float[Array, "q d"]:
        @jax.jit
        @jax.value_and_grad
        def loss(x):
            return -acqusition_fn(x)

        n, d = observed_xs.shape
        lhs_sampler = sp.stats.qmc.LatinHypercube(d=d, seed=np.random.mtrand._rand)
        results = [
            sp.optimize.minimize(
                fun=loss,
                x0=x0,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0, 1) for _ in range(d)],
                options=dict(maxiter=self.max_iterations, ftol=EPS, gtol=0),
            )
            for x0 in lhs_sampler.random(n=self.multi_starts)
        ]
        x = jnp.array([result.x for result in results])
        idx = jnp.argsort(jnp.array([result.fun for result in results]))
        return x[idx[:q]]


@dataclass
class Voronoi(AcquisitionStrategy[Float[Array, "d"]]):
    multi_starts: int
    binary_search_steps: int = 30 
    sampling_strategy: Literal["uniform", "proj"] = "uniform"

    def __call__(
        self, 
        acqusition_fn: AcquisitionFunction, 
        observed_xs: Float[Array, "n d"],
        q: int = 1
    ) -> Float[Array, "q d"]:
        # sample initial guess for candidate points
        n, d = observed_xs.shape
        lhs_sampler = sp.stats.qmc.LatinHypercube(d=d, seed=np.random.mtrand._rand)
        x0 = lhs_sampler.random(n=self.multi_starts)
        cell_centers = np.argmin(sp.spatial.distance.cdist(x0, observed_xs), axis=1)

        # sample random directions for each candidate
        if self.sampling_strategy == "uniform":
            direction = np.random.randn(self.multi_starts, d)
        elif self.sampling_strategy == "proj":
            direction = x0 - observed_xs[cell_centers]
        else:
            raise ValueError(f"Unknown sampling strategy {self.sampling_strategy}")
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        direction *= 1 / jnp.sqrt(d)  # make sure up is outside [0,1]^d

        # binary search (vectorized)
        cell_centers = np.argmin(sp.spatial.distance.cdist(x0, observed_xs), axis=1)
        lb = np.zeros(self.multi_starts)
        up = np.ones(self.multi_starts)
        for i in range(self.binary_search_steps):
            m = (up + lb) / 2
            x = np.clip(x0 + direction * m[..., None], 0, 1)
            new_cell_centers = np.argmin(
                sp.spatial.distance.cdist(x, observed_xs), axis=1
            )
            lb = np.where(new_cell_centers == cell_centers, m, lb)
            up = np.where(new_cell_centers == cell_centers, up, m)

        # evaluate acquisition function and return best
        m = (up + lb) / 2
        x = np.clip(x0 + direction * m[..., None], 0, 1)
        values = jnp.array([acqusition_fn(xi) for xi in x])
        idx = jnp.argsort(values, descending=True)
        return x[idx[:q]]
