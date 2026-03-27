from functools import partial
from typing import Literal
from jaxtyping import Array, Float, Int
import jax
import jax.numpy as jnp
import equinox as eqx
import scipy as sp
import numpy as np
from .gp import GaussianProcess, lbfgs_minimize, EPS


class AcquisitionStrategy(eqx.Module):
    def __call__(self, model: GaussianProcess, q: int = 1) -> Float[Array, "q d"]: ...


class LHS(AcquisitionStrategy):
    multi_starts: int

    def __call__(self, model: GaussianProcess, q: int = 1) -> Float[Array, "q d"]:
        n, d = model.observed_xs.shape
        lhs_sampler = sp.stats.qmc.LatinHypercube(d=d, seed=np.random.mtrand._rand)
        x = lhs_sampler.random(n=self.multi_starts)
        values = np.array([model.expected_improvement(xi) for xi in x])
        idx = jnp.argsort(values, descending=True)
        return x[idx[:q]]


class BFGS(AcquisitionStrategy):
    multi_starts: int
    max_iterations: int = 100
    ftol: float = EPS
    gtol: float = 0.0  # TODO: add gtol stopping condition

    def __call__(self, model: GaussianProcess, q: int = 1) -> Float[Array, "q d"]:
        # sample initial guess for candidate points
        n, d = model.observed_xs.shape
        lhs_sampler = sp.stats.qmc.LatinHypercube(d=d, seed=np.random.mtrand._rand)
        x = lhs_sampler.random(n=self.multi_starts)

        # optimize each initial guess with L-BFGS and return the best q
        x, losses, iters = self.optimize_initial_guesses(model, x)
        idx = jnp.argsort(losses)
        return x[idx[:q]]

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, 0))
    def optimize_initial_guesses(
        self, model: GaussianProcess, x: Float[Array, "n d"]
    ) -> tuple[Float[Array, "n d"], Float[Array, "n"], Int[Array, "n"]]:
        x_logit, loss, iter = lbfgs_minimize(
            lambda x: -model.expected_improvement(jax.nn.sigmoid(x)),
            x0=jnp.log(x) - jnp.log(1 - x),
            max_iterations=self.max_iterations,
            ftol=self.ftol,
            gtol=self.gtol,
        )
        return jax.nn.sigmoid(x_logit), loss, iter


class Voronoi(AcquisitionStrategy):
    multi_starts: int
    binary_search_steps: int = 30
    sampling_strategy: Literal["uniform", "proj"] = "uniform"

    def __call__(self, model: GaussianProcess, q: int = 1) -> Float[Array, "q d"]:
        # sample initial guess for candidate points
        n, d = model.observed_xs.shape
        lhs_sampler = sp.stats.qmc.LatinHypercube(d=d, seed=np.random.mtrand._rand)
        x0 = lhs_sampler.random(n=self.multi_starts)
        cell_centers = np.argmin(
            sp.spatial.distance.cdist(x0, model.observed_xs), axis=1
        )

        # sample random directions for each candidate
        if self.sampling_strategy == "uniform":
            direction = np.random.randn(self.multi_starts, d)
        elif self.sampling_strategy == "proj":
            direction = x0 - model.observed_xs[cell_centers]
        else:
            raise ValueError(f"Unknown sampling strategy {self.sampling_strategy}")
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        direction *= 1 / jnp.sqrt(d)  # make sure up is outside [0,1]^d

        # binary search (vectorized)
        cell_centers = np.argmin(
            sp.spatial.distance.cdist(x0, model.observed_xs), axis=1
        )
        lb = np.zeros(self.multi_starts)
        up = np.ones(self.multi_starts)
        for i in range(self.binary_search_steps):
            m = (up + lb) / 2
            x = np.clip(x0 + direction * m[..., None], 0, 1)
            new_cell_centers = np.argmin(
                sp.spatial.distance.cdist(x, model.observed_xs), axis=1
            )
            lb = np.where(new_cell_centers == cell_centers, m, lb)
            up = np.where(new_cell_centers == cell_centers, up, m)

        # evaluate acquisition function and return best
        m = (up + lb) / 2
        x = np.clip(x0 + direction * m[..., None], 0, 1)
        values = jnp.array([model.expected_improvement(xi) for xi in x])
        idx = jnp.argsort(values, descending=True)
        return x[idx[:q]]

