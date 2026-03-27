from typing import Literal
from jaxtyping import Array, Float
from numpy.typing import NDArray as Array
import jax
import jax.numpy as jnp
import equinox as eqx
import scipy as sp
import numpy as np
from .gp import GaussianProcess, EPS


class AcquisitionStrategy(eqx.Module):
    def __call__(self, model: GaussianProcess, q: int = 1) -> Float[Array, "q d"]: ...


class LHS(AcquisitionStrategy):
    multi_starts: int

    def __call__(self, model: GaussianProcess, q: int = 1) -> Float[Array, "q d"]:
        # sample initial guess for candidate points
        n, d = model.observed_xs.shape
        lhs_sampler = sp.stats.qmc.LatinHypercube(d=d, seed=np.random.mtrand._rand)
        x = lhs_sampler.random(n=self.multi_starts)

        # evaluate acquisition function and return best q
        values = np.array([model.log_expected_improvement(xi) for xi in x])
        best_dx = np.argsort(values)[-q:]
        return x[best_dx[:q]]


class BFGS(AcquisitionStrategy):
    multi_starts: int
    raw_samples: int = 512
    max_iterations: int = 100
    ftol: float = EPS
    gtol: float = 0.0

    def __call__(self, model: GaussianProcess, q: int = 1) -> Float[Array, "q d"]:
        # sample initial guess for candidate points
        n, d = model.observed_xs.shape
        lhs_sampler = sp.stats.qmc.LatinHypercube(d=d, seed=np.random.mtrand._rand)
        x = lhs_sampler.random(n=self.raw_samples)

        # keep the most promising initial guesses
        ei = np.array([model.log_expected_improvement(xi) for xi in x])
        best_idx = np.argsort(ei)[-self.multi_starts :]
        x = x[best_idx]

        # optimize each initial guess with L-BFGS
        @jax.jit
        @jax.value_and_grad
        def loss(x):
            return -model.log_expected_improvement(x)

        results = [
            sp.optimize.minimize(
                fun=loss,
                x0=xi,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0, 1) for _ in range(d)],
                options=dict(
                    maxiter=self.max_iterations,
                    ftol=self.ftol,
                    gtol=self.gtol,
                ),
            )
            for xi in x
        ]

        # sort results and return best q
        x = np.array([result.x for result in results])
        idx = np.argsort(np.array([result.fun for result in results]))
        return x[idx[:q]]


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
        values = np.array([model.log_expected_improvement(xi) for xi in x])
        best_idx = np.argsort(values)[-q:]
        return x[best_idx]
