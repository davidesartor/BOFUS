from typing import Protocol
from jaxtyping import Array, Float
import jax
import jax.numpy as jnp
import scipy as sp
import numpy as np
from . import gp

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


class AcquisitionStrategy(Protocol):
    def __call__(
        self, model: gp.GaussianProcess, x: Float[Array, "n d"], *args, **kwargs
    ) -> Float[Array, "n"]: ...


class LHS(AcquisitionStrategy):
    def __call__(self, model: gp.GaussianProcess, points: int) -> Float[Array, "#n d"]:
        n, d = model.x.shape
        lhs_sampler = sp.stats.qmc.LatinHypercube(d=d, seed=np.random.mtrand._rand)
        x = lhs_sampler.random(n=points)
        ei = model.expected_improvement(x)
        best = jnp.argmax(ei)
        return x[best : best + 1]


class BFGS(AcquisitionStrategy):
    def __call__(
        self,
        model: gp.GaussianProcess,
        multi_starts: int,
        max_iterations: int,
    ) -> Float[Array, "#n d"]:
        @jax.jit
        @jax.value_and_grad
        def loss(x):
            return -model.expected_improvement(x[None]).squeeze()

        def verbose_loss(x):
            v, g = loss(x)
            nan_in_v = jnp.isnan(v).any()
            nan_in_g = jnp.isnan(g).any()
            if nan_in_v or nan_in_g:
                mu, cov = model.predict(x[None])
                sigma = jnp.diag(cov) ** 0.5
                y_star = model.y.min()
                z = (y_star - mu) / sigma
                print()
                print(f"NaN encountered")
                print(f"NaN in loss: {nan_in_v}, NaN in gradient: {nan_in_g}")
                print(f"at z={z}, mu={mu}, sigma={sigma}")
                print()
            return v, g

        n, d = model.x.shape
        lhs_sampler = sp.stats.qmc.LatinHypercube(d=d, seed=np.random.mtrand._rand)
        results = [
            sp.optimize.minimize(
                fun=verbose_loss,
                x0=x0,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0, 1) for _ in range(d)],
                options=dict(maxiter=max_iterations, ftol=EPS, gtol=0),
            )
            for x0 in lhs_sampler.random(n=multi_starts)
        ]
        x = jnp.array([result.x for result in results])
        best = jnp.argmin(jnp.array([result.fun for result in results]))
        return x[best : best + 1]