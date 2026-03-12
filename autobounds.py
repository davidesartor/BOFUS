import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

def squared_distance(
    x1: Float[Array, "n d"],
    x2: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    def dist(a, b):
        return jnp.sum((a - b) ** 2)

    dist = jax.vmap(jax.vmap(dist, (None, 0)), (0, None))
    return dist(x1, x2)


def hetgpy_auto_bounds(
    x: Float[Array, "n d"], min_cor=0.01, max_cor=0.5
) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
    # rescale X to [0,1]^d
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    x = (x - x_min) @ jnp.diag(1 / (x_max - x_min))
    # compute pairwise distances only for proper pair
    dists = squared_distance(x, x)
    dists = dists[jnp.tril(dists, k=-1) > 0]
    # magic initialization using inverse of kernel
    lower = -jnp.quantile(dists, q=0.05) / jnp.log(min_cor) * (x_max - x_min) ** 2
    upper = -jnp.quantile(dists, q=0.95) / jnp.log(max_cor) * (x_max - x_min) ** 2
    return lower, upper
