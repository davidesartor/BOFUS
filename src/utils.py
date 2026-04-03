from typing import Callable, NamedTuple, Self
from jaxtyping import Array, Float, Scalar, Int, PyTree

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import equinox as eqx
import optax

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


class Module(eqx.Module):
    def replace(self, **kwargs)-> Self:
        where = lambda m: tuple(getattr(m, k) for k in kwargs.keys())
        return eqx.tree_at(where, self, kwargs.values(), is_leaf=lambda x: x is None)


def hetgpy_scale_init(
    x: Float[Array, "n d"], min_cor: float = 0.01, max_cor: float = 0.5
) -> Float[Array, "d"]:
    # rescale X to [0,1]^d
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    x = (x - x_min) @ jnp.diag(1 / (x_max - x_min))

    # compute pairwise squared distances only for proper pairs
    d_fn = lambda a, b: jnp.linalg.norm(a - b)
    d_fn = jax.vmap(jax.vmap(d_fn, (None, 0)), (0, None))
    dists = d_fn(x, x) ** 2  # squared distances
    dists = dists[*jnp.tril_indices(len(dists), k=-1)]
    
    # magic hetgpy initialization using inverse of kernel
    lower = -jnp.quantile(dists, q=0.05) / jnp.log(min_cor) * (x_max - x_min) ** 2
    upper = -jnp.quantile(dists, q=0.95) / jnp.log(max_cor) * (x_max - x_min) ** 2
    scale = 0.9 * upper + 0.1 * lower
    return scale


def lbfgs_minimize[T: PyTree](
    f: Callable[[T], Scalar],
    x0: T,
    max_iterations: int = 100,
    ftol: float = EPS,
    gtol: float = 0.0,  # TODO: add gtol stopping condition
) -> tuple[T, Scalar, Int]:
    # initialize L-BFGS optimization
    solver = optax.lbfgs()
    opt = solver.init(x0)
    initial_state = (x0, opt, jnp.inf, 0.0, 0)

    # early stopping condition (false -> stop)
    def while_cond_fn(state):
        x, opt, loss, prev_loss, i = state
        abs_change = jnp.abs(loss - prev_loss)
        return (i < max_iterations) & (abs_change > ftol)

    # optimization step
    def while_body_fn(state):
        x, opt, loss, prev_loss, i = state
        prev_loss = loss
        loss, grad = optax.value_and_grad_from_state(f)(x, state=opt)
        updates, opt = solver.update(grad, opt, x, value=loss, grad=grad, value_fn=f)
        x = optax.apply_updates(x, updates)
        return (x, opt, loss, prev_loss, i + 1)

    # runs the optimization loop (lowered to a single while op)
    x, _, loss, _, iters = jax.lax.while_loop(
        while_cond_fn, while_body_fn, initial_state
    )
    return x, loss, iters  # type: ignore