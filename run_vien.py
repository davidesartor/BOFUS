from typing import Callable
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import equinox as eqx
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from src import kernels, gp, acquisition


jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


def f_from_array(
    rkhs: gp.RKHS,
    p: Float[Array, "k d+1"],
    x_range: tuple[float, float] = (0.0, 1.0),
    a_range: tuple[float, float] = (-1.0, 1.0),
) -> gp.RKHSFunction:
    x, a = p[:, :-1], p[:, -1]
    x = x * (x_range[1] - x_range[0]) + x_range[0]  # [0,1]->x_range
    a = a * (a_range[1] - a_range[0]) + a_range[0]  # [0,1]->a_range
    return gp.RKHSFunction(kernel=rkhs, a=a, x=x)


@eqx.filter_jit
def sparsify(f: gp.RKHSFunction, k: int) -> gp.RKHSFunction:
    """Sparsify using Kernel Matching Pursuit (Vincent & Bengio 2002)"""
    D = f.kernel(f.x, f.x)
    norm = jnp.linalg.norm(D, axis=0)

    def scan_fn(residual, _):
        # select the basis function with the largest correlation to the residual
        gamma = jnp.argmax(jnp.abs((D @ residual) / norm))
        # and compute the corresponding coefficient
        alpha = (D[gamma] @ residual) / norm[gamma] ** 2
        # project the residual onto the remaining basis functions
        residual = residual - alpha * D[gamma]
        return residual, (alpha, gamma)

    residual = jax.vmap(f)(f.x)
    residual, (a, idx) = jax.lax.scan(scan_fn, residual, None, length=k)
    return f._replace(a=a, x=f.x[idx])


# custom version for vien to handle the extra x0 argument
def optimize_lhs_candidates(
    acquisition_loss: Callable[[Float[Array, "k"], Float[Array, "k d"]], Scalar],
    a0: Float[Array, "n k"],
    x0: Float[Array, "n k"],
    max_restarts: int,
    optimizer_options: dict = dict(maxiter=100, ftol=EPS, gtol=0.0),
) -> tuple[Float[Array, "k"], Float[Array, "k d"]]:
    # only keep the best initial candidates
    loss_fn = jax.jit(jax.value_and_grad(acquisition_loss))
    losses = np.array([loss_fn(a, x)[0] for a, x in zip(a0, x0)])
    best_idx = jnp.argsort(losses)[:max_restarts]
    a0, x0 = a0[best_idx], x0[best_idx]

    # optimize each initial guesses with L-BFGS-B
    results = [
        sp.optimize.minimize(
            fun=loss_fn,
            x0=a,
            args=(x,),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * len(a),
            options=optimizer_options,
        )
        for x, a in zip(x0, a0)
    ]

    # sort results and return the best one
    losses = jnp.array([result.fun for result in results])
    best_idx = jnp.argmin(losses)
    best_a, best_x = results[best_idx].x, x0[best_idx]
    return best_a, best_x


def run(
    seed: int,
    target_fn: Callable,
    rkhs: gp.RKHS,
    surrogate_model: gp.FunctionalGaussianProcess,
    # simulation parameters
    initial_acquisitions: int,
    min_basis_points: int,
    max_basis_points: int,
    acquisitions_each: int,
    acquisition_raw_samples: int,
    acquisition_max_restarts: int,
    input_range: tuple[float, float] = (0.0, 1.0),
    coefficient_range: tuple[float, float] = (-1.0, 1.0),
):
    # set random seed
    rng = np.random.default_rng(seed=seed)
    as_function = lambda p: f_from_array(
        rkhs, p.reshape(-1, d + 1), x_range=input_range, a_range=coefficient_range
    )

    # initial acquisition (work with flat, normalized parameters in [0, 1])
    k = min_basis_points
    d = 1  # assuming 1-dimensional input
    sampler = sp.stats.qmc.LatinHypercube(d=k * (d + 1), rng=rng)
    ps = sampler.random(n=initial_acquisitions)

    # evaluate target function at initial points and fit surrogate model
    fs = [as_function(p) for p in ps]
    ys = jnp.array([target_fn(f) for f in fs])
    surrogate_model = surrogate_model.fit(fs, ys)

    # sequential acquisition loop
    for k in range(min_basis_points, max_basis_points + 1):
        sampler = sp.stats.qmc.LatinHypercube(d=k * (d + 1), rng=rng)
        for i in range(acquisitions_each):
            # find all unique x locations used so far
            all_xs = jnp.concatenate([f.x for f in fs], axis=0)
            all_xs = jnp.unique(all_xs, axis=0)

            # sample initial guess for candidate points
            candidate_fs = [
                as_function(pi) for pi in sampler.random(n=acquisition_raw_samples)
            ]

            # expand the basis representation to include all xs
            a0 = [jnp.concat([fi.a, jnp.zeros(len(all_xs))]) for fi in candidate_fs]
            x0 = [jnp.concat([fi.x, all_xs]) for fi in candidate_fs]

            # acquisition function
            def acquisition_loss(
                a: Float[Array, "k"], x: Float[Array, "k d"]
            ) -> Scalar:
                f = gp.RKHSFunction(kernel=rkhs, x=x, a=a)
                mu, cov = surrogate_model.predict([f])
                return -acquisition.log_expected_improvement(
                    mu=mu.squeeze(),
                    sigma=cov.squeeze() ** 0.5,
                    y_best=surrogate_model.observed_ys.min(),
                )

            # custom optimize acquisition function to save the x0 as well
            a, x = optimize_lhs_candidates(
                acquisition_loss=acquisition_loss,
                a0=jnp.array(a0),
                x0=jnp.array(x0),
                max_restarts=acquisition_max_restarts,
            )

            # construct the new candidate function and sparsify it
            f = gp.RKHSFunction(kernel=rkhs, x=x, a=a)
            f = sparsify(f, k=k)

            # evaluate target function at the new point
            y = target_fn(f)

            # fit surrogate model on the new data
            fs = fs + [f]
            ys = jnp.concatenate([ys, y[None]])
            surrogate_model = surrogate_model.fit(fs, ys)

            print(f"Iteration {i+1}: current= {y:.8f}, best = {ys.min():.8f}")
    return ys


if __name__ == "__main__":
    seed = 0
    initial_acquisitions = 10
    min_basis_points = 1
    max_basis_points = 5
    acquisitions_each = 10
    acquisition_raw_samples = 1000
    acquisition_max_restarts = 16

    rkhs = gp.RKHS(
        metric=kernels.Euclidean(),
        profile=kernels.SquaredExponential(),
        rho=1 / (4 * jnp.pi),  # type: ignore
    )

    def target_fn(f, n: int = 1000):
        x = np.linspace(0, 1, n).reshape(-1, 1)
        y = np.sinc(x * 2 * jnp.pi).squeeze()
        pred = np.array([f(xi) for xi in x])
        return np.mean(np.square(pred - y))

    ys = run(
        seed=seed,
        target_fn=target_fn,
        rkhs=rkhs,
        surrogate_model=gp.FunctionalGaussianProcess(),
        initial_acquisitions=initial_acquisitions,
        min_basis_points=min_basis_points,
        max_basis_points=max_basis_points,
        acquisitions_each=acquisitions_each,
        acquisition_raw_samples=acquisition_raw_samples,
        acquisition_max_restarts=acquisition_max_restarts,
    )


    plt.figure(figsize=(5, 3))
    n0, dn = initial_acquisitions, acquisitions_each
    plt.plot(range(0, n0), ys[:n0], "o", label="initial samples")
    for i, degree in enumerate(range(min_basis_points, max_basis_points + 1)):
        y_deg = ys[n0 + i * dn : n0 + (i + 1) * dn]
        plt.plot(
            range(n0 + i * dn, n0 + (i + 1) * dn),
            y_deg,
            "o",
            label=f"acquired samples (degree={degree})",
        )

    plt.yscale("log")
    plt.xlabel("Total Evaluations")
    plt.ylabel("Target fn")
    plt.title("Convergence of Bayesian Optimization")
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig("vien.png", dpi=300, bbox_inches="tight")
    plt.close()
