from jaxtyping import Array, Float, Scalar
import jax
import jax.numpy as jnp
import equinox as eqx
import scipy as sp
import numpy as np

import argparse
import time
import os
import pickle

from src import gp, kernels, acquisition, rkhs, targets

jax.config.update("jax_enable_x64", True)


def run_wycoff(
    seed: int,
    target_fn: targets.TestFunction,
    kernel: rkhs.RKHS,
    surrogate_model: gp.FunctionalGaussianProcess,
    # simulation parameters
    initial_acquisitions: int,
    minimum_k: int,  # number of basis points
    maximum_k: int,  # number of basis points
    acquisitions_each_k: int,
    acquisition_raw_samples: int,
    acquisition_max_restarts: int,
):
    # initialize run rng and timers
    rng = np.random.default_rng(seed=seed)
    surrogate_fit_time = 0.0
    acquisition_time = 0.0
    target_evaluation_time = 0.0

    print("Sampling initial acquisition...")
    timer = time.time()
    k = maximum_k
    candidate_sampler = sp.stats.qmc.LatinHypercube(d=k * (kernel.d + 1), rng=rng)
    ps = candidate_sampler.random(n=initial_acquisitions)
    fs = [rkhs.Function.from_array(kernel, p.reshape(k, kernel.d + 1)) for p in ps]
    acquisition_time += time.time() - timer
    print(f"Done! (total acquisition time: {acquisition_time:.2f}s)\n")

    print("Evaluating target function at initial acquisition points...")
    timer = time.time()
    ys = jnp.array([target_fn(f) for f in fs])
    target_evaluation_time += time.time() - timer
    print(f"Done! (total target eval time: {target_evaluation_time:.2f}s)\n")

    print("Fitting surrogate model on initial acquisition points...")
    timer = time.time()
    surrogate_model = surrogate_model.fit(fs, ys)
    surrogate_fit_time += time.time() - timer
    print(f"Done! (total surrogate fit time: {surrogate_fit_time:.2f}s)\n")

    # sequential acquisition loop
    for k in range(minimum_k, maximum_k + 1):
        candidate_sampler = sp.stats.qmc.LatinHypercube(d=k * (kernel.d + 1), rng=rng)
        for i in range(acquisitions_each_k):
            # define acquisition loss function
            def acquisition_loss(p: Float[Array, "k * (d+1)"]) -> Scalar:
                f = rkhs.Function.from_array(kernel, p.reshape(k, kernel.d + 1))
                mu, cov = surrogate_model.predict([f])
                return -acquisition.log_expected_improvement(
                    mu=mu.squeeze(),
                    sigma=cov.squeeze() ** 0.5,
                    y_best=surrogate_model.observed_ys.min(),
                )

            print(f"Optimizing acquisition function...")
            timer = time.time()
            ps = candidate_sampler.random(n=acquisition_raw_samples)
            p, _ = acquisition.optimize_lhs_candidates(
                acquisition_loss=acquisition_loss,
                candidates=ps,
                max_restarts=acquisition_max_restarts,
            )
            acquisition_time += time.time() - timer
            print(f"Done! (total acquisition time: {acquisition_time:.2f}s)\n")

            print("Evaluating target function at new acquisition point...")
            timer = time.time()
            f = rkhs.Function.from_array(kernel, p.reshape(k, kernel.d + 1))
            y = target_fn(f)
            target_evaluation_time += time.time() - timer
            print(f"Done! (total target eval time: {target_evaluation_time:.2f}s)\n")

            print("Updating surrogate model with new acquisition point...")
            timer = time.time()
            fs = fs + [f]
            ys = jnp.concatenate([ys, y[None]])
            surrogate_model = surrogate_model.fit(fs, ys)
            surrogate_fit_time += time.time() - timer
            print(f"Done! (total surrogate fit time: {surrogate_fit_time:.2f}s)\n")

            print(f"Iteration {i+1}: current= {y:.8f}, best = {ys.min():.8f}\n")

    return dict(
        observation_locations=fs,
        observation_values=ys,
        surrogate_fit_time=surrogate_fit_time,
        acquisition_time=acquisition_time,
        target_evaluation_time=target_evaluation_time,
    )


def run_vellanky(
    seed: int,
    target_fn: targets.TestFunction,
    kernel: rkhs.RKHS,  # ignored, only used to assert 1D input space
    surrogate_model: gp.GaussianProcess,
    # simulation parameters
    initial_acquisitions: int,
    minimum_k: int,  # polynomial degree
    maximum_k: int,  # polynomial degree
    acquisitions_each_k: int,
    acquisition_raw_samples: int,
    acquisition_max_restarts: int,
):
    assert kernel.d == 1, "Vellanky's method only supports 1D input spaces"
    # initialize run rng and timers
    rng = np.random.default_rng(seed=seed)
    surrogate_fit_time = 0.0
    acquisition_time = 0.0
    target_evaluation_time = 0.0

    print("Sampling initial acquisition...")
    timer = time.time()
    candidate_sampler = sp.stats.qmc.LatinHypercube(d=minimum_k + 1, rng=rng)
    cs = candidate_sampler.random(n=initial_acquisitions)
    fs = [rkhs.BernsteinPolynomial.from_array(c) for c in cs]
    acquisition_time += time.time() - timer
    print(f"Done! (total initial acquisition time: {time.time() - timer:.2f}s)")

    print("Evaluating target function at initial acquisition points...")
    timer = time.time()
    ys = jnp.array([target_fn(f) for f in fs])
    target_evaluation_time += time.time() - timer
    print(f"Done! (total target evaluation time: {target_evaluation_time:.2f}s)\n")

    print("Fitting surrogate model on initial acquisition points...")
    timer = time.time()
    surrogate_model = surrogate_model.fit(cs, ys)
    print(f"Done! (total surrogate fit time: {time.time() - timer:.2f}s)\n")

    # sequential acquisition loop
    for degree in range(minimum_k, maximum_k + 1):
        print(f"Adjusting surrogate model to degree {degree}...")
        timer = time.time()
        fs = [f.as_degree(degree) for f in fs]
        cs = jnp.array([f.c for f in fs])
        cs = (cs + 1) / 2  # renormalize from [-1, 1] -> [0, 1]
        surrogate_model = surrogate_model.fit(cs, ys)
        surrogate_fit_time += time.time() - timer
        print(f"Done! (total surrogate fit time: {surrogate_fit_time:.2f}s)\n")

        candidate_sampler = sp.stats.qmc.LatinHypercube(d=degree + 1, rng=rng)
        for i in range(acquisitions_each_k):
            # define acquisition loss function
            def acquisition_loss(c: Float[Array, "n+1"]) -> Scalar:
                mu, cov = surrogate_model.predict(c[None, :])
                return -acquisition.log_expected_improvement(
                    mu=mu.squeeze(),
                    sigma=cov.squeeze() ** 0.5,
                    y_best=surrogate_model.observed_ys.min(),
                )

            print(f"Optimizing acquisition function...")
            timer = time.time()
            c, _ = acquisition.optimize_lhs_candidates(
                acquisition_loss=acquisition_loss,
                candidates=candidate_sampler.random(n=acquisition_raw_samples),
                max_restarts=acquisition_max_restarts,
            )
            acquisition_time += time.time() - timer
            print(f"Done! (total acquisition time: {acquisition_time:.2f}s)\n")

            print("Evaluating target function at new acquisition point...")
            timer = time.time()
            f = rkhs.BernsteinPolynomial.from_array(c)
            y = target_fn(f)
            target_evaluation_time += time.time() - timer
            print(f"Done! (total target eval time: {target_evaluation_time:.2f}s)\n")

            print("Updating surrogate model with new acquisition point...")
            fs = fs + [f]
            cs = jnp.concatenate([cs, c[None]])
            ys = jnp.concatenate([ys, y[None]])
            surrogate_model = surrogate_model.fit(cs, ys)
            surrogate_fit_time += time.time() - timer
            print(f"Done! (total surrogate fit time: {surrogate_fit_time:.2f}s)\n")

            print(f"Iteration {i+1}: current= {y:.8f}, best = {ys.min():.8f}\n")

    return dict(
        observation_locations=fs,
        observation_values=ys,
        surrogate_fit_time=surrogate_fit_time,
        acquisition_time=acquisition_time,
        target_evaluation_time=target_evaluation_time,
    )


def run_kundu(
    seed: int,
    target_fn: targets.TestFunction,
    kernel: rkhs.RKHS,  # ignored by Vellanky's method
    surrogate_model: gp.FunctionalGaussianProcess,
    # simulation parameters
    initial_acquisitions: int,
    minimum_k: int,  # number of basis points
    maximum_k: int,  # number of basis points
    acquisitions_each_k: int,
    acquisition_raw_samples: int,
    acquisition_max_restarts: int,
):

    def linear_combination(
        basis_fs: list[rkhs.Function],
        coefficients: Float[Array, "n"],
    ) -> rkhs.Function:
        x = [fi.x for fi in basis_fs]
        a = [fi.a * ci for fi, ci in zip(basis_fs, coefficients)]
        return rkhs.Function(kernel, x=jnp.concat(x), a=jnp.concat(a))

    # initialize run rng and timers
    rng = np.random.default_rng(seed=seed)
    surrogate_fit_time = 0.0
    acquisition_time = 0.0
    target_evaluation_time = 0.0

    print("Sampling initial acquisition...")
    timer = time.time()
    k = maximum_k  # number of basis points = maximum subspace dimension
    basis_sampler = sp.stats.qmc.LatinHypercube(d=k * (kernel.d + 1), rng=rng)
    fs = [
        rkhs.Function.from_array(kernel, p.reshape(k, kernel.d + 1))
        for p in basis_sampler.random(n=initial_acquisitions)
    ]
    acquisition_time += time.time() - timer
    print(f"Done! (total acquisition time: {acquisition_time:.2f}s)\n")

    print("Evaluating target function at initial acquisition points...")
    timer = time.time()
    ys = jnp.array([target_fn(f) for f in fs])
    target_evaluation_time += time.time() - timer
    print(f"Done! (total target eval time: {target_evaluation_time:.2f}s)\n")

    print("Fitting surrogate model on initial acquisition points...")
    timer = time.time()
    surrogate_model = surrogate_model.fit(fs, ys)
    surrogate_fit_time += time.time() - timer
    print(f"Done! (total surrogate fit time: {surrogate_fit_time:.2f}s)\n")

    # sequential acquisition loop
    for n in range(minimum_k, maximum_k + 1):
        candidate_sampler = sp.stats.qmc.LatinHypercube(d=n, rng=rng)
        for i in range(acquisitions_each_k):
            print(f"Sampling random subspace of dimension {n}...")
            timer = time.time()
            basis_fs = [
                rkhs.Function.from_array(kernel, p.reshape(k, kernel.d + 1))
                for p in basis_sampler.random(n=n)
            ]
            acquisition_time += time.time() - timer
            print(f"Done! (total acquisition time: {acquisition_time:.2f}s)\n")

            # define acquisition loss function
            def acquisition_loss(p: Float[Array, "n"]) -> Scalar:
                f = linear_combination(basis_fs, p)
                mu, cov = surrogate_model.predict([f])
                return -acquisition.log_expected_improvement(
                    mu=mu.squeeze(),
                    sigma=cov.squeeze() ** 0.5,
                    y_best=surrogate_model.observed_ys.min(),
                )

            print(f"Optimizing acquisition function...")
            timer = time.time()
            p, _ = acquisition.optimize_lhs_candidates(
                acquisition_loss=acquisition_loss,
                candidates=candidate_sampler.random(n=acquisition_raw_samples),
                max_restarts=acquisition_max_restarts,
            )
            f = linear_combination(basis_fs, p)
            acquisition_time += time.time() - timer
            print(f"Done! (total acquisition time: {acquisition_time:.2f}s)\n")

            print("Evaluating target function at new acquisition point...")
            timer = time.time()
            y = target_fn(f)
            target_evaluation_time += time.time() - timer
            print(f"Done! (total target eval time: {target_evaluation_time:.2f}s)\n")

            print("Updating surrogate model with new acquisition point...")
            fs = fs + [f]
            ys = jnp.concatenate([ys, y[None]])
            surrogate_model = surrogate_model.fit(fs, ys)
            surrogate_fit_time += time.time() - timer
            print(f"Done! (total surrogate fit time: {surrogate_fit_time:.2f}s)\n")

            print(f"Iteration {i+1}: current= {y:.8f}, best = {ys.min():.8f}\n")

    return dict(
        observation_locations=fs,
        observation_values=ys,
        surrogate_fit_time=surrogate_fit_time,
        acquisition_time=acquisition_time,
        target_evaluation_time=target_evaluation_time,
    )


def run_vien(
    seed: int,
    target_fn: targets.TestFunction,
    kernel: rkhs.RKHS,
    surrogate_model: gp.FunctionalGaussianProcess,
    # simulation parameters
    initial_acquisitions: int,
    minimum_k: int,  # number of basis points
    maximum_k: int,  # number of basis points
    acquisitions_each_k: int,
    acquisition_raw_samples: int,
    acquisition_max_restarts: int,
):
    @eqx.filter_jit
    def sparsify(f: rkhs.Function, k: int) -> rkhs.Function:
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

    # initialize run rng and timers
    rng = np.random.default_rng(seed=seed)
    surrogate_fit_time = 0.0
    acquisition_time = 0.0
    target_evaluation_time = 0.0

    print("Sampling initial acquisition...")
    timer = time.time()
    k = minimum_k
    candidate_sampler = sp.stats.qmc.LatinHypercube(d=k * (kernel.d + 1), rng=rng)
    fs = [
        rkhs.Function.from_array(kernel, p.reshape(k, kernel.d + 1))
        for p in candidate_sampler.random(n=initial_acquisitions)
    ]
    acquisition_time += time.time() - timer
    print(f"Done! (total acquisition time: {acquisition_time:.2f}s)\n")

    print("Evaluating target function at initial acquisition points...")
    timer = time.time()
    ys = jnp.array([target_fn(f) for f in fs])
    target_evaluation_time += time.time() - timer
    print(f"Done! (total target eval time: {target_evaluation_time:.2f}s)\n")

    print("Fitting surrogate model on initial acquisition points...")
    timer = time.time()
    surrogate_model = surrogate_model.fit(fs, ys)
    surrogate_fit_time += time.time() - timer
    print(f"Done! (total surrogate fit time: {surrogate_fit_time:.2f}s)\n")

    # sequential acquisition loop
    for k in range(minimum_k, maximum_k + 1):
        candidate_sampler = sp.stats.qmc.LatinHypercube(d=k * (kernel.d + 1), rng=rng)
        for i in range(acquisitions_each_k):
            print(f"Preparing acquisition candidates...")
            timer = time.time()
            candidate_fs = [
                rkhs.Function.from_array(kernel, p.reshape(k, kernel.d + 1))
                for p in candidate_sampler.random(n=acquisition_raw_samples)
            ]
            # expand the basis representation to include all xs
            all_xs = jnp.unique(jnp.concat([f.x for f in fs]), axis=0)
            a0 = [jnp.concat([fi.a, jnp.zeros(len(all_xs))]) for fi in candidate_fs]
            x0 = [jnp.concat([fi.x, all_xs]) for fi in candidate_fs]
            acquisition_time += time.time() - timer
            print(f"Done! (total acquisition time: {acquisition_time:.2f}s)\n")

            # define acquisition loss
            def acquisition_loss(
                a: Float[Array, "k"], x: Float[Array, "k d"]
            ) -> Scalar:
                f = rkhs.Function(kernel, x=x, a=a)
                mu, cov = surrogate_model.predict([f])
                return -acquisition.log_expected_improvement(
                    mu=mu.squeeze(),
                    sigma=cov.squeeze() ** 0.5,
                    y_best=surrogate_model.observed_ys.min(),
                )

            print(f"Optimizing acquisition function...")
            timer = time.time()
            a, x = acquisition.optimize_lhs_candidates(
                acquisition_loss=acquisition_loss,
                candidates=jnp.array(a0),
                extra_args=x0,
                max_restarts=acquisition_max_restarts,
            )
            acquisition_time += time.time() - timer
            print(f"Done! (total acquisition time: {acquisition_time:.2f}s)\n")

            print("Evaluating target function at new acquisition point...")
            timer = time.time()
            f = sparsify(rkhs.Function(kernel, x=x, a=a), k=k)  # type: ignore
            y = target_fn(f)
            target_evaluation_time += time.time() - timer
            print(f"Done! (total target eval time: {target_evaluation_time:.2f}s)\n")

            print("Updating surrogate model with new acquisition point...")
            timer = time.time()
            fs = fs + [f]
            ys = jnp.concatenate([ys, y[None]])
            surrogate_model = surrogate_model.fit(fs, ys)
            surrogate_fit_time += time.time() - timer
            print(f"Done! (total surrogate fit time: {surrogate_fit_time:.2f}s)\n")

            print(f"Iteration {i+1}: current= {y:.8f}, best = {ys.min():.8f}\n")

    return dict(
        observation_locations=fs,
        observation_values=ys,
        surrogate_fit_time=surrogate_fit_time,
        acquisition_time=acquisition_time,
        target_evaluation_time=target_evaluation_time,
    )


def run_shilton(
    seed: int,
    target_fn: targets.TestFunction,
    kernel: rkhs.RKHS,
    surrogate_model: gp.FunctionalGaussianProcess,
    # simulation parameters
    initial_acquisitions: int,
    minimum_k: int,  # number of basis points
    maximum_k: int,  # number of basis points
    acquisitions_each_k: int,
    acquisition_raw_samples: int,
    acquisition_max_restarts: int,
):
    def sample_from_gp_prior(basis_size: int, grid_size: int = 30):
        xs = grid_sampler.random(n=grid_size)
        mean = jnp.zeros(len(xs))
        cov = kernel(xs, xs)
        ys = rng.multivariate_normal(mean, cov, size=basis_size)
        return xs, ys

    # initialize run rng and timers
    rng = np.random.default_rng(seed=seed)
    surrogate_fit_time = 0.0
    acquisition_time = 0.0
    target_evaluation_time = 0.0

    print("Sampling initial acquisition...")
    timer = time.time()
    grid_sampler = sp.stats.qmc.LatinHypercube(d=kernel.d, rng=rng)
    xs_grid, ys_grids = sample_from_gp_prior(basis_size=initial_acquisitions)
    fs = [rkhs.Function.from_xy(kernel, x=xs_grid, y=ys_grid) for ys_grid in ys_grids]
    acquisition_time += time.time() - timer
    print(f"Done! (total acquisition time: {acquisition_time:.2f}s)\n")

    print("Evaluating target function at initial acquisition points...")
    timer = time.time()
    ys = jnp.array([target_fn(f) for f in fs])
    target_evaluation_time += time.time() - timer
    print(f"Done! (total target eval time: {target_evaluation_time:.2f}s)\n")

    print("Fitting surrogate model on initial acquisition points...")
    timer = time.time()
    surrogate_model = surrogate_model.fit(fs, ys)
    surrogate_fit_time += time.time() - timer
    print(f"Done! (total surrogate fit time: {surrogate_fit_time:.2f}s)\n")

    # sequential acquisition loop
    for n in range(minimum_k, maximum_k + 1):
        sampler = sp.stats.qmc.LatinHypercube(d=n, rng=rng)
        for i in range(acquisitions_each_k):
            print(f"Sampling random subspace of dimension {n}...")
            timer = time.time()
            xs_grid, ys_grids = sample_from_gp_prior(basis_size=n)
            b_grid = np.array([fs[jnp.argmin(ys)](x) for x in xs_grid])
            acquisition_time += time.time() - timer
            print(f"Done! (total acquisition time: {acquisition_time:.2f}s)\n")

            # define acquisition loss function
            def acquisition_loss(c: Float[Array, "n"]) -> Scalar:
                ys_grid = jnp.array(ys_grids.T @ c + b_grid)
                f = rkhs.Function.from_xy(kernel, x=xs_grid, y=ys_grid)
                mu, cov = surrogate_model.predict([f])
                return -acquisition.log_expected_improvement(
                    mu=mu.squeeze(),
                    sigma=cov.squeeze() ** 0.5,
                    y_best=surrogate_model.observed_ys.min(),
                )

            print(f"Optimizing acquisition function...")
            timer = time.time()
            c, _ = acquisition.optimize_lhs_candidates(
                acquisition_loss=acquisition_loss,
                candidates=sampler.random(n=acquisition_raw_samples),
                max_restarts=acquisition_max_restarts,
            )
            acquisition_time += time.time() - timer
            print(f"Done! (total acquisition time: {acquisition_time:.2f}s)\n")

            print("Evaluating target function at new acquisition point...")
            timer = time.time()
            ys_grid = jnp.array(ys_grids.T @ c + b_grid)
            f = rkhs.Function.from_xy(kernel, x=xs_grid, y=ys_grid)
            y = target_fn(f)
            target_evaluation_time += time.time() - timer
            print(f"Done! (total target eval time: {target_evaluation_time:.2f}s)\n")

            print("Updating surrogate model with new acquisition point...")
            timer = time.time()
            fs = fs + [f]
            ys = jnp.concatenate([ys, y[None]])
            surrogate_model = surrogate_model.fit(fs, ys)
            surrogate_fit_time += time.time() - timer
            print(f"Done! (total surrogate fit time: {surrogate_fit_time:.2f}s)\n")

            print(f"Iteration {i+1}: current= {y:.8f}, best = {ys.min():.8f}\n")

    return dict(
        observation_locations=fs,
        observation_values=ys,
        surrogate_fit_time=surrogate_fit_time,
        acquisition_time=acquisition_time,
        target_evaluation_time=target_evaluation_time,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=["wycoff", "vellanky", "vien", "kundu", "shilton"]
    )
    parser.add_argument("--target_fn", choices=["mnist", "sinc", "pendulum", "ackley"])
    parser.add_argument("--lengthscale", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    # simulation parameters
    parser.add_argument("--initial_acquisitions", type=int, default=10)
    parser.add_argument("--minimum_k", type=int, default=1)
    parser.add_argument("--maximum_k", type=int, default=10)
    parser.add_argument("--acquisitions_each_k", type=int, default=10)
    parser.add_argument("--acquisition_raw_samples", type=int, default=1024)
    parser.add_argument("--acquisition_max_restarts", type=int, default=16)
    args = parser.parse_args()

    # problem setup
    target_fn = {
        "mnist": targets.MNIST,
        "sinc": targets.SincProjection,
        "pendulum": targets.Pendulum,
        "ackley": lambda: targets.Ridge(targets.virtual_library.Ackley(), d=2),
    }[args.target_fn]()
    kernel = rkhs.RKHS(
        metric=kernels.Euclidean(),
        profile=kernels.SquaredExponential(),
        rho=jnp.array([args.lengthscale] * target_fn.d),
    )

    # simulation setup
    run_simulation_fn, surrogate_model = {
        "wycoff": (run_wycoff, gp.FunctionalGaussianProcess()),
        "vellanky": (run_vellanky, gp.GaussianProcess()),
        "kundu": (run_kundu, gp.FunctionalGaussianProcess()),
        "vien": (run_vien, gp.FunctionalGaussianProcess()),
        "shilton": (run_shilton, gp.FunctionalGaussianProcess()),
    }[args.method]

    # run simulation
    results = run_simulation_fn(
        seed=args.seed,
        target_fn=target_fn,
        kernel=kernel,
        surrogate_model=surrogate_model,
        initial_acquisitions=args.initial_acquisitions,
        minimum_k=args.minimum_k,
        maximum_k=args.maximum_k,
        acquisitions_each_k=args.acquisitions_each_k,
        acquisition_raw_samples=args.acquisition_raw_samples,
        acquisition_max_restarts=args.acquisition_max_restarts,
    )

    # save results
    save_dir = f"results/{args.method}/{args.target_fn}/lengthscale_{args.lengthscale}/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/seed_{args.seed}"
    pickle.dump(results, open(f"{save_path}.pkl", "wb"))
    print(f"Results saved to {save_path}.npz")
