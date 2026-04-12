from typing import Callable
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
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
    d = 3  # assuming 1-dimensional input
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
            # optimize acquisition function
            def acquisition_loss(p: Float[Array, "k * (d+1)"]) -> Scalar:
                f = as_function(p)
                mu, cov = surrogate_model.predict([f])
                return -acquisition.log_expected_improvement(
                    mu=mu.squeeze(),
                    sigma=cov.squeeze() ** 0.5,
                    y_best=surrogate_model.observed_ys.min(),
                )

            p = acquisition.optimize_lhs_candidates(
                acquisition_loss=acquisition_loss,
                candidates=sampler.random(n=acquisition_raw_samples),
                max_restarts=acquisition_max_restarts,
            )

            # evaluate target function at the new point
            f = as_function(p)
            y = target_fn(f)

            # fit surrogate model on the new data
            fs = fs + [f]
            ys = jnp.concatenate([ys, y[None]])
            surrogate_model = surrogate_model.fit(fs, ys)

            print(f"Iteration {i+1}: current= {y:.8f}, best = {ys.min():.8f}")

        import gymnasium as gym
        from gymnasium.utils.save_video import save_video

        env = gym.make("Pendulum-v1", render_mode="rgb_array_list")

        for i in range(10):
            obs, _ = env.reset(seed=i)
            for _ in range(400):
                # normalize the observation to be in the range [0, 1]
                lb = env.observation_space.low
                ub = env.observation_space.high
                obs = (obs - lb) / (ub - lb)

                # magic happens here and we get an action in [-1, 1]
                action = fs[ys.argmin()](obs)

                # scale the action to the action space of the environment
                action = (action + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                lb = env.action_space.low
                ub = env.action_space.high
                action = lb + (action + 1.0) * 0.5 * (ub - lb)  # [0, 1] -> [lb, ub]

                # step the environment
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

            save_video(
                frames=env.render(),
                video_folder="gifs",
                name_prefix=f"pendulum{k}_iter{i}",
                fps=env.metadata["render_fps"],
                episode_index=0,
            )

    return ys


if __name__ == "__main__":
    seed = 0
    initial_acquisitions = 20
    min_basis_points = 5
    max_basis_points = 10
    acquisitions_each = 30
    acquisition_raw_samples = 1024 * 16
    acquisition_max_restarts = 64

    rkhs = gp.RKHS(
        metric=kernels.Euclidean(),
        profile=kernels.SquaredExponential(),
        rho=0.1,  # type: ignore
    )

    def target_fn(f, n_rollouts=50):
        import gymnasium as gym

        env = gym.make("Pendulum-v1")

        def rollout(seed):
            obs, _ = env.reset(seed=seed)
            rewards = []
            for _ in range(400):
                # normalize the observation to be in the range [0, 1]
                lb = env.observation_space.low
                ub = env.observation_space.high
                obs = (obs - lb) / (ub - lb)

                # magic happens here and we get an action in [-1, 1]
                action = f(obs)

                # scale the action to the action space of the environment
                action = (action + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                lb = env.action_space.low
                ub = env.action_space.high
                action = lb + (action + 1.0) * 0.5 * (ub - lb)  # [0, 1] -> [lb, ub]

                # step the environment
                obs, reward, terminated, truncated, _ = env.step(action)
                rewards.append(reward)
                if terminated or truncated:
                    break

            J = sum([r * (0.99**t) for t, r in enumerate(rewards)])
            return J

        returns = [rollout(seed) for seed in range(n_rollouts)]
        return -np.mean(returns)

    ys = run(
        seed=seed,
        target_fn=target_fn,
        rkhs=rkhs,
        surrogate_model=gp.FunctionalGaussianProcess(
            profile=kernels.SquaredExponential()
        ),
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
    plt.savefig("wycoff.png", dpi=300, bbox_inches="tight")
    plt.close()
