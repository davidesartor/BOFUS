from typing import Callable
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from src import kernels, gp, acquisition, rkhs  # targets


jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


from matplotlib.animation import FuncAnimation


def simulate_arm_pinwheel(
    K1=lambda t: 100.0,
    K2=lambda t: 20.0,
    q1_ref=lambda t: np.pi / 4 * np.cos(2 * np.pi * t / 5.0),
    q2_ref=lambda t: 0.0,
    L1=0.5,
    L2=0.4,
    m1=2.0,
    m2=1.5,
    Lp=0.5,
    mp=5.0,
    pivot=np.array([1.2, 0.0]),
    damping_ratio=0.1,
    k_contact=1e6,
    d_contact=1000,
    contact_radius=0.005,
    T=5.0,
):
    """
    Simulate a 2-link impedance-controlled arm interacting with a pinwheel.

    Parameters
    ----------
    K1, K2 : callable(t) -> float
        Joint stiffness trajectories [Nm/rad].
    q1_ref, q2_ref : callable(t) -> float
        Joint reference angle trajectories [rad].
    L1, L2 : float
        Arm link lengths [m].
    m1, m2 : float
        Arm link masses [kg].
    Lp : float
        Pinwheel rod length [m].
    mp : float
        Pinwheel rod mass [kg].
    pivot : array-like (2,)
        Pinwheel pivot position [m]. Default [0.6, 0.0].
    damping_ratio : float
        D = damping_ratio * K.
    k_contact : float
        Contact penalty stiffness [N/m].
    d_contact : float
        Contact damping [Ns/m].
    contact_radius : float
        Contact detection distance [m].
    T : float
        Simulation duration [s].

    Returns
    -------
    sol : OdeResult
        Full solution from solve_ivp.
    theta_final : float
        Final pinwheel angle [rad].
    omega_final : float
        Final pinwheel angular velocity [rad/s].
    """
    if pivot is None:
        pivot = np.array([0.9, 0.0])
    else:
        pivot = np.asarray(pivot, dtype=float)

    Ip = (1 / 3) * mp * Lp**2  # pinwheel MOI about pivot
    cr = contact_radius
    kc = k_contact
    dc = d_contact
    dr = damping_ratio

    def clamp01(v):
        return max(0.0, min(1.0, v))

    def seg_closest(A, B, C, D):
        """Closest points between 2D segments AB and CD."""
        d1 = B - A
        d2 = D - C
        r = A - C
        a = d1 @ d1
        e = d2 @ d2
        f = d2 @ r

        if a < 1e-12 and e < 1e-12:
            t, s = 0.0, 0.0
        elif a < 1e-12:
            t = 0.0
            s = clamp01(f / e)
        elif e < 1e-12:
            s = 0.0
            t = clamp01(-(d1 @ r) / a)
        else:
            b = d1 @ d2
            c_ = d1 @ r
            den = a * e - b * b
            t = clamp01((b * f - c_ * e) / den) if den > 1e-12 else 0.0
            s = (b * t + f) / e
            if s < 0:
                s = 0.0
                t = clamp01(-c_ / a)
            elif s > 1:
                s = 1.0
                t = clamp01((b - c_) / a)

        ptA = A + t * d1
        ptB = C + s * d2
        dist = np.linalg.norm(ptA - ptB)
        return t, s, dist, ptA, ptB

    def contact_gen_forces(ta, sp, d, ptA, ptP, q1, q2, q1d, q2d, thp, thpd, link):
        """Compute generalized contact forces on arm joints and pinwheel."""
        n = (ptA - ptP) / d
        pen = cr - d

        # Velocity of arm contact point
        e1 = np.array([-np.sin(q1), np.cos(q1)])
        if link == 1:
            v_arm = ta * L1 * e1 * q1d
        else:
            e12 = np.array([-np.sin(q1 + q2), np.cos(q1 + q2)])
            v_arm = L1 * e1 * q1d + ta * L2 * e12 * (q1d + q2d)

        # Velocity of pinwheel contact point
        ep = np.array([-np.sin(thp), np.cos(thp)])
        v_pw = sp * Lp * ep * thpd

        # Relative approach speed
        vn = (v_arm - v_pw) @ n

        # Penalty force (repulsive only)
        fn = max(0.0, kc * pen - dc * vn)
        F = fn * n  # force on arm

        # Jacobian of arm contact point w.r.t. [q1, q2]
        if link == 1:
            Ja = np.column_stack([ta * L1 * e1, np.zeros(2)])
        else:
            e12 = np.array([-np.sin(q1 + q2), np.cos(q1 + q2)])
            Ja = np.column_stack([L1 * e1 + ta * L2 * e12, ta * L2 * e12])

        # Jacobian of pinwheel contact point w.r.t. theta
        Jp = sp * Lp * ep

        tau_arm = Ja.T @ F
        tau_pw = Jp @ (-F)
        return tau_arm, tau_pw

    def dynamics(t, x):
        q1, q2, thp = x[0], x[1], x[2]
        q1d, q2d, thpd = x[3], x[4], x[5]

        # Arm mass matrix (uniform rigid rods)
        c2, s2 = np.cos(q2), np.sin(q2)
        M11 = (1 / 3) * m1 * L1**2 + m2 * (L1**2 + (1 / 3) * L2**2 + L1 * L2 * c2)
        M12 = m2 * ((1 / 3) * L2**2 + 0.5 * L1 * L2 * c2)
        M22 = (1 / 3) * m2 * L2**2
        M = np.array([[M11, M12], [M12, M22]])

        # Coriolis / centrifugal
        h = 0.5 * m2 * L1 * L2 * s2
        C = np.array([[-h * q2d, -h * (q1d + q2d)], [h * q1d, 0.0]])

        # Joint impedance torques (damped to zero velocity)
        K1t, K2t = K1(t), K2(t)
        D1t, D2t = dr * K1t, dr * K2t
        tau = np.array(
            [K1t * (q1_ref(t) - q1) - D1t * q1d, K2t * (q2_ref(t) - q2) - D2t * q2d]
        )

        # Contact
        tau_arm = np.zeros(2)
        tau_pw = 0.0

        p0 = np.array([0.0, 0.0])
        p1 = L1 * np.array([np.cos(q1), np.sin(q1)])
        p2 = p1 + L2 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
        pw0 = pivot
        pw1 = pivot + Lp * np.array([np.cos(thp), np.sin(thp)])

        # Link 1 vs pinwheel
        ta, sp, d, ptA, ptP = seg_closest(p0, p1, pw0, pw1)
        if d < cr and d > 1e-10:
            da, dp = contact_gen_forces(
                ta, sp, d, ptA, ptP, q1, q2, q1d, q2d, thp, thpd, 1
            )
            tau_arm += da
            tau_pw += dp

        # Link 2 vs pinwheel
        ta, sp, d, ptA, ptP = seg_closest(p1, p2, pw0, pw1)
        if d < cr and d > 1e-10:
            da, dp = contact_gen_forces(
                ta, sp, d, ptA, ptP, q1, q2, q1d, q2d, thp, thpd, 2
            )
            tau_arm += da
            tau_pw += dp

        # Equations of motion
        qd = np.array([q1d, q2d])
        qdd = np.linalg.solve(M, tau + tau_arm - C @ qd)
        thpdd = tau_pw / Ip - 3.9 * thpd  # friction at pivot

        return [q1d, q2d, thpd, qdd[0], qdd[1], thpdd]

    # Initial state: arm at reference, pinwheel at pi with zero velocity
    x0 = [q1_ref(0), q2_ref(0), np.pi, 0.0, 0.0, 0.0]

    sol = sp.integrate.solve_ivp(
        dynamics, [0, T], x0, method="Radau", rtol=1e-8, atol=1e-10, max_step=5e-4
    )

    theta_final = sol.y[2, -1]
    omega_final = sol.y[5, -1]
    return sol, theta_final, omega_final


def animate_sim(
    sol, f, target_angle, L1=0.5, L2=0.4, Lp=0.5, pivot=np.array([1.2, 0.0]), fps=30
):
    """Animate the arm-pinwheel simulation result."""
    if pivot is None:
        pivot = np.array([0.6, 0.0])
    else:
        pivot = np.asarray(pivot, dtype=float)

    # Resample to uniform time steps
    dt = 1.0 / fps
    t_uni = np.arange(0, sol.t[-1], dt)
    X_uni = np.array([np.interp(t_uni, sol.t, sol.y[i]) for i in range(6)]).T

    # Precompute all positions for axis limits
    all_x, all_y = [0.0], [0.0]
    positions = []
    for k in range(len(t_uni)):
        q1, q2, thp = X_uni[k, 0], X_uni[k, 1], X_uni[k, 2]
        p1 = L1 * np.array([np.cos(q1), np.sin(q1)])
        p2 = p1 + L2 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
        pw1 = pivot + Lp * np.array([np.cos(thp), np.sin(thp)])
        positions.append((p1, p2, pw1))
        all_x.extend([p1[0], p2[0], pivot[0], pw1[0]])
        all_y.extend([p1[1], p2[1], pivot[1], pw1[1]])

    margin = 0.15
    xl = (min(all_x) - margin, max(all_x) + margin)
    yl = (min(all_y) - margin, max(all_y) + margin)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].set_xlim(xl)
    ax[0].set_ylim(yl)
    ax[0].set_aspect("equal")
    ax[0].grid(True)
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")

    (link1_line,) = ax[0].plot([], [], "b-", linewidth=4)
    (link2_line,) = ax[0].plot([], [], "c-", linewidth=4)
    (pw_line,) = ax[0].plot([], [], "r-", linewidth=3)
    (joints,) = ax[0].plot([], [], "ko", markersize=7)
    ax[0].plot(pivot[0], pivot[1], "rs", markersize=9)
    ax[0].plot(
        [pivot[0], pivot[0] + np.cos(target_angle)],
        [pivot[1], pivot[1] + np.sin(target_angle)],
        "k:",
    )

    (stiffness,) = ax[1].plot([], [], "bo")
    ax[1].plot(t_uni, [f(t) for t in t_uni], "k:", label="Stiffness Profile")
    ax[1].set_xlim(t_uni[0], t_uni[-1])
    ax[1].set_ylim(0, max(f(t) for t in t_uni) * 1.5)
    ax[1].grid(True)
    ax[1].set_ylabel("Stiffness [Nm/rad]")
    ax[1].set_xlabel("Time [s]")

    def init():
        link1_line.set_data([], [])
        link2_line.set_data([], [])
        pw_line.set_data([], [])
        joints.set_data([], [])
        stiffness.set_data([], [])
        plt.tight_layout()
        return link1_line, link2_line, pw_line, joints

    def update(k):
        p1, p2, pw1 = positions[k]
        link1_line.set_data([0, p1[0]], [0, p1[1]])
        link2_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        pw_line.set_data([pivot[0], pw1[0]], [pivot[1], pw1[1]])
        joints.set_data([0, p1[0], p2[0]], [0, p1[1], p2[1]])
        stiffness.set_data([t_uni[k]], [f(t_uni[k])])
        plt.tight_layout()
        return link1_line, link2_line, pw_line, joints

    anim = FuncAnimation(
        fig, update, frames=len(t_uni), init_func=init, interval=1000 / fps, blit=False
    )
    plt.close(fig)
    return anim


def run(
    seed: int,
    target_fn: Callable,
    kernel: rkhs.RKHS,
    surrogate_model: rkhs.FunctionalGaussianProcess,
    # simulation parameters
    initial_acquisitions: int,
    min_basis_points: int,
    max_basis_points: int,
    acquisitions_each: int,
    acquisition_raw_samples: int,
    acquisition_max_restarts: int,
):
    # set random seed
    rng = np.random.default_rng(seed=seed)

    # initial acquisition (work with flat, normalized parameters in [0, 1])
    k = min_basis_points
    d = 1  # assuming 1-dimensional input
    sampler = sp.stats.qmc.LatinHypercube(d=k * (d + 1), rng=rng)
    ps = sampler.random(n=initial_acquisitions)

    # evaluate target function at initial points and fit surrogate model
    fs = [rkhs.RKHSFunction.from_array(kernel, p.reshape(-1, d + 1)) for p in ps]
    ys = jnp.array([target_fn(f) for f in fs])
    surrogate_model = surrogate_model.fit(fs, ys)

    # sequential acquisition loop
    for k in range(min_basis_points, max_basis_points + 1):
        sampler = sp.stats.qmc.LatinHypercube(d=k * (d + 1), rng=rng)
        for i in range(acquisitions_each):
            # optimize acquisition function
            def acquisition_loss(p: Float[Array, "k * (d+1)"]) -> Scalar:
                f = rkhs.RKHSFunction.from_array(kernel, p.reshape(-1, d + 1))
                mu, cov = surrogate_model.predict([f])
                return -acquisition.log_expected_improvement(
                    mu=mu.squeeze(),
                    sigma=cov.squeeze() ** 0.5,
                    y_best=surrogate_model.observed_ys.min(),
                )

            p, _ = acquisition.optimize_lhs_candidates(
                acquisition_loss=acquisition_loss,
                candidates=sampler.random(n=acquisition_raw_samples),
                max_restarts=acquisition_max_restarts,
            )

            # evaluate target function at the new point
            f = rkhs.RKHSFunction.from_array(kernel, p.reshape(-1, d + 1))
            y = target_fn(f)

            # fit surrogate model on the new data
            fs = fs + [f]
            ys = jnp.concatenate([ys, y[None]])
            surrogate_model = surrogate_model.fit(fs, ys)

            print(f"Iteration {i+1}: current= {y:.8f}, best = {ys.min():.8f}")

            target_fn(fs[-1], save_path=f"mh_{k}_{i}")
    return ys


if __name__ == "__main__":
    seed = 0
    initial_acquisitions = 2
    min_basis_points = 1
    max_basis_points = 5
    acquisitions_each = 5
    acquisition_raw_samples = 1024 * 16
    acquisition_max_restarts = 16

    kernel = rkhs.RKHS(
        metric=kernels.Euclidean(),
        profile=kernels.SquaredExponential(),
        rho=0.1,  # type: ignore
    )

    def target_fn(f, target_angle=-0.2 * jnp.pi, save_path=None):
        T = 10.0
        K2 = jax.jit(lambda t: jnp.clip(10 * (f(jnp.array([t / T])) + 1), 0.0, 10.0))
        sol, theta, omega = simulate_arm_pinwheel(K2=K2, T=T)
        if save_path:
            anim = animate_sim(sol, K2, target_angle)
            anim.save(f"{save_path}.gif", writer="pillow", fps=30)
        return 2 * (1 - jnp.cos(theta - target_angle))

    ys = run(
        seed=seed,
        target_fn=target_fn,
        kernel=kernel,
        surrogate_model=rkhs.FunctionalGaussianProcess(
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
