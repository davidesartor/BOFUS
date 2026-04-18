from typing import Protocol, Callable
from jaxtyping import Float, Array, Scalar

import jax
import jax.numpy as jnp
import scipy as sp
import numpy as np


class TestFunction(Protocol):
    d: int

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar: ...


class PinWheel(TestFunction):
    d: int = 1
    """
    2-link impedance-controlled arm interacting with a pinwheel.

    Parameters
    ----------
    robot_arm_lengths : array-like (2,)
        Lengths of the 2 robot arm links [m]. Default [0.5, 0.4].
    robot_arm_masses : array-like (2,)
        Masses of the 2 robot arm links [kg]. Default [2.0, 1.5].
    pinwheel_arm_length : float
        Length of the pinwheel arm [m]. Default 0.5.
    pinwheel_arm_mass : float
        Mass of the pinwheel arm [kg]. Default 5.0.
    pivot : array-like (2,)
        Pinwheel pivot position [m]. Default [0.6, 0.0].
    damping_ratio : float
        D = damping_ratio * K. Damping is proportional to stiffness for critical damping. Default 0.1.
    contact_penalty_stiffness : float
        Stiffness for penalty-based contact forces [N/m]. Default 1e6.
    d_contact : float
        Damping for contact forces [N/(m/s)]. Default 1000.
    contact_radius : float
        Effective radius for contact between arm and pinwheel [m]. Default 0.005.
    simulation_time : float
        Total simulation time [s]. Default 5.0.
    """

    def __init__(
        self,
        robot_arm_lengths: tuple[float, float] = (0.5, 0.4),
        robot_arm_masses: tuple[float, float] = (2.0, 1.5),
        pinwheel_arm_length: float = 0.5,
        pinwheel_arm_mass: float = 5.0,
        pivot: tuple[float, float] = (1.2, 0.0),
        damping_ratio: float = 0.1,
        contact_penalty_stiffness: float = 1e6,
        d_contact: float = 1000,
        contact_radius: float = 0.005,
        simulation_time: float = 3.0,
        target_angle: float = 0.0,
    ):
        self.L1, self.L2 = robot_arm_lengths
        self.m1, self.m2 = robot_arm_masses
        self.Lp = pinwheel_arm_length
        self.mp = pinwheel_arm_mass
        self.pivot = np.array(pivot, dtype=float)
        self.dr = damping_ratio
        self.kc = contact_penalty_stiffness
        self.dc = d_contact
        self.cr = contact_radius
        self.simulation_time = simulation_time
        self.target_angle = target_angle

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar:
        q1 = lambda t: f(jnp.array([t/self.simulation_time]))
        sol, theta_final, omega_final = self.simulate(q1_ref=q1)
        return jnp.log(2 * (1 - jnp.cos(theta_final - self.target_angle)))

    def simulate(
        self,
        K1=lambda t: 100.0,  # stiffness for joint 1
        K2=lambda t: 20.0,  # stiffness for joint 2
        q1_ref=lambda t: 0.0,  # reference trajectory for joint 1
        q2_ref=lambda t: 0.0,  # reference trajectory for joint 2
    ):
        Ip = (1 / 3) * self.mp * self.Lp**2  # pinwheel MOI about pivot

        def dynamics(t, x):
            q1, q2, thp, q1d, q2d, thpd = x

            # Arm mass matrix
            c2, s2 = np.cos(q2), np.sin(q2)
            M11 = (1 / 3) * self.m1 * self.L1**2 + self.m2 * (
                self.L1**2 + (1 / 3) * self.L2**2 + self.L1 * self.L2 * c2
            )
            M12 = self.m2 * ((1 / 3) * self.L2**2 + 0.5 * self.L1 * self.L2 * c2)
            M22 = (1 / 3) * self.m2 * self.L2**2
            M = np.array([[M11, M12], [M12, M22]])

            # Coriolis
            h = 0.5 * self.m2 * self.L1 * self.L2 * s2
            C = np.array([[-h * q2d, -h * (q1d + q2d)], [h * q1d, 0.0]])

            # Impedance torques
            K1t, K2t = K1(t), K2(t)
            tau = np.array(
                [
                    K1t * (q1_ref(t) - q1) - self.dr * K1t * q1d,
                    K2t * (q2_ref(t) - q2) - self.dr * K2t * q2d,
                ]
            )

            # Geometry
            p0 = np.zeros(2)
            p1 = self.L1 * np.array([np.cos(q1), np.sin(q1)])
            p2 = p1 + self.L2 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
            pw0 = self.pivot
            pw1 = self.pivot + self.Lp * np.array([np.cos(thp), np.sin(thp)])

            # Contact forces
            tau_arm, tau_pw = np.zeros(2), 0.0
            for link, (A, B) in enumerate([(p0, p1), (p1, p2)], start=1):
                ta, sp, d, ptA, ptP = self.seg_closest(A, B, pw0, pw1)
                if 1e-10 < d < self.cr:
                    da, dp = self.contact_gen_forces(
                        ta, sp, d, ptA, ptP, q1, q2, q1d, q2d, thp, thpd, link
                    )
                    tau_arm += da
                    tau_pw += dp

            qdd = np.linalg.solve(M, tau + tau_arm - C @ np.array([q1d, q2d]))
            thpdd = tau_pw / Ip - 3.9 * thpd

            return [q1d, q2d, thpd, qdd[0], qdd[1], thpdd]

        # Initial state: arm at reference, pinwheel at pi with zero velocity
        x0 = [q1_ref(0), q2_ref(0), np.pi, 0.0, 0.0, 0.0]

        sol = sp.integrate.solve_ivp(
            dynamics,
            [0, self.simulation_time],
            x0,
            method="Radau",
            rtol=1e-8,
            atol=1e-10,
            max_step=5e-4,
        )

        theta_final = sol.y[2, -1]
        omega_final = sol.y[5, -1]
        return sol, theta_final, omega_final

    @staticmethod
    def seg_closest(A, B, C, D):
        """Closest points between 2D segments AB and CD."""
        d1, d2, r = B - A, D - C, A - C
        a, e, f = d1 @ d1, d2 @ d2, d2 @ r

        if a < 1e-12 and e < 1e-12:
            t = s = 0.0
        elif a < 1e-12:
            t, s = 0.0, np.clip(f / e, 0, 1)
        elif e < 1e-12:
            t, s = np.clip(-(d1 @ r) / a, 0, 1), 0.0
        else:
            b, c_ = d1 @ d2, d1 @ r
            den = a * e - b * b
            t = np.clip((b * f - c_ * e) / den, 0, 1) if den > 1e-12 else 0.0
            s = (b * t + f) / e
            if s < 0:
                t, s = np.clip(-c_ / a, 0, 1), 0.0
            elif s > 1:
                t, s = np.clip((b - c_) / a, 0, 1), 1.0

        ptA, ptB = A + t * d1, C + s * d2
        return t, s, np.linalg.norm(ptA - ptB), ptA, ptB

    def contact_gen_forces(
        self, ta, sp, d, ptA, ptP, q1, q2, q1d, q2d, thp, thpd, link
    ):
        """Compute generalized contact forces on arm joints and pinwheel."""
        n = (ptA - ptP) / d
        pen = self.cr - d

        e1 = np.array([-np.sin(q1), np.cos(q1)])
        ep = np.array([-np.sin(thp), np.cos(thp)])

        if link == 1:
            v_arm = ta * self.L1 * e1 * q1d
            Ja = np.column_stack([ta * self.L1 * e1, np.zeros(2)])
        else:
            e12 = np.array([-np.sin(q1 + q2), np.cos(q1 + q2)])
            v_arm = self.L1 * e1 * q1d + ta * self.L2 * e12 * (q1d + q2d)
            Ja = np.column_stack(
                [self.L1 * e1 + ta * self.L2 * e12, ta * self.L2 * e12]
            )

        vn = (v_arm - sp * self.Lp * ep * thpd) @ n
        F = max(0.0, self.kc * pen - self.dc * vn) * n

        return Ja.T @ F, sp * self.Lp * ep @ (-F)
