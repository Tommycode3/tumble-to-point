"""Rigid-body attitude dynamics with reaction wheels, and an RK4 integrator.

State vector x = [q (4), omega (3), h_w (3)]:

* q      attitude quaternion (inertial -> body), scalar last
* omega  body angular rate w.r.t. inertial, body frame [rad/s]
* h_w    stored wheel angular momentum, body frame [N m s]

Equations (J includes the wheels' transverse inertia):

    J omega_dot = tau_ext + tau_w - omega x (J omega + h_w)       (Euler)
    h_w_dot     = -tau_w                                          (wheels)
    q_dot       = 1/2 [omega; 0] ⊗ q                              (kinematics)

``tau_w`` is the torque the wheels exert *on the bus*; the wheels spin up
with the opposite torque, so total angular momentum J omega + h_w changes
only through ``tau_ext``.  Wheel axes are aligned with body axes (three
orthogonal wheels; a 4-wheel pyramid only changes the allocation matrix).
"""
from __future__ import annotations

import numpy as np

from .quaternion import cross, qmul, qnormalize


def rhs(x: np.ndarray, J: np.ndarray, J_inv: np.ndarray,
        tau_ext: np.ndarray, tau_w: np.ndarray) -> np.ndarray:
    """Time derivative of the state for given (held) torques."""
    q = x[0:4]
    w = x[4:7]
    h = x[7:10]
    qdot = 0.5 * qmul(np.array((w[0], w[1], w[2], 0.0)), q)
    wdot = J_inv @ (tau_ext + tau_w - cross(w, J @ w + h))
    out = np.empty(10)
    out[0:4] = qdot
    out[4:7] = wdot
    out[7:10] = -tau_w
    return out


def rk4_step(x: np.ndarray, dt: float, J: np.ndarray, J_inv: np.ndarray,
             tau_ext: np.ndarray, tau_w: np.ndarray) -> np.ndarray:
    """Classic 4th-order Runge-Kutta step, torques zero-order held.

    The quaternion is re-normalised after the step (the RK4 update is linear
    and does not preserve the unit norm by itself; the drift is O(dt^5) per
    step, the projection removes it without biasing the attitude).
    """
    k1 = rhs(x, J, J_inv, tau_ext, tau_w)
    k2 = rhs(x + 0.5 * dt * k1, J, J_inv, tau_ext, tau_w)
    k3 = rhs(x + 0.5 * dt * k2, J, J_inv, tau_ext, tau_w)
    k4 = rhs(x + dt * k3, J, J_inv, tau_ext, tau_w)
    xn = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    xn[0:4] = qnormalize(xn[0:4])
    return xn


def kinetic_energy(x: np.ndarray, J: np.ndarray) -> float:
    """Rotational kinetic energy of the rigid bus, 1/2 omega^T J omega."""
    w = x[4:7]
    return 0.5 * float(w @ J @ w)


def inertial_momentum(x: np.ndarray, J: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Total angular momentum in inertial components, A^T (J omega + h_w)."""
    return A.T @ (J @ x[4:7] + x[7:10])
