"""Control laws: B-dot detumbling, quaternion-feedback PD pointing with
reaction wheels, cross-product momentum dumping, and actuator saturation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .quaternion import cross, dcm, error_quat, from_dcm


# ------------------------------------------------------------ actuators ---
@dataclass(frozen=True)
class Magnetorquers:
    m_max: float = 0.2  # A m^2 per axis (3U-class torque rods/coils)

    def saturate(self, m: np.ndarray) -> np.ndarray:
        """Scale the whole dipole vector so no axis exceeds m_max
        (preserves direction, unlike per-axis clipping)."""
        peak = np.max(np.abs(m))
        return m if peak <= self.m_max else m * (self.m_max / peak)


@dataclass(frozen=True)
class ReactionWheels:
    tau_max: float = 1.0e-3  # N m per wheel
    h_max: float = 10.0e-3  # N m s per wheel

    def saturate(self, tau_body: np.ndarray, h_w: np.ndarray) -> np.ndarray:
        """Torque limit per wheel, and no further spin-up of a saturated wheel.

        ``tau_body`` is the torque on the bus; the wheel sees -tau_body.
        """
        tau = np.clip(tau_body, -self.tau_max, self.tau_max)
        for i in range(3):
            if abs(h_w[i]) >= self.h_max and -tau[i] * h_w[i] > 0.0:
                tau[i] = 0.0
        return tau


# ------------------------------------------------------------- B-dot -----
def bdot_gain(n: float, inclination_geomag: float, J_min: float) -> float:
    """Avanzini & Giulietti (2012) near-optimal gain k = 2 n (1 + sin xi) J_min."""
    return 2.0 * n * (1.0 + math.sin(inclination_geomag)) * J_min


def bdot(b_now: np.ndarray, b_prev: np.ndarray, dt: float, k: float) -> np.ndarray:
    """m = -(k / |B|) d(b_hat)/dt, with the derivative from finite differences
    of two consecutive magnetometer samples (no gyro needed)."""
    bn = math.sqrt(b_now @ b_now)
    bp = math.sqrt(b_prev @ b_prev)
    bhat_dot = (b_now / bn - b_prev / bp) / dt
    return -(k / bn) * bhat_dot


# ------------------------------------------------------ nadir reference ---
def lvlh_reference(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nadir-pointing LVLH target attitude and its body-frame angular rate.

    z_o = -r_hat (nadir), y_o = -h_hat (minus orbit normal), x_o = y_o x z_o
    (along velocity for a circular orbit).  A_ref has the LVLH axes as rows.
    The LVLH frame rotates at n about -y_o, i.e. omega_ref = [0, -n, 0] in LVLH.
    """
    z = -r / math.sqrt(r @ r)
    h = cross(r, v)
    y = -h / math.sqrt(h @ h)
    x = cross(y, z)
    A_ref = np.vstack((x, y, z))
    n = math.sqrt(h @ h) / (r @ r)
    return from_dcm(A_ref), np.array((0.0, -n, 0.0))


# --------------------------------------------------------- pointing PD ---
@dataclass(frozen=True)
class PDGains:
    """Kp = 2 wn^2 J, Kd = 2 zeta wn J.  For small errors dq_v ~ theta/2, so
    each axis behaves as theta'' + 2 zeta wn theta' + wn^2 theta = 0."""

    wn: float = 0.08  # rad/s
    zeta: float = 0.9


def pd_pointing(q: np.ndarray, w: np.ndarray, q_ref: np.ndarray, w_ref_ref: np.ndarray,
                J: np.ndarray, h_w: np.ndarray, gains: PDGains) -> tuple[np.ndarray, float]:
    """Quaternion-feedback PD with gyroscopic compensation (Wie, 2008, sec. 7.3):

        u = -Kp sgn(dq4) dq_v - Kd (w - A(dq) w_ref) + w x (J w + h_w)

    Returns (torque on the bus [N m], principal error angle [rad]).
    """
    dq = error_quat(q, q_ref)  # dq4 >= 0 already (shortest path)
    w_ref_b = dcm(dq) @ w_ref_ref
    Kp = 2.0 * gains.wn**2 * J
    Kd = 2.0 * gains.zeta * gains.wn * J
    u = -Kp @ dq[:3] - Kd @ (w - w_ref_b) + cross(w, J @ w + h_w)
    return u, 2.0 * math.acos(min(1.0, dq[3]))


# ------------------------------------------------------ momentum dump ---
def momentum_dump(h_w: np.ndarray, b_body: np.ndarray, k: float) -> np.ndarray:
    """Cross-product law: m = (k/|B|^2) (h_w x B)  ->  m x B = -k h_w_perp.

    Only the component of h_w perpendicular to B can be removed at any
    instant; the field rotation along the orbit makes all axes reachable.
    """
    return (k / (b_body @ b_body)) * cross(h_w, b_body)
