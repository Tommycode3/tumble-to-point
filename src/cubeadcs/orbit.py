"""Keplerian circular orbit (two-body, no J2, no drag decay).

For a circular orbit the argument of latitude grows linearly, u(t) = u0 + n t,
and the ECI position is

    r(t) = R3(-RAAN) R1(-i) [a cos u, a sin u, 0]^T

with n = sqrt(mu / a^3).  This is exact for the two-body problem; J2 nodal
regression (~1 deg/day for an SSO) is negligible over the few orbits we
simulate and is deliberately left out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MU_EARTH = 3.986004418e14  # m^3/s^2 (EGM96 / WGS-84)
R_EARTH = 6378137.0  # m (WGS-84 equatorial radius)
J2 = 1.08262668e-3


def sso_inclination(altitude: float) -> float:
    """Sun-synchronous inclination [rad] for a circular orbit (J2 secular rate)."""
    a = R_EARTH + altitude
    n = math.sqrt(MU_EARTH / a**3)
    omega_sun = 2 * math.pi / (365.2421897 * 86400.0)
    cos_i = -omega_sun / (1.5 * n * J2 * (R_EARTH / a) ** 2)
    return math.acos(cos_i)


@dataclass(frozen=True)
class CircularOrbit:
    altitude: float = 500e3  # m
    inclination: float = math.radians(97.4)  # rad (SSO at 500 km)
    raan: float = 0.0  # rad
    u0: float = 0.0  # argument of latitude at t = 0 [rad]

    @property
    def a(self) -> float:
        return R_EARTH + self.altitude

    @property
    def n(self) -> float:
        """Mean motion [rad/s]."""
        return math.sqrt(MU_EARTH / self.a**3)

    @property
    def period(self) -> float:
        return 2 * math.pi / self.n

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """ECI position [m] and velocity [m/s] at time t [s]."""
        u = self.u0 + self.n * t
        cu, su = math.cos(u), math.sin(u)
        ci, si = math.cos(self.inclination), math.sin(self.inclination)
        cO, sO = math.cos(self.raan), math.sin(self.raan)
        # columns of R3(-RAAN) R1(-i): in-plane unit vectors P (node) and Q
        P = np.array((cO, sO, 0.0))
        Q = np.array((-sO * ci, cO * ci, si))
        r = self.a * (cu * P + su * Q)
        v = self.a * self.n * (-su * P + cu * Q)
        return r, v
