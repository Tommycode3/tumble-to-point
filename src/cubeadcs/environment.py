"""Space environment: Earth rotation, geomagnetic field, Sun, eclipse,
atmosphere and disturbance torques.

Every model here is a documented approximation chosen to be *good enough for
ADCS design trades* in LEO, not for precise orbit determination.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from .orbit import MU_EARTH, R_EARTH
from .quaternion import cross

OMEGA_EARTH = 7.2921150e-5  # rad/s, sidereal rotation rate


# ---------------------------------------------------------------- time ----
def julian_date(dt: datetime) -> float:
    """Julian date (UTC treated as UT1; the < 1 s difference is irrelevant here)."""
    dt = dt.astimezone(timezone.utc)
    j2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
    return 2451545.0 + (dt - j2000).total_seconds() / 86400.0


def gmst(jd: float) -> float:
    """Greenwich mean sidereal time [rad], linear IAU-82 approximation."""
    deg = 280.46061837 + 360.98564736629 * (jd - 2451545.0)
    return math.radians(deg % 360.0)


def rot3(theta: float) -> np.ndarray:
    """Frame rotation about z: components in rotated frame = rot3 @ components."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array(((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)))


# ------------------------------------------------------------ magnetics ---
# IGRF-13, epoch 2020.0, degree-1 Gauss coefficients [nT] (Alken et al., 2021).
G10, G11, H11 = -29404.8, -1450.9, 4652.5


@dataclass(frozen=True)
class TiltedDipole:
    """Centred, tilted geomagnetic dipole built from the n = 1 IGRF terms.

    In ECEF:  B(r) = (R_E/|r|)^3 [ 3 (m . r_hat) r_hat - m ],
    with m = [g11, h11, g10] (nT).  Dipole tilt is ~9.7 deg; the north
    geomagnetic pole is at ~80.6 N, 72.7 W.  The n = 1 terms carry ~90 % of
    the field energy at LEO; residual errors vs. full IGRF are typically
    5-15 % in magnitude and a few degrees in direction (South Atlantic
    Anomaly is the worst region).
    """

    m_ecef: np.ndarray = field(default_factory=lambda: np.array((G11, H11, G10)) * 1e-9)

    def b_ecef(self, r_ecef: np.ndarray) -> np.ndarray:
        rn = math.sqrt(r_ecef @ r_ecef)
        rh = r_ecef / rn
        return (R_EARTH / rn) ** 3 * (3.0 * (self.m_ecef @ rh) * rh - self.m_ecef)


class IGRFGrid:
    """Optional full IGRF field via the ``ppigrf`` package (pip install ppigrf).

    ``ppigrf`` re-reads its coefficient file at every call, so we evaluate the
    field once, vectorised, on a coarse time grid along the (deterministic)
    orbit and linearly interpolate in ECI.  10 s spacing ≈ 76 km along-track.
    """

    def __init__(self, orbit, jd0: float, t_end: float, step: float = 10.0, max_degree: int = 13):
        import ppigrf  # optional dependency

        self.t = np.arange(0.0, t_end + 2 * step, step)
        r_eci = np.array([orbit.state(t)[0] for t in self.t])
        th = np.array([gmst(jd0 + t / 86400.0) for t in self.t])
        r_ecef = np.einsum("kij,kj->ki", np.array([rot3(a) for a in th]), r_eci)
        rn = np.linalg.norm(r_ecef, axis=1)
        colat = np.degrees(np.arccos(r_ecef[:, 2] / rn))
        lon = np.degrees(np.arctan2(r_ecef[:, 1], r_ecef[:, 0]))
        date = datetime(2000, 1, 1, 12) + timedelta(days=jd0 - 2451545.0)
        Br, Bt, Bp = ppigrf.igrf_gc(rn / 1e3, colat, lon, date, max_degree=max_degree)
        Br, Bt, Bp = (np.asarray(x).reshape(-1) * 1e-9 for x in (Br, Bt, Bp))
        th_r, ph_r = np.radians(colat), np.radians(lon)
        # spherical (r, theta, phi) unit vectors in ECEF
        e_r = np.stack((np.sin(th_r) * np.cos(ph_r), np.sin(th_r) * np.sin(ph_r), np.cos(th_r)), 1)
        e_t = np.stack((np.cos(th_r) * np.cos(ph_r), np.cos(th_r) * np.sin(ph_r), -np.sin(th_r)), 1)
        e_p = np.stack((-np.sin(ph_r), np.cos(ph_r), np.zeros_like(ph_r)), 1)
        b_ecef = Br[:, None] * e_r + Bt[:, None] * e_t + Bp[:, None] * e_p
        self.b_eci = np.einsum("kji,kj->ki", np.array([rot3(a) for a in th]), b_ecef)

    def __call__(self, t: float) -> np.ndarray:
        k = min(int(t // (self.t[1] - self.t[0])), len(self.t) - 2)
        a = (t - self.t[k]) / (self.t[k + 1] - self.t[k])
        return (1 - a) * self.b_eci[k] + a * self.b_eci[k + 1]


# ------------------------------------------------------------------ Sun ---
def sun_direction(jd: float) -> np.ndarray:
    """Unit Earth->Sun vector in ECI (Vallado, Alg. 29; ~0.01 deg accuracy).

    The ~4.7e-5 rad parallax between Earth centre and a LEO spacecraft is
    ignored.
    """
    T = (jd - 2451545.0) / 36525.0
    lam_m = math.radians(280.460 + 36000.771 * T)
    M = math.radians(357.5291092 + 35999.05034 * T)
    lam = lam_m + math.radians(1.914666471 * math.sin(M) + 0.019994643 * math.sin(2 * M))
    eps = math.radians(23.439291 - 0.0130042 * T)
    return np.array((math.cos(lam), math.cos(eps) * math.sin(lam), math.sin(eps) * math.sin(lam)))


def in_eclipse(r_eci: np.ndarray, s_hat: np.ndarray) -> bool:
    """Cylindrical Earth-shadow model (no penumbra): behind Earth and within
    one Earth radius of the Earth-Sun line."""
    proj = r_eci @ s_hat
    if proj >= 0.0:
        return False
    perp = r_eci - proj * s_hat
    return bool(perp @ perp < R_EARTH * R_EARTH)


# ----------------------------------------------------------- atmosphere ---
def density(altitude: float) -> float:
    """Exponential atmosphere, 500-600 km band of Vallado Table 8-4
    (rho0 = 6.967e-13 kg/m^3 at 500 km, H = 63.822 km).  Solar-activity
    variation (x10 over a cycle) is NOT modelled."""
    return 6.967e-13 * math.exp(-(altitude - 500e3) / 63822.0)


# --------------------------------------------------------- disturbances ---
def gravity_gradient_torque(r_body: np.ndarray, J: np.ndarray) -> np.ndarray:
    """tau_gg = 3 mu / |r|^5  r_b x (J r_b)   (M&C eq. 3.157)."""
    rn2 = r_body @ r_body
    return (3.0 * MU_EARTH / (rn2 * rn2 * math.sqrt(rn2))) * cross(r_body, J @ r_body)


@dataclass(frozen=True)
class AeroModel:
    """Flat-plate drag on a box, force applied at a fixed centre-of-pressure
    offset from the centre of mass.

    F = -1/2 rho Cd A_proj |v|^2 v_hat,  A_proj = sum_i A_i max(0, n_i . v_hat)
    tau = r_cp x F.  Atmosphere co-rotation and wind are neglected (~7 %).
    """

    face_areas: tuple = (0.034, 0.034, 0.01)  # m^2 for +-x, +-y, +-z (3U)
    cd: float = 2.2
    r_cp: tuple = (0.004, -0.003, 0.015)  # m, CoP offset from CoM

    def torque(self, v_body: np.ndarray, rho: float) -> np.ndarray:
        vn = math.sqrt(v_body @ v_body)
        vh = v_body / vn
        area = sum(a * abs(c) for a, c in zip(self.face_areas, vh))
        f = -0.5 * rho * self.cd * area * vn * vn * vh
        return cross(np.asarray(self.r_cp), f)
