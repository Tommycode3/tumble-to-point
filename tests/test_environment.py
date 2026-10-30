import math

import numpy as np

from cubeadcs.environment import (TiltedDipole, gravity_gradient_torque, in_eclipse, julian_date,
                                  sun_direction)
from cubeadcs.orbit import R_EARTH, CircularOrbit, sso_inclination
from cubeadcs.sim import SimConfig


def test_dipole_magnitude_and_direction():
    d = TiltedDipole()
    b0 = np.linalg.norm(d.m_ecef)
    assert 29e-6 < b0 < 30.5e-6
    m_hat = d.m_ecef / b0
    # along the dipole axis |B| = 2 B0, B = 2 m; on the magnetic equator |B| = B0
    np.testing.assert_allclose(d.b_ecef(-R_EARTH * m_hat), 2 * b0 * m_hat, rtol=1e-12)
    eq = np.cross(m_hat, [1.0, 0, 0])
    eq = R_EARTH * eq / np.linalg.norm(eq)
    assert abs(np.linalg.norm(d.b_ecef(eq)) - b0) < 1e-15
    # north geomagnetic pole (where B points down) at ~80.6 N
    lat = math.degrees(math.asin(-m_hat[2]))
    assert 80 < lat < 81.5


def test_gravity_gradient():
    J = np.diag([0.04, 0.04, 0.007])
    r = np.array([0, 0, 6878e3])
    assert np.allclose(gravity_gradient_torque(r, J), 0)  # principal axis along r -> no torque
    r45 = 6878e3 * np.array([0, math.sqrt(0.5), math.sqrt(0.5)])
    n2 = 3.986004418e14 / 6878e3**3
    tau = gravity_gradient_torque(r45, J)
    # |tau| = 3 n^2 (Iy - Iz) sin(theta) cos(theta) for a 45 deg pitch
    assert abs(abs(tau[0]) - 1.5 * n2 * (0.04 - 0.007)) < 1e-15


def test_sun_and_eclipse():
    jd = julian_date(SimConfig().epoch)  # 2026-03-20, close to the March equinox
    s = sun_direction(jd)
    assert np.degrees(np.arccos(s[0])) < 1.0
    r = 6878e3
    assert in_eclipse(-r * s, s)
    assert not in_eclipse(r * s, s)
    assert not in_eclipse(np.array([0, 0, r]), s)


def test_orbit():
    o = CircularOrbit()
    assert abs(math.degrees(sso_inclination(500e3)) - 97.4) < 0.05
    assert abs(o.period - 5677) < 2
    r, v = o.state(1234.0)
    assert abs(np.linalg.norm(r) - o.a) < 1e-6 and abs(r @ v) < 1e-3


def test_igrf_grid_agrees_with_dipole():
    """Optional: full IGRF (ppigrf) along one orbit vs the tilted dipole."""
    import pytest

    pytest.importorskip("ppigrf")
    from cubeadcs.environment import IGRFGrid, gmst, rot3

    o = CircularOrbit()
    jd0 = julian_date(SimConfig().epoch)
    d = TiltedDipole()

    def compare(grid):
        ang, ratio = [], []
        for t in np.arange(0, o.period, 60.0):
            r, _ = o.state(t)
            R = rot3(gmst(jd0 + t / 86400))
            bd = R.T @ d.b_ecef(R @ r)
            bi = grid(t)
            ang.append(np.degrees(np.arccos(np.clip(bd @ bi / np.linalg.norm(bd) / np.linalg.norm(bi), -1, 1))))
            ratio.append(np.linalg.norm(bi) / np.linalg.norm(bd))
        return np.array(ang), np.array(ratio)

    # degree 1 only: same physics as the dipole -> validates frames/units; residual = IGRF-13 2020
    # vs IGRF-14 2026 coefficients (secular variation) plus 10 s linear interpolation
    ang, ratio = compare(IGRFGrid(o, jd0, o.period, max_degree=1))
    print(f"IGRF n=1 vs dipole: max direction diff {ang.max():.2f} deg, |B| ratio {ratio.min():.3f}-{ratio.max():.3f}")
    assert ang.max() < 1.5 and np.all(np.abs(ratio - 1) < 0.02)
    # full IGRF: the dipole approximation error we accept (documented in README)
    ang, ratio = compare(IGRFGrid(o, jd0, o.period))
    print(f"IGRF full vs dipole: direction diff median {np.median(ang):.1f} max {ang.max():.1f} deg; "
          f"|B| ratio {ratio.min():.2f}-{ratio.max():.2f}")
    assert np.median(ang) < 10 and ang.max() < 45
    assert 0.5 < ratio.min() and ratio.max() < 1.4
