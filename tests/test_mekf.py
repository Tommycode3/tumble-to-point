"""MEKF statistical consistency (the estimator's verifier).

Open-loop Monte Carlo: a nadir-pointing spacecraft with a small attitude
wobble, real gyro / magnetometer / sun-sensor models and eclipse.  The filter
is started from an initial error drawn from its own P0.  We check

* per-axis attitude & bias errors inside +-3 sigma >= 95 % of the time;
* ANEES (NEES averaged over M runs) inside the two-sided 95 % chi^2 band
  for >= 90 % of time steps (Bar-Shalom, Li & Kirubarajan 2001, sec. 5.4).
"""
import math

import numpy as np
from scipy.stats import chi2

from cubeadcs.environment import TiltedDipole, gmst, in_eclipse, julian_date, rot3, sun_direction
from cubeadcs.estimation import MEKF, triad
from cubeadcs.orbit import CircularOrbit
from cubeadcs.quaternion import dcm, error_quat, from_rotvec, propagate, qmul
from cubeadcs.sensors import Gyro, Magnetometer, SunSensor
from cubeadcs.sim import SimConfig

M, T, DT = 8, 3400.0, 1.0  # runs, duration [s] (~0.6 orbit, includes eclipse entry), step


def one_run(seed):
    rng = np.random.default_rng(seed)
    orbit, dip = CircularOrbit(), TiltedDipole()
    jd0 = julian_date(SimConfig().epoch)
    gyro, mag, sun = Gyro(), Magnetometer(), SunSensor()
    gyro.reset(rng)
    P0 = np.diag([math.radians(2.0) ** 2] * 3 + [math.radians(0.1) ** 2] * 3)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    # filter initial estimate drawn from P0
    e0 = np.sqrt(np.diag(P0)) * rng.standard_normal(6)
    ekf = MEKF(qmul(from_rotvec(-e0[:3]), q), gyro.beta - e0[3:], P0, gyro.sigma_v, gyro.sigma_u)
    n = int(T / DT)
    nees = np.empty(n)
    inside = np.empty((n, 6), bool)
    phase = rng.uniform(0, 2 * math.pi, 3)
    for k in range(n):
        t = (k + 1) * DT
        tm = t - 0.5 * DT
        w = np.array([0.0, -orbit.n, 0.0]) + 2e-3 * np.sin(0.01 * tm + phase)  # midpoint ~ average
        q = propagate(q, w, DT)
        r, _ = orbit.state(t)
        Rg = rot3(gmst(jd0 + t / 86400))
        b_eci = Rg.T @ dip.b_ecef(Rg @ r)
        s_eci = sun_direction(jd0 + t / 86400)
        A = dcm(q)
        ekf.propagate(gyro.measure(w, DT, rng), DT)
        bm = mag.measure(A @ b_eci, rng)
        bn = np.linalg.norm(bm)
        ekf.update_vector(bm / bn, b_eci / np.linalg.norm(b_eci), mag.sigma / bn)
        sm = sun.measure(A @ s_eci, in_eclipse(r, s_eci), rng)
        if sm is not None:
            ekf.update_vector(sm, s_eci, sun.sigma)
        e = np.concatenate((2 * error_quat(q, ekf.q)[:3], gyro.beta - ekf.beta))
        nees[k] = e @ np.linalg.solve(ekf.P, e)
        inside[k] = np.abs(e) <= ekf.sigma3()
    return nees, inside


def test_mekf_consistency():
    runs = [one_run(s) for s in range(M)]
    nees = np.array([r[0] for r in runs])
    inside = np.concatenate([r[1] for r in runs])
    frac = inside.mean(axis=0)
    anees = nees.mean(axis=0)
    lo, hi = chi2.ppf([0.025, 0.975], 6 * M) / M
    in_band = np.mean((anees >= lo) & (anees <= hi))
    print(f"3-sigma fraction per state: {np.round(frac, 4)}; mean NEES {anees.mean():.2f} (n=6); "
          f"ANEES in 95% band [{lo:.2f}, {hi:.2f}] {100 * in_band:.1f}% of time")
    assert np.all(frac >= 0.95)
    assert 0.8 * 6 < anees.mean() < 1.2 * 6
    assert in_band >= 0.90


def test_triad_exact_without_noise():
    rng = np.random.default_rng(1)
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    r1, r2 = rng.standard_normal(3), rng.standard_normal(3)
    A = dcm(q)
    qt = triad(A @ r1, A @ r2, r1, r2)
    assert abs(abs(qt @ q) - 1) < 1e-12
