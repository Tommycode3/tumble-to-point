"""Torque-free rigid body: conservation laws and the analytic axisymmetric solution."""
import numpy as np

from cubeadcs.dynamics import inertial_momentum, kinetic_energy, rk4_step
from cubeadcs.quaternion import dcm, random_quat
from cubeadcs.sim import inertia_3u

Z = np.zeros(3)


def integrate(J, w0, t_end, dt):
    Jinv = np.linalg.inv(J)
    x = np.zeros(10)
    x[0:4] = random_quat(np.random.default_rng(3))
    x[4:7] = w0
    xs = [x]
    for _ in range(int(t_end / dt)):
        x = rk4_step(x, dt, J, Jinv, Z, Z)
        xs.append(x)
    return np.array(xs)


def test_energy_momentum_and_norm_conserved():
    """Tumbling asymmetric 3U body (|w| ~ 10 deg/s) for one orbit, dt = 0.25 s."""
    J = inertia_3u()
    xs = integrate(J, np.radians([6.0, -5.0, 4.0]), 5677.0, 0.25)
    T = np.array([kinetic_energy(x, J) for x in xs])
    H = np.array([inertial_momentum(x, J, dcm(x[0:4])) for x in xs])
    qn = np.linalg.norm(xs[:, 0:4], axis=1)
    dT = np.max(np.abs(T / T[0] - 1))
    dH = np.max(np.linalg.norm(H - H[0], axis=1)) / np.linalg.norm(H[0])
    print(f"rel. energy drift {dT:.2e}, rel. |H_I| drift {dH:.2e}, max |q|-1 {np.max(np.abs(qn - 1)):.1e}")
    assert dT < 1e-7
    assert dH < 1e-7  # also checks quaternion kinematics (H is mapped to inertial)
    assert np.max(np.abs(qn - 1)) < 1e-14


def test_axisymmetric_analytic_solution():
    """J = diag(I, I, I3): w3 constant, (w1, w2) rotate at lambda = (I3 - I) w3 / I."""
    I, I3 = 0.04, 0.007
    J = np.diag([I, I, I3])
    w0 = np.array([0.05, 0.0, 0.3])
    dt, t_end = 0.1, 600.0
    xs = integrate(J, w0, t_end, dt)
    t = np.arange(xs.shape[0]) * dt
    lam = (I3 - I) / I * w0[2]
    w1 = w0[0] * np.cos(lam * t)
    w2 = w0[0] * np.sin(lam * t)
    err = np.max(np.abs(xs[:, 4] - w1)) + np.max(np.abs(xs[:, 5] - w2)) + np.max(np.abs(xs[:, 6] - w0[2]))
    print(f"max rate error vs analytic: {err:.2e} rad/s")
    assert err < 1e-6  # 2e-5 relative to |w_perp|


def test_rk4_is_fourth_order():
    """Halving the step must shrink the global error by ~2^4 = 16."""
    I, I3 = 0.04, 0.007
    J = np.diag([I, I, I3])
    w0 = np.array([0.05, 0.0, 0.3])
    lam = (I3 - I) / I * w0[2]
    errs = []
    for dt in (0.4, 0.2, 0.1):
        xs = integrate(J, w0, 400.0, dt)
        errs.append(abs(xs[-1, 4] - w0[0] * np.cos(lam * 400.0)))
    ratios = np.array(errs[:-1]) / np.array(errs[1:])
    print("RK4 error ratios on step halving:", np.round(ratios, 2))
    assert np.all((ratios > 14) & (ratios < 18))
