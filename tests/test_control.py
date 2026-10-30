import numpy as np

from cubeadcs import control as ctl
from cubeadcs.quaternion import cross, dcm, from_rotvec, qmul, random_quat

rng = np.random.default_rng(7)


def test_dump_law_removes_perpendicular_momentum():
    for _ in range(50):
        h, B = rng.standard_normal(3), rng.standard_normal(3) * 3e-5
        k = 0.01
        tau = cross(ctl.momentum_dump(h, B, k), B)
        h_perp = h - (h @ B) / (B @ B) * B
        np.testing.assert_allclose(tau, -k * h_perp, atol=1e-15)


def test_bdot_dissipates_energy():
    """For a slowly varying field, b_hat_dot ≈ -w x b_hat, so w·(m x B) <= 0."""
    w = np.array([0.1, -0.05, 0.08])
    B = np.array([2e-5, -1e-5, 3e-5])
    dt = 0.01
    B2 = dcm(from_rotvec(w * dt)) @ B  # field seen by a body rotating at w
    m = ctl.bdot(B2, B, dt, 1e-3)
    assert w @ cross(m, B2) < 0


def test_pd_torque_opposes_error():
    q_ref = random_quat(rng)
    q = qmul(from_rotvec(np.radians([5.0, 0, 0])), q_ref)
    u, err = ctl.pd_pointing(q, np.zeros(3), q_ref, np.zeros(3), np.eye(3) * 0.04, np.zeros(3), ctl.PDGains())
    assert u[0] < 0 and abs(np.degrees(err) - 5) < 1e-9


def test_wheel_saturation():
    rw = ctl.ReactionWheels(tau_max=1e-3, h_max=1e-2)
    tau = rw.saturate(np.array([5e-3, -1e-4, -1e-4]), np.array([0.0, 1e-2, -1e-2]))
    # axis 1: torque clipped; axis 2: wheel at +h_max would spin further up (h_dot = -tau > 0) -> 0;
    # axis 3: wheel at -h_max, h_dot = +1e-4 brings it back -> allowed
    np.testing.assert_allclose(tau, [1e-3, 0.0, -1e-4])


def test_lvlh_reference_points_nadir():
    r = np.array([7e6, 0, 0])
    v = np.array([0, 7.5e3, 0])
    q, w = ctl.lvlh_reference(r, v)
    A = dcm(q)
    np.testing.assert_allclose(A @ (-r / 7e6), [0, 0, 1], atol=1e-12)  # body z -> nadir
    np.testing.assert_allclose(A @ [0, 1.0, 0], [1, 0, 0], atol=1e-12)  # body x -> velocity
    assert w[1] < 0
