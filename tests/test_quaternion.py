"""Quaternion convention checks: product <-> DCM composition, kinematics."""
import numpy as np

from cubeadcs.quaternion import (dcm, error_angle, from_dcm, from_rotvec, propagate, qconj, qmul,
                                 random_quat, skew)

rng = np.random.default_rng(0)


def test_product_matches_dcm_composition():
    for _ in range(100):
        q, p = random_quat(rng), random_quat(rng)
        np.testing.assert_allclose(dcm(qmul(q, p)), dcm(q) @ dcm(p), atol=1e-13)


def test_dcm_is_proper_rotation_and_roundtrip():
    for _ in range(200):
        q = random_quat(rng)
        A = dcm(q)
        np.testing.assert_allclose(A @ A.T, np.eye(3), atol=1e-13)
        assert abs(np.linalg.det(A) - 1) < 1e-13
        np.testing.assert_allclose(from_dcm(A), q, atol=1e-12)


def test_kinematics_consistent_with_dcm_ode():
    """q_dot = 1/2 [w;0] ⊗ q must reproduce A_dot = -[w x] A."""
    q = random_quat(rng)
    A = dcm(q)
    w = np.array([0.3, -0.2, 0.5])
    dt = 1e-4
    for _ in range(10000):  # 1 s, explicit Euler on A with re-orthonormalisation
        A = A - skew(w) @ A * dt
        u, _, vt = np.linalg.svd(A)
        A = u @ vt
    q1 = propagate(q, w, 1.0)
    np.testing.assert_allclose(dcm(q1), A, atol=1e-4)


def test_error_angle():
    q = random_quat(rng)
    dq = from_rotvec(np.radians([0, 0, 30.0]))
    assert abs(np.degrees(error_angle(qmul(dq, q), q)) - 30.0) < 1e-10
    assert error_angle(q, -q) < 1e-7  # double cover
    np.testing.assert_allclose(qmul(q, qconj(q)), [0, 0, 0, 1], atol=1e-15)
