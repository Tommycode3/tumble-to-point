"""Quaternion algebra in the Markley & Crassidis (2014) convention.

Conventions (used consistently in the whole package)
----------------------------------------------------
* Scalar-last quaternion ``q = [q1, q2, q3, q4] = [q_v, q4]``.
* ``A(q)`` is the attitude matrix that maps *inertial* components into
  *body* components: ``b = A(q) r``  (M&C eq. 2.125).
* The product ``q ⊗ p`` is chosen so that ``A(q ⊗ p) = A(q) A(p)``
  (M&C eq. 2.82a / 2.85), i.e. composition reads left-to-right like matrices.
* Kinematics: ``q_dot = 1/2 [omega; 0] ⊗ q`` with ``omega`` the body rate
  w.r.t. inertial, in body components (M&C eq. 3.20).

All functions operate on plain 1-D float arrays; no classes, no magic.
"""
from __future__ import annotations

import math

import numpy as np


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3-vector cross product (much faster than np.cross for single vectors)."""
    return np.array(
        (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
    )


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix: skew(a) @ b == a x b."""
    return np.array(
        (
            (0.0, -v[2], v[1]),
            (v[2], 0.0, -v[0]),
            (-v[1], v[0], 0.0),
        )
    )


def qmul(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Quaternion product with A(q ⊗ p) = A(q) A(p)  (M&C eq. 2.82a)."""
    q1, q2, q3, q4 = q
    p1, p2, p3, p4 = p
    # vector part: p4*qv + q4*pv - qv x pv
    return np.array(
        (
            p4 * q1 + q4 * p1 - (q2 * p3 - q3 * p2),
            p4 * q2 + q4 * p2 - (q3 * p1 - q1 * p3),
            p4 * q3 + q4 * p3 - (q1 * p2 - q2 * p1),
            q4 * p4 - (q1 * p1 + q2 * p2 + q3 * p3),
        )
    )


def qconj(q: np.ndarray) -> np.ndarray:
    """Conjugate = inverse for unit quaternions."""
    return np.array((-q[0], -q[1], -q[2], q[3]))


def qnormalize(q: np.ndarray) -> np.ndarray:
    return q / math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


def dcm(q: np.ndarray) -> np.ndarray:
    """Attitude matrix A(q) (inertial -> body), M&C eq. 2.125."""
    q1, q2, q3, q4 = q
    return np.array(
        (
            (q4 * q4 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 + q3 * q4), 2 * (q1 * q3 - q2 * q4)),
            (2 * (q1 * q2 - q3 * q4), q4 * q4 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 + q1 * q4)),
            (2 * (q1 * q3 + q2 * q4), 2 * (q2 * q3 - q1 * q4), q4 * q4 - q1 * q1 - q2 * q2 + q3 * q3),
        )
    )


def from_dcm(A: np.ndarray) -> np.ndarray:
    """Quaternion from attitude matrix, Shepperd's method (M&C sec. 2.9.3).

    Returns the representative with q4 >= 0.
    """
    tr = A[0, 0] + A[1, 1] + A[2, 2]
    choice = int(np.argmax((A[0, 0], A[1, 1], A[2, 2], tr)))
    if choice == 3:
        q4 = 0.5 * math.sqrt(1.0 + tr)
        d = 4.0 * q4
        q = np.array(((A[1, 2] - A[2, 1]) / d, (A[2, 0] - A[0, 2]) / d, (A[0, 1] - A[1, 0]) / d, q4))
    elif choice == 0:
        q1 = 0.5 * math.sqrt(1.0 + 2 * A[0, 0] - tr)
        d = 4.0 * q1
        q = np.array((q1, (A[0, 1] + A[1, 0]) / d, (A[0, 2] + A[2, 0]) / d, (A[1, 2] - A[2, 1]) / d))
    elif choice == 1:
        q2 = 0.5 * math.sqrt(1.0 + 2 * A[1, 1] - tr)
        d = 4.0 * q2
        q = np.array(((A[0, 1] + A[1, 0]) / d, q2, (A[1, 2] + A[2, 1]) / d, (A[2, 0] - A[0, 2]) / d))
    else:
        q3 = 0.5 * math.sqrt(1.0 + 2 * A[2, 2] - tr)
        d = 4.0 * q3
        q = np.array(((A[0, 2] + A[2, 0]) / d, (A[1, 2] + A[2, 1]) / d, q3, (A[0, 1] - A[1, 0]) / d))
    q = qnormalize(q)
    return q if q[3] >= 0 else -q


def from_rotvec(phi: np.ndarray) -> np.ndarray:
    """Quaternion of a rotation vector phi (angle = |phi|), exact."""
    ang = math.sqrt(phi[0] ** 2 + phi[1] ** 2 + phi[2] ** 2)
    if ang < 1e-12:
        return qnormalize(np.array((0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2], 1.0)))
    s = math.sin(0.5 * ang) / ang
    return np.array((s * phi[0], s * phi[1], s * phi[2], math.cos(0.5 * ang)))


def propagate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Exact propagation for constant body rate: q+ = Θ(ω,dt) q  (M&C eq. 3.32)."""
    return qmul(from_rotvec(omega * dt), q)


def error_quat(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """δq = q ⊗ q_ref^-1, sign fixed so that δq4 >= 0 (shortest rotation)."""
    dq = qmul(q, qconj(q_ref))
    return dq if dq[3] >= 0 else -dq


def error_angle(q: np.ndarray, q_ref: np.ndarray) -> float:
    """Total rotation angle [rad] between two attitudes (principal angle)."""
    dq4 = abs(qmul(q, qconj(q_ref))[3])
    return 2.0 * math.acos(min(1.0, dq4))


def small_angle(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Rotation-vector error δθ ≈ 2 δq_v, with q = δq(δθ) ⊗ q_ref."""
    return 2.0 * error_quat(q, q_ref)[:3]


def random_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed random attitude (Shoemake)."""
    q = rng.standard_normal(4)
    q = qnormalize(q)
    return q if q[3] >= 0 else -q
