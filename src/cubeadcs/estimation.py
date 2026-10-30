"""Attitude determination: TRIAD and the Multiplicative EKF (MEKF).

Reference: F. L. Markley and J. L. Crassidis, *Fundamentals of Spacecraft
Attitude Determination and Control*, Springer, 2014, sec. 5.1 (TRIAD) and
sec. 6.2 (MEKF), in particular Table 6.3 (discrete propagation) and
eqs. 6.83 (Phi), 6.93 (Q_k).

Error state (6):  dx = [dtheta; dbeta]   with   q = dq(dtheta) ⊗ q_hat,
                  dq(dtheta) ≈ [dtheta/2; 1],     beta = beta_hat + dbeta.
The quaternion itself is never part of the covariance, so P stays 6x6 and
non-singular; the unit-norm constraint is handled by the multiplicative reset.
"""
from __future__ import annotations

import math

import numpy as np

from .quaternion import cross, dcm, from_dcm, propagate, qmul, qnormalize, skew

I3 = np.eye(3)


def triad(b1: np.ndarray, b2: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """TRIAD (Black 1964; M&C eq. 5.1).  b1/r1 is the 'anchor' pair (more
    accurate sensor), b2/r2 only fixes the rotation about it."""
    def frame(v1, v2):
        a = v1 / np.linalg.norm(v1)
        c = cross(v1, v2)
        c = c / np.linalg.norm(c)
        return np.column_stack((a, c, cross(a, c)))

    A = frame(b1, b2) @ frame(r1, r2).T
    return from_dcm(A)


class MEKF:
    """Gyro-driven MEKF with gyro-bias estimation (M&C Table 6.3)."""

    def __init__(self, q0: np.ndarray, beta0: np.ndarray, P0: np.ndarray,
                 sigma_v: float, sigma_u: float):
        self.q = qnormalize(np.asarray(q0, float))
        self.beta = np.asarray(beta0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.sv2 = sigma_v**2
        self.su2 = sigma_u**2

    # ---------------------------------------------------------- propagate --
    def propagate(self, w_meas: np.ndarray, dt: float) -> None:
        w = w_meas - self.beta
        wn = math.sqrt(w @ w)
        W = skew(w)
        W2 = W @ W
        if wn > 1e-9:
            s, c = math.sin(wn * dt), math.cos(wn * dt)
            Phi11 = I3 - W * (s / wn) + W2 * ((1 - c) / wn**2)
            Phi12 = W * ((1 - c) / wn**2) - I3 * dt - W2 * ((wn * dt - s) / wn**3)
        else:  # first-order limit
            Phi11 = I3 - W * dt
            Phi12 = -I3 * dt
        Phi = np.zeros((6, 6))
        Phi[:3, :3] = Phi11
        Phi[:3, 3:] = Phi12
        Phi[3:, 3:] = I3
        # G Q G^T with G = blkdiag(-I, I)  (M&C eq. 6.93)
        q11 = (self.sv2 * dt + self.su2 * dt**3 / 3.0)
        q12 = 0.5 * self.su2 * dt**2
        Qd = np.zeros((6, 6))
        Qd[:3, :3] = q11 * I3
        Qd[:3, 3:] = -q12 * I3
        Qd[3:, :3] = -q12 * I3
        Qd[3:, 3:] = self.su2 * dt * I3
        self.q = propagate(self.q, w, dt)
        self.P = Phi @ self.P @ Phi.T + Qd

    # ------------------------------------------------------------- update --
    def update_vector(self, b_meas: np.ndarray, r_ref: np.ndarray, sigma: float) -> np.ndarray:
        """Update with one unit-vector measurement b = A(q) r + v, v ~ N(0, sigma^2 I).

        H = [ [A(q_hat) r x]  0_3 ]  (M&C eq. 6.29).  Joseph-form covariance
        update for numerical robustness.  Returns the innovation.
        """
        Ar = dcm(self.q) @ r_ref
        H = np.zeros((3, 6))
        H[:, :3] = skew(Ar)
        R = sigma**2 * I3
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T  # = P H^T S^-1 (P, S symmetric)
        innov = b_meas - Ar
        dx = K @ innov
        IKH = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._reset(dx)
        return innov

    def _reset(self, dx: np.ndarray) -> None:
        """Multiplicative reset: q <- normalize([dtheta/2; 1] ⊗ q), beta += dbeta."""
        dth = dx[:3]
        self.q = qnormalize(qmul(np.array((0.5 * dth[0], 0.5 * dth[1], 0.5 * dth[2], 1.0)), self.q))
        self.beta = self.beta + dx[3:]

    # ------------------------------------------------------------ helpers --
    def sigma3(self) -> np.ndarray:
        return 3.0 * np.sqrt(np.diag(self.P))
