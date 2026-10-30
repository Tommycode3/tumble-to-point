"""Sensor models: three-axis magnetometer, rate-integrating gyro, sun sensor."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Magnetometer:
    """b_meas = A B + b_bias + v,  v ~ N(0, sigma^2 I).

    Default noise 100 nT/axis (typical of CubeSat AMR/fluxgate units after
    calibration).  ``bias`` is a constant hard-iron residual; it is zero by
    default because the MEKF in this repo assumes a calibrated sensor (it does
    not estimate magnetometer bias).
    """

    sigma: float = 100e-9  # T
    bias: tuple = (0.0, 0.0, 0.0)  # T

    def measure(self, b_body: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return b_body + np.asarray(self.bias) + self.sigma * rng.standard_normal(3)


@dataclass
class Gyro:
    """Farrenkopf gyro model (M&C sec. 4.7.1), discrete form (M&C eq. 6.90):

        beta_{k+1} = beta_k + sigma_u sqrt(dt) N_u
        w_meas     = w_avg + 1/2 (beta_{k+1} + beta_k)
                     + sqrt(sigma_v^2/dt + sigma_u^2 dt/12) N_v

    ``w_avg`` is the true rate averaged over the sample interval (delta-angle
    output, which is what the MEKF propagation assumes).

    sigma_v  angle random walk     [rad/s^0.5]   (8.7e-5 = 0.30 deg/sqrt(h))
    sigma_u  rate random walk      [rad/s^1.5]
    """

    sigma_v: float = 8.7e-5
    sigma_u: float = 1.0e-6
    bias0_sigma: float = math.radians(0.05)  # initial turn-on bias, per axis [rad/s]

    def __post_init__(self):
        self.beta = np.zeros(3)

    def reset(self, rng: np.random.Generator) -> None:
        self.beta = self.bias0_sigma * rng.standard_normal(3)

    def measure(self, w_avg: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        beta_old = self.beta
        self.beta = beta_old + self.sigma_u * math.sqrt(dt) * rng.standard_normal(3)
        sig = math.sqrt(self.sigma_v**2 / dt + self.sigma_u**2 * dt / 12.0)
        return w_avg + 0.5 * (self.beta + beta_old) + sig * rng.standard_normal(3)


@dataclass
class SunSensor:
    """Unit Sun vector in body frame with ~sigma (rad, per axis) noise.

    Models a set of coarse/fine sun sensors with full-sphere coverage (e.g.
    six photodiodes + one fine sensor); returns None in eclipse.  Albedo is
    not modelled.
    """

    sigma: float = math.radians(0.5)

    def measure(self, s_body: np.ndarray, eclipse: bool, rng: np.random.Generator):
        if eclipse:
            return None
        return unit(s_body + self.sigma * rng.standard_normal(3))


def unit(v: np.ndarray) -> np.ndarray:
    return v / math.sqrt(v @ v)

