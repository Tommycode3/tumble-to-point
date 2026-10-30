"""Closed-loop ADCS simulation: truth dynamics + environment + sensors +
mode logic + estimator + controllers.

Timing (per control step k, period ``dt``):

1. sample sensors at t_k (gyro returns the rate averaged over [t_{k-1}, t_k]);
2. MEKF: propagate t_{k-1} -> t_k with that gyro sample, then update with
   magnetometer and (if lit) sun vector;
3. mode logic and control laws -> magnetic dipole m and wheel torque u,
   held constant (ZOH) over [t_k, t_{k+1}];
4. integrate the truth over [t_k, t_{k+1}] with ``n_sub`` RK4 sub-steps,
   re-evaluating environment and disturbance torques at each sub-step.

Modes: DETUMBLE (B-dot) -> POINTING (MEKF + PD on wheels + momentum dump).
The switch happens when the gyro-measured rate stays below
``detumble_exit_rate`` for ``detumble_exit_hold`` seconds *and* the Sun is
visible, so TRIAD (magnetometer anchor + Sun) can initialise the MEKF.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np

from . import control as ctl
from .dynamics import rk4_step
from .environment import (AeroModel, IGRFGrid, TiltedDipole, density, gmst, gravity_gradient_torque,
                          in_eclipse, julian_date, rot3, sun_direction)
from .estimation import MEKF, triad
from .orbit import R_EARTH, CircularOrbit
from .quaternion import cross, dcm, error_quat, qconj, qmul, random_quat
from .sensors import Gyro, Magnetometer, SunSensor

DETUMBLE, POINTING = 0, 1


def inertia_3u() -> np.ndarray:
    """3U CubeSat, 4 kg, 10x10x34 cm, z = long axis, with small products of
    inertia from an asymmetric internal layout [kg m^2]."""
    return np.array(((0.0420, 0.0004, -0.0002),
                     (0.0004, 0.0405, 0.0003),
                     (-0.0002, 0.0003, 0.0072)))


@dataclass
class SimConfig:
    # timing
    t_end: float = 4 * 5677.0  # s (~4 orbits at 500 km)
    dt: float = 1.0  # control / sensor / estimator period [s]
    n_sub: int = 4  # RK4 sub-steps per control period
    log_every: int = 2  # log every N control steps
    seed: int = 1
    epoch: datetime = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    # spacecraft
    J_true: np.ndarray = field(default_factory=inertia_3u)
    J_nominal: np.ndarray = field(default_factory=inertia_3u)  # what GNC believes
    residual_dipole: tuple = (2e-3, -1.5e-3, 3e-3)  # A m^2
    aero: AeroModel = field(default_factory=AeroModel)
    # orbit / environment
    orbit: CircularOrbit = field(default_factory=CircularOrbit)
    field_model: str = "dipole"  # "dipole" or "igrf" (needs ppigrf; filter keeps dipole)
    # initial conditions
    w0: tuple = (6.0, -5.0, 4.0)  # deg/s
    q0: tuple | None = None  # None -> random attitude from seed
    # hardware
    mtq: ctl.Magnetorquers = field(default_factory=ctl.Magnetorquers)
    rw: ctl.ReactionWheels = field(default_factory=ctl.ReactionWheels)
    mag: Magnetometer = field(default_factory=Magnetometer)
    gyro: Gyro = field(default_factory=Gyro)
    sun: SunSensor = field(default_factory=SunSensor)
    # control
    pd: ctl.PDGains = field(default_factory=ctl.PDGains)
    k_dump: float = 5e-3  # 1/s
    detumble_exit_rate: float = math.radians(0.5)  # rad/s
    detumble_exit_hold: float = 120.0  # s
    # estimator initial uncertainty (1-sigma)
    P0_att: float = math.radians(2.0)
    P0_bias: float = math.radians(0.1)


def run(cfg: SimConfig) -> dict:
    """Run one closed-loop simulation and return logged arrays (dict)."""
    rng = np.random.default_rng(cfg.seed)
    orbit = cfg.orbit
    jd0 = julian_date(cfg.epoch)
    J, J_inv = cfg.J_true, np.linalg.inv(cfg.J_true)
    Jn = cfg.J_nominal
    dipole = TiltedDipole()
    igrf = IGRFGrid(orbit, jd0, cfg.t_end) if cfg.field_model == "igrf" else None
    m_res = np.asarray(cfg.residual_dipole, float)
    gyro = replace(cfg.gyro)  # fresh copy (holds the bias state)
    gyro.reset(rng)

    def b_eci_model(t):  # what the on-board model believes (always the dipole)
        r, _ = orbit.state(t)
        R = rot3(gmst(jd0 + t / 86400.0))
        return R.T @ dipole.b_ecef(R @ r)

    b_eci_truth = igrf if igrf is not None else b_eci_model

    # B-dot gain: xi = inclination w.r.t. geomagnetic equator (~ i - tilt for polar orbits)
    xi = abs(orbit.inclination - math.radians(9.7))
    k_bdot = ctl.bdot_gain(orbit.n, xi, float(np.min(np.linalg.eigvalsh(Jn))))

    q0 = np.asarray(cfg.q0, float) if cfg.q0 is not None else random_quat(rng)
    x = np.zeros(10)
    x[0:4] = q0
    x[4:7] = np.radians(cfg.w0)

    n_steps = int(round(cfg.t_end / cfg.dt))
    n_log = n_steps // cfg.log_every + 1
    L = {k: np.full((n_log,) + s, np.nan) for k, s in {
        "t": (), "mode": (), "eclipse": (), "w_true": (3,), "w_norm": (), "q_true": (4,), "h_w": (3,),
        "point_err": (), "est_err": (3,), "bias_err": (3,), "sig3": (6,), "nees": (),
        "m_cmd": (3,), "tau_w": (3,), "tau_dist": (3,), "b_norm": (), "q_est": (4,), "q_ref": (4,),
    }.items()}

    mode = DETUMBLE
    low_rate_time = 0.0
    b_prev = None
    w_avg = x[4:7].copy()
    ekf: MEKF | None = None
    t_pointing = np.nan
    m_cmd = np.zeros(3)
    u = np.zeros(3)
    h_sub = cfg.dt / cfg.n_sub
    li = 0

    for k in range(n_steps + 1):
        t = k * cfg.dt
        r, v = orbit.state(t)
        A = dcm(x[0:4])
        s_eci = sun_direction(jd0 + t / 86400.0)
        ecl = in_eclipse(r, s_eci)
        b_true_eci = b_eci_truth(t)

        # ---- 1. sensors
        b_meas = cfg.mag.measure(A @ b_true_eci, rng)
        s_meas = cfg.sun.measure(A @ s_eci, ecl, rng)
        w_meas = gyro.measure(w_avg, cfg.dt, rng)
        b_ref = b_eci_model(t)
        q_ref, w_ref = ctl.lvlh_reference(r, v)

        # ---- 2. estimation
        if ekf is not None:
            ekf.propagate(w_meas, cfg.dt)
            bn = math.sqrt(b_meas @ b_meas)
            ekf.update_vector(b_meas / bn, b_ref / math.sqrt(b_ref @ b_ref), cfg.mag.sigma / bn)
            if s_meas is not None:
                ekf.update_vector(s_meas, s_eci, cfg.sun.sigma)

        # ---- 3. mode logic + control
        if mode == DETUMBLE:
            low_rate_time = low_rate_time + cfg.dt if np.linalg.norm(w_meas) < cfg.detumble_exit_rate else 0.0
            if low_rate_time >= cfg.detumble_exit_hold and s_meas is not None:
                q_triad = triad(b_meas, s_meas, b_ref, s_eci)
                P0 = np.diag([cfg.P0_att**2] * 3 + [cfg.P0_bias**2] * 3)
                ekf = MEKF(q_triad, np.zeros(3), P0, gyro.sigma_v, gyro.sigma_u)
                mode = POINTING
                t_pointing = t
        if mode == DETUMBLE:
            m_cmd = cfg.mtq.saturate(ctl.bdot(b_meas, b_prev, cfg.dt, k_bdot)) if b_prev is not None else np.zeros(3)
            u = np.zeros(3)
        else:
            w_hat = w_meas - ekf.beta
            u_cmd, _ = ctl.pd_pointing(ekf.q, w_hat, q_ref, w_ref, Jn, x[7:10], cfg.pd)
            u = cfg.rw.saturate(u_cmd, x[7:10])
            m_cmd = cfg.mtq.saturate(ctl.momentum_dump(x[7:10], b_meas, cfg.k_dump))
        b_prev = b_meas

        # ---- logging (truth vs estimate)
        if k % cfg.log_every == 0:
            L["t"][li] = t
            L["mode"][li] = mode
            L["eclipse"][li] = ecl
            L["w_true"][li] = x[4:7]
            L["w_norm"][li] = np.linalg.norm(x[4:7])
            L["q_true"][li] = x[0:4]
            L["q_ref"][li] = q_ref
            L["h_w"][li] = x[7:10]
            L["point_err"][li] = 2 * math.acos(min(1.0, abs(qmul(x[0:4], qconj(q_ref))[3])))
            L["m_cmd"][li] = m_cmd
            L["tau_w"][li] = u
            L["b_norm"][li] = np.linalg.norm(b_true_eci)
            if ekf is not None:
                dth = 2.0 * error_quat(x[0:4], ekf.q)[:3]
                dbeta = gyro.beta - ekf.beta
                e = np.concatenate((dth, dbeta))
                L["est_err"][li] = dth
                L["bias_err"][li] = dbeta
                L["sig3"][li] = ekf.sigma3()
                L["nees"][li] = e @ np.linalg.solve(ekf.P, e)
                L["q_est"][li] = ekf.q
            li += 1
        if k == n_steps:
            break

        # ---- 4. truth propagation over [t, t+dt]
        w_acc = 0.5 * x[4:7]
        for j in range(cfg.n_sub):
            ts = t + j * h_sub
            rs, vs = orbit.state(ts)
            As = dcm(x[0:4])
            b_body = As @ (b_true_eci if j == 0 else b_eci_truth(ts))
            tau_gg = gravity_gradient_torque(As @ rs, J)
            tau_aero = cfg.aero.torque(As @ vs, density(math.sqrt(rs @ rs) - R_EARTH))
            tau_mag = cross(m_cmd + m_res, b_body)
            tau_ext = tau_gg + tau_aero + tau_mag
            x = rk4_step(x, h_sub, J, J_inv, tau_ext, u)
            w_acc += x[4:7] if j < cfg.n_sub - 1 else 0.5 * x[4:7]
        w_avg = w_acc / cfg.n_sub  # trapezoid average of the true rate
        if k % cfg.log_every == 0:
            L["tau_dist"][li - 1] = tau_gg + tau_aero + cross(m_res, b_body)

    out = {k: a[:li] for k, a in L.items()}
    out["t_pointing"] = t_pointing
    out["period"] = orbit.period
    out["config"] = cfg
    return out


def summarize(res: dict) -> dict:
    """Scalar figures of merit from a run."""
    t, P = res["t"], res["period"]
    tp = res["t_pointing"]
    s = {"detumble_time_orbits": tp / P if np.isfinite(tp) else np.nan}
    if np.isfinite(tp):
        last = t >= t[-1] - P  # last orbit
        pe = np.degrees(res["point_err"][last])
        lit = last & (res["eclipse"] == 0)
        ee = np.degrees(np.linalg.norm(res["est_err"][last], axis=1))
        s.update(
            point_err_mean_deg=float(np.mean(pe)),
            point_err_p95_deg=float(np.percentile(pe, 95)),
            point_err_max_deg=float(np.max(pe)),
            point_err_p95_sunlit_deg=float(np.percentile(np.degrees(res["point_err"][lit]), 95)),
            est_err_rms_deg=float(np.sqrt(np.mean(ee**2))),
            h_w_max_mNms=float(1e3 * np.nanmax(np.abs(res["h_w"]))),
        )
        conv = np.degrees(res["point_err"]) < 1.0
        after = t >= tp
        idx = np.where(after & conv)[0]
        s["time_to_1deg_s"] = float(t[idx[0]] - tp) if idx.size else np.nan
    return s
