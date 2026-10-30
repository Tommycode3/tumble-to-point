"""Closed-loop mission tests: B-dot detumbling and wheel pointing."""
import math

import numpy as np

from cubeadcs.sim import SimConfig, run, summarize


def test_bdot_detumbles():
    """From ~10 deg/s: quarter-orbit mean |w| decreases monotonically until it
    is below 0.5 deg/s, which must happen within 2 orbits."""
    cfg = SimConfig(t_end=2.0 * 5677.0, w0=(6.0, -5.0, 4.0), detumble_exit_hold=math.inf, log_every=5)
    res = run(cfg)
    P = res["period"]
    w = np.degrees(res["w_norm"])
    win = np.floor(res["t"] / (P / 4)).astype(int)
    means = np.array([w[win == i].mean() for i in range(win.max())])
    print("quarter-orbit mean |w| [deg/s]:", np.round(means, 3))
    above = means > 0.5
    assert np.all(np.diff(means[: np.argmin(above) + 1]) < 0)
    t_below = res["t"][np.argmax(w < 0.5)] / P
    print(f"|w| < 0.5 deg/s after {t_below:.2f} orbits; final {w[-1]:.3f} deg/s")
    assert t_below < 2.0 and w[-1] < 0.5


def test_pointing_converges_and_mekf_consistent():
    cfg = SimConfig(t_end=2.5 * 5677.0, w0=(1.5, -1.0, 1.0), seed=4)
    res = run(cfg)
    s = summarize(res)
    print(s)
    assert np.isfinite(res["t_pointing"])
    # stated requirement (last orbit, 3-axis, truth): p95 < 1 deg in sunlight, < 2 deg incl. eclipse
    assert s["point_err_p95_sunlit_deg"] < 1.0
    assert s["point_err_p95_deg"] < 2.0
    assert s["time_to_1deg_s"] < 600
    m = np.isfinite(res["nees"])
    inside = np.abs(res["est_err"][m]) <= res["sig3"][m][:, :3]
    print("closed-loop 3-sigma fraction:", inside.mean(axis=0), "mean NEES", res["nees"][m].mean())
    assert np.all(inside.mean(axis=0) >= 0.95)
    # wheels do not saturate and momentum dumping keeps them near zero
    assert s["h_w_max_mNms"] < 10.0
    assert np.max(np.abs(res["h_w"][-200:])) < 1e-3
