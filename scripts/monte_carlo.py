"""Monte Carlo dispersion analysis (sequential on purpose: low memory).

Dispersed per run: initial body rate (random direction, |w0| ~ U(2, 10) deg/s),
initial attitude (uniform), true inertia (principal moments x U(0.9, 1.1)
and a random 0-3 deg misalignment of the principal axes; GNC keeps the
nominal tensor), residual dipole (N(0, 3e-3) A m^2 per axis) and every sensor
noise / gyro-bias seed.  Usage:  python scripts/monte_carlo.py [N]
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from cubeadcs import plotting as pl
from cubeadcs.quaternion import dcm, from_rotvec
from cubeadcs.sim import SimConfig, inertia_3u, run, summarize

OUT = Path(__file__).resolve().parents[1] / "docs"
RESULTS = Path(__file__).resolve().parents[1] / "results"  # raw runs, git-ignored
KEEP = ("t", "w_norm", "point_err", "mode")


def dispersed_config(i: int, rng: np.random.Generator) -> SimConfig:
    J0 = inertia_3u()
    lam, V = np.linalg.eigh(J0)
    lam = lam * rng.uniform(0.9, 1.1, 3)
    axis = rng.standard_normal(3)
    Rm = dcm(from_rotvec(axis / np.linalg.norm(axis) * math.radians(rng.uniform(0, 3))))
    V = Rm @ V
    J = V @ np.diag(lam) @ V.T
    d = rng.standard_normal(3)
    w0 = d / np.linalg.norm(d) * rng.uniform(2.0, 10.0)
    return SimConfig(t_end=3 * 5677.0, seed=1000 + i, J_true=J, w0=tuple(w0), log_every=5,
                     residual_dipole=tuple(3e-3 * rng.standard_normal(3)))


def main(n_runs: int = 50):
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    runs, stats = [], []
    t0 = time.time()
    for i in range(n_runs):
        res = run(dispersed_config(i, rng))
        s = summarize(res)
        stats.append(s)
        runs.append({k: res[k] for k in KEEP})
        P = res["period"]
        del res
        print(f"run {i + 1:2d}/{n_runs}: detumble {s['detumble_time_orbits']:.2f} orb, "
              f"p95 err {s.get('point_err_p95_deg', np.nan):.2f} deg  ({time.time() - t0:.0f} s)", flush=True)

    def col(k):
        return np.array([s.get(k, np.nan) for s in stats], float)

    det = col("detumble_time_orbits")
    summary = {
        "n_runs": n_runs,
        "converged_runs": int(np.sum(np.isfinite(det))),
        "detumble_time_orbits": {"mean": np.nanmean(det), "p95": np.nanpercentile(det, 95), "max": np.nanmax(det)},
    }
    for k in ("point_err_p95_deg", "point_err_p95_sunlit_deg", "est_err_rms_deg", "h_w_max_mNms", "time_to_1deg_s"):
        v = col(k)
        summary[k] = {"median": np.nanmedian(v), "p95": np.nanpercentile(v, 95), "max": np.nanmax(v)}
    summary = json.loads(json.dumps(summary, default=float))
    for v in summary.values():
        if isinstance(v, dict):
            for kk in v:
                v[kk] = round(v[kk], 4)
    print(json.dumps(summary, indent=2))
    (OUT / "monte_carlo.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(RESULTS / "mc_runs.npz", P=P, **{f"{k}_{i}": r[k] for i, r in enumerate(runs) for k in KEEP})
    pl.fig_monte_carlo(runs, P, OUT / "figures" / "monte_carlo.png")


def replot():
    """Re-draw the figure from results/mc_runs.npz without re-simulating."""
    d = np.load(RESULTS / "mc_runs.npz")
    n = sum(1 for k in d.files if k.startswith("t_"))
    runs = [{k: d[f"{k}_{i}"] for k in KEEP} for i in range(n)]
    pl.fig_monte_carlo(runs, float(d["P"]), OUT / "figures" / "monte_carlo.png")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        replot()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
