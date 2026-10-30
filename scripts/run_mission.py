"""Nominal mission: 4 orbits, detumbling from ~9 deg/s, then nadir pointing.

Writes the README figures to docs/figures/ and a summary to docs/results.json.
Usage:  python scripts/run_mission.py [--no-gif]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from cubeadcs import plotting as pl
from cubeadcs.sim import SimConfig, run, summarize

OUT = Path(__file__).resolve().parents[1] / "docs"


def animate(res, path, n_frames=90):
    """3-D attitude of the 3U bus, drawn in the LVLH frame, around the mode switch."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from cubeadcs.quaternion import dcm

    t = res["t"]
    ts = res["t_pointing"]
    idx = np.searchsorted(t, np.linspace(ts - 240, ts + 480, n_frames))
    hx, hy, hz = 0.05, 0.05, 0.17
    V = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_subplot(projection="3d")

    def draw(k):
        ax.cla()
        i = idx[k]
        R = dcm(res["q_ref"][i]) @ dcm(res["q_true"][i]).T  # body -> LVLH components
        P = V @ R.T
        ax.add_collection3d(Poly3DCollection([P[f] for f in faces], facecolor="#c9c8c2", edgecolor="#52514e",
                                             lw=0.6, alpha=0.55))
        for j, c in enumerate(pl.AXIS_COLORS):
            e = R[:, j] * 0.26
            ax.quiver(0, 0, 0, *e, color=c, lw=2, arrow_length_ratio=0.15)
        ax.quiver(0, 0, 0, 0, 0, 0.3, color=pl.INK, lw=1, ls="--", arrow_length_ratio=0.12)
        ax.text(0, 0, 0.33, "nadir", color=pl.INK, fontsize=8)
        ax.set_xlim(-0.22, 0.22); ax.set_ylim(-0.22, 0.22); ax.set_zlim(-0.22, 0.22)
        ax.invert_zaxis()  # LVLH z points down (nadir)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        mode = "DETUMBLE (B-dot)" if res["mode"][i] < 1 else "POINTING (RW + MEKF)"
        ax.set_title(f"t = {t[i] - ts:+.0f} s   {mode}\nerror {np.degrees(res['point_err'][i]):5.1f} deg",
                     fontsize=9, loc="left")

    anim = FuncAnimation(fig, draw, frames=n_frames)
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = run(SimConfig())
    print(f"simulated {res['t'][-1] / res['period']:.1f} orbits in {time.time() - t0:.1f} s wall")
    s = summarize(res)
    m = np.isfinite(res["nees"])
    inside = np.abs(np.c_[res["est_err"], res["bias_err"]][m]) <= res["sig3"][m]
    s["mekf_3sigma_fraction"] = [round(float(f), 4) for f in inside.mean(axis=0)]
    s["mekf_mean_nees"] = float(res["nees"][m].mean())
    print(json.dumps(s, indent=2))
    (OUT / "results.json").write_text(json.dumps(s, indent=2) + "\n")
    f = OUT / "figures"
    pl.fig_detumble(res, f / "detumble.png")
    pl.fig_pointing(res, f / "pointing.png")
    pl.fig_mekf(res, f / "mekf.png")
    pl.fig_wheels(res, f / "wheels.png")
    if "--no-gif" not in sys.argv:
        animate(res, f / "attitude.gif")


if __name__ == "__main__":
    main()
