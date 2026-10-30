"""Figure helpers (matplotlib only).  Colours: validated categorical slots
1-3 (blue / orange / aqua) for body axes x / y / z; neutral grey for bounds."""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AXIS_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
INK, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 1.4,
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def shade_eclipse(ax, t_orb, eclipse):
    """Grey bands where the spacecraft is in Earth's shadow."""
    e = (np.asarray(eclipse) > 0.5).astype(int)
    idx = np.flatnonzero(np.diff(np.r_[0, e, 0]))
    for i, (a, b) in enumerate(zip(idx[::2], idx[1::2] - 1)):
        ax.axvspan(t_orb[a], t_orb[b], color=SHADE, lw=0, zorder=0, label="eclipse" if i == 0 else None)


def mark_switch(ax, t_switch_orb):
    if np.isfinite(t_switch_orb):
        ax.axvline(t_switch_orb, color=MUTED, lw=0.9, ls="--", zorder=1)


def fig_detumble(res, path):
    P = res["period"]
    t = res["t"] / P
    ts = res["t_pointing"] / P
    det = t <= ts if np.isfinite(ts) else np.ones_like(t, bool)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, height_ratios=(1.25, 1))
    for i, c in enumerate(AXIS_COLORS):
        a1.plot(t[det], np.degrees(res["w_true"][det, i]), color=c, lw=1.1, label=f"$\\omega_{'xyz'[i]}$")
    a1.set_ylabel("body rate [deg/s]")
    a1.set_title("B-dot detumbling: body angular rate (truth)")
    a1.legend(ncol=3, loc="upper right")
    a2.semilogy(t[det], np.degrees(res["w_norm"][det]), color=INK, lw=1.2, label=r"$\|\omega\|$")
    a2.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    a2.text(t[det][-1], 0.5, " exit threshold 0.5 deg/s", va="bottom", ha="right", color=MUTED, fontsize=8)
    a2.set_ylabel(r"$\|\omega\|$ [deg/s]")
    a2.set_xlabel("time [orbits]")
    for a in (a1, a2):
        shade_eclipse(a, t[det], res["eclipse"][det])
        mark_switch(a, ts)
    a2.legend(loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def fig_pointing(res, path):
    P = res["period"]
    t = res["t"] / P
    ts = res["t_pointing"] / P
    m = t >= ts
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    shade_eclipse(ax, t[m], res["eclipse"][m])
    ax.semilogy(t[m], np.degrees(res["point_err"][m]), color=AXIS_COLORS[0], lw=1.1,
                label="pointing error (truth vs LVLH)")
    ke = np.degrees(np.linalg.norm(res["est_err"][m], axis=1))
    ax.semilogy(t[m], ke, color=AXIS_COLORS[1], lw=0.9, alpha=0.9, label="knowledge error (truth vs MEKF)")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(1e-2, 200)
    ax.set_xlabel("time [orbits]")
    ax.set_ylabel("3-axis error [deg]")
    ax.set_title("Nadir pointing with reaction wheels: total attitude error", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.savefig(path)
    plt.close(fig)


def fig_mekf(res, path):
    P = res["period"]
    t = res["t"] / P
    m = np.isfinite(res["nees"])
    t = t[m]
    fig, axs = plt.subplots(3, 2, figsize=(8.6, 6.0), sharex=True)
    for i in range(3):
        for j, (err, scale, unit) in enumerate(((res["est_err"], 1.0, "deg"), (res["bias_err"], 3600.0, "deg/h"))):
            a = axs[i, j]
            shade_eclipse(a, t, res["eclipse"][m])
            s3 = np.degrees(res["sig3"][m][:, 3 * j + i]) * scale
            a.plot(t, s3, color=INK, lw=0.8, label=r"MEKF $\pm 3\sigma$" if (i, j) == (0, 0) else None)
            a.plot(t, -s3, color=INK, lw=0.8)
            a.plot(t, np.degrees(err[m][:, i]) * scale, color=AXIS_COLORS[i], lw=0.7, zorder=2)
            lim = np.percentile(s3, 98) * 1.3
            a.set_ylim(-lim, lim)
            name = r"\delta\theta" if j == 0 else r"\Delta\beta"
            a.set_ylabel(f"${name}_{'xyz'[i]}$ [{unit}]")
    axs[0, 0].set_title("MEKF attitude error")
    axs[0, 1].set_title("MEKF gyro-bias error")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.94))
    for a in axs[2]:
        a.set_xlabel("time [orbits]")
    fig.align_ylabels(axs)
    fig.savefig(path)
    plt.close(fig)


def fig_wheels(res, path):
    P = res["period"]
    t = res["t"] / P
    ts = res["t_pointing"] / P
    m = t >= ts
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    for i, c in enumerate(AXIS_COLORS):
        a1.plot(t[m], 1e3 * res["h_w"][m, i], color=c, lw=1.1, label=f"$h_{{w,{'xyz'[i]}}}$")
        a2.plot(t[m], res["m_cmd"][m, i], color=c, lw=0.8, label=f"$m_{'xyz'[i]}$")
    a1.set_ylabel("wheel momentum [mN m s]")
    a1.set_title("Reaction-wheel momentum and magnetorquer dumping")
    a1.legend(ncol=3, loc="upper right")
    a2.set_ylabel(r"dipole [A m$^2$]")
    a2.set_xlabel("time [orbits]")
    a2.legend(ncol=3, loc="upper right")
    for a in (a1, a2):
        shade_eclipse(a, t[m], res["eclipse"][m])
    fig.savefig(path)
    plt.close(fig)


def fig_monte_carlo(runs, P, path):
    """Percentile envelopes over Monte Carlo runs (common time grid)."""
    t = runs[0]["t"] / P
    modes = np.array([r["mode"] for r in runs])
    W = np.degrees(np.array([r["w_norm"] for r in runs]))
    W[modes > 0] = np.nan  # detumbling phase of each run only
    # pointing error re-aligned on each run's own mode switch (t = 0 at TRIAD/MEKF init)
    E = np.full_like(W, np.nan)
    for i, r in enumerate(runs):
        k0 = int(np.argmax(modes[i] > 0))
        if modes[i, k0] > 0:
            e = np.degrees(r["point_err"][k0:])
            E[i, : e.size] = e
    last = int(np.min(np.sum(np.isfinite(E), axis=1)))  # window covered by every run
    tE, E = t[:last], E[:, :last]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for a, tt, Y, title, ylab, xl in (
            (a1, t, W, "Detumbling (each run until its switch)", r"$\|\omega\|$ [deg/s]", "time [orbits]"),
            (a2, tE, E, "Pointing after the switch", "3-axis error [deg]",
             "time since MEKF init / pointing start [orbits]")):
        with warnings.catch_warnings():  # all-NaN columns before any run points
            warnings.simplefilter("ignore", RuntimeWarning)
            p5, p50, p95 = np.nanpercentile(Y, [5, 50, 95], axis=0)
            pmax = np.nanmax(Y, axis=0)
        a.fill_between(tt, p5, p95, color=AXIS_COLORS[0], alpha=0.18, lw=0, label="5-95 %")
        a.plot(tt, pmax, color=MUTED, lw=0.7, ls="--", label="max")
        a.plot(tt, p50, color=AXIS_COLORS[0], lw=1.3, label="median")
        a.set_yscale("log")
        a.set_title(title)
        a.set_ylabel(ylab)
        a.set_xlabel(xl)
    a1.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    a2.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    a2.set_ylim(1e-2, 200)
    a1.legend(loc="upper right")
    fig.suptitle(f"Monte Carlo, N = {len(runs)} runs", x=0.01, ha="left", fontsize=9, color=MUTED)
    fig.savefig(path)
    plt.close(fig)
