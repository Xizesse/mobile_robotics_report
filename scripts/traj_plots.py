#!/usr/bin/env python3
"""Generate square-mission trajectory figures + arrival metrics for the report."""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = os.path.expanduser(
    "~/ros2_ws/src/lily/lily_nav/trajectory_logs/20260704_151926")
OUTDIR = os.path.expanduser("~/Documents/TeX/mobile_robotics_report/figures")

layout = json.load(open(os.path.join(RUN, "layout.json")))
metrics = json.load(open(os.path.join(RUN, "metrics.json")))
R_WP = layout["gains"]["waypoint_radius"]      # 3.0 m
R_ARR = layout["gains"]["arrival_radius"]      # 5.0 m
TOL = metrics["config"]["tolerance_band_m"]    # 1.0 m

# --- load trajectory ------------------------------------------------------
cols = {k: [] for k in ("t_s", "lap", "wp_idx", "x", "y", "e_y", "distance")}
with open(os.path.join(RUN, "trajectory.csv")) as f:
    for row in csv.DictReader(f):
        for k in cols:
            cols[k].append(float(row[k]))
d = {k: np.array(v) for k, v in cols.items()}

# mission-active mask: valid laps (1..N) with a defined cross-track error
active = (d["lap"] >= 1) & np.isfinite(d["e_y"])
t0 = d["t_s"][active][0]

wps = [(w["x"], w["y"]) for w in layout["waypoints_local_enu"]]
wx, wy = zip(*wps)

# ==========================================================================
# Figure 1: executed path vs commanded square
# ==========================================================================
fig, ax = plt.subplots(figsize=(6.0, 6.2))
# closed commanded loop WP1->WP2->WP3->WP4->WP1
loop_x = list(wx) + [wx[0]]
loop_y = list(wy) + [wy[0]]
ax.plot(loop_x, loop_y, "--", color="0.35", lw=1.6, label="Commanded path")
ax.plot(d["x"][active], d["y"][active], color="#1f77b4", lw=1.3,
        label="Executed (3 laps)")
for i, (x, y) in enumerate(wps, 1):
    ax.add_patch(plt.Circle((x, y), R_WP, color="#2ca02c",
                            fill=False, ls="-", lw=1.0, alpha=0.7))
    ax.plot(x, y, "o", color="#2ca02c", ms=6)
    ax.annotate(f"WP{i}", (x, y), textcoords="offset points",
                xytext=(6, 6), fontsize=9)
ax.plot(layout["spawn"]["x"], layout["spawn"]["y"], "ks", ms=7, label="Spawn")
ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.legend(loc="center", fontsize=9, framealpha=0.9)
fig.tight_layout(); fig.savefig(f"{OUTDIR}/traj_square_path.pdf"); plt.close(fig)

# ==========================================================================
# Figure 2: cross-track error e_y(t)
# ==========================================================================
t = d["t_s"][active] - t0
ey = d["e_y"][active]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.axhspan(-TOL, TOL, color="#2ca02c", alpha=0.12,
           label=f"$\\pm{TOL:.0f}$ m tolerance")
ax.axhline(0, color="0.7", lw=0.8)
ax.plot(t, ey, color="#1f77b4", lw=1.0)
ax.set_xlabel("time [s]"); ax.set_ylabel("$e_y$ [m]")
ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, t[-1])
fig.tight_layout(); fig.savefig(f"{OUTDIR}/traj_square_ey.pdf"); plt.close(fig)

# ==========================================================================
# Arrival error: closest approach (min distance) per leg, from the legs that
# actually cover a full waypoint transition (n_samples > 1)
# ==========================================================================
arr = []
key = d["lap"] * 10 + d["wp_idx"]
for k in np.unique(key[active]):
    seg = active & (key == k)
    if seg.sum() > 5:
        arr.append(d["distance"][seg].min())
mean_arrival = float(np.mean(arr))

o = metrics["overall"]
print("=== Square mission summary ===")
print(f"laps                 : {metrics['config']['num_laps']}")
print(f"total time           : {o['total_duration_s']:.1f} s")
print(f"path / ideal length  : {o['total_path_length']:.1f} / {o['total_ideal_length']:.0f} m")
print(f"mean |e_y|           : {o['mean_abs_e_y']:.3f} m")
print(f"RMS  e_y             : {o['rmse_e_y']:.3f} m")
print(f"max  |e_y|           : {o['max_abs_e_y']:.3f} m")
print(f"within +/-{TOL:.0f} m band  : {o['pct_within_tol']:.1f} %")
print(f"mean arrival error   : {mean_arrival:.2f} m  (R_wp = {R_WP:.0f} m)")
