#!/usr/bin/env python3
"""Generate PID step-response figures + metrics for the Lily report."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGDIR = os.path.expanduser("~/ros2_ws/src/lily/lily_control/pid_step_logs/20260704_143654")
OUTDIR = os.path.expanduser("~/Documents/TeX/mobile_robotics_report/figures")

def despike(x, win=5, k=8.0):
    """Remove isolated odometry outliers via a rolling-median filter:
    replace any sample deviating from its local median by > k*MAD."""
    x = x.copy()
    n = len(x)
    for i in range(n):
        lo, hi = max(0, i - win), min(n, i + win + 1)
        seg = x[lo:hi]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med)) + 1e-9
        if abs(x[i] - med) > k * mad:
            x[i] = med
    return x

def load(name):
    t, vm, wm, vc, wc = [], [], [], [], []
    with open(os.path.join(LOGDIR, name)) as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row["t_s"]));   vm.append(float(row["v_meas"]))
            wm.append(float(row["w_meas"])); vc.append(float(row["v_cmd"]))
            wc.append(float(row["w_cmd"]))
    return (np.array(t), despike(np.array(vm)), despike(np.array(wm)),
            np.array(vc), np.array(wc))

def metrics(t, y, cmd):
    """Rise 10-90%, overshoot %, 2% settling time, steady-state error.
    The command is a pulse (0 -> target -> 0); analyse the step-up plateau."""
    target = cmd[np.argmax(np.abs(cmd))]      # plateau value
    if abs(target) < 1e-6:
        return None
    on = np.where(np.abs(cmd) > 0.5 * abs(target))[0]
    step_idx, off_idx = on[0], on[-1]         # plateau start / end
    t0 = t[step_idx]
    tt, yy = t[step_idx:off_idx + 1], y[step_idx:off_idx + 1]
    s = np.sign(target)
    yy_s, tgt = yy * s, target * s          # work in positive convention
    # steady state: mean over last 1 s
    ss = yy_s[tt >= tt[-1] - 1.0].mean()
    sse = (target - ss * s)
    # rise time 10->90 % of target
    def crossing(frac):
        thr = frac * tgt
        idx = np.argmax(yy_s >= thr)
        return tt[idx] if yy_s[idx] >= thr else np.nan
    tr = crossing(0.9) - crossing(0.1)
    # overshoot
    peak = yy_s.max()
    os_pct = max(0.0, (peak - tgt) / tgt * 100.0)
    # 2% settling (on a 0.2 s moving average, so steady-state sensor
    # ripple does not masquerade as "not yet settled")
    w = 11
    yy_sm = np.convolve(np.pad(yy_s, w // 2, mode="edge"),
                        np.ones(w) / w, mode="valid")
    band = 0.02 * abs(tgt)
    outside = np.abs(yy_sm - tgt) > band
    settle = tt[np.max(np.where(outside)[0])] - t0 if outside.any() else 0.0
    return dict(target=target, rise=tr, overshoot=os_pct, settle=settle,
                sse=sse, ss=ss * s)

XLIM = (-1.0, 15.0)   # step-up plateau window

def plot_axis(ax, t, meas, cmd, label, unit, color):
    ax.plot(t, cmd, "--", color="0.4", lw=1.5, label=f"${label}_{{cmd}}$")
    ax.plot(t, meas, color=color, lw=1.6, label=f"${label}_{{meas}}$")
    ax.axvline(0, color="0.7", lw=0.8, ls=":")
    ax.set_xlim(*XLIM)
    ax.set_ylabel(f"${label}$ [{unit}]")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)

os.makedirs(OUTDIR, exist_ok=True)
res = {}

# --- Surge step: v 0 -> 1.0 -----------------------------------------------
t, vm, wm, vc, wc = load("v1_w0.csv")
res["surge"] = metrics(t, vm, vc)
fig, ax = plt.subplots(figsize=(7.0, 3.0))
plot_axis(ax, t, vm, vc, "v", "m/s", "#1f77b4")
ax.set_xlabel("time [s]")
fig.tight_layout(); fig.savefig(f"{OUTDIR}/pid_step_surge.pdf"); plt.close(fig)

# --- Yaw step: w 0 -> 0.5 --------------------------------------------------
t, vm, wm, vc, wc = load("v0_w0p5.csv")
res["yaw"] = metrics(t, wm, wc)
fig, ax = plt.subplots(figsize=(7.0, 3.0))
plot_axis(ax, t, wm, wc, "w", "rad/s", "#d62728")
ax.set_xlabel("time [s]")
fig.tight_layout(); fig.savefig(f"{OUTDIR}/pid_step_yaw.pdf"); plt.close(fig)

# --- Coupled step: v 0.5 & w 0.5 ------------------------------------------
t, vm, wm, vc, wc = load("v0p5_w0p5.csv")
res["coupled_v"] = metrics(t, vm, vc)
res["coupled_w"] = metrics(t, wm, wc)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True)
plot_axis(a1, t, vm, vc, "v", "m/s", "#1f77b4")
plot_axis(a2, t, wm, wc, "w", "rad/s", "#d62728")
a2.set_xlabel("time [s]")
fig.tight_layout(); fig.savefig(f"{OUTDIR}/pid_step_coupled.pdf"); plt.close(fig)

print(f"{'case':12} {'target':>8} {'rise[s]':>8} {'OS[%]':>7} {'settle[s]':>10} {'sse':>10}")
for k, m in res.items():
    if m:
        print(f"{k:12} {m['target']:8.3f} {m['rise']:8.2f} {m['overshoot']:7.1f} "
              f"{m['settle']:10.2f} {m['sse']:10.4f}")
