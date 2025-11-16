"""
Visual diagnostics for spiral-in trajectories.

- Reads replay.npz (.npz with 'obs' and optional 'actions') and optional metrics_energy.json
- Plots: orbit (x,y), radial error over time, thrust histogram, energy curve (if available)
- CLI flags:
    --replay <path>            path to replay.npz
    --metrics <path>           optional metrics_energy.json for annotations
    --outdir <dir>             where to save PNGs (default: figures/new_week_2/run/)
    --save                     if present, save PNGs instead of only showing
    --show                     if present, show window (matplotlib blocking)
    --png-dpi <int>            figure DPI (default 150)
    --target-radius <float>    draw target circle (if provided)
    --title <str>              figure title prefix

Usage example:
    python tools/diagnostics/diag_orbit.py ^
        --replay logs/new_week_1/spiral_in/high_thrust/replay.npz ^
        --metrics logs/new_week_1/spiral_in/high_thrust/metrics_energy.json ^
        --outdir figures/new_week_2/high_thrust --save --png-dpi 180 ^
        --target-radius 7.5e12 --title "NEW_WEEK_1 high_thrust"
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tools.diagnostics.fig_helpers import new_ax, save_fig, ensure_dir

def load_replay_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Replay not found: {path}")
    data = np.load(path, allow_pickle=True)
    # obs: [T, ...] with at least [x,y,vx,vy]
    if "obs" in data.files:
        obs = data["obs"]
    else:
        obs = data[data.files[0]]
    actions = data["actions"] if "actions" in data.files else None
    obs = np.asarray(obs)
    if actions is not None:
        actions = np.asarray(actions)
    return obs, actions

def extract_xy_v(obs: np.ndarray):
    if obs.shape[-1] < 4:
        raise ValueError(f"Expected obs[...,0:4]=[x,y,vx,vy], got shape {obs.shape}")
    x = obs[:, 0]
    y = obs[:, 1]
    vx = obs[:, 2]
    vy = obs[:, 3]
    r = np.sqrt(x*x + y*y)
    v = np.sqrt(vx*vx + vy*vy)
    return x, y, vx, vy, r, v

def maybe_load_metrics(path: str):
    if path is None or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_orbit_xy(x, y, outdir, title="", target_radius=None, dpi=150):
    fig, ax = new_ax(figsize=(5.0, 5.0))
    ax.plot(x, y, linewidth=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title} — Orbit (x, y)")
    if target_radius is not None and target_radius > 0.0:
        # draw target circle
        theta = np.linspace(0, 2*np.pi, 400)
        cx = target_radius * np.cos(theta)
        cy = target_radius * np.sin(theta)
        ax.plot(cx, cy, linestyle="--", linewidth=1.0)
        ax.legend(["trajectory", "target circle"], loc="best")
    return save_fig(fig, outdir, "orbit_xy.png", dpi=dpi)

def plot_radial_error(r, target_radius, outdir, title="", dpi=150):
    fig, ax = new_ax()
    if target_radius is None or target_radius <= 0.0:
        # fall back to mean radius as pseudo-target to at least see fluctuations
        target_radius = float(np.mean(r))
    err = r - target_radius
    ax.plot(err, linewidth=1.2)
    ax.set_xlabel("t (step)")
    ax.set_ylabel("radial error (r - r*)")
    ax.set_title(f"{title} — Radial Error vs Time")
    return save_fig(fig, outdir, "radial_error.png", dpi=dpi)

def plot_thrust_hist(actions, outdir, title="", dpi=150):
    if actions is None:
        return None
    thr = np.linalg.norm(actions, axis=-1)
    fig, ax = new_ax()
    ax.hist(thr, bins=40)
    ax.set_xlabel("|thrust|")
    ax.set_ylabel("count")
    ax.set_title(f"{title} — Thrust Magnitude Histogram")
    return save_fig(fig, outdir, "thrust_hist.png", dpi=dpi)

def plot_energy_curve_if_available(metrics_json, outdir, title="", dpi=150):
    """
    Optional placeholder: if later you store per-step energy array into metrics,
    you can plot E_t here. For Week2 baseline we skip detailed E_t curve.
    """
    if metrics_json is None:
        return None
    # Show a bar of summary metrics as text-only figure (quick reference)
    keys = [
        "energy_convergence_step",
        "energy_drift_percent",
        "angular_momentum_error_final",
        "energy_oscillation_index",
        "thrust_energy_ratio",
        "energy_convergence_ratio_median",
    ]
    lines = []
    for k in keys:
        if k in metrics_json:
            lines.append(f"{k}: {metrics_json[k]}")
    if not lines:
        return None
    fig, ax = new_ax()
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left")
    ax.set_title(f"{title} — Energy Summary")
    return save_fig(fig, outdir, "energy_summary.png", dpi=dpi)

def decide_visual_pass_fail(radial_error_path, metrics_json):
    """
    Heuristic visual PASS/FAIL:
      PASS if (energy_convergence_step >=0 and energy_drift_percent in [ -0.1, 0.1 ])
      and energy_oscillation_index < 0.2
      else FAIL
    You can tighten rules later.
    """
    if metrics_json is None:
        return "UNKNOWN (no metrics)"
    ecs = metrics_json.get("energy_convergence_step", -1)
    drift = metrics_json.get("energy_drift_percent", 1e9)
    osc = metrics_json.get("energy_oscillation_index", 1.0)
    if ecs >= 0 and (-0.10 <= drift <= 0.10) and (osc < 0.20):
        return "PASS (stable, low drift, low oscillation)"
    return "FAIL (slow/unstable or drifted)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True, type=str)
    ap.add_argument("--metrics", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="figures/new_week_2/run")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--png-dpi", type=int, default=150)
    ap.add_argument("--target-radius", type=float, default=0.0)
    ap.add_argument("--title", type=str, default="")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    obs, actions = load_replay_npz(args.replay)
    x, y, vx, vy, r, v = extract_xy_v(obs)
    metrics_json = maybe_load_metrics(args.metrics)

    p1 = plot_orbit_xy(x, y, args.outdir, title=args.title, target_radius=args.target_radius, dpi=args.png_dpi)
    p2 = plot_radial_error(r, args.target_radius, args.outdir, title=args.title, dpi=args.png_dpi)
    p3 = plot_thrust_hist(actions, args.outdir, title=args.title, dpi=args.png_dpi)
    p4 = plot_energy_curve_if_available(metrics_json, args.outdir, title=args.title, dpi=args.png_dpi)

    verdict = decide_visual_pass_fail(p2, metrics_json)
    # Write a small report.txt
    rep = os.path.join(args.outdir, "visual_report.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write(f"Title: {args.title}\n")
        f.write(f"Replay: {args.replay}\n")
        f.write(f"Metrics: {args.metrics}\n")
        f.write(f"Outputs:\n- {p1}\n- {p2}\n")
        if p3: f.write(f"- {p3}\n")
        if p4: f.write(f"- {p4}\n")
        f.write(f"\nVisual Verdict: {verdict}\n")

    if args.show:
        # Show the last figure to keep it open if requested
        plt.show()

if __name__ == "__main__":
    main()
