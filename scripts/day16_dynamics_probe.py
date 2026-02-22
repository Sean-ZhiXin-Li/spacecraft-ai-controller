"""
Day16: Dynamics probe (mechanism-level metrics)
- Extract trajectory indicators per run:
  min_vr, min_r_err, t_flip(vr +->-), t_cross(r crosses target)
- Export a summary CSV
- Plot representative r(t), vr(t) for selected conditions

Notes:
- This script tries to auto-detect trajectory files under a run directory.
- Adjust RUNS_ROOT / file patterns if your project stores runs elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np


# Data structures

@dataclass
class RunMeta:
    run_dir: Path
    controller_variant: str
    thrust_newton: float
    r0_over_target: float
    total_reward: float
    saturation_rate_mean: float
    target_radius: Optional[float] = None


@dataclass
class RunSignals:
    r: np.ndarray
    vr: np.ndarray
    target_r: float


@dataclass
class RunIndicators:
    min_vr: float
    min_r_err: float
    t_flip: Optional[int]
    t_cross: Optional[int]


# Helpers: loading meta

def safe_read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_metrics_json(run_dir: Path) -> Optional[Path]:
    # Common patterns in your repo: metrics.json / metrics_*.json / run_manifest.json
    candidates = [
        run_dir / "metrics.json",
        run_dir / "metrics_summary.json",
        run_dir / "run_manifest.json",
        run_dir / "config_snapshot.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback: first json file containing "metrics"
    for p in run_dir.glob("*.json"):
        if "metric" in p.name.lower():
            return p
    return None


def parse_run_meta(run_dir: Path) -> Optional[RunMeta]:
    mj = find_metrics_json(run_dir)
    if mj is None:
        return None

    data = safe_read_json(mj)
    if not data:
        return None

    # Try a few likely key layouts
    def pick(d: dict, keys: List[str], default=None):
        for k in keys:
            if k in d:
                return d[k]
        return default

    # Some projects store nested dicts; try flatten-ish access
    controller = pick(data, ["controller_variant", "controller", "variant"], default="unknown")
    thrust = pick(data, ["thrust_newton", "THRUST_NEWTON"], default=np.nan)
    r0 = pick(data, ["r0_over_target", "R0_OVER_TARGET"], default=np.nan)
    total_reward = pick(data, ["total_reward", "reward_total", "episode_return"], default=np.nan)
    sat = pick(data, ["saturation_rate_mean", "saturation_mean"], default=np.nan)
    target_r = pick(data, ["target_radius", "target_r"], default=None)

    # Sometimes nested under "config" or "phys"
    if isinstance(thrust, dict) or np.isnan(thrust):
        for k in ["phys", "config", "env", "scenario"]:
            if k in data and isinstance(data[k], dict):
                thrust = data[k].get("thrust_newton", thrust)
                r0 = data[k].get("r0_over_target", r0)
                controller = data[k].get("controller_variant", controller)
                target_r = data[k].get("target_r", target_r)

    try:
        thrust = float(thrust)
    except Exception:
        thrust = float("nan")
    try:
        r0 = float(r0)
    except Exception:
        r0 = float("nan")
    try:
        total_reward = float(total_reward)
    except Exception:
        total_reward = float("nan")
    try:
        sat = float(sat)
    except Exception:
        sat = float("nan")
    if target_r is not None:
        try:
            target_r = float(target_r)
        except Exception:
            target_r = None

    return RunMeta(
        run_dir=run_dir,
        controller_variant=str(controller),
        thrust_newton=thrust,
        r0_over_target=r0,
        total_reward=total_reward,
        saturation_rate_mean=sat,
        target_radius=target_r,
    )


# Helpers: loading trajectories
def load_signals_from_run(run_dir: Path, fallback_target_r: Optional[float]) -> Optional[RunSignals]:
    """
    Auto-detect common trajectory storage patterns:
    - traj.npz containing arrays: r, vr, target_r
    - r.npy + vr.npy
    - orbit.csv with columns r, v_r, target_r (or target_r in meta)
    """
    # 1) NPZ
    for npz_name in ["traj.npz", "trajectory.npz", "rollout.npz", "orbit.npz"]:
        p = run_dir / npz_name
        if p.exists():
            z = np.load(p)
            # common keys
            r = None
            vr = None
            for rk in ["r", "radius", "radial_r"]:
                if rk in z:
                    r = z[rk]
                    break
            for vk in ["vr", "v_r", "radial_v", "v_radial"]:
                if vk in z:
                    vr = z[vk]
                    break
            target_r = None
            for tk in ["target_r", "target_radius"]:
                if tk in z:
                    target_r = float(z[tk])
                    break
            if target_r is None:
                if fallback_target_r is None:
                    # Can't proceed without target
                    return None
                target_r = float(fallback_target_r)
            if r is not None and vr is not None:
                return RunSignals(r=np.asarray(r).astype(float), vr=np.asarray(vr).astype(float), target_r=target_r)

    # 2) NPY pairs
    r_path = None
    vr_path = None
    for name in ["r.npy", "radius.npy", "traj_r.npy"]:
        if (run_dir / name).exists():
            r_path = run_dir / name
            break
    for name in ["vr.npy", "v_r.npy", "traj_vr.npy", "radial_velocity.npy"]:
        if (run_dir / name).exists():
            vr_path = run_dir / name
            break
    if r_path and vr_path:
        if fallback_target_r is None:
            return None
        r = np.load(r_path).astype(float)
        vr = np.load(vr_path).astype(float)
        return RunSignals(r=r, vr=vr, target_r=float(fallback_target_r))

    # 3) CSV (lightweight manual parse without pandas)
    for csv_name in ["orbit.csv", "trajectory.csv", "rollout.csv", "traj.csv"]:
        p = run_dir / csv_name
        if p.exists():
            import csv
            with p.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rs = []
                vrs = []
                target_r = fallback_target_r
                for row in reader:
                    # Try common column names
                    r_val = row.get("r") or row.get("radius")
                    vr_val = row.get("v_r") or row.get("vr") or row.get("radial_velocity")
                    if r_val is None or vr_val is None:
                        continue
                    rs.append(float(r_val))
                    vrs.append(float(vr_val))
                    if target_r is None:
                        tr = row.get("target_r") or row.get("target_radius")
                        if tr is not None:
                            target_r = float(tr)
                if target_r is None or len(rs) == 0:
                    return None
                return RunSignals(r=np.array(rs, dtype=float), vr=np.array(vrs, dtype=float), target_r=float(target_r))

    return None


# Indicators
def compute_indicators(sig: RunSignals) -> RunIndicators:
    r = sig.r
    vr = sig.vr
    target = sig.target_r

    r_err = r - target

    min_vr = float(np.min(vr))
    min_r_err = float(np.min(r_err))

    # t_flip: first index where vr crosses from >=0 to <0
    t_flip = None
    for i in range(1, len(vr)):
        if vr[i - 1] >= 0 and vr[i] < 0:
            t_flip = i
            break

    # t_cross: first crossing of target radius from above to below
    t_cross = None
    for i in range(1, len(r_err)):
        if r_err[i - 1] > 0 and r_err[i] <= 0:
            t_cross = i
            break

    return RunIndicators(min_vr=min_vr, min_r_err=min_r_err, t_flip=t_flip, t_cross=t_cross)


# Selection for representative plots
def pick_representative(runs: List[Tuple[RunMeta, RunIndicators]], controller: str, thrust: float, r0: float) -> Optional[RunMeta]:
    """
    Pick the run closest to the median total_reward among matches.
    """
    matches = [(m, ind) for (m, ind) in runs
               if (m.controller_variant == controller
                   and np.isfinite(m.thrust_newton)
                   and abs(m.thrust_newton - thrust) < 1e-6
                   and np.isfinite(m.r0_over_target)
                   and abs(m.r0_over_target - r0) < 1e-9)]
    if not matches:
        return None
    rewards = np.array([m.total_reward for (m, _) in matches], dtype=float)
    med = float(np.median(rewards))
    matches.sort(key=lambda x: abs(x[0].total_reward - med))
    return matches[0][0]


# Main
def discover_run_dirs(root: Path) -> List[Path]:
    # Heuristic: directories that contain some json or trajectory file
    run_dirs = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        has_json = any(p.suffix == ".json" for p in d.glob("*.json"))
        has_traj = any((d / fn).exists() for fn in ["traj.npz", "trajectory.npz", "r.npy", "vr.npy", "orbit.csv"])
        if has_json and has_traj:
            run_dirs.append(d)
    # If none found, relax: any dir with metrics.json
    if not run_dirs:
        for d in root.rglob("*"):
            if d.is_dir() and (d / "metrics.json").exists():
                run_dirs.append(d)
    return sorted(set(run_dirs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, default="analysis/runs", help="Root directory containing run folders")
    ap.add_argument("--out-csv", type=str, default="analysis/results/day16_dynamics_indicators.csv")
    ap.add_argument("--plot-dir", type=str, default="analysis/figs")
    ap.add_argument("--r0", type=float, default=1.005)
    ap.add_argument("--thrust-a", type=float, default=800.0)
    ap.add_argument("--thrust-b", type=float, default=2000.0)
    ap.add_argument("--controller-a", type=str, default="always_on")
    ap.add_argument("--controller-b", type=str, default="gated")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_csv = Path(args.out_csv)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(runs_root)
    if not run_dirs:
        raise SystemExit(f"[ERR] No run directories found under: {runs_root}")

    rows: List[Tuple[RunMeta, RunIndicators]] = []

    for rd in run_dirs:
        meta = parse_run_meta(rd)
        if meta is None:
            continue

        sig = load_signals_from_run(rd, fallback_target_r=meta.target_radius)
        if sig is None:
            # Skip silently but you can print if you want
            continue

        ind = compute_indicators(sig)
        rows.append((meta, ind))

    if not rows:
        raise SystemExit("[ERR] Found run dirs, but couldn't load any trajectories (r, vr, target_r).")

    # Write CSV
    header = [
        "run_dir",
        "controller_variant",
        "thrust_newton",
        "r0_over_target",
        "total_reward",
        "saturation_rate_mean",
        "target_radius",
        "min_vr",
        "min_r_err",
        "t_flip",
        "t_cross",
    ]
    import csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for (m, ind) in rows:
            w.writerow([
                str(m.run_dir),
                m.controller_variant,
                m.thrust_newton,
                m.r0_over_target,
                m.total_reward,
                m.saturation_rate_mean,
                m.target_radius if m.target_radius is not None else "",
                ind.min_vr,
                ind.min_r_err,
                ind.t_flip if ind.t_flip is not None else "",
                ind.t_cross if ind.t_cross is not None else "",
            ])

    print(f"[OK] wrote: {out_csv} | rows={len(rows)}")

    # Representative plots (4 combos)
    reps = []
    reps.append((args.controller_a, args.thrust_a))
    reps.append((args.controller_a, args.thrust_b))
    reps.append((args.controller_b, args.thrust_a))
    reps.append((args.controller_b, args.thrust_b))

    chosen: List[RunMeta] = []
    for (cv, th) in reps:
        rm = pick_representative(rows, controller=cv, thrust=th, r0=args.r0)
        if rm is None:
            print(f"[WARN] no representative found for controller={cv}, thrust={th}, r0={args.r0}")
        else:
            chosen.append(rm)

    # Plot r(t) and vr(t) for chosen reps
    import matplotlib.pyplot as plt

    # r(t)
    plt.figure()
    for rm in chosen:
        meta = parse_run_meta(rm.run_dir)
        sig = load_signals_from_run(rm.run_dir, fallback_target_r=meta.target_radius if meta else None)
        if sig is None:
            continue
        label = f"{rm.controller_variant}|{rm.thrust_newton:.0f}N"
        plt.plot(sig.r, label=label)
        plt.axhline(sig.target_r, linestyle="--")
    plt.title(f"Day16 r(t) | r0={args.r0}")
    plt.xlabel("step")
    plt.ylabel("radius")
    plt.legend()
    out1 = plot_dir / "day16_r_timeseries_r0_1p005.png"
    plt.savefig(out1, dpi=180, bbox_inches="tight")
    print(f"[OK] saved: {out1}")

    # v_r(t)
    plt.figure()
    for rm in chosen:
        meta = parse_run_meta(rm.run_dir)
        sig = load_signals_from_run(rm.run_dir, fallback_target_r=meta.target_radius if meta else None)
        if sig is None:
            continue
        label = f"{rm.controller_variant}|{rm.thrust_newton:.0f}N"
        plt.plot(sig.vr, label=label)
        plt.axhline(0.0, linestyle="--")
    plt.title(f"Day16 v_r(t) | r0={args.r0}")
    plt.xlabel("step")
    plt.ylabel("radial velocity")
    plt.legend()
    out2 = plot_dir / "day16_vr_timeseries_r0_1p005.png"
    plt.savefig(out2, dpi=180, bbox_inches="tight")
    print(f"[OK] saved: {out2}")


if __name__ == "__main__":
    main()