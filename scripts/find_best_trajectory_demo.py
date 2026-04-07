from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any, default: float = float("inf")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def collect_run_candidates(
    runs_dir: Path,
    controller_variant: Optional[str] = None,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    if not runs_dir.exists():
        return candidates

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        metrics_path = run_dir / "metrics.json"
        traj_path = run_dir / "traj.npz"

        if not metrics_path.exists() or not traj_path.exists():
            continue

        try:
            metrics = load_json(metrics_path)
        except Exception:
            continue

        run_variant = metrics.get("controller_variant")
        if controller_variant is not None and run_variant != controller_variant:
            continue

        total_reward = safe_float(metrics.get("total_reward"), default=-1e18)
        final_radius_error = abs(safe_float(metrics.get("final_radius_error"), default=1e18))
        avg_radius_error = abs(safe_float(metrics.get("avg_radius_error"), default=1e18))
        saturation_rate_mean = safe_float(metrics.get("saturation_rate_mean"), default=1e18)

        candidate = {
            "run_dir": run_dir,
            "metrics_path": metrics_path,
            "traj_path": traj_path,
            "metrics": metrics,
            "score_tuple": (
                total_reward,
                -final_radius_error,
                -avg_radius_error,
                -saturation_rate_mean,
            ),
        }
        candidates.append(candidate)

    return candidates


def find_best_run(
    runs_dir: Path,
    controller_variant: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    candidates = collect_run_candidates(
        runs_dir,
        controller_variant=controller_variant,
    )
    if not candidates:
        return None

    candidates.sort(key=lambda x: x["score_tuple"], reverse=True)
    return candidates[0]


def find_best_checkpoint(ppo_dir: Path) -> Optional[Path]:
    if not ppo_dir.exists():
        return None

    preferred_names = [
        "ppo_best_model.pth",
        "ppo_best.pth",
        "best_model.pth",
    ]

    for name in preferred_names:
        path = ppo_dir / name
        if path.exists():
            return path

    checkpoints = sorted(ppo_dir.glob("*.pth"))
    if not checkpoints:
        return None

    def checkpoint_key(path: Path) -> Tuple[int, str]:
        stem = path.stem
        numbers = [int(x) for x in stem.replace("-", "_").split("_") if x.isdigit()]
        last_num = numbers[-1] if numbers else -1
        return (last_num, path.name)

    checkpoints.sort(key=checkpoint_key, reverse=True)
    return checkpoints[0]


def load_traj(traj_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(traj_path)
    return {key: data[key] for key in data.files}


def build_time_axis(traj: Dict[str, np.ndarray]) -> np.ndarray:
    if "t" in traj:
        return np.asarray(traj["t"])

    if "time" in traj:
        return np.asarray(traj["time"])

    if "step" in traj:
        return np.asarray(traj["step"])

    n = None
    for key in ["r", "vr", "radius"]:
        if key in traj:
            n = len(traj[key])
            break

    if n is None:
        raise ValueError("Cannot infer time axis from traj.npz")

    return np.arange(n)


def get_radius_array(traj: Dict[str, np.ndarray]) -> np.ndarray:
    if "r" in traj:
        return np.asarray(traj["r"])

    if "radius" in traj:
        return np.asarray(traj["radius"])

    if "x" in traj and "y" in traj:
        x = np.asarray(traj["x"])
        y = np.asarray(traj["y"])
        return np.sqrt(x ** 2 + y ** 2)

    raise ValueError("Cannot find radius array in traj.npz")


def get_vr_array(traj: Dict[str, np.ndarray]) -> np.ndarray:
    if "vr" in traj:
        return np.asarray(traj["vr"])

    if "radial_velocity" in traj:
        return np.asarray(traj["radial_velocity"])

    raise ValueError("Cannot find radial velocity array in traj.npz")


def plot_radius_vs_time(
    t: np.ndarray,
    r: np.ndarray,
    target_r: Optional[float],
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(t, r, linewidth=1.5, label="radius")

    if target_r is not None and np.isfinite(target_r):
        plt.axhline(target_r, linestyle="--", linewidth=1.2, label="target_radius")

    plt.xlabel("Time / Step")
    plt.ylabel("Radius")
    plt.title("Radius vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_radius_error_vs_time(
    t: np.ndarray,
    r: np.ndarray,
    target_r: Optional[float],
    out_path: Path,
) -> None:
    if target_r is None or not np.isfinite(target_r):
        return

    r_error = r - target_r

    plt.figure(figsize=(10, 5))
    plt.plot(t, r_error, linewidth=1.5, label="radius_error")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Time / Step")
    plt.ylabel("Radius Error")
    plt.title("Radius Error vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_vr_vs_time(t: np.ndarray, vr: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(t, vr, linewidth=1.5, label="v_r")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Time / Step")
    plt.ylabel("Radial Velocity")
    plt.title("Radial Velocity vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Find best trajectory and export demo plots.")
    parser.add_argument("--runs-dir", type=str, default="analysis/runs")
    parser.add_argument("--ppo-dir", type=str, default="ppo_orbit")
    parser.add_argument("--out-dir", type=str, default="analysis/demo_best")
    parser.add_argument("--controller-variant", type=str, default=None)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    ppo_dir = Path(args.ppo_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_run = find_best_run(
        runs_dir,
        controller_variant=args.controller_variant,
    )
    best_ckpt = find_best_checkpoint(ppo_dir)

    if best_run is None:
        if args.controller_variant is None:
            raise FileNotFoundError(f"No valid run found under: {runs_dir}")
        raise FileNotFoundError(
            f"No valid run found under: {runs_dir} for controller_variant={args.controller_variant}"
        )

    traj = load_traj(best_run["traj_path"])
    metrics = best_run["metrics"]

    t = build_time_axis(traj)
    r = get_radius_array(traj)
    vr = get_vr_array(traj)

    target_r = None
    if "target_r" in traj:
        target_r_arr = np.asarray(traj["target_r"])
        if target_r_arr.size > 0:
            target_r = float(target_r_arr.flat[0])
    elif "target_radius" in metrics:
        target_r = safe_float(metrics.get("target_radius"), default=float("nan"))

    prefix = args.controller_variant if args.controller_variant is not None else "best"

    radius_fig = out_dir / f"{prefix}_radius_vs_time.png"
    vr_fig = out_dir / f"{prefix}_vr_vs_time.png"
    summary_path = out_dir / f"{prefix}_run_summary.txt"
    radius_error_fig = out_dir / f"{prefix}_radius_error_vs_time.png"
    plot_radius_error_vs_time(t, r, target_r, radius_error_fig)

    plot_radius_vs_time(t, r, target_r, radius_fig)
    plot_vr_vs_time(t, vr, vr_fig)

    lines = [
        f"Best run directory: {best_run['run_dir']}",
        f"Trajectory file: {best_run['traj_path']}",
        f"Metrics file: {best_run['metrics_path']}",
        f"Best checkpoint (if found): {best_ckpt if best_ckpt is not None else 'None'}",
        "",
        "Key metrics:",
        f"  total_reward = {metrics.get('total_reward')}",
        f"  final_radius_error = {metrics.get('final_radius_error')}",
        f"  avg_radius_error = {metrics.get('avg_radius_error')}",
        f"  saturation_rate_mean = {metrics.get('saturation_rate_mean')}",
        f"  controller_variant = {metrics.get('controller_variant')}",
        f"  thrust_newton = {metrics.get('thrust_newton')}",
        f"  r0_over_target = {metrics.get('r0_over_target')}",
        "",
        f"Saved: {radius_fig}",
        f"Saved: {vr_fig}",
        f"Saved: {radius_error_fig}",
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()