from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_run(run_dir: Path) -> dict:
    traj_path = run_dir / "traj.npz"
    metrics_path = run_dir / "metrics.json"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {traj_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    traj = np.load(traj_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    r = np.asarray(traj["r"], dtype=np.float64)
    target = np.asarray(traj["target_r"], dtype=np.float64)
    vr = np.asarray(traj["vr"], dtype=np.float64)
    speed = np.asarray(traj["speed"], dtype=np.float64)
    alignment = np.asarray(traj["alignment"], dtype=np.float64)

    tail_len = min(5000, len(r))
    tail = slice(len(r) - tail_len, len(r))
    radius_error = np.abs(r - target)
    radius_delta = float(radius_error[0] - radius_error[-1]) if len(radius_error) else 0.0

    return {
        "run_dir": str(run_dir),
        "checkpoint_path": metrics.get("checkpoint_path"),
        "num_steps": int(metrics.get("num_steps", len(r))),
        "avg_radius_error": float(np.mean(radius_error)) if len(radius_error) else None,
        "final_radius_error": float(radius_error[-1]) if len(radius_error) else None,
        "min_radius_error": float(np.min(radius_error)) if len(radius_error) else None,
        "radius_improvement": radius_delta,
        "tail_avg_radius_error": float(np.mean(radius_error[tail])) if tail_len else None,
        "tail_mean_abs_vr": float(np.mean(np.abs(vr[tail]))) if tail_len else None,
        "tail_mean_alignment": float(np.mean(alignment[tail])) if tail_len else None,
        "tail_speed_mean": float(np.mean(speed[tail])) if tail_len else None,
        "tail_speed_std": float(np.std(speed[tail])) if tail_len else None,
        "tail_speed_cv": float(np.std(speed[tail]) / (np.mean(speed[tail]) + 1e-12)) if tail_len else None,
        "tail_radius_std": float(np.std(r[tail])) if tail_len else None,
        "terminated": bool(metrics.get("terminated", False)),
        "truncated": bool(metrics.get("truncated", False)),
        "total_reward": metrics.get("total_reward"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize behavioral metrics from a rollout run directory.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    summary = summarize_run(args.run_dir.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
