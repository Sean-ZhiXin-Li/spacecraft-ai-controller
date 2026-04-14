from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from scripts.recover_ppo_rollout import build_controller, build_env, rollout_once


def _epoch_from_path(path: Path) -> int:
    match = re.search(r"ppo_epoch_(\d+)\.pth$", path.name)
    if match:
        return int(match.group(1))
    return -1


def list_candidate_checkpoints(run_dir: Path) -> List[Path]:
    paths = list(run_dir.glob("ppo_epoch_*.pth"))
    best_path = run_dir / "ppo_best.pth"
    if best_path.exists():
        paths.append(best_path)

    seen = set()
    unique: List[Path] = []
    for path in sorted(paths, key=lambda p: (_epoch_from_path(p), p.name)):
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path.resolve())
    return unique


def summarize_rollout(traj: Dict[str, np.ndarray], metrics: Dict[str, Any], max_steps: int) -> Dict[str, Any]:
    r = np.asarray(traj["r"], dtype=np.float64)
    vr = np.asarray(traj["vr"], dtype=np.float64)
    speed = np.asarray(traj["speed"], dtype=np.float64)
    alignment = np.asarray(traj["alignment"], dtype=np.float64)
    target = np.asarray(traj["target_r"], dtype=np.float64)

    tail_len = min(5000, len(r))
    tail = slice(len(r) - tail_len, len(r))
    radius_error = np.abs(r - target)

    num_steps = int(metrics.get("num_steps", len(r)))
    tail_speed_cv = float(np.std(speed[tail]) / (np.mean(speed[tail]) + 1e-12)) if tail_len else None
    tail_mean_alignment = float(np.mean(alignment[tail])) if tail_len else None

    rejected_reasons: List[str] = []
    if num_steps < int(0.95 * max_steps):
        rejected_reasons.append("survival_drop")
    if tail_speed_cv is not None and tail_speed_cv > 1e-3:
        rejected_reasons.append("speed_unstable")
    if tail_mean_alignment is not None and tail_mean_alignment > 0.8:
        rejected_reasons.append("strongly_radial")

    return {
        "checkpoint_path": metrics.get("checkpoint_path"),
        "checkpoint_name": Path(str(metrics.get("checkpoint_path", ""))).name,
        "num_steps": num_steps,
        "survival_ok": num_steps >= max_steps,
        "avg_radius_error": float(np.mean(radius_error)) if len(radius_error) else None,
        "final_radius_error": float(radius_error[-1]) if len(radius_error) else None,
        "tail_avg_radius_error": float(np.mean(radius_error[tail])) if tail_len else None,
        "tail_mean_abs_vr": float(np.mean(np.abs(vr[tail]))) if tail_len else None,
        "tail_mean_alignment": tail_mean_alignment,
        "tail_radius_std": float(np.std(r[tail])) if tail_len else None,
        "tail_speed_cv": tail_speed_cv,
        "terminated": bool(metrics.get("terminated", False)),
        "truncated": bool(metrics.get("truncated", False)),
        "total_reward": metrics.get("total_reward"),
        "rejected": bool(rejected_reasons),
        "rejected_reasons": rejected_reasons,
    }


def _ranking_key(summary: Dict[str, Any]) -> tuple:
    return (
        1 if summary["rejected"] else 0,
        0 if summary["survival_ok"] else 1,
        float(summary["tail_mean_abs_vr"]),
        float(summary["final_radius_error"]),
        float(summary["tail_avg_radius_error"]),
        float(summary["tail_mean_alignment"]),
        float(summary["tail_radius_std"]),
        float(summary["tail_speed_cv"]),
        str(summary["checkpoint_path"]),
    )


def select_best_checkpoint(
    run_dir: Path,
    *,
    max_steps: int = 20000,
    thrust_scale: float = 20000.0,
    r0_over_target: float = 1.01,
) -> Dict[str, Any]:
    candidates = list_candidate_checkpoints(run_dir)
    if not candidates:
        raise FileNotFoundError(f"No candidate checkpoints found in {run_dir}")

    evaluated: List[Dict[str, Any]] = []
    for checkpoint_path in candidates:
        env = build_env(
            thrust_scale=thrust_scale,
            r0_over_target=r0_over_target,
            max_steps=max_steps,
        )
        controller = build_controller(checkpoint_path)
        traj, metrics = rollout_once(env=env, controller=controller, max_steps=max_steps)
        evaluated.append(
            summarize_rollout(
                traj=traj,
                metrics=metrics,
                max_steps=max_steps,
            )
        )

    ranked = sorted(evaluated, key=_ranking_key)
    best = ranked[0]
    return {
        "run_dir": str(run_dir.resolve()),
        "evaluated_checkpoints": [str(p) for p in candidates],
        "metrics_table": evaluated,
        "ranking": ranked,
        "selected_checkpoint": best["checkpoint_path"],
        "selected_checkpoint_name": best["checkpoint_name"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best checkpoint from a run directory using behavior-based rollout metrics.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline checkpoint to evaluate alongside the run candidates.")
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--thrust-scale", type=float, default=20000.0)
    parser.add_argument("--r0-over-target", type=float, default=1.01)
    args = parser.parse_args()

    result = select_best_checkpoint(
        args.run_dir.resolve(),
        max_steps=args.max_steps,
        thrust_scale=args.thrust_scale,
        r0_over_target=args.r0_over_target,
    )

    if args.baseline is not None:
        env = build_env(
            thrust_scale=args.thrust_scale,
            r0_over_target=args.r0_over_target,
            max_steps=args.max_steps,
        )
        controller = build_controller(args.baseline.resolve())
        traj, metrics = rollout_once(env=env, controller=controller, max_steps=args.max_steps)
        result["baseline"] = summarize_rollout(traj=traj, metrics=metrics, max_steps=args.max_steps)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
