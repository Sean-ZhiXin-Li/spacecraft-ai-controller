from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import inspect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from controller.ppo_controller import PPOController
from envs.orbit_env import OrbitEnv

def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def make_output_dir(root: Path) -> Path:
    ts = int(time.time())
    out_dir = root / f"ppo_recovered_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def ensure_numpy(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def compute_radius(pos: np.ndarray) -> float:
    return float(np.linalg.norm(pos))


def compute_radial_velocity(pos: np.ndarray, vel: np.ndarray) -> float:
    r = np.linalg.norm(pos)
    if r < 1e-12:
        return 0.0
    return float(np.dot(pos, vel) / r)


def plot_radius_vs_time(t: np.ndarray, r: np.ndarray, target_r: float, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(t, r, label="radius", linewidth=1.5)
    plt.axhline(target_r, linestyle="--", linewidth=1.2, label="target_radius")
    plt.xlabel("Step")
    plt.ylabel("Radius")
    plt.title("PPO Recovered Rollout: Radius vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_vr_vs_time(t: np.ndarray, vr: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(t, vr, label="v_r", linewidth=1.5)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Step")
    plt.ylabel("Radial Velocity")
    plt.title("PPO Recovered Rollout: Radial Velocity vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_radius_error_vs_time(t: np.ndarray, r: np.ndarray, target_r: float, out_path: Path) -> None:
    r_error = r - target_r
    plt.figure(figsize=(10, 5))
    plt.plot(t, r_error, label="radius_error", linewidth=1.5)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Step")
    plt.ylabel("Radius Error")
    plt.title("PPO Recovered Rollout: Radius Error vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_env(
    thrust_newton: float,
    r0_over_target: float,
    max_steps: int,
) -> OrbitEnv:
    """
    Adjust keyword arguments here if your OrbitEnv signature is slightly different.
    """
    env = OrbitEnv(
        thrust_newton=thrust_newton,
        r0_over_target=r0_over_target,
        max_steps=max_steps,
    )
    return env


def build_controller(
    checkpoint_path: Path,
) -> PPOController:
    """
    Adjust keyword arguments here if your PPOController signature is slightly different.
    """
    controller = PPOController(
        model_path=str(checkpoint_path),
    )
    return controller


def rollout_once(
    env: OrbitEnv,
    controller: PPOController,
    max_steps: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    obs, info = env.reset()

    r_list: List[float] = []
    vr_list: List[float] = []
    cos_tr_list: List[float] = []
    cos_tt_list: List[float] = []
    action_list: List[np.ndarray] = []
    reward_list: List[float] = []

    total_reward = 0.0
    target_radius = to_float(getattr(env, "target_radius", None), default=np.nan)

    done = False
    truncated = False
    step = 0

    while not (done or truncated) and step < max_steps:
        action = controller(obs)
        action = ensure_numpy(action)

        next_obs, reward, done, truncated, info = env.step(action)

        pos = ensure_numpy(getattr(env, "pos", info.get("pos", [np.nan, np.nan])))
        vel = ensure_numpy(getattr(env, "vel", info.get("vel", [np.nan, np.nan])))

        r = compute_radius(pos)
        vr = compute_radial_velocity(pos, vel)

        r_hat = pos / (np.linalg.norm(pos) + 1e-12)
        t_hat = np.array([-r_hat[1], r_hat[0]], dtype=np.float64)

        action_norm = np.linalg.norm(action)
        if action_norm > 1e-12:
            a_hat = action / action_norm
            cos_tr = float(np.dot(a_hat, r_hat))
            cos_tt = float(np.dot(a_hat, t_hat))
        else:
            cos_tr = 0.0
            cos_tt = 0.0

        r_list.append(r)
        vr_list.append(vr)
        cos_tr_list.append(cos_tr)
        cos_tt_list.append(cos_tt)
        action_list.append(action.copy())
        reward_list.append(float(reward))

        total_reward += float(reward)
        obs = next_obs
        step += 1

    r_arr = np.asarray(r_list, dtype=np.float64)
    vr_arr = np.asarray(vr_list, dtype=np.float64)
    cos_tr_arr = np.asarray(cos_tr_list, dtype=np.float64)
    cos_tt_arr = np.asarray(cos_tt_list, dtype=np.float64)
    actions_arr = np.asarray(action_list, dtype=np.float64)
    rewards_arr = np.asarray(reward_list, dtype=np.float64)
    t_arr = np.arange(len(r_arr), dtype=np.int64)
    target_r_arr = np.full_like(r_arr, target_radius, dtype=np.float64)

    final_radius_error = None
    avg_radius_error = None
    if np.isfinite(target_radius) and len(r_arr) > 0:
        final_radius_error = float(abs(r_arr[-1] - target_radius))
        avg_radius_error = float(np.mean(np.abs(r_arr - target_radius)))

    saturation_rate_mean = None
    if len(actions_arr) > 0:
        action_norms = np.linalg.norm(actions_arr, axis=1)
        saturation_rate_mean = float(np.mean(action_norms > 0.999))

    traj = {
        "t": t_arr,
        "r": r_arr,
        "vr": vr_arr,
        "cos_tr": cos_tr_arr,
        "cos_tt": cos_tt_arr,
        "target_r": target_r_arr,
        "action": actions_arr,
        "reward": rewards_arr,
    }

    metrics = {
        "controller_variant": "ppo_recovered",
        "total_reward": float(total_reward),
        "final_radius_error": final_radius_error,
        "avg_radius_error": avg_radius_error,
        "saturation_rate_mean": saturation_rate_mean,
        "target_radius": None if not np.isfinite(target_radius) else float(target_radius),
        "num_steps": int(len(r_arr)),
        "terminated": bool(done),
        "truncated": bool(truncated),
        "thrust_newton": to_float(getattr(env, "thrust_newton", None), default=np.nan),
        "r0_over_target": to_float(getattr(env, "r0_over_target", None), default=np.nan),
        "checkpoint_path": str(getattr(controller, "model_path", "")),
    }

    return traj, metrics


def save_outputs(
    out_dir: Path,
    traj: Dict[str, np.ndarray],
    metrics: Dict[str, Any],
) -> None:
    np.savez(out_dir / "traj.npz", **traj)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    t = traj["t"]
    r = traj["r"]
    vr = traj["vr"]
    target_r = metrics.get("target_radius")

    if target_r is not None:
        plot_radius_vs_time(t, r, float(target_r), out_dir / "radius_vs_time.png")
        plot_radius_error_vs_time(t, r, float(target_r), out_dir / "radius_error_vs_time.png")

    plot_vr_vs_time(t, vr, out_dir / "vr_vs_time.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover PPO rollout from saved checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="ppo_orbit/ppo_best_model.pth")
    parser.add_argument("--runs-root", type=str, default="analysis/runs")
    parser.add_argument("--thrust-newton", type=float, default=1500.0)
    parser.add_argument("--r0-over-target", type=float, default=1.01)
    parser.add_argument("--max-steps", type=int, default=4000)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    runs_root = Path(args.runs_root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_dir = make_output_dir(runs_root)

    # Optional sanity check that torch can read the checkpoint file.
    _ = torch.load(checkpoint_path, map_location="cpu")

    print("OrbitEnv signature:")
    print(inspect.signature(OrbitEnv.__init__))

    env = build_env(
        thrust_newton=args.thrust_newton,
        r0_over_target=args.r0_over_target,
        max_steps=args.max_steps,
    )
    controller = build_controller(checkpoint_path)

    traj, metrics = rollout_once(
        env=env,
        controller=controller,
        max_steps=args.max_steps,
    )
    save_outputs(out_dir, traj, metrics)

    print(f"Recovered PPO rollout saved to: {out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()