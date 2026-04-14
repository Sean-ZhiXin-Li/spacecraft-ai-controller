from __future__ import annotations

import argparse
import json
import os
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
import torch.nn as nn

from envs.orbit_env import OrbitEnv
from ppo_orbit.ppo import ActorCritic, device

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


def plot_speed_vs_time(t: np.ndarray, speed: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(t, speed, label="speed", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Speed")
    plt.title("PPO Recovered Rollout: Speed vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_alignment_vs_time(t: np.ndarray, alignment: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(t, alignment, label="|cos(angle)|", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("|cos(angle)|")
    plt.title("PPO Recovered Rollout: Alignment vs Time")
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
    thrust_scale: float,
    r0_over_target: float,
    max_steps: int,
) -> OrbitEnv:
    os.environ["R0_OVER_TARGET"] = str(r0_over_target)
    env = OrbitEnv(
        thrust_scale=thrust_scale,
        dt=2.0,
        max_steps=max_steps,
        reward_mode="orbit_circular_minimal",
        w_radius=0.0,
        w_progress=0.0,
        w_speed=0.0,
        use_action_smoothing=False,
        use_orbit_capture_assist=False,
    )
    return env


class LegacyActorCritic(nn.Module):
    """4D rollout model for pre-v_r checkpoints."""
    def __init__(self, hidden1=256, hidden2=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(4, hidden1),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 1)
        )
        self.log_std = nn.Parameter(torch.log(torch.ones(2, device=device) * 0.35))

    def forward(self, x: torch.Tensor):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


def normalize_rollout_state(state: np.ndarray, input_dim: int) -> np.ndarray:
    """Match checkpoint-era observation normalization for 4D and 5D policies."""
    pos_scale = 7.5e12
    vel_scale = 3e4
    if input_dim == 4:
        return np.array([
            state[0] / pos_scale,
            state[1] / pos_scale,
            state[2] / vel_scale,
            state[3] / vel_scale,
        ], dtype=np.float32)
    vr_norm = state[4] / vel_scale
    vr_scaled = vr_norm * (0.1 + 0.9 * np.exp(-abs(vr_norm) * 10))
    return np.array([
        state[0] / pos_scale,
        state[1] / pos_scale,
        state[2] / vel_scale,
        state[3] / vel_scale,
        vr_scaled,
    ], dtype=np.float32)


class RolloutPPOController:
    def __init__(self, checkpoint_path: Path):
        self.model_path = str(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        shared_weight = state_dict.get("shared.0.weight")
        if shared_weight is None or shared_weight.ndim != 2:
            raise KeyError("Checkpoint missing shared.0.weight; cannot infer input dimension.")
        self.input_dim = int(shared_weight.shape[1])
        if self.input_dim == 4:
            self.model = LegacyActorCritic().to(device)
        elif self.input_dim == 5:
            self.model = ActorCritic().to(device)
        else:
            raise ValueError(f"Unsupported checkpoint input dimension: {self.input_dim}")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        state = normalize_rollout_state(obs[: self.input_dim], self.input_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.model(state_tensor)
        return torch.clamp(mu, -1.0, 1.0).squeeze(0).cpu().numpy().astype(np.float32)


def build_controller(
    checkpoint_path: Path,
) -> RolloutPPOController:
    return RolloutPPOController(checkpoint_path)


def rollout_once(
    env: OrbitEnv,
    controller: RolloutPPOController,
    max_steps: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    obs, info = env.reset()

    r_list: List[float] = []
    vr_list: List[float] = []
    speed_list: List[float] = []
    alignment_list: List[float] = []
    action_list: List[np.ndarray] = []
    reward_list: List[float] = []

    total_reward = 0.0
    target_radius = to_float(getattr(env, "target_radius", None), default=np.nan)

    done = False
    truncated = False
    step = 0

    while not (done or truncated) and step < max_steps:
        action = controller.act(obs)
        action = ensure_numpy(action)

        next_obs, reward, done, truncated, info = env.step(action)

        pos = ensure_numpy(getattr(env, "pos", info.get("pos", [np.nan, np.nan])))
        vel = ensure_numpy(getattr(env, "vel", info.get("vel", [np.nan, np.nan])))

        r = compute_radius(pos)
        vr = compute_radial_velocity(pos, vel)
        speed = float(np.linalg.norm(vel))

        r_hat = pos / (np.linalg.norm(pos) + 1e-12)
        t_hat = np.array([-r_hat[1], r_hat[0]], dtype=np.float64)

        action_norm = np.linalg.norm(action)
        if action_norm > 1e-12:
            a_hat = action / action_norm
            cos_tr = float(np.dot(a_hat, r_hat))
        else:
            cos_tr = 0.0

        r_list.append(r)
        vr_list.append(vr)
        speed_list.append(speed)
        alignment_list.append(abs(cos_tr))
        action_list.append(action.copy())
        reward_list.append(float(reward))

        total_reward += float(reward)
        obs = next_obs
        step += 1

    r_arr = np.asarray(r_list, dtype=np.float64)
    vr_arr = np.asarray(vr_list, dtype=np.float64)
    speed_arr = np.asarray(speed_list, dtype=np.float64)
    alignment_arr = np.asarray(alignment_list, dtype=np.float64)
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
        "speed": speed_arr,
        "alignment": alignment_arr,
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
        "thrust_scale": float(getattr(env, "thrust_scale", np.nan)),
        "r0_over_target": to_float(os.environ.get("R0_OVER_TARGET"), default=np.nan),
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
    speed = traj["speed"]
    alignment = traj["alignment"]
    target_r = metrics.get("target_radius")

    if target_r is not None:
        plot_radius_vs_time(t, r, float(target_r), out_dir / "radius_vs_time.png")
        plot_radius_error_vs_time(t, r, float(target_r), out_dir / "radius_error_vs_time.png")

    plot_vr_vs_time(t, vr, out_dir / "vr_vs_time.png")
    plot_speed_vs_time(t, speed, out_dir / "speed_vs_time.png")
    plot_alignment_vs_time(t, alignment, out_dir / "alignment_vs_time.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover PPO rollout from saved checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="ppo_orbit/ppo_best_model.pth")
    parser.add_argument("--runs-root", type=str, default="analysis/runs")
    parser.add_argument("--thrust-scale", type=float, default=20000.0)
    parser.add_argument("--r0-over-target", type=float, default=1.01)
    parser.add_argument("--max-steps", type=int, default=20000)
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
        thrust_scale=args.thrust_scale,
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
