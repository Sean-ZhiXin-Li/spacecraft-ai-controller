from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from envs.orbit_env import OrbitEnv
from scripts.day20_policy_surface import DEFAULT_CHECKPOINTS, LoadedPolicy
from ppo_orbit.ppo import ActorCritic, normalize_state


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figs" / "bc_policy"
STRICT_CFG = {
    "tol_r": 1.0e-3,
    "tol_v": 1.0e-3,
    "tol_ang": 0.02,
    "success_threshold": 200,
}
DT = 50.0
MAX_STEPS = 100000
R0_OVER_TARGET = 1.00005
THRUST_SCALE = 20000.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BCPolicy(nn.Module):
    def __init__(self, hidden1: int = 256, hidden2: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(5, hidden1),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.shared(x))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_env() -> OrbitEnv:
    return OrbitEnv(
        thrust_scale=THRUST_SCALE,
        dt=DT,
        max_steps=MAX_STEPS,
        reward_mode="orbit_circular_minimal",
        tol_r=STRICT_CFG["tol_r"],
        tol_v=STRICT_CFG["tol_v"],
        tol_ang=STRICT_CFG["tol_ang"],
        success_threshold=STRICT_CFG["success_threshold"],
        use_action_smoothing=False,
        use_orbit_capture_assist=False,
        verbose=False,
    )


def set_default_start(env: OrbitEnv) -> None:
    env.reset(start_mode="default")
    env.steps = 0
    env.success_counter = 0
    env.radial_stall_counter = 0
    env.orbit_lock_counter = 0
    env.prev_action = np.zeros(2, dtype=np.float64)
    env.pos = np.array([0.0, R0_OVER_TARGET * env.target_radius], dtype=np.float64)
    v_mag = math.sqrt(env.mu / np.linalg.norm(env.pos))
    angle = math.radians(170.0)
    env.vel = v_mag * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)


class BCPolicyController:
    def __init__(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        self.model = BCPolicy().to(DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def act(self, obs: Sequence[float]) -> List[float]:
        x_np = normalize_state(np.asarray(obs, dtype=np.float32))
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = torch.clamp(self.model(x), -1.0, 1.0).squeeze(0).cpu().numpy()
        return [float(action[0]), float(action[1])]


class PPOPolicyController:
    def __init__(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        self.model = ActorCritic().to(DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def act(self, obs: Sequence[float]) -> List[float]:
        x_np = normalize_state(np.asarray(obs, dtype=np.float32))
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.model(x)
            action = torch.clamp(mu, -1.0, 1.0).squeeze(0).cpu().numpy()
        return [float(action[0]), float(action[1])]


class MaxRetrogradeController:
    def act(self, obs: Sequence[float]) -> List[float]:
        vel = np.asarray(obs[2:4], dtype=np.float64)
        v = np.linalg.norm(vel)
        if v < 1e-12:
            return [0.0, 0.0]
        action = -(vel / v)
        return [float(action[0]), float(action[1])]


def evaluate(controller) -> tuple[Dict[str, float | int | bool | None], Dict[str, List[float]]]:
    env = make_env()
    set_default_start(env)
    obs = env._get_obs()
    trace = {
        "step": [],
        "radius": [],
        "v_r": [],
    }
    success = False
    terminated = False
    truncated = False
    for step in range(MAX_STEPS):
        action = np.asarray(controller.act(obs), dtype=np.float64)
        obs, _, terminated, truncated, info = env.step(action)
        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1e-12)
        v_r = float(np.dot(env.vel, r_hat))
        trace["step"].append(float(step))
        trace["radius"].append(radius)
        trace["v_r"].append(v_r)
        success = success or bool(info.get("success", False))
        if terminated or truncated:
            break

    radius_arr = np.asarray(trace["radius"], dtype=np.float64)
    re = radius_arr - env.target_radius
    sign_flip_mask = np.signbit(re[1:]) != np.signbit(re[:-1]) if len(re) > 1 else np.zeros(0, dtype=bool)
    crossing_indices = np.flatnonzero(sign_flip_mask)
    metrics = {
        "radius_crossings_total": int(np.sum(sign_flip_mask)),
        "first_crossing_step": int(crossing_indices[0] + 1) if len(crossing_indices) else None,
        "success": bool(success),
        "final_radius_error": float(abs(re[-1])) if len(re) else 0.0,
    }
    return metrics, trace


def save_plot(path: Path, traces: Dict[str, Dict[str, List[float]]], key: str, title: str, ylabel: str, target_value: float | None = None) -> None:
    plt.figure(figsize=(10, 4.8))
    for label, trace in traces.items():
        plt.plot(trace["step"], trace[key], label=label, linewidth=1.3)
    if target_value is not None:
        plt.axhline(target_value, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BC / PPO / explicit / probe policy on the strict baseline.")
    parser.add_argument("--policy-kind", choices=["bc", "ppo", "explicit", "probe"], default="bc")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "bc_policy.pth")
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)
    if args.policy_kind == "bc":
        controller = BCPolicyController(args.checkpoint)
        label = "bc_policy"
    elif args.policy_kind == "ppo":
        controller = PPOPolicyController(args.checkpoint)
        label = "ppo_policy"
    elif args.policy_kind == "explicit":
        env = make_env()
        controller = OrbitLockController(target_radius=env.target_radius, mu=env.mu, config=OrbitLockConfig())
        label = "explicit_policy"
    else:
        controller = MaxRetrogradeController()
        label = "probe_policy"

    metrics, trace = evaluate(controller)
    csv_path = OUTPUT_DIR / f"{label}_metrics.csv"
    json_path = OUTPUT_DIR / f"{label}_metrics.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    env = make_env()
    save_plot(OUTPUT_DIR / f"{label}_radius.png", {label: trace}, "radius", f"{label} | radius vs time", "radius [m]", target_value=env.target_radius)
    save_plot(OUTPUT_DIR / f"{label}_v_r.png", {label: trace}, "v_r", f"{label} | v_r vs time", "v_r [m/s]", target_value=0.0)
    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation artifacts under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
