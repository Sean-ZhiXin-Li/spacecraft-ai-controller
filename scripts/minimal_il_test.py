from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from envs.orbit_env import OrbitEnv
from ppo_orbit.ppo import normalize_state


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "minimal_il"
MODEL_DIR = PROJECT_ROOT / "models"
STATES_PATH = OUTPUT_DIR / "explicit_states.npy"
ACTIONS_PATH = OUTPUT_DIR / "explicit_actions.npy"
MODEL_PATH = MODEL_DIR / "minimal_il_policy.pth"
SUMMARY_PATH = OUTPUT_DIR / "minimal_il_summary.json"
RADIUS_PLOT_PATH = OUTPUT_DIR / "minimal_il_radius.png"
VR_PLOT_PATH = OUTPUT_DIR / "minimal_il_v_r.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = 100.0
THRUST_SCALE = 10000.0
MAX_STEPS = 100000
R0_OVER_TARGET = 1.00005
STRICT_CFG = {
    "tol_r": 1.0e-3,
    "tol_v": 1.0e-3,
    "tol_ang": 0.02,
    "success_threshold": 200,
}


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def collect_dataset() -> tuple[np.ndarray, np.ndarray]:
    env = make_env()
    controller = OrbitLockController(target_radius=env.target_radius, mu=env.mu, config=OrbitLockConfig())
    set_default_start(env)
    obs = np.asarray(env._get_obs(), dtype=np.float32)

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        info = controller.act_with_info(obs)
        action = np.asarray(info["final_action"], dtype=np.float32)
        states.append(obs.copy())
        actions.append(action.copy())
        next_obs, _, terminated, truncated, _ = env.step(action)
        obs = np.asarray(next_obs, dtype=np.float32)

    states_arr = np.asarray(states, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.float32)
    np.save(STATES_PATH, states_arr)
    np.save(ACTIONS_PATH, actions_arr)
    return states_arr, actions_arr


def train_model(states: np.ndarray, actions: np.ndarray) -> tuple[TinyPolicy, Dict[str, float]]:
    states_norm = np.stack([normalize_state(s) for s in states], axis=0).astype(np.float32)
    dataset = TensorDataset(
        torch.tensor(states_norm, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = TinyPolicy().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 60
    last_loss = float("nan")
    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * x_batch.size(0)
            count += x_batch.size(0)
        last_loss = total / max(1, count)

    torch.save({"model_state": model.state_dict()}, MODEL_PATH)
    return model, {"train_loss": float(last_loss), "epochs": float(epochs)}


def evaluate_model(model: TinyPolicy) -> tuple[Dict[str, float | int | bool | None], Dict[str, List[float]]]:
    env = make_env()
    set_default_start(env)
    obs = np.asarray(env._get_obs(), dtype=np.float32)

    trace = {
        "step": [],
        "radius": [],
        "v_r": [],
    }
    success = False
    terminated = False
    truncated = False
    while not (terminated or truncated):
        x = torch.tensor(normalize_state(obs), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = torch.clamp(model(x), -1.0, 1.0).squeeze(0).cpu().numpy()
        next_obs, _, terminated, truncated, info = env.step(action)
        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1e-12)
        v_r = float(np.dot(env.vel, r_hat))
        trace["step"].append(float(env.steps))
        trace["radius"].append(radius)
        trace["v_r"].append(v_r)
        success = success or bool(info.get("success", False))
        obs = np.asarray(next_obs, dtype=np.float32)

    radius_arr = np.asarray(trace["radius"], dtype=np.float64)
    r_error = radius_arr - env.target_radius
    sign_flip_mask = np.signbit(r_error[1:]) != np.signbit(r_error[:-1]) if len(r_error) > 1 else np.zeros(0, dtype=bool)
    crossing_indices = np.flatnonzero(sign_flip_mask)
    metrics = {
        "crossing_occurs": bool(np.any(sign_flip_mask)),
        "radius_crossings_total": int(np.sum(sign_flip_mask)),
        "first_crossing_step": int(crossing_indices[0] + 1) if len(crossing_indices) else None,
        "success": bool(success),
        "final_radius_error": float(abs(r_error[-1])) if len(r_error) else 0.0,
    }
    return metrics, trace


def save_plot(path: Path, trace: Dict[str, List[float]], key: str, title: str, ylabel: str, target_value: float | None = None) -> None:
    plt.figure(figsize=(10, 4.8))
    plt.plot(trace["step"], trace[key], linewidth=1.5)
    if target_value is not None:
        plt.axhline(target_value, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(MODEL_DIR)

    states, actions = collect_dataset()
    model, train_metrics = train_model(states, actions)
    eval_metrics, trace = evaluate_model(model)

    env = make_env()
    save_plot(RADIUS_PLOT_PATH, trace, "radius", "Minimal IL | radius vs time", "radius [m]", target_value=env.target_radius)
    save_plot(VR_PLOT_PATH, trace, "v_r", "Minimal IL | v_r vs time", "v_r [m/s]", target_value=0.0)

    summary = {
        "dataset_states_path": STATES_PATH.as_posix(),
        "dataset_actions_path": ACTIONS_PATH.as_posix(),
        "model_path": MODEL_PATH.as_posix(),
        "num_samples": int(len(states)),
        "train": train_metrics,
        "eval": eval_metrics,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved dataset to: {STATES_PATH} and {ACTIONS_PATH}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
