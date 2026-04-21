from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from envs.orbit_env import OrbitEnv
from ppo_orbit.ppo import normalize_state


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "residual_explicit_il"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "residual_explicit_il_policy.pth"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
RADIUS_PLOT_PATH = OUTPUT_DIR / "radius.png"
VR_PLOT_PATH = OUTPUT_DIR / "v_r.png"
ACTION_COMPARE_PLOT_PATH = OUTPUT_DIR / "action_compare.png"
NOTE_PATH = PROJECT_ROOT / "analysis" / "residual_explicit_il_result.md"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 7
HISTORY_LEN = 4
DT = 100.0
THRUST_SCALE = 10000.0
MAX_STEPS = 100000
R0_OVER_TARGET = 1.00005
ALPHA = 0.2
STRICT_CFG = {
    "tol_r": 1.0e-3,
    "tol_v": 1.0e-3,
    "tol_ang": 0.02,
    "success_threshold": 200,
}


class ResidualPolicy(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def make_controller(env: OrbitEnv) -> OrbitLockController:
    return OrbitLockController(target_radius=env.target_radius, mu=env.mu, config=OrbitLockConfig())


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


def collect_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = make_env()
    controller = make_controller(env)
    set_default_start(env)
    obs = np.asarray(env._get_obs(), dtype=np.float32)

    states: List[np.ndarray] = []
    explicit_actions: List[np.ndarray] = []
    residual_targets: List[np.ndarray] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        info = controller.act_with_info(obs)
        action = np.asarray(info["final_action"], dtype=np.float32)
        states.append(obs.copy())
        explicit_actions.append(action.copy())
        residual_targets.append(np.zeros(2, dtype=np.float32))
        next_obs, _, terminated, truncated, _ = env.step(action)
        obs = np.asarray(next_obs, dtype=np.float32)

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(explicit_actions, dtype=np.float32),
        np.asarray(residual_targets, dtype=np.float32),
    )


def build_history_inputs(states: np.ndarray) -> np.ndarray:
    states_norm = np.stack([normalize_state(state) for state in states], axis=0).astype(np.float32)
    padded = np.repeat(states_norm[:1], HISTORY_LEN - 1, axis=0)
    padded = np.concatenate([padded, states_norm], axis=0)
    windows = []
    for idx in range(len(states_norm)):
        windows.append(padded[idx : idx + HISTORY_LEN].reshape(-1))
    return np.asarray(windows, dtype=np.float32)


def train_model(history_inputs: np.ndarray, residual_targets: np.ndarray) -> tuple[ResidualPolicy, Dict[str, float]]:
    generator = torch.Generator().manual_seed(SEED)
    dataset = TensorDataset(
        torch.tensor(history_inputs, dtype=torch.float32),
        torch.tensor(residual_targets, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=generator)

    model = ResidualPolicy(input_dim=int(history_inputs.shape[1])).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 30
    last_loss = float("nan")
    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for x_batch, residual_batch in loader:
            x_batch = x_batch.to(DEVICE)
            residual_batch = residual_batch.to(DEVICE)
            residual_pred = model(x_batch)
            loss = loss_fn(residual_pred, residual_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = x_batch.size(0)
            total += loss.item() * batch_size
            count += batch_size
        last_loss = total / max(1, count)

    torch.save(
        {
            "model_state": model.state_dict(),
            "history_len": HISTORY_LEN,
            "alpha": ALPHA,
            "seed": SEED,
        },
        MODEL_PATH,
    )
    return model, {
        "residual_mse": float(last_loss),
        "epochs": float(epochs),
    }


def tail_slice(length: int, tail_len: int = 5000) -> slice:
    use_len = min(tail_len, length)
    return slice(length - use_len, length)


def compute_metrics(trace: Dict[str, List[float]], target_radius: float, success: bool) -> Dict[str, float | int | bool | None]:
    radius_arr = np.asarray(trace["radius"], dtype=np.float64)
    vr_arr = np.asarray(trace["v_r"], dtype=np.float64)
    r_error = radius_arr - target_radius
    sign_flip_mask = np.signbit(r_error[1:]) != np.signbit(r_error[:-1]) if len(r_error) > 1 else np.zeros(0, dtype=bool)
    crossing_indices = np.flatnonzero(sign_flip_mask)
    ts = tail_slice(len(vr_arr))
    return {
        "success": bool(success),
        "crossing_occurs": bool(np.any(sign_flip_mask)),
        "radius_crossings_total": int(np.sum(sign_flip_mask)),
        "first_crossing_step": int(crossing_indices[0] + 1) if len(crossing_indices) else None,
        "final_radius_error": float(abs(r_error[-1])) if len(r_error) else 0.0,
        "tail_mean_abs_vr": float(np.mean(np.abs(vr_arr[ts]))) if len(vr_arr) else 0.0,
    }


def rollout_explicit() -> tuple[Dict[str, float | int | bool | None], Dict[str, List[float]]]:
    env = make_env()
    controller = make_controller(env)
    set_default_start(env)
    obs = np.asarray(env._get_obs(), dtype=np.float32)

    trace: Dict[str, List[float]] = {
        "step": [],
        "radius": [],
        "v_r": [],
        "explicit_action_x": [],
        "explicit_action_y": [],
        "final_action_x": [],
        "final_action_y": [],
        "residual_x": [],
        "residual_y": [],
    }
    success = False
    terminated = False
    truncated = False
    while not (terminated or truncated):
        info = controller.act_with_info(obs)
        explicit_action = np.asarray(info["final_action"], dtype=np.float64)
        next_obs, _, terminated, truncated, step_info = env.step(explicit_action)
        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1e-12)
        v_r = float(np.dot(env.vel, r_hat))

        trace["step"].append(float(env.steps))
        trace["radius"].append(radius)
        trace["v_r"].append(v_r)
        trace["explicit_action_x"].append(float(explicit_action[0]))
        trace["explicit_action_y"].append(float(explicit_action[1]))
        trace["final_action_x"].append(float(explicit_action[0]))
        trace["final_action_y"].append(float(explicit_action[1]))
        trace["residual_x"].append(0.0)
        trace["residual_y"].append(0.0)
        success = success or bool(step_info.get("success", False))

        obs = np.asarray(next_obs, dtype=np.float32)

    return compute_metrics(trace, env.target_radius, success), trace


def rollout_residual(model: ResidualPolicy) -> tuple[Dict[str, float | int | bool | None], Dict[str, List[float]]]:
    env = make_env()
    controller = make_controller(env)
    set_default_start(env)
    obs = np.asarray(env._get_obs(), dtype=np.float32)
    history = [np.asarray(normalize_state(obs), dtype=np.float32) for _ in range(HISTORY_LEN)]

    trace: Dict[str, List[float]] = {
        "step": [],
        "radius": [],
        "v_r": [],
        "explicit_action_x": [],
        "explicit_action_y": [],
        "final_action_x": [],
        "final_action_y": [],
        "residual_x": [],
        "residual_y": [],
    }
    success = False
    terminated = False
    truncated = False
    while not (terminated or truncated):
        info = controller.act_with_info(obs)
        explicit_action = np.asarray(info["final_action"], dtype=np.float64)
        x = np.concatenate(history, axis=0).astype(np.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            residual = model(x_tensor).squeeze(0).cpu().numpy().astype(np.float64)
        final_action = np.clip(explicit_action + ALPHA * residual, -1.0, 1.0)

        next_obs, _, terminated, truncated, step_info = env.step(final_action)
        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1e-12)
        v_r = float(np.dot(env.vel, r_hat))

        trace["step"].append(float(env.steps))
        trace["radius"].append(radius)
        trace["v_r"].append(v_r)
        trace["explicit_action_x"].append(float(explicit_action[0]))
        trace["explicit_action_y"].append(float(explicit_action[1]))
        trace["final_action_x"].append(float(final_action[0]))
        trace["final_action_y"].append(float(final_action[1]))
        trace["residual_x"].append(float(residual[0]))
        trace["residual_y"].append(float(residual[1]))
        success = success or bool(step_info.get("success", False))

        obs = np.asarray(next_obs, dtype=np.float32)
        history.pop(0)
        history.append(np.asarray(normalize_state(obs), dtype=np.float32))

    return compute_metrics(trace, env.target_radius, success), trace


def save_timeseries_plot(
    path: Path,
    traces: Dict[str, Dict[str, List[float]]],
    key: str,
    title: str,
    ylabel: str,
    target_value: float | None = None,
) -> None:
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


def save_action_compare_plot(path: Path, trace: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(10, 5.4))
    plt.plot(trace["step"], trace["explicit_action_x"], label="explicit x", linewidth=1.1)
    plt.plot(trace["step"], trace["final_action_x"], label="hybrid x", linewidth=1.1, linestyle="--")
    plt.plot(trace["step"], trace["explicit_action_y"], label="explicit y", linewidth=1.1)
    plt.plot(trace["step"], trace["final_action_y"], label="hybrid y", linewidth=1.1, linestyle="--")
    plt.plot(trace["step"], np.asarray(trace["residual_x"]) * ALPHA, label="alpha residual x", linewidth=0.9, alpha=0.8)
    plt.plot(trace["step"], np.asarray(trace["residual_y"]) * ALPHA, label="alpha residual y", linewidth=0.9, alpha=0.8)
    plt.xlabel("step")
    plt.ylabel("action component")
    plt.title("Residual explicit IL | action comparison")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def metric_improves(candidate: Dict[str, object], baseline: Dict[str, object], key: str) -> bool:
    return float(candidate[key]) < float(baseline[key])


def write_note(summary: Dict[str, object]) -> None:
    explicit = summary["explicit_baseline"]
    residual = summary["residual_eval"]
    preserves_success = bool(residual["success"]) == bool(explicit["success"])
    improves_radius = metric_improves(residual, explicit, "final_radius_error")
    improves_vr = metric_improves(residual, explicit, "tail_mean_abs_vr")
    destabilizes = bool(summary["destabilizes_controller"])

    implication = (
        "Hybrid control preserves the verified explicit structure in this minimal zero-residual test, unlike learning-only policies that had to reproduce the full insertion behavior from scratch."
    )
    if not preserves_success or destabilizes:
        implication = (
            "Even a residual wrapper can destabilize if the learned correction is not tightly constrained; hybrid control should keep residual authority small and explicitly verify preservation."
        )

    lines = [
        "# Residual Explicit IL Result",
        "",
        "## Main Answer",
        "",
        f"- Does the residual policy preserve explicit-controller success? `{'Yes' if preserves_success else 'No'}`",
        f"- Does it improve final radius error? `{'Yes' if improves_radius else 'No'}`",
        f"- Does it improve tail radial velocity? `{'Yes' if improves_vr else 'No'}`",
        f"- Does it destabilize the controller? `{'Yes' if destabilizes else 'No'}`",
        f"- What does this imply about hybrid control vs learning-only control? {implication}",
        "",
        "## Metrics",
        "",
        f"- alpha `{summary['alpha']}`",
        f"- residual success `{residual['success']}` vs explicit `{explicit['success']}`",
        f"- residual crossings `{residual['radius_crossings_total']}` vs explicit `{explicit['radius_crossings_total']}`",
        f"- residual first_crossing_step `{residual['first_crossing_step']}` vs explicit `{explicit['first_crossing_step']}`",
        f"- residual final_radius_error `{residual['final_radius_error']:.3e}` vs explicit `{explicit['final_radius_error']:.3e}`",
        f"- residual tail_mean_abs_vr `{residual['tail_mean_abs_vr']:.3e}` vs explicit `{explicit['tail_mean_abs_vr']:.3e}`",
        f"- mean_abs_residual `{summary['residual_stats']['mean_abs_residual']:.3e}`",
        f"- max_abs_residual `{summary['residual_stats']['max_abs_residual']:.3e}`",
    ]
    NOTE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(MODEL_DIR)
    set_seed(SEED)

    states, explicit_actions, residual_targets = collect_dataset()
    history_inputs = build_history_inputs(states)
    model, train_metrics = train_model(history_inputs, residual_targets)
    explicit_metrics, explicit_trace = rollout_explicit()
    residual_metrics, residual_trace = rollout_residual(model)

    residual_arr = np.stack([residual_trace["residual_x"], residual_trace["residual_y"]], axis=1)
    env = make_env()
    traces = {
        "explicit": explicit_trace,
        "residual_explicit": residual_trace,
    }
    save_timeseries_plot(RADIUS_PLOT_PATH, traces, "radius", "Residual explicit IL | radius vs time", "radius [m]", target_value=env.target_radius)
    save_timeseries_plot(VR_PLOT_PATH, traces, "v_r", "Residual explicit IL | v_r vs time", "v_r [m/s]", target_value=0.0)
    save_action_compare_plot(ACTION_COMPARE_PLOT_PATH, residual_trace)

    summary = {
        "model_path": MODEL_PATH.as_posix(),
        "baseline_setup": {
            "dt": DT,
            "thrust_scale": THRUST_SCALE,
            "max_steps": MAX_STEPS,
            "r0_over_target": R0_OVER_TARGET,
            **STRICT_CFG,
        },
        "history_len": HISTORY_LEN,
        "alpha": ALPHA,
        "seed": SEED,
        "device": str(DEVICE),
        "num_samples": int(len(states)),
        "input_dim": int(history_inputs.shape[1]),
        "state_dim": int(states.shape[1]),
        "residual_target": "zero_correction_on_explicit_controller_action",
        "train": train_metrics,
        "explicit_action_dataset_stats": {
            "mean_action_x": float(np.mean(explicit_actions[:, 0])),
            "mean_action_y": float(np.mean(explicit_actions[:, 1])),
            "mean_action_norm": float(np.mean(np.linalg.norm(explicit_actions, axis=1))),
        },
        "explicit_baseline": explicit_metrics,
        "residual_eval": residual_metrics,
        "residual_stats": {
            "mean_abs_residual": float(np.mean(np.abs(residual_arr))) if len(residual_arr) else 0.0,
            "max_abs_residual": float(np.max(np.abs(residual_arr))) if len(residual_arr) else 0.0,
            "mean_scaled_residual_norm": float(np.mean(np.linalg.norm(ALPHA * residual_arr, axis=1))) if len(residual_arr) else 0.0,
        },
        "improves_final_radius_error": metric_improves(residual_metrics, explicit_metrics, "final_radius_error"),
        "improves_tail_mean_abs_vr": metric_improves(residual_metrics, explicit_metrics, "tail_mean_abs_vr"),
    }
    summary["preserves_explicit_success"] = bool(residual_metrics["success"]) == bool(explicit_metrics["success"])
    summary["destabilizes_controller"] = (
        (bool(explicit_metrics["success"]) and not bool(residual_metrics["success"]))
        or int(residual_metrics["radius_crossings_total"]) < int(explicit_metrics["radius_crossings_total"])
        or float(residual_metrics["final_radius_error"]) > 1.1 * float(explicit_metrics["final_radius_error"])
        or float(residual_metrics["tail_mean_abs_vr"]) > 1.1 * float(explicit_metrics["tail_mean_abs_vr"])
    )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_note(summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")
    print(f"Saved plots to: {RADIUS_PLOT_PATH}, {VR_PLOT_PATH}, {ACTION_COMPARE_PLOT_PATH}")
    print(f"Saved note to: {NOTE_PATH}")


if __name__ == "__main__":
    main()
