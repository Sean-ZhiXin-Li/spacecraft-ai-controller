from __future__ import annotations

import csv
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from envs.orbit_env import OrbitEnv
from ppo_orbit.ppo import normalize_state


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "residual_explicit_alpha_sweep"
MODEL_PATH = PROJECT_ROOT / "models" / "residual_explicit_il_policy.pth"
SUMMARY_JSON_PATH = OUTPUT_DIR / "summary.json"
SUMMARY_CSV_PATH = OUTPUT_DIR / "summary.csv"
FINAL_RADIUS_ERROR_PLOT_PATH = OUTPUT_DIR / "final_radius_error.png"
TAIL_MEAN_ABS_VR_PLOT_PATH = OUTPUT_DIR / "tail_mean_abs_vr.png"
NOTE_PATH = PROJECT_ROOT / "analysis" / "residual_explicit_alpha_sweep_result.md"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 7
HISTORY_LEN = 4
DT = 100.0
THRUST_SCALE = 10000.0
MAX_STEPS = 100000
R0_OVER_TARGET = 1.00005
ALPHAS = [0.0, 0.02, 0.05, 0.1, 0.2]
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


def load_model() -> tuple[ResidualPolicy, Dict[str, object]]:
    input_dim = HISTORY_LEN * 5
    model = ResidualPolicy(input_dim=input_dim).to(DEVICE)
    metadata: Dict[str, object] = {
        "model_path": MODEL_PATH.as_posix(),
        "loaded_checkpoint": False,
        "fallback": "zero_output_initialization",
    }
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        metadata = {
            "model_path": MODEL_PATH.as_posix(),
            "loaded_checkpoint": True,
            "checkpoint_history_len": checkpoint.get("history_len"),
            "checkpoint_seed": checkpoint.get("seed"),
            "checkpoint_alpha": checkpoint.get("alpha"),
        }
    model.eval()
    return model, metadata


def tail_slice(length: int, tail_len: int = 5000) -> slice:
    use_len = min(tail_len, length)
    return slice(length - use_len, length)


def rollout_alpha(model: ResidualPolicy, alpha: float) -> Dict[str, float | int | bool | None]:
    env = make_env()
    controller = make_controller(env)
    set_default_start(env)
    obs = np.asarray(env._get_obs(), dtype=np.float32)
    history = [np.asarray(normalize_state(obs), dtype=np.float32) for _ in range(HISTORY_LEN)]

    radius_values: List[float] = []
    vr_values: List[float] = []
    residual_values: List[np.ndarray] = []
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
        final_action = np.clip(explicit_action + alpha * residual, -1.0, 1.0)

        next_obs, _, terminated, truncated, step_info = env.step(final_action)
        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1e-12)
        v_r = float(np.dot(env.vel, r_hat))

        radius_values.append(radius)
        vr_values.append(v_r)
        residual_values.append(residual)
        success = success or bool(step_info.get("success", False))

        obs = np.asarray(next_obs, dtype=np.float32)
        history.pop(0)
        history.append(np.asarray(normalize_state(obs), dtype=np.float32))

    radius_arr = np.asarray(radius_values, dtype=np.float64)
    vr_arr = np.asarray(vr_values, dtype=np.float64)
    residual_arr = np.asarray(residual_values, dtype=np.float64)
    r_error = radius_arr - env.target_radius
    sign_flip_mask = np.signbit(r_error[1:]) != np.signbit(r_error[:-1]) if len(r_error) > 1 else np.zeros(0, dtype=bool)
    crossing_indices = np.flatnonzero(sign_flip_mask)
    ts = tail_slice(len(vr_arr))
    return {
        "alpha": float(alpha),
        "success": bool(success),
        "crossing_occurs": bool(np.any(sign_flip_mask)),
        "radius_crossings_total": int(np.sum(sign_flip_mask)),
        "first_crossing_step": int(crossing_indices[0] + 1) if len(crossing_indices) else None,
        "final_radius_error": float(abs(r_error[-1])) if len(r_error) else 0.0,
        "tail_mean_abs_vr": float(np.mean(np.abs(vr_arr[ts]))) if len(vr_arr) else 0.0,
        "mean_abs_residual": float(np.mean(np.abs(residual_arr))) if len(residual_arr) else 0.0,
        "max_abs_residual": float(np.max(np.abs(residual_arr))) if len(residual_arr) else 0.0,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "alpha",
        "success",
        "crossing_occurs",
        "radius_crossings_total",
        "first_crossing_step",
        "final_radius_error",
        "tail_mean_abs_vr",
        "mean_abs_residual",
        "max_abs_residual",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def save_metric_plot(path: Path, rows: List[Dict[str, object]], key: str, title: str, ylabel: str) -> None:
    alphas = [float(row["alpha"]) for row in rows]
    values = [float(row[key]) for row in rows]
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(alphas, values, marker="o", linewidth=1.5)
    plt.xlabel("alpha")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_note(summary: Dict[str, object]) -> None:
    rows = summary["rows"]
    baseline = rows[0]
    preserving = [row for row in rows if bool(row["success"]) == bool(baseline["success"])]
    preserving_alphas = [float(row["alpha"]) for row in preserving]
    nonzero_rows = [row for row in rows if float(row["alpha"]) > 0.0]
    improves_radius = [row for row in nonzero_rows if float(row["final_radius_error"]) < float(baseline["final_radius_error"])]
    improves_vr = [row for row in nonzero_rows if float(row["tail_mean_abs_vr"]) < float(baseline["tail_mean_abs_vr"])]

    if preserving_alphas:
        alpha_range = f"{min(preserving_alphas):.2f} to {max(preserving_alphas):.2f}"
    else:
        alpha_range = "none"

    max_metric_delta = max(
        (
            max(
                abs(float(row["final_radius_error"]) - float(baseline["final_radius_error"])),
                abs(float(row["tail_mean_abs_vr"]) - float(baseline["tail_mean_abs_vr"])),
            )
            for row in rows
        ),
        default=0.0,
    )
    sensitivity = (
        "No measurable sensitivity in this sweep because the loaded residual policy outputs zero correction."
        if max_metric_delta == 0.0
        else "The controller shows measurable sensitivity to residual authority; residual magnitude and alpha should be increased cautiously."
    )

    lines = [
        "# Residual Explicit Alpha Sweep Result",
        "",
        "## Main Answer",
        "",
        f"- What alpha range preserves explicit-controller success? `{alpha_range}`",
        f"- Does any nonzero alpha improve final radius error? `{'Yes' if improves_radius else 'No'}`",
        f"- Does any nonzero alpha improve tail radial velocity? `{'Yes' if improves_vr else 'No'}`",
        f"- How sensitive is the controller to residual authority? {sensitivity}",
        "- What does this imply for safe hybrid learning? Start from verified zero-residual preservation, keep residual authority small, and require explicit success-preservation checks before training nonzero corrections.",
        "",
        "## Sweep",
        "",
    ]
    for row in rows:
        lines.append(
            f"- alpha `{float(row['alpha']):.2f}`: success `{row['success']}`, crossings `{row['radius_crossings_total']}`, "
            f"first_crossing_step `{row['first_crossing_step']}`, final_radius_error `{float(row['final_radius_error']):.3e}`, "
            f"tail_mean_abs_vr `{float(row['tail_mean_abs_vr']):.3e}`, max_abs_residual `{float(row['max_abs_residual']):.3e}`"
        )
    NOTE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    set_seed(SEED)
    model, model_metadata = load_model()
    rows = [rollout_alpha(model, alpha) for alpha in ALPHAS]
    summary = {
        "baseline_setup": {
            "dt": DT,
            "thrust_scale": THRUST_SCALE,
            "max_steps": MAX_STEPS,
            "r0_over_target": R0_OVER_TARGET,
            **STRICT_CFG,
        },
        "history_len": HISTORY_LEN,
        "seed": SEED,
        "device": str(DEVICE),
        "alphas": ALPHAS,
        "model": model_metadata,
        "rows": rows,
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(SUMMARY_CSV_PATH, rows)
    save_metric_plot(FINAL_RADIUS_ERROR_PLOT_PATH, rows, "final_radius_error", "Residual explicit alpha sweep | final radius error", "final radius error [m]")
    save_metric_plot(TAIL_MEAN_ABS_VR_PLOT_PATH, rows, "tail_mean_abs_vr", "Residual explicit alpha sweep | tail mean abs v_r", "tail mean abs v_r [m/s]")
    write_note(summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved summary JSON to: {SUMMARY_JSON_PATH}")
    print(f"Saved summary CSV to: {SUMMARY_CSV_PATH}")
    print(f"Saved plots to: {FINAL_RADIUS_ERROR_PLOT_PATH}, {TAIL_MEAN_ABS_VR_PLOT_PATH}")
    print(f"Saved note to: {NOTE_PATH}")


if __name__ == "__main__":
    main()
