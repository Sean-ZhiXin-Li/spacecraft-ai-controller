from __future__ import annotations

import json
import math
import random
import sys
from copy import deepcopy
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


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "residual_explicit_tune"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "residual_explicit_tuned_policy.pth"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
RADIUS_PLOT_PATH = OUTPUT_DIR / "radius.png"
VR_PLOT_PATH = OUTPUT_DIR / "v_r.png"
ACTION_COMPARE_PLOT_PATH = OUTPUT_DIR / "action_compare.png"
NOTE_PATH = PROJECT_ROOT / "analysis" / "residual_explicit_tune_result.md"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 7
HISTORY_LEN = 4
DT = 100.0
THRUST_SCALE = 10000.0
MAX_STEPS = 100000
R0_OVER_TARGET = 1.00005
ALPHA = 0.05
RAW_RESIDUAL_CLIP = 0.2
RESIDUAL_NORM_WEIGHT = 0.01
TAIL_VR_WEIGHT = 0.1
TUNE_LR = 0.02
UPDATE_DIRECTIONS = [
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
]
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


def tail_slice(length: int, tail_len: int = 5000) -> slice:
    use_len = min(tail_len, length)
    return slice(length - use_len, length)


def objective(metrics: Dict[str, object]) -> float:
    return (
        float(metrics["final_radius_error"])
        + TAIL_VR_WEIGHT * float(metrics["tail_mean_abs_vr"])
        + RESIDUAL_NORM_WEIGHT * float(metrics["mean_abs_residual"])
    )


def compute_metrics(
    radius_values: List[float],
    vr_values: List[float],
    residual_values: List[np.ndarray],
    target_radius: float,
    success: bool,
) -> Dict[str, float | int | bool | None]:
    radius_arr = np.asarray(radius_values, dtype=np.float64)
    vr_arr = np.asarray(vr_values, dtype=np.float64)
    residual_arr = np.asarray(residual_values, dtype=np.float64)
    r_error = radius_arr - target_radius
    sign_flip_mask = np.signbit(r_error[1:]) != np.signbit(r_error[:-1]) if len(r_error) > 1 else np.zeros(0, dtype=bool)
    crossing_indices = np.flatnonzero(sign_flip_mask)
    ts = tail_slice(len(vr_arr))
    metrics: Dict[str, float | int | bool | None] = {
        "success": bool(success),
        "crossing_occurs": bool(np.any(sign_flip_mask)),
        "radius_crossings_total": int(np.sum(sign_flip_mask)),
        "first_crossing_step": int(crossing_indices[0] + 1) if len(crossing_indices) else None,
        "final_radius_error": float(abs(r_error[-1])) if len(r_error) else 0.0,
        "tail_mean_abs_vr": float(np.mean(np.abs(vr_arr[ts]))) if len(vr_arr) else 0.0,
        "mean_abs_residual": float(np.mean(np.abs(residual_arr))) if len(residual_arr) else 0.0,
        "max_abs_residual": float(np.max(np.abs(residual_arr))) if len(residual_arr) else 0.0,
    }
    metrics["objective"] = objective(metrics)
    return metrics


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
    radius_values: List[float] = []
    vr_values: List[float] = []
    residual_values: List[np.ndarray] = []
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
        radius_values.append(radius)
        vr_values.append(v_r)
        residual_values.append(np.zeros(2, dtype=np.float64))
        success = success or bool(step_info.get("success", False))
        obs = np.asarray(next_obs, dtype=np.float32)

    return compute_metrics(radius_values, vr_values, residual_values, env.target_radius, success), trace


def rollout_hybrid(model: ResidualPolicy) -> tuple[Dict[str, float | int | bool | None], Dict[str, List[float]]]:
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
            raw_residual = model(x_tensor).squeeze(0)
            clipped_residual = torch.clamp(raw_residual, -RAW_RESIDUAL_CLIP, RAW_RESIDUAL_CLIP)
            residual = clipped_residual.cpu().numpy().astype(np.float64)
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
        radius_values.append(radius)
        vr_values.append(v_r)
        residual_values.append(residual)
        success = success or bool(step_info.get("success", False))

        obs = np.asarray(next_obs, dtype=np.float32)
        history.pop(0)
        history.append(np.asarray(normalize_state(obs), dtype=np.float32))

    return compute_metrics(radius_values, vr_values, residual_values, env.target_radius, success), trace


def make_model() -> ResidualPolicy:
    model = ResidualPolicy(input_dim=HISTORY_LEN * 5).to(DEVICE)
    model.eval()
    return model


def set_output_bias(model: ResidualPolicy, bias: np.ndarray) -> None:
    with torch.no_grad():
        model.net[-1].bias.copy_(torch.tensor(bias, dtype=torch.float32, device=DEVICE))


def get_output_bias(model: ResidualPolicy) -> np.ndarray:
    return model.net[-1].bias.detach().cpu().numpy().astype(np.float64)


def tune_model(explicit_metrics: Dict[str, object]) -> tuple[ResidualPolicy, Dict[str, object], List[Dict[str, object]]]:
    model = make_model()
    zero_metrics, _ = rollout_hybrid(model)
    best_objective = float(zero_metrics["objective"])
    best_state = deepcopy(model.state_dict())
    best_metrics = zero_metrics
    current_bias = get_output_bias(model)
    success_required = bool(explicit_metrics["success"])
    history: List[Dict[str, object]] = [
        {
            "step": 0,
            "candidate_bias": current_bias.tolist(),
            "accepted": True,
            "reason": "zero_initialization",
            "metrics": zero_metrics,
        }
    ]

    for step_idx, direction in enumerate(UPDATE_DIRECTIONS, start=1):
        candidate_bias = np.clip(
            current_bias + TUNE_LR * np.asarray(direction, dtype=np.float64),
            -RAW_RESIDUAL_CLIP,
            RAW_RESIDUAL_CLIP,
        )
        set_output_bias(model, candidate_bias)
        candidate_metrics, _ = rollout_hybrid(model)
        candidate_objective = float(candidate_metrics["objective"])
        preserves_success = bool(candidate_metrics["success"]) == success_required
        improves_objective = candidate_objective < best_objective
        accepted = bool(preserves_success and improves_objective)
        reason = "accepted" if accepted else "rejected"
        if not preserves_success:
            reason = "rejected_success_not_preserved"
        elif not improves_objective:
            reason = "rejected_no_objective_improvement"

        if accepted:
            current_bias = candidate_bias
            best_objective = candidate_objective
            best_state = deepcopy(model.state_dict())
            best_metrics = candidate_metrics
        else:
            set_output_bias(model, current_bias)

        history.append(
            {
                "step": step_idx,
                "direction": direction,
                "candidate_bias": candidate_bias.tolist(),
                "accepted": accepted,
                "reason": reason,
                "metrics": candidate_metrics,
            }
        )

    model.load_state_dict(best_state)
    model.eval()
    best_summary = {
        "best_objective": best_objective,
        "best_bias": get_output_bias(model).tolist(),
        "best_metrics_during_tune": best_metrics,
    }
    return model, best_summary, history


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
    plt.title("Residual explicit tune | action comparison")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def metric_improves(candidate: Dict[str, object], baseline: Dict[str, object], key: str) -> bool:
    return float(candidate[key]) < float(baseline[key])


def write_note(summary: Dict[str, object]) -> None:
    explicit = summary["explicit_baseline"]
    tuned = summary["tuned_eval"]
    preserves_success = bool(summary["preserves_explicit_success"])
    improves_radius = bool(summary["improves_final_radius_error"])
    improves_vr = bool(summary["improves_tail_mean_abs_vr"])
    destabilizes = bool(summary["destabilizes_controller"])
    nonzero_attempts = [entry for entry in summary["optimization_history"] if entry["step"] > 0]
    harmful_attempts = [entry for entry in nonzero_attempts if not bool(entry["metrics"]["success"])]
    accepted_nonzero = [entry for entry in nonzero_attempts if bool(entry["accepted"])]

    if len(harmful_attempts) == len(nonzero_attempts) and nonzero_attempts:
        implication = "Every nonzero residual attempt harmed success, so the safe checkpoint remained at zero residual."
    elif harmful_attempts and not accepted_nonzero:
        implication = "Every accepted safe checkpoint remained at zero residual because nonzero attempts harmed success or failed the objective gate."
    elif accepted_nonzero:
        implication = "A tiny residual can be accepted under strict success-preservation gates, but the update should remain tightly bounded and revalidated."
    else:
        implication = "This run supports safe hybrid control as a guarded residual layer: preserve explicit success first, then accept only measured rollout improvements."

    lines = [
        "# Residual Explicit Tuning Result",
        "",
        "## Main Answer",
        "",
        f"- Does the tuned residual preserve explicit-controller success? `{'Yes' if preserves_success else 'No'}`",
        f"- Does it improve final radius error? `{'Yes' if improves_radius else 'No'}`",
        f"- Does it improve tail radial velocity? `{'Yes' if improves_vr else 'No'}`",
        f"- Does it destabilize the controller? `{'Yes' if destabilizes else 'No'}`",
        f"- Did every nonzero tuned residual harm success? `{'Yes' if len(harmful_attempts) == len(nonzero_attempts) and nonzero_attempts else 'No'}`",
        f"- What does this imply about safe hybrid control? {implication}",
        "",
        "## Metrics",
        "",
        f"- alpha `{summary['alpha']}`",
        f"- best_objective `{summary['best_objective']:.6e}`",
        f"- tuned success `{tuned['success']}` vs explicit `{explicit['success']}`",
        f"- tuned crossings `{tuned['radius_crossings_total']}` vs explicit `{explicit['radius_crossings_total']}`",
        f"- tuned first_crossing_step `{tuned['first_crossing_step']}` vs explicit `{explicit['first_crossing_step']}`",
        f"- tuned final_radius_error `{tuned['final_radius_error']:.3e}` vs explicit `{explicit['final_radius_error']:.3e}`",
        f"- tuned tail_mean_abs_vr `{tuned['tail_mean_abs_vr']:.3e}` vs explicit `{explicit['tail_mean_abs_vr']:.3e}`",
        f"- tuned mean_abs_residual `{tuned['mean_abs_residual']:.3e}`",
        f"- tuned max_abs_residual `{tuned['max_abs_residual']:.3e}`",
        "",
        "## Optimization History",
        "",
    ]
    for entry in summary["optimization_history"]:
        metrics = entry["metrics"]
        lines.append(
            f"- step `{entry['step']}` accepted `{entry['accepted']}` reason `{entry['reason']}` "
            f"objective `{metrics['objective']:.6e}` success `{metrics['success']}` "
            f"final_radius_error `{metrics['final_radius_error']:.3e}` tail_mean_abs_vr `{metrics['tail_mean_abs_vr']:.3e}`"
        )
    NOTE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(MODEL_DIR)
    set_seed(SEED)

    explicit_metrics, explicit_trace = rollout_explicit()
    tuned_model, best_summary, optimization_history = tune_model(explicit_metrics)
    tuned_metrics, tuned_trace = rollout_hybrid(tuned_model)
    env = make_env()

    save_timeseries_plot(
        RADIUS_PLOT_PATH,
        {"explicit": explicit_trace, "tuned_residual": tuned_trace},
        "radius",
        "Residual explicit tune | radius vs time",
        "radius [m]",
        target_value=env.target_radius,
    )
    save_timeseries_plot(
        VR_PLOT_PATH,
        {"explicit": explicit_trace, "tuned_residual": tuned_trace},
        "v_r",
        "Residual explicit tune | v_r vs time",
        "v_r [m/s]",
        target_value=0.0,
    )
    save_action_compare_plot(ACTION_COMPARE_PLOT_PATH, tuned_trace)

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
        "raw_residual_clip": RAW_RESIDUAL_CLIP,
        "objective": {
            "formula": "final_radius_error + 0.1 * tail_mean_abs_vr + 0.01 * mean_abs_residual",
            "tail_vr_weight": TAIL_VR_WEIGHT,
            "residual_norm_weight": RESIDUAL_NORM_WEIGHT,
        },
        "optimization": {
            "method": "deterministic_direct_rollout_bias_search",
            "tune_lr": TUNE_LR,
            "num_update_attempts": len(UPDATE_DIRECTIONS),
            "tuned_parameters": "final_output_bias_only",
            "acceptance_rule": "preserve_explicit_success_and_improve_objective",
        },
        "seed": SEED,
        "device": str(DEVICE),
        "explicit_baseline": explicit_metrics,
        "tuned_eval": tuned_metrics,
        "best_objective": float(best_summary["best_objective"]),
        "best_bias": best_summary["best_bias"],
        "best_metrics_during_tune": best_summary["best_metrics_during_tune"],
        "optimization_history": optimization_history,
        "preserves_explicit_success": bool(tuned_metrics["success"]) == bool(explicit_metrics["success"]),
        "improves_final_radius_error": metric_improves(tuned_metrics, explicit_metrics, "final_radius_error"),
        "improves_tail_mean_abs_vr": metric_improves(tuned_metrics, explicit_metrics, "tail_mean_abs_vr"),
        "destabilizes_controller": (
            (bool(explicit_metrics["success"]) and not bool(tuned_metrics["success"]))
            or int(tuned_metrics["radius_crossings_total"]) < int(explicit_metrics["radius_crossings_total"])
            or float(tuned_metrics["final_radius_error"]) > 1.1 * float(explicit_metrics["final_radius_error"])
            or float(tuned_metrics["tail_mean_abs_vr"]) > 1.1 * float(explicit_metrics["tail_mean_abs_vr"])
        ),
    }

    torch.save(
        {
            "model_state": tuned_model.state_dict(),
            "history_len": HISTORY_LEN,
            "alpha": ALPHA,
            "raw_residual_clip": RAW_RESIDUAL_CLIP,
            "seed": SEED,
            "best_objective": float(best_summary["best_objective"]),
            "best_bias": best_summary["best_bias"],
        },
        MODEL_PATH,
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
