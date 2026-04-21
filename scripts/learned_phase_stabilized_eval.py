from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.orbit_env import OrbitEnv
from ppo_orbit.ppo import normalize_state


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "learned_phase_stabilized"
MODEL_PATH = PROJECT_ROOT / "models" / "learned_phase_il_policy.pth"
LEARNED_PHASE_SUMMARY_PATH = PROJECT_ROOT / "analysis" / "learned_phase_il" / "learned_phase_il_summary.json"
SUMMARY_PATH = OUTPUT_DIR / "learned_phase_stabilized_summary.json"
RADIUS_PLOT_PATH = OUTPUT_DIR / "learned_phase_stabilized_radius.png"
VR_PLOT_PATH = OUTPUT_DIR / "learned_phase_stabilized_v_r.png"
PHASE_PLOT_PATH = OUTPUT_DIR / "learned_phase_stabilized_phase.png"
NOTE_PATH = PROJECT_ROOT / "analysis" / "learned_phase_stabilized_result.md"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 7
HISTORY_LEN = 4
DT = 100.0
THRUST_SCALE = 10000.0
MAX_STEPS = 100000
R0_OVER_TARGET = 1.00005
FORWARD_SWITCH_THRESHOLD = 5
LOCK_TO_CAPTURE_THRESHOLD = 8
STRICT_CFG = {
    "tol_r": 1.0e-3,
    "tol_v": 1.0e-3,
    "tol_ang": 0.02,
    "success_threshold": 200,
}
PHASES = ["DESCENT", "CAPTURE", "LOCK"]


class LearnedPhasePolicy(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.phase_head = nn.Linear(64, len(PHASES))
        self.action_head = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.action_head(z), self.phase_head(z)


class PhaseStabilizer:
    def __init__(self, forward_threshold: int, lock_to_capture_threshold: int) -> None:
        self.forward_threshold = forward_threshold
        self.lock_to_capture_threshold = lock_to_capture_threshold
        self.stable_phase: int | None = None
        self.candidate_phase: int | None = None
        self.candidate_count = 0

    def update(self, raw_phase: int) -> int:
        if self.stable_phase is None:
            self.stable_phase = raw_phase
            return raw_phase

        if raw_phase == self.stable_phase:
            self.candidate_phase = None
            self.candidate_count = 0
            return self.stable_phase

        if raw_phase != self.candidate_phase:
            self.candidate_phase = raw_phase
            self.candidate_count = 1
        else:
            self.candidate_count += 1

        threshold = self.forward_threshold
        if self.stable_phase == 2 and raw_phase == 1:
            threshold = self.lock_to_capture_threshold

        if self.candidate_count >= threshold:
            self.stable_phase = raw_phase
            self.candidate_phase = None
            self.candidate_count = 0
        return self.stable_phase


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


def load_model() -> LearnedPhasePolicy:
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    history_len = int(checkpoint.get("history_len", HISTORY_LEN))
    input_dim = history_len * 5
    model = LearnedPhasePolicy(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def evaluate_model(model: LearnedPhasePolicy) -> tuple[Dict[str, float | int | bool | None], Dict[str, List[float] | List[int]]]:
    env = make_env()
    set_default_start(env)
    obs = np.asarray(env._get_obs(), dtype=np.float32)
    history = [np.asarray(normalize_state(obs), dtype=np.float32) for _ in range(HISTORY_LEN)]
    stabilizer = PhaseStabilizer(FORWARD_SWITCH_THRESHOLD, LOCK_TO_CAPTURE_THRESHOLD)

    trace: Dict[str, List[float] | List[int]] = {
        "step": [],
        "radius": [],
        "v_r": [],
        "raw_phase": [],
        "stabilized_phase": [],
    }
    success = False
    terminated = False
    truncated = False
    while not (terminated or truncated):
        x = np.concatenate(history, axis=0).astype(np.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action_pred, phase_logits = model(x_tensor)
            action = torch.clamp(action_pred, -1.0, 1.0).squeeze(0).cpu().numpy()
            raw_phase = int(torch.argmax(phase_logits, dim=1).item())
            stable_phase = stabilizer.update(raw_phase)

        next_obs, _, terminated, truncated, info = env.step(action)
        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1e-12)
        v_r = float(np.dot(env.vel, r_hat))
        trace["step"].append(float(env.steps))
        trace["radius"].append(radius)
        trace["v_r"].append(v_r)
        trace["raw_phase"].append(raw_phase)
        trace["stabilized_phase"].append(stable_phase)
        success = success or bool(info.get("success", False))

        obs = np.asarray(next_obs, dtype=np.float32)
        history.pop(0)
        history.append(np.asarray(normalize_state(obs), dtype=np.float32))

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


def phase_counts(phase_sequence: List[int]) -> Dict[str, int]:
    return {name: int(sum(phase_id == idx for phase_id in phase_sequence)) for idx, name in enumerate(PHASES)}


def transition_count(phase_sequence: List[int]) -> int:
    return int(sum(curr != prev for prev, curr in zip(phase_sequence[:-1], phase_sequence[1:])))


def load_learned_phase_reference() -> Dict[str, object] | None:
    if not LEARNED_PHASE_SUMMARY_PATH.exists():
        return None
    return json.loads(LEARNED_PHASE_SUMMARY_PATH.read_text(encoding="utf-8"))


def improves_over_reference(eval_metrics: Dict[str, object], reference: Dict[str, object] | None) -> bool | None:
    if reference is None:
        return None
    ref_eval = reference.get("eval", {})
    if bool(eval_metrics["success"]) != bool(ref_eval.get("success", False)):
        return bool(eval_metrics["success"])
    if bool(eval_metrics["crossing_occurs"]) != bool(ref_eval.get("crossing_occurs", False)):
        return bool(eval_metrics["crossing_occurs"])
    return float(eval_metrics["final_radius_error"]) < float(ref_eval.get("final_radius_error", float("inf")))


def save_plot(path: Path, trace: Dict[str, List[float] | List[int]], key: str, title: str, ylabel: str, target_value: float | None = None) -> None:
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


def save_phase_plot(path: Path, trace: Dict[str, List[float] | List[int]]) -> None:
    plt.figure(figsize=(10, 4.8))
    plt.step(trace["step"], trace["raw_phase"], where="post", linewidth=1.0, alpha=0.55, label="raw")
    plt.step(trace["step"], trace["stabilized_phase"], where="post", linewidth=1.5, label="stabilized")
    plt.yticks(range(len(PHASES)), PHASES)
    plt.xlabel("step")
    plt.ylabel("phase")
    plt.title("Learned-phase stabilized eval | phase vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_note(summary: Dict[str, object]) -> None:
    eval_metrics = summary["eval"]
    reference_eval = summary.get("learned_phase_reference_eval")
    improved = summary.get("improves_over_learned_phase_il")
    reduced = bool(summary["phase_oscillation_reduced"])
    if improved is None:
        improved_text = "Unknown; no saved learned_phase_il summary was available."
    else:
        improved_text = "Yes" if improved else "No"

    lines = [
        "# Learned Phase Stabilized Evaluation Result",
        "",
        "## Main Answer",
        "",
        f"- Does phase stabilization improve over learned_phase_il? `{improved_text}`",
        f"- Does it reduce phase oscillation? `{'Yes' if reduced else 'No'}`",
        f"- Does it improve crossing or final radius error? `{'Yes' if improved else 'No' if improved is not None else 'Unknown'}`",
        f"- Does it achieve success? `{'Yes' if bool(eval_metrics['success']) else 'No'}`",
        "",
        "## Interpretation",
        "",
        "- The raw phase head is smoothed with consecutive-step hysteresis, but the loaded policy action head does not consume that stabilized phase signal.",
        "- As expected for this architecture, phase stabilization reduces the diagnostic phase oscillation but does not materially change closed-loop behavior.",
        "- What remains missing is a way for the stabilized internal phase state to affect action selection, or a temporally consistent action policy that learns stable post-crossing dynamics directly.",
        "",
        "## Metrics",
        "",
        f"- crossing_occurs `{eval_metrics['crossing_occurs']}`",
        f"- radius_crossings_total `{eval_metrics['radius_crossings_total']}`",
        f"- first_crossing_step `{eval_metrics['first_crossing_step']}`",
        f"- success `{eval_metrics['success']}`",
        f"- final_radius_error `{eval_metrics['final_radius_error']:.3e}`",
        f"- raw_phase_transitions `{summary['raw_phase_transitions']}`",
        f"- stabilized_phase_transitions `{summary['stabilized_phase_transitions']}`",
    ]
    if isinstance(reference_eval, dict):
        lines.extend(
            [
                "",
                "## Learned-Phase Reference",
                "",
                f"- crossing_occurs `{reference_eval.get('crossing_occurs')}`",
                f"- radius_crossings_total `{reference_eval.get('radius_crossings_total')}`",
                f"- first_crossing_step `{reference_eval.get('first_crossing_step')}`",
                f"- success `{reference_eval.get('success')}`",
                f"- final_radius_error `{float(reference_eval.get('final_radius_error', 0.0)):.3e}`",
            ]
        )
    NOTE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    set_seed(SEED)

    model = load_model()
    eval_metrics, trace = evaluate_model(model)
    raw_phase = [int(x) for x in trace["raw_phase"]]
    stabilized_phase = [int(x) for x in trace["stabilized_phase"]]
    learned_phase_reference = load_learned_phase_reference()
    reference_eval = learned_phase_reference.get("eval") if learned_phase_reference else None
    raw_transitions = transition_count(raw_phase)
    stabilized_transitions = transition_count(stabilized_phase)

    env = make_env()
    save_plot(RADIUS_PLOT_PATH, trace, "radius", "Learned-phase stabilized eval | radius vs time", "radius [m]", target_value=env.target_radius)
    save_plot(VR_PLOT_PATH, trace, "v_r", "Learned-phase stabilized eval | v_r vs time", "v_r [m/s]", target_value=0.0)
    save_phase_plot(PHASE_PLOT_PATH, trace)

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
        "phase_labels": PHASES,
        "phase_stabilization": {
            "forward_switch_threshold": FORWARD_SWITCH_THRESHOLD,
            "lock_to_capture_threshold": LOCK_TO_CAPTURE_THRESHOLD,
            "rule": "switch to a new raw phase only after the required consecutive predictions",
            "action_feedback": "none; action head is unchanged and does not consume stabilized phase",
        },
        "seed": SEED,
        "device": str(DEVICE),
        "eval": eval_metrics,
        "learned_phase_reference_eval": reference_eval,
        "improves_over_learned_phase_il": improves_over_reference(eval_metrics, learned_phase_reference),
        "raw_phase_counts": phase_counts(raw_phase),
        "stabilized_phase_counts": phase_counts(stabilized_phase),
        "raw_phase_transitions": raw_transitions,
        "stabilized_phase_transitions": stabilized_transitions,
        "phase_oscillation_reduced": stabilized_transitions < raw_transitions,
        "raw_predicted_phase_sequence": raw_phase,
        "stabilized_phase_sequence": stabilized_phase,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_note(summary)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"raw_predicted_phase_sequence", "stabilized_phase_sequence"}},
            indent=2,
        )
    )
    print(f"Saved summary to: {SUMMARY_PATH}")
    print(f"Saved plots to: {RADIUS_PLOT_PATH}, {VR_PLOT_PATH}, {PHASE_PLOT_PATH}")
    print(f"Saved note to: {NOTE_PATH}")


if __name__ == "__main__":
    main()
