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


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "soft_phase_conditioned_il"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "soft_phase_conditioned_il_policy.pth"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
RADIUS_PLOT_PATH = OUTPUT_DIR / "radius.png"
VR_PLOT_PATH = OUTPUT_DIR / "v_r.png"
PHASE_PROBS_PLOT_PATH = OUTPUT_DIR / "phase_probs.png"
PHASE_ARGMAX_PLOT_PATH = OUTPUT_DIR / "phase_argmax.png"
NOTE_PATH = PROJECT_ROOT / "analysis" / "soft_phase_conditioned_il_result.md"

LEARNED_PHASE_SUMMARY_PATH = PROJECT_ROOT / "analysis" / "learned_phase_il" / "learned_phase_il_summary.json"
HARD_PHASE_SUMMARY_PATH = PROJECT_ROOT / "analysis" / "phase_conditioned_il" / "summary.json"
HISTORY_AWARE_SUMMARY_PATH = PROJECT_ROOT / "analysis" / "history_aware_il" / "history_aware_il_summary.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 7
HISTORY_LEN = 4
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
PHASES = ["DESCENT", "CAPTURE", "LOCK"]
PHASE_TO_ID = {name: idx for idx, name in enumerate(PHASES)}


class SoftPhaseConditionedPolicy(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.phase_head = nn.Linear(64, len(PHASES))
        self.action_head = nn.Sequential(
            nn.Linear(64 + len(PHASES), 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        phase_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        phase_logits = self.phase_head(z)
        if phase_probs is None:
            phase_probs = torch.softmax(phase_logits, dim=1)
        action = self.action_head(torch.cat([z, phase_probs], dim=1))
        return action, phase_logits, phase_probs


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


def collect_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = make_env()
    controller = OrbitLockController(target_radius=env.target_radius, mu=env.mu, config=OrbitLockConfig())
    set_default_start(env)
    obs = np.asarray(env._get_obs(), dtype=np.float32)

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    labels: List[int] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        info = controller.act_with_info(obs)
        phase = str(info["phase"])
        action = np.asarray(info["final_action"], dtype=np.float32)
        states.append(obs.copy())
        actions.append(action.copy())
        labels.append(PHASE_TO_ID[phase])
        next_obs, _, terminated, truncated, _ = env.step(action)
        obs = np.asarray(next_obs, dtype=np.float32)

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def build_history_inputs(states: np.ndarray) -> np.ndarray:
    states_norm = np.stack([normalize_state(state) for state in states], axis=0).astype(np.float32)
    padded = np.repeat(states_norm[:1], HISTORY_LEN - 1, axis=0)
    padded = np.concatenate([padded, states_norm], axis=0)
    windows = []
    for idx in range(len(states_norm)):
        windows.append(padded[idx : idx + HISTORY_LEN].reshape(-1))
    return np.asarray(windows, dtype=np.float32)


def train_model(
    history_inputs: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
) -> tuple[SoftPhaseConditionedPolicy, Dict[str, float]]:
    generator = torch.Generator().manual_seed(SEED)
    dataset = TensorDataset(
        torch.tensor(history_inputs, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=generator)

    model = SoftPhaseConditionedPolicy(input_dim=int(history_inputs.shape[1])).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    action_loss_fn = nn.MSELoss()
    phase_loss_fn = nn.CrossEntropyLoss()

    epochs = 60
    last_action_loss = float("nan")
    last_phase_loss = float("nan")
    last_total_loss = float("nan")
    for _ in range(epochs):
        model.train()
        action_total = 0.0
        phase_total = 0.0
        total = 0.0
        count = 0
        for x_batch, action_batch, phase_batch in loader:
            x_batch = x_batch.to(DEVICE)
            action_batch = action_batch.to(DEVICE)
            phase_batch = phase_batch.to(DEVICE)
            phase_one_hot = nn.functional.one_hot(phase_batch, num_classes=len(PHASES)).to(dtype=torch.float32)
            action_pred, phase_logits, _ = model(x_batch, phase_probs=phase_one_hot)
            action_loss = action_loss_fn(action_pred, action_batch)
            phase_loss = phase_loss_fn(phase_logits, phase_batch)
            loss = action_loss + 0.5 * phase_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = x_batch.size(0)
            action_total += action_loss.item() * batch_size
            phase_total += phase_loss.item() * batch_size
            total += loss.item() * batch_size
            count += batch_size
        last_action_loss = action_total / max(1, count)
        last_phase_loss = phase_total / max(1, count)
        last_total_loss = total / max(1, count)

    torch.save(
        {
            "model_state": model.state_dict(),
            "history_len": HISTORY_LEN,
            "phase_labels": PHASES,
            "seed": SEED,
        },
        MODEL_PATH,
    )
    return model, {
        "train_loss": float(last_total_loss),
        "action_mse": float(last_action_loss),
        "phase_cross_entropy": float(last_phase_loss),
        "epochs": float(epochs),
    }


def evaluate_model(model: SoftPhaseConditionedPolicy) -> tuple[Dict[str, float | int | bool | None], Dict[str, object]]:
    env = make_env()
    set_default_start(env)
    obs = np.asarray(env._get_obs(), dtype=np.float32)

    history = [np.asarray(normalize_state(obs), dtype=np.float32) for _ in range(HISTORY_LEN)]
    trace: Dict[str, object] = {
        "step": [],
        "radius": [],
        "v_r": [],
        "phase_probs": [],
        "phase_argmax": [],
    }
    success = False
    terminated = False
    truncated = False
    while not (terminated or truncated):
        x = np.concatenate(history, axis=0).astype(np.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action_pred, _, phase_probs = model(x_tensor)
            action = torch.clamp(action_pred, -1.0, 1.0).squeeze(0).cpu().numpy()
            phase_prob_values = phase_probs.squeeze(0).cpu().numpy()
            phase_id = int(np.argmax(phase_prob_values))

        next_obs, _, terminated, truncated, info = env.step(action)
        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1e-12)
        v_r = float(np.dot(env.vel, r_hat))
        trace["step"].append(float(env.steps))
        trace["radius"].append(radius)
        trace["v_r"].append(v_r)
        trace["phase_probs"].append([float(x) for x in phase_prob_values])
        trace["phase_argmax"].append(phase_id)
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


def load_summary(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def phase_counts(phase_sequence: List[int]) -> Dict[str, int]:
    return {name: int(sum(phase_id == idx for phase_id in phase_sequence)) for idx, name in enumerate(PHASES)}


def phase_transition_summary(phase_sequence: List[int], max_examples: int = 50) -> Dict[str, object]:
    transitions: List[Dict[str, int | str]] = []
    total = 0
    if not phase_sequence:
        return {"total": 0, "first_examples": transitions}
    prev = phase_sequence[0]
    for idx, current in enumerate(phase_sequence[1:], start=1):
        if current != prev:
            total += 1
            if len(transitions) < max_examples:
                transitions.append(
                    {
                        "step_index": idx,
                        "from": PHASES[prev],
                        "to": PHASES[current],
                    }
                )
            prev = current
    return {"total": total, "first_examples": transitions}


def get_eval(summary: Dict[str, object] | None) -> Dict[str, object] | None:
    if summary is None:
        return None
    eval_metrics = summary.get("eval")
    return eval_metrics if isinstance(eval_metrics, dict) else None


def improves_over(eval_metrics: Dict[str, object], reference_eval: Dict[str, object] | None) -> bool | None:
    if reference_eval is None:
        return None
    if bool(eval_metrics["success"]) != bool(reference_eval.get("success", False)):
        return bool(eval_metrics["success"])
    if bool(eval_metrics["crossing_occurs"]) != bool(reference_eval.get("crossing_occurs", False)):
        return bool(eval_metrics["crossing_occurs"])
    return float(eval_metrics["final_radius_error"]) < float(reference_eval.get("final_radius_error", float("inf")))


def save_plot(path: Path, trace: Dict[str, object], key: str, title: str, ylabel: str, target_value: float | None = None) -> None:
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


def save_phase_probs_plot(path: Path, trace: Dict[str, object]) -> None:
    probs = np.asarray(trace["phase_probs"], dtype=np.float64)
    plt.figure(figsize=(10, 4.8))
    for idx, phase in enumerate(PHASES):
        plt.plot(trace["step"], probs[:, idx], linewidth=1.2, label=phase)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("step")
    plt.ylabel("predicted phase probability")
    plt.title("Soft phase-conditioned IL | phase probabilities vs time")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_phase_argmax_plot(path: Path, trace: Dict[str, object]) -> None:
    plt.figure(figsize=(10, 4.8))
    plt.step(trace["step"], trace["phase_argmax"], where="post", linewidth=1.5)
    plt.yticks(range(len(PHASES)), PHASES)
    plt.xlabel("step")
    plt.ylabel("predicted phase argmax")
    plt.title("Soft phase-conditioned IL | phase argmax vs time")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def yes_no_unknown(value: bool | None) -> str:
    if value is None:
        return "Unknown"
    return "Yes" if value else "No"


def write_note(summary: Dict[str, object]) -> None:
    eval_metrics = summary["eval"]
    crossing = bool(eval_metrics["crossing_occurs"])
    success = bool(eval_metrics["success"])
    improves_hard = summary.get("improves_over_hard_phase_conditioned_il")
    improves_learned = summary.get("improves_over_learned_phase_il")

    if success:
        missing = "No remaining success gap was observed in this single deterministic rollout."
    elif crossing:
        missing = "Soft phase conditioning recovers at least one radius crossing, but the learned closed loop still lacks stable post-crossing capture and lock."
    else:
        missing = "The learned closed loop still fails to trigger a reliable descent-to-capture transition without oracle phase input."

    lines = [
        "# Soft Phase-Conditioned Imitation Learning Result",
        "",
        "## Main Answer",
        "",
        f"- Does soft phase conditioning recover crossing? `{'Yes' if crossing else 'No'}`",
        f"- Does it improve over hard phase conditioning? `{yes_no_unknown(improves_hard)}`",
        f"- Does it improve over learned_phase_il? `{yes_no_unknown(improves_learned)}`",
        f"- Does it achieve success? `{'Yes' if success else 'No'}`",
        f"- What remains missing if it still fails? {missing}",
        "",
        "## Interpretation",
        "",
        "- Training teacher-forces the action branch with ground-truth phase converted to one-hot probabilities.",
        "- Evaluation uses no oracle phase; the phase head softmax probabilities are fed directly into the action head.",
        "",
        "## Metrics",
        "",
        f"- crossing_occurs `{eval_metrics['crossing_occurs']}`",
        f"- radius_crossings_total `{eval_metrics['radius_crossings_total']}`",
        f"- first_crossing_step `{eval_metrics['first_crossing_step']}`",
        f"- success `{eval_metrics['success']}`",
        f"- final_radius_error `{eval_metrics['final_radius_error']:.3e}`",
        f"- predicted_phase_transitions `{summary['predicted_phase_transitions']['total']}`",
    ]
    for label, key in [
        ("Hard Phase-Conditioned Reference", "hard_phase_conditioned_reference_eval"),
        ("Learned-Phase Reference", "learned_phase_reference_eval"),
        ("History-Aware Reference", "history_aware_reference_eval"),
    ]:
        reference_eval = summary.get(key)
        if isinstance(reference_eval, dict):
            lines.extend(
                [
                    "",
                    f"## {label}",
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
    ensure_dir(MODEL_DIR)
    set_seed(SEED)

    states, actions, labels = collect_dataset()
    history_inputs = build_history_inputs(states)
    model, train_metrics = train_model(history_inputs, actions, labels)
    eval_metrics, trace = evaluate_model(model)
    phase_argmax = [int(x) for x in trace["phase_argmax"]]

    learned_phase_summary = load_summary(LEARNED_PHASE_SUMMARY_PATH)
    hard_phase_summary = load_summary(HARD_PHASE_SUMMARY_PATH)
    history_aware_summary = load_summary(HISTORY_AWARE_SUMMARY_PATH)
    learned_phase_eval = get_eval(learned_phase_summary)
    hard_phase_eval = get_eval(hard_phase_summary)
    history_aware_eval = get_eval(history_aware_summary)

    env = make_env()
    save_plot(RADIUS_PLOT_PATH, trace, "radius", "Soft phase-conditioned IL | radius vs time", "radius [m]", target_value=env.target_radius)
    save_plot(VR_PLOT_PATH, trace, "v_r", "Soft phase-conditioned IL | v_r vs time", "v_r [m/s]", target_value=0.0)
    save_phase_probs_plot(PHASE_PROBS_PLOT_PATH, trace)
    save_phase_argmax_plot(PHASE_ARGMAX_PLOT_PATH, trace)

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
        "input_dim": int(history_inputs.shape[1]),
        "state_dim": int(states.shape[1]),
        "phase_labels": PHASES,
        "action_phase_source_train": "ground_truth_phase_one_hot_teacher_forcing",
        "action_phase_source_eval": "predicted_phase_softmax_probabilities",
        "seed": SEED,
        "device": str(DEVICE),
        "num_samples": int(len(states)),
        "train": train_metrics,
        "eval": eval_metrics,
        "hard_phase_conditioned_reference_eval": hard_phase_eval,
        "learned_phase_reference_eval": learned_phase_eval,
        "history_aware_reference_eval": history_aware_eval,
        "improves_over_hard_phase_conditioned_il": improves_over(eval_metrics, hard_phase_eval),
        "improves_over_learned_phase_il": improves_over(eval_metrics, learned_phase_eval),
        "improves_over_history_aware_il": improves_over(eval_metrics, history_aware_eval),
        "predicted_phase_counts": phase_counts(phase_argmax),
        "predicted_phase_transitions": phase_transition_summary(phase_argmax),
        "predicted_phase_probabilities": trace["phase_probs"],
        "argmax_phase_sequence": phase_argmax,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_note(summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"predicted_phase_probabilities", "argmax_phase_sequence"}}, indent=2))
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")
    print(f"Saved plots to: {RADIUS_PLOT_PATH}, {VR_PLOT_PATH}, {PHASE_PROBS_PLOT_PATH}, {PHASE_ARGMAX_PLOT_PATH}")
    print(f"Saved note to: {NOTE_PATH}")


if __name__ == "__main__":
    main()
