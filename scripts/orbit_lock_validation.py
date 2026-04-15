from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from envs.orbit_env import OrbitEnv
from scripts.day20_policy_surface import DEFAULT_CHECKPOINTS, LoadedPolicy


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figs" / "orbit_lock_validation"
CSV_PATH = OUTPUT_DIR / "aggregate_results.csv"
JSON_PATH = OUTPUT_DIR / "aggregate_results.json"
SUMMARY_PATH = PROJECT_ROOT / "analysis" / "orbit_lock_validation.md"
PHASE_SUMMARY_PATH = PROJECT_ROOT / "analysis" / "orbit_lock_phase_controller.md"

DT = 50.0
MAX_STEPS = 100000
R0_OVER_TARGET = 1.00005
THRUST_SCALE = 20000.0
STRICT_CFG = {
    "tol_r": 1.0e-3,
    "tol_v": 1.0e-3,
    "tol_ang": 0.02,
    "success_threshold": 200,
}


@dataclass
class ValidationRow:
    controller: str
    success_at_start: bool
    success: bool
    terminated: bool
    truncated: bool
    radius_crossings_total: int
    tail_crosses_target_radius: bool
    first_crossing_step: int | None
    final_radius_error: float
    tail_mean_abs_vr: float
    tail_radius_std: float
    tail_speed_cv: float
    amplitude_shrinks: bool
    mean_action_norm: float
    min_abs_radius_error: float
    sustained_crossing_score: int
    phase_transition_count: int


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


def set_default_start(env: OrbitEnv, r0_over_target: float) -> None:
    env.reset(start_mode="default")
    env.steps = 0
    env.success_counter = 0
    env.radial_stall_counter = 0
    env.orbit_lock_counter = 0
    env.prev_action = np.zeros(2, dtype=np.float64)
    env.pos = np.array([0.0, r0_over_target * env.target_radius], dtype=np.float64)
    v_mag = math.sqrt(env.mu / np.linalg.norm(env.pos))
    angle = math.radians(170.0)
    env.vel = v_mag * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)


class MaxRetrogradeController:
    def act(self, obs: Sequence[float]) -> List[float]:
        vel = np.asarray(obs[2:4], dtype=np.float64)
        v = np.linalg.norm(vel)
        if v < 1e-12:
            return [0.0, 0.0]
        action = -(vel / v)
        return [float(action[0]), float(action[1])]


def tail_slice(length: int, tail_len: int = 5000) -> slice:
    use_len = min(tail_len, length)
    return slice(length - use_len, length)


def rollout(controller) -> tuple[ValidationRow, Dict[str, List[float]]]:
    env = make_env()
    set_default_start(env, R0_OVER_TARGET)
    success_at_start = env._inside_tolerance(env.pos, env.vel)
    obs = env._get_obs()

    trace: Dict[str, List[float]] = {
        "step": [],
        "radius": [],
        "r_error": [],
        "v_r": [],
        "speed": [],
        "action_norm": [],
    }

    success = False
    terminated = False
    truncated = False

    for step in range(MAX_STEPS):
        action = np.asarray(controller.act(obs), dtype=np.float64)
        obs, _, terminated, truncated, info = env.step(action)

        radius = float(np.linalg.norm(env.pos))
        speed = float(np.linalg.norm(env.vel))
        r_hat = env.pos / (radius + 1e-12)
        v_r = float(np.dot(env.vel, r_hat))

        trace["step"].append(float(step))
        trace["radius"].append(radius)
        trace["r_error"].append(radius - env.target_radius)
        trace["v_r"].append(v_r)
        trace["speed"].append(speed)
        trace["action_norm"].append(float(np.linalg.norm(np.asarray(info["action_executed"], dtype=np.float64))))
        success = success or bool(info.get("success", False))
        if terminated or truncated:
            break

    radius_arr = np.asarray(trace["radius"], dtype=np.float64)
    speed_arr = np.asarray(trace["speed"], dtype=np.float64)
    vr_arr = np.asarray(trace["v_r"], dtype=np.float64)
    action_arr = np.asarray(trace["action_norm"], dtype=np.float64)
    re = radius_arr - env.target_radius
    sign_flip_mask = np.signbit(re[1:]) != np.signbit(re[:-1]) if len(re) > 1 else np.zeros(0, dtype=bool)
    crossing_indices = np.flatnonzero(sign_flip_mask)
    first_crossing_step = int(crossing_indices[0] + 1) if len(crossing_indices) else None

    ts = tail_slice(len(radius_arr))
    re_tail = re[ts]
    vr_tail = vr_arr[ts]
    speed_tail = speed_arr[ts]
    tail_cross_mask = np.signbit(re_tail[1:]) != np.signbit(re_tail[:-1]) if len(re_tail) > 1 else np.zeros(0, dtype=bool)
    first_half = re[: max(1, len(re) // 2)]
    second_half = re[len(re) // 2 :]

    row = ValidationRow(
        controller=type(controller).__name__.replace("Controller", "").lower(),
        success_at_start=bool(success_at_start),
        success=bool(success),
        terminated=bool(terminated),
        truncated=bool(truncated),
        radius_crossings_total=int(np.sum(sign_flip_mask)),
        tail_crosses_target_radius=bool(np.any(tail_cross_mask)),
        first_crossing_step=first_crossing_step,
        final_radius_error=float(abs(re[-1])) if len(re) else 0.0,
        tail_mean_abs_vr=float(np.mean(np.abs(vr_tail))) if len(vr_tail) else 0.0,
        tail_radius_std=float(np.std(radius_arr[ts])) if len(radius_arr) else 0.0,
        tail_speed_cv=float(np.std(speed_tail) / (np.mean(speed_tail) + 1e-12)) if len(speed_tail) else 0.0,
        amplitude_shrinks=bool(np.std(second_half) < np.std(first_half)) if len(re) >= 20 else False,
        mean_action_norm=float(np.mean(action_arr)) if len(action_arr) else 0.0,
        min_abs_radius_error=float(np.min(np.abs(re))) if len(re) else 0.0,
        sustained_crossing_score=int(np.sum(tail_cross_mask)),
        phase_transition_count=int(len(getattr(controller, "phase_transitions", []))),
    )
    return row, trace


def save_timeseries_plot(path: Path, title: str, ylabel: str, traces: Dict[str, Dict[str, List[float]]], key: str, target_value: float | None = None) -> None:
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


def write_csv(path: Path, rows: List[ValidationRow]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(rows: List[ValidationRow]) -> None:
    probe = next(row for row in rows if row.controller == "probe_max_retrograde")
    explicit = next(row for row in rows if row.controller == "explicit_orbit_lock")
    ppo = next((row for row in rows if row.controller == "ppo_speed_refine_50"), None)

    best = max(rows, key=lambda row: (row.sustained_crossing_score, row.tail_crosses_target_radius, row.radius_crossings_total, -row.tail_mean_abs_vr, row.amplitude_shrinks))
    any_lock = any((row.radius_crossings_total >= 2 and row.tail_crosses_target_radius) for row in rows)

    if any_lock:
        diagnosis = "At least one controller maintains target crossings into the tail, so orbit-lock-like behavior is present."
    else:
        diagnosis = "No controller maintains tail crossings. The failure is not reachability anymore; it is post-crossing state regulation."

    if probe.radius_crossings_total > 0 and explicit.radius_crossings_total == 0:
        missing_behavior = "The explicit controller is missing enough inward energy-removal authority to reach the first crossing under the new baseline."
    elif explicit.radius_crossings_total > 0 and not explicit.tail_crosses_target_radius:
        missing_behavior = "The explicit controller can cross but does not keep a phase-aware corrective cycle after crossing."
    else:
        missing_behavior = "The controllers do not sustain phase-aware control after the first crossing, so the trajectory falls back into one-sided drift or a single-pass transit."

    lines = [
        "# Orbit Lock Validation",
        "",
        "## Baseline Setup",
        "",
        f"- `dt = {int(DT)}`",
        f"- `max_steps = {MAX_STEPS}`",
        f"- `r0_over_target = {R0_OVER_TARGET:.5f}`",
        f"- `thrust_scale = {THRUST_SCALE}`",
        "- Strict tolerances:",
        f"  - `tol_r = {STRICT_CFG['tol_r']}`",
        f"  - `tol_v = {STRICT_CFG['tol_v']}`",
        f"  - `tol_ang = {STRICT_CFG['tol_ang']}`",
        "",
        "## Controllers",
        "",
        "- `probe_max_retrograde`",
        "- `explicit_orbit_lock`",
    ]
    if ppo is not None:
        lines.append("- `ppo_speed_refine_50`")

    lines.extend([
        "",
        "## Results",
        "",
    ])
    for row in rows:
        lines.append(
            f"- `{row.controller}`: crossings `{row.radius_crossings_total}`, tail_crosses `{row.tail_crosses_target_radius}`, "
            f"first_crossing_step `{row.first_crossing_step}`, tail_mean_abs_vr `{row.tail_mean_abs_vr:.3f}`, "
            f"amplitude_shrinks `{row.amplitude_shrinks}`, final_radius_error `{row.final_radius_error:.3e}`"
        )

    lines.extend([
        "",
        "## Main Answers",
        "",
        f"- Can any controller achieve sustained orbit lock? `{'Yes' if any_lock else 'No'}`",
        f"- Best controller under this setup: `{best.controller}`",
        f"- What prevents stability after crossing: {missing_behavior}",
        f"- What control behavior is missing: {missing_behavior}",
        "",
        "## Diagnosis",
        "",
        f"- {diagnosis}",
        f"- Probe crossing count: `{probe.radius_crossings_total}`",
        f"- Explicit crossing count: `{explicit.radius_crossings_total}`",
    ])
    if ppo is not None:
        lines.append(f"- PPO crossing count: `{ppo.radius_crossings_total}`")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    phase_lines = [
        "# Orbit Lock Phase Controller",
        "",
        "## Baseline Setup",
        "",
        f"- `dt = {int(DT)}`",
        f"- `max_steps = {MAX_STEPS}`",
        f"- `r0_over_target = {R0_OVER_TARGET:.5f}`",
        "",
        "## Main Answers",
        "",
        f"- Does the phase controller produce repeated crossings? `{'Yes' if explicit.radius_crossings_total >= 2 else 'No'}`",
        f"- Does it stabilize after crossing? `{'Yes' if explicit.tail_crosses_target_radius and explicit.amplitude_shrinks else 'No'}`",
        f"- What phase transition behavior is critical? `DESCENT -> CAPTURE` must occur early enough to cross, and `CAPTURE -> LOCK` only matters after radial damping and tangential support are both active near the target.`",
        "",
        "## Explicit Controller Metrics",
        "",
        f"- crossings `{explicit.radius_crossings_total}`",
        f"- tail_crosses_target_radius `{explicit.tail_crosses_target_radius}`",
        f"- sustained_crossing_score `{explicit.sustained_crossing_score}`",
        f"- tail_mean_abs_vr `{explicit.tail_mean_abs_vr:.3f}`",
        f"- phase_transition_count `{explicit.phase_transition_count}`",
        "",
        "## Diagnosis",
        "",
        f"- {missing_behavior}",
    ]
    PHASE_SUMMARY_PATH.write_text("\n".join(phase_lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    controllers = {
        "probe_max_retrograde": MaxRetrogradeController(),
        "explicit_orbit_lock": OrbitLockController(target_radius=make_env().target_radius, mu=make_env().mu, config=OrbitLockConfig()),
        "ppo_speed_refine_50": LoadedPolicy(DEFAULT_CHECKPOINTS["speed_refine_50"]),
    }

    rows: List[ValidationRow] = []
    traces: Dict[str, Dict[str, List[float]]] = {}
    for label, controller in controllers.items():
        row, trace = rollout(controller)
        row.controller = label
        rows.append(row)
        traces[label] = trace
        print(json.dumps(asdict(row)))

    write_csv(CSV_PATH, rows)
    JSON_PATH.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    save_timeseries_plot(OUTPUT_DIR / "radius_vs_time.png", "Orbit lock validation | radius", "radius [m]", traces, "radius", target_value=make_env().target_radius)
    save_timeseries_plot(OUTPUT_DIR / "v_r_vs_time.png", "Orbit lock validation | v_r", "v_r [m/s]", traces, "v_r", target_value=0.0)
    save_timeseries_plot(OUTPUT_DIR / "action_norm_vs_time.png", "Orbit lock validation | action norm", "action_norm", traces, "action_norm", target_value=0.0)
    write_summary(rows)
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
