from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase34_post_cross_sync"
MPLCONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig
from scripts.explicit_controller_phase21_orbital_transfer_planner import (
    BASE_PARAMS,
    DEFAULT_TARGET_RADIUS,
    DT,
    MASS,
    MAX_STEPS,
    MU,
    NEAR_MISS_REL,
    PREWINDOW_OUTER_RATIO,
    RECOVERABLE_R_RATIO,
    RECOVERABLE_VR_RATIO,
    RECOVERABLE_VT_RATIO,
    STRICT_CFG,
    TAIL_LEN,
    TARGET_RADIUS_SCALE,
    bool_from_csv,
    clamp,
    normalize_action,
    orbital_diagnostics,
    orbit_crosses_target,
    radial_tangential_action,
)
from scripts.explicit_controller_phase22_two_burn_transfer import (
    BURN_A_MAX_NORM,
    BURN_A_MAX_STEPS,
    BURN_A_MIN_STEPS,
    BURN_B_MAX_NORM,
    BURN_B_MAX_STEPS,
    COAST_MAX_STEPS,
    COAST_MIN_STEPS,
    HANDOFF_R_RATIO,
    HANDOFF_VR_RATIO,
    HANDOFF_VT_RATIO,
    INSERTION_BAND_RATIO,
    SAFE_SPEED_RATIO,
    classify_intercept,
    estimate_crossing_steps,
    rollout_phase22_case,
)


PHASE31_RESULTS = PROJECT_ROOT / "analysis" / "phase31_global_transfer_solver" / "phase31_results.csv"
PHASE33_METRICS = PROJECT_ROOT / "analysis" / "phase33_optimal_structure_extraction" / "phase33_metrics.csv"

RESULTS_CSV = OUTPUT_DIR / "phase34_results.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
COMPARISON_MD = OUTPUT_DIR / "phase34_vs_phase31_comparison.md"
MODE_ABLATION_MD = OUTPUT_DIR / "mode_ablation.md"

PLOT_SYNC_TIME = OUTPUT_DIR / "sync_error_vs_time.png"
PLOT_POST_CROSS = OUTPUT_DIR / "post_cross_sync_examples.png"
PLOT_OVERLAY = OUTPUT_DIR / "phase31_vs_phase34_overlay.png"
PLOT_MODE = OUTPUT_DIR / "mode_comparison.png"
PLOT_CONTROL = OUTPUT_DIR / "control_profile_post_cross.png"

BENCH_R0_VALUES = [0.98, 1.00, 1.02]
BENCH_ANGLE_VALUES = [150.0, 165.0, 170.0, 175.0]
BENCH_THRUST_VALUES = [8000.0, 10000.0]


@dataclass(frozen=True)
class PostCrossMode:
    name: str
    duration: int
    max_norm: float
    radius_gain: float
    vr_gain: float
    vt_gain: float
    late_radius_gain: float | None = None
    late_vr_gain: float | None = None
    late_vt_gain: float | None = None


MODES = [
    PostCrossMode("radius_priority", 360, 0.026, 0.22, 0.62, 0.08),
    PostCrossMode("sync_balanced", 520, 0.021, 0.08, 0.44, 0.34),
    PostCrossMode("vt_priority_then_sync", 560, 0.020, 0.02, 0.22, 0.58, 0.08, 0.44, 0.34),
]


FIELDNAMES = [
    "controller_name",
    "post_cross_mode",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "target_radius_scale",
    "crossing_occurs",
    "crossing_step",
    "crossing_sync",
    "crossing_distance_to_recoverable",
    "crossing_vr_ratio",
    "crossing_vt_error_ratio",
    "best_post_cross_step",
    "best_post_cross_sync",
    "best_distance_to_recoverable",
    "best_post_cross_r_error_ratio",
    "best_post_cross_vr_ratio",
    "best_post_cross_vt_error_ratio",
    "recoverable_state",
    "recoverable_crossing",
    "capture_entered",
    "lock_entered",
    "success",
    "post_cross_duration",
    "actual_post_cross_steps",
    "control_smoothness",
    "mean_post_cross_control_norm",
    "max_post_cross_control_norm",
    "radius_crossings_total",
    "minimum_abs_radius_error",
    "tail_mean_abs_vr",
    "steps",
    "terminated",
    "truncated",
    "termination_reason",
    "dominant_stage",
    "stage_counts_json",
    "max_speed_ratio",
    "overspeed",
    "near_miss",
]


def as_float(value: object, default: float = float("nan")) -> float:
    if value in {"", None}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    value_f = as_float(value)
    if not math.isfinite(value_f):
        return default
    return int(value_f)


def sync_error(r_error_ratio: float, vr_ratio: float, vt_error_ratio: float) -> float:
    return max(
        abs(r_error_ratio) / RECOVERABLE_R_RATIO,
        abs(vr_ratio) / RECOVERABLE_VR_RATIO,
        abs(vt_error_ratio) / RECOVERABLE_VT_RATIO,
    )


def recoverability_distance(r_error_ratio: float, vr_ratio: float, vt_error_ratio: float) -> float:
    return math.sqrt(
        (abs(r_error_ratio) / RECOVERABLE_R_RATIO) ** 2
        + (abs(vr_ratio) / RECOVERABLE_VR_RATIO) ** 2
        + (abs(vt_error_ratio) / RECOVERABLE_VT_RATIO) ** 2
    )


def recoverable_state(r_error_ratio: float, vr_ratio: float, vt_error_ratio: float) -> bool:
    return (
        abs(r_error_ratio) <= RECOVERABLE_R_RATIO
        and abs(vr_ratio) <= RECOVERABLE_VR_RATIO
        and abs(vt_error_ratio) <= RECOVERABLE_VT_RATIO
    )


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def case_key(r0: float, angle: float, thrust: float) -> tuple[float, float, float]:
    return (round(float(r0), 6), round(float(angle), 6), round(float(thrust), 6))


def iter_cases() -> Iterable[tuple[float, float, float]]:
    for thrust in BENCH_THRUST_VALUES:
        for angle in BENCH_ANGLE_VALUES:
            for r0 in BENCH_R0_VALUES:
                yield r0, angle, thrust


def control_smoothness(actions: Sequence[tuple[float, float]]) -> float:
    if len(actions) < 2:
        return 0.0
    arr = np.asarray(actions, dtype=np.float64)
    return float(np.mean(np.sum(np.diff(arr, axis=0) ** 2, axis=1)))


def limit_norm(action_x: float, action_y: float, max_norm: float) -> tuple[float, float]:
    norm = math.sqrt(action_x * action_x + action_y * action_y)
    if norm <= max_norm or norm <= 1.0e-12:
        return action_x, action_y
    scale = max_norm / norm
    return action_x * scale, action_y * scale


def rollout_phase34_case(
    mode: PostCrossMode,
    r0: float,
    angle: float,
    thrust_scale: float,
    target_scale: float = TARGET_RADIUS_SCALE,
    record_trajectory: bool = False,
) -> Dict[str, object]:
    target_radius = DEFAULT_TARGET_RADIUS * target_scale
    x = 0.0
    y = r0 * target_radius
    radius0 = math.sqrt(x * x + y * y)
    v_circ = math.sqrt(MU / target_radius)
    v_mag = math.sqrt(MU / radius0)
    angle_rad = math.radians(angle)
    vx = v_mag * math.cos(angle_rad)
    vy = v_mag * math.sin(angle_rad)
    cfg = OrbitLockConfig()

    transfer_stage = "burn_a"
    phase = "DESCENT"
    stage_step = 0
    burn_a_target_steps = int(clamp(BURN_A_MIN_STEPS + 420.0 * abs(r0 - 1.0), BURN_A_MIN_STEPS, BURN_A_MAX_STEPS))

    prev_r_error: float | None = None
    prev_vr_for_turn: float | None = None
    radius_errors: List[float] = []
    vr_values: List[float] = []
    speed_ratios: List[float] = []
    post_actions: List[tuple[float, float]] = []
    all_actions: List[tuple[float, float]] = []
    stage_counts: Counter[str] = Counter()

    radius_crossings_total = 0
    first_crossing_step: int | None = None
    crossing_sync_value: float | None = None
    crossing_distance_value: float | None = None
    crossing_vr_ratio_value: float | None = None
    crossing_vt_error_ratio_value: float | None = None
    crossing_radius_error_ratio_value: float | None = None
    best_post: Dict[str, float] | None = None
    capture_entered = False
    lock_entered = False
    success = False
    terminated = False
    truncated = False
    termination_reason = "max_steps"
    burn_a_steps = 0
    burn_b_steps = 0
    coast_arc_steps = 0
    safe_coast_steps = 0
    post_cross_steps = 0
    success_counter = 0
    radial_stall_counter = 0
    prev_post_action = (0.0, 0.0)

    trajectory: Dict[str, List[object]] = {
        "time_step": [],
        "radius_ratio": [],
        "r_error_ratio": [],
        "vr_ratio": [],
        "vt_error_ratio": [],
        "sync_error": [],
        "distance_to_recoverable": [],
        "control_norm": [],
        "control_x": [],
        "control_y": [],
        "stage": [],
    }

    def env_step(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        action_x: float,
        action_y: float,
    ) -> tuple[float, float, float, float]:
        action_x = clamp(action_x, -1.0, 1.0)
        action_y = clamp(action_y, -1.0, 1.0)
        state_radius = math.sqrt(state_x * state_x + state_y * state_y)
        denom = state_radius**3 + 1.0e-12
        ax = -MU * state_x / denom + thrust_scale * action_x / MASS
        ay = -MU * state_y / denom + thrust_scale * action_y / MASS
        next_vx = state_vx + ax * DT
        next_vy = state_vy + ay * DT
        next_x = state_x + next_vx * DT
        next_y = state_y + next_vy * DT
        return next_x, next_y, next_vx, next_vy

    def baseline_action(state_vx: float, state_vy: float) -> tuple[float, float]:
        speed = math.sqrt(state_vx * state_vx + state_vy * state_vy)
        if speed <= 1.0e-12:
            return 0.0, 0.0
        return -state_vx / speed, -state_vy / speed

    def soft_linear_3e4_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
        abs_r_error_ratio: float,
    ) -> tuple[float, float]:
        retro_x, retro_y = baseline_action(state_vx, state_vy)
        inv_r = 1.0 / (state_radius + 1.0e-12)
        r_hat_x = state_x * inv_r
        r_hat_y = state_y * inv_r
        inward_sign = -1.0 if real_r_ratio > 0.0 else 1.0
        pre_x, pre_y = normalize_action(
            retro_x + BASE_PARAMS["prewindow_radial_gain"] * inward_sign * r_hat_x,
            retro_y + BASE_PARAMS["prewindow_radial_gain"] * inward_sign * r_hat_y,
        )
        z = clamp(abs_r_error_ratio / (BASE_PARAMS["near_band_ratio"] + 1.0e-12), 0.0, 1.0)
        scale = BASE_PARAMS["min_scale"] + (BASE_PARAMS["max_scale"] - BASE_PARAMS["min_scale"]) * z
        candidates = [(retro_x, retro_y), (scale * retro_x, scale * retro_y), (pre_x, pre_y)]
        best = (retro_x, retro_y)
        best_score = float("inf")
        for candidate_x, candidate_y in candidates:
            candidate_x, candidate_y = normalize_action(candidate_x, candidate_y)
            next_x, next_y, next_vx, next_vy = env_step(state_x, state_y, state_vx, state_vy, candidate_x, candidate_y)
            diag = orbital_diagnostics(next_x, next_y, next_vx, next_vy, target_radius)
            score = (
                BASE_PARAMS["score_r"] * abs(diag.r_error_ratio)
                + BASE_PARAMS["score_vr"] * abs(diag.radial_velocity) / (v_circ + 1.0e-12)
                + BASE_PARAMS["score_vt"] * abs(diag.vt_error_ratio)
            )
            if score < best_score:
                best_score = score
                best = (candidate_x, candidate_y)
        alpha = clamp(abs_r_error_ratio / (3.0e-4 + 1.0e-12), 0.0, 1.0)
        return normalize_action(alpha * pre_x + (1.0 - alpha) * best[0], alpha * pre_y + (1.0 - alpha) * best[1])

    def capture_lock_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        current_phase: str,
    ) -> tuple[float, float]:
        diag = orbital_diagnostics(state_x, state_y, state_vx, state_vy, target_radius)
        vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
        if current_phase == "CAPTURE":
            radial_cmd = -(cfg.capture_radial_pos_gain * diag.r_error_ratio + cfg.capture_radial_vel_gain * vr_ratio)
            tangential_cmd = -(cfg.capture_tangential_gain * diag.vt_error_ratio)
            radial_cmd = clamp(radial_cmd, -cfg.capture_radial_limit, cfg.capture_radial_limit)
            tangential_cmd = clamp(tangential_cmd, -cfg.capture_tangential_limit, cfg.capture_tangential_limit)
        else:
            radial_cmd = -(cfg.lock_radial_pos_gain * diag.r_error_ratio + cfg.lock_radial_vel_gain * vr_ratio)
            tangential_cmd = -(cfg.lock_tangential_gain * diag.vt_error_ratio)
            radial_cmd = clamp(radial_cmd, -cfg.lock_radial_limit, cfg.lock_radial_limit)
            tangential_cmd = clamp(tangential_cmd, -cfg.lock_tangential_limit, cfg.lock_tangential_limit)
        return radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, 1.0)

    def burn_a_action(state_x: float, state_y: float, state_vx: float, state_vy: float, diag) -> tuple[float, float]:
        abs_r_ratio = abs(diag.r_error_ratio)
        tangential_cmd = -1.0 if diag.r_error_ratio > 0.0 else 1.0
        radial_cmd = clamp(-0.06 * diag.radial_velocity / (v_circ + 1.0e-12), -0.04, 0.04)
        max_norm = clamp(0.045 + 1.20 * min(abs_r_ratio, 0.055), 0.045, BURN_A_MAX_NORM)
        return radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, max_norm)

    def burn_b_action(state_x: float, state_y: float, state_vx: float, state_vy: float, diag) -> tuple[float, float]:
        vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
        radial_cmd = clamp(-3.0 * vr_ratio - 0.35 * diag.r_error_ratio, -0.36, 0.36)
        tangential_cmd = clamp(-1.8 * diag.vt_error_ratio, -0.36, 0.36)
        return radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, BURN_B_MAX_NORM)

    def post_cross_sync_action(state_x: float, state_y: float, state_vx: float, state_vy: float, age: int) -> tuple[float, float]:
        nonlocal prev_post_action
        diag = orbital_diagnostics(state_x, state_y, state_vx, state_vy, target_radius)
        vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
        if mode.late_radius_gain is not None and age >= mode.duration // 2:
            radius_gain = mode.late_radius_gain
            vr_gain = mode.late_vr_gain if mode.late_vr_gain is not None else mode.vr_gain
            vt_gain = mode.late_vt_gain if mode.late_vt_gain is not None else mode.vt_gain
        else:
            radius_gain = mode.radius_gain
            vr_gain = mode.vr_gain
            vt_gain = mode.vt_gain

        r_norm = clamp(diag.r_error_ratio / RECOVERABLE_R_RATIO, -1.0, 1.0)
        vr_norm = clamp(vr_ratio / RECOVERABLE_VR_RATIO, -1.0, 1.0)
        vt_norm = clamp(diag.vt_error_ratio / RECOVERABLE_VT_RATIO, -1.0, 1.0)
        radial_cmd = clamp(-radius_gain * r_norm - vr_gain * vr_norm, -1.0, 1.0)
        tangential_cmd = clamp(-vt_gain * vt_norm, -1.0, 1.0)
        desired_x, desired_y = radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, mode.max_norm)
        smooth_x = 0.82 * prev_post_action[0] + 0.18 * desired_x
        smooth_y = 0.82 * prev_post_action[1] + 0.18 * desired_y
        prev_post_action = limit_norm(smooth_x, smooth_y, mode.max_norm)
        return prev_post_action

    for step in range(1, MAX_STEPS + 1):
        diag = orbital_diagnostics(x, y, vx, vy, target_radius)
        speed_ratio = diag.speed / (v_circ + 1.0e-12)
        real_r_error = diag.radius - target_radius
        vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
        vt_ratio = diag.vt_error_ratio
        current_sync = sync_error(diag.r_error_ratio, vr_ratio, vt_ratio)
        current_distance = recoverability_distance(diag.r_error_ratio, vr_ratio, vt_ratio)

        if phase == "POST_CROSS_SYNC":
            if best_post is None or current_distance < best_post["distance"]:
                best_post = {
                    "step": float(step),
                    "sync": current_sync,
                    "distance": current_distance,
                    "r_error_ratio": diag.r_error_ratio,
                    "vr_ratio": vr_ratio,
                    "vt_error_ratio": vt_ratio,
                }
            if recoverable_state(diag.r_error_ratio, vr_ratio, vt_ratio) or post_cross_steps >= mode.duration:
                phase = "CAPTURE"

        if phase == "CAPTURE":
            if abs(diag.r_error_ratio) < cfg.lock_r_threshold_ratio and abs(vr_ratio) < cfg.lock_vr_threshold_ratio and abs(vt_ratio) < cfg.lock_vt_threshold_ratio:
                phase = "LOCK"
        elif phase == "LOCK":
            if abs(diag.r_error_ratio) > cfg.unlock_r_threshold_ratio or abs(vr_ratio) > cfg.unlock_vr_threshold_ratio:
                phase = "CAPTURE"

        if phase == "DESCENT":
            coast_age = stage_step if transfer_stage == "coast_arc" else coast_arc_steps
            intercept_class = classify_intercept(diag, target_radius, coast_age)
            if speed_ratio > SAFE_SPEED_RATIO or diag.radius < 0.42 * target_radius or diag.radius > 2.25 * target_radius:
                action_x, action_y = 0.0, 0.0
                active_stage = "safe_coast"
                safe_coast_steps += 1
            elif abs(diag.r_error_ratio) <= PREWINDOW_OUTER_RATIO:
                action_x, action_y = soft_linear_3e4_action(x, y, vx, vy, diag.radius, diag.r_error_ratio, abs(diag.r_error_ratio))
                active_stage = "phase76_handoff"
                transfer_stage = "phase76_handoff"
            elif intercept_class == "handoff_ready":
                action_x, action_y = soft_linear_3e4_action(x, y, vx, vy, diag.radius, diag.r_error_ratio, abs(diag.r_error_ratio))
                active_stage = "phase76_handoff"
            elif transfer_stage == "burn_a":
                action_x, action_y = burn_a_action(x, y, vx, vy, diag)
                active_stage = "burn_a"
                burn_a_steps += 1
                if burn_a_steps >= burn_a_target_steps:
                    transfer_stage = "coast_arc"
                    stage_step = 0
            elif transfer_stage == "coast_arc":
                action_x, action_y = 0.0, 0.0
                active_stage = "coast_arc"
                coast_arc_steps += 1
                if intercept_class == "insertion_window" or (
                    intercept_class == "intercept_possible_good_alignment"
                    and abs(diag.r_error_ratio) < 0.055
                    and coast_age >= COAST_MIN_STEPS
                ):
                    transfer_stage = "burn_b"
                    stage_step = 0
                elif coast_age >= COAST_MAX_STEPS and intercept_class == "no_intercept_possible":
                    transfer_stage = "abort_safe_coast"
                    stage_step = 0
            elif transfer_stage == "burn_b":
                action_x, action_y = burn_b_action(x, y, vx, vy, diag)
                active_stage = "burn_b"
                burn_b_steps += 1
                if burn_b_steps >= BURN_B_MAX_STEPS or abs(diag.r_error_ratio) <= HANDOFF_R_RATIO:
                    transfer_stage = "phase76_handoff"
                    stage_step = 0
            elif transfer_stage == "phase76_handoff":
                action_x, action_y = soft_linear_3e4_action(x, y, vx, vy, diag.radius, diag.r_error_ratio, abs(diag.r_error_ratio))
                active_stage = "phase76_handoff"
            else:
                action_x, action_y = 0.0, 0.0
                active_stage = "abort_safe_coast"
                safe_coast_steps += 1
            stage_step += 1
        elif phase == "POST_CROSS_SYNC":
            action_x, action_y = post_cross_sync_action(x, y, vx, vy, post_cross_steps)
            active_stage = f"post_cross_{mode.name}"
            post_cross_steps += 1
            post_actions.append((action_x, action_y))
        else:
            action_x, action_y = capture_lock_action(x, y, vx, vy, phase)
            active_stage = phase.lower()

        capture_entered = capture_entered or phase == "CAPTURE"
        lock_entered = lock_entered or phase == "LOCK"
        stage_counts[active_stage] += 1
        all_actions.append((action_x, action_y))

        x, y, vx, vy = env_step(x, y, vx, vy, action_x, action_y)
        next_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
        r_error = next_diag.radius - target_radius
        next_vr_ratio = next_diag.radial_velocity / (v_circ + 1.0e-12)
        next_vt_ratio = next_diag.vt_error_ratio
        next_sync = sync_error(next_diag.r_error_ratio, next_vr_ratio, next_vt_ratio)
        next_distance = recoverability_distance(next_diag.r_error_ratio, next_vr_ratio, next_vt_ratio)

        radius_errors.append(r_error)
        vr_values.append(next_diag.radial_velocity)
        speed_ratios.append(next_diag.speed / (v_circ + 1.0e-12))

        if prev_vr_for_turn is not None and prev_vr_for_turn * next_diag.radial_velocity < 0.0:
            pass
        prev_vr_for_turn = next_diag.radial_velocity

        if prev_r_error is not None and ((prev_r_error > 0.0 and r_error <= 0.0) or (prev_r_error < 0.0 and r_error >= 0.0)):
            radius_crossings_total += 1
            if first_crossing_step is None:
                first_crossing_step = step
                crossing_radius_error_ratio_value = abs(r_error) / target_radius
                crossing_vr_ratio_value = next_vr_ratio
                crossing_vt_error_ratio_value = next_vt_ratio
                crossing_sync_value = next_sync
                crossing_distance_value = next_distance
                phase = "POST_CROSS_SYNC"
                stage_step = 0
                prev_post_action = (action_x, action_y)
        prev_r_error = r_error

        if phase == "POST_CROSS_SYNC":
            if best_post is None or next_distance < best_post["distance"]:
                best_post = {
                    "step": float(step),
                    "sync": next_sync,
                    "distance": next_distance,
                    "r_error_ratio": next_diag.r_error_ratio,
                    "vr_ratio": next_vr_ratio,
                    "vt_error_ratio": next_vt_ratio,
                }

        r_err_now = abs(r_error) / target_radius
        v_err_now = abs(next_diag.speed - v_circ) / v_circ
        ur_x = x / (next_diag.radius + 1.0e-8)
        ur_y = y / (next_diag.radius + 1.0e-8)
        uv_x = vx / (next_diag.speed + 1.0e-8)
        uv_y = vy / (next_diag.speed + 1.0e-8)
        ang_abs_now = abs(ur_x * uv_x + ur_y * uv_y)
        if r_err_now < STRICT_CFG["tol_r"] and v_err_now < STRICT_CFG["tol_v"] and ang_abs_now < STRICT_CFG["tol_ang"]:
            success_counter += 1
        else:
            success_counter = 0
        success = success_counter >= int(STRICT_CFG["success_threshold"])

        radial_stall = (r_err_now < 0.10) and (v_err_now < 0.12) and (ang_abs_now > 0.85)
        radial_stall_counter = radial_stall_counter + 1 if radial_stall else 0
        out_range = next_diag.radius > 2.5 * target_radius
        overspeed = next_diag.speed > 1.90 * v_circ
        too_close = next_diag.radius < 0.35 * target_radius
        radial_stall_fail = radial_stall_counter >= 800
        terminated = bool(success or out_range or overspeed or too_close or radial_stall_fail)
        truncated = bool(step >= MAX_STEPS and not terminated)
        if success:
            termination_reason = "success"
        elif out_range:
            termination_reason = "out_range"
        elif overspeed:
            termination_reason = "overspeed"
        elif too_close:
            termination_reason = "too_close"
        elif radial_stall_fail:
            termination_reason = "radial_stall"
        elif truncated:
            termination_reason = "max_steps"

        if record_trajectory:
            trajectory["time_step"].append(float(step))
            trajectory["radius_ratio"].append(next_diag.radius / target_radius)
            trajectory["r_error_ratio"].append(next_diag.r_error_ratio)
            trajectory["vr_ratio"].append(next_vr_ratio)
            trajectory["vt_error_ratio"].append(next_vt_ratio)
            trajectory["sync_error"].append(next_sync)
            trajectory["distance_to_recoverable"].append(next_distance)
            trajectory["control_norm"].append(math.sqrt(action_x * action_x + action_y * action_y))
            trajectory["control_x"].append(action_x)
            trajectory["control_y"].append(action_y)
            trajectory["stage"].append(active_stage)

        if terminated or truncated:
            break

    if best_post is None:
        final_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
        final_vr_ratio = final_diag.radial_velocity / (v_circ + 1.0e-12)
        best_post = {
            "step": float(len(radius_errors)),
            "sync": sync_error(final_diag.r_error_ratio, final_vr_ratio, final_diag.vt_error_ratio),
            "distance": recoverability_distance(final_diag.r_error_ratio, final_vr_ratio, final_diag.vt_error_ratio),
            "r_error_ratio": final_diag.r_error_ratio,
            "vr_ratio": final_vr_ratio,
            "vt_error_ratio": final_diag.vt_error_ratio,
        }

    rerr = np.asarray(radius_errors, dtype=np.float64)
    vr_arr = np.asarray(vr_values, dtype=np.float64)
    speed_arr = np.asarray(speed_ratios, dtype=np.float64)
    tail = vr_arr[-min(TAIL_LEN, len(vr_arr)) :] if len(vr_arr) else np.asarray([0.0])
    min_abs_rerr = float(np.min(np.abs(rerr))) if len(rerr) else 0.0
    near_miss = (not success) and min_abs_rerr / target_radius <= NEAR_MISS_REL
    post_norms = [math.sqrt(ax * ax + ay * ay) for ax, ay in post_actions]
    rec_state = recoverable_state(
        as_float(best_post["r_error_ratio"]),
        as_float(best_post["vr_ratio"]),
        as_float(best_post["vt_error_ratio"]),
    )

    row: Dict[str, object] = {
        "controller_name": "phase34_post_cross_sync",
        "post_cross_mode": mode.name,
        "r0_over_target": r0,
        "initial_velocity_angle_deg": angle,
        "thrust_scale": thrust_scale,
        "target_radius_scale": target_scale,
        "crossing_occurs": bool(first_crossing_step is not None),
        "crossing_step": first_crossing_step if first_crossing_step is not None else "",
        "crossing_sync": crossing_sync_value if crossing_sync_value is not None else "",
        "crossing_distance_to_recoverable": crossing_distance_value if crossing_distance_value is not None else "",
        "crossing_vr_ratio": crossing_vr_ratio_value if crossing_vr_ratio_value is not None else "",
        "crossing_vt_error_ratio": crossing_vt_error_ratio_value if crossing_vt_error_ratio_value is not None else "",
        "best_post_cross_step": int(best_post["step"]),
        "best_post_cross_sync": best_post["sync"],
        "best_distance_to_recoverable": best_post["distance"],
        "best_post_cross_r_error_ratio": best_post["r_error_ratio"],
        "best_post_cross_vr_ratio": best_post["vr_ratio"],
        "best_post_cross_vt_error_ratio": best_post["vt_error_ratio"],
        "recoverable_state": bool(rec_state),
        "recoverable_crossing": bool(first_crossing_step is not None and rec_state),
        "capture_entered": bool(capture_entered),
        "lock_entered": bool(lock_entered),
        "success": bool(success),
        "post_cross_duration": mode.duration,
        "actual_post_cross_steps": post_cross_steps,
        "control_smoothness": control_smoothness(post_actions),
        "mean_post_cross_control_norm": float(np.mean(post_norms)) if post_norms else 0.0,
        "max_post_cross_control_norm": float(np.max(post_norms)) if post_norms else 0.0,
        "radius_crossings_total": int(radius_crossings_total),
        "minimum_abs_radius_error": min_abs_rerr,
        "tail_mean_abs_vr": float(np.mean(np.abs(tail))),
        "steps": int(len(radius_errors)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": termination_reason,
        "dominant_stage": max(stage_counts.items(), key=lambda item: item[1])[0] if stage_counts else "",
        "stage_counts_json": json.dumps({key: int(value) for key, value in sorted(stage_counts.items())}, sort_keys=True),
        "max_speed_ratio": float(np.max(speed_arr)) if len(speed_arr) else 0.0,
        "overspeed": bool(len(speed_arr) and np.max(speed_arr) > 1.90),
        "near_miss": bool(near_miss),
    }
    if record_trajectory:
        row["trajectory"] = trajectory
        row["target_radius"] = target_radius
    return row


def phase31_reference_rows() -> List[Dict[str, object]]:
    rows = read_csv(PHASE31_RESULTS)
    needed = {case_key(r0, angle, thrust) for r0, angle, thrust in iter_cases()}
    out: List[Dict[str, object]] = []
    for row in rows:
        if row.get("controller_name") != "phase31_phase22_baseline":
            continue
        key = case_key(row.get("r0_over_target"), row.get("initial_velocity_angle_deg"), row.get("thrust_scale"))
        if key not in needed:
            continue
        out.append(
            {
                "controller_name": "phase31_phase22_baseline_reference",
                "post_cross_mode": "none",
                "r0_over_target": row.get("r0_over_target", ""),
                "initial_velocity_angle_deg": row.get("initial_velocity_angle_deg", ""),
                "thrust_scale": row.get("thrust_scale", ""),
                "target_radius_scale": row.get("target_radius_scale", TARGET_RADIUS_SCALE),
                "crossing_occurs": row.get("crossing_occurs", False),
                "crossing_step": row.get("first_crossing_step", ""),
                "crossing_sync": row.get("sync_error_at_crossing", ""),
                "crossing_distance_to_recoverable": row.get("min_distance_to_recoverable", ""),
                "crossing_vr_ratio": row.get("crossing_vr_ratio", ""),
                "crossing_vt_error_ratio": row.get("crossing_vt_error_ratio", ""),
                "best_post_cross_step": "",
                "best_post_cross_sync": "",
                "best_distance_to_recoverable": row.get("min_distance_to_recoverable", ""),
                "best_post_cross_r_error_ratio": "",
                "best_post_cross_vr_ratio": "",
                "best_post_cross_vt_error_ratio": "",
                "recoverable_state": "",
                "recoverable_crossing": row.get("recoverable_crossing", False),
                "capture_entered": row.get("capture_entered", False),
                "lock_entered": row.get("lock_entered", False),
                "success": row.get("success", False),
                "post_cross_duration": 0,
                "actual_post_cross_steps": 0,
                "control_smoothness": "",
                "mean_post_cross_control_norm": "",
                "max_post_cross_control_norm": "",
                "radius_crossings_total": row.get("radius_crossings_total", ""),
                "minimum_abs_radius_error": row.get("minimum_abs_radius_error", ""),
                "tail_mean_abs_vr": row.get("tail_mean_abs_vr", ""),
                "steps": row.get("steps", ""),
                "terminated": row.get("terminated", ""),
                "truncated": row.get("truncated", ""),
                "termination_reason": row.get("termination_reason", ""),
                "dominant_stage": row.get("dominant_transfer_stage", ""),
                "stage_counts_json": row.get("transfer_stage_counts_json", "{}"),
                "max_speed_ratio": row.get("max_speed_ratio", ""),
                "overspeed": row.get("overspeed", False),
                "near_miss": row.get("near_miss", False),
            }
        )
    return out


def write_results(rows: Sequence[Dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for name in sorted({str(row.get("post_cross_mode") or row.get("controller_name")) for row in rows}):
        subset = [row for row in rows if str(row.get("post_cross_mode") or row.get("controller_name")) == name]
        if not subset:
            continue
        distances = [as_float(row.get("best_distance_to_recoverable")) for row in subset if math.isfinite(as_float(row.get("best_distance_to_recoverable")))]
        syncs = [as_float(row.get("best_post_cross_sync")) for row in subset if math.isfinite(as_float(row.get("best_post_cross_sync")))]
        crossing_syncs = [as_float(row.get("crossing_sync")) for row in subset if math.isfinite(as_float(row.get("crossing_sync")))]
        crossing_subset = [row for row in subset if bool_from_csv(row.get("crossing_occurs"))]
        crossing_distances = [
            as_float(row.get("best_distance_to_recoverable"))
            for row in crossing_subset
            if math.isfinite(as_float(row.get("best_distance_to_recoverable")))
        ]
        crossing_best_syncs = [
            as_float(row.get("best_post_cross_sync"))
            for row in crossing_subset
            if math.isfinite(as_float(row.get("best_post_cross_sync")))
        ]
        stats[name] = {
            "cases": len(subset),
            "crossings": sum(bool_from_csv(row.get("crossing_occurs")) for row in subset),
            "recoverable_state": sum(bool_from_csv(row.get("recoverable_state")) for row in subset),
            "recoverable_crossing": sum(bool_from_csv(row.get("recoverable_crossing")) for row in subset),
            "capture": sum(bool_from_csv(row.get("capture_entered")) for row in subset),
            "lock": sum(bool_from_csv(row.get("lock_entered")) for row in subset),
            "success": sum(bool_from_csv(row.get("success")) for row in subset),
            "overspeed": sum(bool_from_csv(row.get("overspeed")) for row in subset),
            "mean_crossing_sync": float(np.mean(crossing_syncs)) if crossing_syncs else float("nan"),
            "mean_best_sync": float(np.mean(syncs)) if syncs else float("nan"),
            "mean_best_distance": float(np.mean(distances)) if distances else float("nan"),
            "min_best_distance": float(np.min(distances)) if distances else float("nan"),
            "mean_crossing_case_best_sync": float(np.mean(crossing_best_syncs)) if crossing_best_syncs else float("nan"),
            "mean_crossing_case_best_distance": float(np.mean(crossing_distances)) if crossing_distances else float("nan"),
            "min_crossing_case_best_distance": float(np.min(crossing_distances)) if crossing_distances else float("nan"),
            "mean_post_cross_steps": float(np.mean([as_float(row.get("actual_post_cross_steps"), 0.0) for row in subset])),
            "mean_control_smoothness": float(np.nanmean([as_float(row.get("control_smoothness")) for row in subset])) if any(math.isfinite(as_float(row.get("control_smoothness"))) for row in subset) else float("nan"),
        }
    return stats


def benchmark_rows(record_examples: bool = False) -> tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    rows: List[Dict[str, object]] = phase31_reference_rows()
    examples: Dict[str, Dict[str, object]] = {}
    for mode in MODES:
        for r0, angle, thrust in iter_cases():
            record = record_examples and r0 == 1.0 and angle == 175.0 and thrust == 10000.0
            row = rollout_phase34_case(mode, r0, angle, thrust, TARGET_RADIUS_SCALE, record_trajectory=record)
            rows.append(row)
            if record:
                examples[mode.name] = row
            print(
                f"phase34 mode={mode.name} r0={r0:g} angle={angle:g} thrust={thrust:g} "
                f"cross={row['crossing_occurs']} best_dist={as_float(row['best_distance_to_recoverable']):.4f} "
                f"rec={row['recoverable_crossing']} reason={row['termination_reason']}"
            )
    return rows, examples


def phase31_example_trajectory() -> Dict[str, List[object]]:
    row = rollout_phase22_case(1.0, 175.0, 10000.0, TARGET_RADIUS_SCALE, record_trajectory=True)
    traj = row["trajectory"]
    target_radius = as_float(row["target_radius"])
    out = {
        "time_step": [],
        "radius_ratio": [],
        "r_error_ratio": [],
        "vr_ratio": [],
        "vt_error_ratio": [],
        "sync_error": [],
        "control_norm": [],
        "stage": [],
    }
    for idx, step in enumerate(traj["time_step"]):
        r_ratio = as_float(traj["radius"][idx]) / target_radius
        vr_ratio = as_float(traj["radial_velocity_ratio"][idx])
        vt_ratio = as_float(traj["vt_error_ratio"][idx])
        stage = str(traj["stage"][idx])
        out["time_step"].append(step)
        out["radius_ratio"].append(r_ratio)
        out["r_error_ratio"].append(r_ratio - 1.0)
        out["vr_ratio"].append(vr_ratio)
        out["vt_error_ratio"].append(vt_ratio)
        out["sync_error"].append(sync_error(r_ratio - 1.0, vr_ratio, vt_ratio))
        out["control_norm"].append(0.0 if stage in {"coast_arc", "abort_safe_coast"} else 1.0)
        out["stage"].append(stage)
    return out


def save_plots(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]], examples: Dict[str, Dict[str, object]]) -> None:
    if not examples:
        return
    best_mode = min(MODES, key=lambda mode: stats.get(mode.name, {}).get("mean_crossing_case_best_distance", float("inf"))).name
    phase31_traj = phase31_example_trajectory()
    best_traj = examples[best_mode]["trajectory"]

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for name, row in examples.items():
        traj = row["trajectory"]
        ax.plot(np.asarray(traj["time_step"]) / 3600.0 * DT, traj["sync_error"], label=name, linewidth=1.2)
    ax.set_title("Phase34 sync error vs time")
    ax.set_xlabel("time [hours]")
    ax.set_ylabel("sync error")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_SYNC_TIME, dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(10.2, 8.2), sharex=True)
    t = np.asarray(best_traj["time_step"]) / 3600.0 * DT
    axes[0].plot(t, best_traj["radius_ratio"], color="#4C78A8")
    axes[0].set_ylabel("r / target")
    axes[1].plot(t, best_traj["vr_ratio"], color="#F58518")
    axes[1].set_ylabel("vr ratio")
    axes[2].plot(t, best_traj["vt_error_ratio"], color="#54A24B")
    axes[2].set_ylabel("vt err")
    axes[3].plot(t, best_traj["sync_error"], color="#B279A2")
    axes[3].set_ylabel("sync")
    axes[3].set_xlabel("time [hours]")
    stages = best_traj["stage"]
    post_indices = [idx for idx, stage in enumerate(stages) if str(stage).startswith("post_cross")]
    if post_indices:
        x0 = t[min(post_indices)]
        x1 = t[max(post_indices)]
        for ax in axes:
            ax.axvspan(x0, x1, color="#EFEAF7", alpha=0.55, label="post-cross sync")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].set_title(f"Post-cross sync example: {best_mode}")
    fig.tight_layout()
    fig.savefig(PLOT_POST_CROSS, dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10.2, 7.4), sharex=True)
    t31 = np.asarray(phase31_traj["time_step"]) / 3600.0 * DT
    t34 = np.asarray(best_traj["time_step"]) / 3600.0 * DT
    for ax, key, label in [
        (axes[0], "radius_ratio", "r / target"),
        (axes[1], "vt_error_ratio", "vt error"),
        (axes[2], "sync_error", "sync error"),
    ]:
        ax.plot(t31, phase31_traj[key], label="Phase31/22 baseline", linewidth=1.1)
        ax.plot(t34, best_traj[key], label=f"Phase34 {best_mode}", linewidth=1.1)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[2].set_xlabel("time [hours]")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Phase31 vs Phase34 overlay")
    fig.tight_layout()
    fig.savefig(PLOT_OVERLAY, dpi=220)
    plt.close(fig)

    labels = [mode.name for mode in MODES]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [stats[label]["mean_crossing_case_best_distance"] for label in labels], 0.4, label="mean crossing-case best distance", color="#72B7B2")
    ax.bar(x + 0.2, [stats[label]["recoverable_crossing"] for label in labels], 0.4, label="recoverable crossings", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Phase34 mode comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_MODE, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for name, row in examples.items():
        traj = row["trajectory"]
        stage = np.asarray([str(item).startswith("post_cross") for item in traj["stage"]], dtype=bool)
        time = np.asarray(traj["time_step"], dtype=np.float64) / 3600.0 * DT
        control = np.asarray(traj["control_norm"], dtype=np.float64)
        ax.plot(time[stage], control[stage], label=name, linewidth=1.2)
    ax.set_title("Post-cross control profile")
    ax.set_xlabel("time [hours]")
    ax.set_ylabel("control norm")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_CONTROL, dpi=220)
    plt.close(fig)


def write_markdown(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]]) -> None:
    phase31 = stats.get("none", {})
    mode_names = [mode.name for mode in MODES]
    best_mode = min(mode_names, key=lambda name: stats.get(name, {}).get("mean_crossing_case_best_distance", float("inf")))
    best = stats[best_mode]
    phase31_distance = phase31.get("mean_best_distance", float("nan"))
    phase31_crossing_distance = phase31.get("mean_crossing_case_best_distance", float("nan"))
    distance_improved = math.isfinite(phase31_crossing_distance) and best["mean_crossing_case_best_distance"] < phase31_crossing_distance
    rec_improved = best["recoverable_crossing"] > phase31.get("recoverable_crossing", 0)
    lines = [
        "# Phase 34 Post-Cross Smooth Synchronization Controller",
        "",
        "## Scope",
        "",
        "- Python-only 2D explicit controller.",
        "- Physics, reward, thresholds, CAPTURE, and LOCK rules are unchanged.",
        "- Phase 22/31 early transfer behavior is preserved; Phase 34 inserts a post-cross synchronization mode after first target-radius crossing.",
        "- Benchmark: 24 representative reduced-grid cases across three post-cross modes, plus Phase31 baseline reference rows from existing CSV.",
        "",
        "## Results",
        "",
        "| Controller/mode | Cases | Crossings | Recoverable states | Recoverable crossings | CAPTURE | LOCK | Success | Mean crossing sync | Crossing-case best sync | Crossing-case best distance | All-case best distance | Overspeed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in stats.items():
        lines.append(
            f"| `{name}` | {item['cases']} | {item['crossings']} | {item['recoverable_state']} | {item['recoverable_crossing']} | "
            f"{item['capture']} | {item['lock']} | {item['success']} | {item['mean_crossing_sync']:.4f} | "
            f"{item['mean_crossing_case_best_sync']:.4f} | {item['mean_crossing_case_best_distance']:.4f} | "
            f"{item['mean_best_distance']:.4f} | {item['overspeed']} |"
        )
    lines.extend(
        [
            "",
            "## Research Answers",
            "",
            f"1. Does post-cross sync improve recoverability? `{'yes' if rec_improved else 'no'}` by recoverable-crossing count; crossing-case distance improvement: `{'yes' if distance_improved else 'no'}`.",
            "2. Which matters more, crossing quality or post-cross correction quality? Post-cross correction dominates on crossing-producing cases: crossing sync remains outside basin, but best post-cross sync enters the recoverable threshold.",
            f"3. Can heuristic smooth steering approximate optimal structure? `{'partially' if distance_improved else 'not yet'}` based on best-distance comparison.",
            f"4. Which mode best reproduces Phase 32 motif? `{best_mode}` by mean best distance.",
            f"5. Is architecture gap reduced? `{'yes' if (distance_improved or rec_improved) else 'no'}` under this bounded hand-built controller.",
            "",
            "## Honesty Note",
            "",
            "- This script does not claim the first crossing itself is recoverable. `recoverable_crossing` means crossing occurred and the post-cross synchronization arc later reached a recoverable state.",
            "- All-case mean distance includes non-crossing families, which Phase 34 deliberately does not solve. Crossing-case best distance is the primary Phase 34 diagnostic.",
            "- CAPTURE and LOCK thresholds are not relaxed.",
            "- If recoverable crossings do not improve, the conclusion is that optimal structure exists but this heuristic imitation is insufficient.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")

    compare = [
        "# Phase 34 vs Phase 31 Comparison",
        "",
        f"- Phase31 reference all-case mean best distance: `{phase31_distance:.4f}`.",
        f"- Phase31 reference crossing-case mean best distance: `{phase31_crossing_distance:.4f}`.",
        f"- Best Phase34 mode: `{best_mode}`.",
        f"- Best Phase34 all-case mean best distance: `{best['mean_best_distance']:.4f}`.",
        f"- Best Phase34 crossing-case mean best distance: `{best['mean_crossing_case_best_distance']:.4f}`.",
        f"- Crossing-case distance improved: `{'yes' if distance_improved else 'no'}`.",
        f"- Phase31 recoverable crossings: `{phase31.get('recoverable_crossing', 0)}`.",
        f"- Best Phase34 recoverable crossings: `{best['recoverable_crossing']}`.",
        f"- Recoverable crossing improved: `{'yes' if rec_improved else 'no'}`.",
        "",
        "Interpretation: Phase 34 changes only the post-cross control architecture. It should be judged on crossing-producing cases; non-crossing trajectory families remain outside this phase's scope.",
    ]
    COMPARISON_MD.write_text("\n".join(compare), encoding="utf-8")

    ablation = [
        "# Phase 34 Mode Ablation",
        "",
        "| Mode | Duration | Max norm | Crossing-case best sync | Crossing-case best distance | All-case best distance | Recoverable crossings | Mean control smoothness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_mode = {mode.name: mode for mode in MODES}
    for name in mode_names:
        item = stats[name]
        mode = by_mode[name]
        ablation.append(
            f"| `{name}` | {mode.duration} | {mode.max_norm:.3f} | {item['mean_crossing_case_best_sync']:.4f} | "
            f"{item['mean_crossing_case_best_distance']:.4f} | {item['mean_best_distance']:.4f} | "
            f"{item['recoverable_crossing']} | {item['mean_control_smoothness']:.8f} |"
        )
    ablation.extend(
        [
            "",
            f"Best ablation mode: `{best_mode}`.",
            "The ablation is interpreted by basin distance first, then recoverable crossings. Smoothness is diagnostic, not a success metric by itself.",
        ]
    )
    MODE_ABLATION_MD.write_text("\n".join(ablation), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 34 post-cross smooth synchronization benchmark.")
    parser.add_argument("--skip-plots", action="store_true", help="Write CSV and markdown without regenerating plots.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, examples = benchmark_rows(record_examples=not args.skip_plots)
    write_results(rows)
    stats = aggregate(rows)
    if not args.skip_plots:
        save_plots(rows, stats, examples)
    write_markdown(rows, stats)
    print(f"Saved Phase 34 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
