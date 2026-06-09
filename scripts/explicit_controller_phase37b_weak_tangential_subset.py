from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase37b_weak_tangential_subset"
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
    PREWINDOW_OUTER_RATIO,
    RECOVERABLE_R_RATIO,
    RECOVERABLE_VR_RATIO,
    RECOVERABLE_VT_RATIO,
    STRICT_CFG,
    TARGET_RADIUS_SCALE,
    bool_from_csv,
    clamp,
    normalize_action,
    orbital_diagnostics,
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
    SAFE_SPEED_RATIO,
    classify_intercept,
)
from scripts.explicit_controller_phase34_post_cross_sync import (
    MODES as PHASE34_MODES,
    PostCrossMode,
    as_float,
    recoverability_distance,
    recoverable_state,
    sync_error,
)
from scripts.explicit_controller_phase35_crossing_basin_expansion import crossing_potential


RESULTS_CSV = OUTPUT_DIR / "phase37b_results.csv"
SUMMARY_MD = OUTPUT_DIR / "phase37b_summary.md"
COMPARISON_PNG = OUTPUT_DIR / "phase37b_comparison.png"
PHASE36B_RESULTS_CSV = PROJECT_ROOT / "analysis" / "phase36b_transfer_family_benchmark" / "phase36b_results.csv"

PHASE34_TERMINAL_MODE = next(mode for mode in PHASE34_MODES if mode.name == "radius_priority")

SELECTED_CASES = [
    (1.02, 150.0, 10000.0),
    (1.02, 165.0, 10000.0),
    (1.02, 170.0, 10000.0),
    (1.02, 175.0, 10000.0),
]

RADIAL_LIMIT = 0.055
WEAK_TANGENTIAL_GAIN = 0.06
WEAK_TANGENTIAL_LIMIT = 0.020
PHASE37B_ACTION_MAX_NORM = 0.060


@dataclass(frozen=True)
class Phase37BSetting:
    name: str
    weak_tangential: bool


SETTINGS = [
    Phase37BSetting("early_commit_low_radial_only", False),
    Phase37BSetting("early_commit_low_plus_weak_tangential", True),
]

FIELDNAMES = [
    "case_id",
    "group",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "setting",
    "crossing_occurs",
    "phase34_compatible_crossing",
    "recoverable_crossing",
    "simulator_success_label",
    "min_abs_radius_error_ratio",
    "closest_approach_step",
    "crossing_vr_ratio",
    "crossing_vt_error_ratio",
    "crossing_sync_error",
    "best_post_cross_distance",
    "overspeed",
    "instability",
    "failure_label",
]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def case_key(r0: object, angle: object, thrust: object) -> tuple[float, float, float]:
    return (round(as_float(r0), 8), round(as_float(angle), 8), round(as_float(thrust), 8))


def selected_case_keys() -> set[tuple[float, float, float]]:
    return {case_key(r0, angle, thrust) for r0, angle, thrust in SELECTED_CASES}


def load_phase36b_regression_crossing_cases() -> list[tuple[float, float, float]]:
    if not PHASE36B_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing Phase36B baseline source: {PHASE36B_RESULTS_CSV}")
    cases: list[tuple[float, float, float]] = []
    with PHASE36B_RESULTS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("transfer_family") != "baseline_phase34":
                continue
            if not bool_from_csv(row.get("crossing_occurs")):
                continue
            cases.append(
                (
                    as_float(row["r0_over_target"]),
                    as_float(row["initial_velocity_angle_deg"]),
                    as_float(row["thrust_scale"]),
                )
            )
    cases = sorted(set(cases), key=lambda item: (item[2], item[1], item[0]))
    if len(cases) != 8:
        raise ValueError(f"Expected 8 Phase36B baseline crossing cases, found {len(cases)}")
    return cases


def iter_phase37b_cases() -> Iterable[tuple[str, str, float, float, float]]:
    for index, (r0, angle, thrust) in enumerate(SELECTED_CASES, start=1):
        yield f"selected_{index:02d}", "selected_non_crossing", r0, angle, thrust
    for index, (r0, angle, thrust) in enumerate(load_phase36b_regression_crossing_cases(), start=1):
        yield f"regression_{index:02d}", "regression_crossing", r0, angle, thrust


def limit_norm(action_x: float, action_y: float, max_norm: float) -> tuple[float, float]:
    norm = math.sqrt(action_x * action_x + action_y * action_y)
    if norm <= max_norm or norm <= 1.0e-12:
        return clamp(action_x, -1.0, 1.0), clamp(action_y, -1.0, 1.0)
    scale = max_norm / norm
    return clamp(action_x * scale, -1.0, 1.0), clamp(action_y * scale, -1.0, 1.0)


def soft_success_check(x: float, y: float, vx: float, vy: float, target_radius: float, v_circ: float) -> bool:
    radius = math.sqrt(x * x + y * y)
    speed = math.sqrt(vx * vx + vy * vy)
    r_err = abs(radius - target_radius) / target_radius
    v_err = abs(speed - v_circ) / v_circ
    ur_x = x / (radius + 1.0e-8)
    ur_y = y / (radius + 1.0e-8)
    uv_x = vx / (speed + 1.0e-8)
    uv_y = vy / (speed + 1.0e-8)
    ang_abs = abs(ur_x * uv_x + ur_y * uv_y)
    return r_err < STRICT_CFG["tol_r"] and v_err < STRICT_CFG["tol_v"] and ang_abs < STRICT_CFG["tol_ang"]


def rollout_case(
    case_id: str,
    group: str,
    setting: Phase37BSetting,
    mode: PostCrossMode,
    r0: float,
    angle: float,
    thrust_scale: float,
    target_scale: float = TARGET_RADIUS_SCALE,
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
    first_crossing_step: int | None = None
    crossing_vr_ratio: float | None = None
    crossing_vt_error_ratio: float | None = None
    crossing_sync_value: float | None = None
    best_post: Dict[str, float] | None = None
    closest: Dict[str, float] | None = None
    best_potential = -1.0

    radius_errors: List[float] = []
    speed_ratios: List[float] = []
    stage_counts: Counter[str] = Counter()
    post_actions: List[tuple[float, float]] = []

    success = False
    termination_reason = "max_steps"
    instability = False
    burn_a_steps = 0
    burn_b_steps = 0
    coast_arc_steps = 0
    safe_coast_steps = 0
    post_cross_steps = 0
    success_counter = 0
    radial_stall_counter = 0
    prev_post_action = (0.0, 0.0)
    key = case_key(r0, angle, thrust_scale)
    selected_keys = selected_case_keys()

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

    def capture_lock_action(state_x: float, state_y: float, state_vx: float, state_vy: float, current_phase: str) -> tuple[float, float]:
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

    def post_cross_sync_action(state_x: float, state_y: float, state_vx: float, state_vy: float, age: int) -> tuple[float, float]:
        nonlocal prev_post_action
        diag = orbital_diagnostics(state_x, state_y, state_vx, state_vy, target_radius)
        vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
        radius_gain = mode.radius_gain
        vr_gain = mode.vr_gain
        vt_gain = mode.vt_gain
        if mode.late_radius_gain is not None and age >= mode.duration // 2:
            radius_gain = mode.late_radius_gain
            vr_gain = mode.late_vr_gain if mode.late_vr_gain is not None else mode.vr_gain
            vt_gain = mode.late_vt_gain if mode.late_vt_gain is not None else mode.vt_gain
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

    def apply_phase37b_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        diag,
        base_action: tuple[float, float],
        active_stage: str,
    ) -> tuple[tuple[float, float], str]:
        vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
        direction_to_target = -1.0 if diag.r_error_ratio > 0.0 else 1.0
        radial_cmd = clamp(RADIAL_LIMIT * direction_to_target - 0.16 * vr_ratio, -RADIAL_LIMIT, RADIAL_LIMIT)
        weak_allowed = (
            setting.weak_tangential
            and key in selected_keys
            and abs(diag.r_error_ratio) > PREWINDOW_OUTER_RATIO
        )
        tangential_cmd = clamp(-WEAK_TANGENTIAL_GAIN * diag.vt_error_ratio, -WEAK_TANGENTIAL_LIMIT, WEAK_TANGENTIAL_LIMIT) if weak_allowed else 0.0
        overlay_x, overlay_y = radial_tangential_action(
            state_x,
            state_y,
            state_vx,
            state_vy,
            radial_cmd,
            tangential_cmd,
            PHASE37B_ACTION_MAX_NORM,
        )
        action_x, action_y = limit_norm(
            base_action[0] + overlay_x,
            base_action[1] + overlay_y,
            PHASE37B_ACTION_MAX_NORM,
        )
        stage = f"{setting.name}_active" if weak_allowed or setting.name == "early_commit_low_radial_only" else f"{active_stage}_gated"
        return (action_x, action_y), stage

    for step in range(1, MAX_STEPS + 1):
        diag = orbital_diagnostics(x, y, vx, vy, target_radius)
        speed_ratio = diag.speed / (v_circ + 1.0e-12)
        vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
        vt_ratio = diag.vt_error_ratio
        current_sync = sync_error(diag.r_error_ratio, vr_ratio, vt_ratio)
        current_distance = recoverability_distance(diag.r_error_ratio, vr_ratio, vt_ratio)
        potential = crossing_potential(diag, target_radius, v_circ)
        best_potential = max(best_potential, potential)

        abs_r_ratio = abs(diag.r_error_ratio)
        if closest is None or abs_r_ratio < closest["abs_r_ratio"]:
            closest = {
                "step": float(step),
                "abs_r_ratio": abs_r_ratio,
                "r_error_ratio": diag.r_error_ratio,
                "vr_ratio": vr_ratio,
                "vt_error_ratio": vt_ratio,
                "energy_error_ratio": diag.energy_error_ratio,
                "angular_momentum_error_ratio": diag.angular_momentum_error_ratio,
            }

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
            (action_x, action_y), active_stage = apply_phase37b_action(x, y, vx, vy, diag, (action_x, action_y), active_stage)
            stage_step += 1
        elif phase == "POST_CROSS_SYNC":
            action_x, action_y = post_cross_sync_action(x, y, vx, vy, post_cross_steps)
            active_stage = f"post_cross_{mode.name}"
            post_cross_steps += 1
            post_actions.append((action_x, action_y))
        else:
            action_x, action_y = capture_lock_action(x, y, vx, vy, phase)
            active_stage = phase.lower()

        stage_counts[active_stage] += 1

        x, y, vx, vy = env_step(x, y, vx, vy, action_x, action_y)
        next_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
        r_error = next_diag.radius - target_radius
        next_vr_ratio = next_diag.radial_velocity / (v_circ + 1.0e-12)
        next_vt_ratio = next_diag.vt_error_ratio
        next_sync = sync_error(next_diag.r_error_ratio, next_vr_ratio, next_vt_ratio)
        next_distance = recoverability_distance(next_diag.r_error_ratio, next_vr_ratio, next_vt_ratio)

        radius_errors.append(r_error)
        speed_ratios.append(next_diag.speed / (v_circ + 1.0e-12))

        if prev_r_error is not None and ((prev_r_error > 0.0 and r_error <= 0.0) or (prev_r_error < 0.0 and r_error >= 0.0)):
            if first_crossing_step is None:
                first_crossing_step = step
                crossing_vr_ratio = next_vr_ratio
                crossing_vt_error_ratio = next_vt_ratio
                crossing_sync_value = next_sync
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
                    "vt_error_ratio": next_diag.vt_error_ratio,
                }

        if soft_success_check(x, y, vx, vy, target_radius, v_circ):
            success_counter += 1
        else:
            success_counter = 0
        success = success_counter >= int(STRICT_CFG["success_threshold"])

        r_err_now = abs(r_error) / target_radius
        v_err_now = abs(next_diag.speed - v_circ) / v_circ
        radius = next_diag.radius
        speed = next_diag.speed
        ur_x = x / (radius + 1.0e-8)
        ur_y = y / (radius + 1.0e-8)
        uv_x = vx / (speed + 1.0e-8)
        uv_y = vy / (speed + 1.0e-8)
        ang_abs_now = abs(ur_x * uv_x + ur_y * uv_y)
        radial_stall = (r_err_now < 0.10) and (v_err_now < 0.12) and (ang_abs_now > 0.85)
        radial_stall_counter = radial_stall_counter + 1 if radial_stall else 0
        out_range = next_diag.radius > 2.5 * target_radius
        overspeed_now = next_diag.speed > 1.90 * v_circ
        too_close = next_diag.radius < 0.35 * target_radius
        radial_stall_fail = radial_stall_counter >= 800
        terminated = bool((success and first_crossing_step is not None) or out_range or overspeed_now or too_close or radial_stall_fail)
        truncated = bool(step >= MAX_STEPS and not terminated)
        if success and first_crossing_step is not None:
            termination_reason = "simulator_success_label"
        elif out_range:
            termination_reason = "out_range"
            instability = True
        elif overspeed_now:
            termination_reason = "overspeed"
        elif too_close:
            termination_reason = "too_close"
            instability = True
        elif radial_stall_fail:
            termination_reason = "radial_stall"
            instability = True
        elif truncated:
            termination_reason = "max_steps"

        if terminated or truncated:
            break

    final_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
    final_vr_ratio = final_diag.radial_velocity / (v_circ + 1.0e-12)
    if best_post is None:
        best_post = {
            "step": float(len(radius_errors)),
            "sync": sync_error(final_diag.r_error_ratio, final_vr_ratio, final_diag.vt_error_ratio),
            "distance": recoverability_distance(final_diag.r_error_ratio, final_vr_ratio, final_diag.vt_error_ratio),
            "r_error_ratio": final_diag.r_error_ratio,
            "vr_ratio": final_vr_ratio,
            "vt_error_ratio": final_diag.vt_error_ratio,
        }
    if closest is None:
        closest = {
            "step": 0.0,
            "abs_r_ratio": abs(final_diag.r_error_ratio),
            "r_error_ratio": final_diag.r_error_ratio,
            "vr_ratio": final_vr_ratio,
            "vt_error_ratio": final_diag.vt_error_ratio,
            "energy_error_ratio": final_diag.energy_error_ratio,
            "angular_momentum_error_ratio": final_diag.angular_momentum_error_ratio,
        }

    speed_arr = np.asarray(speed_ratios, dtype=np.float64)
    rec_state = recoverable_state(
        as_float(best_post["r_error_ratio"]),
        as_float(best_post["vr_ratio"]),
        as_float(best_post["vt_error_ratio"]),
    )
    dominant_stage = max(stage_counts.items(), key=lambda item: item[1])[0] if stage_counts else ""
    overspeed_label = bool(len(speed_arr) and np.max(speed_arr) > 1.90)
    phase34_compatible = bool(first_crossing_step is not None and not overspeed_label and not instability)
    if overspeed_label:
        failure_label = "overspeed_failure"
    elif first_crossing_step is not None and not rec_state:
        failure_label = "phase34_bad_handoff"
    elif first_crossing_step is not None:
        failure_label = ""
    elif closest["abs_r_ratio"] <= 0.0035 or best_potential >= 0.72:
        failure_label = "near_crossing"
    elif "coast" in dominant_stage and closest["abs_r_ratio"] <= 0.025:
        failure_label = "over_conservative_transfer"
    elif abs(closest["vt_error_ratio"]) >= 0.65 or abs(closest["angular_momentum_error_ratio"]) >= 0.65:
        failure_label = "wrong_tangential_corridor"
    elif abs(closest["vr_ratio"]) < 0.020 and closest["abs_r_ratio"] > 0.012:
        failure_label = "bad_radial_timing"
    elif abs(closest["energy_error_ratio"]) >= 0.55:
        failure_label = "insufficient_radial_energy"
    else:
        failure_label = "dead_geometry"

    return {
        "case_id": case_id,
        "group": group,
        "r0_over_target": r0,
        "initial_velocity_angle_deg": angle,
        "thrust_scale": thrust_scale,
        "setting": setting.name,
        "crossing_occurs": bool(first_crossing_step is not None),
        "phase34_compatible_crossing": phase34_compatible,
        "recoverable_crossing": bool(first_crossing_step is not None and rec_state),
        "simulator_success_label": bool(success),
        "min_abs_radius_error_ratio": closest["abs_r_ratio"],
        "closest_approach_step": int(closest["step"]),
        "crossing_vr_ratio": crossing_vr_ratio if crossing_vr_ratio is not None else "",
        "crossing_vt_error_ratio": crossing_vt_error_ratio if crossing_vt_error_ratio is not None else "",
        "crossing_sync_error": crossing_sync_value if crossing_sync_value is not None else "",
        "best_post_cross_distance": best_post["distance"],
        "overspeed": overspeed_label,
        "instability": bool(instability),
        "failure_label": failure_label,
    }


def aggregate(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for setting in SETTINGS:
        subset = [row for row in rows if row["setting"] == setting.name]
        selected = [row for row in subset if row["group"] == "selected_non_crossing"]
        regression = [row for row in subset if row["group"] == "regression_crossing"]
        stats[setting.name] = {
            "rollouts": len(subset),
            "selected_crossings": sum(bool_from_csv(row["crossing_occurs"]) for row in selected),
            "selected_recoverable": sum(bool_from_csv(row["recoverable_crossing"]) for row in selected),
            "regression_crossings": sum(bool_from_csv(row["crossing_occurs"]) for row in regression),
            "regression_recoverable": sum(bool_from_csv(row["recoverable_crossing"]) for row in regression),
            "overspeed": sum(bool_from_csv(row["overspeed"]) for row in subset),
            "instability": sum(bool_from_csv(row["instability"]) for row in subset),
        }
    return stats


def closest_approach_delta(rows: Sequence[Dict[str, object]]) -> tuple[int, int, int, list[str]]:
    baseline = {
        row["case_id"]: as_float(row["min_abs_radius_error_ratio"])
        for row in rows
        if row["group"] == "selected_non_crossing" and row["setting"] == "early_commit_low_radial_only"
    }
    weak = {
        row["case_id"]: as_float(row["min_abs_radius_error_ratio"])
        for row in rows
        if row["group"] == "selected_non_crossing" and row["setting"] == "early_commit_low_plus_weak_tangential"
    }
    improved = 0
    worsened = 0
    unchanged = 0
    details: list[str] = []
    for case_id in sorted(baseline):
        delta = weak[case_id] - baseline[case_id]
        if delta < -1.0e-12:
            improved += 1
            label = "improved"
        elif delta > 1.0e-12:
            worsened += 1
            label = "worsened"
        else:
            unchanged += 1
            label = "unchanged"
        details.append(f"- `{case_id}`: delta `{delta:.8e}` ({label})")
    return improved, worsened, unchanged, details


def save_plot(stats: Dict[str, Dict[str, int]]) -> None:
    labels = [setting.name for setting in SETTINGS]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.bar(x - 0.24, [stats[label]["selected_crossings"] for label in labels], 0.24, label="selected crossings", color="#F58518")
    ax.bar(x, [stats[label]["selected_recoverable"] for label in labels], 0.24, label="selected recoverable", color="#54A24B")
    ax.bar(x + 0.24, [stats[label]["regression_crossings"] for label in labels], 0.24, label="regression crossings", color="#4C78A8")
    ax.axhline(8, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7, label="Phase36B regression crossing count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 8.5)
    ax.set_ylabel("cases")
    ax.set_title("Phase37B weak tangential subset diagnostic")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(COMPARISON_PNG, dpi=220)
    plt.close(fig)


def write_summary(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, int]]) -> None:
    improved, worsened, unchanged, delta_details = closest_approach_delta(rows)
    weak = stats["early_commit_low_plus_weak_tangential"]
    radial = stats["early_commit_low_radial_only"]
    total_overspeed = sum(item["overspeed"] for item in stats.values())
    total_instability = sum(item["instability"] for item in stats.values())

    phase36b_regression_crossing_count = 8
    regression_preserved = (
        weak["regression_crossings"] >= phase36b_regression_crossing_count
        and weak["regression_recoverable"] >= phase36b_regression_crossing_count
    )

    if weak["regression_crossings"] < phase36b_regression_crossing_count or weak["regression_recoverable"] < phase36b_regression_crossing_count:
        decision = "unsafe globally"
        interpretation = "The diagnostic action did not preserve the Phase36B baseline regression crossing set."
    elif weak["selected_crossings"] >= 1 and regression_preserved:
        decision = "positive diagnostic"
        interpretation = "Weak tangential shaping created at least one selected-case crossing without degrading the radial-only regression count."
    elif improved > 0 and weak["selected_crossings"] == 0 and regression_preserved:
        decision = "weak/partial diagnostic"
        interpretation = "Weak tangential shaping improved selected-case closest approach but did not create a new crossing."
    else:
        decision = "negative diagnostic"
        interpretation = "Weak tangential shaping did not create a crossing and did not improve closest approach."

    lines = [
        "# Phase37B Weak Tangential Subset Diagnostic",
        "",
        "## Scope",
        "",
        "- Subset diagnostic only: 4 selected Phase37A-improved non-crossing cases plus 8 Phase36B baseline crossing-producing regression cases.",
        "- Two settings: `early_commit_low_radial_only` and `early_commit_low_plus_weak_tangential`.",
        "- Fixed Phase34 `radius_priority` terminal/post-cross controller after first target-radius crossing.",
        "- No full 24-case tangential grid, no MPC, no RL, and no coast-duration search.",
        "- This remains a simplified 2D orbital-control sandbox result, not real spacecraft validation.",
        "",
        "## Control Perturbation",
        "",
        "- Radial overlay: `u_r = clamp(0.055 * direction_to_target - 0.16 * vr_ratio, -0.055, 0.055)`.",
        "- Weak tangential overlay: `u_t = clamp(-0.06 * vt_error_ratio, -0.020, 0.020)`.",
        "- Combined diagnostic action norm limit: `0.060`.",
        "- Weak tangential shaping is gated to the four selected `r0 = 1.02`, `thrust = 10000` cases and only before first crossing while outside the target-radius corridor.",
        "",
        "## Aggregate Results",
        "",
        "| Setting | Rollouts | Selected crossings | Selected recoverable crossings | Regression crossings | Regression recoverable crossings | Overspeed | Instability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        item = stats[setting.name]
        lines.append(
            f"| `{setting.name}` | {item['rollouts']} | {item['selected_crossings']} | {item['selected_recoverable']} | "
            f"{item['regression_crossings']} | {item['regression_recoverable']} | {item['overspeed']} | {item['instability']} |"
        )
    lines.extend(
        [
            "",
            "## Required Report",
            "",
            f"- Total rollouts: `{len(rows)}`.",
            f"- Selected-case new crossings under weak tangential shaping: `{weak['selected_crossings']} / 4`.",
            f"- Selected-case recoverable crossings under weak tangential shaping: `{weak['selected_recoverable']} / 4`.",
            f"- Regression crossing preservation under weak tangential shaping: `{weak['regression_crossings']} / 8` crossings and `{weak['regression_recoverable']} / 8` recoverable crossings, compared with the Phase36B baseline requirement of `8 / 8`.",
            f"- Overspeed count: `{total_overspeed}`.",
            f"- Instability count: `{total_instability}`.",
            f"- Closest-approach comparison versus radial-only on selected cases: `{improved}` improved, `{worsened}` worsened, `{unchanged}` unchanged.",
            "",
            "## Closest-Approach Deltas",
            "",
            *delta_details,
            "",
            "## Decision",
            "",
            f"- Diagnostic classification: `{decision}`.",
            f"- Interpretation: {interpretation}",
            "",
            "Phase37B should not be expanded unless this subset result creates at least one new selected-case crossing without damaging the regression set. If it only improves closest approach, that is a weak signal requiring analysis before adding any larger search dimension.",
            "",
            "## Artifacts",
            "",
            "- `phase37b_results.csv`",
            "- `phase37b_summary.md`",
            "- `phase37b_comparison.png`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark() -> List[Dict[str, object]]:
    cases = list(iter_phase37b_cases())
    rows: List[Dict[str, object]] = []
    for setting in SETTINGS:
        for case_id, group, r0, angle, thrust in cases:
            row = rollout_case(case_id, group, setting, PHASE34_TERMINAL_MODE, r0, angle, thrust)
            rows.append(row)
            print(
                f"phase37b setting={setting.name} case={case_id} group={group} r0={r0:g} "
                f"angle={angle:g} thrust={thrust:g} cross={row['crossing_occurs']} "
                f"rec={row['recoverable_crossing']} min_r={as_float(row['min_abs_radius_error_ratio']):.5f} "
                f"failure={row['failure_label']}"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase37B weak tangential subset diagnostic.")
    parser.add_argument("--skip-plots", action="store_true", help="Write CSV and markdown without regenerating plots.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = benchmark()
    if len(rows) != 24:
        raise RuntimeError(f"Phase37B expected 24 rows, generated {len(rows)}")
    stats = aggregate(rows)
    write_csv(RESULTS_CSV, FIELDNAMES, rows)
    if not args.skip_plots:
        save_plot(stats)
    write_summary(rows, stats)
    print(f"Saved Phase37B outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
