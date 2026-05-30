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
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase36b_transfer_family_benchmark"
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
    orbit_crosses_target,
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
from scripts.explicit_controller_phase35_crossing_basin_expansion import (
    crossing_potential,
)


RESULTS_CSV = OUTPUT_DIR / "phase36b_results.csv"
FAMILY_SUMMARY_CSV = OUTPUT_DIR / "phase36b_family_summary.csv"
FAILURE_MODES_CSV = OUTPUT_DIR / "phase36b_failure_modes.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
PLOT_FAMILY_COMPARISON = OUTPUT_DIR / "phase36b_family_comparison.png"

BENCH_R0_VALUES = [0.98, 1.00, 1.02]
BENCH_ANGLE_VALUES = [150.0, 165.0, 170.0, 175.0]
BENCH_THRUST_VALUES = [8000.0, 10000.0]

PHASE34_TERMINAL_MODE = next(mode for mode in PHASE34_MODES if mode.name == "radius_priority")


@dataclass(frozen=True)
class TransferFamily:
    name: str
    qualitative_label: str


FAMILIES = [
    TransferFamily("baseline_phase34", "Phase34 reference transfer"),
    TransferFamily("spiral_approach", "gradual low-thrust radius shaping"),
    TransferFamily("grazing_corridor", "near-target loiter with soft crossing windows"),
    TransferFamily("redesigned_delayed_crossing", "bounded delayed commitment before crossing"),
]


FIELDNAMES = [
    "controller_name",
    "transfer_family",
    "family_qualitative_label",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "target_radius_scale",
    "crossing_occurs",
    "first_crossing_step",
    "phase34_compatible_crossing",
    "recoverable_crossing",
    "recoverable_state",
    "crossing_vr_ratio",
    "crossing_vt_error_ratio",
    "crossing_sync_error",
    "best_post_cross_distance",
    "best_post_cross_sync",
    "min_abs_radius_error",
    "min_abs_radius_error_ratio",
    "closest_approach_step",
    "best_crossing_potential",
    "dominant_failure_label",
    "overspeed",
    "instability",
    "max_speed_ratio",
    "termination_reason",
    "simulator_success_label",
]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def iter_cases() -> Iterable[tuple[float, float, float]]:
    for thrust in BENCH_THRUST_VALUES:
        for angle in BENCH_ANGLE_VALUES:
            for r0 in BENCH_R0_VALUES:
                yield r0, angle, thrust


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


def rollout_family_case(
    family: TransferFamily,
    mode: PostCrossMode,
    case_label: str,
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
    first_crossing_step: int | None = None
    crossing_vr_ratio: float | None = None
    crossing_vt_error_ratio: float | None = None
    crossing_sync_value: float | None = None
    best_post: Dict[str, float] | None = None
    closest: Dict[str, float] | None = None
    best_potential = -1.0

    radius_errors: List[float] = []
    vr_values: List[float] = []
    speed_ratios: List[float] = []
    stage_counts: Counter[str] = Counter()
    post_actions: List[tuple[float, float]] = []

    radius_crossings_total = 0
    capture_entered = False
    lock_entered = False
    success = False
    terminated = False
    truncated = False
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

    trajectory: Dict[str, List[object]] = {
        "time_step": [],
        "x_over_target": [],
        "y_over_target": [],
        "radius_ratio": [],
        "r_error_ratio": [],
        "vr_ratio": [],
        "vt_error_ratio": [],
        "crossing_potential": [],
        "distance_to_recoverable": [],
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

    def apply_family_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        diag,
        base_action: tuple[float, float],
        active_stage: str,
        step_index: int,
    ) -> tuple[tuple[float, float], str]:
        name = family.name
        if name == "baseline_phase34":
            return base_action, active_stage

        direction_to_target = -1.0 if diag.r_error_ratio > 0.0 else 1.0
        abs_r = abs(diag.r_error_ratio)
        vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)

        if name == "spiral_approach":
            radial_cmd = clamp(0.11 * direction_to_target - 0.20 * vr_ratio, -0.14, 0.14)
            tangential_cmd = clamp(-0.16 * diag.vt_error_ratio, -0.10, 0.10)
            spiral = radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, 0.15)
            return limit_norm(0.45 * base_action[0] + spiral[0], 0.45 * base_action[1] + spiral[1], 0.16), "family_spiral_approach"

        if name == "redesigned_delayed_crossing":
            delay_steps = 1400
            close_band = abs_r < 0.030
            ready_to_commit = step_index >= delay_steps and (close_band or orbit_crosses_target(diag, target_radius))
            if not ready_to_commit:
                radial_cmd = clamp(0.030 * direction_to_target - 0.22 * vr_ratio, -0.075, 0.075)
                tangential_cmd = clamp(-0.18 * diag.vt_error_ratio - 0.05 * diag.angular_momentum_error_ratio, -0.10, 0.10)
                hold = radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, 0.11)
                return hold, "family_redesigned_delayed_shape"
            radial_cmd = clamp(0.115 * direction_to_target - 0.16 * vr_ratio, -0.14, 0.14)
            tangential_cmd = clamp(-0.14 * diag.vt_error_ratio - 0.06 * diag.angular_momentum_error_ratio, -0.12, 0.12)
            commit = radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, 0.16)
            return limit_norm(0.40 * base_action[0] + commit[0], 0.40 * base_action[1] + commit[1], 0.17), "family_redesigned_delayed_commit"

        if name == "energy_bleed_then_cross":
            if step_index < 2200:
                bleed = radial_tangential_action(state_x, state_y, state_vx, state_vy, -0.03 * vr_ratio, -0.18, 0.18)
                return bleed, "family_energy_bleed"
            commit = radial_tangential_action(state_x, state_y, state_vx, state_vy, 0.16 * direction_to_target, -0.10 * diag.vt_error_ratio, 0.18)
            return limit_norm(0.60 * base_action[0] + commit[0], 0.60 * base_action[1] + commit[1], 0.19), "family_energy_cross_commit"

        if name == "overshoot_return":
            if step_index < 1800 and abs_r < 0.040:
                away = radial_tangential_action(state_x, state_y, state_vx, state_vy, -0.13 * direction_to_target, -0.06 * diag.vt_error_ratio, 0.15)
                return away, "family_overshoot_shape"
            ret = radial_tangential_action(state_x, state_y, state_vx, state_vy, 0.19 * direction_to_target - 0.10 * vr_ratio, -0.10 * diag.vt_error_ratio, 0.20)
            return limit_norm(0.45 * base_action[0] + ret[0], 0.45 * base_action[1] + ret[1], 0.20), "family_overshoot_return"

        if name == "grazing_corridor":
            if abs_r < 0.035 and step_index < 7000:
                radial_cmd = clamp(0.035 * direction_to_target - 0.28 * vr_ratio, -0.09, 0.09)
                tangential_cmd = clamp(-0.12 * diag.vt_error_ratio, -0.08, 0.08)
                graze = radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, 0.10)
                return graze, "family_grazing_corridor"
            commit = radial_tangential_action(state_x, state_y, state_vx, state_vy, 0.10 * direction_to_target, -0.08 * diag.vt_error_ratio, 0.12)
            return limit_norm(0.55 * base_action[0] + commit[0], 0.55 * base_action[1] + commit[1], 0.14), "family_grazing_commit"

        if name == "two_stage_transfer":
            if step_index < 2600 or abs_r > 0.055:
                shape = radial_tangential_action(state_x, state_y, state_vx, state_vy, -0.05 * vr_ratio, -0.14 if diag.r_error_ratio > 0.0 else 0.14, 0.16)
                return shape, "family_two_stage_shape"
            commit = radial_tangential_action(state_x, state_y, state_vx, state_vy, 0.18 * direction_to_target - 0.12 * vr_ratio, -0.14 * diag.vt_error_ratio, 0.20)
            return limit_norm(0.50 * base_action[0] + commit[0], 0.50 * base_action[1] + commit[1], 0.20), "family_two_stage_commit"

        return base_action, active_stage

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
                "abs_r": abs(diag.radius - target_radius),
                "abs_r_ratio": abs_r_ratio,
                "r_error_ratio": diag.r_error_ratio,
                "vr_ratio": vr_ratio,
                "vt_error_ratio": vt_ratio,
                "energy_error_ratio": diag.energy_error_ratio,
                "angular_momentum_error_ratio": diag.angular_momentum_error_ratio,
                "potential": potential,
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
            (action_x, action_y), active_stage = apply_family_action(x, y, vx, vy, diag, (action_x, action_y), active_stage, step)
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

        x, y, vx, vy = env_step(x, y, vx, vy, action_x, action_y)
        next_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
        r_error = next_diag.radius - target_radius
        next_vr_ratio = next_diag.radial_velocity / (v_circ + 1.0e-12)
        next_vt_ratio = next_diag.vt_error_ratio
        next_sync = sync_error(next_diag.r_error_ratio, next_vr_ratio, next_vt_ratio)
        next_distance = recoverability_distance(next_diag.r_error_ratio, next_vr_ratio, next_vt_ratio)
        next_potential = crossing_potential(next_diag, target_radius, v_circ)

        radius_errors.append(r_error)
        vr_values.append(next_diag.radial_velocity)
        speed_ratios.append(next_diag.speed / (v_circ + 1.0e-12))

        if prev_r_error is not None and ((prev_r_error > 0.0 and r_error <= 0.0) or (prev_r_error < 0.0 and r_error >= 0.0)):
            radius_crossings_total += 1
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
                    "vt_error_ratio": next_vt_ratio,
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
        # Some r0=1.0 cases can satisfy the simulator success label before a
        # geometric crossing is recorded. Phase36B keeps running until crossing
        # or a safety/timeout condition so crossing production is measured.
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

        if record_trajectory:
            trajectory["time_step"].append(float(step))
            trajectory["x_over_target"].append(next_diag.radius * 0.0 + x / target_radius)
            trajectory["y_over_target"].append(y / target_radius)
            trajectory["radius_ratio"].append(next_diag.radius / target_radius)
            trajectory["r_error_ratio"].append(next_diag.r_error_ratio)
            trajectory["vr_ratio"].append(next_vr_ratio)
            trajectory["vt_error_ratio"].append(next_vt_ratio)
            trajectory["crossing_potential"].append(next_potential)
            trajectory["distance_to_recoverable"].append(next_distance)
            trajectory["stage"].append(active_stage)

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
            "abs_r": abs(final_diag.radius - target_radius),
            "abs_r_ratio": abs(final_diag.r_error_ratio),
            "r_error_ratio": final_diag.r_error_ratio,
            "vr_ratio": final_vr_ratio,
            "vt_error_ratio": final_diag.vt_error_ratio,
            "energy_error_ratio": final_diag.energy_error_ratio,
            "angular_momentum_error_ratio": final_diag.angular_momentum_error_ratio,
            "potential": crossing_potential(final_diag, target_radius, v_circ),
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
        dominant_failure_label = "overspeed_failure"
    elif first_crossing_step is not None and not rec_state:
        dominant_failure_label = "phase34_bad_handoff"
    elif first_crossing_step is not None:
        dominant_failure_label = ""
    elif closest["abs_r_ratio"] <= 0.0035 or best_potential >= 0.72:
        dominant_failure_label = "near_crossing"
    elif "coast" in dominant_stage and closest["abs_r_ratio"] <= 0.025:
        dominant_failure_label = "over_conservative_transfer"
    elif abs(closest["vt_error_ratio"]) >= 0.65 or abs(closest["angular_momentum_error_ratio"]) >= 0.65:
        dominant_failure_label = "wrong_tangential_corridor"
    elif abs(closest["vr_ratio"]) < 0.020 and closest["abs_r_ratio"] > 0.012:
        dominant_failure_label = "bad_radial_timing"
    elif abs(closest["energy_error_ratio"]) >= 0.55:
        dominant_failure_label = "insufficient_radial_energy"
    else:
        dominant_failure_label = "dead_geometry"

    row: Dict[str, object] = {
        "controller_name": "phase36b_transfer_family_benchmark",
        "transfer_family": family.name,
        "family_qualitative_label": family.qualitative_label,
        "r0_over_target": r0,
        "initial_velocity_angle_deg": angle,
        "thrust_scale": thrust_scale,
        "target_radius_scale": target_scale,
        "crossing_occurs": bool(first_crossing_step is not None),
        "first_crossing_step": first_crossing_step if first_crossing_step is not None else "",
        "phase34_compatible_crossing": phase34_compatible,
        "recoverable_crossing": bool(first_crossing_step is not None and rec_state),
        "recoverable_state": bool(rec_state),
        "crossing_vr_ratio": crossing_vr_ratio if crossing_vr_ratio is not None else "",
        "crossing_vt_error_ratio": crossing_vt_error_ratio if crossing_vt_error_ratio is not None else "",
        "crossing_sync_error": crossing_sync_value if crossing_sync_value is not None else "",
        "min_abs_radius_error": closest["abs_r"],
        "min_abs_radius_error_ratio": closest["abs_r_ratio"],
        "closest_approach_step": int(closest["step"]),
        "best_crossing_potential": best_potential,
        "best_post_cross_distance": best_post["distance"],
        "best_post_cross_sync": best_post["sync"],
        "dominant_failure_label": dominant_failure_label,
        "overspeed": overspeed_label,
        "instability": bool(instability),
        "max_speed_ratio": float(np.max(speed_arr)) if len(speed_arr) else 0.0,
        "termination_reason": termination_reason,
        "simulator_success_label": bool(success),
    }
    if record_trajectory:
        row["trajectory"] = trajectory
    return row


def aggregate(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for family in FAMILIES:
        subset = [row for row in rows if row.get("transfer_family") == family.name]
        if not subset:
            continue
        crossing = [row for row in subset if bool_from_csv(row.get("crossing_occurs"))]
        compatible = [row for row in subset if bool_from_csv(row.get("phase34_compatible_crossing"))]
        recoverable = [row for row in subset if bool_from_csv(row.get("recoverable_crossing"))]
        out[family.name] = {
            "cases": len(subset),
            "crossings": len(crossing),
            "phase34_compatible_crossings": len(compatible),
            "recoverable_crossings": len(recoverable),
            "simulator_success_labels": sum(bool_from_csv(row.get("simulator_success_label")) for row in subset),
            "overspeed": sum(bool_from_csv(row.get("overspeed")) for row in subset),
            "instability": sum(bool_from_csv(row.get("instability")) for row in subset),
            "mean_min_abs_radius_error_ratio": float(np.mean([as_float(row.get("min_abs_radius_error_ratio")) for row in subset])),
            "mean_best_crossing_potential": float(np.mean([as_float(row.get("best_crossing_potential")) for row in subset])),
            "mean_crossing_sync": float(np.mean([as_float(row.get("crossing_sync_error")) for row in crossing])) if crossing else float("nan"),
            "mean_best_post_cross_distance": float(np.mean([as_float(row.get("best_post_cross_distance")) for row in crossing])) if crossing else float("nan"),
        }
    return out


def write_family_summary_csv(stats: Dict[str, Dict[str, object]]) -> None:
    fields = [
        "transfer_family",
        "cases",
        "crossing_count",
        "phase34_compatible_crossing_count",
        "recoverable_crossing_count",
        "simulator_success_label_count",
        "overspeed_count",
        "instability_count",
        "mean_min_abs_radius_error_ratio",
        "mean_best_crossing_potential",
        "mean_crossing_sync",
        "mean_best_post_cross_distance",
    ]
    rows = []
    for family in FAMILIES:
        item = stats[family.name]
        rows.append(
            {
                "transfer_family": family.name,
                "cases": item["cases"],
                "crossing_count": item["crossings"],
                "phase34_compatible_crossing_count": item["phase34_compatible_crossings"],
                "recoverable_crossing_count": item["recoverable_crossings"],
                "simulator_success_label_count": item["simulator_success_labels"],
                "overspeed_count": item["overspeed"],
                "instability_count": item["instability"],
                "mean_min_abs_radius_error_ratio": item["mean_min_abs_radius_error_ratio"],
                "mean_best_crossing_potential": item["mean_best_crossing_potential"],
                "mean_crossing_sync": item["mean_crossing_sync"],
                "mean_best_post_cross_distance": item["mean_best_post_cross_distance"],
            }
        )
    write_csv(FAMILY_SUMMARY_CSV, fields, rows)


def write_failure_modes_csv(rows: Sequence[Dict[str, object]]) -> None:
    fields = ["transfer_family", "dominant_failure_label", "count"]
    out = []
    for family in FAMILIES:
        subset = [row for row in rows if row.get("transfer_family") == family.name and str(row.get("dominant_failure_label", ""))]
        counts = Counter(str(row.get("dominant_failure_label")) for row in subset)
        for label, count in sorted(counts.items()):
            out.append({"transfer_family": family.name, "dominant_failure_label": label, "count": count})
    write_csv(FAILURE_MODES_CSV, fields, out)


def save_plots(rows: Sequence[Dict[str, object]], examples: Sequence[Dict[str, object]]) -> None:
    stats = aggregate(rows)
    labels = [family.name for family in FAMILIES]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.bar(x - 0.25, [stats[label]["crossings"] for label in labels], 0.25, label="geometric crossings", color="#4C78A8")
    ax.bar(x, [stats[label]["phase34_compatible_crossings"] for label in labels], 0.25, label="Phase34-compatible crossings", color="#72B7B2")
    ax.bar(x + 0.25, [stats[label]["recoverable_crossings"] for label in labels], 0.25, label="recoverable crossings", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 24)
    ax.set_ylabel("cases out of 24")
    ax.set_title("Phase36B transfer-family benchmark")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_FAMILY_COMPARISON, dpi=220)
    plt.close(fig)


def write_markdown(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]], cases: Sequence[tuple[float, float, float]]) -> None:
    def fmt_optional(value: object, digits: int = 4) -> str:
        value_f = as_float(value)
        if not math.isfinite(value_f):
            return "N/A"
        return f"{value_f:.{digits}f}"

    baseline = stats["baseline_phase34"]
    best_cross = max(stats, key=lambda name: stats[name]["crossings"])
    best_recoverable = max(stats, key=lambda name: stats[name]["recoverable_crossings"])
    best_quality = min(
        [name for name, item in stats.items() if item["crossings"] > 0],
        key=lambda name: stats[name]["mean_crossing_sync"],
        default="none",
    )
    crossing_improved = stats[best_cross]["crossings"] > baseline["crossings"]
    recoverable_improved = stats[best_recoverable]["recoverable_crossings"] > baseline["recoverable_crossings"]
    baseline_crossing_keys = {
        (as_float(row.get("r0_over_target")), as_float(row.get("initial_velocity_angle_deg")), as_float(row.get("thrust_scale")))
        for row in rows
        if row.get("transfer_family") == "baseline_phase34" and bool_from_csv(row.get("crossing_occurs"))
    }
    compatible_new = any(
        bool_from_csv(row.get("phase34_compatible_crossing"))
        and row.get("transfer_family") != "baseline_phase34"
        and (as_float(row.get("r0_over_target")), as_float(row.get("initial_velocity_angle_deg")), as_float(row.get("thrust_scale"))) not in baseline_crossing_keys
        for row in rows
    )
    overspeed_families = [name for name, item in stats.items() if item["overspeed"] > 0]
    instability_families = [name for name, item in stats.items() if item["instability"] > 0]
    failure_counts = Counter(str(row.get("dominant_failure_label")) for row in rows if str(row.get("dominant_failure_label", "")))
    near_count = failure_counts.get("near_crossing", 0)
    conservative_count = failure_counts.get("over_conservative_transfer", 0)
    if crossing_improved and recoverable_improved:
        geometry_reading = "Phase36B found a transfer-family candidate that improves handoff into Phase34."
        next_step = "family formalization before planner escalation"
    elif crossing_improved:
        geometry_reading = "Phase36B expanded geometric crossing but did not improve recoverable handoff."
        next_step = "family redesign focused on Phase34 handoff quality"
    else:
        geometry_reading = "Phase36B narrowed the transfer-family hypothesis space but did not expand the crossing basin under the tested family set."
        next_step = "planner-level transfer-family search before MPC-lite"

    lines = [
        "# Phase36B Transfer Family Benchmark",
        "",
        "## Scope",
        "",
        "- Full 24-case reduced benchmark used for Phase34 and Phase35.",
        "- Simplified 2D orbital-control sandbox only; this is not real spacecraft validation.",
        "- Phase34 `radius_priority` post-cross synchronization is the fixed terminal controller.",
        "- Physics, CAPTURE/LOCK thresholds, recoverability thresholds, and termination checks are unchanged.",
        "- Phase36A representative-subset results are not used as full benchmark evidence.",
        "",
        "## Families",
        "",
        "- `baseline_phase34`: Phase34 reference transfer.",
        "- `spiral_approach`: gradual low-thrust radius shaping.",
        "- `grazing_corridor`: near-target loiter with soft crossing windows.",
        "- `redesigned_delayed_crossing`: bounded delayed commitment before crossing.",
        "",
        "Excluded unless redesigned: `energy_bleed_then_cross`, `overshoot_return`, and `two_stage_transfer`.",
        "",
        "## Aggregate Results",
        "",
        "| Transfer family | Cases | Crossings | Phase34-compatible crossings | Recoverable crossings | Simulator success label | Overspeed | Instability | Mean crossing sync | Mean best post-cross distance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        item = stats[family.name]
        lines.append(
            f"| `{family.name}` | {item['cases']} | {item['crossings']} | {item['phase34_compatible_crossings']} | "
            f"{item['recoverable_crossings']} | {item['simulator_success_labels']} | {item['overspeed']} | {item['instability']} | "
            f"{fmt_optional(item['mean_crossing_sync'])} | {fmt_optional(item['mean_best_post_cross_distance'])} |"
        )
    lines.extend(
        [
            "",
            "## Required Questions",
            "",
            f"- Did any family improve geometric crossing count over `baseline_phase34`? `{'yes' if crossing_improved else 'no'}`. Best crossing family: `{best_cross}` with `{stats[best_cross]['crossings']} / 24` crossings; baseline has `{baseline['crossings']} / 24`.",
            f"- Did any family improve recoverable crossing count over `baseline_phase34`? `{'yes' if recoverable_improved else 'no'}`. Best recoverable family: `{best_recoverable}` with `{stats[best_recoverable]['recoverable_crossings']} / 24`; baseline has `{baseline['recoverable_crossings']} / 24`.",
            f"- Did any new crossing remain Phase34-compatible? `{'yes' if compatible_new else 'no'}`.",
            f"- Which family produced the best crossing-state quality? `{best_quality}` by mean crossing sync among families with crossings.",
            f"- Which family caused overspeed or instability? Overspeed: `{', '.join(overspeed_families) if overspeed_families else 'none'}`. Instability: `{', '.join(instability_families) if instability_families else 'none'}`.",
            f"- Which failures were near-crossing or over-conservative? `near_crossing={near_count}`, `over_conservative_transfer={conservative_count}`.",
            f"- Does the result support or weaken the transfer-family geometry hypothesis? {geometry_reading}",
            f"- Should the next step be planner-level search, MPC-lite, or family redesign? `{next_step}`.",
            "",
            "## Failure Mode Counts",
            "",
            "| Failure label | Count |",
            "|---|---:|",
        ]
    )
    for label, count in sorted(failure_counts.items()):
        lines.append(f"| `{label}` | {count} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `phase36b_results.csv`",
            "- `phase36b_family_summary.csv`",
            "- `phase36b_failure_modes.csv`",
            "- `phase36b_family_comparison.png`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def benchmark() -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[tuple[float, float, float]]]:
    cases = list(iter_cases())
    rows: List[Dict[str, object]] = []
    examples: List[Dict[str, object]] = []
    for family in FAMILIES:
        for r0, angle, thrust in cases:
            row = rollout_family_case(family, PHASE34_TERMINAL_MODE, "full_24_case", r0, angle, thrust, record_trajectory=False)
            rows.append(row)
            print(
                f"phase36b family={family.name} r0={r0:g} angle={angle:g} thrust={thrust:g} "
                f"cross={row['crossing_occurs']} rec={row['recoverable_crossing']} "
                f"min_r={as_float(row['min_abs_radius_error_ratio']):.5f} reason={row['termination_reason']}"
            )
    return rows, examples, cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase36B full transfer-family benchmark.")
    parser.add_argument("--skip-plots", action="store_true", help="Write CSV and markdown without regenerating plots.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, examples, cases = benchmark()
    stats = aggregate(rows)
    write_csv(RESULTS_CSV, FIELDNAMES, rows)
    write_family_summary_csv(stats)
    write_failure_modes_csv(rows)
    if not args.skip_plots:
        save_plots(rows, examples)
    write_markdown(rows, stats, cases)
    print(f"Saved Phase36B outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
