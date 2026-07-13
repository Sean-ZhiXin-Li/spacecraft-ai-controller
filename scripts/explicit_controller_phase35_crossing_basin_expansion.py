from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase35_crossing_basin_expansion"
MPLCONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig
from runtime_assurance.final_veto_runner_types import (
    ActionInterceptionResult,
    PostTransitionObservation,
    PostTransitionObservationHook,
    PreTransitionActionContext,
    PreTransitionActionHook,
    RolloutCaseContext,
)
from simulator.phase34_35_transition import (
    CartesianState2D,
    NormalizedAction2D,
    Phase3435DynamicsContext,
    step_phase34_35_transition,
)
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
    INSERTION_BAND_RATIO,
    SAFE_SPEED_RATIO,
    classify_intercept,
)
from scripts.explicit_controller_phase34_post_cross_sync import (
    MODES as PHASE34_MODES,
    PostCrossMode,
    RESULTS_CSV as PHASE34_RESULTS,
    as_float,
    recoverability_distance,
    recoverable_state,
    sync_error,
)


RESULTS_CSV = OUTPUT_DIR / "phase35_results.csv"
DIAGNOSIS_CSV = OUTPUT_DIR / "non_crossing_diagnosis.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
FAILURE_MD = OUTPUT_DIR / "failure_mode_summary.md"
COMPARE_MD = OUTPUT_DIR / "phase35_vs_phase34.md"

PLOT_CROSSING_COUNTS = OUTPUT_DIR / "crossing_count_comparison.png"
PLOT_FAILURE_MODES = OUTPUT_DIR / "non_crossing_failure_modes.png"
PLOT_POTENTIAL = OUTPUT_DIR / "crossing_potential_distribution.png"
PLOT_MIN_RADIUS = OUTPUT_DIR / "min_radius_error_by_variant.png"
PLOT_HANDOFF = OUTPUT_DIR / "phase34_terminal_handoff_examples.png"

BENCH_R0_VALUES = [0.98, 1.00, 1.02]
BENCH_ANGLE_VALUES = [150.0, 165.0, 170.0, 175.0]
BENCH_THRUST_VALUES = [8000.0, 10000.0]


@dataclass(frozen=True)
class PreCrossVariant:
    name: str
    kind: str


VARIANTS = [
    PreCrossVariant("baseline_phase34", "baseline"),
    PreCrossVariant("radial_energy_push", "radial_energy"),
    PreCrossVariant("tangential_corridor_entry", "tangential_corridor"),
    PreCrossVariant("predictive_crossing_bias", "predictive"),
]

PHASE34_TERMINAL_MODE = next(mode for mode in PHASE34_MODES if mode.name == "radius_priority")


FIELDNAMES = [
    "controller_name",
    "upstream_variant",
    "post_cross_mode",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "target_radius_scale",
    "crossing_occurs",
    "crossing_step",
    "recoverable_crossing",
    "recoverable_state",
    "capture_entered",
    "lock_entered",
    "success",
    "non_crossing",
    "best_crossing_potential",
    "final_crossing_potential",
    "crossing_potential_at_closest",
    "min_abs_radius_error",
    "min_abs_radius_error_ratio",
    "closest_approach_step",
    "final_radius_error",
    "final_radius_error_ratio",
    "radial_velocity_trend",
    "closest_vr_ratio",
    "closest_vt_error_ratio",
    "closest_energy_error_ratio",
    "closest_angular_momentum_error_ratio",
    "final_vr_ratio",
    "final_vt_error_ratio",
    "final_energy_error_ratio",
    "final_angular_momentum_error_ratio",
    "best_post_cross_step",
    "best_post_cross_sync",
    "best_post_cross_distance",
    "radius_crossings_total",
    "post_cross_steps",
    "steps",
    "terminated",
    "truncated",
    "termination_reason",
    "dominant_stage",
    "stage_counts_json",
    "max_speed_ratio",
    "overspeed",
    "instability",
    "failure_label",
    "deadness",
]

DIAGNOSIS_FIELDS = [
    "case_id",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "minimum_radius_error",
    "minimum_radius_error_ratio",
    "final_radius_error",
    "final_radius_error_ratio",
    "radial_velocity_trend",
    "closest_vr_ratio",
    "final_vr_ratio",
    "closest_vt_error_ratio",
    "final_vt_error_ratio",
    "closest_energy_error_ratio",
    "final_energy_error_ratio",
    "closest_angular_momentum_error_ratio",
    "final_angular_momentum_error_ratio",
    "closest_approach_step",
    "best_crossing_potential",
    "crossing_potential_at_closest",
    "near_or_dead",
    "failure_label",
]


def iter_cases() -> Iterable[tuple[float, float, float]]:
    for thrust in BENCH_THRUST_VALUES:
        for angle in BENCH_ANGLE_VALUES:
            for r0 in BENCH_R0_VALUES:
                yield r0, angle, thrust


def case_id(r0: float, angle: float, thrust: float) -> str:
    return f"r0={r0:.5g}|angle={angle:.5g}|thrust={thrust:.5g}"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def estimate_crossing_steps(radius_error: float, radial_velocity: float) -> int | None:
    if radius_error * radial_velocity >= 0.0 or abs(radial_velocity) < 1.0e-12:
        return None
    steps = abs(radius_error) / (abs(radial_velocity) * DT + 1.0e-12)
    if not math.isfinite(steps) or steps < 0.0:
        return None
    return int(round(steps))


def limit_norm(action_x: float, action_y: float, max_norm: float) -> tuple[float, float]:
    norm = math.sqrt(action_x * action_x + action_y * action_y)
    if norm <= max_norm or norm <= 1.0e-12:
        return clamp(action_x, -1.0, 1.0), clamp(action_y, -1.0, 1.0)
    scale = max_norm / norm
    return clamp(action_x * scale, -1.0, 1.0), clamp(action_y * scale, -1.0, 1.0)


def crossing_potential(diag, target_radius: float, v_circ: float) -> float:
    radius_term = math.exp(-abs(diag.r_error_ratio) / 0.030)
    approach_term = 1.0 if diag.r_error_ratio * diag.radial_velocity < 0.0 else 0.18
    vt_term = math.exp(-abs(diag.vt_error_ratio) / 0.55)
    energy_term = math.exp(-abs(diag.energy_error_ratio) / 0.80)
    h_term = math.exp(-abs(diag.angular_momentum_error_ratio) / 0.80)
    orbit_term = 1.0 if orbit_crosses_target(diag, target_radius) else 0.15
    predicted = estimate_crossing_steps(diag.radius - target_radius, diag.radial_velocity)
    if predicted is None:
        prediction_term = 0.10
    else:
        prediction_term = math.exp(-min(predicted, 50000) / 18000.0)
    speed_term = math.exp(-abs(diag.speed / (v_circ + 1.0e-12) - 1.0) / 0.70)
    return float(
        0.24 * radius_term
        + 0.15 * approach_term
        + 0.15 * vt_term
        + 0.12 * energy_term
        + 0.12 * h_term
        + 0.14 * orbit_term
        + 0.06 * prediction_term
        + 0.02 * speed_term
    )


def classify_failure(
    min_abs_ratio: float,
    final_r_ratio: float,
    closest_vr_ratio: float,
    final_vr_ratio: float,
    closest_vt_error: float,
    closest_energy_error: float,
    closest_h_error: float,
    best_potential: float,
    dominant_stage: str,
) -> tuple[str, str]:
    if min_abs_ratio <= 0.0035 or best_potential >= 0.72:
        return "near_crossing", "near_crossing"
    if "coast" in dominant_stage and min_abs_ratio <= 0.025:
        return "over_conservative_transfer", "near_crossing"
    if abs(closest_vt_error) >= 0.65 or abs(closest_h_error) >= 0.65:
        return "wrong_tangential_corridor", "truly_dead"
    if final_r_ratio * final_vr_ratio > 0.0 and abs(closest_energy_error) >= 0.55:
        return "insufficient_radial_energy", "truly_dead"
    if abs(closest_vr_ratio) < 0.020 and min_abs_ratio > 0.012:
        return "bad_initial_geometry", "truly_dead"
    return "dead_geometry", "truly_dead"


def summarize_radial_velocity(vr_values: Sequence[float]) -> str:
    if len(vr_values) < 10:
        return "insufficient_samples"
    early = float(np.mean(vr_values[: min(200, len(vr_values))]))
    late = float(np.mean(vr_values[-min(200, len(vr_values)) :]))
    if abs(late) < 0.15 * max(abs(early), 1.0):
        return "damped_to_low_radial_motion"
    if early * late < 0.0:
        return "reversed_direction"
    if abs(late) > abs(early):
        return "radial_motion_growing"
    return "same_direction_weakening"


def rollout_phase35_case(
    variant: PreCrossVariant,
    mode: PostCrossMode,
    r0: float,
    angle: float,
    thrust_scale: float,
    target_scale: float = TARGET_RADIUS_SCALE,
    record_trajectory: bool = False,
    *,
    case_id: str = "",
    pre_transition_action_hook: PreTransitionActionHook | None = None,
    post_transition_observation_hook: PostTransitionObservationHook | None = None,
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
    radius_errors: List[float] = []
    vr_values: List[float] = []
    vr_ratio_values: List[float] = []
    speed_ratios: List[float] = []
    post_actions: List[tuple[float, float]] = []
    stage_counts: Counter[str] = Counter()

    radius_crossings_total = 0
    first_crossing_step: int | None = None
    crossing_sync_value: float | None = None
    crossing_distance_value: float | None = None
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
    best_potential = -1.0
    final_potential = 0.0
    closest: Dict[str, float] | None = None
    instability = False

    trajectory: Dict[str, List[object]] = {
        "time_step": [],
        "radius_ratio": [],
        "r_error_ratio": [],
        "vr_ratio": [],
        "vt_error_ratio": [],
        "crossing_potential": [],
        "distance_to_recoverable": [],
        "stage": [],
    }

    transition_context = Phase3435DynamicsContext(
        mu=MU,
        dt=DT,
        mass=MASS,
        thrust_scale=thrust_scale,
    )
    case_context = RolloutCaseContext(
        case_id=case_id,
        controller_id="phase35_crossing_basin_expansion",
        controller_family="explicit_controller",
        r0_over_target=r0,
        initial_velocity_angle_deg=angle,
        thrust_scale=thrust_scale,
        target_radius=target_radius,
        target_circular_speed=v_circ,
        post_cross_mode=mode.name,
        upstream_variant=variant.name,
    )

    def predict_transition(
        state: CartesianState2D,
        action: tuple[float, float],
    ):
        return step_phase34_35_transition(
            state,
            NormalizedAction2D(action[0], action[1]),
            transition_context,
        )

    def compute_speed_ratio(state: CartesianState2D) -> float:
        return math.sqrt(state.vx * state.vx + state.vy * state.vy) / (v_circ + 1.0e-12)

    def env_step(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        action_x: float,
        action_y: float,
    ) -> tuple[float, float, float, float]:
        transition = step_phase34_35_transition(
            CartesianState2D(state_x, state_y, state_vx, state_vy),
            NormalizedAction2D(action_x, action_y),
            transition_context,
        )
        next_state = transition.next_state
        return next_state.x, next_state.y, next_state.vx, next_state.vy

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

    def add_pre_cross_bias(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        diag,
        base_action: tuple[float, float],
    ) -> tuple[tuple[float, float], str]:
        if variant.kind == "baseline":
            return base_action, ""

        direction_to_target = -1.0 if diag.r_error_ratio > 0.0 else 1.0
        radial_strength = clamp(0.08 + 2.8 * abs(diag.r_error_ratio), 0.08, 0.20)
        radial_action = radial_tangential_action(state_x, state_y, state_vx, state_vy, direction_to_target * radial_strength, 0.0, 0.20)
        tangential_cmd = clamp(-0.42 * diag.vt_error_ratio - 0.16 * diag.angular_momentum_error_ratio, -0.20, 0.20)
        tangent_action = radial_tangential_action(state_x, state_y, state_vx, state_vy, 0.0, tangential_cmd, 0.20)

        if variant.kind == "radial_energy":
            mixed = (base_action[0] + radial_action[0], base_action[1] + radial_action[1])
            return limit_norm(mixed[0], mixed[1], 0.20), "radial_energy_push"
        if variant.kind == "tangential_corridor":
            mixed = (base_action[0] + tangent_action[0], base_action[1] + tangent_action[1])
            return limit_norm(mixed[0], mixed[1], 0.18), "tangential_corridor_entry"

        candidates = [
            ("baseline_candidate", base_action),
            ("radial_candidate", radial_action),
            ("tangential_candidate", tangent_action),
            ("blend_candidate", limit_norm(base_action[0] + 0.65 * radial_action[0] + 0.65 * tangent_action[0], base_action[1] + 0.65 * radial_action[1] + 0.65 * tangent_action[1], 0.19)),
        ]
        best_label = "baseline_candidate"
        best_action = base_action
        best_score = -1.0e9
        for label, action in candidates:
            nx, ny, nvx, nvy = env_step(state_x, state_y, state_vx, state_vy, action[0], action[1])
            ndiag = orbital_diagnostics(nx, ny, nvx, nvy, target_radius)
            potential = crossing_potential(ndiag, target_radius, v_circ)
            speed_penalty = max(0.0, ndiag.speed / (v_circ + 1.0e-12) - SAFE_SPEED_RATIO)
            score = potential - 0.40 * speed_penalty - 0.04 * abs(ndiag.vt_error_ratio)
            if score > best_score:
                best_score = score
                best_label = label
                best_action = action
        return best_action, f"predictive_crossing_bias_{best_label}"

    for step in range(1, MAX_STEPS + 1):
        diag = orbital_diagnostics(x, y, vx, vy, target_radius)
        speed_ratio = diag.speed / (v_circ + 1.0e-12)
        vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
        vt_ratio = diag.vt_error_ratio
        current_sync = sync_error(diag.r_error_ratio, vr_ratio, vt_ratio)
        current_distance = recoverability_distance(diag.r_error_ratio, vr_ratio, vt_ratio)

        potential = crossing_potential(diag, target_radius, v_circ)
        if potential > best_potential:
            best_potential = potential
        abs_r_ratio = abs(diag.r_error_ratio)
        if closest is None or abs_r_ratio < closest["abs_r_ratio"]:
            closest = {
                "step": float(step),
                "abs_r": abs(diag.radius - target_radius),
                "abs_r_ratio": abs_r_ratio,
                "r_error": diag.radius - target_radius,
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

            (action_x, action_y), bias_label = add_pre_cross_bias(x, y, vx, vy, diag, (action_x, action_y))
            if bias_label:
                active_stage = bias_label
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

        previous_state = CartesianState2D(x, y, vx, vy)
        nominal_action = (action_x, action_y)
        executed_action = nominal_action
        intervention_applied = False
        decision_metadata: object | None = None
        if pre_transition_action_hook is not None:
            interception = pre_transition_action_hook(
                PreTransitionActionContext(
                    step=step,
                    phase=phase,
                    active_stage=active_stage,
                    current_state=previous_state,
                    nominal_action=nominal_action,
                    predict_transition=predict_transition,
                    compute_speed_ratio=compute_speed_ratio,
                    case=case_context,
                )
            )
            if not isinstance(interception, ActionInterceptionResult):
                raise TypeError("pre-transition hook must return ActionInterceptionResult")
            if interception.nominal_action != nominal_action:
                raise ValueError("pre-transition hook must preserve the nominal action evidence")
            executed_action = interception.executed_action
            intervention_applied = interception.intervention_applied
            decision_metadata = interception.decision_metadata
        action_x, action_y = executed_action
        x, y, vx, vy = env_step(x, y, vx, vy, action_x, action_y)
        next_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
        realized_next_state = CartesianState2D(x, y, vx, vy)
        realized_speed_ratio = next_diag.speed / (v_circ + 1.0e-12)
        if post_transition_observation_hook is not None:
            post_transition_observation_hook(
                PostTransitionObservation(
                    step=step,
                    phase=phase,
                    active_stage=active_stage,
                    previous_state=previous_state,
                    nominal_action=nominal_action,
                    executed_action=executed_action,
                    realized_next_state=realized_next_state,
                    realized_next_speed_ratio=realized_speed_ratio,
                    intervention_applied=intervention_applied,
                    decision_metadata=decision_metadata,
                    case=case_context,
                )
            )
        r_error = next_diag.radius - target_radius
        next_vr_ratio = next_diag.radial_velocity / (v_circ + 1.0e-12)
        next_vt_ratio = next_diag.vt_error_ratio
        next_sync = sync_error(next_diag.r_error_ratio, next_vr_ratio, next_vt_ratio)
        next_distance = recoverability_distance(next_diag.r_error_ratio, next_vr_ratio, next_vt_ratio)

        radius_errors.append(r_error)
        vr_values.append(next_diag.radial_velocity)
        vr_ratio_values.append(next_vr_ratio)
        speed_ratios.append(next_diag.speed / (v_circ + 1.0e-12))

        if prev_r_error is not None and ((prev_r_error > 0.0 and r_error <= 0.0) or (prev_r_error < 0.0 and r_error >= 0.0)):
            radius_crossings_total += 1
            if first_crossing_step is None:
                first_crossing_step = step
                crossing_sync_value = next_sync
                crossing_distance_value = next_distance
                phase = "POST_CROSS_SYNC"
                stage_step = 0
                prev_post_action = nominal_action
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
            instability = True
        elif overspeed:
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
            trajectory["radius_ratio"].append(next_diag.radius / target_radius)
            trajectory["r_error_ratio"].append(next_diag.r_error_ratio)
            trajectory["vr_ratio"].append(next_vr_ratio)
            trajectory["vt_error_ratio"].append(next_vt_ratio)
            trajectory["crossing_potential"].append(crossing_potential(next_diag, target_radius, v_circ))
            trajectory["distance_to_recoverable"].append(next_distance)
            trajectory["stage"].append(active_stage)

        if terminated or truncated:
            break

    final_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
    final_vr_ratio = final_diag.radial_velocity / (v_circ + 1.0e-12)
    final_potential = crossing_potential(final_diag, target_radius, v_circ)
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
            "r_error": final_diag.radius - target_radius,
            "r_error_ratio": final_diag.r_error_ratio,
            "vr_ratio": final_vr_ratio,
            "vt_error_ratio": final_diag.vt_error_ratio,
            "energy_error_ratio": final_diag.energy_error_ratio,
            "angular_momentum_error_ratio": final_diag.angular_momentum_error_ratio,
            "potential": final_potential,
        }

    rerr = np.asarray(radius_errors, dtype=np.float64)
    speed_arr = np.asarray(speed_ratios, dtype=np.float64)
    rec_state = recoverable_state(
        as_float(best_post["r_error_ratio"]),
        as_float(best_post["vr_ratio"]),
        as_float(best_post["vt_error_ratio"]),
    )
    dominant_stage = max(stage_counts.items(), key=lambda item: item[1])[0] if stage_counts else ""
    failure_label = ""
    deadness = ""
    if first_crossing_step is None:
        failure_label, deadness = classify_failure(
            closest["abs_r_ratio"],
            final_diag.r_error_ratio,
            closest["vr_ratio"],
            final_vr_ratio,
            closest["vt_error_ratio"],
            closest["energy_error_ratio"],
            closest["angular_momentum_error_ratio"],
            best_potential,
            dominant_stage,
        )

    row: Dict[str, object] = {
        "controller_name": "phase35_crossing_basin_expansion",
        "upstream_variant": variant.name,
        "post_cross_mode": mode.name,
        "r0_over_target": r0,
        "initial_velocity_angle_deg": angle,
        "thrust_scale": thrust_scale,
        "target_radius_scale": target_scale,
        "crossing_occurs": bool(first_crossing_step is not None),
        "crossing_step": first_crossing_step if first_crossing_step is not None else "",
        "recoverable_crossing": bool(first_crossing_step is not None and rec_state),
        "recoverable_state": bool(rec_state),
        "capture_entered": bool(capture_entered),
        "lock_entered": bool(lock_entered),
        "success": bool(success),
        "non_crossing": bool(first_crossing_step is None),
        "best_crossing_potential": best_potential,
        "final_crossing_potential": final_potential,
        "crossing_potential_at_closest": closest["potential"],
        "min_abs_radius_error": closest["abs_r"],
        "min_abs_radius_error_ratio": closest["abs_r_ratio"],
        "closest_approach_step": int(closest["step"]),
        "final_radius_error": final_diag.radius - target_radius,
        "final_radius_error_ratio": final_diag.r_error_ratio,
        "radial_velocity_trend": summarize_radial_velocity(vr_values),
        "closest_vr_ratio": closest["vr_ratio"],
        "closest_vt_error_ratio": closest["vt_error_ratio"],
        "closest_energy_error_ratio": closest["energy_error_ratio"],
        "closest_angular_momentum_error_ratio": closest["angular_momentum_error_ratio"],
        "final_vr_ratio": final_vr_ratio,
        "final_vt_error_ratio": final_diag.vt_error_ratio,
        "final_energy_error_ratio": final_diag.energy_error_ratio,
        "final_angular_momentum_error_ratio": final_diag.angular_momentum_error_ratio,
        "best_post_cross_step": int(best_post["step"]),
        "best_post_cross_sync": best_post["sync"],
        "best_post_cross_distance": best_post["distance"],
        "radius_crossings_total": int(radius_crossings_total),
        "post_cross_steps": int(post_cross_steps),
        "steps": int(len(radius_errors)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": termination_reason,
        "dominant_stage": dominant_stage,
        "stage_counts_json": json.dumps({key: int(value) for key, value in sorted(stage_counts.items())}, sort_keys=True),
        "max_speed_ratio": float(np.max(speed_arr)) if len(speed_arr) else 0.0,
        "overspeed": bool(len(speed_arr) and np.max(speed_arr) > 1.90),
        "instability": bool(instability),
        "failure_label": failure_label,
        "deadness": deadness,
    }
    if record_trajectory:
        row["trajectory"] = trajectory
        row["target_radius"] = target_radius
    return row


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for name in [variant.name for variant in VARIANTS]:
        subset = [row for row in rows if row.get("upstream_variant") == name]
        if not subset:
            continue
        crossing_subset = [row for row in subset if bool_from_csv(row.get("crossing_occurs"))]
        post_distances = [as_float(row.get("best_post_cross_distance")) for row in crossing_subset if math.isfinite(as_float(row.get("best_post_cross_distance")))]
        stats[name] = {
            "cases": len(subset),
            "crossing_count": sum(bool_from_csv(row.get("crossing_occurs")) for row in subset),
            "recoverable_crossing_count": sum(bool_from_csv(row.get("recoverable_crossing")) for row in subset),
            "success_count": sum(bool_from_csv(row.get("success")) for row in subset),
            "non_crossing_count": sum(bool_from_csv(row.get("non_crossing")) for row in subset),
            "overspeed": sum(bool_from_csv(row.get("overspeed")) for row in subset),
            "instability": sum(bool_from_csv(row.get("instability")) for row in subset),
            "mean_best_crossing_potential": float(np.mean([as_float(row.get("best_crossing_potential")) for row in subset])),
            "mean_min_abs_radius_error_ratio": float(np.mean([as_float(row.get("min_abs_radius_error_ratio")) for row in subset])),
            "mean_best_post_cross_distance": float(np.mean(post_distances)) if post_distances else float("nan"),
        }
    return stats


def build_non_crossing_diagnosis(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    baseline = [row for row in rows if row.get("upstream_variant") == "baseline_phase34" and not bool_from_csv(row.get("crossing_occurs"))]
    out: List[Dict[str, object]] = []
    for row in baseline:
        out.append(
            {
                "case_id": case_id(as_float(row["r0_over_target"]), as_float(row["initial_velocity_angle_deg"]), as_float(row["thrust_scale"])),
                "r0_over_target": row["r0_over_target"],
                "initial_velocity_angle_deg": row["initial_velocity_angle_deg"],
                "thrust_scale": row["thrust_scale"],
                "minimum_radius_error": row["min_abs_radius_error"],
                "minimum_radius_error_ratio": row["min_abs_radius_error_ratio"],
                "final_radius_error": row["final_radius_error"],
                "final_radius_error_ratio": row["final_radius_error_ratio"],
                "radial_velocity_trend": row.get("radial_velocity_trend", ""),
                "closest_vr_ratio": row["closest_vr_ratio"],
                "final_vr_ratio": row["final_vr_ratio"],
                "closest_vt_error_ratio": row["closest_vt_error_ratio"],
                "final_vt_error_ratio": row["final_vt_error_ratio"],
                "closest_energy_error_ratio": row["closest_energy_error_ratio"],
                "final_energy_error_ratio": row["final_energy_error_ratio"],
                "closest_angular_momentum_error_ratio": row["closest_angular_momentum_error_ratio"],
                "final_angular_momentum_error_ratio": row["final_angular_momentum_error_ratio"],
                "closest_approach_step": row["closest_approach_step"],
                "best_crossing_potential": row["best_crossing_potential"],
                "crossing_potential_at_closest": row["crossing_potential_at_closest"],
                "near_or_dead": row["deadness"],
                "failure_label": row["failure_label"],
            }
        )
    return out


def load_phase34_baseline_stats() -> Dict[str, object]:
    rows = [row for row in read_csv(PHASE34_RESULTS) if row.get("post_cross_mode") == "radius_priority"]
    return {
        "cases": len(rows),
        "crossing_count": sum(bool_from_csv(row.get("crossing_occurs")) for row in rows),
        "recoverable_crossing_count": sum(bool_from_csv(row.get("recoverable_crossing")) for row in rows),
        "success_count": sum(bool_from_csv(row.get("success")) for row in rows),
        "non_crossing_count": sum(not bool_from_csv(row.get("crossing_occurs")) for row in rows),
    }


def save_plots(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]], examples: Dict[str, Dict[str, object]]) -> None:
    labels = [variant.name for variant in VARIANTS]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.bar(x - 0.2, [stats[label]["crossing_count"] for label in labels], 0.4, label="geometric crossings", color="#4C78A8")
    ax.bar(x + 0.2, [stats[label]["recoverable_crossing_count"] for label in labels], 0.4, label="recoverable crossings", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 24)
    ax.set_ylabel("cases out of 24")
    ax.set_title("Phase35 crossing basin comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_CROSSING_COUNTS, dpi=220)
    plt.close(fig)

    diagnosis_rows = build_non_crossing_diagnosis(rows)
    failure_counts = Counter(str(row["failure_label"]) for row in diagnosis_rows)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    names = list(failure_counts.keys())
    values = [failure_counts[name] for name in names]
    ax.bar(names, values, color="#E45756")
    ax.set_ylabel("baseline non-crossing cases")
    ax.set_title("Phase35 non-crossing failure modes")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_FAILURE_MODES, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    data = [[as_float(row.get("best_crossing_potential")) for row in rows if row.get("upstream_variant") == label] for label in labels]
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("crossing potential score")
    ax.set_title("Crossing potential distribution")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_POTENTIAL, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    data = [[as_float(row.get("min_abs_radius_error_ratio")) for row in rows if row.get("upstream_variant") == label] for label in labels]
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("minimum |radius error| / target")
    ax.set_title("Minimum radius error by upstream variant")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_MIN_RADIUS, dpi=220)
    plt.close(fig)

    if examples:
        fig, axes = plt.subplots(3, 1, figsize=(10.4, 7.5), sharex=True)
        for name, row in examples.items():
            traj = row["trajectory"]
            t = np.asarray(traj["time_step"], dtype=np.float64) * DT / 3600.0
            axes[0].plot(t, traj["radius_ratio"], linewidth=1.1, label=name)
            axes[1].plot(t, traj["crossing_potential"], linewidth=1.1, label=name)
            axes[2].plot(t, traj["distance_to_recoverable"], linewidth=1.1, label=name)
        axes[0].axhline(1.0, color="#333333", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("r / target")
        axes[1].set_ylabel("crossing potential")
        axes[2].set_ylabel("recoverability distance")
        axes[2].set_xlabel("time [hours]")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        axes[0].legend(fontsize=8)
        axes[0].set_title("Phase34 terminal handoff examples")
        fig.tight_layout()
        fig.savefig(PLOT_HANDOFF, dpi=220)
        plt.close(fig)


def write_markdown(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]]) -> None:
    phase34_stats = load_phase34_baseline_stats()
    best_cross = max(stats, key=lambda name: stats[name]["crossing_count"])
    best_recoverable = max(stats, key=lambda name: stats[name]["recoverable_crossing_count"])
    baseline = stats["baseline_phase34"]
    diagnosis_rows = build_non_crossing_diagnosis(rows)
    failure_counts = Counter(str(row["failure_label"]) for row in diagnosis_rows)
    top_count = max(failure_counts.values()) if failure_counts else 0
    dominant_failures = sorted([name for name, count in failure_counts.items() if count == top_count])
    dominant_failure = ", ".join(dominant_failures) if dominant_failures else "none"
    crossing_improved = stats[best_cross]["crossing_count"] > baseline["crossing_count"]
    recoverable_improved = stats[best_recoverable]["recoverable_crossing_count"] > baseline["recoverable_crossing_count"]
    preserves_downstream = all(
        item["recoverable_crossing_count"] == item["crossing_count"]
        for item in stats.values()
        if item["crossing_count"] > 0 and item["overspeed"] == 0
    )
    if crossing_improved and recoverable_improved:
        honesty = "Phase35 successfully feeds new trajectories into the Phase34 terminal controller."
    elif crossing_improved:
        honesty = "Phase35 expanded geometric crossing but not recoverable insertion."
    else:
        honesty = "Phase35 diagnosed the upstream bottleneck but did not expand the crossing basin."

    lines = [
        "# Phase 35 Crossing Basin Expansion",
        "",
        "## Scope",
        "",
        "- Python-only 2D explicit-controller benchmark.",
        "- Phase35 acts only before the first target-radius crossing.",
        "- Phase34 `radius_priority` post-cross synchronization remains the terminal controller after crossing.",
        "- Physics, CAPTURE/LOCK thresholds, reward assumptions, and recoverability thresholds are unchanged.",
        "- Benchmark: same 24-case reduced grid used in Phase34.",
        "",
        "## Results",
        "",
        "| Upstream variant | Cases | Geometric crossings | Recoverable crossings | Success label | Non-crossing | Mean crossing potential | Mean min radius error ratio | Overspeed | Instability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in [variant.name for variant in VARIANTS]:
        item = stats[name]
        lines.append(
            f"| `{name}` | {item['cases']} | {item['crossing_count']} | {item['recoverable_crossing_count']} | {item['success_count']} | "
            f"{item['non_crossing_count']} | {item['mean_best_crossing_potential']:.4f} | {item['mean_min_abs_radius_error_ratio']:.6f} | "
            f"{item['overspeed']} | {item['instability']} |"
        )
    lines.extend(
        [
            "",
            "## Scientific Questions",
            "",
            f"1. Did Phase35 increase crossing-producing cases? `{'yes' if crossing_improved else 'no'}`. Best crossing count: `{stats[best_cross]['crossing_count']} / 24` from `{best_cross}`; baseline Phase34 count: `{baseline['crossing_count']} / 24`.",
            f"2. Which failure mode dominates the 16 non-crossing cases? `{dominant_failure}` with `{top_count}` cases each in the baseline diagnosis.",
            f"3. Did upstream expansion preserve Phase34 downstream recoverability? `{'yes' if preserves_downstream else 'no'}` under this benchmark aggregation.",
            f"4. Is the bottleneck pre-cross energy, tangential corridor, or initial geometry? Current labels point primarily to `{dominant_failure}` rather than post-cross recovery.",
            f"5. Should Phase36 integrate a planner, MPC-lite, or stronger transfer family? `{'yes' if not crossing_improved else 'probably'}`. The pre-cross module needs a more expressive transfer planner than these local biases.",
            "",
            "## Interpretation",
            "",
            honesty,
            "",
            "Phase35 keeps the Phase34 terminal law fixed. Any change in crossing count comes from upstream routing, not from relaxed recoverability or post-cross threshold changes.",
            "",
            "## Artifacts",
            "",
            "- `phase35_results.csv`",
            "- `non_crossing_diagnosis.csv`",
            "- `failure_mode_summary.md`",
            "- `phase35_vs_phase34.md`",
            "- `crossing_count_comparison.png`",
            "- `non_crossing_failure_modes.png`",
            "- `crossing_potential_distribution.png`",
            "- `min_radius_error_by_variant.png`",
            "- `phase34_terminal_handoff_examples.png`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")

    failure_lines = [
        "# Phase 35 Non-Crossing Failure Mode Summary",
        "",
        "Baseline Phase34 non-crossing cases are diagnosed using the Phase35 metric set, not by changing Phase34 behavior.",
        "",
        "| Failure label | Cases |",
        "|---|---:|",
    ]
    for name, count in failure_counts.most_common():
        failure_lines.append(f"| `{name}` | {count} |")
    failure_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Baseline non-crossing cases diagnosed: `{len(diagnosis_rows)}`.",
            f"- Dominant label: `{dominant_failure}`.",
            "- `near_crossing` means the trajectory came close enough that better local timing may matter.",
            "- `over_conservative_transfer` means the trajectory stayed near the target-radius boundary but did not commit to crossing.",
            "- `wrong_tangential_corridor` means angular momentum or tangential velocity alignment is the main blocker.",
            "- `insufficient_radial_energy` means the trajectory did not carry enough useful radial motion toward the target radius.",
            "- `bad_initial_geometry` and `dead_geometry` are stronger signs that a local bias is not enough.",
        ]
    )
    FAILURE_MD.write_text("\n".join(failure_lines), encoding="utf-8")

    compare_lines = [
        "# Phase 35 vs Phase 34",
        "",
        "| Metric | Phase34 radius_priority reference | Phase35 best crossing variant | Phase35 best recoverable variant |",
        "|---|---:|---:|---:|",
        f"| Cases | {phase34_stats['cases']} | {stats[best_cross]['cases']} | {stats[best_recoverable]['cases']} |",
        f"| Geometric crossings | {phase34_stats['crossing_count']} | {stats[best_cross]['crossing_count']} | {stats[best_recoverable]['crossing_count']} |",
        f"| Recoverable crossings | {phase34_stats['recoverable_crossing_count']} | {stats[best_cross]['recoverable_crossing_count']} | {stats[best_recoverable]['recoverable_crossing_count']} |",
        f"| Success label | {phase34_stats['success_count']} | {stats[best_cross]['success_count']} | {stats[best_recoverable]['success_count']} |",
        f"| Non-crossing cases | {phase34_stats['non_crossing_count']} | {stats[best_cross]['non_crossing_count']} | {stats[best_recoverable]['non_crossing_count']} |",
        "",
        "## Structural Reading",
        "",
        "- Phase34 is the fixed terminal/post-cross controller.",
        "- Phase35 changes only pre-cross routing before first target-radius crossing.",
        f"- Crossing count improved over Phase34: `{'yes' if crossing_improved else 'no'}`.",
        f"- Recoverable crossing count improved over Phase34: `{'yes' if recoverable_improved else 'no'}`.",
        f"- Final honesty statement: {honesty}",
    ]
    COMPARE_MD.write_text("\n".join(compare_lines), encoding="utf-8")


def benchmark_rows(record_examples: bool = False) -> tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    examples: Dict[str, Dict[str, object]] = {}
    for variant in VARIANTS:
        for r0, angle, thrust in iter_cases():
            record = record_examples and r0 == 1.0 and angle == 175.0 and thrust == 10000.0
            row = rollout_phase35_case(variant, PHASE34_TERMINAL_MODE, r0, angle, thrust, TARGET_RADIUS_SCALE, record_trajectory=record)
            rows.append(row)
            if record:
                examples[variant.name] = row
            print(
                f"phase35 variant={variant.name} r0={r0:g} angle={angle:g} thrust={thrust:g} "
                f"cross={row['crossing_occurs']} rec={row['recoverable_crossing']} "
                f"potential={as_float(row['best_crossing_potential']):.4f} reason={row['termination_reason']}"
            )
    return rows, examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase35 crossing basin expansion benchmark.")
    parser.add_argument("--skip-plots", action="store_true", help="Write CSV and markdown without regenerating plots.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, examples = benchmark_rows(record_examples=not args.skip_plots)
    stats = aggregate(rows)
    diagnosis_rows = build_non_crossing_diagnosis(rows)
    write_csv(RESULTS_CSV, FIELDNAMES, rows)
    write_csv(DIAGNOSIS_CSV, DIAGNOSIS_FIELDS, diagnosis_rows)
    if not args.skip_plots:
        save_plots(rows, stats, examples)
    write_markdown(rows, stats)
    print(f"Saved Phase35 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
