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
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase26_tangential_velocity_corridor"
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
    POST_CROSS_WINDOW,
    PREWINDOW_OUTER_RATIO,
    RECOVERABLE_R_RATIO,
    RECOVERABLE_VR_RATIO,
    RECOVERABLE_VT_RATIO,
    STRICT_CFG,
    TAIL_LEN,
    TARGET_RADIUS_SCALE,
    THRUST_VALUES,
    ANGLE_VALUES,
    R0_VALUES,
    WS_BAND_RATIO,
    bool_from_csv,
    clamp,
    normalize_action,
    orbital_diagnostics,
    orbit_crosses_target,
    radial_tangential_action,
)


PHASE20_CSV = PROJECT_ROOT / "analysis" / "phase20_predictive_planner" / "reduced_grid_results.csv"
PHASE21_CSV = PROJECT_ROOT / "analysis" / "phase21_orbital_transfer_planner" / "phase21_results.csv"
PHASE22_CSV = PROJECT_ROOT / "analysis" / "phase22_two_burn_transfer" / "phase22_results.csv"
PHASE23_CSV = PROJECT_ROOT / "analysis" / "phase23_windowed_insertion_solver" / "phase23_results.csv"
PHASE24_CSV = PROJECT_ROOT / "analysis" / "phase24_precision_insertion_geometry" / "phase24_results.csv"

RESULTS_CSV = OUTPUT_DIR / "phase26_results.csv"
CORRIDOR_CSV = OUTPUT_DIR / "phase26_vt_corridor_analysis.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"

VT_TIME_PNG = OUTPUT_DIR / "vt_error_over_time.png"
CROSSING_VT_PNG = OUTPUT_DIR / "crossing_vt_distribution.png"
BAND_PROGRESSION_PNG = OUTPUT_DIR / "vt_corridor_band_progression.png"
CONTROLLER_COMPARISON_PNG = OUTPUT_DIR / "controller_comparison.png"
PRE_POST_BURN_PNG = OUTPUT_DIR / "pre_post_burn_vt_correction.png"
RECOVERABILITY_DISTANCE_PNG = OUTPUT_DIR / "recoverability_distance_progression.png"

TRAJECTORY_DIR = OUTPUT_DIR / "trajectories"

BURN_A_MAX_STEPS = 24
BURN_A_MIN_STEPS = 10
BURN_A_MAX_NORM = 0.11
COAST_MIN_STEPS = 80
COAST_MAX_STEPS = 42000
INSERTION_BAND_RATIO = 2.5e-2
HANDOFF_R_RATIO = 7.5e-4
HANDOFF_VR_RATIO = 2.8e-2
HANDOFF_VT_RATIO = 2.8e-1
SAFE_SPEED_RATIO = 1.65


REFERENCE_FIELDNAMES = [
    "controller_name",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "target_radius_scale",
    "success",
    "capture_entered",
    "lock_entered",
    "crossing_occurs",
    "crossing_vr",
    "crossing_vt_error",
    "crossing_vr_ratio",
    "crossing_vt_error_ratio",
    "crossing_radius_error",
    "recoverable_crossing",
    "crossing_score",
    "recoverability_score",
    "near_miss",
    "radius_crossings_total",
    "first_crossing_step",
    "minimum_abs_radius_error",
    "final_radius_error",
    "tail_mean_abs_vr",
    "steps",
    "terminated",
    "truncated",
    "termination_reason",
    "dominant_transfer_stage",
    "transfer_stage_counts_json",
    "intercept_classification_counts_json",
    "initial_energy_error_ratio",
    "final_energy_error_ratio",
    "initial_angular_momentum_error_ratio",
    "final_angular_momentum_error_ratio",
    "mean_energy_error_ratio",
    "mean_angular_momentum_error_ratio",
    "max_speed_ratio",
    "overspeed",
    "turning_point_count",
    "burn_a_steps",
    "burn_b_steps",
    "coast_arc_steps",
    "safe_coast_steps",
    "burn_a_count",
    "burn_b_count",
    "insertion_attempts",
    "insertion_windows",
    "successful_insertion_windows",
    "handoff_ready_count",
    "handoff_entered",
    "handoff_rate",
    "predicted_crossing_step",
    "burn_b_solver_evals",
    "burn_b_solver_score",
    "insertion_window_conversion_rate",
    "chosen_burn_timing",
    "chosen_burn_angle_deg",
    "chosen_burn_magnitude",
    "chosen_burn_duration",
    "chosen_insertion_mode",
    "pre_burn_state_json",
    "post_burn_state_json",
    "basin_score",
    "handoff_success",
]

PHASE26_FIELDNAMES = REFERENCE_FIELDNAMES + [
    "controller_variant",
    "corridor_strategy",
    "initial_vt_error_ratio",
    "final_vt_error_ratio",
    "min_abs_vt_error_ratio",
    "mean_abs_vt_error_ratio",
    "best_vt_error_before_crossing",
    "best_vt_band",
    "best_vt_band_rank",
    "crossing_vt_band",
    "min_distance_to_recoverable",
    "mean_distance_to_recoverable",
    "pre_burn_vt_error_ratio",
    "post_burn_vt_error_ratio",
    "pre_post_burn_vt_correction",
    "vt_improvement_burn_a_to_b",
    "corridor_burn_steps",
    "corridor_hold_steps",
    "corridor_stage_counts_json",
    "instability",
]

CORRIDOR_FIELDNAMES = [
    "controller_name",
    "case_id",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "target_radius_scale",
    "crossing_occurs",
    "recoverable_crossing",
    "capture_entered",
    "success",
    "crossing_vt_error_ratio",
    "crossing_vt_band",
    "min_abs_vt_error_ratio",
    "mean_abs_vt_error_ratio",
    "best_vt_error_before_crossing",
    "best_vt_band",
    "min_distance_to_recoverable",
    "pre_burn_vt_error_ratio",
    "post_burn_vt_error_ratio",
    "pre_post_burn_vt_correction",
    "vt_improvement_burn_a_to_b",
    "overspeed",
    "instability",
    "termination_reason",
]


@dataclass(frozen=True)
class CorridorVariant:
    name: str
    strategy: str
    vt_gain: float
    vr_gain: float
    radius_gain: float
    max_norm: float
    first_burn_steps: int
    hold_steps: int
    second_burn_steps: int
    timing_bias_steps: int


@dataclass
class CorridorPlan:
    original_wait_steps: int
    wait_steps: int
    first_remaining: int
    hold_remaining: int
    second_remaining: int
    chosen_angle_deg: float
    chosen_magnitude: float
    chosen_duration: int
    pre_burn_state: Dict[str, float]


VARIANTS = [
    CorridorVariant(
        "phase26_vt_aware_scoring",
        "radius_vr_vt_corridor",
        vt_gain=2.7,
        vr_gain=2.2,
        radius_gain=0.30,
        max_norm=0.36,
        first_burn_steps=18,
        hold_steps=0,
        second_burn_steps=0,
        timing_bias_steps=-18,
    ),
    CorridorVariant(
        "phase26_two_step_corridor",
        "two_step_radius_then_vt",
        vt_gain=3.4,
        vr_gain=2.0,
        radius_gain=0.35,
        max_norm=0.34,
        first_burn_steps=10,
        hold_steps=0,
        second_burn_steps=14,
        timing_bias_steps=-30,
    ),
    CorridorVariant(
        "phase26_burn_hold_burn",
        "burn_hold_burn_corridor",
        vt_gain=3.2,
        vr_gain=1.6,
        radius_gain=0.18,
        max_norm=0.32,
        first_burn_steps=8,
        hold_steps=18,
        second_burn_steps=12,
        timing_bias_steps=-42,
    ),
    CorridorVariant(
        "phase26_stronger_tangential",
        "stronger_tangential_correction",
        vt_gain=4.6,
        vr_gain=1.8,
        radius_gain=0.20,
        max_norm=0.46,
        first_burn_steps=22,
        hold_steps=0,
        second_burn_steps=8,
        timing_bias_steps=-24,
    ),
]


def set_seed(seed: int = 7) -> None:
    np.random.seed(seed)


def iter_cases(variants: Sequence[CorridorVariant]) -> Iterable[tuple[CorridorVariant, float, float, float, float]]:
    for variant in variants:
        for thrust in THRUST_VALUES:
            for angle in ANGLE_VALUES:
                for r0 in R0_VALUES:
                    yield variant, r0, angle, thrust, TARGET_RADIUS_SCALE


def as_float(value: object, default: float = float("nan")) -> float:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_csv_rows(path: Path, names: set[str]) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [normalize_reference_row(row) for row in rows if str(row.get("controller_name")) in names]


def normalize_reference_row(row: Dict[str, object]) -> Dict[str, object]:
    out = dict(row)
    if "dominant_transfer_stage" not in out:
        out["dominant_transfer_stage"] = out.get("dominant_planner_mode", "")
    if "transfer_stage_counts_json" not in out:
        out["transfer_stage_counts_json"] = out.get("planner_mode_counts_json", "{}")
    out.setdefault("intercept_classification_counts_json", "{}")
    if "initial_angular_momentum_error_ratio" not in out:
        ratio0 = as_float(out.get("initial_angular_momentum_ratio"))
        ratiof = as_float(out.get("final_angular_momentum_ratio"))
        out["initial_angular_momentum_error_ratio"] = ratio0 - 1.0 if math.isfinite(ratio0) else ""
        out["final_angular_momentum_error_ratio"] = ratiof - 1.0 if math.isfinite(ratiof) else ""
    for key in PHASE26_FIELDNAMES:
        out.setdefault(key, "")
    out.setdefault("controller_variant", out.get("controller_name", ""))
    out.setdefault("corridor_strategy", "")
    out.setdefault("instability", bool_from_csv(out.get("overspeed")))
    return out


def estimate_crossing_steps(radius_error: float, radial_velocity: float) -> int | None:
    if radius_error * radial_velocity >= 0.0 or abs(radial_velocity) < 1.0e-9:
        return None
    steps = abs(radius_error) / (abs(radial_velocity) * DT + 1.0e-12)
    if not math.isfinite(steps) or steps < 0.0:
        return None
    return int(round(steps))


def classify_intercept(diag, target_radius: float, coast_steps: int) -> str:
    v_circ = math.sqrt(MU / target_radius)
    vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
    if (
        abs(diag.r_error_ratio) <= HANDOFF_R_RATIO
        and abs(vr_ratio) <= HANDOFF_VR_RATIO
        and abs(diag.vt_error_ratio) <= HANDOFF_VT_RATIO
    ):
        return "handoff_ready"
    if abs(diag.r_error_ratio) <= INSERTION_BAND_RATIO and diag.r_error_ratio * diag.radial_velocity <= 0.0:
        return "insertion_window"
    if not orbit_crosses_target(diag, target_radius):
        return "no_intercept_possible"
    if abs(diag.vt_error_ratio) <= 0.35 and abs(vr_ratio) <= 0.08 and coast_steps >= COAST_MIN_STEPS:
        return "intercept_possible_good_alignment"
    return "intercept_possible_poor_alignment"


def recoverability_distance(r_ratio: float, vr_ratio: float, vt_ratio: float) -> float:
    if not all(math.isfinite(value) for value in [r_ratio, vr_ratio, vt_ratio]):
        return float("nan")
    dr = abs(r_ratio) / RECOVERABLE_R_RATIO
    dvr = abs(vr_ratio) / RECOVERABLE_VR_RATIO
    dvt = abs(vt_ratio) / RECOVERABLE_VT_RATIO
    return math.sqrt(dr * dr + dvr * dvr + dvt * dvt)


def vt_band(vt_error_ratio: float) -> str:
    value = abs(vt_error_ratio)
    if not math.isfinite(value):
        return "unknown"
    if value <= RECOVERABLE_VT_RATIO:
        return "D_1x_recoverable"
    if value <= 1.5 * RECOVERABLE_VT_RATIO:
        return "C_1p5x"
    if value <= 2.0 * RECOVERABLE_VT_RATIO:
        return "B_2x"
    if value <= 3.0 * RECOVERABLE_VT_RATIO:
        return "A_3x"
    return "outside_3x"


def vt_band_rank(band: str) -> int:
    return {
        "unknown": -1,
        "outside_3x": 0,
        "A_3x": 1,
        "B_2x": 2,
        "C_1p5x": 3,
        "D_1x_recoverable": 4,
    }.get(band, -1)


def state_payload(diag, v_circ: float) -> Dict[str, float]:
    return {
        "r_error_ratio": float(diag.r_error_ratio),
        "vr_ratio": float(diag.radial_velocity / (v_circ + 1.0e-12)),
        "vt_error_ratio": float(diag.vt_error_ratio),
        "energy_error_ratio": float(diag.energy_error_ratio),
        "angular_momentum_error_ratio": float(diag.angular_momentum_error_ratio),
    }


def rollout_phase26_case(
    variant: CorridorVariant,
    r0: float,
    angle: float,
    thrust_scale: float,
    target_scale: float,
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
    prev_controller_r_error: float | None = None
    prev_r_error: float | None = None
    prev_vr_for_turn: float | None = None
    radius_errors: List[float] = []
    vr_values: List[float] = []
    speed_ratios: List[float] = []
    energy_errors: List[float] = []
    h_errors: List[float] = []
    vt_errors_abs: List[float] = []
    distance_samples: List[float] = []
    crossing_recovery_samples: List[float] = []
    stage_counts: Counter[str] = Counter()
    corridor_stage_counts: Counter[str] = Counter()
    intercept_counts: Counter[str] = Counter()
    success_counter = 0
    radial_stall_counter = 0
    radius_crossings_total = 0
    first_crossing_step: int | None = None
    crossing_vr: float | None = None
    crossing_vt_error: float | None = None
    crossing_radius_error: float | None = None
    capture_entered = False
    lock_entered = False
    success = False
    terminated = False
    truncated = False
    termination_reason = "max_steps"
    turning_point_count = 0
    overspeed_hit = False
    burn_a_steps = 0
    burn_b_steps = 0
    corridor_burn_steps = 0
    corridor_hold_steps = 0
    coast_arc_steps = 0
    safe_coast_steps = 0
    burn_a_count = 1
    burn_b_count = 0
    insertion_attempts = 0
    insertion_windows = 0
    successful_insertion_windows = 0
    successful_insertion_window_seen = False
    handoff_ready_count = 0
    handoff_entered = False
    predicted_crossing_step: int | None = None
    plan: CorridorPlan | None = None
    post_burn_state: Dict[str, float] = {}
    chosen_mode = variant.strategy
    stage_step = 0
    burn_a_target_steps = int(clamp(BURN_A_MIN_STEPS + 420.0 * abs(r0 - 1.0), BURN_A_MIN_STEPS, BURN_A_MAX_STEPS))
    initial_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
    initial_energy_error = abs(initial_diag.energy_error_ratio)
    initial_h_error = initial_diag.angular_momentum_error_ratio
    initial_vt_error = initial_diag.vt_error_ratio
    best_vt_before_crossing = abs(initial_vt_error)
    vt_at_burn_a_end: float | None = None
    pre_burn_vt_error: float | None = None
    post_burn_vt_error: float | None = None

    trajectory: Dict[str, List[float | str]] = {
        "time_step": [],
        "radius_ratio": [],
        "vt_error_ratio": [],
        "radial_velocity_ratio": [],
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
        next_vy = state_y * 0.0 + state_vy + ay * DT
        next_x = state_x + next_vx * DT
        next_y = state_y + next_vy * DT
        return next_x, next_y, next_vx, next_vy

    def baseline_action(state_vx: float, state_vy: float) -> tuple[float, float]:
        speed = math.sqrt(state_vx * state_vx + state_vy * state_vy)
        if speed <= 1.0e-12:
            return 0.0, 0.0
        return -state_vx / speed, -state_vy / speed

    def candidate_score(state_x: float, state_y: float, state_vx: float, state_vy: float) -> float:
        diag = orbital_diagnostics(state_x, state_y, state_vx, state_vy, target_radius)
        return (
            BASE_PARAMS["score_r"] * abs(diag.r_error_ratio)
            + BASE_PARAMS["score_vr"] * abs(diag.radial_velocity) / (v_circ + 1.0e-12)
            + BASE_PARAMS["score_vt"] * abs(diag.vt_error_ratio)
            + BASE_PARAMS["score_energy"] * abs(diag.energy_error_ratio)
        )

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
        z = clamp(abs_r_error_ratio / (WS_BAND_RATIO + 1.0e-12), 0.0, 1.0)
        scale = BASE_PARAMS["min_scale"] + (BASE_PARAMS["max_scale"] - BASE_PARAMS["min_scale"]) * z
        candidates = [
            (retro_x, retro_y),
            (clamp(retro_x - BASE_PARAMS["inward_bias"] * r_hat_x, -1.0, 1.0), clamp(retro_y - BASE_PARAMS["inward_bias"] * r_hat_y, -1.0, 1.0)),
            (scale * retro_x, scale * retro_y),
        ]
        best = (retro_x, retro_y)
        best_score = float("inf")
        for candidate_x, candidate_y in candidates:
            candidate_x, candidate_y = normalize_action(candidate_x, candidate_y)
            next_x, next_y, next_vx, next_vy = env_step(state_x, state_y, state_vx, state_vy, candidate_x, candidate_y)
            score = candidate_score(next_x, next_y, next_vx, next_vy)
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
        if current_phase == "CAPTURE":
            radial_cmd = -(cfg.capture_radial_pos_gain * diag.r_error_ratio + cfg.capture_radial_vel_gain * diag.radial_velocity / (v_circ + 1.0e-12))
            tangential_cmd = -(cfg.capture_tangential_gain * diag.vt_error_ratio)
            radial_cmd = clamp(radial_cmd, -cfg.capture_radial_limit, cfg.capture_radial_limit)
            tangential_cmd = clamp(tangential_cmd, -cfg.capture_tangential_limit, cfg.capture_tangential_limit)
        else:
            radial_cmd = -(cfg.lock_radial_pos_gain * diag.r_error_ratio + cfg.lock_radial_vel_gain * diag.radial_velocity / (v_circ + 1.0e-12))
            tangential_cmd = -(cfg.lock_tangential_gain * diag.vt_error_ratio)
            radial_cmd = clamp(radial_cmd, -cfg.lock_radial_limit, cfg.lock_radial_limit)
            tangential_cmd = clamp(tangential_cmd, -cfg.lock_tangential_limit, cfg.lock_tangential_limit)
        return radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, 1.0)

    def burn_a_action(state_x: float, state_y: float, state_vx: float, state_vy: float, diag) -> tuple[float, float]:
        abs_r_ratio = abs(diag.r_error_ratio)
        if diag.r_error_ratio > 0.0:
            tangential_cmd = -1.0
        elif diag.r_error_ratio < 0.0:
            tangential_cmd = 1.0
        else:
            tangential_cmd = 1.0 if diag.angular_momentum_error_ratio < 0.0 else -1.0
        radial_cmd = clamp(-0.06 * diag.radial_velocity / (v_circ + 1.0e-12), -0.04, 0.04)
        max_norm = clamp(0.045 + 1.20 * min(abs_r_ratio, 0.055), 0.045, BURN_A_MAX_NORM)
        return radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, max_norm)

    def build_plan(diag) -> CorridorPlan:
        predicted = estimate_crossing_steps(diag.radius - target_radius, diag.radial_velocity)
        wait_steps = int(clamp(float((predicted or 0) + variant.timing_bias_steps), 0.0, 96.0))
        if variant.strategy == "radius_vr_vt_corridor":
            first = variant.first_burn_steps
            hold = 0
            second = 0
        elif variant.strategy == "two_step_radius_then_vt":
            first = variant.first_burn_steps
            hold = 0
            second = variant.second_burn_steps
        elif variant.strategy == "burn_hold_burn_corridor":
            first = variant.first_burn_steps
            hold = variant.hold_steps
            second = variant.second_burn_steps
        else:
            first = variant.first_burn_steps
            hold = 0
            second = variant.second_burn_steps
        vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
        radial_cmd = clamp(-variant.vr_gain * vr_ratio - variant.radius_gain * diag.r_error_ratio, -1.0, 1.0)
        tangential_cmd = clamp(-variant.vt_gain * diag.vt_error_ratio, -1.0, 1.0)
        norm = math.sqrt(radial_cmd * radial_cmd + tangential_cmd * tangential_cmd)
        magnitude = min(variant.max_norm, norm)
        chosen_angle = math.degrees(math.atan2(radial_cmd, tangential_cmd)) if norm > 1.0e-12 else 0.0
        return CorridorPlan(
            original_wait_steps=wait_steps,
            wait_steps=wait_steps,
            first_remaining=first,
            hold_remaining=hold,
            second_remaining=second,
            chosen_angle_deg=chosen_angle,
            chosen_magnitude=magnitude,
            chosen_duration=first + hold + second,
            pre_burn_state=state_payload(diag, v_circ),
        )

    def corridor_action(state_x: float, state_y: float, state_vx: float, state_vy: float, diag) -> tuple[float, float, str]:
        nonlocal plan, pre_burn_vt_error, post_burn_vt_error, post_burn_state
        if plan is None:
            plan = build_plan(diag)
            pre_burn_vt_error = diag.vt_error_ratio
        if plan.wait_steps > 0:
            plan.wait_steps -= 1
            return 0.0, 0.0, "burn_b_timing_wait"
        if plan.first_remaining > 0:
            plan.first_remaining -= 1
            vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
            if variant.strategy in {"two_step_radius_then_vt", "burn_hold_burn_corridor"}:
                radial_cmd = clamp(-variant.vr_gain * vr_ratio - variant.radius_gain * diag.r_error_ratio, -0.85, 0.85)
                tangential_cmd = clamp(-0.55 * variant.vt_gain * diag.vt_error_ratio, -0.85, 0.85)
            else:
                radial_cmd = clamp(-variant.vr_gain * vr_ratio - variant.radius_gain * diag.r_error_ratio, -0.9, 0.9)
                tangential_cmd = clamp(-variant.vt_gain * diag.vt_error_ratio, -1.0, 1.0)
            return (*radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, variant.max_norm), "corridor_burn_1")
        if plan.hold_remaining > 0:
            plan.hold_remaining -= 1
            return 0.0, 0.0, "corridor_hold"
        if plan.second_remaining > 0:
            plan.second_remaining -= 1
            vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
            if variant.strategy == "two_step_radius_then_vt":
                radial_cmd = clamp(-0.95 * variant.vr_gain * vr_ratio, -0.65, 0.65)
                tangential_cmd = clamp(-1.25 * variant.vt_gain * diag.vt_error_ratio, -1.0, 1.0)
            elif variant.strategy == "burn_hold_burn_corridor":
                radial_cmd = clamp(-variant.vr_gain * vr_ratio - 0.12 * diag.r_error_ratio, -0.65, 0.65)
                tangential_cmd = clamp(-1.35 * variant.vt_gain * diag.vt_error_ratio, -1.0, 1.0)
            else:
                radial_cmd = clamp(-0.70 * variant.vr_gain * vr_ratio, -0.55, 0.55)
                tangential_cmd = clamp(-1.50 * variant.vt_gain * diag.vt_error_ratio, -1.0, 1.0)
            action_x, action_y = radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, variant.max_norm)
            if plan.second_remaining == 0:
                post_burn_vt_error = diag.vt_error_ratio
                post_burn_state = state_payload(diag, v_circ)
            return action_x, action_y, "corridor_burn_2"
        if post_burn_vt_error is None:
            post_burn_vt_error = diag.vt_error_ratio
            post_burn_state = state_payload(diag, v_circ)
        return 0.0, 0.0, "phase76_handoff"

    for step in range(1, MAX_STEPS + 1):
        diag = orbital_diagnostics(x, y, vx, vy, target_radius)
        speed_ratio = diag.speed / (v_circ + 1.0e-12)
        coast_age = stage_step if transfer_stage == "coast_arc" else coast_arc_steps
        intercept_class = classify_intercept(diag, target_radius, coast_age)
        intercept_counts[intercept_class] += 1
        predicted = estimate_crossing_steps(diag.radius - target_radius, diag.radial_velocity)
        if predicted is not None and (predicted_crossing_step is None or predicted < predicted_crossing_step):
            predicted_crossing_step = predicted

        real_r_error = diag.radius - target_radius
        v_r_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
        v_t_ratio = diag.vt_error_ratio
        distance = recoverability_distance(abs(diag.r_error_ratio), v_r_ratio, v_t_ratio)
        if first_crossing_step is None and phase == "DESCENT" and math.isfinite(distance):
            distance_samples.append(distance)
        vt_errors_abs.append(abs(v_t_ratio))
        if first_crossing_step is None:
            best_vt_before_crossing = min(best_vt_before_crossing, abs(v_t_ratio))
        turning_point_now = prev_vr_for_turn is not None and (prev_vr_for_turn * diag.radial_velocity < 0.0)
        if turning_point_now:
            turning_point_count += 1

        crossed_now = False
        if prev_controller_r_error is not None:
            crossed_now = (prev_controller_r_error > 0.0 and real_r_error <= 0.0) or (
                prev_controller_r_error < 0.0 and real_r_error >= 0.0
            )
        prev_controller_r_error = real_r_error

        if phase == "DESCENT" and crossed_now:
            phase = "CAPTURE"
        elif phase == "CAPTURE":
            if abs(diag.r_error_ratio) < cfg.lock_r_threshold_ratio and abs(v_r_ratio) < cfg.lock_vr_threshold_ratio and abs(v_t_ratio) < cfg.lock_vt_threshold_ratio:
                phase = "LOCK"
        elif phase == "LOCK":
            if abs(diag.r_error_ratio) > cfg.unlock_r_threshold_ratio or abs(v_r_ratio) > cfg.unlock_vr_threshold_ratio:
                phase = "CAPTURE"

        energy_errors.append(abs(diag.energy_error_ratio))
        h_errors.append(abs(diag.angular_momentum_error_ratio))

        if phase == "DESCENT":
            if speed_ratio > SAFE_SPEED_RATIO or diag.radius < 0.42 * target_radius or diag.radius > 2.25 * target_radius:
                action_x, action_y = 0.0, 0.0
                active_stage = "safe_coast"
                safe_coast_steps += 1
            elif abs(diag.r_error_ratio) <= PREWINDOW_OUTER_RATIO:
                action_x, action_y = soft_linear_3e4_action(x, y, vx, vy, diag.radius, diag.r_error_ratio, abs(diag.r_error_ratio))
                active_stage = "phase76_handoff"
                handoff_ready_count += 1
                handoff_entered = True
                transfer_stage = "phase76_handoff"
            elif intercept_class == "handoff_ready":
                action_x, action_y = soft_linear_3e4_action(x, y, vx, vy, diag.radius, diag.r_error_ratio, abs(diag.r_error_ratio))
                active_stage = "phase76_handoff"
                handoff_ready_count += 1
                handoff_entered = True
            elif transfer_stage == "burn_a":
                action_x, action_y = burn_a_action(x, y, vx, vy, diag)
                active_stage = "burn_a"
                burn_a_steps += 1
                if burn_a_steps >= burn_a_target_steps:
                    vt_at_burn_a_end = diag.vt_error_ratio
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
                    burn_b_count += 1
                    insertion_attempts += 1
                    insertion_windows += 1
                elif coast_age >= COAST_MAX_STEPS and intercept_class == "no_intercept_possible":
                    transfer_stage = "abort_safe_coast"
                    stage_step = 0
            elif transfer_stage == "burn_b":
                action_x, action_y, active_stage = corridor_action(x, y, vx, vy, diag)
                if active_stage.startswith("corridor_burn"):
                    burn_b_steps += 1
                    corridor_burn_steps += 1
                if active_stage == "corridor_hold":
                    corridor_hold_steps += 1
                if active_stage.startswith("corridor") or active_stage == "burn_b_timing_wait":
                    corridor_stage_counts[active_stage] += 1
                if intercept_class in {"handoff_ready", "insertion_window"} and not successful_insertion_window_seen:
                    successful_insertion_windows += 1
                    successful_insertion_window_seen = True
                if active_stage == "phase76_handoff" or abs(diag.r_error_ratio) <= HANDOFF_R_RATIO:
                    transfer_stage = "phase76_handoff"
                    stage_step = 0
            elif transfer_stage == "phase76_handoff":
                action_x, action_y = soft_linear_3e4_action(x, y, vx, vy, diag.radius, diag.r_error_ratio, abs(diag.r_error_ratio))
                active_stage = "phase76_handoff"
                handoff_entered = True
            else:
                action_x, action_y = 0.0, 0.0
                active_stage = "abort_safe_coast"
                safe_coast_steps += 1
            stage_counts[active_stage] += 1
            stage_step += 1
        else:
            action_x, action_y = capture_lock_action(x, y, vx, vy, phase)
            active_stage = phase.lower()

        capture_entered = capture_entered or phase == "CAPTURE"
        lock_entered = lock_entered or phase == "LOCK"
        x, y, vx, vy = env_step(x, y, vx, vy, action_x, action_y)
        next_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
        r_error = next_diag.radius - target_radius
        radius_errors.append(r_error)
        vr_values.append(next_diag.radial_velocity)
        speed_ratios.append(next_diag.speed / (v_circ + 1.0e-12))

        if record_trajectory:
            next_vr_ratio = next_diag.radial_velocity / (v_circ + 1.0e-12)
            trajectory["time_step"].append(float(step))
            trajectory["radius_ratio"].append(next_diag.radius / target_radius)
            trajectory["vt_error_ratio"].append(next_diag.vt_error_ratio)
            trajectory["radial_velocity_ratio"].append(next_vr_ratio)
            trajectory["distance_to_recoverable"].append(recoverability_distance(abs(next_diag.r_error_ratio), next_vr_ratio, next_diag.vt_error_ratio))
            trajectory["stage"].append(active_stage)

        if prev_r_error is not None and ((prev_r_error > 0.0 and r_error <= 0.0) or (prev_r_error < 0.0 and r_error >= 0.0)):
            radius_crossings_total += 1
            if first_crossing_step is None:
                first_crossing_step = step
                crossing_vr = next_diag.radial_velocity
                crossing_vt_error = next_diag.tangential_velocity - v_circ
                crossing_radius_error = r_error
        prev_r_error = r_error
        if first_crossing_step is not None and len(crossing_recovery_samples) < POST_CROSS_WINDOW:
            crossing_recovery_samples.append(
                abs(r_error) / target_radius
                + 0.70 * abs(next_diag.radial_velocity) / (v_circ + 1.0e-12)
                + 0.50 * abs(next_diag.vt_error_ratio)
            )

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
        overspeed_hit = overspeed_hit or overspeed
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
        if terminated or truncated:
            break
        prev_vr_for_turn = next_diag.radial_velocity

    rerr = np.asarray(radius_errors, dtype=np.float64)
    vr_arr = np.asarray(vr_values, dtype=np.float64)
    tail = vr_arr[-min(TAIL_LEN, len(vr_arr)) :] if len(vr_arr) else np.asarray([0.0])
    min_abs_rerr = float(np.min(np.abs(rerr))) if len(rerr) else 0.0
    near_miss = (not success) and min_abs_rerr / target_radius <= NEAR_MISS_REL
    final_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
    dominant_stage = max(stage_counts.items(), key=lambda item: item[1])[0] if stage_counts else ""
    crossing_vr_ratio_value = float(crossing_vr / (v_circ + 1.0e-12)) if crossing_vr is not None else None
    crossing_vt_error_ratio_value = float(crossing_vt_error / (v_circ + 1.0e-12)) if crossing_vt_error is not None else None
    crossing_r_ratio_value = float(abs(crossing_radius_error) / target_radius) if crossing_radius_error is not None else None
    crossing_score = (
        crossing_r_ratio_value + 1.20 * abs(crossing_vr_ratio_value) + 0.80 * abs(crossing_vt_error_ratio_value)
        if crossing_vr_ratio_value is not None and crossing_vt_error_ratio_value is not None and crossing_r_ratio_value is not None
        else ""
    )
    recoverability_score = float(np.mean(crossing_recovery_samples)) if crossing_recovery_samples else ""
    recoverable_crossing = bool(
        crossing_vr_ratio_value is not None
        and crossing_vt_error_ratio_value is not None
        and crossing_r_ratio_value is not None
        and recoverability_score != ""
        and crossing_r_ratio_value < RECOVERABLE_R_RATIO
        and abs(crossing_vr_ratio_value) < RECOVERABLE_VR_RATIO
        and abs(crossing_vt_error_ratio_value) < RECOVERABLE_VT_RATIO
        and float(recoverability_score) < 0.20
        and not overspeed_hit
    )
    min_abs_vt_error = float(best_vt_before_crossing)
    mean_abs_vt_error = float(np.mean(vt_errors_abs)) if vt_errors_abs else abs(final_diag.vt_error_ratio)
    best_band = vt_band(best_vt_before_crossing)
    crossing_band = vt_band(crossing_vt_error_ratio_value) if crossing_vt_error_ratio_value is not None else "unknown"
    min_distance = float(np.min(distance_samples)) if distance_samples else ""
    mean_distance = float(np.mean(distance_samples)) if distance_samples else ""
    if pre_burn_vt_error is None and plan is not None:
        pre_burn_vt_error = plan.pre_burn_state.get("vt_error_ratio")
    if post_burn_vt_error is None:
        post_burn_vt_error = final_diag.vt_error_ratio
        post_burn_state = state_payload(final_diag, v_circ)
    pre_post_correction = (
        abs(pre_burn_vt_error) - abs(post_burn_vt_error)
        if pre_burn_vt_error is not None and post_burn_vt_error is not None
        else ""
    )
    vt_improvement_burn_a_to_b = (
        abs(vt_at_burn_a_end) - abs(pre_burn_vt_error)
        if vt_at_burn_a_end is not None and pre_burn_vt_error is not None
        else ""
    )
    instability = bool(overspeed_hit or termination_reason in {"out_range", "too_close", "radial_stall"})
    row: Dict[str, object] = {
        "controller_name": variant.name,
        "controller_variant": variant.name,
        "corridor_strategy": variant.strategy,
        "r0_over_target": r0,
        "initial_velocity_angle_deg": angle,
        "thrust_scale": thrust_scale,
        "target_radius_scale": target_scale,
        "success": bool(success),
        "capture_entered": bool(capture_entered),
        "lock_entered": bool(lock_entered),
        "crossing_occurs": bool(radius_crossings_total > 0),
        "crossing_vr": float(crossing_vr) if crossing_vr is not None else "",
        "crossing_vt_error": float(crossing_vt_error) if crossing_vt_error is not None else "",
        "crossing_vr_ratio": crossing_vr_ratio_value if crossing_vr_ratio_value is not None else "",
        "crossing_vt_error_ratio": crossing_vt_error_ratio_value if crossing_vt_error_ratio_value is not None else "",
        "crossing_radius_error": float(crossing_radius_error) if crossing_radius_error is not None else "",
        "recoverable_crossing": bool(recoverable_crossing),
        "crossing_score": crossing_score,
        "recoverability_score": recoverability_score,
        "near_miss": bool(near_miss),
        "radius_crossings_total": int(radius_crossings_total),
        "first_crossing_step": first_crossing_step if first_crossing_step is not None else "",
        "minimum_abs_radius_error": min_abs_rerr,
        "final_radius_error": float(abs(rerr[-1])) if len(rerr) else 0.0,
        "tail_mean_abs_vr": float(np.mean(np.abs(tail))),
        "steps": int(len(rerr)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": termination_reason,
        "dominant_transfer_stage": dominant_stage,
        "transfer_stage_counts_json": json.dumps({key: int(value) for key, value in sorted(stage_counts.items())}, sort_keys=True),
        "intercept_classification_counts_json": json.dumps({key: int(value) for key, value in sorted(intercept_counts.items())}, sort_keys=True),
        "initial_energy_error_ratio": float(initial_energy_error),
        "final_energy_error_ratio": float(abs(final_diag.energy_error_ratio)),
        "initial_angular_momentum_error_ratio": float(initial_h_error),
        "final_angular_momentum_error_ratio": float(final_diag.angular_momentum_error_ratio),
        "mean_energy_error_ratio": float(np.mean(energy_errors)) if energy_errors else 0.0,
        "mean_angular_momentum_error_ratio": float(np.mean(h_errors)) if h_errors else 0.0,
        "max_speed_ratio": float(max(speed_ratios)) if speed_ratios else 0.0,
        "overspeed": bool(overspeed_hit),
        "turning_point_count": int(turning_point_count),
        "burn_a_steps": int(burn_a_steps),
        "burn_b_steps": int(burn_b_steps),
        "coast_arc_steps": int(coast_arc_steps),
        "safe_coast_steps": int(safe_coast_steps),
        "burn_a_count": int(burn_a_count),
        "burn_b_count": int(burn_b_count),
        "insertion_attempts": int(insertion_attempts),
        "insertion_windows": int(insertion_windows),
        "successful_insertion_windows": int(successful_insertion_windows),
        "handoff_ready_count": int(handoff_ready_count),
        "handoff_entered": bool(handoff_entered),
        "handoff_rate": float(handoff_ready_count / max(1, len(rerr))),
        "predicted_crossing_step": predicted_crossing_step if predicted_crossing_step is not None else "",
        "burn_b_solver_evals": 0,
        "burn_b_solver_score": "",
        "insertion_window_conversion_rate": float(1.0 if recoverable_crossing and insertion_windows > 0 else 0.0),
        "chosen_burn_timing": plan.original_wait_steps if plan is not None else "",
        "chosen_burn_angle_deg": plan.chosen_angle_deg if plan is not None else "",
        "chosen_burn_magnitude": plan.chosen_magnitude if plan is not None else "",
        "chosen_burn_duration": plan.chosen_duration if plan is not None else "",
        "chosen_insertion_mode": chosen_mode,
        "pre_burn_state_json": json.dumps(plan.pre_burn_state if plan is not None else {}, sort_keys=True),
        "post_burn_state_json": json.dumps(post_burn_state, sort_keys=True),
        "basin_score": min_distance,
        "handoff_success": bool(success),
        "initial_vt_error_ratio": float(initial_vt_error),
        "final_vt_error_ratio": float(final_diag.vt_error_ratio),
        "min_abs_vt_error_ratio": min_abs_vt_error,
        "mean_abs_vt_error_ratio": mean_abs_vt_error,
        "best_vt_error_before_crossing": float(best_vt_before_crossing),
        "best_vt_band": best_band,
        "best_vt_band_rank": vt_band_rank(best_band),
        "crossing_vt_band": crossing_band,
        "min_distance_to_recoverable": min_distance,
        "mean_distance_to_recoverable": mean_distance,
        "pre_burn_vt_error_ratio": pre_burn_vt_error if pre_burn_vt_error is not None else "",
        "post_burn_vt_error_ratio": post_burn_vt_error if post_burn_vt_error is not None else "",
        "pre_post_burn_vt_correction": pre_post_correction,
        "vt_improvement_burn_a_to_b": vt_improvement_burn_a_to_b,
        "corridor_burn_steps": int(corridor_burn_steps),
        "corridor_hold_steps": int(corridor_hold_steps),
        "corridor_stage_counts_json": json.dumps({key: int(value) for key, value in sorted(corridor_stage_counts.items())}, sort_keys=True),
        "instability": bool(instability),
    }
    if record_trajectory:
        row["trajectory"] = trajectory
    return row


def run_phase26_rows(variants: Sequence[CorridorVariant]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    cases = list(iter_cases(variants))
    for idx, (variant, r0, angle, thrust, target_scale) in enumerate(cases, start=1):
        row = rollout_phase26_case(variant, r0, angle, thrust, target_scale)
        rows.append(row)
        print(
            f"phase26 {idx}/{len(cases)} {variant.name} r0={r0:g} angle={angle:g} thrust={thrust:g} "
            f"capture={row['capture_entered']} success={row['success']} crossings={row['radius_crossings_total']} "
            f"recoverable={row['recoverable_crossing']} min_vt={row['min_abs_vt_error_ratio']:.4f} "
            f"band={row['best_vt_band']} reason={row['termination_reason']}"
        )
    return rows


def write_results(rows: Sequence[Dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PHASE26_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PHASE26_FIELDNAMES})


def write_corridor_csv(rows: Sequence[Dict[str, object]]) -> None:
    phase26_rows = [row for row in rows if str(row.get("controller_name", "")).startswith("phase26_")]
    with CORRIDOR_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CORRIDOR_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in phase26_rows:
            case_id = (
                f"r0={as_float(row.get('r0_over_target')):.5g}|"
                f"angle={as_float(row.get('initial_velocity_angle_deg')):.5g}|"
                f"thrust={as_float(row.get('thrust_scale')):.5g}|"
                f"target={as_float(row.get('target_radius_scale'), 1.0):.5g}"
            )
            payload = {key: row.get(key, "") for key in CORRIDOR_FIELDNAMES}
            payload["case_id"] = case_id
            writer.writerow(payload)


def numeric_values(rows: Sequence[Dict[str, object]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key, "")
        if value in {"", None}:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            values.append(value_f)
    return values


def aggregate(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for name in sorted({str(row["controller_name"]) for row in rows}):
        subset = [row for row in rows if str(row["controller_name"]) == name]
        windows = sum(int(float(row.get("insertion_windows") or 0)) for row in subset)
        recoverable = sum(bool_from_csv(row.get("recoverable_crossing")) for row in subset)
        stats[name] = {
            "total": len(subset),
            "crossing_cases": sum(bool_from_csv(row.get("crossing_occurs")) for row in subset),
            "recoverable_crossings": recoverable,
            "capture": sum(bool_from_csv(row.get("capture_entered")) for row in subset),
            "success": sum(bool_from_csv(row.get("success")) for row in subset),
            "overspeed": sum(bool_from_csv(row.get("overspeed")) or str(row.get("termination_reason")) == "overspeed" for row in subset),
            "instability": sum(bool_from_csv(row.get("instability")) for row in subset),
            "insertion_windows": windows,
            "insertion_attempts": sum(int(float(row.get("insertion_attempts") or 0)) for row in subset),
            "burn_a_steps": sum(int(float(row.get("burn_a_steps") or 0)) for row in subset),
            "burn_b_steps": sum(int(float(row.get("burn_b_steps") or 0)) for row in subset),
            "corridor_burn_steps": sum(int(float(row.get("corridor_burn_steps") or 0)) for row in subset),
            "corridor_hold_steps": sum(int(float(row.get("corridor_hold_steps") or 0)) for row in subset),
            "conversion_rate": recoverable / max(1, windows),
            "mean_crossing_vt": float(np.mean([abs(v) for v in numeric_values(subset, "crossing_vt_error_ratio")])) if numeric_values(subset, "crossing_vt_error_ratio") else float("nan"),
            "min_crossing_vt": float(np.min([abs(v) for v in numeric_values(subset, "crossing_vt_error_ratio")])) if numeric_values(subset, "crossing_vt_error_ratio") else float("nan"),
            "min_vt": float(np.min(numeric_values(subset, "min_abs_vt_error_ratio"))) if numeric_values(subset, "min_abs_vt_error_ratio") else float("nan"),
            "mean_vt": float(np.mean(numeric_values(subset, "mean_abs_vt_error_ratio"))) if numeric_values(subset, "mean_abs_vt_error_ratio") else float("nan"),
            "best_band_rank": int(max(numeric_values(subset, "best_vt_band_rank"))) if numeric_values(subset, "best_vt_band_rank") else -1,
            "min_distance": float(np.min(numeric_values(subset, "min_distance_to_recoverable"))) if numeric_values(subset, "min_distance_to_recoverable") else float("nan"),
            "mean_distance": float(np.mean(numeric_values(subset, "mean_distance_to_recoverable"))) if numeric_values(subset, "mean_distance_to_recoverable") else float("nan"),
            "mean_pre_post_vt_correction": float(np.mean(numeric_values(subset, "pre_post_burn_vt_correction"))) if numeric_values(subset, "pre_post_burn_vt_correction") else float("nan"),
        }
    return stats


def phase26_rows_only(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return [row for row in rows if str(row.get("controller_name", "")).startswith("phase26_")]


def save_controller_comparison(stats: Dict[str, Dict[str, object]]) -> None:
    names = [name for name in stats if name.startswith("phase26_") or name == "phase24_precision_insertion_geometry"]
    labels = [name.replace("phase26_", "p26_").replace("phase24_precision_insertion_geometry", "p24_baseline") for name in names]
    metrics = ["crossing_cases", "recoverable_crossings", "capture", "success"]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 1.5) * width, [stats[name][metric] for name in names], width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("cases")
    ax.set_title("Phase 26 controller comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CONTROLLER_COMPARISON_PNG, dpi=220)
    plt.close(fig)


def save_crossing_vt_distribution(rows: Sequence[Dict[str, object]]) -> None:
    phase_rows = phase26_rows_only(rows)
    names = sorted({str(row["controller_name"]) for row in phase_rows})
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    data = [[abs(v) for v in numeric_values([row for row in phase_rows if row["controller_name"] == name], "crossing_vt_error_ratio")] for name in names]
    if any(data):
        ax.boxplot(data, labels=[name.replace("phase26_", "") for name in names], showfliers=True)
    ax.axhline(RECOVERABLE_VT_RATIO, color="#54A24B", linestyle="--", linewidth=1.0, label="1.0x recoverable vt")
    ax.axhline(1.5 * RECOVERABLE_VT_RATIO, color="#72B7B2", linestyle=":", linewidth=1.0, label="1.5x")
    ax.axhline(2.0 * RECOVERABLE_VT_RATIO, color="#F58518", linestyle=":", linewidth=1.0, label="2.0x")
    ax.axhline(3.0 * RECOVERABLE_VT_RATIO, color="#E45756", linestyle=":", linewidth=1.0, label="3.0x")
    ax.set_ylabel("|crossing vt error ratio|")
    ax.set_title("Crossing tangential-velocity distribution")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CROSSING_VT_PNG, dpi=220)
    plt.close(fig)


def save_band_progression(rows: Sequence[Dict[str, object]]) -> None:
    phase_rows = phase26_rows_only(rows)
    names = sorted({str(row["controller_name"]) for row in phase_rows})
    band_order = ["outside_3x", "A_3x", "B_2x", "C_1p5x", "D_1x_recoverable"]
    counts: Dict[str, Counter[str]] = {}
    for name in names:
        counts[name] = Counter(str(row.get("best_vt_band") or "unknown") for row in phase_rows if row["controller_name"] == name)
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    bottom = np.zeros(len(names))
    colors = ["#9D755D", "#E45756", "#F58518", "#72B7B2", "#54A24B"]
    for band, color in zip(band_order, colors):
        values = np.asarray([counts[name].get(band, 0) for name in names])
        ax.bar([name.replace("phase26_", "") for name in names], values, bottom=bottom, label=band, color=color)
        bottom += values
    ax.set_title("Best vt corridor band reached")
    ax.set_ylabel("cases")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(BAND_PROGRESSION_PNG, dpi=220)
    plt.close(fig)


def save_pre_post_burn_plot(rows: Sequence[Dict[str, object]]) -> None:
    phase_rows = phase26_rows_only(rows)
    names = sorted({str(row["controller_name"]) for row in phase_rows})
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    positions = np.arange(len(names))
    pre_values = [np.nanmean([abs(v) for v in numeric_values([row for row in phase_rows if row["controller_name"] == name], "pre_burn_vt_error_ratio")]) for name in names]
    post_values = [np.nanmean([abs(v) for v in numeric_values([row for row in phase_rows if row["controller_name"] == name], "post_burn_vt_error_ratio")]) for name in names]
    width = 0.34
    ax.bar(positions - width / 2, pre_values, width, label="pre Burn B", color="#4C78A8")
    ax.bar(positions + width / 2, post_values, width, label="post Burn B", color="#F58518")
    ax.axhline(RECOVERABLE_VT_RATIO, color="#54A24B", linestyle="--", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([name.replace("phase26_", "") for name in names], rotation=20, ha="right")
    ax.set_ylabel("mean |vt error ratio|")
    ax.set_title("Pre/post insertion vt correction")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PRE_POST_BURN_PNG, dpi=220)
    plt.close(fig)


def representative_cases() -> List[tuple[CorridorVariant, float, float, float]]:
    return [
        (VARIANTS[0], 1.00, 175.0, 10000.0),
        (VARIANTS[1], 1.02, 170.0, 10000.0),
        (VARIANTS[2], 0.98, 165.0, 10000.0),
        (VARIANTS[3], 1.05, 150.0, 12000.0),
    ]


def save_time_series_plots() -> None:
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    trace_rows = [
        rollout_phase26_case(variant, r0, angle, thrust, TARGET_RADIUS_SCALE, record_trajectory=True)
        for variant, r0, angle, thrust in representative_cases()
    ]
    fig_vt, ax_vt = plt.subplots(figsize=(10.0, 5.4))
    fig_dist, ax_dist = plt.subplots(figsize=(10.0, 5.4))
    for idx, row in enumerate(trace_rows, start=1):
        traj = row["trajectory"]
        steps = np.asarray(traj["time_step"], dtype=np.float64)
        time_days = steps * DT / 86400.0
        vt = np.asarray(traj["vt_error_ratio"], dtype=np.float64)
        distance = np.asarray(traj["distance_to_recoverable"], dtype=np.float64)
        radius_ratio = np.asarray(traj["radius_ratio"], dtype=np.float64)
        vr = np.asarray(traj["radial_velocity_ratio"], dtype=np.float64)
        label = f"{row['controller_name'].replace('phase26_', '')} r0={row['r0_over_target']:g} a={row['initial_velocity_angle_deg']:g}"
        ax_vt.plot(time_days, np.abs(vt), linewidth=1.1, label=label)
        ax_dist.plot(time_days, distance, linewidth=1.1, label=label)

        fig, axes = plt.subplots(4, 1, figsize=(9.4, 8.6), sharex=True)
        axes[0].plot(time_days, radius_ratio, color="#4C78A8", linewidth=1.0)
        axes[0].axhline(1.0, color="#333333", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("r / target")
        axes[1].plot(time_days, np.abs(vt), color="#E45756", linewidth=1.0)
        axes[1].axhline(RECOVERABLE_VT_RATIO, color="#54A24B", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("|vt err|")
        axes[2].plot(time_days, vr, color="#F58518", linewidth=1.0)
        axes[2].axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
        axes[2].set_ylabel("vr ratio")
        axes[3].plot(time_days, distance, color="#72B7B2", linewidth=1.0)
        axes[3].set_ylabel("recoverability distance")
        axes[3].set_xlabel("time [days]")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        fig.suptitle(f"Phase 26 representative transfer {idx}: {label}")
        fig.tight_layout()
        fig.savefig(TRAJECTORY_DIR / f"phase26_representative_{idx:02d}.png", dpi=220)
        plt.close(fig)
    for ax, title, ylabel in [
        (ax_vt, "Phase 26 vt error over time", "|vt error ratio|"),
        (ax_dist, "Phase 26 recoverability distance progression", "distance to recoverable"),
    ]:
        ax.axhline(RECOVERABLE_VT_RATIO if ax is ax_vt else 1.0, color="#54A24B", linestyle="--", linewidth=0.9)
        ax.set_xlabel("time [days]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig_vt.tight_layout()
    fig_dist.tight_layout()
    fig_vt.savefig(VT_TIME_PNG, dpi=220)
    fig_dist.savefig(RECOVERABILITY_DISTANCE_PNG, dpi=220)
    plt.close(fig_vt)
    plt.close(fig_dist)


def save_plots(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]]) -> None:
    save_controller_comparison(stats)
    save_crossing_vt_distribution(rows)
    save_band_progression(rows)
    save_pre_post_burn_plot(rows)
    save_time_series_plots()


def write_summary(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]]) -> None:
    phase26_names = [variant.name for variant in VARIANTS]
    phase24 = stats.get("phase24_precision_insertion_geometry", {})
    best_vt_name = min(
        phase26_names,
        key=lambda name: stats.get(name, {}).get("min_vt", float("inf")),
    )
    best_dist_name = min(
        phase26_names,
        key=lambda name: stats.get(name, {}).get("min_distance", float("inf")),
    )
    best_recoverable = max(int(stats.get(name, {}).get("recoverable_crossings", 0)) for name in phase26_names)
    best_capture = max(int(stats.get(name, {}).get("capture", 0)) for name in phase26_names)
    best_success = max(int(stats.get(name, {}).get("success", 0)) for name in phase26_names)
    phase24_recoverable = int(phase24.get("recoverable_crossings", 0))
    phase24_capture = int(phase24.get("capture", 0))
    phase24_success = int(phase24.get("success", 0))
    local_vt_reduced = any(float(stats.get(name, {}).get("mean_pre_post_vt_correction", float("nan"))) > 0.0 for name in phase26_names)
    phase24_crossing_vt = float(phase24.get("mean_crossing_vt", float("nan")))
    crossing_vt_reduced = any(
        math.isfinite(float(stats.get(name, {}).get("mean_crossing_vt", float("nan"))))
        and math.isfinite(phase24_crossing_vt)
        and float(stats[name]["mean_crossing_vt"]) < phase24_crossing_vt
        for name in phase26_names
    )
    multi_step_required = stats.get("phase26_two_step_corridor", {}).get("min_vt", float("inf")) <= stats.get("phase26_vt_aware_scoring", {}).get("min_vt", float("inf"))
    bottleneck = "vt shaping"
    if best_recoverable > 0 and best_capture <= phase24_capture:
        bottleneck = "CAPTURE architecture after recoverability"
    elif best_recoverable == 0 and stats.get(best_dist_name, {}).get("min_distance", float("inf")) <= 1.5:
        bottleneck = "handoff basin width"
    elif best_recoverable == 0 and local_vt_reduced and not crossing_vt_reduced:
        bottleneck = "vt-radius timing synchronization"
    elif best_recoverable == 0 and local_vt_reduced:
        bottleneck = "vr/r coupling after vt improvement"
    elif not local_vt_reduced:
        bottleneck = "physical or architecture limit on vt corridor shaping"

    lines = [
        "# Phase 26 Tangential Velocity Corridor Engineering",
        "",
        "## Scope",
        "",
        "- Python-only 2D benchmark using the same reduced 48-case grid per Phase 26 variant.",
        "- Phase 22 Burn A and coast concepts are preserved.",
        "- CAPTURE, LOCK, physics, PPO, reward, and recoverability thresholds are unchanged.",
        "- Burn B is replaced with deterministic vt-corridor shaping variants.",
        "- Phase 24 is included as the loaded baseline, not overwritten.",
        "",
        "## Controller Comparison",
        "",
            "| Controller | Crossings | Recoverable | CAPTURE | Success | Min pre-cross vt | Mean crossing vt | Min crossing vt | Best distance | Overspeed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    names_for_table = ["phase24_precision_insertion_geometry"] + phase26_names
    for name in names_for_table:
        item = stats.get(name, {})
        lines.append(
            f"| `{name}` | {item.get('crossing_cases', 0)} | {item.get('recoverable_crossings', 0)} | "
            f"{item.get('capture', 0)} | {item.get('success', 0)} | "
            f"{float(item.get('min_vt', float('nan'))):.4f} | {float(item.get('mean_crossing_vt', float('nan'))):.4f} | "
            f"{float(item.get('min_crossing_vt', float('nan'))):.4f} | {float(item.get('min_distance', float('nan'))):.4f} | {item.get('overspeed', 0)} |"
        )
    lines.extend(
        [
            "",
            "## VT Corridor Findings",
            "",
            f"- Recoverable vt threshold: `{RECOVERABLE_VT_RATIO:.4f}`.",
            f"- Soft corridor bands: `3.0x={3.0 * RECOVERABLE_VT_RATIO:.4f}`, `2.0x={2.0 * RECOVERABLE_VT_RATIO:.4f}`, `1.5x={1.5 * RECOVERABLE_VT_RATIO:.4f}`, `1.0x={RECOVERABLE_VT_RATIO:.4f}`.",
            f"- Best vt variant: `{best_vt_name}` with min |vt error ratio| `{stats[best_vt_name]['min_vt']:.4f}`.",
            f"- Best recoverability-distance variant: `{best_dist_name}` with min distance `{stats[best_dist_name]['min_distance']:.4f}`.",
            f"- Strongest best-band rank reached: `{max(int(stats.get(name, {}).get('best_band_rank', -1)) for name in phase26_names)}`.",
            f"- Phase 24 mean crossing |vt error ratio|: `{phase24_crossing_vt:.4f}`.",
            f"- Best Phase 26 mean crossing |vt error ratio|: `{min(float(stats.get(name, {}).get('mean_crossing_vt', float('inf'))) for name in phase26_names):.4f}`.",
            f"- Best Phase 26 mean pre/post Burn B vt correction: `{max(float(stats.get(name, {}).get('mean_pre_post_vt_correction', float('-inf'))) for name in phase26_names):.4f}`.",
            "",
            "## Research Answers",
            "",
            f"1. Can vt-focused shaping reduce vt mismatch? `{'yes locally, no at crossing' if local_vt_reduced and not crossing_vt_reduced else ('yes' if crossing_vt_reduced else 'no')}`.",
            f"2. Does reduced vt mismatch create recoverable crossings? `{'yes' if best_recoverable > phase24_recoverable else 'no'}`.",
            f"3. Is Burn B architecture too shallow? `{'yes' if multi_step_required and best_recoverable == 0 else 'not isolated'}`.",
            f"4. Is multi-step shaping required? `{'possible' if multi_step_required else 'not supported by this run'}`.",
            f"5. Dominant bottleneck now: `{bottleneck}`.",
            "",
            "## Success Criteria",
            "",
            f"- Minimum, reduce vt mismatch measurably: `{'met locally, not at crossing' if local_vt_reduced and not crossing_vt_reduced else ('met' if crossing_vt_reduced else 'not met')}`.",
            f"- Moderate, reach tighter vt corridor bands: `{'met pre-cross, not at crossing' if max(int(stats.get(name, {}).get('best_band_rank', -1)) for name in phase26_names) >= 3 and not crossing_vt_reduced else ('met' if crossing_vt_reduced else 'not met')}`.",
            f"- Strong, first recoverable crossing: `{'met' if best_recoverable > phase24_recoverable else 'not met'}`.",
            f"- Major, increase CAPTURE or success: `{'met' if best_capture > phase24_capture or best_success > phase24_success else 'not met'}`.",
            "",
            "## Honesty Note",
            "",
            "- Positive CAPTURE or success is not inferred from relaxed thresholds.",
            "- vt-band progress is reported separately from recoverability and CAPTURE.",
            "- Overspeed and unstable outcomes are retained in both CSV outputs.",
            "",
            "## Artifacts",
            "",
            "- `phase26_results.csv`",
            "- `phase26_vt_corridor_analysis.csv`",
            "- `vt_error_over_time.png`",
            "- `crossing_vt_distribution.png`",
            "- `vt_corridor_band_progression.png`",
            "- `controller_comparison.png`",
            "- `pre_post_burn_vt_correction.png`",
            "- `recoverability_distance_progression.png`",
            "- `trajectories/phase26_representative_*.png`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 26 tangential velocity corridor benchmark.")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--variants", nargs="*", default=[variant.name for variant in VARIANTS])
    args = parser.parse_args()
    set_seed(7)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_variants = [variant for variant in VARIANTS if variant.name in set(args.variants)]
    reference_rows: List[Dict[str, object]] = []
    reference_rows.extend(load_csv_rows(PHASE20_CSV, {"baseline_soft_linear_3e4", "phase20_predictive_planner"}))
    reference_rows.extend(load_csv_rows(PHASE21_CSV, {"phase21_orbital_transfer_planner"}))
    reference_rows.extend(load_csv_rows(PHASE22_CSV, {"phase22_two_burn_transfer"}))
    reference_rows.extend(load_csv_rows(PHASE23_CSV, {"phase23_windowed_insertion_solver"}))
    reference_rows.extend(load_csv_rows(PHASE24_CSV, {"phase24_precision_insertion_geometry"}))
    phase26_rows = run_phase26_rows(selected_variants)
    rows = reference_rows + phase26_rows
    write_results(rows)
    write_corridor_csv(rows)
    stats = aggregate(rows)
    if not args.skip_plots:
        save_plots(rows, stats)
    write_summary(rows, stats)
    print(f"Saved Phase 26 tangential velocity corridor outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
