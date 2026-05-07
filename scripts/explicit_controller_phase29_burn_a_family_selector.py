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
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase29_repo_audit_and_family_selector"
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


PHASE22_CSV = PROJECT_ROOT / "analysis" / "phase22_two_burn_transfer" / "phase22_results.csv"
PHASE24_CSV = PROJECT_ROOT / "analysis" / "phase24_precision_insertion_geometry" / "phase24_results.csv"
PHASE26_CSV = PROJECT_ROOT / "analysis" / "phase26_tangential_velocity_corridor" / "phase26_results.csv"
PHASE27_CSV = PROJECT_ROOT / "analysis" / "phase27_timing_synchronization" / "phase27_results.csv"
PHASE28_DATASET = PROJECT_ROOT / "analysis" / "phase28_trajectory_family_mapping" / "phase28_family_dataset.csv"
PHASE28_SUMMARY = PROJECT_ROOT / "analysis" / "phase28_trajectory_family_mapping" / "summary.md"

RESULTS_CSV = OUTPUT_DIR / "phase29_results.csv"
ANALYSIS_CSV = OUTPUT_DIR / "phase29_family_selector_analysis.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
REPO_AUDIT_MD = OUTPUT_DIR / "repo_audit.md"

PLOT_BURN_A_MAP = OUTPUT_DIR / "burn_a_family_selection_map.png"
PLOT_INITIAL_REGION = OUTPUT_DIR / "initial_family_region.png"
PLOT_EH_TARGET = OUTPUT_DIR / "energy_angular_momentum_targeting.png"
PLOT_WINDOW_CROSS = OUTPUT_DIR / "window_vs_crossing_comparison.png"
PLOT_PHASE29_PHASE28 = OUTPUT_DIR / "phase29_vs_phase28_family_quality.png"
PLOT_CONTROLLER = OUTPUT_DIR / "controller_comparison.png"

BURN_A_MAX_STEPS = 24
BURN_A_MIN_STEPS = 10
BURN_A_MAX_NORM = 0.11
COAST_MIN_STEPS = 80
COAST_MAX_STEPS = 42000
BURN_B_MAX_STEPS = 46
BURN_B_MAX_NORM = 0.42
INSERTION_BAND_RATIO = 2.5e-2
HANDOFF_R_RATIO = 7.5e-4
HANDOFF_VR_RATIO = 2.8e-2
HANDOFF_VT_RATIO = 2.8e-1
SAFE_SPEED_RATIO = 1.65


@dataclass(frozen=True)
class BurnAFamilyVariant:
    name: str
    strategy: str
    burn_a_steps_delta: int = 0
    max_norm_scale: float = 1.0
    radial_gain: float = 0.06
    h_gain: float = 2.2
    e_gain: float = 1.0
    family_bias: float = 0.0


VARIANTS = [
    BurnAFamilyVariant("phase29_baseline_phase22_burn_a", "baseline_phase22_burn_a"),
    BurnAFamilyVariant(
        "phase29_crossing_family_targeted_burn_a",
        "crossing_family_targeted_burn_a",
        burn_a_steps_delta=-4,
        max_norm_scale=0.72,
        radial_gain=0.04,
        h_gain=1.4,
        family_bias=0.20,
    ),
    BurnAFamilyVariant(
        "phase29_angular_momentum_targeted_burn_a",
        "angular_momentum_targeted_burn_a",
        burn_a_steps_delta=0,
        max_norm_scale=0.90,
        radial_gain=0.04,
        h_gain=3.6,
    ),
    BurnAFamilyVariant(
        "phase29_energy_angular_momentum_balanced_burn_a",
        "energy_angular_momentum_balanced_burn_a",
        burn_a_steps_delta=2,
        max_norm_scale=0.86,
        radial_gain=0.05,
        h_gain=2.4,
        e_gain=1.8,
    ),
    BurnAFamilyVariant(
        "phase29_avoid_dead_window_family_burn_a",
        "avoid_dead_window_family_burn_a",
        burn_a_steps_delta=-6,
        max_norm_scale=0.55,
        radial_gain=0.025,
        h_gain=1.2,
        e_gain=0.8,
        family_bias=-0.15,
    ),
]

FIELDNAMES = [
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
    "crossing_radius_error_ratio",
    "sync_error_at_crossing",
    "min_distance_to_recoverable",
    "recoverable_crossing",
    "near_recoverable_crossing",
    "family_label",
    "window_quality_label",
    "family_quality_score",
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
    "instability",
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
    "good_windows",
    "dead_windows",
    "handoff_ready_count",
    "handoff_entered",
    "handoff_rate",
    "predicted_crossing_step",
    "burn_a_strategy",
    "burn_a_end_step",
    "burn_a_end_radius_ratio",
    "burn_a_end_r_error_ratio",
    "burn_a_end_vr_ratio",
    "burn_a_end_vt_error_ratio",
    "burn_a_end_energy_error_ratio",
    "burn_a_end_angular_momentum_error_ratio",
    "burn_a_end_eccentricity_proxy",
    "burn_a_end_periapsis_ratio",
    "burn_a_end_apoapsis_ratio",
    "burn_a_end_orbit_crosses_target",
    "burn_a_end_predicted_crossing_step",
    "phase28_reference_family_label",
    "phase28_reference_window_quality",
]

ANALYSIS_FIELDS = [
    "controller_name",
    "case_id",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "phase28_reference_family_label",
    "phase28_reference_window_quality",
    "phase29_family_label",
    "window_quality_label",
    "crossing_occurs",
    "recoverable_crossing",
    "near_recoverable_crossing",
    "capture_entered",
    "success",
    "insertion_windows",
    "good_windows",
    "dead_windows",
    "sync_error_at_crossing",
    "min_distance_to_recoverable",
    "crossing_vr_ratio",
    "crossing_vt_error_ratio",
    "burn_a_end_radius_ratio",
    "burn_a_end_vr_ratio",
    "burn_a_end_vt_error_ratio",
    "burn_a_end_energy_error_ratio",
    "burn_a_end_angular_momentum_error_ratio",
    "burn_a_end_eccentricity_proxy",
    "burn_a_end_periapsis_ratio",
    "burn_a_end_apoapsis_ratio",
    "burn_a_end_orbit_crosses_target",
    "burn_a_end_predicted_crossing_step",
    "family_quality_score",
    "overspeed",
    "instability",
]


def set_seed(seed: int = 7) -> None:
    np.random.seed(seed)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: object, default: float = float("nan")) -> float:
    if value in {None, ""}:
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


def case_id_from_values(r0: float, angle: float, thrust: float, target_scale: float) -> str:
    return f"r0={r0:.5g}|angle={angle:.5g}|thrust={thrust:.5g}|target={target_scale:.5g}"


def case_id(row: Dict[str, object]) -> str:
    return case_id_from_values(
        as_float(row.get("r0_over_target")),
        as_float(row.get("initial_velocity_angle_deg")),
        as_float(row.get("thrust_scale")),
        as_float(row.get("target_radius_scale"), 1.0),
    )


def iter_cases(variants: Sequence[BurnAFamilyVariant]) -> Iterable[tuple[BurnAFamilyVariant, float, float, float, float]]:
    for variant in variants:
        for thrust in THRUST_VALUES:
            for angle in ANGLE_VALUES:
                for r0 in R0_VALUES:
                    yield variant, r0, angle, thrust, TARGET_RADIUS_SCALE


def estimate_crossing_steps(radius_error: float, radial_velocity: float) -> int | None:
    if radius_error * radial_velocity >= 0.0 or abs(radial_velocity) < 1.0e-9:
        return None
    steps = abs(radius_error) / (abs(radial_velocity) * DT + 1.0e-12)
    if not math.isfinite(steps) or steps < 0.0:
        return None
    return int(round(steps))


def sync_error(r_ratio: float, vr_ratio: float, vt_ratio: float) -> float:
    if not all(math.isfinite(value) for value in [r_ratio, vr_ratio, vt_ratio]):
        return float("nan")
    return max(
        abs(r_ratio) / RECOVERABLE_R_RATIO,
        abs(vr_ratio) / RECOVERABLE_VR_RATIO,
        abs(vt_ratio) / RECOVERABLE_VT_RATIO,
    )


def recoverability_distance(r_ratio: float, vr_ratio: float, vt_ratio: float) -> float:
    if not all(math.isfinite(value) for value in [r_ratio, vr_ratio, vt_ratio]):
        return float("nan")
    return math.sqrt(
        (abs(r_ratio) / RECOVERABLE_R_RATIO) ** 2
        + (abs(vr_ratio) / RECOVERABLE_VR_RATIO) ** 2
        + (abs(vt_ratio) / RECOVERABLE_VT_RATIO) ** 2
    )


def near_recoverable(sync_value: float, distance: float) -> bool:
    return (math.isfinite(sync_value) and sync_value <= 3.0) or (math.isfinite(distance) and distance <= 3.0)


def family_label(crossing: bool, windows: int, capture: bool, success: bool, recoverable: bool, sync_value: float, distance: float) -> str:
    if crossing and (recoverable or near_recoverable(sync_value, distance)):
        return "near_recoverable_crossing"
    if crossing:
        return "crossing_bad_sync"
    if capture or success:
        return "capture_success_existing"
    if windows > 0:
        return "window_no_crossing"
    return "dead_geometry"


def window_quality_label(windows: int, crossing: bool, sync_value: float, distance: float) -> str:
    if windows <= 0:
        return "no_window"
    if crossing and near_recoverable(sync_value, distance):
        return "good_window"
    if crossing:
        return "dead_window_bad_crossing"
    return "dead_window_no_crossing"


def quality_score(crossing: bool, near: bool, recoverable: bool, capture: bool, success: bool, overspeed: bool, instability: bool, sync_value: float) -> float:
    score = 0.0
    score += 1.0 if crossing else 0.0
    score += 2.0 if near else 0.0
    score += 3.0 if recoverable else 0.0
    score += 4.0 if capture else 0.0
    score += 5.0 if success else 0.0
    score -= 2.0 if overspeed else 0.0
    score -= 1.5 if instability else 0.0
    if math.isfinite(sync_value):
        score -= min(4.0, max(0.0, sync_value - 1.0) * 0.5)
    return score


def classify_intercept(diag, target_radius: float, coast_steps: int) -> str:
    v_circ = math.sqrt(MU / target_radius)
    vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
    if abs(diag.r_error_ratio) <= HANDOFF_R_RATIO and abs(vr_ratio) <= HANDOFF_VR_RATIO and abs(diag.vt_error_ratio) <= HANDOFF_VT_RATIO:
        return "handoff_ready"
    if abs(diag.r_error_ratio) <= INSERTION_BAND_RATIO and diag.r_error_ratio * diag.radial_velocity <= 0.0:
        return "insertion_window"
    if not orbit_crosses_target(diag, target_radius):
        return "no_intercept_possible"
    if abs(diag.vt_error_ratio) <= 0.35 and abs(vr_ratio) <= 0.08 and coast_steps >= COAST_MIN_STEPS:
        return "intercept_possible_good_alignment"
    return "intercept_possible_poor_alignment"


def diagnostic_payload(diag, target_radius: float, v_circ: float) -> Dict[str, object]:
    predicted = estimate_crossing_steps(diag.radius - target_radius, diag.radial_velocity)
    eccentricity = ""
    if diag.periapsis is not None and diag.apoapsis is not None and diag.apoapsis + diag.periapsis > 0.0:
        eccentricity = (diag.apoapsis - diag.periapsis) / (diag.apoapsis + diag.periapsis)
    return {
        "radius_ratio": diag.radius / target_radius,
        "r_error_ratio": diag.r_error_ratio,
        "vr_ratio": diag.radial_velocity / (v_circ + 1.0e-12),
        "vt_error_ratio": diag.vt_error_ratio,
        "energy_error_ratio": diag.energy_error_ratio,
        "angular_momentum_error_ratio": diag.angular_momentum_error_ratio,
        "eccentricity_proxy": eccentricity,
        "periapsis_ratio": diag.periapsis / target_radius if diag.periapsis is not None else "",
        "apoapsis_ratio": diag.apoapsis / target_radius if diag.apoapsis is not None else "",
        "orbit_crosses_target": orbit_crosses_target(diag, target_radius),
        "predicted_crossing_step": predicted if predicted is not None else "",
    }


def load_phase28_reference() -> Dict[str, Dict[str, object]]:
    refs: Dict[str, Dict[str, object]] = {}
    for row in read_csv(PHASE28_DATASET):
        cid = str(row.get("case_id", ""))
        current = refs.get(cid)
        score = as_float(row.get("quality_score"), -999.0)
        if current is None or score > as_float(current.get("quality_score"), -999.0):
            refs[cid] = row
    return refs


def load_reference_rows() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for path, names in [
        (PHASE22_CSV, {"phase22_two_burn_transfer"}),
        (PHASE24_CSV, {"phase24_precision_insertion_geometry"}),
        (PHASE26_CSV, {"phase26_vt_aware_scoring"}),
        (PHASE27_CSV, {"phase27_predicted_cross_vt_targeting"}),
    ]:
        for row in read_csv(path):
            if str(row.get("controller_name")) in names:
                out.append(dict(row))
    return out


def rollout_case(variant: BurnAFamilyVariant, r0: float, angle: float, thrust_scale: float, target_scale: float, phase28_ref: Dict[str, Dict[str, object]]) -> Dict[str, object]:
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
    base_burn_a_steps = int(clamp(BURN_A_MIN_STEPS + 420.0 * abs(r0 - 1.0), BURN_A_MIN_STEPS, BURN_A_MAX_STEPS))
    burn_a_target_steps = int(clamp(base_burn_a_steps + variant.burn_a_steps_delta, 4, BURN_A_MAX_STEPS + 8))
    prev_controller_r_error: float | None = None
    prev_r_error: float | None = None
    prev_vr_for_turn: float | None = None
    radius_errors: List[float] = []
    vr_values: List[float] = []
    speed_ratios: List[float] = []
    energy_errors: List[float] = []
    h_errors: List[float] = []
    crossing_recovery_samples: List[float] = []
    stage_counts: Counter[str] = Counter()
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
    burn_a_end: Dict[str, object] = {}

    def env_step(state_x: float, state_y: float, state_vx: float, state_vy: float, action_x: float, action_y: float) -> tuple[float, float, float, float]:
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

    def candidate_score(state_x: float, state_y: float, state_vx: float, state_vy: float) -> float:
        diag = orbital_diagnostics(state_x, state_y, state_vx, state_vy, target_radius)
        return (
            BASE_PARAMS["score_r"] * abs(diag.r_error_ratio)
            + BASE_PARAMS["score_vr"] * abs(diag.radial_velocity) / (v_circ + 1.0e-12)
            + BASE_PARAMS["score_vt"] * abs(diag.vt_error_ratio)
            + BASE_PARAMS["score_energy"] * abs(diag.energy_error_ratio)
        )

    def soft_linear_3e4_action(state_x: float, state_y: float, state_vx: float, state_vy: float, state_radius: float, real_r_ratio: float, abs_r_error_ratio: float) -> tuple[float, float]:
        retro_x, retro_y = baseline_action(state_vx, state_vy)
        inv_r = 1.0 / (state_radius + 1.0e-12)
        r_hat_x = state_x * inv_r
        r_hat_y = state_y * inv_r
        inward_sign = -1.0 if real_r_ratio > 0.0 else 1.0
        pre_x, pre_y = normalize_action(retro_x + BASE_PARAMS["prewindow_radial_gain"] * inward_sign * r_hat_x, retro_y + BASE_PARAMS["prewindow_radial_gain"] * inward_sign * r_hat_y)
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

    def capture_lock_action(state_x: float, state_y: float, state_vx: float, state_vy: float, current_phase: str) -> tuple[float, float]:
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
        vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
        if variant.strategy == "baseline_phase22_burn_a":
            tangential_cmd = -1.0 if diag.r_error_ratio > 0.0 else (1.0 if diag.r_error_ratio < 0.0 else (1.0 if diag.angular_momentum_error_ratio < 0.0 else -1.0))
            radial_cmd = clamp(-0.06 * vr_ratio, -0.04, 0.04)
            max_norm = clamp(0.045 + 1.20 * min(abs_r_ratio, 0.055), 0.045, BURN_A_MAX_NORM)
        elif variant.strategy == "crossing_family_targeted_burn_a":
            radius_family_cmd = clamp(-0.9 * diag.r_error_ratio + variant.family_bias * (1.0 if diag.r_error_ratio < 0.0 else -1.0), -0.8, 0.8)
            h_cmd = clamp(-variant.h_gain * diag.angular_momentum_error_ratio, -0.8, 0.8)
            tangential_cmd = clamp(0.65 * radius_family_cmd + 0.35 * h_cmd, -1.0, 1.0)
            radial_cmd = clamp(-variant.radial_gain * vr_ratio - 0.08 * diag.r_error_ratio, -0.05, 0.05)
            max_norm = clamp((0.040 + 0.65 * min(abs_r_ratio, 0.055)) * variant.max_norm_scale, 0.025, BURN_A_MAX_NORM)
        elif variant.strategy == "angular_momentum_targeted_burn_a":
            tangential_cmd = clamp(-variant.h_gain * diag.angular_momentum_error_ratio, -1.0, 1.0)
            radial_cmd = clamp(-variant.radial_gain * vr_ratio, -0.04, 0.04)
            max_norm = clamp((0.040 + 0.90 * min(abs_r_ratio, 0.055)) * variant.max_norm_scale, 0.030, BURN_A_MAX_NORM)
        elif variant.strategy == "energy_angular_momentum_balanced_burn_a":
            e_cmd = clamp(-variant.e_gain * diag.energy_error_ratio, -1.0, 1.0)
            h_cmd = clamp(-variant.h_gain * diag.angular_momentum_error_ratio, -1.0, 1.0)
            tangential_cmd = clamp(0.5 * e_cmd + 0.5 * h_cmd, -1.0, 1.0)
            radial_cmd = clamp(-variant.radial_gain * vr_ratio - 0.05 * diag.r_error_ratio, -0.05, 0.05)
            max_norm = clamp((0.040 + 1.00 * min(abs_r_ratio, 0.055)) * variant.max_norm_scale, 0.030, BURN_A_MAX_NORM)
        else:
            dead_window_avoid = -1.0 if diag.r_error_ratio < 0.0 else 1.0
            h_cmd = clamp(-variant.h_gain * diag.angular_momentum_error_ratio, -0.6, 0.6)
            tangential_cmd = clamp(0.45 * h_cmd + 0.55 * variant.family_bias * dead_window_avoid, -0.7, 0.7)
            radial_cmd = clamp(-variant.radial_gain * vr_ratio - 0.03 * diag.r_error_ratio, -0.03, 0.03)
            max_norm = clamp((0.035 + 0.50 * min(abs_r_ratio, 0.055)) * variant.max_norm_scale, 0.018, 0.075)
        return radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, max_norm)

    def burn_b_action(state_x: float, state_y: float, state_vx: float, state_vy: float, diag) -> tuple[float, float]:
        vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
        radial_cmd = clamp(-3.0 * vr_ratio - 0.35 * diag.r_error_ratio, -0.36, 0.36)
        tangential_cmd = clamp(-1.8 * diag.vt_error_ratio, -0.36, 0.36)
        return radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, BURN_B_MAX_NORM)

    initial_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
    initial_energy_error = abs(initial_diag.energy_error_ratio)
    initial_h_error = initial_diag.angular_momentum_error_ratio

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
        turning_point_now = prev_vr_for_turn is not None and (prev_vr_for_turn * diag.radial_velocity < 0.0)
        if turning_point_now:
            turning_point_count += 1

        crossed_now = False
        if prev_controller_r_error is not None:
            crossed_now = (prev_controller_r_error > 0.0 and real_r_error <= 0.0) or (prev_controller_r_error < 0.0 and real_r_error >= 0.0)
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
                    end_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
                    burn_a_end = diagnostic_payload(end_diag, target_radius, v_circ)
                    burn_a_end["step"] = step
                    transfer_stage = "coast_arc"
                    stage_step = 0
            elif transfer_stage == "coast_arc":
                action_x, action_y = 0.0, 0.0
                active_stage = "coast_arc"
                coast_arc_steps += 1
                if intercept_class == "insertion_window" or (intercept_class == "intercept_possible_good_alignment" and abs(diag.r_error_ratio) < 0.055 and coast_age >= COAST_MIN_STEPS):
                    transfer_stage = "burn_b"
                    stage_step = 0
                    burn_b_count += 1
                    insertion_attempts += 1
                    insertion_windows += 1
                elif coast_age >= COAST_MAX_STEPS and intercept_class == "no_intercept_possible":
                    transfer_stage = "abort_safe_coast"
                    stage_step = 0
            elif transfer_stage == "burn_b":
                action_x, action_y = burn_b_action(x, y, vx, vy, diag)
                active_stage = "burn_b"
                burn_b_steps += 1
                if intercept_class in {"handoff_ready", "insertion_window"} and not successful_insertion_window_seen:
                    successful_insertion_windows += 1
                    successful_insertion_window_seen = True
                if burn_b_steps >= BURN_B_MAX_STEPS or abs(diag.r_error_ratio) <= HANDOFF_R_RATIO:
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

        if prev_r_error is not None and ((prev_r_error > 0.0 and r_error <= 0.0) or (prev_r_error < 0.0 and r_error >= 0.0)):
            radius_crossings_total += 1
            if first_crossing_step is None:
                first_crossing_step = step
                crossing_vr = next_diag.radial_velocity
                crossing_vt_error = next_diag.tangential_velocity - v_circ
                crossing_radius_error = r_error
        prev_r_error = r_error
        if first_crossing_step is not None and len(crossing_recovery_samples) < POST_CROSS_WINDOW:
            crossing_recovery_samples.append(abs(r_error) / target_radius + 0.70 * abs(next_diag.radial_velocity) / (v_circ + 1.0e-12) + 0.50 * abs(next_diag.vt_error_ratio))

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
    sync_value = sync_error(crossing_r_ratio_value, crossing_vr_ratio_value, crossing_vt_error_ratio_value) if crossing_r_ratio_value is not None and crossing_vr_ratio_value is not None and crossing_vt_error_ratio_value is not None else float("nan")
    distance_value = recoverability_distance(crossing_r_ratio_value, crossing_vr_ratio_value, crossing_vt_error_ratio_value) if crossing_r_ratio_value is not None and crossing_vr_ratio_value is not None and crossing_vt_error_ratio_value is not None else float("nan")
    crossing_score = crossing_r_ratio_value + 1.20 * abs(crossing_vr_ratio_value) + 0.80 * abs(crossing_vt_error_ratio_value) if crossing_vr_ratio_value is not None and crossing_vt_error_ratio_value is not None and crossing_r_ratio_value is not None else ""
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
    crossing_occurs = radius_crossings_total > 0
    near_rec = crossing_occurs and near_recoverable(sync_value, distance_value)
    instability = bool(overspeed_hit or termination_reason in {"out_range", "too_close", "radial_stall"})
    fam_label = family_label(crossing_occurs, insertion_windows, capture_entered, success, recoverable_crossing, sync_value, distance_value)
    win_quality = window_quality_label(insertion_windows, crossing_occurs, sync_value, distance_value)
    good_windows = 1 if win_quality == "good_window" else 0
    dead_windows = 1 if win_quality.startswith("dead_window") else 0
    cid = case_id_from_values(r0, angle, thrust_scale, target_scale)
    ref = phase28_ref.get(cid, {})
    row: Dict[str, object] = {
        "controller_name": variant.name,
        "r0_over_target": r0,
        "initial_velocity_angle_deg": angle,
        "thrust_scale": thrust_scale,
        "target_radius_scale": target_scale,
        "success": bool(success),
        "capture_entered": bool(capture_entered),
        "lock_entered": bool(lock_entered),
        "crossing_occurs": bool(crossing_occurs),
        "crossing_vr": float(crossing_vr) if crossing_vr is not None else "",
        "crossing_vt_error": float(crossing_vt_error) if crossing_vt_error is not None else "",
        "crossing_vr_ratio": crossing_vr_ratio_value if crossing_vr_ratio_value is not None else "",
        "crossing_vt_error_ratio": crossing_vt_error_ratio_value if crossing_vt_error_ratio_value is not None else "",
        "crossing_radius_error": float(crossing_radius_error) if crossing_radius_error is not None else "",
        "crossing_radius_error_ratio": crossing_r_ratio_value if crossing_r_ratio_value is not None else "",
        "sync_error_at_crossing": sync_value if math.isfinite(sync_value) else "",
        "min_distance_to_recoverable": distance_value if math.isfinite(distance_value) else "",
        "recoverable_crossing": bool(recoverable_crossing),
        "near_recoverable_crossing": bool(near_rec),
        "family_label": fam_label,
        "window_quality_label": win_quality,
        "family_quality_score": quality_score(crossing_occurs, near_rec, recoverable_crossing, capture_entered, success, overspeed_hit, instability, sync_value),
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
        "instability": bool(instability),
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
        "good_windows": int(good_windows),
        "dead_windows": int(dead_windows),
        "handoff_ready_count": int(handoff_ready_count),
        "handoff_entered": bool(handoff_entered),
        "handoff_rate": float(handoff_ready_count / max(1, len(rerr))),
        "predicted_crossing_step": predicted_crossing_step if predicted_crossing_step is not None else "",
        "burn_a_strategy": variant.strategy,
        "burn_a_end_step": burn_a_end.get("step", ""),
        "burn_a_end_radius_ratio": burn_a_end.get("radius_ratio", ""),
        "burn_a_end_r_error_ratio": burn_a_end.get("r_error_ratio", ""),
        "burn_a_end_vr_ratio": burn_a_end.get("vr_ratio", ""),
        "burn_a_end_vt_error_ratio": burn_a_end.get("vt_error_ratio", ""),
        "burn_a_end_energy_error_ratio": burn_a_end.get("energy_error_ratio", ""),
        "burn_a_end_angular_momentum_error_ratio": burn_a_end.get("angular_momentum_error_ratio", ""),
        "burn_a_end_eccentricity_proxy": burn_a_end.get("eccentricity_proxy", ""),
        "burn_a_end_periapsis_ratio": burn_a_end.get("periapsis_ratio", ""),
        "burn_a_end_apoapsis_ratio": burn_a_end.get("apoapsis_ratio", ""),
        "burn_a_end_orbit_crosses_target": burn_a_end.get("orbit_crosses_target", ""),
        "burn_a_end_predicted_crossing_step": burn_a_end.get("predicted_crossing_step", ""),
        "phase28_reference_family_label": ref.get("family_label", ""),
        "phase28_reference_window_quality": ref.get("window_quality_label", ""),
    }
    return row


def run_phase29_rows(variants: Sequence[BurnAFamilyVariant]) -> List[Dict[str, object]]:
    phase28_ref = load_phase28_reference()
    rows: List[Dict[str, object]] = []
    cases = list(iter_cases(variants))
    for idx, (variant, r0, angle, thrust, target_scale) in enumerate(cases, start=1):
        row = rollout_case(variant, r0, angle, thrust, target_scale, phase28_ref)
        rows.append(row)
        print(
            f"phase29 {idx}/{len(cases)} {variant.name} r0={r0:g} angle={angle:g} thrust={thrust:g} "
            f"cross={row['crossing_occurs']} win={row['insertion_windows']} fam={row['family_label']} "
            f"cap={row['capture_entered']} succ={row['success']} reason={row['termination_reason']}"
        )
    return rows


def write_results(rows: Sequence[Dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def write_analysis_csv(rows: Sequence[Dict[str, object]]) -> None:
    with ANALYSIS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ANALYSIS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key, "") for key in ANALYSIS_FIELDS}
            payload["case_id"] = case_id(row)
            payload["phase29_family_label"] = row.get("family_label", "")
            writer.writerow(payload)


def numeric_values(rows: Sequence[Dict[str, object]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = as_float(row.get(key))
        if math.isfinite(value):
            values.append(value)
    return values


def aggregate(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for name in sorted({str(row.get("controller_name")) for row in rows}):
        subset = [row for row in rows if row.get("controller_name") == name]
        stats[name] = {
            "total": len(subset),
            "crossings": sum(bool_from_csv(row.get("crossing_occurs")) for row in subset),
            "recoverable": sum(bool_from_csv(row.get("recoverable_crossing")) for row in subset),
            "near_recoverable": sum(bool_from_csv(row.get("near_recoverable_crossing")) for row in subset),
            "capture": sum(bool_from_csv(row.get("capture_entered")) for row in subset),
            "success": sum(bool_from_csv(row.get("success")) for row in subset),
            "windows": sum(as_int(row.get("insertion_windows")) > 0 for row in subset),
            "good_windows": sum(as_int(row.get("good_windows")) > 0 for row in subset),
            "dead_windows": sum(as_int(row.get("dead_windows")) > 0 for row in subset),
            "crossing_bad_sync": sum(row.get("family_label") == "crossing_bad_sync" for row in subset),
            "overspeed": sum(bool_from_csv(row.get("overspeed")) for row in subset),
            "instability": sum(bool_from_csv(row.get("instability")) for row in subset),
            "mean_sync": float(np.mean(numeric_values(subset, "sync_error_at_crossing"))) if numeric_values(subset, "sync_error_at_crossing") else float("nan"),
            "mean_distance": float(np.mean(numeric_values(subset, "min_distance_to_recoverable"))) if numeric_values(subset, "min_distance_to_recoverable") else float("nan"),
            "mean_burn_a_e": float(np.mean([abs(v) for v in numeric_values(subset, "burn_a_end_energy_error_ratio")])) if numeric_values(subset, "burn_a_end_energy_error_ratio") else float("nan"),
            "mean_burn_a_h": float(np.mean([abs(v) for v in numeric_values(subset, "burn_a_end_angular_momentum_error_ratio")])) if numeric_values(subset, "burn_a_end_angular_momentum_error_ratio") else float("nan"),
            "mean_quality": float(np.mean(numeric_values(subset, "family_quality_score"))) if numeric_values(subset, "family_quality_score") else float("nan"),
        }
    return stats


def write_repo_audit() -> None:
    lines = [
        "# Phase 29 Repository Audit",
        "",
        "## Scope Reviewed",
        "",
        "- `README.md` and current milestone narrative.",
        "- `project_log/`, especially Phase 6.5-7.6 logs and `sprint_ppo22-27.md`.",
        "- Analysis outputs for Phases 20, 20.5, 21, 22, 23, 24, 25, 26, 27, and 28.",
        "- Scripts `scripts/explicit_controller_phase20*` through `scripts/explicit_controller_phase28*`.",
        "",
        "## What Has Already Been Tested",
        "",
        "- Phase 20 tested short-horizon predictive crossing-state planning and did not improve crossings, recoverability, CAPTURE, or success.",
        "- Phase 20.5 tested horizon, replan interval, score, and action-space ablations; all stayed at crossing `12`, recoverable `0`, CAPTURE `12`, success `12`.",
        "- Phase 21 tested explicit orbital energy/angular-momentum shaping; it did not expand crossing or capture reachability.",
        "- Phase 22 introduced Burn A -> coast -> Burn B and created insertion windows, but not recoverable crossings.",
        "- Phase 23 and 24 upgraded Burn B with deterministic and precision insertion solvers; conversion remained zero.",
        "- Phase 25 mapped the recoverability basin and found tangential velocity was the dominant crossing-state blocker under relaxed threshold analysis.",
        "- Phase 26 showed local vt correction exists but crossing-state vt does not improve.",
        "- Phase 27 showed timing synchronization does not improve crossing-state sync and Burn B timing has no leverage on crossing-producing cases.",
        "- Phase 28 mapped trajectory families and found dead windows dominate: window existence is not useful orbital geometry.",
        "",
        "## Eliminated Hypotheses",
        "",
        "- PPO/reward tuning is not the active path for this 2D explicit-controller sequence.",
        "- Local reactive action changes do not expand global reachability.",
        "- Short-horizon predictive planning does not solve orbital-class transition.",
        "- Naive energy/angular-momentum shaping alone does not create useful crossings.",
        "- Burn B brute force, precision insertion, vt correction, and timing synchronization are not sufficient.",
        "- Relaxing thresholds can classify some states as recoverable, but does not improve real CAPTURE count.",
        "",
        "## Controller Phases",
        "",
        "- Controller/performance phases: 20, 21, 22, 23, 24, 26, 27, and Phase 29.",
        "- Phase 20.5 is a controller ablation around Phase 20.",
        "- Phase 22 is the first staged transfer architecture.",
        "- Phase 23-24 modify Burn B only.",
        "- Phase 26-27 modify insertion/corridor timing but preserve Burn A/coast concepts.",
        "",
        "## Diagnostic Phases",
        "",
        "- Phase 20.5: predictive-planner failure diagnosis.",
        "- Phase 25: recoverability basin and threshold ablation.",
        "- Phase 28: historical trajectory-family mapping.",
        "- Phase 29 Part 0: repository audit and synthesis.",
        "",
        "## True Performance Results",
        "",
        "- Phase 7.6 remains the best broad 2D milestone in README: `217 / 270` successes in its local grid.",
        "- On the reduced 48-case benchmark used by Phases 20-29, the repeated true performance result is crossing `12`, recoverable `0`, CAPTURE `12`, success `12`, overspeed usually `0`.",
        "- Phase 22's insertion windows are a structural result, not an improved success result.",
        "",
        "## Structural Negative Findings",
        "",
        "- More horizon and more local candidates mostly increase coast selection and computation, not reachability.",
        "- Window creation is not equivalent to useful crossing geometry.",
        "- Burn B corrections can improve local quantities without changing crossing-state distributions.",
        "- The current evidence points upstream of Burn B: early trajectory-family selection.",
        "",
        "## README Narrative Files To Emphasize",
        "",
        "- `README.md` for the Phase 7.6 milestone.",
        "- `analysis/phase76_soft_hybrid/phase76_summary.md` for the best historical 2D controller milestone.",
        "- `analysis/phase20_5_ablation/summary.md` for the local predictive-planner boundary.",
        "- `analysis/phase22_two_burn_transfer/summary.md` for window creation.",
        "- `analysis/phase25_recoverability_basin_mapping/summary.md` for basin structure.",
        "- `analysis/phase28_trajectory_family_mapping/summary.md` for the current manifold-level conclusion.",
        "- `analysis/phase29_repo_audit_and_family_selector/summary.md` after this run.",
        "",
        "## Redundant Or Lower-Emphasis Files",
        "",
        "- Repeated plot variants inside intermediate phase folders are useful evidence but should not dominate README narrative.",
        "- Phase 23/24 Burn B solver plots are diagnostic negative results; emphasize their summaries rather than every candidate plot.",
        "- Phase 26/27 local correction plots should be framed as negative structural evidence, not as performance progress.",
        "- Untracked generated phase directories should be preserved, not reorganized or deleted.",
    ]
    REPO_AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")


def plot_controller_comparison(stats: Dict[str, Dict[str, object]]) -> None:
    names = list(stats)
    labels = [name.replace("phase29_", "").replace("_burn_a", "") for name in names]
    metrics = ["crossings", "near_recoverable", "recoverable", "capture", "success"]
    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 2) * width, [stats[name][metric] for name in names], width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("cases")
    ax.set_title("Phase 29 controller comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_CONTROLLER, dpi=220)
    plt.close(fig)


def save_plots(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]]) -> None:
    names = sorted({str(row["controller_name"]) for row in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(names))))

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    for color, name in zip(colors, names):
        subset = [row for row in rows if row["controller_name"] == name]
        pairs = [(as_float(row.get("burn_a_end_angular_momentum_error_ratio")), as_float(row.get("family_quality_score"))) for row in subset if math.isfinite(as_float(row.get("burn_a_end_angular_momentum_error_ratio")))]
        if pairs:
            ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], color=color, label=name.replace("phase29_", ""), s=38, alpha=0.75)
    ax.set_title("Burn A family selection map")
    ax.set_xlabel("Burn-A-end angular momentum error")
    ax.set_ylabel("family quality score")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT_BURN_A_MAP, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    family_colors = {"dead_geometry": "#9D755D", "window_no_crossing": "#F58518", "crossing_bad_sync": "#E45756", "near_recoverable_crossing": "#72B7B2", "capture_success_existing": "#54A24B"}
    for family, color in family_colors.items():
        subset = [row for row in rows if row.get("family_label") == family]
        if subset:
            ax.scatter([as_float(row.get("r0_over_target")) for row in subset], [as_float(row.get("initial_velocity_angle_deg")) for row in subset], color=color, label=family, s=42, alpha=0.75)
    ax.set_title("Initial family region")
    ax.set_xlabel("r0 / target")
    ax.set_ylabel("initial velocity angle [deg]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT_INITIAL_REGION, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    for color, name in zip(colors, names):
        subset = [row for row in rows if row["controller_name"] == name]
        pairs = [(as_float(row.get("burn_a_end_energy_error_ratio")), as_float(row.get("burn_a_end_angular_momentum_error_ratio"))) for row in subset if math.isfinite(as_float(row.get("burn_a_end_energy_error_ratio"))) and math.isfinite(as_float(row.get("burn_a_end_angular_momentum_error_ratio")))]
        if pairs:
            ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], color=color, label=name.replace("phase29_", ""), s=38, alpha=0.75)
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_title("Burn-A-end energy/angular momentum targeting")
    ax.set_xlabel("energy error ratio")
    ax.set_ylabel("angular momentum error ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT_EH_TARGET, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    labels = [name.replace("phase29_", "") for name in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [stats[name]["windows"] for name in names], 0.2, label="window cases", color="#F58518")
    ax.bar(x, [stats[name]["crossings"] for name in names], 0.2, label="crossing cases", color="#4C78A8")
    ax.bar(x + 0.2, [stats[name]["good_windows"] for name in names], 0.2, label="good windows", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Window vs crossing comparison")
    ax.set_ylabel("cases")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_WINDOW_CROSS, dpi=220)
    plt.close(fig)

    phase28_counts = Counter(row.get("phase28_reference_family_label", "") for row in rows if row.get("phase28_reference_family_label"))
    phase29_counts = Counter(row.get("family_label", "") for row in rows)
    families = ["dead_geometry", "window_no_crossing", "crossing_bad_sync", "near_recoverable_crossing", "capture_success_existing"]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(families))
    ax.bar(x - 0.18, [phase28_counts.get(f, 0) for f in families], 0.36, label="Phase 28 reference", color="#9D755D")
    ax.bar(x + 0.18, [phase29_counts.get(f, 0) for f in families], 0.36, label="Phase 29 rows", color="#72B7B2")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=20, ha="right")
    ax.set_title("Phase 29 vs Phase 28 family quality")
    ax.set_ylabel("rows")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PHASE29_PHASE28, dpi=220)
    plt.close(fig)

    plot_controller_comparison(stats)


def write_summary(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]]) -> None:
    baseline = stats.get("phase29_baseline_phase22_burn_a", {})
    best_cross = max(stats, key=lambda name: stats[name]["crossings"])
    best_quality = max(stats, key=lambda name: stats[name]["mean_quality"])
    best_dead = min(stats, key=lambda name: stats[name]["dead_windows"])
    dead_reduced = any(stats[name]["dead_windows"] < baseline.get("dead_windows", 999) for name in stats if name != "phase29_baseline_phase22_burn_a")
    crossings_increased = any(stats[name]["crossings"] > baseline.get("crossings", 0) for name in stats if name != "phase29_baseline_phase22_burn_a")
    sync_improved = any(
        math.isfinite(stats[name]["mean_sync"]) and math.isfinite(baseline.get("mean_sync", float("nan"))) and stats[name]["mean_sync"] < baseline["mean_sync"]
        for name in stats
        if name != "phase29_baseline_phase22_burn_a"
    )
    best_recoverable = max(stats[name]["recoverable"] for name in stats)
    best_capture = max(stats[name]["capture"] for name in stats)
    best_success = max(stats[name]["success"] for name in stats)
    lines = [
        "# Phase 29 Burn-A Family Selector",
        "",
        "## Scope",
        "",
        "- Full repository audit written to `repo_audit.md` before controller conclusions.",
        "- Python-only 2D Burn-A family selector benchmark on the same reduced 48-case grid.",
        "- Physics, CAPTURE/LOCK, thresholds, reward, PPO, coast, and Burn B concept are unchanged.",
        "- Phase 29 directly instruments Burn-A-end geometry.",
        "",
        "## Controller Results",
        "",
        "| Controller | Crossings | Near recoverable | Recoverable | CAPTURE | Success | Windows | Good windows | Dead windows | Mean sync | Burn-A abs h | Burn-A abs E |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in stats.items():
        lines.append(
            f"| `{name}` | {item['crossings']} | {item['near_recoverable']} | {item['recoverable']} | {item['capture']} | {item['success']} | "
            f"{item['windows']} | {item['good_windows']} | {item['dead_windows']} | {item['mean_sync']:.4f} | {item['mean_burn_a_h']:.4f} | {item['mean_burn_a_e']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Research Answers",
            "",
            f"1. Does Burn-A family selection reduce dead windows? `{'yes' if dead_reduced else 'no'}`. Best dead-window count: `{stats[best_dead]['dead_windows']}` from `{best_dead}`.",
            f"2. Does it increase crossing-producing trajectories? `{'yes' if crossings_increased else 'no'}`. Best crossing count: `{stats[best_cross]['crossings']}` from `{best_cross}`.",
            f"3. Does it improve crossing sync error? `{'yes' if sync_improved else 'no'}`.",
            f"4. Does it produce near-recoverable or recoverable crossing? Near recoverable best `{max(stats[name]['near_recoverable'] for name in stats)}`; recoverable best `{best_recoverable}`.",
            f"5. Was the Phase 28 wrong-trajectory-family hypothesis supported? `{'partially' if dead_reduced or crossings_increased or sync_improved else 'not by this selector run'}`.",
            "6. What should Phase 30 test? A wider Burn-A search over transfer-family endpoints with explicit constraints on Burn-A-end orbit crossing, not another Burn-B correction.",
            "",
            "## Success Criteria",
            "",
            "- Minimum, full repo audit completed and Burn-A-end geometry instrumented: `met`.",
            f"- Moderate, dead windows reduced: `{'met' if dead_reduced else 'not met'}`.",
            f"- Strong, crossing-producing family increased: `{'met' if crossings_increased else 'not met'}`.",
            f"- Major, recoverable/CAPTURE/success improve: `{'met' if best_recoverable > 0 or best_capture > baseline.get('capture', 0) or best_success > baseline.get('success', 0) else 'not met'}`.",
            "",
            "## Honest Interpretation",
            "",
            f"- Best mean quality variant: `{best_quality}`.",
            "- This phase tests a hypothesis generated by Phase 28; it does not prove causality from historical correlation alone.",
            "- If performance does not move, the next bottleneck is likely that the selector needs a broader Burn-A endpoint search, not that late insertion should be tuned again.",
            "",
            "## Artifacts",
            "",
            "- `repo_audit.md`",
            "- `phase29_results.csv`",
            "- `phase29_family_selector_analysis.csv`",
            "- `burn_a_family_selection_map.png`",
            "- `initial_family_region.png`",
            "- `energy_angular_momentum_targeting.png`",
            "- `window_vs_crossing_comparison.png`",
            "- `phase29_vs_phase28_family_quality.png`",
            "- `controller_comparison.png`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 29 Burn-A family selector benchmark.")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--variants", nargs="*", default=[variant.name for variant in VARIANTS])
    args = parser.parse_args()
    set_seed(7)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_repo_audit()
    selected = [variant for variant in VARIANTS if variant.name in set(args.variants)]
    rows = run_phase29_rows(selected)
    write_results(rows)
    write_analysis_csv(rows)
    stats = aggregate(rows)
    if not args.skip_plots:
        save_plots(rows, stats)
    write_summary(rows, stats)
    print(f"Saved Phase 29 repo audit and family selector outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
