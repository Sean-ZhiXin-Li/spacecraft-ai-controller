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
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase31_global_transfer_solver"
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
PHASE29_RESULTS = PROJECT_ROOT / "analysis" / "phase29_repo_audit_and_family_selector" / "phase29_results.csv"
PHASE29_ANALYSIS = PROJECT_ROOT / "analysis" / "phase29_repo_audit_and_family_selector" / "phase29_family_selector_analysis.csv"
PHASE29_SUMMARY = PROJECT_ROOT / "analysis" / "phase29_repo_audit_and_family_selector" / "summary.md"
PHASE30_RESULTS = PROJECT_ROOT / "analysis" / "phase30_burn_a_endpoint_search" / "phase30_results.csv"
PHASE30_SUMMARY = PROJECT_ROOT / "analysis" / "phase30_burn_a_endpoint_search" / "summary.md"

RESULTS_CSV = OUTPUT_DIR / "phase31_results.csv"
TRANSFER_CSV = OUTPUT_DIR / "phase31_transfer_dataset.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
RESEARCH_CONTEXT_MD = OUTPUT_DIR / "research_context.md"

PLOT_BURN_A_MAP = OUTPUT_DIR / "transfer_family_map.png"
PLOT_INITIAL_REGION = OUTPUT_DIR / "periapsis_apoapsis_transfer_map.png"
PLOT_EH_TARGET = OUTPUT_DIR / "energy_h_transfer_space.png"
PLOT_WINDOW_CROSS = OUTPUT_DIR / "transfer_quality_distribution.png"
PLOT_PHASE29_PHASE28 = OUTPUT_DIR / "phase31_vs_phase30_comparison.png"
PLOT_CONTROLLER = OUTPUT_DIR / "controller_comparison.png"

TRANSFER_PREVIEW_STEPS = 160

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
    endpoint_search: bool = False
    transfer_search: bool = False
    burn_a_steps_delta: int = 0
    max_norm_scale: float = 1.0
    radial_gain: float = 0.06
    h_gain: float = 2.2
    e_gain: float = 1.0
    family_bias: float = 0.0


VARIANTS = [
    BurnAFamilyVariant("phase31_phase22_baseline", "baseline_phase22_burn_a"),
    BurnAFamilyVariant(
        "phase31_phase30_best_baseline",
        "endpoint_crossing_quality_search",
        endpoint_search=True,
    ),
    BurnAFamilyVariant(
        "phase31_direct_transfer",
        "direct_transfer",
        transfer_search=True,
    ),
    BurnAFamilyVariant(
        "phase31_hohmann_like_transfer",
        "hohmann_like_transfer",
        transfer_search=True,
    ),
    BurnAFamilyVariant(
        "phase31_lambert_like_transfer",
        "lambert_like_transfer",
        transfer_search=True,
    ),
    BurnAFamilyVariant(
        "phase31_energy_ladder_transfer",
        "energy_ladder_transfer",
        transfer_search=True,
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
    "endpoint_candidate_id",
    "endpoint_duration",
    "endpoint_norm",
    "endpoint_radial_bias",
    "endpoint_tangential_bias",
    "endpoint_score",
    "endpoint_preview_crossing",
    "endpoint_preview_sync",
    "endpoint_preview_distance",
    "endpoint_preview_min_radius_error_ratio",
    "transfer_candidate_id",
    "transfer_family",
    "transfer_score",
    "transfer_first_coast",
    "transfer_burn1_duration",
    "transfer_burn1_norm",
    "transfer_burn1_radial_bias",
    "transfer_burn1_tangential_bias",
    "transfer_coast_duration",
    "transfer_burn2_duration",
    "transfer_burn2_norm",
    "transfer_burn2_radial_bias",
    "transfer_burn2_tangential_bias",
    "transfer_preview_crossing",
    "transfer_preview_sync",
    "transfer_preview_distance",
]

TRANSFER_FIELDS = [
    "controller_name",
    "case_id",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "candidate_id",
    "selected",
    "transfer_family",
    "first_coast",
    "burn1_duration",
    "burn1_norm",
    "burn1_radial_bias",
    "burn1_tangential_bias",
    "transfer_coast",
    "burn2_duration",
    "burn2_norm",
    "burn2_radial_bias",
    "burn2_tangential_bias",
    "transfer_score",
    "transfer_end_radius_ratio",
    "transfer_end_vr_ratio",
    "transfer_end_vt_error_ratio",
    "transfer_end_energy_error_ratio",
    "transfer_end_angular_momentum_error_ratio",
    "transfer_end_eccentricity_proxy",
    "transfer_end_periapsis_ratio",
    "transfer_end_apoapsis_ratio",
    "transfer_end_orbit_crosses_target",
    "transfer_end_predicted_crossing_step",
    "preview_crossing",
    "preview_crossing_step",
    "preview_sync_error",
    "preview_distance_to_recoverable",
    "preview_min_radius_error_ratio",
    "preview_vr_ratio",
    "preview_vt_error_ratio",
    "overspeed",
    "instability",
]

TRANSFER_ROWS: List[Dict[str, object]] = []


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


def endpoint_case_id(controller_name: str, r0: float, angle: float, thrust: float, target_scale: float) -> str:
    return f"{controller_name}|{case_id_from_values(r0, angle, thrust, target_scale)}"


def endpoint_score(strategy: str, payload: Dict[str, object], preview: Dict[str, object]) -> float:
    crosses = bool_from_csv(payload.get("orbit_crosses_target"))
    preview_crossing = bool_from_csv(preview.get("preview_crossing"))
    preview_sync = as_float(preview.get("preview_sync_error"), 12.0)
    preview_distance = as_float(preview.get("preview_distance_to_recoverable"), 12.0)
    min_radius = as_float(preview.get("preview_min_radius_error_ratio"), 1.0)
    energy = abs(as_float(payload.get("energy_error_ratio"), 4.0))
    h_error = abs(as_float(payload.get("angular_momentum_error_ratio"), 4.0))
    vt_error = abs(as_float(payload.get("vt_error_ratio"), 4.0))
    eccentricity = abs(as_float(payload.get("eccentricity_proxy"), 1.0))
    predicted = as_float(payload.get("predicted_crossing_step"), 999999.0)

    score = 0.0
    score += 8.0 if not crosses else 0.0
    score += 5.0 if not preview_crossing else 0.0
    score += 0.45 * min(preview_sync, 12.0)
    score += 0.20 * min(preview_distance, 16.0)
    score += 18.0 * min_radius
    score += 0.08 * min(predicted / 10000.0, 8.0)

    if strategy == "endpoint_energy_h_target":
        score += 2.2 * energy + 2.6 * h_error + 0.7 * vt_error
    elif strategy == "endpoint_crossing_quality_search":
        score += 0.8 * energy + 1.0 * h_error + 1.6 * vt_error + 0.5 * eccentricity
    else:
        score += 1.4 * energy + 1.5 * h_error + 1.0 * vt_error + 0.3 * eccentricity
    return float(score)


def build_endpoint_candidates(initial_diag, strategy: str) -> List[Dict[str, object]]:
    if initial_diag.r_error_ratio > 0.0:
        base_tangential = -1.0
    elif initial_diag.r_error_ratio < 0.0:
        base_tangential = 1.0
    else:
        base_tangential = 1.0 if initial_diag.angular_momentum_error_ratio < 0.0 else -1.0

    if strategy == "endpoint_energy_h_target":
        h_target_cmd = clamp(-2.2 * initial_diag.angular_momentum_error_ratio, -1.0, 1.0)
        e_target_cmd = clamp(-1.5 * initial_diag.energy_error_ratio, -1.0, 1.0)
        tangential_centers = [0.5 * base_tangential + 0.25 * h_target_cmd + 0.25 * e_target_cmd]
        durations = [8, 18]
        norms = [0.045, 0.095]
    elif strategy == "endpoint_crossing_quality_search":
        tangential_centers = [base_tangential, 0.5 * base_tangential]
        durations = [6, 16, 26]
        norms = [0.050, 0.105]
    else:
        tangential_centers = [base_tangential, 0.0]
        durations = [8, 18]
        norms = [0.050, 0.100]

    candidates: List[Dict[str, object]] = []
    candidate_id = 0
    radial_values = [-0.08, 0.0, 0.08]
    tangential_offsets = [-0.30, 0.30]
    for duration in durations:
        for norm in norms:
            for radial_bias in radial_values:
                for center in tangential_centers:
                    for offset in tangential_offsets:
                        candidate_id += 1
                        candidates.append(
                            {
                                "candidate_id": candidate_id,
                                "duration": int(duration),
                                "norm": float(norm),
                                "radial_bias": float(radial_bias),
                                "tangential_bias": float(clamp(center + offset, -1.0, 1.0)),
                            }
                        )
    return candidates


def build_transfer_candidates(initial_diag, strategy: str) -> List[Dict[str, object]]:
    if initial_diag.r_error_ratio > 0.0:
        radius_sign = -1.0
    elif initial_diag.r_error_ratio < 0.0:
        radius_sign = 1.0
    else:
        radius_sign = 1.0 if initial_diag.angular_momentum_error_ratio < 0.0 else -1.0
    h_cmd = clamp(-2.0 * initial_diag.angular_momentum_error_ratio, -1.0, 1.0)
    e_cmd = clamp(-1.2 * initial_diag.energy_error_ratio, -1.0, 1.0)

    if strategy == "direct_transfer":
        templates = [
            (0, 8, 0.075, 0.0, radius_sign, 24, 8, 0.105, -0.12 * radius_sign, h_cmd),
            (6, 10, 0.095, -0.08 * radius_sign, radius_sign, 40, 10, 0.110, 0.0, 0.5 * h_cmd + 0.5 * e_cmd),
            (12, 14, 0.110, 0.08 * radius_sign, radius_sign, 56, 8, 0.090, -0.18 * radius_sign, h_cmd),
        ]
    elif strategy == "hohmann_like_transfer":
        templates = [
            (0, 10, 0.080, 0.0, radius_sign, 80, 10, 0.095, 0.0, -radius_sign),
            (4, 12, 0.100, 0.0, radius_sign, 120, 12, 0.105, 0.0, -radius_sign),
            (8, 16, 0.110, 0.04 * radius_sign, radius_sign, 160, 10, 0.090, -0.04 * radius_sign, -radius_sign),
        ]
    elif strategy == "lambert_like_transfer":
        templates = [
            (0, 8, 0.090, -0.16 * radius_sign, 0.75 * radius_sign, 48, 12, 0.110, 0.20 * radius_sign, h_cmd),
            (8, 10, 0.100, -0.10 * radius_sign, 0.5 * radius_sign + 0.5 * h_cmd, 72, 12, 0.110, -0.22 * radius_sign, e_cmd),
            (16, 12, 0.110, 0.16 * radius_sign, 0.5 * radius_sign, 96, 14, 0.105, -0.18 * radius_sign, h_cmd),
        ]
    else:
        templates = [
            (0, 8, 0.060, 0.0, 0.5 * radius_sign, 40, 8, 0.070, 0.0, 0.5 * h_cmd),
            (8, 10, 0.080, -0.06 * radius_sign, radius_sign, 80, 10, 0.090, 0.06 * radius_sign, h_cmd),
            (16, 12, 0.100, -0.08 * radius_sign, e_cmd, 120, 12, 0.110, 0.08 * radius_sign, h_cmd),
        ]

    candidates: List[Dict[str, object]] = []
    for idx, item in enumerate(templates, start=1):
        (
            first_coast,
            burn1_duration,
            burn1_norm,
            burn1_radial,
            burn1_tangential,
            transfer_coast,
            burn2_duration,
            burn2_norm,
            burn2_radial,
            burn2_tangential,
        ) = item
        candidates.append(
            {
                "candidate_id": idx,
                "transfer_family": strategy,
                "first_coast": int(first_coast),
                "burn1_duration": int(burn1_duration),
                "burn1_norm": float(burn1_norm),
                "burn1_radial_bias": float(clamp(burn1_radial, -1.0, 1.0)),
                "burn1_tangential_bias": float(clamp(burn1_tangential, -1.0, 1.0)),
                "transfer_coast": int(transfer_coast),
                "burn2_duration": int(burn2_duration),
                "burn2_norm": float(burn2_norm),
                "burn2_radial_bias": float(clamp(burn2_radial, -1.0, 1.0)),
                "burn2_tangential_bias": float(clamp(burn2_tangential, -1.0, 1.0)),
            }
        )
    return candidates


def transfer_score(strategy: str, payload: Dict[str, object], preview: Dict[str, object], crossed_during_transfer: bool) -> float:
    crosses_orbit = bool_from_csv(payload.get("orbit_crosses_target"))
    preview_crossing = bool_from_csv(preview.get("preview_crossing"))
    preview_sync = as_float(preview.get("preview_sync_error"), 12.0)
    preview_distance = as_float(preview.get("preview_distance_to_recoverable"), 14.0)
    min_radius = as_float(preview.get("preview_min_radius_error_ratio"), 1.0)
    energy = abs(as_float(payload.get("energy_error_ratio"), 4.0))
    h_error = abs(as_float(payload.get("angular_momentum_error_ratio"), 4.0))
    vt_error = abs(as_float(payload.get("vt_error_ratio"), 4.0))
    predicted = as_float(payload.get("predicted_crossing_step"), 999999.0)
    score = 0.0
    score += 6.0 if not crosses_orbit else 0.0
    score += 4.0 if not preview_crossing and not crossed_during_transfer else 0.0
    score += 0.55 * min(preview_sync, 12.0)
    score += 0.22 * min(preview_distance, 16.0)
    score += 16.0 * min_radius
    score += 0.08 * min(predicted / 10000.0, 8.0)
    if strategy == "hohmann_like_transfer":
        score += 1.6 * energy + 1.8 * h_error + 0.8 * vt_error
    elif strategy == "lambert_like_transfer":
        score += 0.8 * energy + 1.3 * h_error + 1.8 * vt_error
    elif strategy == "energy_ladder_transfer":
        score += 2.0 * energy + 1.4 * h_error + 1.0 * vt_error
    else:
        score += 1.1 * energy + 1.2 * h_error + 1.1 * vt_error
    return float(score)


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
        if variant.endpoint_search:
            radial_cmd = as_float(selected_endpoint.get("radial_bias"), 0.0)
            tangential_cmd = as_float(selected_endpoint.get("tangential_bias"), 0.0)
            max_norm = as_float(selected_endpoint.get("norm"), 0.06)
        elif variant.strategy == "baseline_phase22_burn_a":
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

    def simulate_endpoint_candidate(candidate: Dict[str, object], candidate_index: int) -> Dict[str, object]:
        sim_x, sim_y, sim_vx, sim_vy = x, y, vx, vy
        overspeed = False
        instability = False
        for _ in range(as_int(candidate["duration"])):
            ax_cmd, ay_cmd = radial_tangential_action(
                sim_x,
                sim_y,
                sim_vx,
                sim_vy,
                as_float(candidate["radial_bias"]),
                as_float(candidate["tangential_bias"]),
                as_float(candidate["norm"]),
            )
            sim_x, sim_y, sim_vx, sim_vy = env_step(sim_x, sim_y, sim_vx, sim_vy, ax_cmd, ay_cmd)
            diag_now = orbital_diagnostics(sim_x, sim_y, sim_vx, sim_vy, target_radius)
            speed_ratio_now = diag_now.speed / (v_circ + 1.0e-12)
            if speed_ratio_now > SAFE_SPEED_RATIO:
                overspeed = True
            if diag_now.radius < 0.42 * target_radius or diag_now.radius > 2.25 * target_radius:
                instability = True

        endpoint_diag = orbital_diagnostics(sim_x, sim_y, sim_vx, sim_vy, target_radius)
        payload = diagnostic_payload(endpoint_diag, target_radius, v_circ)
        preview = preview_endpoint(sim_x, sim_y, sim_vx, sim_vy)
        score = endpoint_score(variant.strategy, payload, preview)
        return {
            "controller_name": variant.name,
            "case_id": endpoint_case_id(variant.name, r0, angle, thrust_scale, target_scale),
            "r0_over_target": r0,
            "initial_velocity_angle_deg": angle,
            "thrust_scale": thrust_scale,
            "candidate_id": candidate_index,
            "selected": False,
            "duration": candidate["duration"],
            "norm": candidate["norm"],
            "radial_bias": candidate["radial_bias"],
            "tangential_bias": candidate["tangential_bias"],
            "endpoint_score": score,
            "endpoint_radius_ratio": payload["radius_ratio"],
            "endpoint_vr_ratio": payload["vr_ratio"],
            "endpoint_vt_error_ratio": payload["vt_error_ratio"],
            "endpoint_energy_error_ratio": payload["energy_error_ratio"],
            "endpoint_angular_momentum_error_ratio": payload["angular_momentum_error_ratio"],
            "endpoint_eccentricity_proxy": payload["eccentricity_proxy"],
            "endpoint_periapsis_ratio": payload["periapsis_ratio"],
            "endpoint_apoapsis_ratio": payload["apoapsis_ratio"],
            "endpoint_orbit_crosses_target": payload["orbit_crosses_target"],
            "endpoint_predicted_crossing_step": payload["predicted_crossing_step"],
            "preview_crossing": preview["preview_crossing"],
            "preview_crossing_step": preview["preview_crossing_step"],
            "preview_sync_error": preview["preview_sync_error"],
            "preview_distance_to_recoverable": preview["preview_distance_to_recoverable"],
            "preview_min_radius_error_ratio": preview["preview_min_radius_error_ratio"],
            "preview_vr_ratio": preview["preview_vr_ratio"],
            "preview_vt_error_ratio": preview["preview_vt_error_ratio"],
            "overspeed": overspeed,
            "instability": instability,
        }

    def preview_endpoint(start_x: float, start_y: float, start_vx: float, start_vy: float) -> Dict[str, object]:
        sim_x, sim_y, sim_vx, sim_vy = start_x, start_y, start_vx, start_vy
        prev_error = sim_radius_error(sim_x, sim_y)
        best_radius_ratio = abs(prev_error) / target_radius
        best_sync = float("inf")
        best_distance = float("inf")
        best_vr = ""
        best_vt = ""
        crossing = False
        crossing_step = ""
        for preview_step in range(1, TRANSFER_PREVIEW_STEPS + 1):
            sim_x, sim_y, sim_vx, sim_vy = env_step(sim_x, sim_y, sim_vx, sim_vy, 0.0, 0.0)
            diag_now = orbital_diagnostics(sim_x, sim_y, sim_vx, sim_vy, target_radius)
            error = diag_now.radius - target_radius
            vr_ratio_now = diag_now.radial_velocity / (v_circ + 1.0e-12)
            radius_ratio_now = abs(error) / target_radius
            sync_now = sync_error(radius_ratio_now, vr_ratio_now, diag_now.vt_error_ratio)
            distance_now = recoverability_distance(radius_ratio_now, vr_ratio_now, diag_now.vt_error_ratio)
            if radius_ratio_now < best_radius_ratio:
                best_radius_ratio = radius_ratio_now
                best_vr = vr_ratio_now
                best_vt = diag_now.vt_error_ratio
            if math.isfinite(sync_now) and sync_now < best_sync:
                best_sync = sync_now
            if math.isfinite(distance_now) and distance_now < best_distance:
                best_distance = distance_now
            if (prev_error > 0.0 and error <= 0.0) or (prev_error < 0.0 and error >= 0.0):
                crossing = True
                crossing_step = preview_step
                best_vr = vr_ratio_now
                best_vt = diag_now.vt_error_ratio
                best_sync = sync_now
                best_distance = distance_now
                best_radius_ratio = radius_ratio_now
                break
            prev_error = error
        return {
            "preview_crossing": crossing,
            "preview_crossing_step": crossing_step,
            "preview_sync_error": best_sync if math.isfinite(best_sync) else "",
            "preview_distance_to_recoverable": best_distance if math.isfinite(best_distance) else "",
            "preview_min_radius_error_ratio": best_radius_ratio,
            "preview_vr_ratio": best_vr,
            "preview_vt_error_ratio": best_vt,
        }

    def sim_radius_error(state_x: float, state_y: float) -> float:
        return math.sqrt(state_x * state_x + state_y * state_y) - target_radius

    def apply_open_loop_burn(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        duration: int,
        norm: float,
        radial_bias: float,
        tangential_bias: float,
    ) -> tuple[float, float, float, float, bool, bool, bool, float, float, float]:
        sim_x, sim_y, sim_vx, sim_vy = state_x, state_y, state_vx, state_vy
        overspeed = False
        instability = False
        crossed = False
        previous_error = sim_radius_error(sim_x, sim_y)
        best_radius_ratio = abs(previous_error) / target_radius
        best_sync = float("inf")
        best_distance = float("inf")
        for _ in range(duration):
            ax_cmd, ay_cmd = radial_tangential_action(sim_x, sim_y, sim_vx, sim_vy, radial_bias, tangential_bias, norm)
            sim_x, sim_y, sim_vx, sim_vy = env_step(sim_x, sim_y, sim_vx, sim_vy, ax_cmd, ay_cmd)
            diag_now = orbital_diagnostics(sim_x, sim_y, sim_vx, sim_vy, target_radius)
            error = diag_now.radius - target_radius
            vr_ratio_now = diag_now.radial_velocity / (v_circ + 1.0e-12)
            radius_ratio_now = abs(error) / target_radius
            sync_now = sync_error(radius_ratio_now, vr_ratio_now, diag_now.vt_error_ratio)
            distance_now = recoverability_distance(radius_ratio_now, vr_ratio_now, diag_now.vt_error_ratio)
            best_radius_ratio = min(best_radius_ratio, radius_ratio_now)
            if math.isfinite(sync_now):
                best_sync = min(best_sync, sync_now)
            if math.isfinite(distance_now):
                best_distance = min(best_distance, distance_now)
            crossed = crossed or ((previous_error > 0.0 and error <= 0.0) or (previous_error < 0.0 and error >= 0.0))
            previous_error = error
            overspeed = overspeed or diag_now.speed / (v_circ + 1.0e-12) > SAFE_SPEED_RATIO
            instability = instability or diag_now.radius < 0.42 * target_radius or diag_now.radius > 2.25 * target_radius
        return (
            sim_x,
            sim_y,
            sim_vx,
            sim_vy,
            overspeed,
            instability,
            crossed,
            best_radius_ratio,
            best_sync,
            best_distance,
        )

    def apply_open_loop_coast(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        duration: int,
    ) -> tuple[float, float, float, float, bool, bool, bool, float, float, float]:
        sim_x, sim_y, sim_vx, sim_vy = state_x, state_y, state_vx, state_vy
        overspeed = False
        instability = False
        crossed = False
        previous_error = sim_radius_error(sim_x, sim_y)
        best_radius_ratio = abs(previous_error) / target_radius
        best_sync = float("inf")
        best_distance = float("inf")
        for _ in range(duration):
            sim_x, sim_y, sim_vx, sim_vy = env_step(sim_x, sim_y, sim_vx, sim_vy, 0.0, 0.0)
            diag_now = orbital_diagnostics(sim_x, sim_y, sim_vx, sim_vy, target_radius)
            error = diag_now.radius - target_radius
            vr_ratio_now = diag_now.radial_velocity / (v_circ + 1.0e-12)
            radius_ratio_now = abs(error) / target_radius
            sync_now = sync_error(radius_ratio_now, vr_ratio_now, diag_now.vt_error_ratio)
            distance_now = recoverability_distance(radius_ratio_now, vr_ratio_now, diag_now.vt_error_ratio)
            best_radius_ratio = min(best_radius_ratio, radius_ratio_now)
            if math.isfinite(sync_now):
                best_sync = min(best_sync, sync_now)
            if math.isfinite(distance_now):
                best_distance = min(best_distance, distance_now)
            crossed = crossed or ((previous_error > 0.0 and error <= 0.0) or (previous_error < 0.0 and error >= 0.0))
            previous_error = error
            overspeed = overspeed or diag_now.speed / (v_circ + 1.0e-12) > SAFE_SPEED_RATIO
            instability = instability or diag_now.radius < 0.42 * target_radius or diag_now.radius > 2.25 * target_radius
        return (
            sim_x,
            sim_y,
            sim_vx,
            sim_vy,
            overspeed,
            instability,
            crossed,
            best_radius_ratio,
            best_sync,
            best_distance,
        )

    def simulate_transfer_candidate(candidate: Dict[str, object], candidate_index: int) -> Dict[str, object]:
        sim_x, sim_y, sim_vx, sim_vy = x, y, vx, vy
        overspeed = False
        instability = False
        crossed = False
        min_radius_ratio = float("inf")
        best_sync = float("inf")
        best_distance = float("inf")
        segments = [
            ("coast", as_int(candidate["first_coast"]), 0.0, 0.0, 0.0),
            ("burn", as_int(candidate["burn1_duration"]), as_float(candidate["burn1_norm"]), as_float(candidate["burn1_radial_bias"]), as_float(candidate["burn1_tangential_bias"])),
            ("coast", as_int(candidate["transfer_coast"]), 0.0, 0.0, 0.0),
            ("burn", as_int(candidate["burn2_duration"]), as_float(candidate["burn2_norm"]), as_float(candidate["burn2_radial_bias"]), as_float(candidate["burn2_tangential_bias"])),
        ]
        for kind, duration, norm, radial_bias, tangential_bias in segments:
            if kind == "coast":
                result = apply_open_loop_coast(sim_x, sim_y, sim_vx, sim_vy, duration)
            else:
                result = apply_open_loop_burn(sim_x, sim_y, sim_vx, sim_vy, duration, norm, radial_bias, tangential_bias)
            sim_x, sim_y, sim_vx, sim_vy, over, unstable, did_cross, radius_best, sync_best, distance_best = result
            overspeed = overspeed or over
            instability = instability or unstable
            crossed = crossed or did_cross
            min_radius_ratio = min(min_radius_ratio, radius_best)
            best_sync = min(best_sync, sync_best)
            best_distance = min(best_distance, distance_best)
        transfer_diag = orbital_diagnostics(sim_x, sim_y, sim_vx, sim_vy, target_radius)
        payload = diagnostic_payload(transfer_diag, target_radius, v_circ)
        preview = preview_endpoint(sim_x, sim_y, sim_vx, sim_vy)
        if not math.isfinite(as_float(preview.get("preview_min_radius_error_ratio"))):
            preview["preview_min_radius_error_ratio"] = min_radius_ratio
        if not preview.get("preview_sync_error") and math.isfinite(best_sync):
            preview["preview_sync_error"] = best_sync
        if not preview.get("preview_distance_to_recoverable") and math.isfinite(best_distance):
            preview["preview_distance_to_recoverable"] = best_distance
        score = transfer_score(variant.strategy, payload, preview, crossed)
        return {
            "controller_name": variant.name,
            "case_id": endpoint_case_id(variant.name, r0, angle, thrust_scale, target_scale),
            "r0_over_target": r0,
            "initial_velocity_angle_deg": angle,
            "thrust_scale": thrust_scale,
            "candidate_id": candidate_index,
            "selected": False,
            "transfer_family": candidate["transfer_family"],
            "first_coast": candidate["first_coast"],
            "burn1_duration": candidate["burn1_duration"],
            "burn1_norm": candidate["burn1_norm"],
            "burn1_radial_bias": candidate["burn1_radial_bias"],
            "burn1_tangential_bias": candidate["burn1_tangential_bias"],
            "transfer_coast": candidate["transfer_coast"],
            "burn2_duration": candidate["burn2_duration"],
            "burn2_norm": candidate["burn2_norm"],
            "burn2_radial_bias": candidate["burn2_radial_bias"],
            "burn2_tangential_bias": candidate["burn2_tangential_bias"],
            "transfer_score": score,
            "transfer_end_radius_ratio": payload["radius_ratio"],
            "transfer_end_vr_ratio": payload["vr_ratio"],
            "transfer_end_vt_error_ratio": payload["vt_error_ratio"],
            "transfer_end_energy_error_ratio": payload["energy_error_ratio"],
            "transfer_end_angular_momentum_error_ratio": payload["angular_momentum_error_ratio"],
            "transfer_end_eccentricity_proxy": payload["eccentricity_proxy"],
            "transfer_end_periapsis_ratio": payload["periapsis_ratio"],
            "transfer_end_apoapsis_ratio": payload["apoapsis_ratio"],
            "transfer_end_orbit_crosses_target": payload["orbit_crosses_target"],
            "transfer_end_predicted_crossing_step": payload["predicted_crossing_step"],
            "preview_crossing": preview["preview_crossing"] or crossed,
            "preview_crossing_step": preview["preview_crossing_step"],
            "preview_sync_error": preview["preview_sync_error"],
            "preview_distance_to_recoverable": preview["preview_distance_to_recoverable"],
            "preview_min_radius_error_ratio": preview["preview_min_radius_error_ratio"],
            "preview_vr_ratio": preview["preview_vr_ratio"],
            "preview_vt_error_ratio": preview["preview_vt_error_ratio"],
            "overspeed": overspeed,
            "instability": instability,
        }

    initial_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
    initial_energy_error = abs(initial_diag.energy_error_ratio)
    initial_h_error = initial_diag.angular_momentum_error_ratio
    selected_endpoint: Dict[str, object] = {}
    selected_transfer: Dict[str, object] = {}
    if variant.endpoint_search:
        endpoint_rows = [
            simulate_endpoint_candidate(candidate, idx)
            for idx, candidate in enumerate(build_endpoint_candidates(initial_diag, variant.strategy), start=1)
        ]
        stable_rows = [row for row in endpoint_rows if not bool_from_csv(row.get("overspeed")) and not bool_from_csv(row.get("instability"))]
        selected_endpoint = min(stable_rows or endpoint_rows, key=lambda row: as_float(row.get("endpoint_score"), 999999.0))
        burn_a_target_steps = as_int(selected_endpoint.get("duration"), burn_a_target_steps)
    if variant.transfer_search:
        transfer_rows = [
            simulate_transfer_candidate(candidate, idx)
            for idx, candidate in enumerate(build_transfer_candidates(initial_diag, variant.strategy), start=1)
        ]
        stable_rows = [row for row in transfer_rows if not bool_from_csv(row.get("overspeed")) and not bool_from_csv(row.get("instability"))]
        selected_transfer = min(stable_rows or transfer_rows, key=lambda row: as_float(row.get("transfer_score"), 999999.0))
        for row in transfer_rows:
            row["selected"] = int(row["candidate_id"]) == int(selected_transfer["candidate_id"])
            TRANSFER_ROWS.append(row)
        transfer_stage = "global_first_coast"
        if as_int(selected_transfer.get("first_coast"), 0) <= 0:
            transfer_stage = "global_burn1"

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
            elif abs(diag.r_error_ratio) <= PREWINDOW_OUTER_RATIO and not variant.transfer_search:
                action_x, action_y = soft_linear_3e4_action(x, y, vx, vy, diag.radius, diag.r_error_ratio, abs(diag.r_error_ratio))
                active_stage = "phase76_handoff"
                handoff_ready_count += 1
                handoff_entered = True
                transfer_stage = "phase76_handoff"
            elif intercept_class == "handoff_ready" and not variant.transfer_search:
                action_x, action_y = soft_linear_3e4_action(x, y, vx, vy, diag.radius, diag.r_error_ratio, abs(diag.r_error_ratio))
                active_stage = "phase76_handoff"
                handoff_ready_count += 1
                handoff_entered = True
            elif transfer_stage == "global_first_coast":
                action_x, action_y = 0.0, 0.0
                active_stage = "global_first_coast"
                coast_arc_steps += 1
                if stage_step + 1 >= as_int(selected_transfer.get("first_coast"), 0):
                    transfer_stage = "global_burn1"
                    stage_step = -1
            elif transfer_stage == "global_burn1":
                action_x, action_y = radial_tangential_action(
                    x,
                    y,
                    vx,
                    vy,
                    as_float(selected_transfer.get("burn1_radial_bias"), 0.0),
                    as_float(selected_transfer.get("burn1_tangential_bias"), 0.0),
                    as_float(selected_transfer.get("burn1_norm"), 0.0),
                )
                active_stage = "global_burn1"
                burn_a_steps += 1
                if stage_step + 1 >= as_int(selected_transfer.get("burn1_duration"), 0):
                    transfer_stage = "global_transfer_coast"
                    stage_step = -1
            elif transfer_stage == "global_transfer_coast":
                action_x, action_y = 0.0, 0.0
                active_stage = "global_transfer_coast"
                coast_arc_steps += 1
                if stage_step + 1 >= as_int(selected_transfer.get("transfer_coast"), 0):
                    transfer_stage = "global_burn2"
                    stage_step = -1
            elif transfer_stage == "global_burn2":
                action_x, action_y = radial_tangential_action(
                    x,
                    y,
                    vx,
                    vy,
                    as_float(selected_transfer.get("burn2_radial_bias"), 0.0),
                    as_float(selected_transfer.get("burn2_tangential_bias"), 0.0),
                    as_float(selected_transfer.get("burn2_norm"), 0.0),
                )
                active_stage = "global_burn2"
                burn_a_steps += 1
                if stage_step + 1 >= as_int(selected_transfer.get("burn2_duration"), 0):
                    end_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
                    burn_a_end = diagnostic_payload(end_diag, target_radius, v_circ)
                    burn_a_end["step"] = step
                    transfer_stage = "coast_arc"
                    stage_step = -1
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
        "endpoint_candidate_id": selected_endpoint.get("candidate_id", ""),
        "endpoint_duration": selected_endpoint.get("duration", ""),
        "endpoint_norm": selected_endpoint.get("norm", ""),
        "endpoint_radial_bias": selected_endpoint.get("radial_bias", ""),
        "endpoint_tangential_bias": selected_endpoint.get("tangential_bias", ""),
        "endpoint_score": selected_endpoint.get("endpoint_score", ""),
        "endpoint_preview_crossing": selected_endpoint.get("preview_crossing", ""),
        "endpoint_preview_sync": selected_endpoint.get("preview_sync_error", ""),
        "endpoint_preview_distance": selected_endpoint.get("preview_distance_to_recoverable", ""),
        "endpoint_preview_min_radius_error_ratio": selected_endpoint.get("preview_min_radius_error_ratio", ""),
        "transfer_candidate_id": selected_transfer.get("candidate_id", ""),
        "transfer_family": selected_transfer.get("transfer_family", ""),
        "transfer_score": selected_transfer.get("transfer_score", ""),
        "transfer_first_coast": selected_transfer.get("first_coast", ""),
        "transfer_burn1_duration": selected_transfer.get("burn1_duration", ""),
        "transfer_burn1_norm": selected_transfer.get("burn1_norm", ""),
        "transfer_burn1_radial_bias": selected_transfer.get("burn1_radial_bias", ""),
        "transfer_burn1_tangential_bias": selected_transfer.get("burn1_tangential_bias", ""),
        "transfer_coast_duration": selected_transfer.get("transfer_coast", ""),
        "transfer_burn2_duration": selected_transfer.get("burn2_duration", ""),
        "transfer_burn2_norm": selected_transfer.get("burn2_norm", ""),
        "transfer_burn2_radial_bias": selected_transfer.get("burn2_radial_bias", ""),
        "transfer_burn2_tangential_bias": selected_transfer.get("burn2_tangential_bias", ""),
        "transfer_preview_crossing": selected_transfer.get("preview_crossing", ""),
        "transfer_preview_sync": selected_transfer.get("preview_sync_error", ""),
        "transfer_preview_distance": selected_transfer.get("preview_distance_to_recoverable", ""),
    }
    return row


def run_phase31_rows(variants: Sequence[BurnAFamilyVariant]) -> List[Dict[str, object]]:
    phase28_ref = load_phase28_reference()
    rows: List[Dict[str, object]] = []
    cases = list(iter_cases(variants))
    for idx, (variant, r0, angle, thrust, target_scale) in enumerate(cases, start=1):
        row = rollout_case(variant, r0, angle, thrust, target_scale, phase28_ref)
        rows.append(row)
        print(
            f"phase31 {idx}/{len(cases)} {variant.name} r0={r0:g} angle={angle:g} thrust={thrust:g} "
            f"cross={row['crossing_occurs']} win={row['insertion_windows']} fam={row['family_label']} "
            f"cap={row['capture_entered']} succ={row['success']} transfer={row.get('transfer_candidate_id', '')} reason={row['termination_reason']}"
        )
    return rows


def write_results(rows: Sequence[Dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def write_transfer_csv(rows: Sequence[Dict[str, object]]) -> None:
    with TRANSFER_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRANSFER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key, "") for key in TRANSFER_FIELDS}
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


def write_research_context() -> None:
    lines = [
        "# Phase 31 Research Context",
        "",
        "## Scope Reviewed",
        "",
        "- `README.md` for the Phase 7.6 milestone and current 2D framing.",
        "- Phase 25 recoverability basin mapping.",
        "- Phase 28 trajectory-family mapping.",
        "- Phase 29 repo audit and Burn-A family selector outputs.",
        "- Phase 30 Burn-A endpoint search outputs.",
        "- Phase 22 and Phase 30 controller scripts.",
        "",
        "## Why Burn A/B May Be Structurally Insufficient",
        "",
        "- Phase 23 and Phase 24 changed Burn B search and insertion geometry without producing recoverable crossings.",
        "- Phase 26 showed tangential correction can occur locally but does not move the crossing-state distribution.",
        "- Phase 27 showed Burn-B timing does not reduce crossing sync error on the cases that actually cross.",
        "- Those results imply Burn B is acting after the relevant orbital family has already been selected.",
        "- Phase 30 searched Burn-A endpoints and still did not move crossing-state structure.",
        "- The Burn A -> coast -> Burn B architecture may not be expressive enough to synchronize radius, vr, and vt.",
        "",
        "## Why Endpoint Manifold Search Failed",
        "",
        "- Phase 25 identified tangential velocity as the dominant recoverability blocker.",
        "- Phase 30 mapped 6912 bounded Burn-A endpoint candidates.",
        "- Selected endpoints altered energy and angular momentum but left crossings, near-recoverability, CAPTURE, and success unchanged.",
        "- Endpoint selection optimizes a state, not a complete transfer path.",
        "",
        "## Why Local Geometry May Not Solve Global Transfer",
        "",
        "- A useful endpoint is not enough if the transfer arc and insertion timing are not globally designed.",
        "- Transfer path geometry determines when radius crossing, radial velocity, and tangential velocity align.",
        "- Prior phases repeatedly changed local behavior without changing crossing-state families.",
        "",
        "## Why Transfer-Class Reasoning Is Required",
        "",
        "- Phase 31 searches named transfer classes instead of action heuristics.",
        "- Direct, Hohmann-like, Lambert-like, and energy-ladder families encode different physical assumptions.",
        "- Each candidate explicitly contains burn timing, burn magnitudes, burn directions, coast arcs, and insertion-state preview.",
        "- This is architecture-level transfer design, not random brute force.",
        "",
        "## Control Architecture Comparison",
        "",
        "- Local control chooses immediate action and failed to expand reachability.",
        "- Staged burn control creates insertion windows but not recoverable crossings.",
        "- Endpoint search chooses a post-Burn-A state but not a complete orbital path.",
        "- Global transfer search chooses a path class connecting initial state to a target-family crossing geometry.",
        "",
        "## Physical Definition Of Global Transfer",
        "",
        "- Orbital energy ladder: the sequence of energy changes across burns and coast arcs.",
        "- Angular momentum transition: deliberate changes in tangential support before insertion.",
        "- Periapsis shaping: lowering or raising closest approach to create target-radius intersection.",
        "- Apoapsis shaping: controlling the far point of the transfer ellipse.",
        "- Transfer arc geometry: the coast segment that brings the spacecraft to the target radius.",
        "- Insertion-state targeting: previewing radius error, vr, vt, energy, and angular momentum near the future crossing.",
        "",
        "## Phase 31 Hypothesis",
        "",
        "- If architecture class is the missing layer, at least one global transfer family should reduce dead windows or improve crossing-state quality.",
        "- If this still fails, Phase 32 should move from explicit transfer families to true optimal control, direct collocation, or MPC.",
    ]
    RESEARCH_CONTEXT_MD.write_text("\n".join(lines), encoding="utf-8")


def plot_controller_comparison(stats: Dict[str, Dict[str, object]]) -> None:
    names = list(stats)
    labels = [name.replace("phase31_", "") for name in names]
    metrics = ["crossings", "near_recoverable", "recoverable", "capture", "success"]
    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 2) * width, [stats[name][metric] for name in names], width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("cases")
    ax.set_title("Phase 31 controller comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_CONTROLLER, dpi=220)
    plt.close(fig)


def save_plots(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]]) -> None:
    names = sorted({str(row["controller_name"]) for row in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(names))))

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    transfer_rows = TRANSFER_ROWS
    selected_rows = [row for row in transfer_rows if bool_from_csv(row.get("selected"))]
    if transfer_rows:
        all_x = [as_float(row.get("transfer_end_energy_error_ratio")) for row in transfer_rows]
        all_y = [as_float(row.get("transfer_end_angular_momentum_error_ratio")) for row in transfer_rows]
        all_c = [as_float(row.get("transfer_score"), 0.0) for row in transfer_rows]
        scatter = ax.scatter(all_x, all_y, c=all_c, cmap="viridis_r", s=20, alpha=0.45, label="candidate transfers")
        if selected_rows:
            ax.scatter(
                [as_float(row.get("transfer_end_energy_error_ratio")) for row in selected_rows],
                [as_float(row.get("transfer_end_angular_momentum_error_ratio")) for row in selected_rows],
                facecolors="none",
                edgecolors="#E45756",
                s=52,
                linewidths=1.0,
                label="selected transfers",
            )
        fig.colorbar(scatter, ax=ax, label="transfer score")
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_title("Transfer family map")
    ax.set_xlabel("transfer-end energy error ratio")
    ax.set_ylabel("transfer-end angular momentum error ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT_BURN_A_MAP, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for color, name in zip(colors, names):
        subset = [row for row in rows if row["controller_name"] == name and math.isfinite(as_float(row.get("burn_a_end_periapsis_ratio"))) and math.isfinite(as_float(row.get("burn_a_end_apoapsis_ratio")))]
        if subset:
            ax.scatter(
                [as_float(row.get("burn_a_end_periapsis_ratio")) for row in subset],
                [as_float(row.get("burn_a_end_apoapsis_ratio")) for row in subset],
                color=color,
                label=name.replace("phase31_", ""),
                s=42,
                alpha=0.75,
            )
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_title("Periapsis/apoapsis transfer map")
    ax.set_xlabel("periapsis / target")
    ax.set_ylabel("apoapsis / target")
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
            ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], color=color, label=name.replace("phase31_", ""), s=38, alpha=0.75)
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_title("Energy/angular-momentum transfer space")
    ax.set_xlabel("transfer-end energy error ratio")
    ax.set_ylabel("transfer-end angular momentum error ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT_EH_TARGET, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    scores = [as_float(row.get("transfer_score")) for row in transfer_rows if math.isfinite(as_float(row.get("transfer_score")))]
    if scores:
        ax.hist(scores, bins=34, color="#4C78A8", alpha=0.82)
    ax.set_title("Transfer quality distribution")
    ax.set_xlabel("transfer score, lower is better")
    ax.set_ylabel("candidate transfers")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_WINDOW_CROSS, dpi=220)
    plt.close(fig)

    phase30_stats = aggregate(read_csv(PHASE30_RESULTS)) if PHASE30_RESULTS.exists() else {}
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    labels = ["crossings", "near_recoverable", "recoverable", "capture", "success", "dead_windows"]
    p30 = phase30_stats.get("phase30_endpoint_crossing_quality_search", {})
    p31 = stats.get("phase31_lambert_like_transfer", {})
    x = np.arange(len(labels))
    ax.bar(x - 0.18, [p30.get(label, 0) for label in labels], 0.36, label="Phase 30 endpoint search", color="#9D755D")
    ax.bar(x + 0.18, [p31.get(label, 0) for label in labels], 0.36, label="Phase 31 Lambert-like", color="#72B7B2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Phase 31 vs Phase 30 comparison")
    ax.set_ylabel("cases")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PHASE29_PHASE28, dpi=220)
    plt.close(fig)

    plot_controller_comparison(stats)


def write_summary(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]]) -> None:
    baseline = stats.get("phase31_phase22_baseline", {})
    best_cross = max(stats, key=lambda name: stats[name]["crossings"])
    best_quality = max(stats, key=lambda name: stats[name]["mean_quality"])
    best_dead = min(stats, key=lambda name: stats[name]["dead_windows"])
    dead_reduced = any(stats[name]["dead_windows"] < baseline.get("dead_windows", 999) for name in stats if name != "phase31_phase22_baseline")
    crossings_increased = any(stats[name]["crossings"] > baseline.get("crossings", 0) for name in stats if name != "phase31_phase22_baseline")
    sync_improved = any(
        math.isfinite(stats[name]["mean_sync"]) and math.isfinite(baseline.get("mean_sync", float("nan"))) and stats[name]["mean_sync"] < baseline["mean_sync"]
        for name in stats
        if name != "phase31_phase22_baseline"
    )
    best_recoverable = max(stats[name]["recoverable"] for name in stats)
    best_capture = max(stats[name]["capture"] for name in stats)
    best_success = max(stats[name]["success"] for name in stats)
    transfer_rows = TRANSFER_ROWS
    selected_transfer_rows = [row for row in transfer_rows if bool_from_csv(row.get("selected"))]
    mapped_count = len(transfer_rows)
    selected_crossing_preview = sum(bool_from_csv(row.get("preview_crossing")) for row in selected_transfer_rows)
    phase30_stats = aggregate(read_csv(PHASE30_RESULTS)) if PHASE30_RESULTS.exists() else {}
    phase30_best = phase30_stats.get("phase30_endpoint_crossing_quality_search", {})
    transfer_better_than_phase30 = bool(best_recoverable > phase30_best.get("recoverable", 0) or best_capture > phase30_best.get("capture", 0) or best_success > phase30_best.get("success", 0))
    lines = [
        "# Phase 31 Global Transfer Solver",
        "",
        "## Scope",
        "",
        "- Research context written to `research_context.md` before controller conclusions.",
        "- Python-only 2D global transfer-family benchmark on the same reduced 48-case grid.",
        "- Physics, CAPTURE/LOCK, thresholds, reward, PPO, coast, and Burn B concept are unchanged.",
        "- Phase 31 searches bounded transfer families over burn timing, burn magnitude, burn direction, and coast duration.",
        f"- Transfer candidates mapped: `{mapped_count}`. Selected transfer previews with crossing: `{selected_crossing_preview}`.",
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
            f"1. Does global transfer improve crossings? `{'yes' if crossings_increased else 'no'}`. Best crossing count: `{stats[best_cross]['crossings']}` from `{best_cross}`.",
            f"2. Does it reduce dead windows? `{'yes' if dead_reduced else 'no'}`. Best dead-window count: `{stats[best_dead]['dead_windows']}` from `{best_dead}`.",
            f"3. Does it improve near-recoverable crossings? `{'yes' if max(stats[name]['near_recoverable'] for name in stats) > baseline.get('near_recoverable', 0) else 'no'}`. Best near-recoverable count: `{max(stats[name]['near_recoverable'] for name in stats)}`.",
            f"4. Does it improve recoverable crossings? `{'yes' if best_recoverable > baseline.get('recoverable', 0) else 'no'}`.",
            f"5. Does it improve CAPTURE? `{'yes' if best_capture > baseline.get('capture', 0) else 'no'}`.",
            f"6. Which transfer family performs best? `{best_quality}` by mean family quality.",
            f"7. Is architecture class the real bottleneck? `{'weakly suggested but not proven' if sync_improved and not (dead_reduced or crossings_increased or transfer_better_than_phase30) else ('supported' if (dead_reduced or crossings_increased or transfer_better_than_phase30) else 'not supported by this bounded family search')}`.",
            "8. What is Phase 32? If Phase 31 does not move crossing-state structure, Phase 32 should test true optimal control: direct collocation, MPC, or CasADi-based trajectory optimization.",
            "",
            "## Success Criteria",
            "",
            "- Minimum, global transfer families mapped: `met`.",
            f"- Moderate, dead windows reduced: `{'met' if dead_reduced else 'not met'}`.",
            f"- Strong, crossing count improves: `{'met' if crossings_increased else 'not met'}`.",
            f"- Major, recoverable crossings improve: `{'met' if best_recoverable > baseline.get('recoverable', 0) else 'not met'}`.",
            f"- Breakthrough, CAPTURE/success improves: `{'met' if best_capture > baseline.get('capture', 0) or best_success > baseline.get('success', 0) else 'not met'}`.",
            "",
            "## Honest Interpretation",
            "",
            f"- Best mean quality variant: `{best_quality}`.",
            f"- Compared with Phase 30 endpoint search, global transfer improvement in recoverable/CAPTURE/success: `{'yes' if transfer_better_than_phase30 else 'no'}`.",
            f"- Mean sync improvement without crossing/CAPTURE improvement: `{'yes' if sync_improved and not crossings_increased else 'no'}`.",
            "- This phase maps named transfer families and tests whether transfer class alters downstream crossing structure.",
            "- If performance does not move, even global transfer family redesign does not solve the current orbital insertion structure.",
            "",
            "## Artifacts",
            "",
            "- `research_context.md`",
            "- `phase31_results.csv`",
            "- `phase31_transfer_dataset.csv`",
            "- `transfer_family_map.png`",
            "- `energy_h_transfer_space.png`",
            "- `periapsis_apoapsis_transfer_map.png`",
            "- `transfer_quality_distribution.png`",
            "- `phase31_vs_phase30_comparison.png`",
            "- `controller_comparison.png`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 31 global transfer solver benchmark.")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--variants", nargs="*", default=[variant.name for variant in VARIANTS])
    args = parser.parse_args()
    set_seed(7)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_research_context()
    selected = [variant for variant in VARIANTS if variant.name in set(args.variants)]
    rows = run_phase31_rows(selected)
    write_results(rows)
    write_transfer_csv(TRANSFER_ROWS)
    stats = aggregate(rows)
    if not args.skip_plots:
        save_plots(rows, stats)
    write_summary(rows, stats)
    print(f"Saved Phase 31 global transfer solver outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
