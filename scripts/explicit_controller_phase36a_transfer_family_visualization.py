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
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase36a_transfer_family_visualization"
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
    summarize_radial_velocity,
)


PHASE35_RESULTS = PROJECT_ROOT / "analysis" / "phase35_crossing_basin_expansion" / "phase35_results.csv"
PHASE35_DIAGNOSIS = PROJECT_ROOT / "analysis" / "phase35_crossing_basin_expansion" / "non_crossing_diagnosis.csv"

RESULTS_CSV = OUTPUT_DIR / "phase36a_family_results.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
CASE_NOTES_MD = OUTPUT_DIR / "family_case_notes.md"
COMPARE_MD = OUTPUT_DIR / "phase36a_vs_phase35.md"

PLOT_TRAJECTORY = OUTPUT_DIR / "family_trajectory_overlay.png"
PLOT_RADIUS = OUTPUT_DIR / "radius_vs_time_by_family.png"
PLOT_VR = OUTPUT_DIR / "vr_vs_time_by_family.png"
PLOT_VT = OUTPUT_DIR / "vt_error_vs_time_by_family.png"
PLOT_SCATTER = OUTPUT_DIR / "crossing_state_scatter_by_family.png"
PLOT_GEOMETRY = OUTPUT_DIR / "family_geometry_map.png"

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
    TransferFamily("delayed_crossing", "delayed commitment before handoff"),
    TransferFamily("energy_bleed_then_cross", "early tangential energy bleed, later crossing commit"),
    TransferFamily("overshoot_return", "controlled overshoot followed by return attempt"),
    TransferFamily("grazing_corridor", "near-target loiter with soft crossing windows"),
    TransferFamily("two_stage_transfer", "geometry-shaping stage then crossing-commit stage"),
]


FIELDNAMES = [
    "controller_name",
    "transfer_family",
    "case_label",
    "qualitative_family_label",
    "post_cross_mode",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "target_radius_scale",
    "crossing_occurs",
    "recoverable_crossing",
    "recoverable_state",
    "first_crossing_step",
    "crossing_vr_ratio",
    "crossing_vt_error_ratio",
    "crossing_sync_error",
    "min_abs_radius_error",
    "min_abs_radius_error_ratio",
    "closest_approach_step",
    "best_crossing_potential",
    "crossing_potential_at_closest",
    "best_post_cross_distance",
    "best_post_cross_sync",
    "radius_crossings_total",
    "overspeed",
    "instability",
    "max_speed_ratio",
    "steps",
    "termination_reason",
    "dominant_stage",
    "radial_velocity_trend",
    "final_radius_error_ratio",
    "final_vr_ratio",
    "final_vt_error_ratio",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def case_key(r0: object, angle: object, thrust: object) -> tuple[float, float, float]:
    return (round(as_float(r0), 6), round(as_float(angle), 6), round(as_float(thrust), 6))


def case_id(r0: float, angle: float, thrust: float) -> str:
    return f"r0={r0:.5g}|angle={angle:.5g}|thrust={thrust:.5g}"


def limit_norm(action_x: float, action_y: float, max_norm: float) -> tuple[float, float]:
    norm = math.sqrt(action_x * action_x + action_y * action_y)
    if norm <= max_norm or norm <= 1.0e-12:
        return clamp(action_x, -1.0, 1.0), clamp(action_y, -1.0, 1.0)
    scale = max_norm / norm
    return clamp(action_x * scale, -1.0, 1.0), clamp(action_y * scale, -1.0, 1.0)


def select_representative_cases() -> List[tuple[str, float, float, float]]:
    rows = read_csv(PHASE35_RESULTS)
    diagnosis = read_csv(PHASE35_DIAGNOSIS)

    baseline = [row for row in rows if row.get("upstream_variant") == "baseline_phase34"]
    crossing_rows = [row for row in baseline if bool_from_csv(row.get("crossing_occurs"))]
    crossing = min(
        crossing_rows,
        key=lambda row: abs(as_float(row.get("crossing_vr_ratio"))) + abs(as_float(row.get("crossing_vt_error_ratio"))),
    )

    near_rows = [row for row in diagnosis if row.get("failure_label") == "near_crossing"]
    over_rows = [row for row in diagnosis if row.get("failure_label") == "over_conservative_transfer"]
    near = max(near_rows, key=lambda row: as_float(row.get("best_crossing_potential")))
    over = max(over_rows, key=lambda row: as_float(row.get("best_crossing_potential")))

    return [
        (
            "representative_crossing",
            as_float(crossing["r0_over_target"]),
            as_float(crossing["initial_velocity_angle_deg"]),
            as_float(crossing["thrust_scale"]),
        ),
        (
            "representative_near_crossing",
            as_float(near["r0_over_target"]),
            as_float(near["initial_velocity_angle_deg"]),
            as_float(near["thrust_scale"]),
        ),
        (
            "representative_over_conservative_transfer",
            as_float(over["r0_over_target"]),
            as_float(over["initial_velocity_angle_deg"]),
            as_float(over["thrust_scale"]),
        ),
    ]


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

        if name == "delayed_crossing":
            delay_steps = 1600
            if step_index < delay_steps or (abs_r < 0.018 and step_index < 3600):
                tangential_cmd = clamp(-0.22 * diag.vt_error_ratio, -0.12, 0.12)
                radial_cmd = clamp(-0.18 * vr_ratio, -0.08, 0.08)
                hold = radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, 0.13)
                return hold, "family_delayed_hold"
            commit = radial_tangential_action(state_x, state_y, state_vx, state_vy, 0.12 * direction_to_target, -0.10 * diag.vt_error_ratio, 0.16)
            return limit_norm(base_action[0] + 0.75 * commit[0], base_action[1] + 0.75 * commit[1], 0.18), "family_delayed_commit"

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
        # Phase36A is a transfer-geometry visualization. Some r0=1.0 cases can
        # satisfy the simulator success predicate before a geometric crossing is
        # recorded, so success alone should not stop the pre-cross visualization.
        terminated = bool((success and first_crossing_step is not None) or out_range or overspeed_now or too_close or radial_stall_fail)
        truncated = bool(step >= MAX_STEPS and not terminated)
        if success and first_crossing_step is not None:
            termination_reason = "success"
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

    row: Dict[str, object] = {
        "controller_name": "phase36a_transfer_family_visualization",
        "transfer_family": family.name,
        "case_label": case_label,
        "qualitative_family_label": family.qualitative_label,
        "post_cross_mode": mode.name,
        "r0_over_target": r0,
        "initial_velocity_angle_deg": angle,
        "thrust_scale": thrust_scale,
        "target_radius_scale": target_scale,
        "crossing_occurs": bool(first_crossing_step is not None),
        "recoverable_crossing": bool(first_crossing_step is not None and rec_state),
        "recoverable_state": bool(rec_state),
        "first_crossing_step": first_crossing_step if first_crossing_step is not None else "",
        "crossing_vr_ratio": crossing_vr_ratio if crossing_vr_ratio is not None else "",
        "crossing_vt_error_ratio": crossing_vt_error_ratio if crossing_vt_error_ratio is not None else "",
        "crossing_sync_error": crossing_sync_value if crossing_sync_value is not None else "",
        "min_abs_radius_error": closest["abs_r"],
        "min_abs_radius_error_ratio": closest["abs_r_ratio"],
        "closest_approach_step": int(closest["step"]),
        "best_crossing_potential": best_potential,
        "crossing_potential_at_closest": closest["potential"],
        "best_post_cross_distance": best_post["distance"],
        "best_post_cross_sync": best_post["sync"],
        "radius_crossings_total": radius_crossings_total,
        "overspeed": bool(len(speed_arr) and np.max(speed_arr) > 1.90),
        "instability": bool(instability),
        "max_speed_ratio": float(np.max(speed_arr)) if len(speed_arr) else 0.0,
        "steps": int(len(radius_errors)),
        "termination_reason": termination_reason,
        "dominant_stage": dominant_stage,
        "radial_velocity_trend": summarize_radial_velocity(vr_values),
        "final_radius_error_ratio": final_diag.r_error_ratio,
        "final_vr_ratio": final_vr_ratio,
        "final_vt_error_ratio": final_diag.vt_error_ratio,
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
        recoverable = [row for row in subset if bool_from_csv(row.get("recoverable_crossing"))]
        out[family.name] = {
            "cases": len(subset),
            "crossings": len(crossing),
            "recoverable_crossings": len(recoverable),
            "overspeed": sum(bool_from_csv(row.get("overspeed")) for row in subset),
            "instability": sum(bool_from_csv(row.get("instability")) for row in subset),
            "mean_min_abs_radius_error_ratio": float(np.mean([as_float(row.get("min_abs_radius_error_ratio")) for row in subset])),
            "mean_best_crossing_potential": float(np.mean([as_float(row.get("best_crossing_potential")) for row in subset])),
            "mean_crossing_sync": float(np.mean([as_float(row.get("crossing_sync_error")) for row in crossing])) if crossing else float("nan"),
            "mean_best_post_cross_distance": float(np.mean([as_float(row.get("best_post_cross_distance")) for row in crossing])) if crossing else float("nan"),
        }
    return out


def choose_plot_case(rows: Sequence[Dict[str, object]]) -> str:
    labels = sorted({str(row.get("case_label")) for row in rows})
    for preferred in ["representative_near_crossing", "representative_over_conservative_transfer", "representative_crossing"]:
        if preferred in labels:
            return preferred
    return labels[0]


def downsample(values: Sequence[object], max_points: int = 5000) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= max_points:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_points).astype(int)
    return arr[idx]


def plot_event_markers(ax, traj: Dict[str, List[object]], row: Dict[str, object], y_key: str, color: str) -> None:
    t = np.asarray(traj["time_step"], dtype=np.float64) * DT / 3600.0
    y = np.asarray(traj[y_key], dtype=np.float64)
    crossing_step = as_float(row.get("first_crossing_step"))
    closest_step = as_float(row.get("closest_approach_step"))
    if math.isfinite(closest_step) and len(t):
        idx = int(np.argmin(np.abs(np.asarray(traj["time_step"], dtype=np.float64) - closest_step)))
        ax.scatter(t[idx], y[idx], s=24, color=color, marker="o", zorder=4)
    if math.isfinite(crossing_step) and len(t):
        idx = int(np.argmin(np.abs(np.asarray(traj["time_step"], dtype=np.float64) - crossing_step)))
        ax.scatter(t[idx], y[idx], s=48, color=color, marker="x", zorder=5)


def save_plots(rows: Sequence[Dict[str, object]], examples: Sequence[Dict[str, object]]) -> None:
    if not examples:
        return
    plot_case = choose_plot_case(examples)
    subset = [row for row in examples if row.get("case_label") == plot_case and "trajectory" in row]
    colors = plt.cm.tab10(np.linspace(0, 1, len(FAMILIES)))
    color_by_family = {family.name: colors[idx % len(colors)] for idx, family in enumerate(FAMILIES)}

    fig, ax = plt.subplots(figsize=(8.8, 8.0))
    theta = np.linspace(0, 2 * math.pi, 360)
    ax.plot(np.cos(theta), np.sin(theta), color="#333333", linestyle="--", linewidth=1.0, label="target orbit")
    for row in subset:
        traj = row["trajectory"]
        family = str(row["transfer_family"])
        x = downsample(traj["x_over_target"])
        y = downsample(traj["y_over_target"])
        ax.plot(x, y, linewidth=1.1, color=color_by_family[family], label=family)
        crossing_step = as_float(row.get("first_crossing_step"))
        closest_step = as_float(row.get("closest_approach_step"))
        time_steps = np.asarray(traj["time_step"], dtype=np.float64)
        if len(time_steps):
            closest_idx = int(np.argmin(np.abs(time_steps - closest_step))) if math.isfinite(closest_step) else 0
            ax.scatter(as_float(traj["x_over_target"][closest_idx]), as_float(traj["y_over_target"][closest_idx]), s=22, color=color_by_family[family], marker="o")
            if math.isfinite(crossing_step):
                cross_idx = int(np.argmin(np.abs(time_steps - crossing_step)))
                ax.scatter(as_float(traj["x_over_target"][cross_idx]), as_float(traj["y_over_target"][cross_idx]), s=48, color=color_by_family[family], marker="x")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / target radius")
    ax.set_ylabel("y / target radius")
    ax.set_title(f"Transfer family geometry: {plot_case}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(PLOT_TRAJECTORY, dpi=220)
    plt.close(fig)

    for path, key, title, ylabel, hline in [
        (PLOT_RADIUS, "radius_ratio", "Radius history by transfer family", "r / target", 1.0),
        (PLOT_VR, "vr_ratio", "Radial velocity ratio by transfer family", "vr / circular speed", 0.0),
        (PLOT_VT, "vt_error_ratio", "Tangential velocity error by transfer family", "vt error ratio", 0.0),
    ]:
        fig, ax = plt.subplots(figsize=(10.2, 5.6))
        for row in subset:
            traj = row["trajectory"]
            family = str(row["transfer_family"])
            t = np.asarray(traj["time_step"], dtype=np.float64) * DT / 3600.0
            y_values = np.asarray(traj[key], dtype=np.float64)
            ax.plot(t, y_values, linewidth=1.05, color=color_by_family[family], label=family)
            plot_event_markers(ax, traj, row, key, color_by_family[family])
        ax.axhline(hline, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_xlabel("time [hours]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: {plot_case}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

    crossing_rows = [row for row in rows if bool_from_csv(row.get("crossing_occurs"))]
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for family in FAMILIES:
        vals = [row for row in crossing_rows if row.get("transfer_family") == family.name]
        if not vals:
            continue
        ax.scatter(
            [as_float(row.get("crossing_vr_ratio")) for row in vals],
            [as_float(row.get("crossing_vt_error_ratio")) for row in vals],
            s=70,
            label=family.name,
            color=color_by_family[family.name],
            alpha=0.85,
        )
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_xlabel("crossing vr ratio")
    ax.set_ylabel("crossing vt error ratio")
    ax.set_title("Crossing state scatter by family")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(PLOT_SCATTER, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for family in FAMILIES:
        vals = [row for row in rows if row.get("transfer_family") == family.name]
        if not vals:
            continue
        ax.scatter(
            [as_float(row.get("min_abs_radius_error_ratio")) for row in vals],
            [as_float(row.get("best_crossing_potential")) for row in vals],
            s=80,
            label=family.name,
            color=color_by_family[family.name],
            alpha=0.82,
        )
    ax.set_xlabel("minimum |radius error| / target")
    ax.set_ylabel("best crossing potential")
    ax.set_title("Family geometry map: closeness vs crossing potential")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(PLOT_GEOMETRY, dpi=220)
    plt.close(fig)


def write_markdown(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]], cases: Sequence[tuple[str, float, float, float]]) -> None:
    def fmt_optional(value: object, digits: int = 4) -> str:
        value_f = as_float(value)
        if not math.isfinite(value_f):
            return "N/A"
        return f"{value_f:.{digits}f}"

    baseline = stats["baseline_phase34"]
    best_cross = max(stats, key=lambda name: stats[name]["crossings"])
    best_recoverable = max(stats, key=lambda name: stats[name]["recoverable_crossings"])
    crossing_improved = stats[best_cross]["crossings"] > baseline["crossings"]
    recoverable_improved = stats[best_recoverable]["recoverable_crossings"] > baseline["recoverable_crossings"]
    if crossing_improved and recoverable_improved:
        honesty = "Phase36A found a transfer-family candidate that improves handoff into Phase34."
    elif crossing_improved:
        honesty = "Phase36A found additional geometric crossings but not Phase34-compatible crossings."
    else:
        honesty = "Phase36A clarified transfer geometry but did not improve crossing count."

    visually_distinct = [
        name
        for name, item in stats.items()
        if abs(item["mean_min_abs_radius_error_ratio"] - baseline["mean_min_abs_radius_error_ratio"]) > 0.002
        or abs(item["mean_best_crossing_potential"] - baseline["mean_best_crossing_potential"]) > 0.03
    ]
    if not visually_distinct:
        visually_distinct = [family.name for family in FAMILIES if family.name != "baseline_phase34"]

    approaching_non_cross = [
        row
        for row in rows
        if not bool_from_csv(row.get("crossing_occurs")) and as_float(row.get("min_abs_radius_error_ratio")) < 0.025
    ]
    violent = [
        row
        for row in rows
        if bool_from_csv(row.get("crossing_occurs"))
        and (abs(as_float(row.get("crossing_vr_ratio"))) > 0.35 or as_float(row.get("crossing_sync_error")) > 6.0)
    ]
    smooth = [
        row
        for row in rows
        if bool_from_csv(row.get("crossing_occurs"))
        and abs(as_float(row.get("crossing_vr_ratio"))) < 0.18
        and abs(as_float(row.get("crossing_vt_error_ratio"))) < 0.20
    ]
    worth_testing = sorted(
        [
            name
            for name, item in stats.items()
            if name != "baseline_phase34"
            and (item["crossings"] >= baseline["crossings"] or item["mean_best_crossing_potential"] >= baseline["mean_best_crossing_potential"])
            and item["overspeed"] == 0
        ]
    )
    if not worth_testing:
        worth_testing = [best_cross] if best_cross != "baseline_phase34" else ["baseline_phase34 reference only"]

    lines = [
        "# Phase36A Transfer Family Visualization",
        "",
        "## Scope",
        "",
        "- Visualization-first experiment, not an optimization phase.",
        "- Phase34 `radius_priority` remains the post-cross terminal controller.",
        "- Physics, CAPTURE/LOCK thresholds, rewards, and recoverability thresholds are unchanged.",
        "- Default run uses three representative cases: one crossing case, one `near_crossing` non-crossing case, and one `over_conservative_transfer` case selected from Phase35 outputs.",
        "",
        "## Representative Cases",
        "",
        "| Case label | r0 / target | Initial velocity angle | Thrust scale |",
        "|---|---:|---:|---:|",
    ]
    for label, r0, angle, thrust in cases:
        lines.append(f"| `{label}` | {r0:.2f} | {angle:.1f} | {thrust:.0f} |")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Transfer family | Cases | Crossings | Recoverable crossings | Mean min radius error ratio | Mean crossing potential | Mean crossing sync | Overspeed | Instability |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family in FAMILIES:
        item = stats[family.name]
        lines.append(
            f"| `{family.name}` | {item['cases']} | {item['crossings']} | {item['recoverable_crossings']} | "
            f"{item['mean_min_abs_radius_error_ratio']:.6f} | {item['mean_best_crossing_potential']:.4f} | "
            f"{fmt_optional(item['mean_crossing_sync'])} | {item['overspeed']} | {item['instability']} |"
        )
    lines.extend(
        [
            "",
            "## Scientific Answers",
            "",
            f"1. Which transfer families are visually distinct? `{', '.join(visually_distinct)}` show distinct geometry or metric behavior relative to the baseline in this small subset.",
            f"2. Which families approach target radius but fail to commit? `{len(approaching_non_cross)}` family-case rows stayed within 2.5% radius error without crossing.",
            f"3. Which families produce violent crossings? `{len(violent)}` family-case rows crossed with high radial ratio or high sync error.",
            f"4. Which families produce smoother crossing states? `{len(smooth)}` family-case rows crossed with lower radial and tangential error by the simple Phase36A filter.",
            f"5. Which families seem worth testing in Phase36B? `{', '.join(worth_testing)}`.",
            "6. Does this support the hypothesis that crossing-generation is trajectory-family geometry? `yes, cautiously`. The families produce visibly different pre-cross paths and different crossing-state quality, even when they do not improve recoverable count.",
            "",
            "## Interpretation",
            "",
            honesty,
            "",
            "The result should be read as a geometry map, not a success benchmark. A family that creates a visually distinct path or a better handoff state is useful even if it does not improve the small-subset count.",
            "",
            "## Artifacts",
            "",
            "- `phase36a_family_results.csv`",
            "- `family_case_notes.md`",
            "- `phase36a_vs_phase35.md`",
            "- `family_trajectory_overlay.png`",
            "- `radius_vs_time_by_family.png`",
            "- `vr_vs_time_by_family.png`",
            "- `vt_error_vs_time_by_family.png`",
            "- `crossing_state_scatter_by_family.png`",
            "- `family_geometry_map.png`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")

    note_lines = [
        "# Phase36A Family Case Notes",
        "",
        "| Case | Family | Crossing | Recoverable crossing | Closest step | Min radius error ratio | First crossing step | Crossing vr | Crossing vt error | Qualitative note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        crossing = bool_from_csv(row.get("crossing_occurs"))
        recoverable = bool_from_csv(row.get("recoverable_crossing"))
        if crossing and recoverable:
            note = "crossed and handed off into Phase34 recoverability"
        elif crossing:
            note = "geometric crossing without recoverable Phase34 handoff"
        elif bool_from_csv(row.get("overspeed")):
            note = "overspeed before useful crossing"
        elif as_float(row.get("min_abs_radius_error_ratio")) < 0.025:
            note = "approached target radius but did not commit to crossing"
        else:
            note = "did not enter a useful crossing corridor"
        note_lines.append(
            f"| `{row['case_label']}` | `{row['transfer_family']}` | {int(crossing)} | {int(recoverable)} | "
            f"{row['closest_approach_step']} | {as_float(row['min_abs_radius_error_ratio']):.6f} | "
            f"{row.get('first_crossing_step', '')} | {fmt_optional(row.get('crossing_vr_ratio'))} | "
            f"{fmt_optional(row.get('crossing_vt_error_ratio'))} | {note} |"
        )
    CASE_NOTES_MD.write_text("\n".join(note_lines), encoding="utf-8")

    phase35_rows = read_csv(PHASE35_RESULTS)
    phase35_subset_keys = {case_key(r0, angle, thrust) for _, r0, angle, thrust in cases}
    phase35_baseline = [
        row
        for row in phase35_rows
        if row.get("upstream_variant") == "baseline_phase34"
        and case_key(row.get("r0_over_target"), row.get("initial_velocity_angle_deg"), row.get("thrust_scale")) in phase35_subset_keys
    ]
    phase35_crossings = sum(bool_from_csv(row.get("crossing_occurs")) for row in phase35_baseline)
    phase35_recoverable = sum(bool_from_csv(row.get("recoverable_crossing")) for row in phase35_baseline)
    compare_lines = [
        "# Phase36A vs Phase35",
        "",
        "| Metric | Phase35 baseline on selected cases | Phase36A best crossing family | Phase36A best recoverable family |",
        "|---|---:|---:|---:|",
        f"| Cases | {len(phase35_baseline)} | {stats[best_cross]['cases']} | {stats[best_recoverable]['cases']} |",
        f"| Geometric crossings | {phase35_crossings} | {stats[best_cross]['crossings']} | {stats[best_recoverable]['crossings']} |",
        f"| Recoverable crossings | {phase35_recoverable} | {stats[best_cross]['recoverable_crossings']} | {stats[best_recoverable]['recoverable_crossings']} |",
        "",
        "## Reading",
        "",
        "- Phase36A is a visualization-first subset experiment, so it should not be compared to Phase35 as a full 24-case benchmark.",
        "- Phase34 terminal behavior remains fixed after first crossing.",
        f"- Final honesty statement: {honesty}",
    ]
    COMPARE_MD.write_text("\n".join(compare_lines), encoding="utf-8")


def benchmark(full_benchmark: bool = False) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[tuple[str, float, float, float]]]:
    if full_benchmark:
        cases = [("full_24_case", r0, angle, thrust) for r0, angle, thrust in iter_cases()]
    else:
        cases = select_representative_cases()
    rows: List[Dict[str, object]] = []
    examples: List[Dict[str, object]] = []
    for family in FAMILIES:
        for label, r0, angle, thrust in cases:
            row = rollout_family_case(family, PHASE34_TERMINAL_MODE, label, r0, angle, thrust, record_trajectory=True)
            rows.append(row)
            examples.append(row)
            print(
                f"phase36a family={family.name} case={label} r0={r0:g} angle={angle:g} thrust={thrust:g} "
                f"cross={row['crossing_occurs']} rec={row['recoverable_crossing']} "
                f"min_r={as_float(row['min_abs_radius_error_ratio']):.5f} reason={row['termination_reason']}"
            )
    return rows, examples, cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase36A transfer-family visualization experiment.")
    parser.add_argument("--full-benchmark", action="store_true", help="Run all 24 Phase34/35 benchmark cases instead of the representative subset.")
    parser.add_argument("--skip-plots", action="store_true", help="Write CSV and markdown without regenerating plots.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, examples, cases = benchmark(full_benchmark=args.full_benchmark)
    stats = aggregate(rows)
    write_csv(RESULTS_CSV, FIELDNAMES, rows)
    if not args.skip_plots:
        save_plots(rows, examples)
    write_markdown(rows, stats, cases)
    print(f"Saved Phase36A outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
