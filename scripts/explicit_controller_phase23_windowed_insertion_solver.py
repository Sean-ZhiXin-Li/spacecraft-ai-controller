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
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase23_windowed_insertion_solver"
MPLCONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig
from scripts.explicit_controller_phase21_orbital_transfer_planner import (
    ANGULAR_MOMENTUM_TOL,
    BASE_PARAMS,
    DEFAULT_TARGET_RADIUS,
    DT,
    G,
    M,
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
RESULTS_CSV = OUTPUT_DIR / "phase23_results.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
TRAJECTORY_DIR = OUTPUT_DIR / "trajectories"
STAGE_USAGE_PNG = OUTPUT_DIR / "transfer_stage_usage.png"
INSERTION_WINDOW_PNG = OUTPUT_DIR / "insertion_window_success.png"
ENERGY_ERROR_PNG = OUTPUT_DIR / "energy_error_vs_time.png"
ANGULAR_MOMENTUM_ERROR_PNG = OUTPUT_DIR / "angular_momentum_error_vs_time.png"

BURN_A_MAX_STEPS = 24
BURN_A_MIN_STEPS = 10
BURN_A_MAX_NORM = 0.11
COAST_MIN_STEPS = 80
COAST_MAX_STEPS = 42000
BURN_B_MAX_STEPS = 46
BURN_B_MAX_NORM = 0.42
SOLVER_HOLD_STEPS = 10
SOLVER_HORIZON_STEPS = 90
INSERTION_BAND_RATIO = 2.5e-2
HANDOFF_R_RATIO = 7.5e-4
HANDOFF_VR_RATIO = 2.8e-2
HANDOFF_VT_RATIO = 2.8e-1
SAFE_SPEED_RATIO = 1.65


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
]


@dataclass(frozen=True)
class ControllerSpec:
    name: str


@dataclass(frozen=True)
class SolverCandidate:
    name: str
    radial_cmd: float
    tangential_cmd: float
    max_norm: float


def set_seed(seed: int = 7) -> None:
    np.random.seed(seed)


def iter_cases(controllers: Sequence[ControllerSpec]) -> Iterable[tuple[ControllerSpec, float, float, float, float]]:
    for controller in controllers:
        for thrust in THRUST_VALUES:
            for angle in ANGLE_VALUES:
                for r0 in R0_VALUES:
                    yield controller, r0, angle, thrust, TARGET_RADIUS_SCALE


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
        ratio0 = float(out.get("initial_angular_momentum_ratio") or 0.0)
        ratiof = float(out.get("final_angular_momentum_ratio") or 0.0)
        out["initial_angular_momentum_error_ratio"] = ratio0 - 1.0 if ratio0 else ""
        out["final_angular_momentum_error_ratio"] = ratiof - 1.0 if ratiof else ""
    out.setdefault("mean_energy_error_ratio", "")
    out.setdefault("mean_angular_momentum_error_ratio", "")
    out.setdefault("burn_a_steps", 0)
    out.setdefault("burn_b_steps", 0)
    out.setdefault("coast_arc_steps", out.get("coast_steps", 0))
    out.setdefault("safe_coast_steps", out.get("safety_coast_steps", 0))
    out.setdefault("burn_a_count", 0)
    out.setdefault("burn_b_count", 0)
    out.setdefault("insertion_attempts", 0)
    out.setdefault("insertion_windows", 0)
    out.setdefault("successful_insertion_windows", 0)
    out.setdefault("handoff_ready_count", 0)
    out.setdefault("handoff_entered", bool_from_csv(out.get("capture_entered")))
    out.setdefault("handoff_rate", 0.0)
    out.setdefault("predicted_crossing_step", "")
    out.setdefault("burn_b_solver_evals", 0)
    out.setdefault("burn_b_solver_score", "")
    out.setdefault("insertion_window_conversion_rate", 0.0)
    return out


def basis(
    state_x: float,
    state_y: float,
    state_vx: float,
    state_vy: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    radius = math.sqrt(state_x * state_x + state_y * state_y)
    inv_r = 1.0 / (radius + 1.0e-12)
    r_hat_x = state_x * inv_r
    r_hat_y = state_y * inv_r
    t_hat_x = -r_hat_y
    t_hat_y = r_hat_x
    radial_velocity = state_vx * r_hat_x + state_vy * r_hat_y
    tangential_velocity = state_vx * t_hat_x + state_vy * t_hat_y
    speed = math.sqrt(state_vx * state_vx + state_vy * state_vy)
    return r_hat_x, r_hat_y, t_hat_x, t_hat_y, radius, radial_velocity, tangential_velocity, speed


def estimate_crossing_steps(radius_error: float, radial_velocity: float) -> int | None:
    if radius_error * radial_velocity >= 0.0 or abs(radial_velocity) < 1.0e-9:
        return None
    steps = abs(radius_error) / (abs(radial_velocity) * DT + 1.0e-12)
    if not math.isfinite(steps) or steps < 0.0:
        return None
    return int(round(steps))


def classify_intercept(diag, target_radius: float, coast_steps: int) -> str:
    v_circ = math.sqrt(MU / target_radius)
    if abs(diag.r_error_ratio) <= HANDOFF_R_RATIO and abs(diag.radial_velocity) / (v_circ + 1.0e-12) <= HANDOFF_VR_RATIO and abs(diag.vt_error_ratio) <= HANDOFF_VT_RATIO:
        return "handoff_ready"
    if abs(diag.r_error_ratio) <= INSERTION_BAND_RATIO and diag.r_error_ratio * diag.radial_velocity <= 0.0:
        return "insertion_window"
    if not orbit_crosses_target(diag, target_radius):
        return "no_intercept_possible"
    if abs(diag.vt_error_ratio) <= 0.35 and abs(diag.radial_velocity) / (v_circ + 1.0e-12) <= 0.08 and coast_steps >= COAST_MIN_STEPS:
        return "intercept_possible_good_alignment"
    return "intercept_possible_poor_alignment"


def rollout_phase23_case(
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
    burn_b_solver_evals = 0
    burn_b_solver_scores: List[float] = []
    stage_step = 0
    burn_a_target_steps = int(clamp(BURN_A_MIN_STEPS + 420.0 * abs(r0 - 1.0), BURN_A_MIN_STEPS, BURN_A_MAX_STEPS))

    trajectory: Dict[str, List[float | str]] = {
        "time_step": [],
        "radius": [],
        "x": [],
        "y": [],
        "energy_error_ratio": [],
        "angular_momentum_error_ratio": [],
        "radial_velocity_ratio": [],
        "vt_error_ratio": [],
        "stage": [],
        "intercept_class": [],
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
        state_radius: float,
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

    def burn_b_action(state_x: float, state_y: float, state_vx: float, state_vy: float, diag) -> tuple[float, float]:
        nonlocal burn_b_solver_evals

        vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
        base_radial = clamp(-3.0 * vr_ratio - 0.35 * diag.r_error_ratio, -0.36, 0.36)
        base_tangential = clamp(-1.8 * diag.vt_error_ratio, -0.36, 0.36)

        candidates = [
            SolverCandidate("proportional", base_radial, base_tangential, BURN_B_MAX_NORM),
            SolverCandidate("radial_strong", clamp(1.35 * base_radial, -0.42, 0.42), base_tangential, BURN_B_MAX_NORM),
            SolverCandidate("tangential_strong", base_radial, clamp(1.35 * base_tangential, -0.42, 0.42), BURN_B_MAX_NORM),
            SolverCandidate("radial_first", clamp(-4.2 * vr_ratio, -0.42, 0.42), 0.20 * base_tangential, 0.36),
            SolverCandidate("tangent_first", 0.35 * base_radial, clamp(-2.6 * diag.vt_error_ratio, -0.42, 0.42), 0.36),
            SolverCandidate("soft_blend", 0.65 * base_radial, 0.65 * base_tangential, 0.28),
            SolverCandidate("coast", 0.0, 0.0, 0.0),
        ]

        def basin_score(test_diag, speed_ratio: float) -> float:
            test_vr_ratio = test_diag.radial_velocity / (v_circ + 1.0e-12)
            score = abs(test_diag.r_error_ratio) + 0.90 * abs(test_vr_ratio) + 0.70 * abs(test_diag.vt_error_ratio)
            score += 0.12 * abs(test_diag.energy_error_ratio) + 0.10 * abs(test_diag.angular_momentum_error_ratio)
            if abs(test_diag.r_error_ratio) < RECOVERABLE_R_RATIO:
                score -= 0.10
            if abs(test_vr_ratio) < RECOVERABLE_VR_RATIO:
                score -= 0.08
            if abs(test_diag.vt_error_ratio) < RECOVERABLE_VT_RATIO:
                score -= 0.06
            if speed_ratio > SAFE_SPEED_RATIO:
                score += 8.0 * (speed_ratio - SAFE_SPEED_RATIO)
            return score

        def evaluate(candidate: SolverCandidate) -> tuple[float, tuple[float, float]]:
            candidate_x, candidate_y = radial_tangential_action(
                state_x,
                state_y,
                state_vx,
                state_vy,
                candidate.radial_cmd,
                candidate.tangential_cmd,
                candidate.max_norm,
            )
            sim_x, sim_y, sim_vx, sim_vy = state_x, state_y, state_vx, state_vy
            best_score = float("inf")
            for horizon_step in range(SOLVER_HORIZON_STEPS):
                hold = horizon_step < SOLVER_HOLD_STEPS and candidate.max_norm > 1.0e-12
                ax = candidate_x if hold else 0.0
                ay = candidate_y if hold else 0.0
                sim_x, sim_y, sim_vx, sim_vy = env_step(sim_x, sim_y, sim_vx, sim_vy, ax, ay)
                sim_diag = orbital_diagnostics(sim_x, sim_y, sim_vx, sim_vy, target_radius)
                sim_speed_ratio = sim_diag.speed / (v_circ + 1.0e-12)
                if sim_diag.radius < 0.40 * target_radius or sim_diag.radius > 2.35 * target_radius:
                    return float("inf"), (candidate_x, candidate_y)
                best_score = min(best_score, basin_score(sim_diag, sim_speed_ratio))
            return best_score, (candidate_x, candidate_y)

        best_score = float("inf")
        best_action = (0.0, 0.0)
        for candidate in candidates:
            score, action = evaluate(candidate)
            burn_b_solver_evals += 1
            if score < best_score:
                best_score = score
                best_action = action
        burn_b_solver_scores.append(best_score)
        return best_action

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
            action_x, action_y = capture_lock_action(x, y, vx, vy, diag.radius, phase)
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
            trajectory["time_step"].append(float(step))
            trajectory["radius"].append(next_diag.radius)
            trajectory["x"].append(x)
            trajectory["y"].append(y)
            trajectory["energy_error_ratio"].append(next_diag.energy_error_ratio)
            trajectory["angular_momentum_error_ratio"].append(next_diag.angular_momentum_error_ratio)
            trajectory["radial_velocity_ratio"].append(next_diag.radial_velocity / (v_circ + 1.0e-12))
            trajectory["vt_error_ratio"].append(next_diag.vt_error_ratio)
            trajectory["stage"].append(active_stage)
            trajectory["intercept_class"].append(intercept_class)

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
    row: Dict[str, object] = {
        "controller_name": "phase23_windowed_insertion_solver",
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
        "first_crossing_step": first_crossing_step,
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
        "burn_b_solver_evals": int(burn_b_solver_evals),
        "burn_b_solver_score": float(np.min(burn_b_solver_scores)) if burn_b_solver_scores else "",
        "insertion_window_conversion_rate": float(1.0 if recoverable_crossing and insertion_windows > 0 else 0.0),
    }
    if record_trajectory:
        row["trajectory"] = trajectory
        row["target_radius"] = target_radius
    return row


def run_phase23_rows() -> List[Dict[str, object]]:
    controller = ControllerSpec("phase23_windowed_insertion_solver")
    rows: List[Dict[str, object]] = []
    cases = list(iter_cases([controller]))
    for idx, (_controller, r0, angle, thrust, target_scale) in enumerate(cases, start=1):
        row = rollout_phase23_case(r0, angle, thrust, target_scale)
        rows.append(row)
        print(
            f"phase23 {idx}/{len(cases)} r0={r0:g} angle={angle:g} thrust={thrust:g} "
            f"capture={row['capture_entered']} success={row['success']} crossings={row['radius_crossings_total']} "
            f"recoverable={row['recoverable_crossing']} stage={row['dominant_transfer_stage']} "
            f"ins={row['insertion_windows']} burnA={row['burn_a_steps']} burnB={row['burn_b_steps']} "
            f"reason={row['termination_reason']}"
        )
    return rows


def write_results(rows: Sequence[Dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def numeric_values(rows: Sequence[Dict[str, object]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key, "")
        if value in {"", None}:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def aggregate(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for name in sorted({str(row["controller_name"]) for row in rows}):
        subset = [row for row in rows if str(row["controller_name"]) == name]
        energy_errors = numeric_values(subset, "final_energy_error_ratio")
        h_errors = numeric_values(subset, "final_angular_momentum_error_ratio")
        stats[name] = {
            "total": len(subset),
            "crossing_cases": sum(bool_from_csv(row.get("crossing_occurs")) for row in subset),
            "recoverable_crossings": sum(bool_from_csv(row.get("recoverable_crossing")) for row in subset),
            "capture": sum(bool_from_csv(row.get("capture_entered")) for row in subset),
            "success": sum(bool_from_csv(row.get("success")) for row in subset),
            "overspeed": sum(bool_from_csv(row.get("overspeed")) or str(row.get("termination_reason")) == "overspeed" for row in subset),
            "mean_energy_error": float(np.mean([abs(value) for value in energy_errors])) if energy_errors else float("nan"),
            "mean_angular_momentum_error": float(np.mean([abs(value) for value in h_errors])) if h_errors else float("nan"),
            "insertion_attempts": sum(int(float(row.get("insertion_attempts") or 0)) for row in subset),
            "insertion_windows": sum(int(float(row.get("insertion_windows") or 0)) for row in subset),
            "successful_insertion_windows": sum(int(float(row.get("successful_insertion_windows") or 0)) for row in subset),
            "burn_a_steps": sum(int(float(row.get("burn_a_steps") or 0)) for row in subset),
            "burn_b_steps": sum(int(float(row.get("burn_b_steps") or 0)) for row in subset),
            "handoff_entered": sum(bool_from_csv(row.get("handoff_entered")) for row in subset),
            "burn_b_solver_evals": sum(int(float(row.get("burn_b_solver_evals") or 0)) for row in subset),
            "insertion_window_conversion_rate": (
                sum(bool_from_csv(row.get("recoverable_crossing")) for row in subset)
                / max(1, sum(int(float(row.get("insertion_windows") or 0)) for row in subset))
            ),
        }
    return stats


def phase23_stage_counts(rows: Sequence[Dict[str, object]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if str(row.get("controller_name")) != "phase23_windowed_insertion_solver":
            continue
        payload = json.loads(str(row.get("transfer_stage_counts_json") or "{}"))
        for key, value in payload.items():
            counts[key] += int(value)
    return counts


def save_stage_usage_plot(rows: Sequence[Dict[str, object]]) -> None:
    counts = phase23_stage_counts(rows)
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#9D755D"][: len(labels)])
    ax.set_ylabel("DESCENT steps")
    ax.set_title("Phase 23 transfer stage usage")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(STAGE_USAGE_PNG, dpi=220)
    plt.close(fig)


def save_insertion_window_plot(stats: Dict[str, Dict[str, object]]) -> None:
    phase23 = stats["phase23_windowed_insertion_solver"]
    labels = ["attempts", "windows", "successful windows", "handoffs", "CAPTURE", "success"]
    values = [
        phase23["insertion_attempts"],
        phase23["insertion_windows"],
        phase23["successful_insertion_windows"],
        phase23["handoff_entered"],
        phase23["capture"],
        phase23["success"],
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(labels, values, color=["#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#E45756", "#B279A2"])
    ax.set_title("Phase 23 insertion-window outcomes")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(INSERTION_WINDOW_PNG, dpi=220)
    plt.close(fig)


def representative_cases() -> List[tuple[float, float, float, str]]:
    return [
        (1.00, 175.0, 10000.0, "successful"),
        (1.02, 170.0, 10000.0, "failed_outer"),
        (0.98, 165.0, 10000.0, "failed_inner"),
        (1.05, 150.0, 12000.0, "failed_far_outer"),
    ]


def save_diagnostic_plots() -> None:
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    for old_plot in TRAJECTORY_DIR.glob("phase23_representative_*.png"):
        old_plot.unlink()
    trace_rows = [rollout_phase23_case(r0, angle, thrust, TARGET_RADIUS_SCALE, record_trajectory=True) for r0, angle, thrust, _label in representative_cases()]
    fig_e, ax_e = plt.subplots(figsize=(9.4, 5.2))
    fig_h, ax_h = plt.subplots(figsize=(9.4, 5.2))
    for idx, row in enumerate(trace_rows, start=1):
        traj = row["trajectory"]
        steps = np.asarray(traj["time_step"], dtype=np.float64)
        time_days = steps * DT / 86400.0
        energy_error = np.asarray(traj["energy_error_ratio"], dtype=np.float64)
        h_error = np.asarray(traj["angular_momentum_error_ratio"], dtype=np.float64)
        radius = np.asarray(traj["radius"], dtype=np.float64)
        vr = np.asarray(traj["radial_velocity_ratio"], dtype=np.float64)
        vt = np.asarray(traj["vt_error_ratio"], dtype=np.float64)
        target_radius = float(row["target_radius"])
        label = f"r0={row['r0_over_target']:g}, a={row['initial_velocity_angle_deg']:g}, T={row['thrust_scale']:g}"
        ax_e.plot(time_days, np.abs(energy_error), linewidth=1.1, label=label)
        ax_h.plot(time_days, np.abs(h_error), linewidth=1.1, label=label)

        fig, axes = plt.subplots(4, 1, figsize=(9.2, 8.4), sharex=True)
        axes[0].plot(time_days, radius / target_radius, color="#4C78A8", linewidth=1.0)
        axes[0].axhline(1.0, color="#333333", linestyle="--", linewidth=0.9)
        axes[0].set_ylabel("r / target")
        axes[1].plot(time_days, np.abs(energy_error), color="#54A24B", linewidth=1.0)
        axes[1].set_ylabel("|E err|")
        axes[2].plot(time_days, np.abs(h_error), color="#F58518", linewidth=1.0)
        axes[2].set_ylabel("|h err|")
        axes[3].plot(time_days, vr, color="#E45756", linewidth=0.9, label="v_r")
        axes[3].plot(time_days, vt, color="#72B7B2", linewidth=0.9, label="v_t err")
        axes[3].axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
        axes[3].set_ylabel("velocity ratio")
        axes[3].set_xlabel("time [days]")
        axes[3].legend(fontsize=8)
        for ax in axes:
            ax.grid(True, alpha=0.25)
        outcome = "success" if bool_from_csv(row["success"]) else "failed"
        fig.suptitle(f"Phase 23 representative {outcome} transfer {idx}: {label}")
        fig.tight_layout()
        fig.savefig(TRAJECTORY_DIR / f"phase23_representative_{outcome}_{idx:02d}.png", dpi=220)
        plt.close(fig)

    for ax, title, ylabel in [
        (ax_e, "Phase 23 energy error vs time", "|E - E_target| / |E_target|"),
        (ax_h, "Phase 23 angular momentum error vs time", "|h - h_target| / |h_target|"),
    ]:
        ax.set_xlabel("time [days]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig_e.tight_layout()
    fig_h.tight_layout()
    fig_e.savefig(ENERGY_ERROR_PNG, dpi=220)
    fig_h.savefig(ANGULAR_MOMENTUM_ERROR_PNG, dpi=220)
    plt.close(fig_e)
    plt.close(fig_h)


def write_summary(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]]) -> None:
    baseline = stats["baseline_soft_linear_3e4"]
    phase20 = stats["phase20_predictive_planner"]
    phase21 = stats["phase21_orbital_transfer_planner"]
    phase22 = stats["phase22_two_burn_transfer"]
    phase23 = stats["phase23_windowed_insertion_solver"]
    crossing_improved = int(phase23["crossing_cases"]) > int(baseline["crossing_cases"])
    recoverable_improved = int(phase23["recoverable_crossings"]) > int(phase22["recoverable_crossings"])
    capture_improved = int(phase23["capture"]) > int(baseline["capture"])
    success_improved = int(phase23["success"]) > int(baseline["success"])
    bottleneck = "windowed Burn B insertion geometry"
    if int(phase23["recoverable_crossings"]) > 0 and not capture_improved:
        bottleneck = "CAPTURE architecture"
    if int(phase23["insertion_windows"]) == 0 and not crossing_improved:
        bottleneck = "transfer timing before Burn B"
    mode_counts = phase23_stage_counts(rows)
    lines = [
        "# Phase 23 Windowed Orbital Insertion Solver",
        "",
        "## Scope",
        "",
        "- Python-only 2D staged transfer controller.",
        "- Burn A and coast arc are kept from Phase 22.",
        "- Burn B is upgraded to a deterministic windowed insertion solver.",
        "- Physics, reward, PPO, and CAPTURE/LOCK logic are unchanged.",
        "",
        "## Reduced-Grid Result",
        "",
        "| Controller | Recoverable crossings | CAPTURE | Success | Insertion windows | Conversion rate | Overspeed |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Phase 7.6 `baseline_soft_linear_3e4` | {baseline['recoverable_crossings']} | {baseline['capture']} | {baseline['success']} | {baseline['insertion_windows']} | {baseline['insertion_window_conversion_rate']:.3f} | {baseline['overspeed']} |",
        f"| Phase 20 `phase20_predictive_planner` | {phase20['recoverable_crossings']} | {phase20['capture']} | {phase20['success']} | {phase20['insertion_windows']} | {phase20['insertion_window_conversion_rate']:.3f} | {phase20['overspeed']} |",
        f"| Phase 21 `phase21_orbital_transfer_planner` | {phase21['recoverable_crossings']} | {phase21['capture']} | {phase21['success']} | {phase21['insertion_windows']} | {phase21['insertion_window_conversion_rate']:.3f} | {phase21['overspeed']} |",
        f"| Phase 22 `phase22_two_burn_transfer` | {phase22['recoverable_crossings']} | {phase22['capture']} | {phase22['success']} | {phase22['insertion_windows']} | {phase22['insertion_window_conversion_rate']:.3f} | {phase22['overspeed']} |",
        f"| Phase 23 `phase23_windowed_insertion_solver` | {phase23['recoverable_crossings']} | {phase23['capture']} | {phase23['success']} | {phase23['insertion_windows']} | {phase23['insertion_window_conversion_rate']:.3f} | {phase23['overspeed']} |",
        "",
        "## Phase 23 Burn B Solver Diagnostics",
        "",
        f"- Burn A steps: `{phase23['burn_a_steps']}`.",
        f"- Burn B steps: `{phase23['burn_b_steps']}`.",
        f"- Burn B solver evaluations: `{phase23['burn_b_solver_evals']}`.",
        f"- Insertion attempts: `{phase23['insertion_attempts']}`.",
        f"- Insertion windows: `{phase23['insertion_windows']}`.",
        f"- Successful insertion windows: `{phase23['successful_insertion_windows']}`.",
        f"- Handoff-entered cases: `{phase23['handoff_entered']}`.",
        f"- Insertion window conversion rate: `{phase23['insertion_window_conversion_rate']:.3f}`.",
        f"- Stage usage: `{json.dumps(dict(mode_counts), sort_keys=True)}`.",
        f"- Mean final energy error: `{phase23['mean_energy_error']:.4f}`.",
        f"- Mean final angular momentum error: `{phase23['mean_angular_momentum_error']:.4f}`.",
        "",
        "## Research Answers",
        "",
        f"1. Can the windowed solver convert Phase 22 insertion windows into recoverable crossings? `{'yes' if int(phase23['recoverable_crossings']) > int(phase22['recoverable_crossings']) else 'no'}`.",
        f"2. Does Phase 23 improve CAPTURE? `{'yes' if capture_improved else 'no'}`. CAPTURE changed from `{baseline['capture']}` to `{phase23['capture']}`.",
        f"3. Does Phase 23 improve success? `{'yes' if success_improved else 'no'}`. Success changed from `{baseline['success']}` to `{phase23['success']}`.",
        f"4. Current bottleneck: `{bottleneck}`.",
        "",
        "## Success Criteria",
        "",
        f"- Minimum, crossings > Phase 7.6: `{'met' if crossing_improved else 'not met'}`.",
        f"- Moderate, recoverable crossings > 0: `{'met' if int(phase23['recoverable_crossings']) > 0 else 'not met'}`.",
        f"- Strong, CAPTURE > Phase 7.6: `{'met' if capture_improved else 'not met'}`.",
        f"- Major, success > Phase 7.6: `{'met' if success_improved else 'not met'}`.",
        f"- Research success, insertion window conversion rate > Phase 22: `{'met' if phase23['insertion_window_conversion_rate'] > phase22['insertion_window_conversion_rate'] else 'not met'}`.",
        "",
        "## Honesty Note",
        "",
        "- This script does not force a positive insertion result.",
        "- Burn A and coast arc are inherited from Phase 22; only Burn B is changed.",
        "- Overspeed, failed windows, and no-gain outcomes are preserved in the CSV and summary.",
        "",
        "## Artifacts",
        "",
        "- `phase23_results.csv`",
        "- `transfer_stage_usage.png`",
        "- `insertion_window_success.png`",
        "- `energy_error_vs_time.png`",
        "- `angular_momentum_error_vs_time.png`",
        "- `trajectories/phase23_representative_*.png`",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 23 windowed insertion solver benchmark.")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    set_seed(7)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_rows = []
    reference_rows.extend(load_csv_rows(PHASE20_CSV, {"baseline_soft_linear_3e4", "phase20_predictive_planner"}))
    reference_rows.extend(load_csv_rows(PHASE21_CSV, {"phase21_orbital_transfer_planner"}))
    reference_rows.extend(load_csv_rows(PHASE22_CSV, {"phase22_two_burn_transfer"}))
    phase23_rows = run_phase23_rows()
    rows = reference_rows + phase23_rows
    write_results(rows)
    stats = aggregate(rows)
    if not args.skip_plots:
        save_stage_usage_plot(rows)
        save_insertion_window_plot(stats)
        save_diagnostic_plots()
    write_summary(rows, stats)
    print(f"Saved Phase 23 windowed insertion solver outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
