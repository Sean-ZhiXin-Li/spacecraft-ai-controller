from __future__ import annotations

import csv
import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase20_5_ablation"
MPLCONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig


COMPARISON_CSV = OUTPUT_DIR / "ablation_results.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
TRAJECTORY_DIR = OUTPUT_DIR / "trajectories"
HORIZON_SWEEP_PNG = OUTPUT_DIR / "horizon_sweep.png"
REPLAN_INTERVAL_SWEEP_PNG = OUTPUT_DIR / "replan_interval_sweep.png"
ACTION_DISTRIBUTION_PNG = OUTPUT_DIR / "action_distribution.png"
SCORE_ABLATION_PNG = OUTPUT_DIR / "score_ablation_comparison.png"
CROSSING_STATE_HIST_PNG = OUTPUT_DIR / "crossing_state_histogram.png"
VR_NEAR_CROSSING_PNG = OUTPUT_DIR / "vr_vs_time_near_crossing.png"
VT_ERROR_PNG = OUTPUT_DIR / "vt_error_vs_time.png"
CAPTURE_CROSSING_PNG = OUTPUT_DIR / "capture_crossing_comparison.png"

SEED = 7
DT = 100.0
MAX_STEPS = 100000
TAIL_LEN = 5000
STRICT_CFG = {
    "tol_r": 1.0e-3,
    "tol_v": 1.0e-3,
    "tol_ang": 0.02,
    "success_threshold": 200,
}
DEFAULT_TARGET_RADIUS = 7.5e12

R0_VALUES = [0.98, 1.00, 1.02, 1.05]
ANGLE_VALUES = [150.0, 165.0, 170.0, 175.0]
THRUST_VALUES = [8000.0, 10000.0, 12000.0]
TARGET_RADIUS_SCALE = 1.00

G = 6.67430e-11
M = 1.989e30
MU = G * M
MASS = 722.0
NEAR_MISS_REL = 3.0e-5
WS_BAND_RATIO = 8.0e-5
PREWINDOW_OUTER_RATIO = 5.0e-4

BASE_PARAMS: Dict[str, float] = {
    "near_band_ratio": WS_BAND_RATIO,
    "inward_bias": 0.12,
    "score_r": 1.0,
    "score_vr": 0.65,
    "score_vt": 0.45,
    "score_energy": 0.30,
    "capture_bonus": 0.20,
    "inner_capture_bonus": 0.10,
    "min_scale": 0.65,
    "max_scale": 0.85,
    "prewindow_radial_gain": 0.08,
}

FIELDNAMES = [
    "variant_name",
    "sweep",
    "predictive_horizon",
    "replan_interval",
    "score_mode",
    "action_space",
    "burn_hold_steps",
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
    "dominant_planner_mode",
    "planner_mode_counts_json",
    "initial_energy_error_ratio",
    "final_energy_error_ratio",
    "initial_angular_momentum_ratio",
    "final_angular_momentum_ratio",
    "max_speed_ratio",
    "overspeed",
    "turning_point_count",
    "mean_abs_tracking_error",
    "initial_transfer_energy_error_ratio",
    "final_transfer_energy_error_ratio",
    "initial_transfer_angular_momentum_error_ratio",
    "final_transfer_angular_momentum_error_ratio",
    "burn_steps",
    "coast_steps",
    "burn_ratio",
    "event_count",
    "burn_count",
    "predictive_replans",
    "candidate_eval_count",
    "runtime_seconds",
    "planner_choice_counts_json",
    "planner_choice_named_counts_json",
    "coast_chosen_pct",
    "prograde_pct",
    "retrograde_pct",
    "radial_inward_pct",
    "radial_outward_pct",
    "diagonal_pct",
    "avg_chosen_thrust_norm",
    "avg_score_margin",
    "event_steps_json",
    "burn_steps_json",
]

PLANNER_MODES = [
    "predictive_descent",
    "phase76_handoff",
    "injection",
    "burn",
    "coast",
    "event_burn",
    "oscillation",
    "trajectory_tracking",
    "elliptical_guidance",
    "targeting",
    "targeting_close",
    "soft_window",
]
MAX_BURN_STEPS = 30
COOLDOWN_STEPS = 20
EVENT_BURN_STEPS = 10
TURNING_VR_THRESHOLD = 2.5e-3
NEAR_MISS_WINDOW_RATIO = 8.0e-3
TRAJECTORY_PERIOD_STEPS = 9000000.0
TRAJECTORY_DECAY_PER_STEP = 1.8e-5
REPLAN_INTERVAL = 40
PREDICTIVE_HORIZON = 260
POST_CROSS_WINDOW = 80
RECOVERABLE_VR_RATIO = 2.0e-2
RECOVERABLE_VT_RATIO = 2.5e-1
RECOVERABLE_R_RATIO = 2.5e-3
INJECTION_NORM = 0.10
INJECTION_MIN_STEPS = 20
INJECTION_MAX_STEPS = 80
FORCED_COAST_STEPS = 20
TARGETING_BAND_RATIO = 0.035


@dataclass(frozen=True)
class AblationConfig:
    name: str
    sweep: str
    horizon: int = PREDICTIVE_HORIZON
    replan_interval: int = REPLAN_INTERVAL
    score_mode: str = "recoverability"
    action_space: str = "baseline"
    burn_hold_steps: int = 0


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    planner_kind: str
    ablation: AblationConfig | None = None


def define_controllers() -> List[ControllerSpec]:
    return [ControllerSpec(config.name, "predictive", config) for config in define_ablation_configs()]


def define_ablation_configs() -> List[AblationConfig]:
    return [
        AblationConfig("baseline_h260_r40_recoverability_base", "baseline"),
        AblationConfig("horizon_500", "horizon", horizon=500),
        AblationConfig("horizon_1000", "horizon", horizon=1000),
        AblationConfig("horizon_2000", "horizon", horizon=2000),
        AblationConfig("replan_20", "replan_interval", replan_interval=20),
        AblationConfig("replan_80", "replan_interval", replan_interval=80),
        AblationConfig("replan_150", "replan_interval", replan_interval=150),
        AblationConfig("score_crossing_only", "score", score_mode="crossing_only"),
        AblationConfig("action_expanded", "action_space", action_space="expanded"),
        AblationConfig("action_expanded_burn_hold", "action_space", action_space="expanded", burn_hold_steps=2),
    ]


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize_action(action_x: float, action_y: float, max_norm: float = 1.0) -> tuple[float, float]:
    action_norm = math.sqrt(action_x * action_x + action_y * action_y)
    if action_norm > max_norm and action_norm > 1.0e-12:
        scale = max_norm / action_norm
        return action_x * scale, action_y * scale
    return clamp(action_x, -1.0, 1.0), clamp(action_y, -1.0, 1.0)


def classify_state(pos: Sequence[float], vel: Sequence[float], target_radius: float) -> str:
    radius = math.sqrt(pos[0] * pos[0] + pos[1] * pos[1])
    if radius <= 1.0e-12:
        return "angle_misaligned"
    r_hat_x = pos[0] / radius
    r_hat_y = pos[1] / radius
    t_hat_x = -r_hat_y
    t_hat_y = r_hat_x
    speed = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1])
    v_circ = math.sqrt(MU / target_radius)
    vr = vel[0] * r_hat_x + vel[1] * r_hat_y
    vt = vel[0] * t_hat_x + vel[1] * t_hat_y
    r_ratio = (radius - target_radius) / target_radius
    abs_r_ratio = abs(r_ratio)
    radial_fraction = abs(vr) / (speed + 1.0e-12)
    tangential_error = abs(vt - v_circ) / (v_circ + 1.0e-12)

    if abs_r_ratio <= PREWINDOW_OUTER_RATIO:
        return "near_window"
    moving_away = r_ratio * vr > 0.0
    if moving_away and radial_fraction > 0.08:
        return "angle_misaligned"
    if radial_fraction > 0.75 or tangential_error > 0.80:
        return "angle_misaligned"
    if r_ratio > 0.0:
        return "outer_orbit"
    return "inner_orbit"


def iter_cases(controllers: Sequence[ControllerSpec]) -> Iterable[tuple[ControllerSpec, float, float, float, float]]:
    for controller in controllers:
        for thrust in THRUST_VALUES:
            for angle in ANGLE_VALUES:
                for r0 in R0_VALUES:
                    yield controller, r0, angle, thrust, TARGET_RADIUS_SCALE


def case_key(row_or_case: Dict[str, object] | tuple[ControllerSpec, float, float, float, float]) -> tuple[str, float, float, float, float]:
    if isinstance(row_or_case, tuple):
        controller, r0, angle, thrust, target_scale = row_or_case
        return (controller.name, float(r0), float(angle), float(thrust), float(target_scale))
    return (
        str(row_or_case["variant_name"]),
        float(row_or_case["r0_over_target"]),
        float(row_or_case["initial_velocity_angle_deg"]),
        float(row_or_case["thrust_scale"]),
        float(row_or_case["target_radius_scale"]),
    )


def run_case_tuple(case: tuple[ControllerSpec, float, float, float, float]) -> Dict[str, object]:
    controller, r0, angle, thrust, target_scale = case
    return rollout_case(controller, r0, angle, thrust, target_scale)


def rollout_case(
    controller: ControllerSpec,
    r0: float,
    angle: float,
    thrust_scale: float,
    target_scale: float,
    record_trajectory: bool = False,
) -> Dict[str, object]:
    runtime_start = time.perf_counter()
    ablation = controller.ablation or AblationConfig(controller.name, "baseline")
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
    phase = "DESCENT"
    prev_controller_r_error: float | None = None
    prev_r_error: float | None = None
    radius_errors: List[float] = []
    vr_values: List[float] = []
    success_counter = 0
    radial_stall_counter = 0
    radius_crossings_total = 0
    first_crossing_step: int | None = None
    crossing_vr: float | None = None
    crossing_vt_error: float | None = None
    crossing_radius_error: float | None = None
    crossing_recovery_samples: List[float] = []
    capture_entered = False
    lock_entered = False
    success = False
    terminated = False
    truncated = False
    termination_reason = ""
    planner_mode_counts: Counter[str] = Counter()
    energy_error_ratios: List[float] = []
    angular_momentum_ratios: List[float] = []
    tracking_error_ratios: List[float] = []
    speed_ratios: List[float] = []
    trajectory_radius: List[float] = []
    trajectory_x: List[float] = []
    trajectory_y: List[float] = []
    trajectory_energy: List[float] = []
    trajectory_delta_energy: List[float] = []
    trajectory_angular_momentum: List[float] = []
    trajectory_radial_velocity: List[float] = []
    trajectory_vt_error: List[float] = []
    trajectory_radius_error_ratio: List[float] = []
    trajectory_desired_radius: List[float] = []
    trajectory_desired_radial_velocity: List[float] = []
    trajectory_tracking_error: List[float] = []
    trajectory_speed_ratio: List[float] = []
    trajectory_burn_flag: List[int] = []
    trajectory_event_flag: List[int] = []
    transfer_energy_error_ratios: List[float] = []
    transfer_l_error_ratios: List[float] = []
    burn_steps = 0
    coast_steps = 0
    burn_streak = 0
    cooldown_remaining = 0
    event_count = 0
    burn_count = 0
    event_steps: List[int] = []
    burn_start_steps: List[int] = []
    event_burn_remaining = 0
    event_burn_type = ""
    planned_action_x = 0.0
    planned_action_y = 0.0
    plan_steps_remaining = 0
    latest_plan_score = float("inf")
    latest_recoverability_score = float("inf")
    planner_choice_counts: Counter[str] = Counter()
    planner_choice_named_counts: Counter[str] = Counter()
    chosen_action_norms: List[float] = []
    score_margins: List[float] = []
    predictive_replans = 0
    candidate_eval_count = 0
    prev_vr_for_event: float | None = None
    turning_point_count = 0
    overspeed_hit = False
    initial_radius_error = radius0 - target_radius
    trajectory_amplitude0 = clamp(abs(initial_radius_error), 0.005 * target_radius, 0.08 * target_radius)
    initial_radius_error_ratio = initial_radius_error / target_radius
    injection_duration = 0
    if abs(initial_radius_error_ratio) > PREWINDOW_OUTER_RATIO:
        injection_duration = int(
            round(
                clamp(
                    float(INJECTION_MIN_STEPS) + 1200.0 * abs(initial_radius_error_ratio),
                    float(INJECTION_MIN_STEPS),
                    float(INJECTION_MAX_STEPS),
                )
            )
        )
    transfer_a = 0.5 * (radius0 + target_radius)
    transfer_e = abs(radius0 - target_radius) / (radius0 + target_radius + 1.0e-12)
    transfer_energy = -MU / (2.0 * transfer_a)
    transfer_l = math.sqrt(max(MU * transfer_a * (1.0 - transfer_e * transfer_e), 1.0e-12))

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

    def state_energy(state_x: float, state_y: float, state_vx: float, state_vy: float) -> float:
        state_radius = math.sqrt(state_x * state_x + state_y * state_y)
        speed2 = state_vx * state_vx + state_vy * state_vy
        return 0.5 * speed2 - MU / (state_radius + 1.0e-12)

    def state_energy_error_ratio(state_x: float, state_y: float, state_vx: float, state_vy: float) -> float:
        energy = state_energy(state_x, state_y, state_vx, state_vy)
        target_energy = -MU / (2.0 * target_radius)
        return abs(energy - target_energy) / (abs(target_energy) + 1.0e-12)

    def delta_energy(state_x: float, state_y: float, state_vx: float, state_vy: float) -> float:
        target_energy = -MU / (2.0 * target_radius)
        return target_energy - state_energy(state_x, state_y, state_vx, state_vy)

    def angular_momentum(state_x: float, state_y: float, state_vx: float, state_vy: float) -> float:
        return state_x * state_vy - state_y * state_vx

    def angular_momentum_ratio(state_x: float, state_y: float, state_vx: float, state_vy: float) -> float:
        state_radius = math.sqrt(state_x * state_x + state_y * state_y)
        target_l = state_radius * v_circ
        return abs(angular_momentum(state_x, state_y, state_vx, state_vy)) / (abs(target_l) + 1.0e-12)

    def candidate_score(state_x: float, state_y: float, state_vx: float, state_vy: float) -> float:
        state_radius = math.sqrt(state_x * state_x + state_y * state_y)
        inv_r = 1.0 / (state_radius + 1.0e-12)
        vr_state = (state_x * state_vx + state_y * state_vy) * inv_r
        vt_state = state_vx * (-state_y * inv_r) + state_vy * (state_x * inv_r)
        v_t_error = vt_state - v_circ
        r_err_ratio = abs(state_radius - target_radius) / target_radius
        v_r_ratio = abs(vr_state) / v_circ
        v_t_ratio = abs(v_t_error) / v_circ
        energy_ratio = state_energy_error_ratio(state_x, state_y, state_vx, state_vy)
        score = (
            BASE_PARAMS["score_r"] * r_err_ratio
            + BASE_PARAMS["score_vr"] * v_r_ratio
            + BASE_PARAMS["score_vt"] * v_t_ratio
            + BASE_PARAMS["score_energy"] * energy_ratio
        )
        if (state_radius - target_radius) > 0.0 and (state_radius - target_radius) / target_radius < WS_BAND_RATIO and vr_state < 0.0:
            score -= BASE_PARAMS["capture_bonus"]
        if r_err_ratio < 3.0e-4 and v_r_ratio < 2.0e-2:
            score -= BASE_PARAMS["inner_capture_bonus"]
        return score

    def basis(state_x: float, state_y: float, state_vx: float, state_vy: float) -> tuple[float, float, float, float, float, float, float, float]:
        state_radius = math.sqrt(state_x * state_x + state_y * state_y)
        inv_r = 1.0 / (state_radius + 1.0e-12)
        r_hat_x = state_x * inv_r
        r_hat_y = state_y * inv_r
        t_hat_x = -r_hat_y
        t_hat_y = r_hat_x
        speed = math.sqrt(state_vx * state_vx + state_vy * state_vy)
        vr = state_vx * r_hat_x + state_vy * r_hat_y
        vt = state_vx * t_hat_x + state_vy * t_hat_y
        return r_hat_x, r_hat_y, t_hat_x, t_hat_y, speed, vr, vt, state_radius

    def baseline_action(state_vx: float, state_vy: float) -> tuple[float, float]:
        speed = math.sqrt(state_vx * state_vx + state_vy * state_vy)
        if speed <= 1.0e-12:
            return 0.0, 0.0
        return -state_vx / speed, -state_vy / speed

    def prograde_action(state_vx: float, state_vy: float) -> tuple[float, float]:
        speed = math.sqrt(state_vx * state_vx + state_vy * state_vy)
        if speed <= 1.0e-12:
            return 0.0, 0.0
        return state_vx / speed, state_vy / speed

    def capture_lock_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        current_phase: str,
    ) -> tuple[float, float]:
        inv_r = 1.0 / (state_radius + 1.0e-12)
        r_hat_x = state_x * inv_r
        r_hat_y = state_y * inv_r
        t_hat_x = -r_hat_y
        t_hat_y = r_hat_x
        real_r_ratio = (state_radius - target_radius) / target_radius
        vr_ratio = ((state_x * state_vx + state_y * state_vy) * inv_r) / v_circ
        vt_ratio = (state_vx * t_hat_x + state_vy * t_hat_y - v_circ) / v_circ
        if current_phase == "CAPTURE":
            radial_cmd = -(cfg.capture_radial_pos_gain * real_r_ratio + cfg.capture_radial_vel_gain * vr_ratio)
            tangential_cmd = -(cfg.capture_tangential_gain * vt_ratio)
            radial_cmd = clamp(radial_cmd, -cfg.capture_radial_limit, cfg.capture_radial_limit)
            tangential_cmd = clamp(tangential_cmd, -cfg.capture_tangential_limit, cfg.capture_tangential_limit)
        else:
            radial_cmd = -(cfg.lock_radial_pos_gain * real_r_ratio + cfg.lock_radial_vel_gain * vr_ratio)
            tangential_cmd = -(cfg.lock_tangential_gain * vt_ratio)
            radial_cmd = clamp(radial_cmd, -cfg.lock_radial_limit, cfg.lock_radial_limit)
            tangential_cmd = clamp(tangential_cmd, -cfg.lock_tangential_limit, cfg.lock_tangential_limit)
        return (
            clamp(radial_cmd * r_hat_x + tangential_cmd * t_hat_x, -1.0, 1.0),
            clamp(radial_cmd * r_hat_y + tangential_cmd * t_hat_y, -1.0, 1.0),
        )

    def adaptive_soft_ws_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        abs_r_error_ratio: float,
    ) -> tuple[float, float]:
        inv_r = 1.0 / (state_radius + 1.0e-12)
        r_hat_x = state_x * inv_r
        r_hat_y = state_y * inv_r
        retro_x, retro_y = baseline_action(state_vx, state_vy)
        z = clamp(abs_r_error_ratio / (WS_BAND_RATIO + 1.0e-12), 0.0, 1.0)
        scale = BASE_PARAMS["min_scale"] + (BASE_PARAMS["max_scale"] - BASE_PARAMS["min_scale"]) * z
        candidates = [
            (retro_x, retro_y),
            (
                clamp(retro_x - BASE_PARAMS["inward_bias"] * r_hat_x, -1.0, 1.0),
                clamp(retro_y - BASE_PARAMS["inward_bias"] * r_hat_y, -1.0, 1.0),
            ),
            (scale * retro_x, scale * retro_y),
        ]
        best_action = (retro_x, retro_y)
        best_score = float("inf")
        for candidate_x, candidate_y in candidates:
            candidate_x, candidate_y = normalize_action(candidate_x, candidate_y)
            next_x, next_y, next_vx, next_vy = env_step(state_x, state_y, state_vx, state_vy, candidate_x, candidate_y)
            score = candidate_score(next_x, next_y, next_vx, next_vy)
            if score < best_score:
                best_score = score
                best_action = (candidate_x, candidate_y)
        return best_action

    def prewindow_radial_medium_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
    ) -> tuple[float, float]:
        retro_x, retro_y = baseline_action(state_vx, state_vy)
        if abs(real_r_ratio) > PREWINDOW_OUTER_RATIO:
            return retro_x, retro_y
        inv_r = 1.0 / (state_radius + 1.0e-12)
        r_hat_x = state_x * inv_r
        r_hat_y = state_y * inv_r
        inward_sign = -1.0 if real_r_ratio > 0.0 else 1.0
        gain = BASE_PARAMS["prewindow_radial_gain"]
        return normalize_action(retro_x + gain * inward_sign * r_hat_x, retro_y + gain * inward_sign * r_hat_y)

    def soft_linear_3e4_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
        abs_r_error_ratio: float,
    ) -> tuple[float, float]:
        pre_x, pre_y = prewindow_radial_medium_action(state_x, state_y, state_vx, state_vy, state_radius, real_r_ratio)
        ws_x, ws_y = adaptive_soft_ws_action(state_x, state_y, state_vx, state_vy, state_radius, abs_r_error_ratio)
        alpha = clamp(abs_r_error_ratio / (3.0e-4 + 1.0e-12), 0.0, 1.0)
        return normalize_action(alpha * pre_x + (1.0 - alpha) * ws_x, alpha * pre_y + (1.0 - alpha) * ws_y)

    def energy_planner_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
        abs_r_error_ratio: float,
    ) -> tuple[float, float, str, float, float]:
        soft_x, soft_y = soft_linear_3e4_action(state_x, state_y, state_vx, state_vy, state_radius, real_r_ratio, abs_r_error_ratio)
        d_energy = delta_energy(state_x, state_y, state_vx, state_vy)
        target_energy = -MU / (2.0 * target_radius)
        energy_error_ratio = abs(d_energy) / (abs(target_energy) + 1.0e-12)
        if abs_r_error_ratio <= PREWINDOW_OUTER_RATIO or energy_error_ratio < 4.0e-4:
            return soft_x, soft_y, "soft_window", energy_error_ratio, d_energy

        r_hat_x, r_hat_y, _t_hat_x, _t_hat_y, _speed, _vr, _vt, _radius = basis(state_x, state_y, state_vx, state_vy)
        if d_energy > 0.0:
            tangential_x, tangential_y = prograde_action(state_vx, state_vy)
        else:
            tangential_x, tangential_y = baseline_action(state_vx, state_vy)
        radial_sign = -1.0 if real_r_ratio > 0.0 else 1.0
        w1 = clamp(0.55 + 0.45 * energy_error_ratio / 0.04, 0.55, 1.00)
        w2 = clamp(0.10 + 0.18 * min(abs_r_error_ratio / 0.05, 1.0), 0.10, 0.28)
        max_norm = clamp(0.35 + 0.65 * energy_error_ratio / 0.03, 0.35, 1.00)
        guide_x, guide_y = normalize_action(
            w1 * tangential_x + w2 * radial_sign * r_hat_x,
            w1 * tangential_y + w2 * radial_sign * r_hat_y,
            max_norm=max_norm,
        )
        planner_alpha = clamp(max(energy_error_ratio / 0.015, abs_r_error_ratio / 0.02), 0.0, 1.0)
        action_x, action_y = normalize_action(
            planner_alpha * guide_x + (1.0 - planner_alpha) * soft_x,
            planner_alpha * guide_y + (1.0 - planner_alpha) * soft_y,
            max_norm=max_norm,
        )
        mode = "energy_correction" if planner_alpha > 0.75 else "blend_window"
        return action_x, action_y, mode, energy_error_ratio, d_energy

    def orbit_intersection_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
        abs_r_error_ratio: float,
    ) -> tuple[float, float, str, float, float]:
        soft_x, soft_y = soft_linear_3e4_action(state_x, state_y, state_vx, state_vy, state_radius, real_r_ratio, abs_r_error_ratio)
        d_energy = delta_energy(state_x, state_y, state_vx, state_vy)
        target_energy = -MU / (2.0 * target_radius)
        energy_error_ratio = abs(d_energy) / (abs(target_energy) + 1.0e-12)
        if abs_r_error_ratio <= PREWINDOW_OUTER_RATIO:
            return soft_x, soft_y, "soft_window", energy_error_ratio, d_energy

        r_hat_x, r_hat_y, t_hat_x, t_hat_y, speed, vr, vt, _radius = basis(state_x, state_y, state_vx, state_vy)
        radial_sign = -1.0 if real_r_ratio > 0.0 else 1.0
        moving_away = (real_r_ratio > 0.0 and vr >= 0.0) or (real_r_ratio < 0.0 and vr <= 0.0)
        current_l = angular_momentum(state_x, state_y, state_vx, state_vy)
        target_l = state_radius * v_circ
        l_ratio = abs(current_l) / (abs(target_l) + 1.0e-12)
        tangent_orientation = 1.0 if vt >= 0.0 else -1.0
        if l_ratio < 0.96:
            tangential_sign = tangent_orientation
            tangential_weight = clamp(0.20 + (0.96 - l_ratio) * 1.8, 0.20, 0.70)
        elif l_ratio > 1.04:
            tangential_sign = -tangent_orientation
            tangential_weight = clamp(0.16 + (l_ratio - 1.04) * 1.4, 0.16, 0.62)
        else:
            tangential_sign = tangent_orientation
            tangential_weight = 0.12
        radial_weight = 0.62 if moving_away else 0.38
        radial_weight = clamp(radial_weight + min(abs_r_error_ratio / 0.08, 1.0) * 0.22, 0.38, 0.84)
        energy_weight = clamp(energy_error_ratio / 0.05, 0.0, 0.28)
        if d_energy > 0.0:
            energy_x, energy_y = prograde_action(state_vx, state_vy)
        else:
            energy_x, energy_y = baseline_action(state_vx, state_vy)
        guide_x, guide_y = normalize_action(
            radial_weight * radial_sign * r_hat_x
            + tangential_weight * tangential_sign * t_hat_x
            + energy_weight * energy_x,
            radial_weight * radial_sign * r_hat_y
            + tangential_weight * tangential_sign * t_hat_y
            + energy_weight * energy_y,
            max_norm=0.78,
        )
        blend = clamp(max(abs_r_error_ratio / 0.015, abs(l_ratio - 1.0) / 0.18), 0.0, 1.0)
        action_x, action_y = normalize_action(
            blend * guide_x + (1.0 - blend) * soft_x,
            blend * guide_y + (1.0 - blend) * soft_y,
            max_norm=0.78,
        )
        mode = "orbit_intersection" if blend > 0.75 else "blend_window"
        return action_x, action_y, mode, energy_error_ratio, d_energy

    def burn_coast_guidance(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
        abs_r_error_ratio: float,
    ) -> tuple[float, float, str, float, float, bool]:
        soft_x, soft_y = soft_linear_3e4_action(state_x, state_y, state_vx, state_vy, state_radius, real_r_ratio, abs_r_error_ratio)
        d_energy = delta_energy(state_x, state_y, state_vx, state_vy)
        target_energy = -MU / (2.0 * target_radius)
        energy_error_ratio = abs(d_energy) / (abs(target_energy) + 1.0e-12)
        if abs_r_error_ratio <= PREWINDOW_OUTER_RATIO:
            return soft_x, soft_y, "soft_window", energy_error_ratio, d_energy, True

        r_hat_x, r_hat_y, t_hat_x, t_hat_y, _speed, vr, vt, _radius = basis(state_x, state_y, state_vx, state_vy)
        wrong_radial_sign = (real_r_ratio > 0.0 and vr >= 0.0) or (real_r_ratio < 0.0 and vr <= 0.0)
        near_turning_point = abs(vr) / (v_circ + 1.0e-12) < 2.5e-3
        should_burn = energy_error_ratio > 0.02 or wrong_radial_sign or near_turning_point
        if not should_burn:
            return 0.0, 0.0, "coast", energy_error_ratio, d_energy, False

        radial_sign = -1.0 if real_r_ratio > 0.0 else 1.0
        current_l = angular_momentum(state_x, state_y, state_vx, state_vy)
        target_l = state_radius * v_circ
        l_ratio = abs(current_l) / (abs(target_l) + 1.0e-12)
        tangent_orientation = 1.0 if vt >= 0.0 else -1.0
        if l_ratio < 0.97:
            tangential_sign = tangent_orientation
            tangential_weight = clamp(0.14 + (0.97 - l_ratio) * 0.95, 0.14, 0.42)
        elif l_ratio > 1.03:
            tangential_sign = -tangent_orientation
            tangential_weight = clamp(0.12 + (l_ratio - 1.03) * 0.75, 0.12, 0.36)
        else:
            tangential_sign = tangent_orientation
            tangential_weight = 0.08
        radial_weight = 0.22 if not wrong_radial_sign else 0.42
        radial_weight = clamp(radial_weight + min(abs_r_error_ratio / 0.08, 1.0) * 0.10, 0.22, 0.52)
        energy_weight = clamp(energy_error_ratio / 0.08, 0.0, 0.16)
        if d_energy > 0.0:
            energy_x, energy_y = prograde_action(state_vx, state_vy)
        else:
            energy_x, energy_y = baseline_action(state_vx, state_vy)
        max_norm = 0.32 if energy_error_ratio < 0.05 else 0.42
        action_x, action_y = normalize_action(
            radial_weight * radial_sign * r_hat_x
            + tangential_weight * tangential_sign * t_hat_x
            + energy_weight * energy_x,
            radial_weight * radial_sign * r_hat_y
            + tangential_weight * tangential_sign * t_hat_y
            + energy_weight * energy_y,
            max_norm=max_norm,
        )
        return action_x, action_y, "burn", energy_error_ratio, d_energy, True

    def event_tangential_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        burn_type: str,
    ) -> tuple[float, float]:
        _r_hat_x, _r_hat_y, t_hat_x, t_hat_y, _speed, _vr, vt, _radius = basis(state_x, state_y, state_vx, state_vy)
        tangent_orientation = 1.0 if vt >= 0.0 else -1.0
        if burn_type == "prograde":
            tangential_sign = tangent_orientation
        else:
            tangential_sign = -tangent_orientation
        return normalize_action(0.34 * tangential_sign * t_hat_x, 0.34 * tangential_sign * t_hat_y, max_norm=0.34)

    def oscillation_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
        abs_r_error_ratio: float,
    ) -> tuple[float, float, str, float, float]:
        soft_x, soft_y = soft_linear_3e4_action(state_x, state_y, state_vx, state_vy, state_radius, real_r_ratio, abs_r_error_ratio)
        d_energy = delta_energy(state_x, state_y, state_vx, state_vy)
        target_energy = -MU / (2.0 * target_radius)
        energy_error_ratio = abs(d_energy) / (abs(target_energy) + 1.0e-12)
        if abs_r_error_ratio <= PREWINDOW_OUTER_RATIO:
            return soft_x, soft_y, "soft_window", energy_error_ratio, d_energy

        r_hat_x, r_hat_y, t_hat_x, t_hat_y, _speed, vr, vt, _radius = basis(state_x, state_y, state_vx, state_vy)
        r_sign = 1.0 if real_r_ratio > 0.0 else -1.0
        vr_sign = 1.0 if vr > 0.0 else -1.0 if vr < 0.0 else 0.0
        moving_away = real_r_ratio * vr > 0.0
        moving_toward = real_r_ratio * vr < 0.0
        current_l = angular_momentum(state_x, state_y, state_vx, state_vy)
        target_l = state_radius * v_circ
        l_ratio = abs(current_l) / (abs(target_l) + 1.0e-12)
        vr_ratio_abs = abs(vr) / (v_circ + 1.0e-12)

        k_r = 0.16 + min(abs_r_error_ratio / 0.08, 1.0) * 0.10
        k_v = 0.07
        if moving_away:
            k_r += 0.14
            k_v += 0.04
        elif moving_toward:
            coast_scale = 0.45 if vr_ratio_abs > 2.0e-3 and 0.86 <= l_ratio <= 1.14 else 0.70
            k_r *= coast_scale
            k_v *= coast_scale

        radial_cmd = -k_r * r_sign - k_v * vr_sign
        radial_cmd = clamp(radial_cmd, -0.38, 0.38)

        tangent_orientation = 1.0 if vt >= 0.0 else -1.0
        vt_abs_ratio = abs(vt) / (v_circ + 1.0e-12)
        tangential_cmd = 0.0
        if vt_abs_ratio < 0.96:
            tangential_cmd = tangent_orientation * clamp((0.96 - vt_abs_ratio) * 0.55, 0.04, 0.16)
        elif vt_abs_ratio > 1.04:
            tangential_cmd = -tangent_orientation * clamp((vt_abs_ratio - 1.04) * 0.45, 0.04, 0.14)
        elif l_ratio < 0.94:
            tangential_cmd = tangent_orientation * 0.06
        elif l_ratio > 1.06:
            tangential_cmd = -tangent_orientation * 0.06

        max_norm = 0.42 if moving_away else 0.30 if moving_toward else 0.34
        return (
            *normalize_action(
                radial_cmd * r_hat_x + tangential_cmd * t_hat_x,
                radial_cmd * r_hat_y + tangential_cmd * t_hat_y,
                max_norm=max_norm,
            ),
            "oscillation",
            energy_error_ratio,
            d_energy,
        )

    def desired_radial_trajectory(step_index: int) -> tuple[float, float]:
        time_sec = step_index * DT
        omega = 2.0 * math.pi / (TRAJECTORY_PERIOD_STEPS * DT)
        decay = TRAJECTORY_DECAY_PER_STEP / DT
        amplitude = trajectory_amplitude0 * math.exp(-decay * time_sec)
        phase_arg = omega * time_sec
        desired_radius = target_radius + amplitude * math.sin(phase_arg)
        desired_vr = amplitude * (-decay * math.sin(phase_arg) + omega * math.cos(phase_arg))
        return desired_radius, desired_vr

    def trajectory_tracking_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
        abs_r_error_ratio: float,
        step_index: int,
    ) -> tuple[float, float, str, float, float, float, float, float]:
        soft_x, soft_y = soft_linear_3e4_action(state_x, state_y, state_vx, state_vy, state_radius, real_r_ratio, abs_r_error_ratio)
        d_energy = delta_energy(state_x, state_y, state_vx, state_vy)
        target_energy = -MU / (2.0 * target_radius)
        energy_error_ratio = abs(d_energy) / (abs(target_energy) + 1.0e-12)
        desired_radius, desired_vr = desired_radial_trajectory(step_index)
        tracking_error_ratio = (state_radius - desired_radius) / target_radius
        if abs_r_error_ratio <= PREWINDOW_OUTER_RATIO:
            return soft_x, soft_y, "soft_window", energy_error_ratio, d_energy, desired_radius, desired_vr, tracking_error_ratio

        r_hat_x, r_hat_y, t_hat_x, t_hat_y, _speed, vr, vt, _radius = basis(state_x, state_y, state_vx, state_vy)
        vr_track_error_ratio = (vr - desired_vr) / (v_circ + 1.0e-12)
        vt_error_ratio = (vt - v_circ) / (v_circ + 1.0e-12)
        k_p = 0.06
        k_d = 0.10
        k_t = 0.025
        radial_cmd = -(k_p * tracking_error_ratio + k_d * vr_track_error_ratio)
        tangential_cmd = -(k_t * vt_error_ratio)
        radial_cmd = clamp(radial_cmd, -0.045, 0.045)
        tangential_cmd = clamp(tangential_cmd, -0.025, 0.025)
        max_norm = 0.30
        action_x, action_y = normalize_action(
            radial_cmd * r_hat_x + tangential_cmd * t_hat_x,
            radial_cmd * r_hat_y + tangential_cmd * t_hat_y,
            max_norm=max_norm,
        )
        return action_x, action_y, "trajectory_tracking", energy_error_ratio, d_energy, desired_radius, desired_vr, tracking_error_ratio

    def transfer_error_ratios(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
    ) -> tuple[float, float]:
        current_energy = state_energy(state_x, state_y, state_vx, state_vy)
        current_l = angular_momentum(state_x, state_y, state_vx, state_vy)
        energy_error_ratio = (current_energy - transfer_energy) / (abs(transfer_energy) + 1.0e-12)
        l_error_ratio = (current_l - transfer_l) / (abs(transfer_l) + 1.0e-12)
        return energy_error_ratio, l_error_ratio

    def elliptical_orbit_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
        abs_r_error_ratio: float,
    ) -> tuple[float, float, str, float, float, float, float]:
        soft_x, soft_y = soft_linear_3e4_action(state_x, state_y, state_vx, state_vy, state_radius, real_r_ratio, abs_r_error_ratio)
        target_energy = -MU / (2.0 * target_radius)
        d_energy = target_energy - state_energy(state_x, state_y, state_vx, state_vy)
        energy_error_ratio = abs(d_energy) / (abs(target_energy) + 1.0e-12)
        transfer_e_ratio, transfer_l_ratio = transfer_error_ratios(state_x, state_y, state_vx, state_vy)
        if abs_r_error_ratio <= PREWINDOW_OUTER_RATIO:
            return soft_x, soft_y, "soft_window", energy_error_ratio, d_energy, transfer_e_ratio, transfer_l_ratio

        r_hat_x, r_hat_y, t_hat_x, t_hat_y, _speed, vr, vt, _radius = basis(state_x, state_y, state_vx, state_vy)
        tangent_orientation = 1.0 if vt >= 0.0 else -1.0
        if abs(transfer_e_ratio) < 0.01 and abs(transfer_l_ratio) < 0.015 and abs(real_r_ratio) > PREWINDOW_OUTER_RATIO:
            return 0.0, 0.0, "coast", energy_error_ratio, d_energy, transfer_e_ratio, transfer_l_ratio
        tangential_cmd = -(0.012 * transfer_e_ratio + 0.020 * transfer_l_ratio)
        radial_sign = -1.0 if real_r_ratio > 0.0 else 1.0
        moving_away = real_r_ratio * vr > 0.0
        radial_cmd = (0.004 + (0.003 if moving_away else 0.0)) * radial_sign
        tangential_cmd = clamp(tangential_cmd, -0.018, 0.018)
        radial_cmd = clamp(radial_cmd, -0.007, 0.007)
        action_x, action_y = normalize_action(
            radial_cmd * r_hat_x + tangential_cmd * tangent_orientation * t_hat_x,
            radial_cmd * r_hat_y + tangential_cmd * tangent_orientation * t_hat_y,
            max_norm=0.022,
        )
        return action_x, action_y, "elliptical_guidance", energy_error_ratio, d_energy, transfer_e_ratio, transfer_l_ratio

    def crossing_targeting_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
        abs_r_error_ratio: float,
    ) -> tuple[float, float, str, float, float, float, float]:
        soft_x, soft_y = soft_linear_3e4_action(state_x, state_y, state_vx, state_vy, state_radius, real_r_ratio, abs_r_error_ratio)
        target_energy = -MU / (2.0 * target_radius)
        d_energy = target_energy - state_energy(state_x, state_y, state_vx, state_vy)
        energy_error_ratio = abs(d_energy) / (abs(target_energy) + 1.0e-12)
        transfer_e_ratio, transfer_l_ratio = transfer_error_ratios(state_x, state_y, state_vx, state_vy)
        if abs_r_error_ratio <= PREWINDOW_OUTER_RATIO:
            return soft_x, soft_y, "soft_window", energy_error_ratio, d_energy, transfer_e_ratio, transfer_l_ratio

        r_hat_x, r_hat_y, t_hat_x, t_hat_y, _speed, vr, vt, _radius = basis(state_x, state_y, state_vx, state_vy)
        approaching_crossing = real_r_ratio * vr < 0.0
        targeting_band = abs_r_error_ratio < 0.035
        if not (approaching_crossing and targeting_band):
            return elliptical_orbit_action(state_x, state_y, state_vx, state_vy, state_radius, real_r_ratio, abs_r_error_ratio)

        vr_ratio = vr / (v_circ + 1.0e-12)
        vt_error_ratio = (vt - v_circ) / (v_circ + 1.0e-12)
        close_band = abs_r_error_ratio < 0.006
        k_vr = 0.20 if close_band else 0.13
        k_vt = 0.035 if close_band else 0.025
        radial_cmd = -k_vr * vr_ratio
        tangential_cmd = -k_vt * vt_error_ratio
        radial_cmd = clamp(radial_cmd, -0.18 if close_band else -0.10, 0.18 if close_band else 0.10)
        tangential_cmd = clamp(tangential_cmd, -0.035, 0.035)

        if abs(vr_ratio) < 2.0e-3 and abs(vt_error_ratio) < 1.5e-2:
            return 0.0, 0.0, "coast", energy_error_ratio, d_energy, transfer_e_ratio, transfer_l_ratio

        max_norm = 0.24 if close_band else 0.18
        action_x, action_y = normalize_action(
            radial_cmd * r_hat_x + tangential_cmd * t_hat_x,
            radial_cmd * r_hat_y + tangential_cmd * t_hat_y,
            max_norm=max_norm,
        )
        mode = "targeting_close" if close_band else "targeting"
        return action_x, action_y, mode, energy_error_ratio, d_energy, transfer_e_ratio, transfer_l_ratio

    def crossing_targeting_only_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
        abs_r_error_ratio: float,
    ) -> tuple[float, float, str, float, float, float, float]:
        target_energy = -MU / (2.0 * target_radius)
        d_energy = target_energy - state_energy(state_x, state_y, state_vx, state_vy)
        energy_error_ratio = abs(d_energy) / (abs(target_energy) + 1.0e-12)
        transfer_e_ratio, transfer_l_ratio = transfer_error_ratios(state_x, state_y, state_vx, state_vy)

        r_hat_x, r_hat_y, t_hat_x, t_hat_y, _speed, vr, vt, _radius = basis(state_x, state_y, state_vx, state_vy)
        approaching_crossing = real_r_ratio * vr < 0.0
        if not approaching_crossing or abs_r_error_ratio > TARGETING_BAND_RATIO:
            return 0.0, 0.0, "coast", energy_error_ratio, d_energy, transfer_e_ratio, transfer_l_ratio

        vr_ratio = vr / (v_circ + 1.0e-12)
        vt_error_ratio = (vt - v_circ) / (v_circ + 1.0e-12)
        close_band = abs_r_error_ratio < 0.006
        k_vr = 0.20 if close_band else 0.13
        k_vt = 0.035 if close_band else 0.025
        radial_cmd = clamp(-k_vr * vr_ratio, -0.18 if close_band else -0.10, 0.18 if close_band else 0.10)
        tangential_cmd = clamp(-k_vt * vt_error_ratio, -0.035, 0.035)

        if abs(vr_ratio) < 2.0e-3 and abs(vt_error_ratio) < 1.5e-2:
            return 0.0, 0.0, "coast", energy_error_ratio, d_energy, transfer_e_ratio, transfer_l_ratio

        max_norm = 0.24 if close_band else 0.18
        action_x, action_y = normalize_action(
            radial_cmd * r_hat_x + tangential_cmd * t_hat_x,
            radial_cmd * r_hat_y + tangential_cmd * t_hat_y,
            max_norm=max_norm,
        )
        mode = "targeting_close" if close_band else "targeting"
        return action_x, action_y, mode, energy_error_ratio, d_energy, transfer_e_ratio, transfer_l_ratio

    def minimal_transfer_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
        abs_r_error_ratio: float,
        step_index: int,
    ) -> tuple[float, float, str, float, float, float, float]:
        soft_x, soft_y = soft_linear_3e4_action(state_x, state_y, state_vx, state_vy, state_radius, real_r_ratio, abs_r_error_ratio)
        target_energy = -MU / (2.0 * target_radius)
        d_energy = target_energy - state_energy(state_x, state_y, state_vx, state_vy)
        energy_error_ratio = abs(d_energy) / (abs(target_energy) + 1.0e-12)
        transfer_e_ratio, transfer_l_ratio = transfer_error_ratios(state_x, state_y, state_vx, state_vy)

        if step_index <= injection_duration and abs(initial_radius_error_ratio) > PREWINDOW_OUTER_RATIO:
            _r_hat_x, _r_hat_y, t_hat_x, t_hat_y, _speed, _vr, vt, _radius = basis(
                state_x, state_y, state_vx, state_vy
            )
            tangent_orientation = 1.0 if vt >= 0.0 else -1.0
            if initial_radius_error_ratio > 0.0:
                unit_x, unit_y = -tangent_orientation * t_hat_x, -tangent_orientation * t_hat_y
            else:
                unit_x, unit_y = tangent_orientation * t_hat_x, tangent_orientation * t_hat_y
            return INJECTION_NORM * unit_x, INJECTION_NORM * unit_y, "injection", energy_error_ratio, d_energy, transfer_e_ratio, transfer_l_ratio

        if step_index <= injection_duration + FORCED_COAST_STEPS:
            return 0.0, 0.0, "coast", energy_error_ratio, d_energy, transfer_e_ratio, transfer_l_ratio

        return crossing_targeting_only_action(state_x, state_y, state_vx, state_vy, state_radius, real_r_ratio, abs_r_error_ratio)

    def predictive_candidates(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
    ) -> List[tuple[str, float, float]]:
        r_hat_x, r_hat_y, t_hat_x, t_hat_y, _speed, _vr, vt, _radius = basis(state_x, state_y, state_vx, state_vy)
        tangent_orientation = 1.0 if vt >= 0.0 else -1.0
        pro_x, pro_y = tangent_orientation * t_hat_x, tangent_orientation * t_hat_y
        ret_x, ret_y = -pro_x, -pro_y
        inward_x, inward_y = -r_hat_x, -r_hat_y
        outward_x, outward_y = r_hat_x, r_hat_y
        raw = [
            ("coast", 0.0, 0.0),
            ("small_prograde", 0.07 * pro_x, 0.07 * pro_y),
            ("medium_prograde", 0.14 * pro_x, 0.14 * pro_y),
            ("small_retrograde", 0.07 * ret_x, 0.07 * ret_y),
            ("medium_retrograde", 0.14 * ret_x, 0.14 * ret_y),
            ("radial_inward", 0.10 * inward_x, 0.10 * inward_y),
            ("radial_outward", 0.10 * outward_x, 0.10 * outward_y),
            ("prograde_inward", 0.08 * pro_x + 0.05 * inward_x, 0.08 * pro_y + 0.05 * inward_y),
            ("retrograde_outward", 0.08 * ret_x + 0.05 * outward_x, 0.08 * ret_y + 0.05 * outward_y),
        ]
        if ablation.action_space == "expanded":
            raw.extend(
                [
                    ("large_prograde", 0.28 * pro_x, 0.28 * pro_y),
                    ("large_retrograde", 0.28 * ret_x, 0.28 * ret_y),
                    ("strong_radial_inward", 0.24 * inward_x, 0.24 * inward_y),
                    ("strong_radial_outward", 0.24 * outward_x, 0.24 * outward_y),
                    ("strong_prograde_inward", 0.20 * pro_x + 0.16 * inward_x, 0.20 * pro_y + 0.16 * inward_y),
                    ("strong_retrograde_outward", 0.20 * ret_x + 0.16 * outward_x, 0.20 * ret_y + 0.16 * outward_y),
                    ("strong_retrograde_inward", 0.20 * ret_x + 0.16 * inward_x, 0.20 * ret_y + 0.16 * inward_y),
                    ("strong_prograde_outward", 0.20 * pro_x + 0.16 * outward_x, 0.20 * pro_y + 0.16 * outward_y),
                ]
            )
            max_norm = 0.34
        else:
            max_norm = 0.16
        return [(name, *normalize_action(ax, ay, max_norm=max_norm)) for name, ax, ay in raw]

    def candidate_category(candidate_name: str) -> str:
        if candidate_name == "coast":
            return "coast"
        if "prograde" in candidate_name and "radial" not in candidate_name and "inward" not in candidate_name and "outward" not in candidate_name:
            return "prograde"
        if "retrograde" in candidate_name and "radial" not in candidate_name and "inward" not in candidate_name and "outward" not in candidate_name:
            return "retrograde"
        if "inward" in candidate_name and "prograde" not in candidate_name and "retrograde" not in candidate_name:
            return "radial_inward"
        if "outward" in candidate_name and "prograde" not in candidate_name and "retrograde" not in candidate_name:
            return "radial_outward"
        return "diagonal"

    def simulate_predictive_candidate(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        candidate_x: float,
        candidate_y: float,
    ) -> Dict[str, float | bool]:
        sim_x, sim_y, sim_vx, sim_vy = state_x, state_y, state_vx, state_vy
        prev_sim_error = math.sqrt(sim_x * sim_x + sim_y * sim_y) - target_radius
        min_abs_r_ratio = abs(prev_sim_error) / target_radius
        crossed = False
        crossing_step = ablation.horizon + 1
        crossing_r_ratio = 1.0
        crossing_vr_ratio = 1.0
        crossing_vt_error_ratio = 1.0
        recovery_values: List[float] = []
        overspeed_risk = False
        instability_risk = False

        for sim_step in range(1, ablation.horizon + 1):
            sim_radius = math.sqrt(sim_x * sim_x + sim_y * sim_y)
            sim_error_ratio = (sim_radius - target_radius) / target_radius
            if crossed:
                sim_action_x, sim_action_y = soft_linear_3e4_action(
                    sim_x, sim_y, sim_vx, sim_vy, sim_radius, sim_error_ratio, abs(sim_error_ratio)
                )
            elif sim_step <= (int(ablation.burn_hold_steps) if ablation.burn_hold_steps > 0 else int(ablation.replan_interval)):
                sim_action_x, sim_action_y = candidate_x, candidate_y
            elif abs(sim_error_ratio) <= PREWINDOW_OUTER_RATIO:
                sim_action_x, sim_action_y = soft_linear_3e4_action(
                    sim_x, sim_y, sim_vx, sim_vy, sim_radius, sim_error_ratio, abs(sim_error_ratio)
                )
            else:
                sim_action_x, sim_action_y = 0.0, 0.0

            sim_x, sim_y, sim_vx, sim_vy = env_step(sim_x, sim_y, sim_vx, sim_vy, sim_action_x, sim_action_y)
            sim_radius = math.sqrt(sim_x * sim_x + sim_y * sim_y)
            sim_r_error = sim_radius - target_radius
            min_abs_r_ratio = min(min_abs_r_ratio, abs(sim_r_error) / target_radius)
            inv_sim_r = 1.0 / (sim_radius + 1.0e-12)
            sim_vr = (sim_x * sim_vx + sim_y * sim_vy) * inv_sim_r
            sim_vt = sim_vx * (-sim_y * inv_sim_r) + sim_vy * (sim_x * inv_sim_r)
            sim_speed = math.sqrt(sim_vx * sim_vx + sim_vy * sim_vy)
            overspeed_risk = overspeed_risk or sim_speed > 1.90 * v_circ
            instability_risk = instability_risk or sim_radius > 2.5 * target_radius or sim_radius < 0.35 * target_radius

            if not crossed and ((prev_sim_error > 0.0 and sim_r_error <= 0.0) or (prev_sim_error < 0.0 and sim_r_error >= 0.0)):
                crossed = True
                crossing_step = sim_step
                crossing_r_ratio = abs(sim_r_error) / target_radius
                crossing_vr_ratio = sim_vr / (v_circ + 1.0e-12)
                crossing_vt_error_ratio = (sim_vt - v_circ) / (v_circ + 1.0e-12)
            prev_sim_error = sim_r_error

            if crossed and len(recovery_values) < POST_CROSS_WINDOW:
                recovery_values.append(
                    abs(sim_r_error) / target_radius
                    + 0.70 * abs(sim_vr) / (v_circ + 1.0e-12)
                    + 0.50 * abs(sim_vt - v_circ) / (v_circ + 1.0e-12)
                )
                if len(recovery_values) >= POST_CROSS_WINDOW:
                    break

            if overspeed_risk or instability_risk:
                break

        recoverability = float(np.mean(recovery_values)) if recovery_values else 1.0 + min_abs_r_ratio
        safety_penalty = (6.0 if overspeed_risk else 0.0) + (8.0 if instability_risk else 0.0)
        if crossed:
            crossing_quality = crossing_r_ratio + 1.20 * abs(crossing_vr_ratio) + 0.80 * abs(crossing_vt_error_ratio)
            if ablation.score_mode == "crossing_only":
                score = crossing_quality + 0.001 * crossing_step + safety_penalty
            else:
                score = crossing_quality + 1.80 * recoverability + 0.001 * crossing_step + safety_penalty
        else:
            crossing_quality = 1.0 + min_abs_r_ratio
            if ablation.score_mode == "crossing_only":
                score = 0.45 + min_abs_r_ratio + safety_penalty
            else:
                score = 0.45 + min_abs_r_ratio + recoverability + safety_penalty
        recoverable = bool(
            crossed
            and abs(crossing_r_ratio) < RECOVERABLE_R_RATIO
            and abs(crossing_vr_ratio) < RECOVERABLE_VR_RATIO
            and abs(crossing_vt_error_ratio) < RECOVERABLE_VT_RATIO
            and recoverability < 0.20
            and not overspeed_risk
            and not instability_risk
        )
        return {
            "score": float(score),
            "crossing_quality": float(crossing_quality),
            "recoverability": float(recoverability),
            "crossed": bool(crossed),
            "recoverable": bool(recoverable),
            "overspeed_risk": bool(overspeed_risk),
            "instability_risk": bool(instability_risk),
            "min_abs_r_ratio": float(min_abs_r_ratio),
        }

    def predictive_planner_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
        abs_r_error_ratio: float,
    ) -> tuple[float, float, str, float, float, float, float]:
        nonlocal planned_action_x, planned_action_y, plan_steps_remaining
        nonlocal latest_plan_score, latest_recoverability_score, predictive_replans, candidate_eval_count

        target_energy = -MU / (2.0 * target_radius)
        d_energy = target_energy - state_energy(state_x, state_y, state_vx, state_vy)
        energy_error_ratio = abs(d_energy) / (abs(target_energy) + 1.0e-12)
        transfer_e_ratio, transfer_l_ratio = transfer_error_ratios(state_x, state_y, state_vx, state_vy)

        if abs_r_error_ratio <= PREWINDOW_OUTER_RATIO:
            planned_action_x, planned_action_y = 0.0, 0.0
            plan_steps_remaining = 0
            soft_x, soft_y = soft_linear_3e4_action(state_x, state_y, state_vx, state_vy, state_radius, real_r_ratio, abs_r_error_ratio)
            return soft_x, soft_y, "phase76_handoff", energy_error_ratio, d_energy, transfer_e_ratio, transfer_l_ratio

        if plan_steps_remaining <= 0:
            predictive_replans += 1
            best_name = "coast"
            best_action = (0.0, 0.0)
            best_score = float("inf")
            second_best_score = float("inf")
            best_recoverability = float("inf")
            for candidate_name, candidate_x, candidate_y in predictive_candidates(state_x, state_y, state_vx, state_vy):
                candidate_eval_count += 1
                forecast = simulate_predictive_candidate(
                    state_x, state_y, state_vx, state_vy, candidate_x, candidate_y
                )
                forecast_score = float(forecast["score"])
                if forecast_score < best_score:
                    second_best_score = best_score
                    best_name = candidate_name
                    best_score = forecast_score
                    best_recoverability = float(forecast["recoverability"])
                    best_action = (candidate_x, candidate_y)
                elif forecast_score < second_best_score:
                    second_best_score = forecast_score
            planned_action_x, planned_action_y = best_action
            latest_plan_score = best_score
            latest_recoverability_score = best_recoverability
            plan_steps_remaining = int(ablation.replan_interval)
            category = candidate_category(best_name)
            planner_choice_counts[category] += 1
            planner_choice_named_counts[best_name] += 1
            chosen_action_norms.append(math.sqrt(best_action[0] * best_action[0] + best_action[1] * best_action[1]))
            if math.isfinite(second_best_score):
                score_margins.append(float(second_best_score - best_score))
            if best_name == "coast":
                planned_action_x, planned_action_y = 0.0, 0.0

        plan_steps_remaining -= 1
        action_x, action_y = planned_action_x, planned_action_y
        if ablation.burn_hold_steps > 0 and plan_steps_remaining < int(ablation.replan_interval) - int(ablation.burn_hold_steps):
            action_x, action_y = 0.0, 0.0
        mode = "coast" if abs(action_x) + abs(action_y) <= 1.0e-12 else "predictive_descent"
        return (
            action_x,
            action_y,
            mode,
            energy_error_ratio,
            d_energy,
            transfer_e_ratio,
            transfer_l_ratio,
        )

    for step in range(1, MAX_STEPS + 1):
        radius = math.sqrt(x * x + y * y)
        real_r_error = radius - target_radius
        inv_r = 1.0 / (radius + 1.0e-12)
        vr_current = (x * vx + y * vy) * inv_r
        vt_error = vx * (-y * inv_r) + vy * (x * inv_r) - v_circ
        real_r_ratio = real_r_error / target_radius
        abs_r_ratio = abs(real_r_ratio)
        v_r_ratio = vr_current / v_circ
        v_t_ratio = vt_error / v_circ
        turning_point_now = prev_vr_for_event is not None and (prev_vr_for_event * vr_current < 0.0)
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
            if (
                abs(real_r_ratio) < cfg.lock_r_threshold_ratio
                and abs(v_r_ratio) < cfg.lock_vr_threshold_ratio
                and abs(v_t_ratio) < cfg.lock_vt_threshold_ratio
            ):
                phase = "LOCK"
        elif phase == "LOCK":
            if abs(real_r_ratio) > cfg.unlock_r_threshold_ratio or abs(v_r_ratio) > cfg.unlock_vr_threshold_ratio:
                phase = "CAPTURE"

        planner_mode = ""
        current_energy_error_ratio = state_energy_error_ratio(x, y, vx, vy)
        current_delta_energy = delta_energy(x, y, vx, vy)
        current_l_ratio = angular_momentum_ratio(x, y, vx, vy)
        current_desired_radius, current_desired_vr = desired_radial_trajectory(step)
        current_tracking_error_ratio = (radius - current_desired_radius) / target_radius
        current_transfer_e_ratio, current_transfer_l_ratio = transfer_error_ratios(x, y, vx, vy)
        energy_error_ratios.append(current_energy_error_ratio)
        angular_momentum_ratios.append(current_l_ratio)
        tracking_error_ratios.append(abs(current_tracking_error_ratio))
        transfer_energy_error_ratios.append(abs(current_transfer_e_ratio))
        transfer_l_error_ratios.append(abs(current_transfer_l_ratio))
        if phase == "DESCENT":
            if controller.planner_kind == "orbit":
                action_x, action_y, planner_mode, current_energy_error_ratio, current_delta_energy = orbit_intersection_action(
                    x, y, vx, vy, radius, real_r_ratio, abs_r_ratio
                )
            elif controller.planner_kind == "burn_coast":
                action_x, action_y, planner_mode, current_energy_error_ratio, current_delta_energy, should_burn = burn_coast_guidance(
                    x, y, vx, vy, radius, real_r_ratio, abs_r_ratio
                )
                if planner_mode == "soft_window":
                    burn_streak = 0
                elif not should_burn:
                    action_x, action_y = 0.0, 0.0
                    planner_mode = "coast"
                    coast_steps += 1
                    burn_streak = 0
                elif cooldown_remaining > 0:
                    action_x, action_y = 0.0, 0.0
                    planner_mode = "cooldown"
                    cooldown_remaining -= 1
                    coast_steps += 1
                elif burn_streak >= MAX_BURN_STEPS:
                    action_x, action_y = 0.0, 0.0
                    planner_mode = "cooldown"
                    cooldown_remaining = COOLDOWN_STEPS - 1
                    burn_streak = 0
                    coast_steps += 1
                else:
                    burn_streak += 1
                    burn_steps += 1
            elif controller.planner_kind == "event":
                if abs_r_ratio <= PREWINDOW_OUTER_RATIO:
                    planner_mode = "soft_window"
                    action_x, action_y = soft_linear_3e4_action(x, y, vx, vy, radius, real_r_ratio, abs_r_ratio)
                elif event_burn_remaining > 0:
                    action_x, action_y = event_tangential_action(x, y, vx, vy, event_burn_type)
                    event_burn_remaining -= 1
                    burn_steps += 1
                    planner_mode = "event_burn"
                else:
                    near_turning_point = abs(vr_current) / (v_circ + 1.0e-12) < TURNING_VR_THRESHOLD
                    near_miss_event = abs_r_ratio < NEAR_MISS_WINDOW_RATIO and (real_r_ratio * vr_current > 0.0)
                    event_detected = turning_point_now or near_miss_event
                    if event_detected:
                        event_count += 1
                        event_steps.append(step)
                        burn_count += 1
                        burn_start_steps.append(step)
                        event_burn_type = "retrograde" if real_r_ratio > 0.0 else "prograde"
                        event_burn_remaining = EVENT_BURN_STEPS - 1
                        action_x, action_y = event_tangential_action(x, y, vx, vy, event_burn_type)
                        burn_steps += 1
                        planner_mode = "event_burn"
                    else:
                        action_x, action_y = 0.0, 0.0
                        coast_steps += 1
                        planner_mode = "coast" if not near_turning_point else "coast"
            elif controller.planner_kind == "oscillation":
                action_x, action_y, planner_mode, current_energy_error_ratio, current_delta_energy = oscillation_action(
                    x, y, vx, vy, radius, real_r_ratio, abs_r_ratio
                )
                if planner_mode == "coast":
                    coast_steps += 1
            elif controller.planner_kind == "trajectory":
                (
                    action_x,
                    action_y,
                    planner_mode,
                    current_energy_error_ratio,
                    current_delta_energy,
                    current_desired_radius,
                    current_desired_vr,
                    current_tracking_error_ratio,
                ) = trajectory_tracking_action(x, y, vx, vy, radius, real_r_ratio, abs_r_ratio, step)
            elif controller.planner_kind == "elliptical":
                (
                    action_x,
                    action_y,
                    planner_mode,
                    current_energy_error_ratio,
                    current_delta_energy,
                    current_transfer_e_ratio,
                    current_transfer_l_ratio,
                ) = elliptical_orbit_action(x, y, vx, vy, radius, real_r_ratio, abs_r_ratio)
            elif controller.planner_kind == "crossing_targeting":
                (
                    action_x,
                    action_y,
                    planner_mode,
                    current_energy_error_ratio,
                    current_delta_energy,
                    current_transfer_e_ratio,
                    current_transfer_l_ratio,
                ) = crossing_targeting_action(x, y, vx, vy, radius, real_r_ratio, abs_r_ratio)
            elif controller.planner_kind == "minimal_transfer":
                (
                    action_x,
                    action_y,
                    planner_mode,
                    current_energy_error_ratio,
                    current_delta_energy,
                    current_transfer_e_ratio,
                    current_transfer_l_ratio,
                ) = minimal_transfer_action(x, y, vx, vy, radius, real_r_ratio, abs_r_ratio, step)
                if planner_mode == "injection":
                    burn_steps += 1
                elif planner_mode == "coast":
                    coast_steps += 1
            elif controller.planner_kind == "predictive":
                (
                    action_x,
                    action_y,
                    planner_mode,
                    current_energy_error_ratio,
                    current_delta_energy,
                    current_transfer_e_ratio,
                    current_transfer_l_ratio,
                ) = predictive_planner_action(x, y, vx, vy, radius, real_r_ratio, abs_r_ratio)
                if planner_mode == "coast":
                    coast_steps += 1
            else:
                planner_mode = "soft_window" if abs_r_ratio <= PREWINDOW_OUTER_RATIO else "unconditioned_descent"
                action_x, action_y = soft_linear_3e4_action(x, y, vx, vy, radius, real_r_ratio, abs_r_ratio)
            planner_mode_counts[planner_mode] += 1
        else:
            action_x, action_y = capture_lock_action(x, y, vx, vy, radius, phase)

        capture_entered = capture_entered or phase == "CAPTURE"
        lock_entered = lock_entered or phase == "LOCK"
        x, y, vx, vy = env_step(x, y, vx, vy, action_x, action_y)
        radius = math.sqrt(x * x + y * y)
        r_error = radius - target_radius
        speed = math.sqrt(vx * vx + vy * vy)
        vr = (x * vx + y * vy) / (radius + 1.0e-12)
        radius_errors.append(r_error)
        vr_values.append(vr)
        speed_ratios.append(speed / (v_circ + 1.0e-12))
        if record_trajectory:
            trajectory_x.append(x)
            trajectory_y.append(y)
            trajectory_radius.append(radius)
            trajectory_energy.append(state_energy(x, y, vx, vy))
            trajectory_delta_energy.append(delta_energy(x, y, vx, vy))
            trajectory_angular_momentum.append(angular_momentum(x, y, vx, vy))
            trajectory_radial_velocity.append(vr)
            next_inv_r = 1.0 / (radius + 1.0e-12)
            next_vt = vx * (-y * next_inv_r) + vy * (x * next_inv_r)
            trajectory_vt_error.append(next_vt - v_circ)
            trajectory_radius_error_ratio.append(r_error / target_radius)
            trajectory_desired_radius.append(current_desired_radius)
            trajectory_desired_radial_velocity.append(current_desired_vr)
            trajectory_tracking_error.append((radius - current_desired_radius) / target_radius)
            trajectory_speed_ratio.append(speed / (v_circ + 1.0e-12))
            trajectory_burn_flag.append(1 if planner_mode in {"burn", "event_burn", "injection"} else 0)
            trajectory_event_flag.append(1 if event_steps and event_steps[-1] == step else 0)
        if prev_r_error is not None and ((prev_r_error > 0.0 and r_error <= 0.0) or (prev_r_error < 0.0 and r_error >= 0.0)):
            radius_crossings_total += 1
            if first_crossing_step is None:
                first_crossing_step = step
                crossing_vr = vr
                next_inv_r = 1.0 / (radius + 1.0e-12)
                next_vt = vx * (-y * next_inv_r) + vy * (x * next_inv_r)
                crossing_vt_error = next_vt - v_circ
                crossing_radius_error = r_error
        prev_r_error = r_error
        if first_crossing_step is not None and len(crossing_recovery_samples) < POST_CROSS_WINDOW:
            rec_inv_r = 1.0 / (radius + 1.0e-12)
            rec_vt = vx * (-y * rec_inv_r) + vy * (x * rec_inv_r)
            crossing_recovery_samples.append(
                abs(r_error) / target_radius
                + 0.70 * abs(vr) / (v_circ + 1.0e-12)
                + 0.50 * abs(rec_vt - v_circ) / (v_circ + 1.0e-12)
            )

        r_err_now = abs(r_error) / target_radius
        v_err_now = abs(speed - v_circ) / v_circ
        ur_x = x / (radius + 1.0e-8)
        ur_y = y / (radius + 1.0e-8)
        uv_x = vx / (speed + 1.0e-8)
        uv_y = vy / (speed + 1.0e-8)
        ang_abs_now = abs(ur_x * uv_x + ur_y * uv_y)

        if r_err_now < STRICT_CFG["tol_r"] and v_err_now < STRICT_CFG["tol_v"] and ang_abs_now < STRICT_CFG["tol_ang"]:
            success_counter += 1
        else:
            success_counter = 0

        success = success_counter >= int(STRICT_CFG["success_threshold"])

        radial_stall = (r_err_now < 0.10) and (v_err_now < 0.12) and (ang_abs_now > 0.85)
        radial_stall_counter = radial_stall_counter + 1 if radial_stall else 0

        out_range = radius > 2.5 * target_radius
        overspeed = speed > 1.90 * v_circ
        overspeed_hit = overspeed_hit or overspeed
        too_close = radius < 0.35 * target_radius
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
        prev_vr_for_event = vr

    rerr = np.asarray(radius_errors, dtype=np.float64)
    vr_arr = np.asarray(vr_values, dtype=np.float64)
    tail = vr_arr[-min(TAIL_LEN, len(vr_arr)) :] if len(vr_arr) else np.asarray([0.0])
    min_abs_rerr = float(np.min(np.abs(rerr))) if len(rerr) else 0.0
    near_miss = (not success) and min_abs_rerr / target_radius <= NEAR_MISS_REL
    dominant_mode = max(planner_mode_counts.items(), key=lambda item: item[1])[0] if planner_mode_counts else ""
    mode_counts_payload = {key: int(value) for key, value in sorted(planner_mode_counts.items())}
    scheduled_steps = burn_steps + coast_steps
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
    total_planner_choices = sum(planner_choice_counts.values())

    def choice_pct(category: str) -> float:
        if total_planner_choices <= 0:
            return 0.0
        return 100.0 * float(planner_choice_counts.get(category, 0)) / float(total_planner_choices)

    result = {
        "variant_name": ablation.name,
        "sweep": ablation.sweep,
        "predictive_horizon": int(ablation.horizon),
        "replan_interval": int(ablation.replan_interval),
        "score_mode": ablation.score_mode,
        "action_space": ablation.action_space,
        "burn_hold_steps": int(ablation.burn_hold_steps),
        "controller_name": controller.name,
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
        "dominant_planner_mode": dominant_mode,
        "planner_mode_counts_json": json.dumps(mode_counts_payload, sort_keys=True),
        "initial_energy_error_ratio": float(energy_error_ratios[0]) if energy_error_ratios else 0.0,
        "final_energy_error_ratio": float(energy_error_ratios[-1]) if energy_error_ratios else 0.0,
        "initial_angular_momentum_ratio": float(angular_momentum_ratios[0]) if angular_momentum_ratios else 0.0,
        "final_angular_momentum_ratio": float(angular_momentum_ratios[-1]) if angular_momentum_ratios else 0.0,
        "max_speed_ratio": float(max(speed_ratios)) if speed_ratios else 0.0,
        "overspeed": bool(overspeed_hit),
        "turning_point_count": int(turning_point_count),
        "mean_abs_tracking_error": float(np.mean(tracking_error_ratios)) if tracking_error_ratios else 0.0,
        "initial_transfer_energy_error_ratio": float(transfer_energy_error_ratios[0]) if transfer_energy_error_ratios else 0.0,
        "final_transfer_energy_error_ratio": float(transfer_energy_error_ratios[-1]) if transfer_energy_error_ratios else 0.0,
        "initial_transfer_angular_momentum_error_ratio": float(transfer_l_error_ratios[0]) if transfer_l_error_ratios else 0.0,
        "final_transfer_angular_momentum_error_ratio": float(transfer_l_error_ratios[-1]) if transfer_l_error_ratios else 0.0,
        "burn_steps": int(burn_steps),
        "coast_steps": int(coast_steps),
        "burn_ratio": float(burn_steps / scheduled_steps) if scheduled_steps else 0.0,
        "event_count": int(event_count),
        "burn_count": int(burn_count),
        "predictive_replans": int(predictive_replans),
        "candidate_eval_count": int(candidate_eval_count),
        "runtime_seconds": float(time.perf_counter() - runtime_start),
        "planner_choice_counts_json": json.dumps({key: int(value) for key, value in sorted(planner_choice_counts.items())}),
        "planner_choice_named_counts_json": json.dumps({key: int(value) for key, value in sorted(planner_choice_named_counts.items())}),
        "coast_chosen_pct": choice_pct("coast"),
        "prograde_pct": choice_pct("prograde"),
        "retrograde_pct": choice_pct("retrograde"),
        "radial_inward_pct": choice_pct("radial_inward"),
        "radial_outward_pct": choice_pct("radial_outward"),
        "diagonal_pct": choice_pct("diagonal"),
        "avg_chosen_thrust_norm": float(np.mean(chosen_action_norms)) if chosen_action_norms else 0.0,
        "avg_score_margin": float(np.mean(score_margins)) if score_margins else 0.0,
        "event_steps_json": json.dumps(event_steps),
        "burn_steps_json": json.dumps(burn_start_steps),
    }
    if record_trajectory:
        result["trajectory"] = {
            "radius": trajectory_radius,
            "x": trajectory_x,
            "y": trajectory_y,
            "energy": trajectory_energy,
            "delta_energy": trajectory_delta_energy,
            "angular_momentum": trajectory_angular_momentum,
            "radial_velocity": trajectory_radial_velocity,
            "vt_error": trajectory_vt_error,
            "radius_error_ratio": trajectory_radius_error_ratio,
            "desired_radius": trajectory_desired_radius,
            "desired_radial_velocity": trajectory_desired_radial_velocity,
            "tracking_error": trajectory_tracking_error,
            "speed_ratio": trajectory_speed_ratio,
            "burn_flag": trajectory_burn_flag,
            "event_flag": trajectory_event_flag,
            "target_radius": target_radius,
            "target_energy": -MU / (2.0 * target_radius),
            "target_angular_momentum": target_radius * v_circ,
            "transfer_energy": transfer_energy,
            "transfer_angular_momentum": transfer_l,
        }
    return result


def write_comparison(rows: Sequence[Dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in FIELDNAMES})


def read_comparison_rows() -> List[Dict[str, object]]:
    if not COMPARISON_CSV.exists():
        return []
    with COMPARISON_CSV.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_comparison_row(row: Dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not COMPARISON_CSV.exists()
    with COMPARISON_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in FIELDNAMES})


def controller_stats(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for name in sorted({str(row["controller_name"]) for row in rows}):
        subset = [row for row in rows if str(row["controller_name"]) == name]
        stats[name] = {
            "total": len(subset),
            "capture": sum(bool(row["capture_entered"]) for row in subset),
            "success": sum(bool(row["success"]) for row in subset),
            "near_miss": sum((not bool(row["success"])) and bool(row["near_miss"]) for row in subset),
        }
    return stats


def write_planner_diagnostics(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    planner_rows = [row for row in rows if str(row["controller_name"]) == "phase20_predictive_planner"]
    step_counts: Counter[str] = Counter()
    case_counts: Counter[str] = Counter()
    case_success: Counter[str] = Counter()
    case_capture: Counter[str] = Counter()
    for row in planner_rows:
        counts = json.loads(str(row.get("planner_mode_counts_json", "{}")))
        for mode, count in counts.items():
            step_counts[mode] += int(count)
            if int(count) > 0:
                case_counts[mode] += 1
                if bool(row["success"]):
                    case_success[mode] += 1
                if bool(row["capture_entered"]):
                    case_capture[mode] += 1
    outcome_by_mode = {}
    for mode in PLANNER_MODES:
        cases = int(case_counts.get(mode, 0))
        outcome_by_mode[mode] = {
            "cases_triggered": cases,
            "successes": int(case_success.get(mode, 0)),
            "captures": int(case_capture.get(mode, 0)),
            "success_rate": float(case_success.get(mode, 0) / cases) if cases else 0.0,
            "capture_rate": float(case_capture.get(mode, 0) / cases) if cases else 0.0,
        }
    payload = {
        "descent_step_counts": {key: int(value) for key, value in sorted(step_counts.items())},
        "case_counts": {key: int(value) for key, value in sorted(case_counts.items())},
        "outcome_by_mode": outcome_by_mode,
    }
    return payload


def crossing_rows(rows: Sequence[Dict[str, object]], controller_name: str) -> List[Dict[str, object]]:
    return [
        row
        for row in rows
        if str(row["controller_name"]) == controller_name
        and bool(row["crossing_occurs"])
        and row.get("crossing_vr_ratio", "") != ""
        and row.get("crossing_vt_error_ratio", "") != ""
    ]


def save_crossing_state_histogram(rows: Sequence[Dict[str, object]]) -> None:
    phase17_rows = crossing_rows(rows, "phase17_elliptical_orbit")
    phase18_rows = crossing_rows(rows, "phase18_crossing_targeting")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for subset, label, color in [
        (phase17_rows, "phase17", "#D95F02"),
        (phase18_rows, "phase18", "#386CB0"),
    ]:
        vr_values = [float(row["crossing_vr_ratio"]) for row in subset]
        vt_values = [float(row["crossing_vt_error_ratio"]) for row in subset]
        if vr_values:
            axes[0].hist(vr_values, bins=12, alpha=0.55, label=label, color=color)
        if vt_values:
            axes[1].hist(vt_values, bins=12, alpha=0.55, label=label, color=color)
    axes[0].axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0].set_title("v_r at first crossing")
    axes[0].set_xlabel("v_r / v_circular")
    axes[0].set_ylabel("count")
    axes[1].axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[1].set_title("v_t error at first crossing")
    axes[1].set_xlabel("(v_t - v_circular) / v_circular")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CROSSING_STATE_HIST_PNG, dpi=220)
    plt.close(fig)


def run_trace(controller: ControllerSpec, r0: float, angle: float, thrust: float) -> Dict[str, object]:
    return rollout_case(controller, r0, angle, thrust, TARGET_RADIUS_SCALE, record_trajectory=True)


def save_trajectory_plot(controller: ControllerSpec, r0: float, angle: float, thrust: float, idx: int) -> None:
    row = rollout_case(controller, r0, angle, thrust, TARGET_RADIUS_SCALE, record_trajectory=True)
    traj = row["trajectory"]
    target_radius = float(traj["target_radius"])
    target_energy = float(traj["target_energy"])
    target_l = float(traj["target_angular_momentum"])
    radius = np.asarray(traj["radius"], dtype=np.float64)
    energy = np.asarray(traj["energy"], dtype=np.float64)
    angular_momentum = np.asarray(traj["angular_momentum"], dtype=np.float64)
    radial_velocity = np.asarray(traj["radial_velocity"], dtype=np.float64)
    time_days = np.arange(len(radius), dtype=np.float64) * DT / 86400.0

    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.0), sharex=True)
    axes[0].plot(time_days, radius / target_radius, color="#386CB0", linewidth=1.1)
    axes[0].axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("radius / target")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_days, energy / abs(target_energy), color="#1B9E77", linewidth=1.1)
    axes[1].axhline(target_energy / abs(target_energy), color="#333333", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("energy / |target|")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time_days, angular_momentum / abs(target_l), color="#D95F02", linewidth=1.1, label="L / |target L|")
    axes[2].plot(time_days, np.sign(radial_velocity), color="#7570B3", linewidth=0.8, alpha=0.65, label="sign(vr)")
    axes[2].axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("L ratio / sign(vr)")
    axes[2].set_xlabel("time [days]")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best", fontsize=8)

    desired_radius = np.asarray(traj.get("desired_radius", []), dtype=np.float64)
    if len(desired_radius) == len(radius):
        axes[0].plot(time_days, desired_radius / target_radius, color="#D95F02", linewidth=0.9, linestyle=":", label="desired")
        axes[0].legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Phase 20 predictive planner r0={r0:g}, angle={angle:g}, thrust={thrust:g}, "
        f"capture={row['capture_entered']}, success={row['success']}, reason={row['termination_reason']}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = TRAJECTORY_DIR / f"case_{idx:02d}_r0_{r0:g}_angle_{angle:g}_thrust_{thrust:g}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_trajectory_diagnostics() -> None:
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    controller = ControllerSpec("phase20_predictive_planner", "predictive")
    cases = [
        (0.98, 150.0, 8000.0),
        (1.00, 170.0, 10000.0),
        (1.02, 175.0, 12000.0),
        (1.05, 165.0, 10000.0),
    ]
    for idx, (r0, angle, thrust) in enumerate(cases, start=1):
        save_trajectory_plot(controller, r0, angle, thrust, idx)


def diagnostic_cases() -> List[tuple[float, float, float]]:
    return [
        (0.98, 150.0, 8000.0),
        (1.00, 170.0, 10000.0),
        (1.02, 175.0, 12000.0),
        (1.05, 165.0, 10000.0),
    ]


def save_vr_near_crossing_plot() -> None:
    controller = ControllerSpec("phase18_crossing_targeting", "crossing_targeting")
    cases = diagnostic_cases()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for r0, angle, thrust in cases:
        row = run_trace(controller, r0, angle, thrust)
        traj = row["trajectory"]
        target_radius = float(traj["target_radius"])
        v_circ_trace = math.sqrt(MU / target_radius)
        radius_error = np.asarray(traj["radius_error_ratio"], dtype=np.float64)
        vr_values = np.asarray(traj["radial_velocity"], dtype=np.float64) / (v_circ_trace + 1.0e-12)
        if len(radius_error) == 0:
            continue
        idx = int(np.argmin(np.abs(radius_error)))
        lo = max(0, idx - 800)
        hi = min(len(vr_values), idx + 801)
        local_steps = np.arange(lo, hi, dtype=np.float64) - idx
        ax.plot(local_steps * DT / 3600.0, vr_values[lo:hi], linewidth=1.1, label=f"r0={r0:g}, a={angle:g}, T={thrust:g}")
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlabel("hours from closest radius approach")
    ax.set_ylabel("v_r / v_circular")
    ax.set_title("Phase 18 radial velocity near crossing")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(VR_NEAR_CROSSING_PNG, dpi=220)
    plt.close(fig)


def save_vt_error_plot() -> None:
    controller = ControllerSpec("phase18_crossing_targeting", "crossing_targeting")
    cases = diagnostic_cases()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for r0, angle, thrust in cases:
        row = run_trace(controller, r0, angle, thrust)
        traj = row["trajectory"]
        target_radius = float(traj["target_radius"])
        v_circ_trace = math.sqrt(MU / target_radius)
        time_days = np.arange(len(traj["vt_error"]), dtype=np.float64) * DT / 86400.0
        vt_error = np.asarray(traj["vt_error"], dtype=np.float64) / (v_circ_trace + 1.0e-12)
        ax.plot(time_days, vt_error, linewidth=1.1, label=f"r0={r0:g}, a={angle:g}, T={thrust:g}")
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlabel("time [days]")
    ax.set_ylabel("(v_t - v_circular) / v_circular")
    ax.set_title("Phase 18 tangential velocity error")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(VT_ERROR_PNG, dpi=220)
    plt.close(fig)


def save_capture_crossing_comparison(rows: Sequence[Dict[str, object]]) -> None:
    names = ["baseline_soft_linear_3e4", "phase19_minimal_transfer", "phase20_predictive_planner"]
    labels = ["Phase 7.6", "Phase 19", "Phase 20"]
    capture_counts = []
    success_counts = []
    crossing_counts = []
    recoverable_counts = []
    for name in names:
        subset = [row for row in rows if str(row["controller_name"]) == name]
        capture_counts.append(sum(bool(row["capture_entered"]) for row in subset))
        success_counts.append(sum(bool(row["success"]) for row in subset))
        crossing_counts.append(sum(bool(row["crossing_occurs"]) for row in subset))
        recoverable_counts.append(sum(bool(row["recoverable_crossing"]) for row in subset))

    x = np.arange(len(labels), dtype=np.float64)
    width = 0.20
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.bar(x - 1.5 * width, crossing_counts, width, label="crossing cases", color="#80B1D3")
    ax.bar(x - 0.5 * width, recoverable_counts, width, label="recoverable crossings", color="#1B9E77")
    ax.bar(x + 0.5 * width, capture_counts, width, label="CAPTURE", color="#FDB462")
    ax.bar(x + 1.5 * width, success_counts, width, label="success", color="#386CB0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("cases out of 48")
    ax.set_title("Phase 20 crossing and capture comparison")
    ax.set_ylim(0, 48)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CAPTURE_CROSSING_PNG, dpi=220)
    plt.close(fig)


def write_summary(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, int]], planner_diagnostics: Dict[str, object]) -> None:
    baseline = stats["baseline_soft_linear_3e4"]
    phase19 = stats["phase19_minimal_transfer"]
    planner = stats["phase20_predictive_planner"]
    baseline_rows = [row for row in rows if str(row["controller_name"]) == "baseline_soft_linear_3e4"]
    phase19_rows = [row for row in rows if str(row["controller_name"]) == "phase19_minimal_transfer"]
    planner_rows = [row for row in rows if str(row["controller_name"]) == "phase20_predictive_planner"]
    baseline_crossings = sum(int(float(row["radius_crossings_total"])) for row in baseline_rows)
    phase19_crossings = sum(int(float(row["radius_crossings_total"])) for row in phase19_rows)
    planner_crossings = sum(int(float(row["radius_crossings_total"])) for row in planner_rows)
    baseline_crossing_cases = sum(bool(row["crossing_occurs"]) for row in baseline_rows)
    phase19_crossing_cases = sum(bool(row["crossing_occurs"]) for row in phase19_rows)
    planner_crossing_cases = sum(bool(row["crossing_occurs"]) for row in planner_rows)
    baseline_recoverable = sum(bool(row["recoverable_crossing"]) for row in baseline_rows)
    phase19_recoverable = sum(bool(row["recoverable_crossing"]) for row in phase19_rows)
    planner_recoverable = sum(bool(row["recoverable_crossing"]) for row in planner_rows)
    capture_improvement = planner["capture"] - baseline["capture"]
    success_improvement = planner["success"] - baseline["success"]
    crossing_case_improvement = planner_crossing_cases - baseline_crossing_cases
    capture_improvement_vs_phase19 = planner["capture"] - phase19["capture"]
    success_improvement_vs_phase19 = planner["success"] - phase19["success"]
    crossing_case_improvement_vs_phase19 = planner_crossing_cases - phase19_crossing_cases
    baseline_overspeed = sum(str(row["termination_reason"]) == "overspeed" for row in baseline_rows)
    phase19_overspeed = sum(str(row["termination_reason"]) == "overspeed" for row in phase19_rows)
    planner_overspeed = sum(str(row["termination_reason"]) == "overspeed" for row in planner_rows)
    coast_steps = sum(int(float(row["coast_steps"])) for row in planner_rows)
    predictive_replans_total = sum(int(float(row["predictive_replans"])) for row in planner_rows)
    candidate_eval_total = sum(int(float(row["candidate_eval_count"])) for row in planner_rows)
    predictive_steps = 0
    handoff_steps = 0
    for row in planner_rows:
        counts = json.loads(str(row.get("planner_mode_counts_json", "{}")))
        predictive_steps += int(counts.get("predictive_descent", 0))
        handoff_steps += int(counts.get("phase76_handoff", 0))

    def mean_numeric(key: str, subset: Sequence[Dict[str, object]]) -> float:
        values = []
        for row in subset:
            value = row.get(key, "")
            if value == "":
                continue
            values.append(float(value))
        return float(np.mean(values)) if values else float("nan")

    baseline_crossing_score = mean_numeric("crossing_score", baseline_rows)
    planner_crossing_score = mean_numeric("crossing_score", planner_rows)
    baseline_recoverability = mean_numeric("recoverability_score", baseline_rows)
    planner_recoverability = mean_numeric("recoverability_score", planner_rows)
    improves_recoverable = planner_recoverable > baseline_recoverable
    improves_capture = capture_improvement > 0
    outperforms_reactive = capture_improvement > 0 or success_improvement > 0
    lines = [
        "# Phase 20 Predictive Crossing-State Planning",
        "",
        "## Scope",
        "",
        "- 2D Python-only explicit short-horizon predictive planner.",
        "- Physics, CAPTURE/LOCK logic, and success definition are unchanged.",
        f"- DESCENT replans every `{REPLAN_INTERVAL}` steps over a `{PREDICTIVE_HORIZON}`-step short horizon.",
        "- Candidate actions include coast, prograde, retrograde, radial, and diagonal low-thrust actions.",
        "- Scoring uses target-radius crossing quality, post-cross recoverability, and safety penalties.",
        "- Near the target radius, the planner hands off to the existing Phase 7.6 `soft_linear_3e4` controller.",
        "",
        "## Reduced-Grid Result",
        "",
        "| Controller | CAPTURE | Success | Crossing cases | Recoverable crossings | Overspeed | Total |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| `baseline_soft_linear_3e4` | {baseline['capture']} | {baseline['success']} | {baseline_crossing_cases} | {baseline_recoverable} | {baseline_overspeed} | {baseline['total']} |",
        f"| `phase19_minimal_transfer` | {phase19['capture']} | {phase19['success']} | {phase19_crossing_cases} | {phase19_recoverable} | {phase19_overspeed} | {phase19['total']} |",
        f"| `phase20_predictive_planner` | {planner['capture']} | {planner['success']} | {planner_crossing_cases} | {planner_recoverable} | {planner_overspeed} | {planner['total']} |",
        "",
        f"- Capture improvement vs baseline: `{capture_improvement}`.",
        f"- Success improvement vs baseline: `{success_improvement}`.",
        f"- Crossing-case improvement vs baseline: `{crossing_case_improvement}`.",
        f"- Recoverable-crossing improvement vs baseline: `{planner_recoverable - baseline_recoverable}`.",
        f"- Capture improvement vs Phase 19: `{capture_improvement_vs_phase19}`.",
        f"- Success improvement vs Phase 19: `{success_improvement_vs_phase19}`.",
        f"- Crossing-case improvement vs Phase 19: `{crossing_case_improvement_vs_phase19}`.",
        f"- Total radius crossings: baseline `{baseline_crossings}`, Phase 19 `{phase19_crossings}`, Phase 20 `{planner_crossings}`.",
        f"- Overspeed terminations: baseline `{baseline_overspeed}`, Phase 19 `{phase19_overspeed}`, Phase 20 `{planner_overspeed}`.",
        f"- Mean crossing score: baseline `{baseline_crossing_score:.4f}`, Phase 20 `{planner_crossing_score:.4f}`.",
        f"- Mean recoverability score: baseline `{baseline_recoverability:.4f}`, Phase 20 `{planner_recoverability:.4f}`.",
        f"- Phase 20 replans: `{predictive_replans_total}`.",
        f"- Phase 20 candidate simulations: `{candidate_eval_total}`.",
        f"- Phase 20 predictive steps: `{predictive_steps}`.",
        f"- Phase 20 coast steps: `{coast_steps}`.",
        f"- Phase 20 Phase 7.6 handoff steps: `{handoff_steps}`.",
        f"- Phase 20 mode usage: `{json.dumps(planner_diagnostics.get('descent_step_counts', {}), sort_keys=True)}`.",
        "",
        "## Answers",
        "",
        f"1. Can predictive short-horizon search produce more recoverable crossings than reactive local control? `{'yes' if improves_recoverable else 'no'}`. Recoverable crossings change from `{baseline_recoverable}` to `{planner_recoverable}`.",
        "2. Does recoverability-aware planning outperform crossing-only planning? `not directly tested`. This script implements recoverability-aware scoring but does not include a crossing-only ablation.",
        f"3. Can planning improve no-CAPTURE scenarios without degrading stable cases? `{'yes' if improves_capture and planner['success'] >= baseline['success'] else 'no'}`. CAPTURE changes by `{capture_improvement}`, success changes by `{success_improvement}`.",
        f"4. Does planning outperform reactive control? `{'yes' if outperforms_reactive else 'no'}` on this reduced grid.",
        "",
        "## Interpretation",
        "",
        "Phase 20 is the first guidance-layer test that chooses actions by forecasting future crossing quality and post-cross recoverability. A positive result requires more than crossing the target radius; the planner must create states that the Phase 7.6 handoff can exploit.",
        "",
        "## Artifacts",
        "",
        "- `reduced_grid_results.csv`",
        "- `summary.md`",
        "- `capture_crossing_comparison.png`",
        "- `trajectories/*.png`",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def numeric_mean(rows: Sequence[Dict[str, object]], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        values.append(float(value))
    return float(np.mean(values)) if values else 0.0


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def aggregate_by_variant(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    aggregates: List[Dict[str, object]] = []
    for variant_name in sorted({str(row["variant_name"]) for row in rows}):
        subset = [row for row in rows if str(row["variant_name"]) == variant_name]
        first = subset[0]
        choice_totals: Counter[str] = Counter()
        total_replans = 0
        weighted_action_norm = 0.0
        weighted_score_margin = 0.0
        for row in subset:
            counts = json.loads(str(row.get("planner_choice_counts_json", "{}") or "{}"))
            for key, value in counts.items():
                choice_totals[key] += int(value)
            replans = int(float(row["predictive_replans"]))
            total_replans += replans
            weighted_action_norm += float(row["avg_chosen_thrust_norm"]) * replans
            weighted_score_margin += float(row["avg_score_margin"]) * replans
        total_choices = sum(choice_totals.values())

        def aggregate_choice_pct(category: str) -> float:
            if total_choices <= 0:
                return 0.0
            return 100.0 * float(choice_totals.get(category, 0)) / float(total_choices)

        aggregates.append(
            {
                "variant_name": variant_name,
                "sweep": first["sweep"],
                "predictive_horizon": int(first["predictive_horizon"]),
                "replan_interval": int(first["replan_interval"]),
                "score_mode": first["score_mode"],
                "action_space": first["action_space"],
                "burn_hold_steps": int(first["burn_hold_steps"]),
                "total": len(subset),
                "crossing_cases": sum(as_bool(row["crossing_occurs"]) for row in subset),
                "recoverable_crossings": sum(as_bool(row["recoverable_crossing"]) for row in subset),
                "capture": sum(as_bool(row["capture_entered"]) for row in subset),
                "success": sum(as_bool(row["success"]) for row in subset),
                "overspeed": sum(as_bool(row["overspeed"]) for row in subset),
                "instability": sum(str(row["termination_reason"]) in {"out_range", "too_close", "radial_stall"} for row in subset),
                "mean_crossing_score": numeric_mean(subset, "crossing_score"),
                "mean_recoverability": numeric_mean(subset, "recoverability_score"),
                "mean_action_norm": weighted_action_norm / float(total_replans) if total_replans else 0.0,
                "mean_score_margin": weighted_score_margin / float(total_replans) if total_replans else 0.0,
                "mean_runtime_seconds": numeric_mean(subset, "runtime_seconds"),
                "runtime_seconds_total": sum(float(row["runtime_seconds"]) for row in subset),
                "candidate_eval_count": sum(int(row["candidate_eval_count"]) for row in subset),
                "predictive_replans": total_replans,
                "coast_chosen_pct": aggregate_choice_pct("coast"),
                "prograde_pct": aggregate_choice_pct("prograde"),
                "retrograde_pct": aggregate_choice_pct("retrograde"),
                "radial_inward_pct": aggregate_choice_pct("radial_inward"),
                "radial_outward_pct": aggregate_choice_pct("radial_outward"),
                "diagonal_pct": aggregate_choice_pct("diagonal"),
            }
        )
    order = {config.name: idx for idx, config in enumerate(define_ablation_configs())}
    return sorted(aggregates, key=lambda row: order.get(str(row["variant_name"]), 999))


def metric_score(row: Dict[str, object]) -> int:
    return (
        4 * int(row["success"])
        + 3 * int(row["capture"])
        + 2 * int(row["recoverable_crossings"])
        + int(row["crossing_cases"])
    )


def bottleneck_classification(aggregates: Sequence[Dict[str, object]]) -> tuple[str, str]:
    baseline = next(row for row in aggregates if str(row["variant_name"]) == "baseline_h260_r40_recoverability_base")
    baseline_score = metric_score(baseline)
    best_by_group: Dict[str, int] = {"Horizon bottleneck": 0, "Action bottleneck": 0, "Objective bottleneck": 0}
    for row in aggregates:
        improvement = metric_score(row) - baseline_score
        if str(row["sweep"]) == "horizon":
            best_by_group["Horizon bottleneck"] = max(best_by_group["Horizon bottleneck"], improvement)
        elif str(row["sweep"]) == "action_space":
            best_by_group["Action bottleneck"] = max(best_by_group["Action bottleneck"], improvement)
        elif str(row["sweep"]) == "score":
            best_by_group["Objective bottleneck"] = max(best_by_group["Objective bottleneck"], improvement)
    best_label, best_improvement = max(best_by_group.items(), key=lambda item: item[1])
    if best_improvement > 0:
        return best_label, f"{best_label} produced the largest positive composite metric gain over the Phase 20 baseline."
    return (
        "Structural orbital-class bottleneck",
        "No tested horizon, score, replan, or expanded local-action variant improved the composite reachability metrics over the Phase 20 baseline.",
    )


def save_sweep_plot(
    aggregates: Sequence[Dict[str, object]],
    sweep_name: str,
    x_key: str,
    output_path: Path,
    title: str,
) -> None:
    selected = [row for row in aggregates if str(row["sweep"]) == sweep_name]
    baseline = next(row for row in aggregates if str(row["variant_name"]) == "baseline_h260_r40_recoverability_base")
    selected = [baseline] + selected
    selected = sorted(selected, key=lambda row: int(row[x_key]))
    x = np.arange(len(selected), dtype=np.float64)
    labels = [str(int(row[x_key])) for row in selected]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for key, label, color in [
        ("crossing_cases", "crossing cases", "#80B1D3"),
        ("recoverable_crossings", "recoverable crossings", "#1B9E77"),
        ("capture", "CAPTURE", "#FDB462"),
        ("success", "success", "#386CB0"),
        ("overspeed", "overspeed", "#D95F02"),
    ]:
        ax.plot(x, [int(row[key]) for row in selected], marker="o", linewidth=1.5, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(x_key)
    ax.set_ylabel("cases out of 48")
    ax.set_ylim(0, 48)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_action_distribution_plot(aggregates: Sequence[Dict[str, object]]) -> None:
    labels = [str(row["variant_name"]).replace("_", "\n") for row in aggregates]
    x = np.arange(len(aggregates), dtype=np.float64)
    bottoms = np.zeros(len(aggregates), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    for key, label, color in [
        ("coast_chosen_pct", "coast", "#BDBDBD"),
        ("prograde_pct", "prograde", "#386CB0"),
        ("retrograde_pct", "retrograde", "#D95F02"),
        ("radial_inward_pct", "radial inward", "#1B9E77"),
        ("radial_outward_pct", "radial outward", "#7570B3"),
        ("diagonal_pct", "diagonal", "#FDB462"),
    ]:
        values = np.asarray([float(row[key]) for row in aggregates], dtype=np.float64)
        ax.bar(x, values, bottom=bottoms, label=label, color=color)
        bottoms += values
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("planner choices at replans (%)")
    ax.set_title("Phase 20.5 planner action distribution")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncols=3)
    fig.tight_layout()
    fig.savefig(ACTION_DISTRIBUTION_PNG, dpi=220)
    plt.close(fig)


def save_score_ablation_plot(aggregates: Sequence[Dict[str, object]]) -> None:
    names = ["baseline_h260_r40_recoverability_base", "score_crossing_only"]
    selected = [row for name in names for row in aggregates if str(row["variant_name"]) == name]
    x = np.arange(len(selected), dtype=np.float64)
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for offset, key, label, color in [
        (-1.5, "crossing_cases", "crossing cases", "#80B1D3"),
        (-0.5, "recoverable_crossings", "recoverable crossings", "#1B9E77"),
        (0.5, "capture", "CAPTURE", "#FDB462"),
        (1.5, "success", "success", "#386CB0"),
    ]:
        ax.bar(x + offset * width, [int(row[key]) for row in selected], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(["recoverability-aware", "crossing-only"])
    ax.set_ylabel("cases out of 48")
    ax.set_ylim(0, 48)
    ax.set_title("Phase 20.5 score ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SCORE_ABLATION_PNG, dpi=220)
    plt.close(fig)


def write_phase20_5_summary(aggregates: Sequence[Dict[str, object]]) -> None:
    baseline = next(row for row in aggregates if str(row["variant_name"]) == "baseline_h260_r40_recoverability_base")
    bottleneck, bottleneck_reason = bottleneck_classification(aggregates)

    def table_rows(selected: Sequence[Dict[str, object]]) -> List[str]:
        lines = [
            "| Variant | Crossing | Recoverable | CAPTURE | Success | Overspeed | Mean action norm | Coast % | Runtime s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in selected:
            lines.append(
                f"| `{row['variant_name']}` | {row['crossing_cases']} | {row['recoverable_crossings']} | "
                f"{row['capture']} | {row['success']} | {row['overspeed']} | "
                f"{float(row['mean_action_norm']):.4f} | {float(row['coast_chosen_pct']):.1f} | "
                f"{float(row['runtime_seconds_total']):.1f} |"
            )
        return lines

    horizon_rows = [baseline] + [row for row in aggregates if str(row["sweep"]) == "horizon"]
    replan_rows = [baseline] + [row for row in aggregates if str(row["sweep"]) == "replan_interval"]
    score_rows = [baseline] + [row for row in aggregates if str(row["sweep"]) == "score"]
    action_rows = [baseline] + [row for row in aggregates if str(row["sweep"]) == "action_space"]
    max_horizon = max(horizon_rows, key=metric_score)
    max_action = max(action_rows, key=metric_score)
    score_crossing = score_rows[-1]
    lines = [
        "# Phase 20.5 Predictive Planner Failure Diagnosis",
        "",
        "## Structural Conclusion",
        "",
        f"- Dominant bottleneck: `{bottleneck}`.",
        f"- Reason: {bottleneck_reason}",
        f"- Phase 20 baseline: crossing `{baseline['crossing_cases']}`, recoverable `{baseline['recoverable_crossings']}`, CAPTURE `{baseline['capture']}`, success `{baseline['success']}`, overspeed `{baseline['overspeed']}`.",
        "",
        "## Research Answers",
        "",
        f"1. Does longer horizon create new reachability? `{'yes' if metric_score(max_horizon) > metric_score(baseline) else 'no'}`. Best horizon variant is `{max_horizon['variant_name']}`.",
        f"2. Does planner mostly choose coast? `{'yes' if float(baseline['coast_chosen_pct']) >= 50.0 else 'no'}`. Baseline coast choice rate is `{float(baseline['coast_chosen_pct']):.1f}%`.",
        f"3. Does recoverability scoring suppress useful aggressive transfers? `{'yes' if metric_score(score_crossing) > metric_score(baseline) else 'no'}`. Crossing-only score composite change is `{metric_score(score_crossing) - metric_score(baseline)}`.",
        f"4. Is predictive local search fundamentally incapable of orbital-class transition? `{'yes' if bottleneck == 'Structural orbital-class bottleneck' else 'not under the strongest observed ablation'}`.",
        "",
        "## Horizon Sweep",
        "",
        *table_rows(horizon_rows),
        "",
        "## Replan Interval Sweep",
        "",
        *table_rows(replan_rows),
        "",
        "## Score Ablation",
        "",
        *table_rows(score_rows),
        "",
        "## Action Space Ablation",
        "",
        *table_rows(action_rows),
        "",
        "## Action Distribution",
        "",
        "| Variant | Coast % | Prograde % | Retrograde % | Radial inward % | Radial outward % | Diagonal % | Mean margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| `{row['variant_name']}` | {float(row['coast_chosen_pct']):.1f} | {float(row['prograde_pct']):.1f} | "
            f"{float(row['retrograde_pct']):.1f} | {float(row['radial_inward_pct']):.1f} | "
            f"{float(row['radial_outward_pct']):.1f} | {float(row['diagonal_pct']):.1f} | "
            f"{float(row['mean_score_margin']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `ablation_results.csv`",
            "- `horizon_sweep.png`",
            "- `replan_interval_sweep.png`",
            "- `action_distribution.png`",
            "- `score_ablation_comparison.png`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def print_case_progress(done: int, total: int, row: Dict[str, object]) -> None:
    print(
        f"phase20.5 {done}/{total} variant={row['variant_name']} r0={float(row['r0_over_target']):g} "
        f"angle={float(row['initial_velocity_angle_deg']):g} thrust={float(row['thrust_scale']):g} "
        f"capture={row['capture_entered']} success={row['success']} reason={row['termination_reason']} "
        f"crossings={row['radius_crossings_total']} recoverable={row['recoverable_crossing']} "
        f"coast_choice={float(row['coast_chosen_pct']):.1f}% norm={float(row['avg_chosen_thrust_norm']):.4f} "
        f"runtime={float(row['runtime_seconds']):.2f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 20.5 predictive-planner ablation benchmark.")
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = list(iter_cases(define_controllers()))
    if args.no_resume and COMPARISON_CSV.exists():
        COMPARISON_CSV.unlink()
    rows = read_comparison_rows()
    completed = {case_key(row) for row in rows}
    remaining = [case for case in cases if case_key(case) not in completed]
    print(
        f"phase20.5 starting total_cases={len(cases)} completed={len(rows)} "
        f"remaining={len(remaining)} workers={max(1, int(args.workers))}",
        flush=True,
    )
    completed_count = len(rows)
    if remaining:
        if int(args.workers) <= 1:
            for case in remaining:
                row = run_case_tuple(case)
                append_comparison_row(row)
                rows.append(row)
                completed_count += 1
                print_case_progress(completed_count, len(cases), row)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
                future_to_case = {executor.submit(run_case_tuple, case): case for case in remaining}
                for future in concurrent.futures.as_completed(future_to_case):
                    row = future.result()
                    append_comparison_row(row)
                    rows.append(row)
                    completed_count += 1
                    print_case_progress(completed_count, len(cases), row)
    aggregates = aggregate_by_variant(rows)
    save_sweep_plot(aggregates, "horizon", "predictive_horizon", HORIZON_SWEEP_PNG, "Phase 20.5 horizon sweep")
    save_sweep_plot(
        aggregates,
        "replan_interval",
        "replan_interval",
        REPLAN_INTERVAL_SWEEP_PNG,
        "Phase 20.5 replan interval sweep",
    )
    save_action_distribution_plot(aggregates)
    save_score_ablation_plot(aggregates)
    write_phase20_5_summary(aggregates)
    print(f"Saved Phase 20.5 ablation outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
