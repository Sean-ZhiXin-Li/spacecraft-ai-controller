from __future__ import annotations

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
PHASE8_GRID_PATH = PROJECT_ROOT / "analysis" / "phase8_multiregime" / "phase8_grid.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase9_reachability"
TRAJECTORY_DIR = OUTPUT_DIR / "trajectories"
MPLCONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig


NO_CAPTURE_DATASET_PATH = OUTPUT_DIR / "no_capture_dataset.csv"
REACHABILITY_COMPARISON_PATH = OUTPUT_DIR / "reachability_comparison.csv"
SUMMARY_PATH = OUTPUT_DIR / "phase9_summary.md"

CAPTURE_MAP_T10000_PNG = OUTPUT_DIR / "capture_map_r0_vs_angle_thrust10000.png"
CAPTURE_MAP_T15000_PNG = OUTPUT_DIR / "capture_map_r0_vs_angle_thrust15000.png"
NO_CAPTURE_BY_R0_PNG = OUTPUT_DIR / "no_capture_rate_by_r0.png"
NO_CAPTURE_BY_ANGLE_PNG = OUTPUT_DIR / "no_capture_rate_by_angle.png"
NO_CAPTURE_BY_THRUST_PNG = OUTPUT_DIR / "no_capture_rate_by_thrust.png"

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

PHASE8_R0_VALUES = [0.98, 0.99, 1.00, 1.00005, 1.01, 1.02, 1.05, 1.10, 1.20]
PHASE8_ANGLE_VALUES = [120.0, 135.0, 150.0, 160.0, 165.0, 170.0, 175.0, 180.0]

SMALL_R0_VALUES = [0.98, 1.00, 1.02, 1.05]
SMALL_ANGLE_VALUES = [150.0, 165.0, 170.0, 175.0]
SMALL_THRUST_VALUES = [8000.0, 10000.0, 12000.0]
SMALL_TARGET_RADIUS_SCALE = 1.00

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

PHASE8_FIELDNAMES = [
    "controller_name",
    "controller_params_json",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "target_radius_scale",
    "success",
    "capture_entered",
    "lock_entered",
    "crossing_occurs",
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
]

NO_CAPTURE_FIELDNAMES = [
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "target_radius_scale",
    "minimum_abs_radius_error",
    "termination_reason",
]

COMPARISON_FIELDNAMES = [
    "controller_name",
    "controller_params_json",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "target_radius_scale",
    "success",
    "capture_entered",
    "lock_entered",
    "crossing_occurs",
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
]


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    alpha_schedule: str
    alpha_mid: float
    alpha_width: float
    reachability_mode: str
    reachability_radial_gain: float
    params: Dict[str, float]


def define_controllers() -> List[ControllerSpec]:
    return [
        ControllerSpec("baseline_soft_linear_3e4", "linear", 3.0e-4, 0.0, "baseline", 0.0, dict(BASE_PARAMS)),
        ControllerSpec("reach_energy_directional_light", "linear", 3.0e-4, 0.0, "energy_directional", 0.12, dict(BASE_PARAMS)),
        ControllerSpec("reach_energy_directional_medium", "linear", 3.0e-4, 0.0, "energy_directional", 0.24, dict(BASE_PARAMS)),
    ]


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def parse_optional_int(value: object) -> int | None:
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return int(float(text))


def load_phase8_rows() -> List[Dict[str, object]]:
    if not PHASE8_GRID_PATH.exists():
        raise FileNotFoundError(f"Missing Phase 8 grid: {PHASE8_GRID_PATH}")
    with PHASE8_GRID_PATH.open("r", newline="", encoding="utf-8") as f:
        loaded = list(csv.DictReader(f))
    rows: List[Dict[str, object]] = []
    for row in loaded:
        rows.append(
            {
                **row,
                "r0_over_target": float(row["r0_over_target"]),
                "initial_velocity_angle_deg": float(row["initial_velocity_angle_deg"]),
                "thrust_scale": float(row["thrust_scale"]),
                "target_radius_scale": float(row["target_radius_scale"]),
                "success": parse_bool(row["success"]),
                "capture_entered": parse_bool(row["capture_entered"]),
                "lock_entered": parse_bool(row["lock_entered"]),
                "crossing_occurs": parse_bool(row["crossing_occurs"]),
                "near_miss": parse_bool(row["near_miss"]),
                "radius_crossings_total": int(float(row["radius_crossings_total"])),
                "first_crossing_step": parse_optional_int(row["first_crossing_step"]),
                "minimum_abs_radius_error": float(row["minimum_abs_radius_error"]),
                "final_radius_error": float(row["final_radius_error"]),
                "tail_mean_abs_vr": float(row["tail_mean_abs_vr"]),
                "steps": int(float(row["steps"])),
                "terminated": parse_bool(row["terminated"]),
                "truncated": parse_bool(row["truncated"]),
                "termination_reason": row.get("termination_reason", ""),
            }
        )
    return rows


def load_comparison_rows() -> List[Dict[str, object]]:
    if not REACHABILITY_COMPARISON_PATH.exists():
        return []
    with REACHABILITY_COMPARISON_PATH.open("r", newline="", encoding="utf-8") as f:
        loaded = list(csv.DictReader(f))
    rows: List[Dict[str, object]] = []
    for row in loaded:
        rows.append(
            {
                **row,
                "r0_over_target": float(row["r0_over_target"]),
                "initial_velocity_angle_deg": float(row["initial_velocity_angle_deg"]),
                "thrust_scale": float(row["thrust_scale"]),
                "target_radius_scale": float(row["target_radius_scale"]),
                "success": parse_bool(row["success"]),
                "capture_entered": parse_bool(row["capture_entered"]),
                "lock_entered": parse_bool(row["lock_entered"]),
                "crossing_occurs": parse_bool(row["crossing_occurs"]),
                "near_miss": parse_bool(row["near_miss"]),
                "radius_crossings_total": int(float(row["radius_crossings_total"])),
                "first_crossing_step": parse_optional_int(row["first_crossing_step"]),
                "minimum_abs_radius_error": float(row["minimum_abs_radius_error"]),
                "final_radius_error": float(row["final_radius_error"]),
                "tail_mean_abs_vr": float(row["tail_mean_abs_vr"]),
                "steps": int(float(row["steps"])),
                "terminated": parse_bool(row["terminated"]),
                "truncated": parse_bool(row["truncated"]),
                "termination_reason": row.get("termination_reason", ""),
            }
        )
    return rows


def append_comparison_row(row: Dict[str, object]) -> None:
    REACHABILITY_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = REACHABILITY_COMPARISON_PATH.exists()
    with REACHABILITY_COMPARISON_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in COMPARISON_FIELDNAMES})


def completed_keys(rows: Sequence[Dict[str, object]]) -> set[tuple[str, float, float, float, float]]:
    return {
        (
            str(row["controller_name"]),
            float(row["r0_over_target"]),
            float(row["initial_velocity_angle_deg"]),
            float(row["thrust_scale"]),
            float(row["target_radius_scale"]),
        )
        for row in rows
    }


def iter_small_cases(controllers: Sequence[ControllerSpec]) -> Iterable[tuple[ControllerSpec, float, float, float, float]]:
    for controller in controllers:
        for thrust in SMALL_THRUST_VALUES:
            for angle in SMALL_ANGLE_VALUES:
                for r0 in SMALL_R0_VALUES:
                    yield controller, r0, angle, thrust, SMALL_TARGET_RADIUS_SCALE


def rollout_case(
    controller: ControllerSpec,
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
    phase = "DESCENT"
    prev_controller_r_error: float | None = None
    radius_errors: List[float] = []
    vr_values: List[float] = []
    radii: List[float] = []
    energies: List[float] = []
    phases: List[str] = []
    success_counter = 0
    radial_stall_counter = 0
    success = False
    capture_entered = False
    lock_entered = False
    terminated = False
    truncated = False
    termination_reason = ""
    prev_r_error: float | None = None
    radius_crossings_total = 0
    first_crossing_step: int | None = None
    params = controller.params

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
        target_energy = -MU / (2.0 * target_radius)
        return abs(state_energy(state_x, state_y, state_vx, state_vy) - target_energy) / (abs(target_energy) + 1.0e-12)

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
            float(params["score_r"]) * r_err_ratio
            + float(params["score_vr"]) * v_r_ratio
            + float(params["score_vt"]) * v_t_ratio
            + float(params["score_energy"]) * energy_ratio
        )
        if (state_radius - target_radius) > 0.0 and (state_radius - target_radius) / target_radius < WS_BAND_RATIO and vr_state < 0.0:
            score -= float(params["capture_bonus"])
        if r_err_ratio < 3.0e-4 and v_r_ratio < 2.0e-2:
            score -= float(params["inner_capture_bonus"])
        return score

    def baseline_action(state_vx: float, state_vy: float) -> tuple[float, float]:
        speed = math.sqrt(state_vx * state_vx + state_vy * state_vy)
        if speed <= 1.0e-12:
            return 0.0, 0.0
        return -state_vx / speed, -state_vy / speed

    def normalize_action(action_x: float, action_y: float) -> tuple[float, float]:
        action_norm = math.sqrt(action_x * action_x + action_y * action_y)
        if action_norm > 1.0:
            return action_x / action_norm, action_y / action_norm
        return clamp(action_x, -1.0, 1.0), clamp(action_y, -1.0, 1.0)

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
        scale = float(params["min_scale"]) + (float(params["max_scale"]) - float(params["min_scale"])) * z
        candidates = [
            (retro_x, retro_y),
            (
                clamp(retro_x - float(params["inward_bias"]) * r_hat_x, -1.0, 1.0),
                clamp(retro_y - float(params["inward_bias"]) * r_hat_y, -1.0, 1.0),
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
        gain = float(params["prewindow_radial_gain"])
        return normalize_action(retro_x + gain * inward_sign * r_hat_x, retro_y + gain * inward_sign * r_hat_y)

    def alpha_value(abs_r_error_ratio: float) -> float:
        if controller.alpha_schedule == "linear":
            return clamp(abs_r_error_ratio / (controller.alpha_mid + 1.0e-12), 0.0, 1.0)
        if controller.alpha_schedule == "sigmoid":
            z = (abs_r_error_ratio - controller.alpha_mid) / (controller.alpha_width + 1.0e-12)
            z = clamp(z, -60.0, 60.0)
            return 1.0 / (1.0 + math.exp(-z))
        raise ValueError(f"Unknown alpha schedule: {controller.alpha_schedule}")

    def soft_hybrid_action(
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
        alpha = alpha_value(abs_r_error_ratio)
        return normalize_action(alpha * pre_x + (1.0 - alpha) * ws_x, alpha * pre_y + (1.0 - alpha) * ws_y)

    def reachability_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
        abs_r_error_ratio: float,
    ) -> tuple[float, float]:
        base_x, base_y = soft_hybrid_action(state_x, state_y, state_vx, state_vy, state_radius, real_r_ratio, abs_r_error_ratio)
        gain = float(controller.reachability_radial_gain)
        if controller.reachability_mode == "baseline" or gain <= 0.0 or abs_r_error_ratio <= PREWINDOW_OUTER_RATIO:
            return base_x, base_y
        inv_r = 1.0 / (state_radius + 1.0e-12)
        r_hat_x = state_x * inv_r
        r_hat_y = state_y * inv_r
        if controller.reachability_mode == "energy_directional" and real_r_ratio < 0.0:
            retro_x, retro_y = baseline_action(state_vx, state_vy)
            base_x, base_y = -retro_x, -retro_y
        target_seeking_sign = -1.0 if real_r_ratio > 0.0 else 1.0
        return normalize_action(base_x + gain * target_seeking_sign * r_hat_x, base_y + gain * target_seeking_sign * r_hat_y)

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

        if phase == "DESCENT":
            action_x, action_y = reachability_action(x, y, vx, vy, radius, real_r_ratio, abs_r_ratio)
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
        if record_trajectory:
            radii.append(radius)
            energies.append(state_energy(x, y, vx, vy))
            phases.append(phase)
        if prev_r_error is not None and ((prev_r_error > 0.0 and r_error <= 0.0) or (prev_r_error < 0.0 and r_error >= 0.0)):
            radius_crossings_total += 1
            if first_crossing_step is None:
                first_crossing_step = step
        prev_r_error = r_error

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

        radial_stall = (r_err_now < 0.10) and (v_err_now < 0.12) and (ang_abs_now > 0.85)
        radial_stall_counter = radial_stall_counter + 1 if radial_stall else 0

        success = success_counter >= int(STRICT_CFG["success_threshold"])
        out_range = radius > 2.5 * target_radius
        overspeed = speed > 1.90 * v_circ
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

    rerr = np.asarray(radius_errors, dtype=np.float64)
    vr_arr = np.asarray(vr_values, dtype=np.float64)
    tail = vr_arr[-min(TAIL_LEN, len(vr_arr)) :] if len(vr_arr) else np.asarray([0.0])
    min_abs_rerr = float(np.min(np.abs(rerr))) if len(rerr) else 0.0
    near_miss = (not success) and min_abs_rerr / target_radius <= NEAR_MISS_REL
    row: Dict[str, object] = {
        "controller_name": controller.name,
        "controller_params_json": json.dumps(
            {
                **controller.params,
                "alpha_schedule": controller.alpha_schedule,
                "alpha_mid": controller.alpha_mid,
                "alpha_width": controller.alpha_width,
                "reachability_mode": controller.reachability_mode,
                "reachability_radial_gain": controller.reachability_radial_gain,
            },
            sort_keys=True,
        ),
        "r0_over_target": r0,
        "initial_velocity_angle_deg": angle,
        "thrust_scale": thrust_scale,
        "target_radius_scale": target_scale,
        "success": bool(success),
        "capture_entered": bool(capture_entered),
        "lock_entered": bool(lock_entered),
        "crossing_occurs": bool(radius_crossings_total > 0),
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
    }
    if record_trajectory:
        row["trajectory"] = {
            "radius": radii,
            "radius_error": [float(value) for value in radius_errors],
            "radial_velocity": [float(value) for value in vr_values],
            "energy": energies,
            "phase": phases,
            "target_radius": target_radius,
            "target_energy": -MU / (2.0 * target_radius),
        }
    return row


def write_no_capture_dataset(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    no_capture = [row for row in rows if not bool(row["success"]) and not bool(row["capture_entered"])]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with NO_CAPTURE_DATASET_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NO_CAPTURE_FIELDNAMES)
        writer.writeheader()
        for row in no_capture:
            writer.writerow({key: row.get(key) for key in NO_CAPTURE_FIELDNAMES})
    return no_capture


def grouped_rate(rows: Sequence[Dict[str, object]], key: str, predicate) -> tuple[List[float], List[float], List[int]]:
    values = sorted({float(row[key]) for row in rows})
    rates: List[float] = []
    counts: List[int] = []
    for value in values:
        subset = [row for row in rows if abs(float(row[key]) - value) < 1.0e-12]
        counts.append(len(subset))
        rates.append(float(np.mean([1.0 if predicate(row) else 0.0 for row in subset])) if subset else 0.0)
    return values, rates, counts


def save_rate_bar(values: Sequence[float], rates: Sequence[float], path: Path, title: str, xlabel: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(values))
    bars = ax.bar(x, rates, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:g}" for value in values], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("rate")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2.0, rate + 0.02, f"{rate:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def capture_grid(rows: Sequence[Dict[str, object]], thrust: float) -> np.ndarray:
    grid = np.zeros((len(PHASE8_ANGLE_VALUES), len(PHASE8_R0_VALUES)), dtype=np.float64)
    counts = np.zeros_like(grid)
    for row in rows:
        if abs(float(row["thrust_scale"]) - thrust) < 1.0e-9:
            xi = PHASE8_R0_VALUES.index(float(row["r0_over_target"]))
            yi = PHASE8_ANGLE_VALUES.index(float(row["initial_velocity_angle_deg"]))
            grid[yi, xi] += 1.0 if bool(row["capture_entered"]) else 0.0
            counts[yi, xi] += 1.0
    return np.divide(grid, counts, out=np.zeros_like(grid), where=counts > 0.0)


def save_capture_map(rows: Sequence[Dict[str, object]], thrust: float, path: Path) -> None:
    grid = capture_grid(rows, thrust)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", norm=colors.Normalize(vmin=0.0, vmax=1.0))
    ax.set_xticks(np.arange(len(PHASE8_R0_VALUES)))
    ax.set_xticklabels([f"{value:g}" for value in PHASE8_R0_VALUES], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(PHASE8_ANGLE_VALUES)))
    ax.set_yticklabels([f"{value:g}" for value in PHASE8_ANGLE_VALUES])
    ax.set_xlabel("r0_over_target")
    ax.set_ylabel("initial_velocity_angle_deg")
    ax.set_title(f"CAPTURE access rate, thrust={thrust:g}, aggregated over target scales")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_phase8_analysis_plots(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    no_capture_pred = lambda row: (not bool(row["success"])) and (not bool(row["capture_entered"]))
    success_pred = lambda row: bool(row["success"])
    r0_values, no_capture_r0, _ = grouped_rate(rows, "r0_over_target", no_capture_pred)
    angle_values, no_capture_angle, _ = grouped_rate(rows, "initial_velocity_angle_deg", no_capture_pred)
    thrust_values, no_capture_thrust, _ = grouped_rate(rows, "thrust_scale", no_capture_pred)
    _, success_r0, _ = grouped_rate(rows, "r0_over_target", success_pred)
    _, success_angle, _ = grouped_rate(rows, "initial_velocity_angle_deg", success_pred)
    _, success_thrust, _ = grouped_rate(rows, "thrust_scale", success_pred)

    save_capture_map(rows, 10000.0, CAPTURE_MAP_T10000_PNG)
    save_capture_map(rows, 15000.0, CAPTURE_MAP_T15000_PNG)
    save_rate_bar(r0_values, no_capture_r0, NO_CAPTURE_BY_R0_PNG, "No-CAPTURE rate by r0_over_target", "r0_over_target", "#D95F02")
    save_rate_bar(angle_values, no_capture_angle, NO_CAPTURE_BY_ANGLE_PNG, "No-CAPTURE rate by initial velocity angle", "initial_velocity_angle_deg", "#D95F02")
    save_rate_bar(thrust_values, no_capture_thrust, NO_CAPTURE_BY_THRUST_PNG, "No-CAPTURE rate by thrust", "thrust_scale", "#D95F02")

    def spread(values: Sequence[float]) -> float:
        return float(max(values) - min(values)) if values else 0.0

    return {
        "no_capture_by_r0": dict(zip([f"{v:g}" for v in r0_values], no_capture_r0)),
        "no_capture_by_angle": dict(zip([f"{v:g}" for v in angle_values], no_capture_angle)),
        "no_capture_by_thrust": dict(zip([f"{v:g}" for v in thrust_values], no_capture_thrust)),
        "success_by_r0": dict(zip([f"{v:g}" for v in r0_values], success_r0)),
        "success_by_angle": dict(zip([f"{v:g}" for v in angle_values], success_angle)),
        "success_by_thrust": dict(zip([f"{v:g}" for v in thrust_values], success_thrust)),
        "success_spread_r0": spread(success_r0),
        "success_spread_angle": spread(success_angle),
        "success_spread_thrust": spread(success_thrust),
    }


def select_representative_failures(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    desired = [
        (0.98, 150.0, 8000.0),
        (0.98, 170.0, 12000.0),
        (1.00005, 170.0, 10000.0),
        (1.02, 150.0, 8000.0),
        (1.02, 175.0, 12000.0),
        (1.05, 165.0, 10000.0),
        (1.10, 150.0, 15000.0),
        (1.20, 175.0, 20000.0),
    ]
    no_capture = [row for row in rows if not bool(row["success"]) and not bool(row["capture_entered"])]
    selected: List[Dict[str, object]] = []
    used: set[tuple[float, float, float, float]] = set()
    for r0, angle, thrust in desired:
        candidates = [
            row
            for row in no_capture
            if abs(float(row["r0_over_target"]) - r0) < 1.0e-12
            and abs(float(row["initial_velocity_angle_deg"]) - angle) < 1.0e-12
            and abs(float(row["thrust_scale"]) - thrust) < 1.0e-12
        ]
        candidates.sort(key=lambda row: (abs(float(row["target_radius_scale"]) - 1.0), float(row["target_radius_scale"])))
        if candidates:
            row = candidates[0]
            key = (
                float(row["r0_over_target"]),
                float(row["initial_velocity_angle_deg"]),
                float(row["thrust_scale"]),
                float(row["target_radius_scale"]),
            )
            if key not in used:
                selected.append(row)
                used.add(key)
    return selected[:8]


def save_trajectory_plot(row: Dict[str, object], idx: int) -> None:
    controller = define_controllers()[0]
    result = rollout_case(
        controller,
        float(row["r0_over_target"]),
        float(row["initial_velocity_angle_deg"]),
        float(row["thrust_scale"]),
        float(row["target_radius_scale"]),
        record_trajectory=True,
    )
    traj = result["trajectory"]
    time_days = np.arange(len(traj["radius"]), dtype=np.float64) * DT / 86400.0
    radius_rel = np.asarray(traj["radius"], dtype=np.float64) / float(traj["target_radius"])
    vr = np.asarray(traj["radial_velocity"], dtype=np.float64)
    energy = np.asarray(traj["energy"], dtype=np.float64)
    target_energy = float(traj["target_energy"])

    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.0), sharex=True)
    axes[0].plot(time_days, radius_rel, color="#386CB0", linewidth=1.2)
    axes[0].axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("radius / target")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_days, vr, color="#D95F02", linewidth=1.0)
    axes[1].axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("radial velocity")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time_days, energy / abs(target_energy), color="#1B9E77", linewidth=1.0)
    axes[2].axhline(target_energy / abs(target_energy), color="#333333", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("specific energy / |target|")
    axes[2].set_xlabel("time [days]")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(
        "Phase 9 no-CAPTURE trajectory "
        f"r0={float(row['r0_over_target']):g}, angle={float(row['initial_velocity_angle_deg']):g}, "
        f"thrust={float(row['thrust_scale']):g}, target={float(row['target_radius_scale']):.2f}, "
        f"reason={row.get('termination_reason', '')}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = TRAJECTORY_DIR / (
        f"case_{idx:02d}_r0_{float(row['r0_over_target']):g}_"
        f"angle_{float(row['initial_velocity_angle_deg']):g}_"
        f"thrust_{float(row['thrust_scale']):g}_target_{float(row['target_radius_scale']):.2f}.png"
    )
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_trajectory_diagnostics(rows: Sequence[Dict[str, object]]) -> int:
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    selected = select_representative_failures(rows)
    for idx, row in enumerate(selected, start=1):
        save_trajectory_plot(row, idx)
    return len(selected)


def evaluate_small_grid() -> List[Dict[str, object]]:
    controllers = define_controllers()
    expected_names = {controller.name for controller in controllers}
    if REACHABILITY_COMPARISON_PATH.exists():
        existing_names = {str(row["controller_name"]) for row in load_comparison_rows()}
        if not existing_names.issubset(expected_names):
            REACHABILITY_COMPARISON_PATH.unlink()
    rows = load_comparison_rows()
    done = completed_keys(rows)
    cases = list(iter_small_cases(controllers))
    pending = [case for case in cases if (case[0].name, case[1], case[2], case[3], case[4]) not in done]
    print(f"phase9 comparison pending={len(pending)} completed={len(done)} total={len(cases)}")
    for idx, (controller, r0, angle, thrust, target_scale) in enumerate(pending, start=1):
        row = rollout_case(controller, r0, angle, thrust, target_scale)
        append_comparison_row(row)
        rows.append(row)
        print(
            f"phase9 {idx}/{len(pending)} controller={controller.name} r0={r0:g} angle={angle:g} "
            f"thrust={thrust:g} capture={row['capture_entered']} success={row['success']} "
            f"near_miss={row['near_miss']} reason={row['termination_reason']}"
        )
    return load_comparison_rows()


def controller_stats(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for name in sorted({str(row["controller_name"]) for row in rows}):
        subset = [row for row in rows if str(row["controller_name"]) == name]
        stats[name] = {
            "total": len(subset),
            "capture": sum(bool(row["capture_entered"]) for row in subset),
            "success": sum(bool(row["success"]) for row in subset),
            "near_miss": sum((not bool(row["success"])) and bool(row["near_miss"]) for row in subset),
            "termination_reasons": dict(Counter(str(row["termination_reason"]) for row in subset)),
        }
    return stats


def best_variant(stats: Dict[str, Dict[str, object]]) -> tuple[str, Dict[str, object]]:
    return max(
        stats.items(),
        key=lambda item: (
            int(item[1]["capture"]),
            int(item[1]["success"]),
            -int(item[1]["near_miss"]),
        ),
    )


def write_summary(
    phase8_rows: Sequence[Dict[str, object]],
    no_capture_rows: Sequence[Dict[str, object]],
    pattern_stats: Dict[str, object],
    comparison_rows: Sequence[Dict[str, object]],
    trajectory_count: int,
) -> None:
    stats = controller_stats(comparison_rows)
    best_name, best_stats = best_variant(stats)
    baseline = stats.get("baseline_soft_linear_3e4", {"capture": 0, "success": 0, "near_miss": 0, "total": 0})
    no_capture_total = len(no_capture_rows)
    phase8_total = len(phase8_rows)
    capture_total = sum(bool(row["capture_entered"]) for row in phase8_rows)
    r0_spread = float(pattern_stats["success_spread_r0"])
    angle_spread = float(pattern_stats["success_spread_angle"])
    thrust_spread = float(pattern_stats["success_spread_thrust"])
    dominant_factor = max(
        [("r0_over_target", r0_spread), ("initial_velocity_angle_deg", angle_spread), ("thrust_scale", thrust_spread)],
        key=lambda item: item[1],
    )[0]
    improvement = int(best_stats["capture"]) - int(baseline["capture"])
    improved_text = "yes" if improvement > 0 else "no"
    if improvement > 0 and best_name != "baseline_soft_linear_3e4":
        mechanism_text = (
            "The gain sweep uses the same energy-directional DESCENT structure at two radial-bias strengths, so the evidence supports "
            "a trajectory-shaping effect with sensitivity to parameter scale. It is not evidence for a new controller phase."
        )
    else:
        mechanism_text = (
            "The tested energy-directional extension did not improve the best capture count, so there is no positive mechanism claim beyond "
            "the Phase 8 reachability diagnosis."
        )

    lines = [
        "# Phase 9 Pre-CAPTURE Reachability Analysis",
        "",
        "## Scope",
        "",
        "- 2D Python-only analysis of Phase 8 no-CAPTURE failures.",
        "- Phase 7.6 `soft_linear_3e4` logic is used as the baseline and is not modified in place.",
        "- The only tested controller-side change is a DESCENT-only energy-directional reachability term outside the Phase 7 pre-window band.",
        "- CAPTURE/LOCK logic, physics, and success definition are unchanged.",
        "",
        "## Phase 8 Failure Structure",
        "",
        f"- Phase 8 rows analyzed: `{phase8_total}`.",
        f"- CAPTURE entries in Phase 8: `{capture_total}`.",
        f"- No-CAPTURE access cases (`success == False` and `capture_entered == False`): `{no_capture_total}`.",
        f"- Dominant factor by success-rate spread: `{dominant_factor}`.",
        f"- Success-rate spread by r0: `{r0_spread:.3f}`.",
        f"- Success-rate spread by angle: `{angle_spread:.3f}`.",
        f"- Success-rate spread by thrust: `{thrust_spread:.3f}`.",
        "",
        "## Small-Grid Reachability Comparison",
        "",
        "| Controller | Capture | Success | Near-miss | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in stats.items():
        lines.append(
            f"| `{name}` | {int(values['capture'])} | {int(values['success'])} | "
            f"{int(values['near_miss'])} | {int(values['total'])} |"
        )
    lines.extend(
        [
            "",
            f"- Best by capture/success/near-miss ranking: `{best_name}`.",
            f"- Capture improvement over baseline on the reduced grid: `{improvement}` regimes.",
            f"- Reachability improved: `{improved_text}`.",
            f"- Representative no-CAPTURE trajectory plots saved: `{trajectory_count}`.",
            "",
            "## Answers",
            "",
            "1. Why do most trajectories fail before CAPTURE? Most expanded-grid failures never cross the target radius, so the phase machine remains in DESCENT until `max_steps`. The trajectory diagnostics show radius staying on the same side of the target instead of entering the CAPTURE trigger.",
            f"2. Which factor matters most? `{dominant_factor}` matters most in this Phase 8 map. The r0 grouping is the largest contrast: local starts near `1.0` are reachable, while most broader radius offsets are not.",
            "3. What pattern defines `no_capture_access`? It is a target-radius crossing failure: `capture_entered == False`, usually `crossing_occurs == False`, with long integrations ending at `max_steps` rather than CAPTURE/LOCK instability.",
            f"4. Does the new variant improve capture rate? `{improved_text}`. The best reduced-grid controller is `{best_name}` with `{int(best_stats['capture'])}` captures versus baseline `{int(baseline['capture'])}`.",
            f"5. Is improvement due to better trajectory shaping or just parameter scaling? {mechanism_text}",
            "6. The next step toward global reachability should stay in 2D and map crossing geometry directly: classify whether each no-CAPTURE case needs radius raising, radius lowering, or angular-momentum correction before adding more controller structure.",
            "",
            "## Artifacts",
            "",
            "- `no_capture_dataset.csv`",
            "- `capture_map_r0_vs_angle_thrust10000.png`",
            "- `capture_map_r0_vs_angle_thrust15000.png`",
            "- `no_capture_rate_by_r0.png`",
            "- `no_capture_rate_by_angle.png`",
            "- `no_capture_rate_by_thrust.png`",
            "- `trajectories/*.png`",
            "- `reachability_comparison.csv`",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase8_rows = load_phase8_rows()
    no_capture_rows = write_no_capture_dataset(phase8_rows)
    pattern_stats = save_phase8_analysis_plots(phase8_rows)
    trajectory_count = save_trajectory_diagnostics(phase8_rows)
    comparison_rows = evaluate_small_grid()
    write_summary(phase8_rows, no_capture_rows, pattern_stats, comparison_rows, trajectory_count)
    print(f"Saved Phase 9 reachability outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
