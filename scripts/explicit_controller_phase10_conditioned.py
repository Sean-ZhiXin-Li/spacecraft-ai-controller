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
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase10_conditioned"
MPLCONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig


COMPARISON_CSV = OUTPUT_DIR / "comparison.csv"
MODE_USAGE_JSON = OUTPUT_DIR / "mode_usage.json"
SUMMARY_MD = OUTPUT_DIR / "phase10_summary.md"
SUCCESS_BY_MODE_PNG = OUTPUT_DIR / "success_by_mode.png"
CAPTURE_IMPROVEMENT_PNG = OUTPUT_DIR / "capture_improvement_vs_baseline.png"

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
    "controller_name",
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
    "dominant_descent_mode",
    "mode_counts_json",
]

MODES = ["outer_orbit", "inner_orbit", "angle_misaligned", "near_window"]


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    conditioned: bool


def define_controllers() -> List[ControllerSpec]:
    return [
        ControllerSpec("baseline_soft_linear_3e4", False),
        ControllerSpec("phase10_conditioned", True),
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


def rollout_case(controller: ControllerSpec, r0: float, angle: float, thrust_scale: float, target_scale: float) -> Dict[str, object]:
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
    capture_entered = False
    lock_entered = False
    success = False
    terminated = False
    truncated = False
    termination_reason = ""
    mode_counts: Counter[str] = Counter()
    mode_success_steps: Counter[str] = Counter()

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

    def state_energy_error_ratio(state_x: float, state_y: float, state_vx: float, state_vy: float) -> float:
        state_radius = math.sqrt(state_x * state_x + state_y * state_y)
        speed2 = state_vx * state_vx + state_vy * state_vy
        energy = 0.5 * speed2 - MU / (state_radius + 1.0e-12)
        target_energy = -MU / (2.0 * target_radius)
        return abs(energy - target_energy) / (abs(target_energy) + 1.0e-12)

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

    def strategy_action(mode: str, state_x: float, state_y: float, state_vx: float, state_vy: float, state_radius: float, real_r_ratio: float, abs_r_error_ratio: float) -> tuple[float, float]:
        if mode == "near_window":
            return soft_linear_3e4_action(state_x, state_y, state_vx, state_vy, state_radius, real_r_ratio, abs_r_error_ratio)
        r_hat_x, r_hat_y, t_hat_x, t_hat_y, _speed, vr, vt, _radius = basis(state_x, state_y, state_vx, state_vy)
        if mode == "outer_orbit":
            retro_x, retro_y = baseline_action(state_vx, state_vy)
            return normalize_action(retro_x - 0.18 * r_hat_x, retro_y - 0.18 * r_hat_y)
        if mode == "inner_orbit":
            prog_x, prog_y = prograde_action(state_vx, state_vy)
            return normalize_action(0.34 * prog_x + 0.20 * r_hat_x, 0.34 * prog_y + 0.20 * r_hat_y, max_norm=0.48)
        target_radial_sign = -1.0 if real_r_ratio > 0.0 else 1.0
        tangent_sign = 1.0 if vt < v_circ else -1.0
        radial_damp_sign = -1.0 if vr > 0.0 else 1.0
        radial_sign = target_radial_sign if real_r_ratio * vr > 0.0 else radial_damp_sign
        return normalize_action(
            0.64 * radial_sign * r_hat_x + 0.28 * tangent_sign * t_hat_x,
            0.64 * radial_sign * r_hat_y + 0.28 * tangent_sign * t_hat_y,
            max_norm=0.72,
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

        descent_mode = ""
        if phase == "DESCENT":
            if controller.conditioned:
                descent_mode = classify_state((x, y), (vx, vy), target_radius)
                action_x, action_y = strategy_action(descent_mode, x, y, vx, vy, radius, real_r_ratio, abs_r_ratio)
            else:
                descent_mode = "near_window" if abs_r_ratio <= PREWINDOW_OUTER_RATIO else "unconditioned_descent"
                action_x, action_y = soft_linear_3e4_action(x, y, vx, vy, radius, real_r_ratio, abs_r_ratio)
            mode_counts[descent_mode] += 1
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

        success = success_counter >= int(STRICT_CFG["success_threshold"])
        if descent_mode and success:
            mode_success_steps[descent_mode] += 1

        radial_stall = (r_err_now < 0.10) and (v_err_now < 0.12) and (ang_abs_now > 0.85)
        radial_stall_counter = radial_stall_counter + 1 if radial_stall else 0

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
    dominant_mode = max(mode_counts.items(), key=lambda item: item[1])[0] if mode_counts else ""
    mode_counts_payload = {key: int(value) for key, value in sorted(mode_counts.items())}
    return {
        "controller_name": controller.name,
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
        "dominant_descent_mode": dominant_mode,
        "mode_counts_json": json.dumps(mode_counts_payload, sort_keys=True),
    }


def write_comparison(rows: Sequence[Dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
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


def write_mode_usage(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    conditioned_rows = [row for row in rows if str(row["controller_name"]) == "phase10_conditioned"]
    step_counts: Counter[str] = Counter()
    case_counts: Counter[str] = Counter()
    case_success: Counter[str] = Counter()
    case_capture: Counter[str] = Counter()
    for row in conditioned_rows:
        counts = json.loads(str(row.get("mode_counts_json", "{}")))
        for mode, count in counts.items():
            step_counts[mode] += int(count)
            if int(count) > 0:
                case_counts[mode] += 1
                if bool(row["success"]):
                    case_success[mode] += 1
                if bool(row["capture_entered"]):
                    case_capture[mode] += 1
    success_by_mode = {}
    for mode in MODES:
        cases = int(case_counts.get(mode, 0))
        success_by_mode[mode] = {
            "cases_triggered": cases,
            "successes": int(case_success.get(mode, 0)),
            "captures": int(case_capture.get(mode, 0)),
            "success_rate": float(case_success.get(mode, 0) / cases) if cases else 0.0,
            "capture_rate": float(case_capture.get(mode, 0) / cases) if cases else 0.0,
        }
    payload = {
        "descent_step_counts": {key: int(value) for key, value in sorted(step_counts.items())},
        "case_counts": {key: int(value) for key, value in sorted(case_counts.items())},
        "success_by_mode": success_by_mode,
    }
    MODE_USAGE_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def save_success_by_mode(mode_usage: Dict[str, object]) -> None:
    success_by_mode = dict(mode_usage["success_by_mode"])
    labels = MODES
    success_rates = [float(success_by_mode[mode]["success_rate"]) for mode in labels]
    capture_rates = [float(success_by_mode[mode]["capture_rate"]) for mode in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - 0.18, capture_rates, width=0.36, label="CAPTURE", color="#386CB0")
    ax.bar(x + 0.18, success_rates, width=0.36, label="success", color="#1B9E77")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("case rate when mode triggered")
    ax.set_title("Phase 10 outcome rate by triggered mode")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SUCCESS_BY_MODE_PNG, dpi=220)
    plt.close(fig)


def save_capture_improvement(stats: Dict[str, Dict[str, int]]) -> None:
    labels = ["baseline", "phase10"]
    baseline = stats["baseline_soft_linear_3e4"]
    conditioned = stats["phase10_conditioned"]
    captures = [baseline["capture"], conditioned["capture"]]
    successes = [baseline["success"], conditioned["success"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(x - 0.18, captures, width=0.36, label="CAPTURE", color="#386CB0")
    ax.bar(x + 0.18, successes, width=0.36, label="success", color="#1B9E77")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("count over 48 regimes")
    ax.set_title("Phase 10 conditioned controller vs baseline")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    for idx, value in enumerate(captures):
        ax.text(idx - 0.18, value + 0.4, str(value), ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(successes):
        ax.text(idx + 0.18, value + 0.4, str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(CAPTURE_IMPROVEMENT_PNG, dpi=220)
    plt.close(fig)


def write_summary(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, int]], mode_usage: Dict[str, object]) -> None:
    baseline = stats["baseline_soft_linear_3e4"]
    conditioned = stats["phase10_conditioned"]
    capture_improvement = conditioned["capture"] - baseline["capture"]
    success_improvement = conditioned["success"] - baseline["success"]
    success_by_mode = dict(mode_usage["success_by_mode"])
    most_used_mode = max(dict(mode_usage["case_counts"]).items(), key=lambda item: item[1])[0] if mode_usage["case_counts"] else "none"
    best_capture_mode = max(success_by_mode.items(), key=lambda item: item[1]["capture_rate"])[0] if success_by_mode else "none"
    helps = capture_improvement > 0 or success_improvement > 0
    honest_result = "improves" if helps else "does not improve"
    lines = [
        "# Phase 10 Failure-Conditioned Reachability Controller",
        "",
        "## Scope",
        "",
        "- 2D Python-only explicit controller test.",
        "- CAPTURE/LOCK logic, physics, and success definition are unchanged.",
        "- DESCENT chooses among `outer_orbit`, `inner_orbit`, `angle_misaligned`, and `near_window` strategies.",
        "- `near_window` uses the existing Phase 7.6 `soft_linear_3e4` behavior unchanged.",
        "",
        "## Reduced-Grid Result",
        "",
        "| Controller | CAPTURE | Success | Near-miss | Total |",
        "|---|---:|---:|---:|---:|",
        f"| `baseline_soft_linear_3e4` | {baseline['capture']} | {baseline['success']} | {baseline['near_miss']} | {baseline['total']} |",
        f"| `phase10_conditioned` | {conditioned['capture']} | {conditioned['success']} | {conditioned['near_miss']} | {conditioned['total']} |",
        "",
        f"- Capture improvement: `{capture_improvement}`.",
        f"- Success improvement: `{success_improvement}`.",
        f"- Most frequently triggered failure type: `{most_used_mode}`.",
        f"- Highest capture-rate triggered mode: `{best_capture_mode}`.",
        "",
        "## Answers",
        "",
        f"1. Does failure-conditioned control improve reachability? It `{honest_result}` reachability on this reduced Phase 9 grid: CAPTURE changes by `{capture_improvement}` and success changes by `{success_improvement}`.",
        f"2. Which failure type benefits most? By triggered-mode capture rate, `{best_capture_mode}` is strongest. By frequency, `{most_used_mode}` is the dominant diagnosed mode.",
        "3. Is geometry-aware control necessary? The Phase 8 and Phase 9 evidence says geometry diagnosis is necessary for understanding the failures. This first simple conditioned controller is not sufficient proof that the tested strategy set solves them.",
        f"4. Does this outperform single-controller design? `{ 'yes' if helps else 'no' }` on the reduced grid. The comparison should be treated as a controller-structure diagnostic, not a global 2D solution.",
        "",
        "## Artifacts",
        "",
        "- `comparison.csv`",
        "- `mode_usage.json`",
        "- `success_by_mode.png`",
        "- `capture_improvement_vs_baseline.png`",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    cases = list(iter_cases(define_controllers()))
    for idx, (controller, r0, angle, thrust, target_scale) in enumerate(cases, start=1):
        row = rollout_case(controller, r0, angle, thrust, target_scale)
        rows.append(row)
        print(
            f"phase10 {idx}/{len(cases)} controller={controller.name} r0={r0:g} angle={angle:g} "
            f"thrust={thrust:g} capture={row['capture_entered']} success={row['success']} "
            f"mode={row['dominant_descent_mode']} reason={row['termination_reason']}"
        )
    write_comparison(rows)
    stats = controller_stats(rows)
    mode_usage = write_mode_usage(rows)
    save_success_by_mode(mode_usage)
    save_capture_improvement(stats)
    write_summary(rows, stats, mode_usage)
    print(f"Saved Phase 10 conditioned outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
