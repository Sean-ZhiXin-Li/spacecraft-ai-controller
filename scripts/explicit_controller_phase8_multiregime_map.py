from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase8_multiregime"
MPLCONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig


GRID_CSV_PATH = OUTPUT_DIR / "phase8_grid.csv"
FAILURE_MODES_PATH = OUTPUT_DIR / "phase8_failure_modes.json"
SUMMARY_PATH = OUTPUT_DIR / "phase8_summary.md"

SUCCESS_BY_THRUST_PNG = OUTPUT_DIR / "success_rate_by_thrust.png"
SUCCESS_BY_ANGLE_PNG = OUTPUT_DIR / "success_rate_by_angle.png"
SUCCESS_BY_R0_PNG = OUTPUT_DIR / "success_rate_by_r0.png"
CAPTURE_BY_THRUST_PNG = OUTPUT_DIR / "capture_rate_by_thrust.png"
FAILURE_MODE_PNG = OUTPUT_DIR / "failure_mode_distribution.png"
SUCCESS_MAP_T10000_PNG = OUTPUT_DIR / "success_map_r0_vs_angle_thrust10000.png"
SUCCESS_MAP_T15000_PNG = OUTPUT_DIR / "success_map_r0_vs_angle_thrust15000.png"
MIN_RERR_MAP_T10000_PNG = OUTPUT_DIR / "min_radius_error_map_r0_vs_angle_thrust10000.png"
CAPTURE_SUCCESS_SUMMARY_PNG = OUTPUT_DIR / "capture_vs_success_summary.png"

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

R0_VALUES = [0.98, 0.99, 1.00, 1.00005, 1.01, 1.02, 1.05, 1.10, 1.20]
ANGLE_VALUES = [120.0, 135.0, 150.0, 160.0, 165.0, 170.0, 175.0, 180.0]
THRUST_VALUES = [5000.0, 8000.0, 10000.0, 12000.0, 15000.0, 20000.0]
TARGET_RADIUS_SCALES = [0.98, 1.00, 1.02]

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
    params: Dict[str, float]


def define_controllers() -> List[ControllerSpec]:
    return [
        ControllerSpec("soft_linear_3e4", "linear", 3.0e-4, 0.0, dict(BASE_PARAMS)),
    ]


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)


def recommended_workers(limit: int | None = None) -> int:
    cpu_count = os.cpu_count() or 2
    workers = max(1, min(cpu_count - 1, 12))
    if limit is not None:
        workers = min(workers, max(1, int(limit)))
    return workers


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


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
    radius_errors: List[float] = []
    vr_values: List[float] = []
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
            action_x, action_y = soft_hybrid_action(x, y, vx, vy, radius, real_r_ratio, abs_r_ratio)
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
    return {
        "controller_name": controller.name,
        "controller_params_json": json.dumps(
            {
                **controller.params,
                "alpha_schedule": controller.alpha_schedule,
                "alpha_mid": controller.alpha_mid,
                "alpha_width": controller.alpha_width,
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


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def parse_optional_int(value: object) -> int | None:
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return int(float(text))


def append_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in FIELDNAMES})


def load_rows(path: Path = GRID_CSV_PATH) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
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


def iter_cases(controllers: Sequence[ControllerSpec]) -> Iterable[tuple[ControllerSpec, float, float, float, float]]:
    for controller in controllers:
        for target_scale in TARGET_RADIUS_SCALES:
            for thrust in THRUST_VALUES:
                for angle in ANGLE_VALUES:
                    for r0 in R0_VALUES:
                        yield controller, r0, angle, thrust, target_scale


def classify_failure(row: Dict[str, object]) -> str | None:
    if bool(row["success"]):
        return None
    if bool(row["near_miss"]):
        return "near_miss"
    if str(row.get("termination_reason", "")) in {"out_range", "overspeed", "too_close", "radial_stall"}:
        return "escaped_or_diverged"
    if not bool(row["capture_entered"]):
        return "no_capture_access"
    return "unstable_after_capture"


def write_failure_modes(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    failures = [row for row in rows if not bool(row["success"])]
    counts = Counter(classify_failure(row) for row in failures)
    counts.pop(None, None)
    by_controller: Dict[str, Dict[str, int]] = {}
    for row in failures:
        controller_name = str(row["controller_name"])
        mode = classify_failure(row) or "success"
        by_controller.setdefault(controller_name, {})
        by_controller[controller_name][mode] = by_controller[controller_name].get(mode, 0) + 1
    payload = {
        "total_rows": len(rows),
        "failure_count": len(failures),
        "failure_modes": dict(counts),
        "by_controller": by_controller,
    }
    FAILURE_MODES_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def grouped_rate(rows: Sequence[Dict[str, object]], key: str, metric: str) -> tuple[List[float], List[float]]:
    values = sorted({float(row[key]) for row in rows})
    rates: List[float] = []
    for value in values:
        subset = [row for row in rows if abs(float(row[key]) - value) < 1.0e-12]
        rates.append(float(np.mean([1.0 if bool(row[metric]) else 0.0 for row in subset])) if subset else 0.0)
    return values, rates


def save_rate_bar(rows: Sequence[Dict[str, object]], key: str, metric: str, path: Path, title: str, xlabel: str) -> None:
    values, rates = grouped_rate(rows, key, metric)
    labels = [f"{value:g}" for value in values]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(values))
    bars = ax.bar(x, rates, color="#1B9E77" if metric == "success" else "#386CB0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
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


def success_grid(rows: Sequence[Dict[str, object]], thrust: float, target_scale: float = 1.0) -> np.ndarray:
    grid = np.zeros((len(ANGLE_VALUES), len(R0_VALUES)), dtype=np.float64)
    counts = np.zeros_like(grid)
    for row in rows:
        if abs(float(row["thrust_scale"]) - thrust) < 1.0e-9 and abs(float(row["target_radius_scale"]) - target_scale) < 1.0e-9:
            xi = R0_VALUES.index(float(row["r0_over_target"]))
            yi = ANGLE_VALUES.index(float(row["initial_velocity_angle_deg"]))
            grid[yi, xi] += 1.0 if bool(row["success"]) else 0.0
            counts[yi, xi] += 1.0
    return np.divide(grid, counts, out=np.zeros_like(grid), where=counts > 0.0)


def min_rerr_grid(rows: Sequence[Dict[str, object]], thrust: float, target_scale: float = 1.0) -> np.ndarray:
    grid = np.full((len(ANGLE_VALUES), len(R0_VALUES)), np.nan, dtype=np.float64)
    for row in rows:
        if abs(float(row["thrust_scale"]) - thrust) < 1.0e-9 and abs(float(row["target_radius_scale"]) - target_scale) < 1.0e-9:
            xi = R0_VALUES.index(float(row["r0_over_target"]))
            yi = ANGLE_VALUES.index(float(row["initial_velocity_angle_deg"]))
            grid[yi, xi] = float(row["minimum_abs_radius_error"]) / (DEFAULT_TARGET_RADIUS * float(row["target_radius_scale"]))
    return grid


def format_r0_angle_axes(ax: plt.Axes, title: str) -> None:
    ax.set_xticks(np.arange(len(R0_VALUES)))
    ax.set_xticklabels([f"{x:g}" for x in R0_VALUES], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ANGLE_VALUES)))
    ax.set_yticklabels([f"{y:g}" for y in ANGLE_VALUES])
    ax.set_xlabel("r0_over_target")
    ax.set_ylabel("initial_velocity_angle_deg")
    ax.set_title(title)


def save_success_map(rows: Sequence[Dict[str, object]], thrust: float, path: Path) -> None:
    grid = success_grid(rows, thrust)
    cmap = colors.ListedColormap(["#F2F2F2", "#1B9E77"])
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=colors.BoundaryNorm([-0.5, 0.5, 1.5], 2))
    format_r0_angle_axes(ax, f"soft_linear_3e4 success map, thrust={thrust:g}, target scale=1.00")
    fig.colorbar(image, ax=ax, ticks=[0, 1], fraction=0.045, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_min_rerr_map(rows: Sequence[Dict[str, object]], thrust: float, path: Path) -> None:
    grid = min_rerr_grid(rows, thrust)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    image = ax.imshow(np.log10(grid + 1.0e-12), origin="lower", aspect="auto", cmap="viridis")
    format_r0_angle_axes(ax, f"log10 minimum relative radius error, thrust={thrust:g}, target scale=1.00")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_failure_mode_plot(failure_payload: Dict[str, object]) -> None:
    modes = dict(failure_payload.get("failure_modes", {}))
    labels = list(modes.keys())
    values = [int(modes[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    if labels:
        bars = ax.bar(np.arange(len(labels)), values, color="#D95F02")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(values + [1]) * 0.01, str(value), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("failed regimes")
    ax.set_title("Phase 8 failure mode distribution")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FAILURE_MODE_PNG, dpi=220)
    plt.close(fig)


def save_capture_success_summary(rows: Sequence[Dict[str, object]]) -> None:
    labels = ["success", "CAPTURE", "crossing", "near-miss failures"]
    values = [
        sum(bool(row["success"]) for row in rows),
        sum(bool(row["capture_entered"]) for row in rows),
        sum(bool(row["crossing_occurs"]) for row in rows),
        sum((not bool(row["success"])) and bool(row["near_miss"]) for row in rows),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(np.arange(len(labels)), values, color=["#1B9E77", "#386CB0", "#7570B3", "#D95F02"])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("count over regimes")
    ax.set_title("Phase 8 capture vs success summary")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(values + [1]) * 0.01, str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(CAPTURE_SUCCESS_SUMMARY_PNG, dpi=220)
    plt.close(fig)


def save_plots(rows: Sequence[Dict[str, object]], failure_payload: Dict[str, object]) -> None:
    save_rate_bar(rows, "thrust_scale", "success", SUCCESS_BY_THRUST_PNG, "Success rate by thrust", "thrust_scale")
    save_rate_bar(rows, "initial_velocity_angle_deg", "success", SUCCESS_BY_ANGLE_PNG, "Success rate by initial velocity angle", "initial_velocity_angle_deg")
    save_rate_bar(rows, "r0_over_target", "success", SUCCESS_BY_R0_PNG, "Success rate by r0_over_target", "r0_over_target")
    save_rate_bar(rows, "thrust_scale", "capture_entered", CAPTURE_BY_THRUST_PNG, "CAPTURE rate by thrust", "thrust_scale")
    save_failure_mode_plot(failure_payload)
    save_success_map(rows, 10000.0, SUCCESS_MAP_T10000_PNG)
    save_success_map(rows, 15000.0, SUCCESS_MAP_T15000_PNG)
    save_min_rerr_map(rows, 10000.0, MIN_RERR_MAP_T10000_PNG)
    save_capture_success_summary(rows)


def best_group(rows: Sequence[Dict[str, object]], key: str) -> tuple[float, float]:
    values, rates = grouped_rate(rows, key, "success")
    if not values:
        return 0.0, 0.0
    idx = int(np.argmax(rates))
    return values[idx], rates[idx]


def write_summary(rows: Sequence[Dict[str, object]], failure_payload: Dict[str, object]) -> None:
    total = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    captures = sum(bool(row["capture_entered"]) for row in rows)
    locks = sum(bool(row["lock_entered"]) for row in rows)
    crossings = sum(bool(row["crossing_occurs"]) for row in rows)
    near_misses = sum((not bool(row["success"])) and bool(row["near_miss"]) for row in rows)
    success_rate = successes / total if total else 0.0
    capture_rate = captures / total if total else 0.0
    failure_modes = dict(failure_payload.get("failure_modes", {}))
    dominant_mode = max(failure_modes.items(), key=lambda item: item[1])[0] if failure_modes else "none"
    best_thrust, best_thrust_rate = best_group(rows, "thrust_scale")
    best_angle, best_angle_rate = best_group(rows, "initial_velocity_angle_deg")
    best_r0, best_r0_rate = best_group(rows, "r0_over_target")
    no_capture = int(failure_modes.get("no_capture_access", 0))
    unstable = int(failure_modes.get("unstable_after_capture", 0))
    access_text = "pre-CAPTURE access failures" if no_capture >= unstable else "post-CAPTURE instability"
    supports_text = (
        "partially supports"
        if successes > 0 and success_rate < 0.5
        else "supports"
        if success_rate >= 0.5
        else "does not yet support"
    )

    lines = [
        "# Phase 8 Multi-Regime Generalization Map",
        "",
        "## Setup",
        "",
        "- Scope: 2D Python-only expanded regime map for the Phase 7.6 `soft_linear_3e4` controller.",
        "- Controller structure, physics, CAPTURE/LOCK logic, and success definition are unchanged from Phase 7.6.",
        f"- Grid size: `{total}` completed regimes.",
        "- This is not a final project result; it is a current 2D Phase 8 generalization diagnostic.",
        "",
        "## Aggregate Result",
        "",
        f"- Successes: `{successes}` / `{total}` (`{success_rate:.3f}`).",
        f"- CAPTURE entries: `{captures}` / `{total}` (`{capture_rate:.3f}`).",
        f"- LOCK entries: `{locks}` / `{total}`.",
        f"- Radius crossings: `{crossings}` / `{total}`.",
        f"- Near-miss failures: `{near_misses}`.",
        f"- Dominant failure mode: `{dominant_mode}`.",
        "",
        "## Best Slices",
        "",
        f"- Best thrust slice: `{best_thrust:g}` with success rate `{best_thrust_rate:.3f}`.",
        f"- Best angle slice: `{best_angle:g}` with success rate `{best_angle_rate:.3f}`.",
        f"- Best r0 slice: `{best_r0:g}` with success rate `{best_r0_rate:.3f}`.",
        "",
        "## Answers",
        "",
        f"1. Does `soft_linear_3e4` generalize beyond the Phase 7.6 local grid? It generalizes only if the expanded map retains a meaningful success region. On this Phase 8 grid, success is `{successes}` / `{total}` (`{success_rate:.3f}`), so the evidence should be read at that scope.",
        f"2. Where does it work best? The strongest one-dimensional slices are thrust `{best_thrust:g}`, angle `{best_angle:g}`, and r0 `{best_r0:g}` by success rate.",
        f"3. Where does it fail? Failures concentrate in the dominant class `{dominant_mode}`; the plots break this down by r0, angle, and thrust.",
        f"4. Are failures mostly pre-CAPTURE access failures or post-CAPTURE instability? In this classification they are mostly `{access_text}`: no_capture_access `{no_capture}`, unstable_after_capture `{unstable}`.",
        f"5. Does the result support the continuous-coordination insight? It `{supports_text}` the insight in 2D: the same coordinated controller was tested without structural changes, and the map shows where the structure remains useful versus where expanded initial conditions exceed its reach.",
        "6. Phase 9 should stay in 2D and analyze the boundary of the successful Phase 8 regions before adding controller complexity: focus on failure-mode-specific diagnostics, especially whether no-capture cases need broader pre-window shaping or whether captured failures need CAPTURE/LOCK robustness checks.",
        "",
        "## Artifacts",
        "",
        "- `phase8_grid.csv`",
        "- `phase8_failure_modes.json`",
        "- `success_rate_by_thrust.png`",
        "- `success_rate_by_angle.png`",
        "- `success_rate_by_r0.png`",
        "- `capture_rate_by_thrust.png`",
        "- `failure_mode_distribution.png`",
        "- `success_map_r0_vs_angle_thrust10000.png`",
        "- `success_map_r0_vs_angle_thrust15000.png`",
        "- `min_radius_error_map_r0_vs_angle_thrust10000.png`",
        "- `capture_vs_success_summary.png`",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    controllers = define_controllers()
    rows = load_rows()
    done = completed_keys(rows)
    cases = list(iter_cases(controllers))
    total = len(cases)
    pending = [(idx, case) for idx, case in enumerate(cases, start=1) if (case[0].name, case[1], case[2], case[3], case[4]) not in done]

    if pending:
        workers = recommended_workers()
        print(f"phase8 pending={len(pending)} completed={len(done)} total={total} workers={workers}")
        if workers == 1:
            for idx, (controller, r0, angle, thrust, target_scale) in pending:
                row = rollout_case(controller, r0, angle, thrust, target_scale)
                append_row(GRID_CSV_PATH, row)
                rows.append(row)
                done.add((controller.name, r0, angle, thrust, target_scale))
                print(
                    f"phase8 {idx}/{total} controller={controller.name} r0={r0:g} angle={angle:g} "
                    f"thrust={thrust:g} target={target_scale:.2f} success={row['success']} "
                    f"capture={row['capture_entered']} near_miss={row['near_miss']} "
                    f"reason={row['termination_reason']} min_abs_rerr={row['minimum_abs_radius_error']:.3e}"
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_case = {
                    executor.submit(rollout_case, controller, r0, angle, thrust, target_scale): (idx, controller, r0, angle, thrust, target_scale)
                    for idx, (controller, r0, angle, thrust, target_scale) in pending
                }
                completed_now = 0
                for future in as_completed(future_to_case):
                    idx, controller, r0, angle, thrust, target_scale = future_to_case[future]
                    row = future.result()
                    append_row(GRID_CSV_PATH, row)
                    rows.append(row)
                    done.add((controller.name, r0, angle, thrust, target_scale))
                    completed_now += 1
                    print(
                        f"phase8 {idx}/{total} done_now={completed_now}/{len(pending)} controller={controller.name} "
                        f"r0={r0:g} angle={angle:g} thrust={thrust:g} target={target_scale:.2f} "
                        f"success={row['success']} capture={row['capture_entered']} near_miss={row['near_miss']} "
                        f"reason={row['termination_reason']} min_abs_rerr={row['minimum_abs_radius_error']:.3e}"
                    )
    else:
        print(f"phase8 no pending rows; completed={len(done)} total={total}")

    rows = load_rows()
    failure_payload = write_failure_modes(rows)
    save_plots(rows, failure_payload)
    write_summary(rows, failure_payload)
    print(f"Saved Phase 8 multiregime outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
