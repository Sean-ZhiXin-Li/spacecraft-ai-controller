from __future__ import annotations

import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase7_pre_window_shaping"
MPLCONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig


GRID_CSV_PATH = OUTPUT_DIR / "pre_window_grid.csv"
RANKING_CSV_PATH = OUTPUT_DIR / "pre_window_ranking.csv"
SUMMARY_PATH = OUTPUT_DIR / "phase7_summary.md"

SUCCESS_PNG = OUTPUT_DIR / "success_by_prewindow_variant.png"
CAPTURE_PNG = OUTPUT_DIR / "capture_by_prewindow_variant.png"
NEAR_MISS_PNG = OUTPUT_DIR / "near_miss_by_prewindow_variant.png"
COMPARISON_PNG = OUTPUT_DIR / "original_vs_adaptive_vs_best_prewindow.png"
BEST_MAP_PNG = OUTPUT_DIR / "best_prewindow_success_map.png"

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

R0_VALUES = [1.00002, 1.00004, 1.00005, 1.00006, 1.00008, 1.00010]
ANGLE_VALUES = [165.0, 168.0, 170.0, 172.0, 175.0]
THRUST_VALUES = [9000.0, 10000.0, 11000.0]
TARGET_RADIUS_SCALES = [0.99, 1.00, 1.01]

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
}

FIELDNAMES = [
    "variant_name",
    "variant_params_json",
    "schedule",
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
]

RANKING_FIELDNAMES = [
    "rank",
    "variant_name",
    "schedule",
    "success_count",
    "capture_count",
    "lock_count",
    "near_miss_count",
    "mean_minimum_abs_radius_error",
    "mean_tail_abs_vr",
    "mean_steps",
    "gained_vs_original",
    "lost_vs_original",
    "retained_vs_original",
    "gained_vs_adaptive_soft",
    "lost_vs_adaptive_soft",
    "retained_vs_adaptive_soft",
]


@dataclass(frozen=True)
class Phase7Variant:
    name: str
    schedule: str
    params: Dict[str, float | str]


def define_variants() -> List[Phase7Variant]:
    variants = [
        Phase7Variant("original_ws1", "fixed_0p70", {**BASE_PARAMS, "fixed_scale": 0.70, "prewindow_kind": "none"}),
        Phase7Variant("adaptive_soft", "soft_rerr", {**BASE_PARAMS, "prewindow_kind": "none"}),
        Phase7Variant(
            "prewindow_radial_light",
            "soft_rerr+prewindow_radial",
            {**BASE_PARAMS, "prewindow_kind": "radial", "prewindow_radial_gain": 0.04},
        ),
        Phase7Variant(
            "prewindow_radial_medium",
            "soft_rerr+prewindow_radial",
            {**BASE_PARAMS, "prewindow_kind": "radial", "prewindow_radial_gain": 0.08},
        ),
        Phase7Variant(
            "prewindow_energy_guided",
            "soft_rerr+prewindow_energy",
            {
                **BASE_PARAMS,
                "prewindow_kind": "energy",
                "energy_norm": 5.0e-4,
                "energy_scale_low": 0.72,
                "energy_scale_high": 1.00,
                "energy_radial_gain": 0.05,
            },
        ),
        Phase7Variant(
            "prewindow_geometry_guided",
            "soft_rerr+prewindow_geometry",
            {
                **BASE_PARAMS,
                "prewindow_kind": "geometry",
                "geometry_vr_fast": -0.020,
                "geometry_vr_slow": -0.006,
                "geometry_fast_retro_scale": 0.72,
                "geometry_slow_retro_scale": 1.00,
                "geometry_tangential_gain": 0.06,
                "geometry_radial_gain": 0.07,
            },
        ),
    ]
    if len(variants) > 8:
        raise RuntimeError(f"Phase 7 variant cap exceeded: {len(variants)}")
    return variants


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


def reduced_retro_scale(
    variant: Phase7Variant,
    abs_r_error_ratio: float,
    energy_error_ratio: float,
) -> float:
    params = variant.params
    if variant.schedule.startswith("fixed"):
        return float(params["fixed_scale"])
    if variant.schedule.startswith("soft_rerr"):
        z = clamp(abs_r_error_ratio / (WS_BAND_RATIO + 1.0e-12), 0.0, 1.0)
        return float(params["min_scale"]) + (float(params["max_scale"]) - float(params["min_scale"])) * z
    if variant.schedule == "energy":
        z = clamp(energy_error_ratio / float(params["energy_norm"]), 0.0, 1.0)
        return float(params["min_scale"]) + (float(params["max_scale"]) - float(params["min_scale"])) * z
    raise ValueError(f"Unknown schedule: {variant.schedule}")


def rollout_case(variant: Phase7Variant, r0: float, angle: float, thrust_scale: float, target_scale: float) -> Dict[str, object]:
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
    prev_r_error: float | None = None
    radius_crossings_total = 0
    first_crossing_step: int | None = None
    params = variant.params

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

    def signed_energy_error_ratio(state_x: float, state_y: float, state_vx: float, state_vy: float) -> float:
        state_radius = math.sqrt(state_x * state_x + state_y * state_y)
        speed2 = state_vx * state_vx + state_vy * state_vy
        energy = 0.5 * speed2 - MU / (state_radius + 1.0e-12)
        target_energy = -MU / (2.0 * target_radius)
        return (energy - target_energy) / (abs(target_energy) + 1.0e-12)

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

    def normalize_action(action_x: float, action_y: float) -> tuple[float, float]:
        action_norm = math.sqrt(action_x * action_x + action_y * action_y)
        if action_norm > 1.0:
            return action_x / action_norm, action_y / action_norm
        return clamp(action_x, -1.0, 1.0), clamp(action_y, -1.0, 1.0)

    def window_action(
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
        energy_ratio = state_energy_error_ratio(state_x, state_y, state_vx, state_vy)
        scale = reduced_retro_scale(variant, abs_r_error_ratio, energy_ratio)
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

    def prewindow_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
        real_r_ratio: float,
    ) -> tuple[float, float]:
        kind = str(params.get("prewindow_kind", "none"))
        retro_x, retro_y = baseline_action(state_vx, state_vy)
        if kind == "none":
            return retro_x, retro_y

        inv_r = 1.0 / (state_radius + 1.0e-12)
        r_hat_x = state_x * inv_r
        r_hat_y = state_y * inv_r
        t_hat_x = -r_hat_y
        t_hat_y = r_hat_x
        vr_ratio = ((state_x * state_vx + state_y * state_vy) * inv_r) / v_circ
        vt_ratio = (state_vx * t_hat_x + state_vy * t_hat_y - v_circ) / v_circ
        inward_sign = -1.0 if real_r_ratio > 0.0 else 1.0

        if kind == "radial":
            gain = float(params["prewindow_radial_gain"])
            return normalize_action(retro_x + gain * inward_sign * r_hat_x, retro_y + gain * inward_sign * r_hat_y)

        if kind == "energy":
            signed_energy = signed_energy_error_ratio(state_x, state_y, state_vx, state_vy)
            z = clamp(max(0.0, signed_energy) / float(params["energy_norm"]), 0.0, 1.0)
            scale = float(params["energy_scale_low"]) + (float(params["energy_scale_high"]) - float(params["energy_scale_low"])) * z
            radial_gain = float(params["energy_radial_gain"]) * z
            return normalize_action(
                scale * retro_x + radial_gain * inward_sign * r_hat_x,
                scale * retro_y + radial_gain * inward_sign * r_hat_y,
            )

        if kind == "geometry":
            fast = float(params["geometry_vr_fast"])
            slow = float(params["geometry_vr_slow"])
            if vr_ratio < fast:
                retro_scale = float(params["geometry_fast_retro_scale"])
                tangential_cmd = -float(params["geometry_tangential_gain"]) * clamp(vt_ratio, -1.0, 1.0)
                return normalize_action(
                    retro_scale * retro_x + tangential_cmd * t_hat_x,
                    retro_scale * retro_y + tangential_cmd * t_hat_y,
                )
            if vr_ratio > slow:
                retro_scale = float(params["geometry_slow_retro_scale"])
                radial_gain = float(params["geometry_radial_gain"])
                return normalize_action(
                    retro_scale * retro_x + radial_gain * inward_sign * r_hat_x,
                    retro_scale * retro_y + radial_gain * inward_sign * r_hat_y,
                )
            return retro_x, retro_y

        raise ValueError(f"Unknown prewindow_kind: {kind}")

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
            if abs_r_ratio < WS_BAND_RATIO:
                action_x, action_y = window_action(x, y, vx, vy, radius, abs_r_ratio)
            elif abs_r_ratio <= PREWINDOW_OUTER_RATIO:
                action_x, action_y = prewindow_action(x, y, vx, vy, radius, real_r_ratio)
            else:
                action_x, action_y = baseline_action(vx, vy)
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
        if terminated or truncated:
            break

    rerr = np.asarray(radius_errors, dtype=np.float64)
    vr_arr = np.asarray(vr_values, dtype=np.float64)
    tail = vr_arr[-min(TAIL_LEN, len(vr_arr)) :] if len(vr_arr) else np.asarray([0.0])
    min_abs_rerr = float(np.min(np.abs(rerr))) if len(rerr) else 0.0
    near_miss = (not success) and min_abs_rerr / target_radius <= NEAR_MISS_REL
    return {
        "variant_name": variant.name,
        "variant_params_json": json.dumps(variant.params, sort_keys=True),
        "schedule": variant.schedule,
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
            }
        )
    return rows


def completed_keys(rows: Sequence[Dict[str, object]]) -> set[tuple[str, float, float, float, float]]:
    return {
        (
            str(row["variant_name"]),
            float(row["r0_over_target"]),
            float(row["initial_velocity_angle_deg"]),
            float(row["thrust_scale"]),
            float(row["target_radius_scale"]),
        )
        for row in rows
    }


def case_key(row: Dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(row["r0_over_target"]),
        float(row["initial_velocity_angle_deg"]),
        float(row["thrust_scale"]),
        float(row["target_radius_scale"]),
    )


def rank_rows(rows: Sequence[Dict[str, object]], variants: Sequence[Phase7Variant]) -> List[Dict[str, object]]:
    original_success = {
        case_key(row)
        for row in rows
        if row["variant_name"] == "original_ws1" and bool(row["success"])
    }
    adaptive_success = {
        case_key(row)
        for row in rows
        if row["variant_name"] == "adaptive_soft" and bool(row["success"])
    }
    ranking: List[Dict[str, object]] = []
    for variant in variants:
        subset = [row for row in rows if row["variant_name"] == variant.name]
        if not subset:
            continue
        success_set = {case_key(row) for row in subset if bool(row["success"])}
        ranking.append(
            {
                "variant_name": variant.name,
                "schedule": variant.schedule,
                "success_count": sum(bool(row["success"]) for row in subset),
                "capture_count": sum(bool(row["capture_entered"]) for row in subset),
                "lock_count": sum(bool(row["lock_entered"]) for row in subset),
                "near_miss_count": sum(bool(row["near_miss"]) for row in subset),
                "mean_minimum_abs_radius_error": float(np.mean([float(row["minimum_abs_radius_error"]) for row in subset])),
                "mean_tail_abs_vr": float(np.mean([float(row["tail_mean_abs_vr"]) for row in subset])),
                "mean_steps": float(np.mean([float(row["steps"]) for row in subset])),
                "gained_vs_original": len(success_set - original_success),
                "lost_vs_original": len(original_success - success_set),
                "retained_vs_original": len(success_set & original_success),
                "gained_vs_adaptive_soft": len(success_set - adaptive_success),
                "lost_vs_adaptive_soft": len(adaptive_success - success_set),
                "retained_vs_adaptive_soft": len(success_set & adaptive_success),
            }
        )
    ranking.sort(
        key=lambda row: (
            -int(row["success_count"]),
            -int(row["capture_count"]),
            int(row["near_miss_count"]),
            float(row["mean_minimum_abs_radius_error"]),
        )
    )
    for idx, row in enumerate(ranking, start=1):
        row["rank"] = idx
    return ranking


def write_ranking(ranking: Sequence[Dict[str, object]]) -> None:
    with RANKING_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RANKING_FIELDNAMES)
        writer.writeheader()
        for row in ranking:
            writer.writerow({key: row.get(key) for key in RANKING_FIELDNAMES})


def save_variant_bar(ranking: Sequence[Dict[str, object]], metric: str, path: Path, title: str, ylabel: str, color: str) -> None:
    labels = [str(row["variant_name"]) for row in ranking]
    values = [float(row[metric]) for row in ranking]
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(values + [1.0]) * 0.01, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def success_grid(rows: Sequence[Dict[str, object]], variant_name: str, thrust: float = 10000.0, target_scale: float = 1.00) -> np.ndarray:
    grid = np.zeros((len(ANGLE_VALUES), len(R0_VALUES)), dtype=np.float64)
    for row in rows:
        if (
            row["variant_name"] == variant_name
            and abs(float(row["thrust_scale"]) - thrust) < 1.0e-9
            and abs(float(row["target_radius_scale"]) - target_scale) < 1.0e-9
        ):
            xi = R0_VALUES.index(float(row["r0_over_target"]))
            yi = ANGLE_VALUES.index(float(row["initial_velocity_angle_deg"]))
            grid[yi, xi] = 1.0 if bool(row["success"]) else 0.0
    return grid


def format_success_map(ax: plt.Axes, grid: np.ndarray, title: str) -> object:
    cmap = colors.ListedColormap(["#F2F2F2", "#1B9E77"])
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=colors.BoundaryNorm([-0.5, 0.5, 1.5], 2))
    ax.set_xticks(np.arange(len(R0_VALUES)))
    ax.set_xticklabels([f"{x:.5f}" for x in R0_VALUES], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ANGLE_VALUES)))
    ax.set_yticklabels([f"{y:g}" for y in ANGLE_VALUES])
    ax.set_xlabel("r0_over_target")
    ax.set_ylabel("initial_velocity_angle_deg")
    ax.set_title(title)
    return image


def save_plots(rows: Sequence[Dict[str, object]], ranking: Sequence[Dict[str, object]]) -> None:
    save_variant_bar(ranking, "success_count", SUCCESS_PNG, "Strict success count by pre-window variant", "count over 270 regimes", "#1B9E77")
    save_variant_bar(ranking, "capture_count", CAPTURE_PNG, "CAPTURE count by pre-window variant", "count over 270 regimes", "#386CB0")
    save_variant_bar(ranking, "near_miss_count", NEAR_MISS_PNG, "Near-miss count by pre-window variant", "failed near-misses over 270 regimes", "#D95F02")

    prewindow_rows = [row for row in ranking if str(row["variant_name"]).startswith("prewindow_")]
    best_prewindow = prewindow_rows[0] if prewindow_rows else ranking[0]
    best_prewindow_name = str(best_prewindow["variant_name"])

    best_grid = success_grid(rows, best_prewindow_name)
    fig, ax = plt.subplots(figsize=(8.0, 5.1))
    image = format_success_map(ax, best_grid, f"{best_prewindow_name} success map (thrust=10000, target scale=1.00)")
    fig.colorbar(image, ax=ax, ticks=[0, 1], fraction=0.045, pad=0.03)
    fig.tight_layout()
    fig.savefig(BEST_MAP_PNG, dpi=220)
    plt.close(fig)

    names = ["original_ws1", "adaptive_soft", best_prewindow_name]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharey=True)
    image = None
    for ax, name in zip(axes, names):
        image = format_success_map(ax, success_grid(rows, name), name)
    fig.suptitle("Original WS-1 vs adaptive_soft vs best pre-window shaping")
    fig.colorbar(image, ax=axes.ravel().tolist(), ticks=[0, 1], fraction=0.03, pad=0.02)
    fig.savefig(COMPARISON_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(rows: Sequence[Dict[str, object]], ranking: Sequence[Dict[str, object]]) -> None:
    original = next(row for row in ranking if row["variant_name"] == "original_ws1")
    adaptive = next(row for row in ranking if row["variant_name"] == "adaptive_soft")
    prewindow_rows = [row for row in ranking if str(row["variant_name"]).startswith("prewindow_")]
    best_prewindow = prewindow_rows[0] if prewindow_rows else ranking[0]
    best_overall = ranking[0]

    success_delta = int(best_prewindow["success_count"]) - int(adaptive["success_count"])
    capture_delta = int(best_prewindow["capture_count"]) - int(adaptive["capture_count"])
    near_delta = int(best_prewindow["near_miss_count"]) - int(adaptive["near_miss_count"])
    mean_min_delta = float(best_prewindow["mean_minimum_abs_radius_error"]) - float(adaptive["mean_minimum_abs_radius_error"])
    shifted = int(best_prewindow["gained_vs_adaptive_soft"]) + int(best_prewindow["lost_vs_adaptive_soft"])
    widened = int(best_prewindow["gained_vs_adaptive_soft"]) > int(best_prewindow["lost_vs_adaptive_soft"])
    if success_delta > 0:
        improvement_text = "yes"
    elif near_delta < 0 or mean_min_delta < 0.0:
        improvement_text = "partially: quality improved without increasing strict success"
    else:
        improvement_text = "no"
    if success_delta > 0:
        count_text = f"increases strict success by `{success_delta:+d}`"
    elif near_delta < 0:
        count_text = f"does not increase strict success, but reduces near-misses by `{near_delta:+d}`"
    else:
        count_text = f"does not increase strict success or reduce near-misses; near-miss delta is `{near_delta:+d}`"
    window_text = "widen" if widened else "shift"

    lines = [
        "# Phase 7 Pre-Window Trajectory Shaping Summary",
        "",
        "## Setup",
        "",
        "- Scope: 2D Python-only pre-window shaping before WS activation.",
        "- Grid: same 270 regimes from Phase 6.5-6.7.",
        "- WS band: `8e-5` relative radius error.",
        "- Pre-window band: `8e-5` to `5e-4` relative radius error.",
        "- Outside the pre-window band, DESCENT remains baseline retrograde.",
        "- Inside the WS band, the same Phase 6.7 WS candidate structure and CAPTURE/LOCK logic are used.",
        "",
        "## Ranking Result",
        "",
        f"- Original WS-1: success `{original['success_count']}`, CAPTURE `{original['capture_count']}`, near-miss `{original['near_miss_count']}`.",
        f"- Adaptive soft reference: success `{adaptive['success_count']}`, CAPTURE `{adaptive['capture_count']}`, near-miss `{adaptive['near_miss_count']}`.",
        f"- Best pre-window variant: `{best_prewindow['variant_name']}` with success `{best_prewindow['success_count']}`, CAPTURE `{best_prewindow['capture_count']}`, near-miss `{best_prewindow['near_miss_count']}`.",
        f"- Best overall by ranking: `{best_overall['variant_name']}`.",
        "",
        "## Answers",
        "",
        f"1. Does pre-window shaping improve over adaptive_soft? `{improvement_text}`. Best pre-window delta vs adaptive_soft: success `{success_delta:+d}`, CAPTURE `{capture_delta:+d}`, near-miss `{near_delta:+d}`, mean minimum radius error `{mean_min_delta:+.3e}`.",
        f"2. Does it increase success count or only reduce near-misses? It {count_text}.",
        f"3. Which shaping strategy works best? `{best_prewindow['variant_name']}` by success count, CAPTURE count, near-miss count, and mean minimum radius error.",
        f"4. Does it widen the capture window or shift it? It appears to `{window_text}` the success set relative to adaptive_soft: gained `{best_prewindow['gained_vs_adaptive_soft']}`, lost `{best_prewindow['lost_vs_adaptive_soft']}`, retained `{best_prewindow['retained_vs_adaptive_soft']}`, changed memberships `{shifted}`.",
        "5. Best next 2D Python-only step: keep the same evaluator and run a small parameter refinement around the best pre-window family, using gained/lost membership vs adaptive_soft as the primary diagnostic before changing any CAPTURE/LOCK logic.",
        "",
        "## Caution",
        "",
        "This phase changes only DESCENT action selection in the pre-window band. It does not alter physics, PPO, learned experiments, CAPTURE/LOCK equations, or Phase 3-6.7 output directories.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def iter_cases(variants: Sequence[Phase7Variant]) -> Iterable[tuple[Phase7Variant, float, float, float, float]]:
    for variant in variants:
        for target_scale in TARGET_RADIUS_SCALES:
            for thrust in THRUST_VALUES:
                for angle in ANGLE_VALUES:
                    for r0 in R0_VALUES:
                        yield variant, r0, angle, thrust, target_scale


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = define_variants()
    rows = load_rows()
    done = completed_keys(rows)
    cases = list(iter_cases(variants))
    total = len(cases)
    pending = [(idx, case) for idx, case in enumerate(cases, start=1) if (case[0].name, case[1], case[2], case[3], case[4]) not in done]

    if pending:
        workers = recommended_workers()
        print(f"phase7 pending={len(pending)} completed={len(done)} total={total} workers={workers}")
        if workers == 1:
            for idx, (variant, r0, angle, thrust, target_scale) in pending:
                row = rollout_case(variant, r0, angle, thrust, target_scale)
                append_row(GRID_CSV_PATH, row)
                rows.append(row)
                done.add((variant.name, r0, angle, thrust, target_scale))
                print(
                    f"phase7 {idx}/{total} variant={variant.name} r0={r0:.5f} angle={angle:g} "
                    f"thrust={thrust:g} target={target_scale:.2f} success={row['success']} "
                    f"capture={row['capture_entered']} near_miss={row['near_miss']} "
                    f"min_abs_rerr={row['minimum_abs_radius_error']:.3e}"
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_case = {
                    executor.submit(rollout_case, variant, r0, angle, thrust, target_scale): (idx, variant, r0, angle, thrust, target_scale)
                    for idx, (variant, r0, angle, thrust, target_scale) in pending
                }
                completed_now = 0
                for future in as_completed(future_to_case):
                    idx, variant, r0, angle, thrust, target_scale = future_to_case[future]
                    row = future.result()
                    append_row(GRID_CSV_PATH, row)
                    rows.append(row)
                    done.add((variant.name, r0, angle, thrust, target_scale))
                    completed_now += 1
                    print(
                        f"phase7 {idx}/{total} done_now={completed_now}/{len(pending)} variant={variant.name} "
                        f"r0={r0:.5f} angle={angle:g} thrust={thrust:g} target={target_scale:.2f} "
                        f"success={row['success']} capture={row['capture_entered']} near_miss={row['near_miss']} "
                        f"min_abs_rerr={row['minimum_abs_radius_error']:.3e}"
                    )
    else:
        print(f"phase7 no pending rows; completed={len(done)} total={total}")

    rows = load_rows()
    ranking = rank_rows(rows, variants)
    write_ranking(ranking)
    save_plots(rows, ranking)
    write_summary(rows, ranking)
    print(f"Saved Phase 7 pre-window shaping outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
