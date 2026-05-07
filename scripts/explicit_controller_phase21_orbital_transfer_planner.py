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
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase21_orbital_transfer_planner"
MPLCONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig


PHASE20_CSV = PROJECT_ROOT / "analysis" / "phase20_predictive_planner" / "reduced_grid_results.csv"
RESULTS_CSV = OUTPUT_DIR / "phase21_results.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
TRAJECTORY_DIR = OUTPUT_DIR / "trajectories"
MODE_USAGE_PNG = OUTPUT_DIR / "transfer_mode_usage.png"
ENERGY_ERROR_PNG = OUTPUT_DIR / "energy_error_vs_time.png"
ANGULAR_MOMENTUM_ERROR_PNG = OUTPUT_DIR / "angular_momentum_error_vs_time.png"
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
POST_CROSS_WINDOW = 80
RECOVERABLE_VR_RATIO = 2.0e-2
RECOVERABLE_VT_RATIO = 2.5e-1
RECOVERABLE_R_RATIO = 2.5e-3

ENERGY_TOL = 2.5e-2
ANGULAR_MOMENTUM_TOL = 4.0e-2
PRECAPTURE_BAND_RATIO = 1.5e-2
ORBIT_CROSS_MARGIN_RATIO = 2.0e-3
MAX_TRANSFER_BURN_STEPS = 12
TRANSFER_COOLDOWN_STEPS = 180

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
    "initial_angular_momentum_error_ratio",
    "final_angular_momentum_error_ratio",
    "mean_energy_error_ratio",
    "mean_angular_momentum_error_ratio",
    "mean_crossing_score",
    "mean_recoverability_score",
    "initial_angular_momentum_ratio",
    "final_angular_momentum_ratio",
    "max_speed_ratio",
    "overspeed",
    "turning_point_count",
    "burn_steps",
    "coast_steps",
    "burn_ratio",
    "transfer_burn_count",
    "safety_coast_steps",
    "classification_counts_json",
]


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    planner_kind: str


@dataclass(frozen=True)
class OrbitalDiagnostics:
    radius: float
    radial_velocity: float
    tangential_velocity: float
    speed: float
    specific_energy: float
    angular_momentum: float
    target_energy: float
    target_angular_momentum: float
    energy_error_ratio: float
    angular_momentum_error_ratio: float
    semi_major_axis: float | None
    periapsis: float | None
    apoapsis: float | None
    vt_error_ratio: float
    r_error_ratio: float


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


def radius_of(pos: Sequence[float]) -> float:
    return math.sqrt(pos[0] * pos[0] + pos[1] * pos[1])


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


def orbital_diagnostics(
    state_x: float,
    state_y: float,
    state_vx: float,
    state_vy: float,
    target_radius: float,
) -> OrbitalDiagnostics:
    _rx, _ry, _tx, _ty, radius, radial_velocity, tangential_velocity, speed = basis(state_x, state_y, state_vx, state_vy)
    speed2 = state_vx * state_vx + state_vy * state_vy
    energy = 0.5 * speed2 - MU / (radius + 1.0e-12)
    angular_momentum = radius * tangential_velocity
    target_energy = -MU / (2.0 * target_radius)
    target_angular_momentum = math.sqrt(MU * target_radius)
    energy_error_ratio = (energy - target_energy) / (abs(target_energy) + 1.0e-12)
    angular_momentum_error_ratio = (angular_momentum - target_angular_momentum) / (abs(target_angular_momentum) + 1.0e-12)
    semi_major_axis: float | None = None
    periapsis: float | None = None
    apoapsis: float | None = None
    if energy < -1.0e-12:
        semi_major_axis = -MU / (2.0 * energy)
        e2 = 1.0 + 2.0 * energy * angular_momentum * angular_momentum / (MU * MU)
        if 0.0 <= e2 < 1.0 and math.isfinite(semi_major_axis):
            eccentricity = math.sqrt(max(0.0, e2))
            periapsis = semi_major_axis * (1.0 - eccentricity)
            apoapsis = semi_major_axis * (1.0 + eccentricity)
    v_circ = math.sqrt(MU / target_radius)
    return OrbitalDiagnostics(
        radius=radius,
        radial_velocity=radial_velocity,
        tangential_velocity=tangential_velocity,
        speed=speed,
        specific_energy=energy,
        angular_momentum=angular_momentum,
        target_energy=target_energy,
        target_angular_momentum=target_angular_momentum,
        energy_error_ratio=energy_error_ratio,
        angular_momentum_error_ratio=angular_momentum_error_ratio,
        semi_major_axis=semi_major_axis,
        periapsis=periapsis,
        apoapsis=apoapsis,
        vt_error_ratio=(tangential_velocity - v_circ) / (v_circ + 1.0e-12),
        r_error_ratio=(radius - target_radius) / target_radius,
    )


def classify_transfer_state(diag: OrbitalDiagnostics, target_radius: float) -> List[str]:
    regimes: List[str] = []
    if diag.energy_error_ratio > ENERGY_TOL:
        regimes.append("too_high_energy")
    elif diag.energy_error_ratio < -ENERGY_TOL:
        regimes.append("too_low_energy")
    if diag.angular_momentum_error_ratio > ANGULAR_MOMENTUM_TOL:
        regimes.append("too_high_angular_momentum")
    elif diag.angular_momentum_error_ratio < -ANGULAR_MOMENTUM_TOL:
        regimes.append("too_low_angular_momentum")
    if diag.r_error_ratio > PREWINDOW_OUTER_RATIO:
        regimes.append("outside_target_orbit")
    elif diag.r_error_ratio < -PREWINDOW_OUTER_RATIO:
        regimes.append("inside_target_orbit")
    if diag.r_error_ratio * diag.radial_velocity < 0.0:
        regimes.append("approaching_target")
    elif abs(diag.radial_velocity) > 1.0e-12:
        regimes.append("leaving_target")
    v_circ = math.sqrt(MU / target_radius)
    if (
        abs(diag.r_error_ratio) < PRECAPTURE_BAND_RATIO
        and abs(diag.radial_velocity) / (v_circ + 1.0e-12) < 4.0e-2
        and abs(diag.vt_error_ratio) < 3.0e-1
    ):
        regimes.append("near_capture_window")
    return regimes


def orbit_crosses_target(diag: OrbitalDiagnostics, target_radius: float) -> bool:
    if diag.periapsis is None or diag.apoapsis is None:
        return False
    margin = ORBIT_CROSS_MARGIN_RATIO * target_radius
    return diag.periapsis - margin <= target_radius <= diag.apoapsis + margin


def radial_tangential_action(
    state_x: float,
    state_y: float,
    state_vx: float,
    state_vy: float,
    radial_cmd: float,
    tangential_cmd: float,
    max_norm: float,
) -> tuple[float, float]:
    r_hat_x, r_hat_y, t_hat_x, t_hat_y, _radius, _vr, _vt, _speed = basis(state_x, state_y, state_vx, state_vy)
    return normalize_action(
        radial_cmd * r_hat_x + tangential_cmd * t_hat_x,
        radial_cmd * r_hat_y + tangential_cmd * t_hat_y,
        max_norm=max_norm,
    )


def iter_cases(controllers: Sequence[ControllerSpec]) -> Iterable[tuple[ControllerSpec, float, float, float, float]]:
    for controller in controllers:
        for thrust in THRUST_VALUES:
            for angle in ANGLE_VALUES:
                for r0 in R0_VALUES:
                    yield controller, r0, angle, thrust, TARGET_RADIUS_SCALE


def bool_from_csv(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_reference_rows() -> List[Dict[str, object]]:
    if PHASE20_CSV.exists():
        with PHASE20_CSV.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        selected = [row for row in rows if row.get("controller_name") in {"baseline_soft_linear_3e4", "phase20_predictive_planner"}]
        return [normalize_reference_row(row) for row in selected]

    from scripts.explicit_controller_phase20_predictive_planner import ControllerSpec as Phase20ControllerSpec
    from scripts.explicit_controller_phase20_predictive_planner import rollout_case as phase20_rollout_case

    reference_specs = [
        Phase20ControllerSpec("baseline_soft_linear_3e4", "baseline"),
        Phase20ControllerSpec("phase20_predictive_planner", "predictive"),
    ]
    rows: List[Dict[str, object]] = []
    for controller, r0, angle, thrust, target_scale in iter_cases(reference_specs):
        rows.append(normalize_reference_row(phase20_rollout_case(controller, r0, angle, thrust, target_scale)))
    return rows


def normalize_reference_row(row: Dict[str, object]) -> Dict[str, object]:
    out = dict(row)
    if "initial_angular_momentum_error_ratio" not in out:
        ratio0 = float(out.get("initial_angular_momentum_ratio") or 0.0)
        ratiof = float(out.get("final_angular_momentum_ratio") or 0.0)
        out["initial_angular_momentum_error_ratio"] = ratio0 - 1.0 if ratio0 else ""
        out["final_angular_momentum_error_ratio"] = ratiof - 1.0 if ratiof else ""
    out.setdefault("mean_energy_error_ratio", "")
    out.setdefault("mean_angular_momentum_error_ratio", "")
    out.setdefault("mean_crossing_score", out.get("crossing_score", ""))
    out.setdefault("mean_recoverability_score", out.get("recoverability_score", ""))
    out.setdefault("transfer_burn_count", out.get("burn_count", 0))
    out.setdefault("safety_coast_steps", "")
    out.setdefault("classification_counts_json", "{}")
    return out


def rollout_phase21_case(
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
    prev_r_error: float | None = None
    prev_vr_for_turn: float | None = None
    radius_errors: List[float] = []
    vr_values: List[float] = []
    speed_ratios: List[float] = []
    energy_error_values: List[float] = []
    angular_momentum_error_values: List[float] = []
    classification_counts: Counter[str] = Counter()
    planner_mode_counts: Counter[str] = Counter()
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
    termination_reason = "max_steps"
    turning_point_count = 0
    overspeed_hit = False
    burn_steps = 0
    coast_steps = 0
    transfer_burn_count = 0
    safety_coast_steps = 0
    burn_streak = 0
    cooldown_remaining = 0

    trajectory: Dict[str, List[float | str]] = {
        "time_step": [],
        "radius": [],
        "x": [],
        "y": [],
        "energy_error_ratio": [],
        "angular_momentum_error_ratio": [],
        "radial_velocity_ratio": [],
        "vt_error_ratio": [],
        "mode": [],
        "burn_flag": [],
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

    def candidate_score(state_x: float, state_y: float, state_vx: float, state_vy: float) -> float:
        diag = orbital_diagnostics(state_x, state_y, state_vx, state_vy, target_radius)
        score = (
            BASE_PARAMS["score_r"] * abs(diag.r_error_ratio)
            + BASE_PARAMS["score_vr"] * abs(diag.radial_velocity) / (v_circ + 1.0e-12)
            + BASE_PARAMS["score_vt"] * abs(diag.vt_error_ratio)
            + BASE_PARAMS["score_energy"] * abs(diag.energy_error_ratio)
        )
        if diag.r_error_ratio > 0.0 and diag.r_error_ratio < WS_BAND_RATIO and diag.radial_velocity < 0.0:
            score -= BASE_PARAMS["capture_bonus"]
        if abs(diag.r_error_ratio) < 3.0e-4 and abs(diag.radial_velocity) / (v_circ + 1.0e-12) < 2.0e-2:
            score -= BASE_PARAMS["inner_capture_bonus"]
        return score

    def baseline_action(state_vx: float, state_vy: float) -> tuple[float, float]:
        speed = math.sqrt(state_vx * state_vx + state_vy * state_vy)
        if speed <= 1.0e-12:
            return 0.0, 0.0
        return -state_vx / speed, -state_vy / speed

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

    def transfer_planner_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        diag: OrbitalDiagnostics,
        regimes: Sequence[str],
    ) -> tuple[float, float, str]:
        nonlocal burn_streak, cooldown_remaining, transfer_burn_count, safety_coast_steps

        vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
        abs_r_ratio = abs(diag.r_error_ratio)
        speed_ratio = diag.speed / (v_circ + 1.0e-12)
        if speed_ratio > 1.35 or diag.radius < 0.42 * target_radius or diag.radius > 2.25 * target_radius:
            safety_coast_steps += 1
            return 0.0, 0.0, "safety_guard"

        if abs_r_ratio <= PREWINDOW_OUTER_RATIO:
            action_x, action_y = soft_linear_3e4_action(state_x, state_y, state_vx, state_vy, diag.radius, diag.r_error_ratio, abs_r_ratio)
            burn_streak = 0
            cooldown_remaining = 0
            return action_x, action_y, "phase76_handoff"

        if "near_capture_window" in regimes and abs(diag.vt_error_ratio) < 0.25 and abs(vr_ratio) < 0.025:
            action_x, action_y = soft_linear_3e4_action(state_x, state_y, state_vx, state_vy, diag.radius, diag.r_error_ratio, abs_r_ratio)
            burn_streak = 0
            cooldown_remaining = 0
            return action_x, action_y, "phase76_handoff"

        if abs_r_ratio < PRECAPTURE_BAND_RATIO:
            radial_cmd = clamp(-2.2 * vr_ratio - 0.8 * diag.r_error_ratio, -0.40, 0.40)
            tangential_cmd = clamp(-1.6 * diag.vt_error_ratio, -0.48, 0.48)
            action = radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, 0.58)
            burn_streak = 0
            cooldown_remaining = min(cooldown_remaining, 8)
            return action[0], action[1], "pre_capture_alignment"

        crosses = orbit_crosses_target(diag, target_radius)
        approaching = diag.r_error_ratio * diag.radial_velocity < 0.0
        if crosses and approaching and abs_r_ratio < 0.006 and abs(diag.angular_momentum_error_ratio) < 0.18:
            burn_streak = 0
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            return 0.0, 0.0, "coast_to_crossing"

        if abs_r_ratio >= PRECAPTURE_BAND_RATIO:
            if diag.r_error_ratio > 0.0 and diag.angular_momentum_error_ratio < 0.02:
                mode = "angular_momentum_shape"
                tangential_cmd = 1.0
            else:
                mode = "energy_shape"
                tangential_cmd = -1.0 if diag.r_error_ratio > 0.0 else 1.0
            radial_cmd = clamp(-0.08 * vr_ratio, -0.06, 0.06)
            max_norm = clamp(0.05 + 0.90 * min(abs_r_ratio, 0.10), 0.05, 0.14)
        elif abs(diag.angular_momentum_error_ratio) > max(ANGULAR_MOMENTUM_TOL, 0.75 * abs(diag.energy_error_ratio)):
            mode = "angular_momentum_shape"
            tangential_cmd = 1.0 if diag.angular_momentum_error_ratio < 0.0 else -1.0
            radial_cmd = clamp(-0.35 * vr_ratio, -0.18, 0.18)
            max_norm = clamp(0.24 + 1.75 * abs(diag.angular_momentum_error_ratio), 0.24, 0.72)
        elif abs(diag.energy_error_ratio) > ENERGY_TOL:
            mode = "energy_shape"
            tangential_cmd = -1.0 if diag.energy_error_ratio > 0.0 else 1.0
            radial_cmd = clamp(-0.20 * vr_ratio, -0.14, 0.14)
            max_norm = clamp(0.20 + 1.35 * abs(diag.energy_error_ratio), 0.20, 0.62)
        else:
            burn_streak = 0
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            return 0.0, 0.0, "coast_to_crossing"

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            return 0.0, 0.0, "coast_to_crossing"
        if burn_streak == 0:
            transfer_burn_count += 1
        if burn_streak >= MAX_TRANSFER_BURN_STEPS:
            burn_streak = 0
            cooldown_remaining = TRANSFER_COOLDOWN_STEPS
            return 0.0, 0.0, "coast_to_crossing"
        burn_streak += 1
        action = radial_tangential_action(state_x, state_y, state_vx, state_vy, radial_cmd, tangential_cmd, max_norm)
        return action[0], action[1], mode

    initial_diag = orbital_diagnostics(x, y, vx, vy, target_radius)
    initial_energy_error = abs(initial_diag.energy_error_ratio)
    initial_h_error = initial_diag.angular_momentum_error_ratio
    initial_h_ratio = initial_diag.angular_momentum / (initial_diag.target_angular_momentum + 1.0e-12)

    for step in range(1, MAX_STEPS + 1):
        diag = orbital_diagnostics(x, y, vx, vy, target_radius)
        regimes = classify_transfer_state(diag, target_radius)
        classification_counts.update(regimes)
        real_r_error = diag.radius - target_radius
        abs_r_ratio = abs(diag.r_error_ratio)
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
            if (
                abs(diag.r_error_ratio) < cfg.lock_r_threshold_ratio
                and abs(v_r_ratio) < cfg.lock_vr_threshold_ratio
                and abs(v_t_ratio) < cfg.lock_vt_threshold_ratio
            ):
                phase = "LOCK"
        elif phase == "LOCK":
            if abs(diag.r_error_ratio) > cfg.unlock_r_threshold_ratio or abs(v_r_ratio) > cfg.unlock_vr_threshold_ratio:
                phase = "CAPTURE"

        energy_error_values.append(abs(diag.energy_error_ratio))
        angular_momentum_error_values.append(abs(diag.angular_momentum_error_ratio))

        if phase == "DESCENT":
            action_x, action_y, planner_mode = transfer_planner_action(x, y, vx, vy, diag, regimes)
            action_norm = math.sqrt(action_x * action_x + action_y * action_y)
            if planner_mode in {"energy_shape", "angular_momentum_shape", "pre_capture_alignment"} and action_norm > 1.0e-12:
                burn_steps += 1
            else:
                coast_steps += 1
            planner_mode_counts[planner_mode] += 1
        else:
            action_x, action_y = capture_lock_action(x, y, vx, vy, diag.radius, phase)
            planner_mode = phase.lower()

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
            trajectory["mode"].append(planner_mode)
            trajectory["burn_flag"].append(1.0 if planner_mode in {"energy_shape", "angular_momentum_shape", "pre_capture_alignment"} else 0.0)

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
    final_h_ratio = final_diag.angular_momentum / (final_diag.target_angular_momentum + 1.0e-12)
    dominant_mode = max(planner_mode_counts.items(), key=lambda item: item[1])[0] if planner_mode_counts else ""
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
    row: Dict[str, object] = {
        "controller_name": "phase21_orbital_transfer_planner",
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
        "planner_mode_counts_json": json.dumps({key: int(value) for key, value in sorted(planner_mode_counts.items())}, sort_keys=True),
        "initial_energy_error_ratio": float(initial_energy_error),
        "final_energy_error_ratio": float(abs(final_diag.energy_error_ratio)),
        "initial_angular_momentum_error_ratio": float(initial_h_error),
        "final_angular_momentum_error_ratio": float(final_diag.angular_momentum_error_ratio),
        "mean_energy_error_ratio": float(np.mean(energy_error_values)) if energy_error_values else 0.0,
        "mean_angular_momentum_error_ratio": float(np.mean(angular_momentum_error_values)) if angular_momentum_error_values else 0.0,
        "mean_crossing_score": crossing_score,
        "mean_recoverability_score": recoverability_score,
        "initial_angular_momentum_ratio": float(initial_h_ratio),
        "final_angular_momentum_ratio": float(final_h_ratio),
        "max_speed_ratio": float(max(speed_ratios)) if speed_ratios else 0.0,
        "overspeed": bool(overspeed_hit),
        "turning_point_count": int(turning_point_count),
        "burn_steps": int(burn_steps),
        "coast_steps": int(coast_steps),
        "burn_ratio": float(burn_steps / scheduled_steps) if scheduled_steps else 0.0,
        "transfer_burn_count": int(transfer_burn_count),
        "safety_coast_steps": int(safety_coast_steps),
        "classification_counts_json": json.dumps({key: int(value) for key, value in sorted(classification_counts.items())}, sort_keys=True),
    }
    if record_trajectory:
        row["trajectory"] = trajectory
        row["target_radius"] = target_radius
    return row


def run_phase21_rows() -> List[Dict[str, object]]:
    controller = ControllerSpec("phase21_orbital_transfer_planner", "orbital_transfer")
    rows: List[Dict[str, object]] = []
    cases = list(iter_cases([controller]))
    for idx, (_controller, r0, angle, thrust, target_scale) in enumerate(cases, start=1):
        row = rollout_phase21_case(r0, angle, thrust, target_scale)
        rows.append(row)
        print(
            f"phase21 {idx}/{len(cases)} r0={r0:g} angle={angle:g} thrust={thrust:g} "
            f"capture={row['capture_entered']} success={row['success']} crossings={row['radius_crossings_total']} "
            f"recoverable={row['recoverable_crossing']} mode={row['dominant_planner_mode']} "
            f"reason={row['termination_reason']} E0={row['initial_energy_error_ratio']:.4f} "
            f"Ef={row['final_energy_error_ratio']:.4f} H0={row['initial_angular_momentum_error_ratio']:.4f} "
            f"Hf={row['final_angular_momentum_error_ratio']:.4f}"
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
        crossing_scores = numeric_values(subset, "crossing_score")
        recoverability_scores = numeric_values(subset, "recoverability_score")
        energy_errors = numeric_values(subset, "final_energy_error_ratio")
        h_errors = numeric_values(subset, "final_angular_momentum_error_ratio")
        burn_ratios = numeric_values(subset, "burn_ratio")
        stats[name] = {
            "total": len(subset),
            "crossing_cases": sum(bool_from_csv(row.get("crossing_occurs")) for row in subset),
            "recoverable_crossings": sum(bool_from_csv(row.get("recoverable_crossing")) for row in subset),
            "capture": sum(bool_from_csv(row.get("capture_entered")) for row in subset),
            "success": sum(bool_from_csv(row.get("success")) for row in subset),
            "overspeed": sum(bool_from_csv(row.get("overspeed")) or str(row.get("termination_reason")) == "overspeed" for row in subset),
            "mean_energy_error": float(np.mean([abs(value) for value in energy_errors])) if energy_errors else float("nan"),
            "mean_angular_momentum_error": float(np.mean([abs(value) for value in h_errors])) if h_errors else float("nan"),
            "mean_crossing_score": float(np.mean(crossing_scores)) if crossing_scores else float("nan"),
            "mean_recoverability_score": float(np.mean(recoverability_scores)) if recoverability_scores else float("nan"),
            "burn_ratio": float(np.mean(burn_ratios)) if burn_ratios else 0.0,
        }
    return stats


def phase21_mode_counts(rows: Sequence[Dict[str, object]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if str(row.get("controller_name")) != "phase21_orbital_transfer_planner":
            continue
        payload = json.loads(str(row.get("planner_mode_counts_json") or "{}"))
        for key, value in payload.items():
            counts[key] += int(value)
    return counts


def save_capture_crossing_comparison(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]]) -> None:
    names = ["baseline_soft_linear_3e4", "phase20_predictive_planner", "phase21_orbital_transfer_planner"]
    labels = ["Phase 7.6", "Phase 20", "Phase 21"]
    x = np.arange(len(names), dtype=np.float64)
    width = 0.20
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for offset, key, label, color in [
        (-1.5, "crossing_cases", "crossing cases", "#80B1D3"),
        (-0.5, "recoverable_crossings", "recoverable crossings", "#1B9E77"),
        (0.5, "capture", "CAPTURE", "#FDB462"),
        (1.5, "success", "success", "#B3DE69"),
    ]:
        ax.bar(x + offset * width, [int(stats[name][key]) for name in names], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 48)
    ax.set_ylabel("cases out of 48")
    ax.set_title("Reduced-grid capture and crossing comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CAPTURE_CROSSING_PNG, dpi=220)
    plt.close(fig)


def save_mode_usage_plot(rows: Sequence[Dict[str, object]]) -> None:
    counts = phase21_mode_counts(rows)
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"][: len(labels)])
    ax.set_ylabel("DESCENT steps")
    ax.set_title("Phase 21 transfer mode usage")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(MODE_USAGE_PNG, dpi=220)
    plt.close(fig)


def representative_cases() -> List[tuple[float, float, float]]:
    return [
        (0.98, 150.0, 10000.0),
        (1.00, 150.0, 10000.0),
        (1.02, 165.0, 10000.0),
        (1.05, 175.0, 12000.0),
    ]


def save_diagnostic_plots() -> None:
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    trace_rows = [rollout_phase21_case(r0, angle, thrust, TARGET_RADIUS_SCALE, record_trajectory=True) for r0, angle, thrust in representative_cases()]
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
        fig.suptitle(f"Phase 21 representative trajectory {idx}: {label}")
        fig.tight_layout()
        fig.savefig(TRAJECTORY_DIR / f"phase21_representative_{idx:02d}.png", dpi=220)
        plt.close(fig)

    for ax, title, ylabel in [
        (ax_e, "Phase 21 energy error vs time", "|E - E_target| / |E_target|"),
        (ax_h, "Phase 21 angular momentum error vs time", "|h - h_target| / |h_target|"),
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
    phase21_rows = [row for row in rows if str(row["controller_name"]) == "phase21_orbital_transfer_planner"]
    mode_counts = phase21_mode_counts(rows)
    initial_energy = numeric_values(phase21_rows, "initial_energy_error_ratio")
    final_energy = numeric_values(phase21_rows, "final_energy_error_ratio")
    initial_h = numeric_values(phase21_rows, "initial_angular_momentum_error_ratio")
    final_h = numeric_values(phase21_rows, "final_angular_momentum_error_ratio")
    energy_improved = float(np.mean(final_energy)) < float(np.mean(initial_energy)) if initial_energy and final_energy else False
    h_improved = float(np.mean([abs(v) for v in final_h])) < float(np.mean([abs(v) for v in initial_h])) if initial_h and final_h else False
    crossing_improved = int(phase21["crossing_cases"]) > int(baseline["crossing_cases"])
    recoverable_improved = int(phase21["recoverable_crossings"]) > int(baseline["recoverable_crossings"])
    capture_improved = int(phase21["capture"]) > int(baseline["capture"])
    success_improved = int(phase21["success"]) > int(baseline["success"])
    bottleneck = "transfer design"
    if crossing_improved and not recoverable_improved:
        bottleneck = "capture design after nonrecoverable transfer crossings"
    elif recoverable_improved and not capture_improved:
        bottleneck = "handoff/capture design"
    elif crossing_improved and capture_improved:
        bottleneck = "physical reachability beyond this reduced grid"

    lines = [
        "# Phase 21 Orbital-Class Transfer Planner",
        "",
        "## Scope",
        "",
        "- Python-only 2D explicit transfer-regime planner.",
        "- Physics, CAPTURE/LOCK logic, success definition, and prior phase outputs are unchanged.",
        "- DESCENT uses orbital diagnostics: specific energy, angular momentum, semi-major-axis proxy, periapsis/apoapsis proxy, radial velocity, and tangential velocity error.",
        "- The planner uses burn/coast transfer modes instead of short-horizon local action search.",
        "",
        "## Reduced-Grid Result",
        "",
        "| Controller | Crossing cases | Recoverable crossings | CAPTURE | Success | Overspeed | Mean energy err | Mean h err | Burn ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Phase 7.6 `baseline_soft_linear_3e4` | {baseline['crossing_cases']} | {baseline['recoverable_crossings']} | {baseline['capture']} | {baseline['success']} | {baseline['overspeed']} | {baseline['mean_energy_error']:.4f} | {baseline['mean_angular_momentum_error']:.4f} | {baseline['burn_ratio']:.3f} |",
        f"| Phase 20 `phase20_predictive_planner` | {phase20['crossing_cases']} | {phase20['recoverable_crossings']} | {phase20['capture']} | {phase20['success']} | {phase20['overspeed']} | {phase20['mean_energy_error']:.4f} | {phase20['mean_angular_momentum_error']:.4f} | {phase20['burn_ratio']:.3f} |",
        f"| Phase 21 `phase21_orbital_transfer_planner` | {phase21['crossing_cases']} | {phase21['recoverable_crossings']} | {phase21['capture']} | {phase21['success']} | {phase21['overspeed']} | {phase21['mean_energy_error']:.4f} | {phase21['mean_angular_momentum_error']:.4f} | {phase21['burn_ratio']:.3f} |",
        "",
        "## Phase 21 Diagnostics",
        "",
        f"- Mean initial |energy error|: `{float(np.mean(initial_energy)):.4f}`; mean final |energy error|: `{float(np.mean(final_energy)):.4f}`.",
        f"- Mean initial |angular momentum error|: `{float(np.mean([abs(v) for v in initial_h])):.4f}`; mean final |angular momentum error|: `{float(np.mean([abs(v) for v in final_h])):.4f}`.",
        f"- Mean crossing score: `{phase21['mean_crossing_score']:.4f}`.",
        f"- Mean recoverability score: `{phase21['mean_recoverability_score']:.4f}`.",
        f"- Mode usage: `{json.dumps(dict(mode_counts), sort_keys=True)}`.",
        f"- Transfer burn/coast ratio: `{phase21['burn_ratio']:.4f}`.",
        "",
        "## Research Answers",
        "",
        f"1. Can orbital-class transfer planning create new crossings beyond Phase 7.6? `{'yes' if crossing_improved else 'no'}`. Crossing cases changed from `{baseline['crossing_cases']}` to `{phase21['crossing_cases']}`.",
        f"2. Can energy/angular-momentum shaping create recoverable crossings? `{'yes' if int(phase21['recoverable_crossings']) > 0 else 'no'}`. Recoverable crossings: `{phase21['recoverable_crossings']}`.",
        f"3. Does Phase 21 improve CAPTURE or success? CAPTURE improvement: `{int(phase21['capture']) - int(baseline['capture'])}`; success improvement: `{int(phase21['success']) - int(baseline['success'])}`.",
        f"4. Is the bottleneck now transfer design, capture design, or physical reachability? Current evidence points to `{bottleneck}`.",
        "",
        "## Success Criteria",
        "",
        f"- Minimum success, crossing cases > Phase 7.6: `{'met' if crossing_improved else 'not met'}`.",
        f"- Moderate success, recoverable crossings > 0: `{'met' if int(phase21['recoverable_crossings']) > 0 else 'not met'}`.",
        f"- Strong success, CAPTURE > Phase 7.6: `{'met' if capture_improved else 'not met'}`.",
        f"- Major success, success > Phase 7.6: `{'met' if success_improved else 'not met'}`.",
        "",
        "## Honesty Note",
        "",
        f"- Energy moved toward target values: `{'yes' if energy_improved else 'no'}`.",
        f"- Angular momentum moved toward target values: `{'yes' if h_improved else 'no'}`.",
        "- Negative or mixed results are treated as diagnostic evidence, not forced into a positive claim.",
        "",
        "## Artifacts",
        "",
        "- `phase21_results.csv`",
        "- `transfer_mode_usage.png`",
        "- `energy_error_vs_time.png`",
        "- `angular_momentum_error_vs_time.png`",
        "- `capture_crossing_comparison.png`",
        "- `trajectories/phase21_representative_*.png`",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 21 orbital-class transfer planner benchmark.")
    parser.add_argument("--skip-plots", action="store_true", help="Run benchmark and summary without regenerating plots.")
    args = parser.parse_args()
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_rows = load_reference_rows()
    phase21_rows = run_phase21_rows()
    rows = reference_rows + phase21_rows
    write_results(rows)
    stats = aggregate(rows)
    if not args.skip_plots:
        save_capture_crossing_comparison(rows, stats)
        save_mode_usage_plot(rows)
        save_diagnostic_plots()
    write_summary(rows, stats)
    print(f"Saved Phase 21 orbital-transfer outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
