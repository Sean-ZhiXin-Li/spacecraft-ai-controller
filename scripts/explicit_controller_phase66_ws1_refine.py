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
MPLCONFIG_DIR = PROJECT_ROOT / "analysis" / "phase66_ws1_refine" / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig


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


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase66_ws1_refine"
GRID_CSV_PATH = OUTPUT_DIR / "ws1_refine_grid.csv"
RANKING_CSV_PATH = OUTPUT_DIR / "ws1_refine_ranking.csv"
SUMMARY_PATH = OUTPUT_DIR / "phase66_summary.md"

SUCCESS_PNG = OUTPUT_DIR / "success_by_ws_variant.png"
CAPTURE_PNG = OUTPUT_DIR / "capture_by_ws_variant.png"
NEAR_MISS_PNG = OUTPUT_DIR / "near_miss_by_ws_variant.png"
BEST_MAP_PNG = OUTPUT_DIR / "best_ws_success_map.png"
ORIGINAL_VS_BEST_PNG = OUTPUT_DIR / "original_ws1_vs_best_ws.png"

ORIGINAL_WS1_PARAMS: Dict[str, float] = {
    "near_band_ratio": 8.0e-5,
    "inward_bias": 0.12,
    "reduced_retro_scale": 0.70,
    "score_r": 1.0,
    "score_vr": 0.65,
    "score_vt": 0.45,
    "score_energy": 0.30,
    "capture_bonus": 0.20,
    "inner_capture_bonus": 0.10,
}

FIELDNAMES = [
    "variant_name",
    "variant_params_json",
    "knob",
    "knob_value",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "target_radius_scale",
    "success",
    "capture_entered",
    "lock_entered",
    "crossing_occurs",
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
    "knob",
    "knob_value",
    "near_band_ratio",
    "reduced_retro_scale",
    "score_energy",
    "success_count",
    "capture_count",
    "lock_count",
    "near_miss_count",
    "mean_minimum_abs_radius_error",
    "mean_tail_abs_vr",
    "mean_steps",
]


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def norm2(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def compute_vr(pos: np.ndarray, vel: np.ndarray) -> float:
    radius = norm2(pos)
    return float(np.dot(pos, vel) / (radius + 1.0e-12))


def compute_vt(pos: np.ndarray, vel: np.ndarray) -> float:
    radius = norm2(pos)
    r_hat = pos / (radius + 1.0e-12)
    t_hat = np.array([-r_hat[1], r_hat[0]], dtype=np.float64)
    return float(np.dot(vel, t_hat))


def initial_state(r0_over_target: float, angle_deg: float, target_radius: float) -> tuple[np.ndarray, np.ndarray]:
    pos = np.array([0.0, r0_over_target * target_radius], dtype=np.float64)
    v_mag = math.sqrt(MU / norm2(pos))
    angle = math.radians(angle_deg)
    vel = v_mag * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    return pos, vel


def inside_tolerance(pos: np.ndarray, vel: np.ndarray, target_radius: float) -> bool:
    radius = norm2(pos)
    speed = norm2(vel)
    v_target = math.sqrt(MU / target_radius)
    r_err = abs(radius - target_radius) / target_radius
    v_err = abs(speed - v_target) / v_target
    ur = pos / (radius + 1.0e-8)
    uv = vel / (speed + 1.0e-8)
    ang = abs(float(np.dot(ur, uv)))
    return r_err < STRICT_CFG["tol_r"] and v_err < STRICT_CFG["tol_v"] and ang < STRICT_CFG["tol_ang"]


def env_equivalent_step(pos: np.ndarray, vel: np.ndarray, action: np.ndarray, thrust_scale: float) -> tuple[np.ndarray, np.ndarray]:
    action = np.clip(action, -1.0, 1.0).astype(np.float64)
    thrust = thrust_scale * action
    acc_thrust = thrust / MASS
    radius = norm2(pos)
    acc_gravity = -MU * pos / ((radius**3) + 1.0e-12)
    vel_next = vel + (acc_gravity + acc_thrust) * DT
    pos_next = pos + vel_next * DT
    return pos_next, vel_next


def recommended_workers(limit: int | None = None) -> int:
    cpu_count = os.cpu_count() or 2
    workers = max(1, min(cpu_count - 1, 12))
    if limit is not None:
        workers = min(workers, max(1, int(limit)))
    return workers


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    params: Dict[str, float | str]


class ExplicitPhaseController:
    def __init__(self, *, target_radius: float, spec: ControllerSpec) -> None:
        self.target_radius = float(target_radius)
        self.v_circ = math.sqrt(MU / target_radius)
        self.cfg = OrbitLockConfig()
        self.spec = spec
        self.phase = "DESCENT"
        self.prev_real_r_error: float | None = None
        self.phase_transition_count = 0
        self.window_used_steps = 0

    def set_phase(self, new_phase: str) -> None:
        if new_phase != self.phase:
            self.phase = new_phase
            self.phase_transition_count += 1

    def baseline_descent_action(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        v_norm = norm2(vel)
        if v_norm <= 1.0e-12:
            return np.zeros(2, dtype=np.float64)
        return np.array([-vel[0] / v_norm, -vel[1] / v_norm], dtype=np.float64)

    def capture_lock_action(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        radius = norm2(pos)
        real_r_error = radius - self.target_radius
        v_r = compute_vr(pos, vel)
        v_t_error = compute_vt(pos, vel) - self.v_circ
        real_r_ratio = real_r_error / self.target_radius
        v_r_ratio = v_r / self.v_circ
        v_t_ratio = v_t_error / self.v_circ
        r_hat = pos / (radius + 1.0e-12)
        t_hat = np.array([-r_hat[1], r_hat[0]], dtype=np.float64)

        if self.phase == "CAPTURE":
            radial_cmd = -(
                self.cfg.capture_radial_pos_gain * real_r_ratio
                + self.cfg.capture_radial_vel_gain * v_r_ratio
            )
            tangential_cmd = -(self.cfg.capture_tangential_gain * v_t_ratio)
            radial_cmd = clamp(radial_cmd, -self.cfg.capture_radial_limit, self.cfg.capture_radial_limit)
            tangential_cmd = clamp(
                tangential_cmd,
                -self.cfg.capture_tangential_limit,
                self.cfg.capture_tangential_limit,
            )
        else:
            radial_cmd = -(
                self.cfg.lock_radial_pos_gain * real_r_ratio
                + self.cfg.lock_radial_vel_gain * v_r_ratio
            )
            tangential_cmd = -(self.cfg.lock_tangential_gain * v_t_ratio)
            radial_cmd = clamp(radial_cmd, -self.cfg.lock_radial_limit, self.cfg.lock_radial_limit)
            tangential_cmd = clamp(
                tangential_cmd,
                -self.cfg.lock_tangential_limit,
                self.cfg.lock_tangential_limit,
            )
        return np.clip(radial_cmd * r_hat + tangential_cmd * t_hat, -1.0, 1.0)

    def candidate_score(self, pos: np.ndarray, vel: np.ndarray, target_radius: float) -> float:
        radius = norm2(pos)
        v_target = math.sqrt(MU / target_radius)
        v_r = compute_vr(pos, vel)
        v_t_error = compute_vt(pos, vel) - v_target
        r_err_ratio = abs(radius - target_radius) / target_radius
        v_r_ratio = abs(v_r) / v_target
        v_t_ratio = abs(v_t_error) / v_target
        energy = 0.5 * norm2(vel) ** 2 - MU / (radius + 1.0e-12)
        target_energy = -MU / (2.0 * target_radius)
        energy_ratio = abs(energy - target_energy) / (abs(target_energy) + 1.0e-12)
        params = self.spec.params
        score = (
            float(params["score_r"]) * r_err_ratio
            + float(params["score_vr"]) * v_r_ratio
            + float(params["score_vt"]) * v_t_ratio
            + float(params["score_energy"]) * energy_ratio
        )
        if (radius - target_radius) > 0.0 and (radius - target_radius) / target_radius < float(params["near_band_ratio"]) and v_r < 0.0:
            score -= float(params["capture_bonus"])
        if r_err_ratio < 3.0e-4 and v_r_ratio < 2.0e-2:
            score -= float(params["inner_capture_bonus"])
        return float(score)

    def window_seeking_action(self, pos: np.ndarray, vel: np.ndarray, thrust_scale: float) -> np.ndarray:
        radius = norm2(pos)
        r_hat = pos / (radius + 1.0e-12)
        retro = self.baseline_descent_action(pos, vel)
        inward_bias = np.clip(retro - float(self.spec.params["inward_bias"]) * r_hat, -1.0, 1.0)
        reduced = float(self.spec.params["reduced_retro_scale"]) * retro
        candidates = [
            ("pure_retro", retro),
            ("retro_plus_inward", inward_bias),
            ("reduced_retro", reduced),
        ]
        best_action = retro
        best_score = float("inf")
        for _, candidate in candidates:
            candidate_norm = norm2(candidate)
            if candidate_norm > 1.0:
                candidate = candidate / candidate_norm
            next_pos, next_vel = env_equivalent_step(pos, vel, candidate, thrust_scale)
            score = self.candidate_score(next_pos, next_vel, self.target_radius)
            if score < best_score:
                best_score = score
                best_action = candidate
        self.window_used_steps += 1
        return np.clip(best_action, -1.0, 1.0)


@dataclass(frozen=True)
class WSVariant:
    name: str
    params: Dict[str, float]
    knob: str
    knob_value: float | str

    def spec(self) -> ControllerSpec:
        return ControllerSpec(self.name, dict(self.params))


def with_param(name: str, knob: str, value: float) -> WSVariant:
    params = dict(ORIGINAL_WS1_PARAMS)
    params[knob] = float(value)
    return WSVariant(name=name, params=params, knob=knob, knob_value=float(value))


def define_variants() -> List[WSVariant]:
    variants = [
        WSVariant("original_ws1", dict(ORIGINAL_WS1_PARAMS), "reference", "original"),
        with_param("near_band_5e-5", "near_band_ratio", 5.0e-5),
        with_param("near_band_1p2e-4", "near_band_ratio", 1.2e-4),
        with_param("near_band_2e-4", "near_band_ratio", 2.0e-4),
        with_param("retro_scale_0p60", "reduced_retro_scale", 0.60),
        with_param("retro_scale_0p80", "reduced_retro_scale", 0.80),
        with_param("energy_weight_0p20", "score_energy", 0.20),
        with_param("energy_weight_0p50", "score_energy", 0.50),
    ]
    if len(variants) > 16:
        raise RuntimeError(f"Phase 6.6 variant cap exceeded: {len(variants)}")
    return variants


class Phase66WSController(ExplicitPhaseController):
    def act(self, pos: np.ndarray, vel: np.ndarray, thrust_scale: float) -> tuple[np.ndarray, str]:
        radius = norm2(pos)
        real_r_error = radius - self.target_radius
        v_r = compute_vr(pos, vel)
        v_t_error = self._compute_vt_error(pos, vel)
        real_r_ratio = real_r_error / self.target_radius
        v_r_ratio = v_r / self.v_circ
        v_t_ratio = v_t_error / self.v_circ

        crossed_now = False
        if self.prev_real_r_error is not None:
            crossed_now = (self.prev_real_r_error > 0.0 and real_r_error <= 0.0) or (
                self.prev_real_r_error < 0.0 and real_r_error >= 0.0
            )
        self.prev_real_r_error = real_r_error

        if self.phase == "DESCENT" and crossed_now:
            self.set_phase("CAPTURE")
        elif self.phase == "CAPTURE":
            if (
                abs(real_r_ratio) < self.cfg.lock_r_threshold_ratio
                and abs(v_r_ratio) < self.cfg.lock_vr_threshold_ratio
                and abs(v_t_ratio) < self.cfg.lock_vt_threshold_ratio
            ):
                self.set_phase("LOCK")
        elif self.phase == "LOCK":
            if abs(real_r_ratio) > self.cfg.unlock_r_threshold_ratio or abs(v_r_ratio) > self.cfg.unlock_vr_threshold_ratio:
                self.set_phase("CAPTURE")

        if self.phase == "DESCENT":
            if abs(real_r_ratio) < float(self.spec.params["near_band_ratio"]):
                return self.window_seeking_action(pos, vel, thrust_scale), self.phase
            return self.baseline_descent_action(pos, vel), self.phase
        return self.capture_lock_action(pos, vel), self.phase

    def _compute_vt_error(self, pos: np.ndarray, vel: np.ndarray) -> float:
        radius = norm2(pos)
        r_hat = pos / (radius + 1.0e-12)
        t_hat = np.array([-r_hat[1], r_hat[0]], dtype=np.float64)
        return float(np.dot(vel, t_hat)) - self.v_circ


def rollout_case(variant: WSVariant, r0: float, angle: float, thrust_scale: float, target_scale: float) -> Dict[str, object]:
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

    def scalar_clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def vt_error_for(state_x: float, state_y: float, state_vx: float, state_vy: float, state_radius: float) -> float:
        inv_r = 1.0 / (state_radius + 1.0e-12)
        return state_vx * (-state_y * inv_r) + state_vy * (state_x * inv_r) - v_circ

    def env_step(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        action_x: float,
        action_y: float,
    ) -> tuple[float, float, float, float]:
        action_x = scalar_clip(action_x, -1.0, 1.0)
        action_y = scalar_clip(action_y, -1.0, 1.0)
        state_radius = math.sqrt(state_x * state_x + state_y * state_y)
        denom = state_radius**3 + 1.0e-12
        ax = -MU * state_x / denom + thrust_scale * action_x / MASS
        ay = -MU * state_y / denom + thrust_scale * action_y / MASS
        next_vx = state_vx + ax * DT
        next_vy = state_vy + ay * DT
        next_x = state_x + next_vx * DT
        next_y = state_y + next_vy * DT
        return next_x, next_y, next_vx, next_vy

    def candidate_score(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
    ) -> float:
        state_radius = math.sqrt(state_x * state_x + state_y * state_y)
        inv_r = 1.0 / (state_radius + 1.0e-12)
        speed = math.sqrt(state_vx * state_vx + state_vy * state_vy)
        vr_state = (state_x * state_vx + state_y * state_vy) * inv_r
        vt_state = state_vx * (-state_y * inv_r) + state_vy * (state_x * inv_r)
        v_t_error = vt_state - v_circ
        r_err_ratio = abs(state_radius - target_radius) / target_radius
        v_r_ratio = abs(vr_state) / v_circ
        v_t_ratio = abs(v_t_error) / v_circ
        energy = 0.5 * speed * speed - MU / (state_radius + 1.0e-12)
        target_energy = -MU / (2.0 * target_radius)
        energy_ratio = abs(energy - target_energy) / (abs(target_energy) + 1.0e-12)
        score = (
            float(params["score_r"]) * r_err_ratio
            + float(params["score_vr"]) * v_r_ratio
            + float(params["score_vt"]) * v_t_ratio
            + float(params["score_energy"]) * energy_ratio
        )
        if (state_radius - target_radius) > 0.0 and (state_radius - target_radius) / target_radius < float(params["near_band_ratio"]) and vr_state < 0.0:
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
            radial_cmd = scalar_clip(radial_cmd, -cfg.capture_radial_limit, cfg.capture_radial_limit)
            tangential_cmd = scalar_clip(tangential_cmd, -cfg.capture_tangential_limit, cfg.capture_tangential_limit)
        else:
            radial_cmd = -(cfg.lock_radial_pos_gain * real_r_ratio + cfg.lock_radial_vel_gain * vr_ratio)
            tangential_cmd = -(cfg.lock_tangential_gain * vt_ratio)
            radial_cmd = scalar_clip(radial_cmd, -cfg.lock_radial_limit, cfg.lock_radial_limit)
            tangential_cmd = scalar_clip(tangential_cmd, -cfg.lock_tangential_limit, cfg.lock_tangential_limit)
        return (
            scalar_clip(radial_cmd * r_hat_x + tangential_cmd * t_hat_x, -1.0, 1.0),
            scalar_clip(radial_cmd * r_hat_y + tangential_cmd * t_hat_y, -1.0, 1.0),
        )

    def window_action(
        state_x: float,
        state_y: float,
        state_vx: float,
        state_vy: float,
        state_radius: float,
    ) -> tuple[float, float]:
        inv_r = 1.0 / (state_radius + 1.0e-12)
        r_hat_x = state_x * inv_r
        r_hat_y = state_y * inv_r
        retro_x, retro_y = baseline_action(state_vx, state_vy)
        candidates = [
            (retro_x, retro_y),
            (
                scalar_clip(retro_x - float(params["inward_bias"]) * r_hat_x, -1.0, 1.0),
                scalar_clip(retro_y - float(params["inward_bias"]) * r_hat_y, -1.0, 1.0),
            ),
            (float(params["reduced_retro_scale"]) * retro_x, float(params["reduced_retro_scale"]) * retro_y),
        ]
        best_action = (retro_x, retro_y)
        best_score = float("inf")
        for candidate_x, candidate_y in candidates:
            candidate_norm = math.sqrt(candidate_x * candidate_x + candidate_y * candidate_y)
            if candidate_norm > 1.0:
                candidate_x /= candidate_norm
                candidate_y /= candidate_norm
            next_x, next_y, next_vx, next_vy = env_step(state_x, state_y, state_vx, state_vy, candidate_x, candidate_y)
            score = candidate_score(next_x, next_y, next_vx, next_vy)
            if score < best_score:
                best_score = score
                best_action = (candidate_x, candidate_y)
        return best_action

    for step in range(1, MAX_STEPS + 1):
        radius = math.sqrt(x * x + y * y)
        real_r_error = radius - target_radius
        inv_r = 1.0 / (radius + 1.0e-12)
        vr_current = (x * vx + y * vy) * inv_r
        vt_error = vt_error_for(x, y, vx, vy, radius)
        real_r_ratio = real_r_error / target_radius
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
            if abs(real_r_ratio) < float(params["near_band_ratio"]):
                action_x, action_y = window_action(x, y, vx, vy, radius)
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

        v_target = v_circ
        r_err_now = abs(r_error) / target_radius
        v_err_now = abs(speed - v_target) / v_target
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
        overspeed = speed > 1.90 * v_target
        too_close = radius < 0.35 * target_radius
        radial_stall_fail = radial_stall_counter >= 800
        terminated = bool(success or out_range or overspeed or too_close or radial_stall_fail)
        truncated = bool(step >= MAX_STEPS and not terminated)
        if terminated or truncated:
            break

    rerr = np.asarray(radius_errors, dtype=np.float64)
    vr_arr = np.asarray(vr_values, dtype=np.float64)
    tail = vr_arr[-min(TAIL_LEN, len(vr_arr)) :] if len(vr_arr) else np.asarray([0.0])
    return {
        "variant_name": variant.name,
        "variant_params_json": json.dumps(variant.params, sort_keys=True),
        "knob": variant.knob,
        "knob_value": variant.knob_value,
        "r0_over_target": r0,
        "initial_velocity_angle_deg": angle,
        "thrust_scale": thrust_scale,
        "target_radius_scale": target_scale,
        "success": bool(success),
        "capture_entered": bool(capture_entered),
        "lock_entered": bool(lock_entered),
        "crossing_occurs": bool(radius_crossings_total > 0),
        "radius_crossings_total": int(radius_crossings_total),
        "first_crossing_step": first_crossing_step,
        "minimum_abs_radius_error": float(np.min(np.abs(rerr))) if len(rerr) else 0.0,
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
                "knob_value": row["knob_value"],
                "r0_over_target": float(row["r0_over_target"]),
                "initial_velocity_angle_deg": float(row["initial_velocity_angle_deg"]),
                "thrust_scale": float(row["thrust_scale"]),
                "target_radius_scale": float(row["target_radius_scale"]),
                "success": parse_bool(row["success"]),
                "capture_entered": parse_bool(row["capture_entered"]),
                "lock_entered": parse_bool(row["lock_entered"]),
                "crossing_occurs": parse_bool(row["crossing_occurs"]),
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


def is_near_miss(row: Dict[str, object]) -> bool:
    target_radius = DEFAULT_TARGET_RADIUS * float(row["target_radius_scale"])
    return (not bool(row["success"])) and float(row["minimum_abs_radius_error"]) / target_radius <= NEAR_MISS_REL


def rank_rows(rows: Sequence[Dict[str, object]], variants: Sequence[WSVariant]) -> List[Dict[str, object]]:
    ranking: List[Dict[str, object]] = []
    for variant in variants:
        subset = [row for row in rows if row["variant_name"] == variant.name]
        if not subset:
            continue
        ranking.append(
            {
                "variant_name": variant.name,
                "knob": variant.knob,
                "knob_value": variant.knob_value,
                "near_band_ratio": variant.params["near_band_ratio"],
                "reduced_retro_scale": variant.params["reduced_retro_scale"],
                "score_energy": variant.params["score_energy"],
                "success_count": sum(bool(row["success"]) for row in subset),
                "capture_count": sum(bool(row["capture_entered"]) for row in subset),
                "lock_count": sum(bool(row["lock_entered"]) for row in subset),
                "near_miss_count": sum(is_near_miss(row) for row in subset),
                "mean_minimum_abs_radius_error": float(np.mean([float(row["minimum_abs_radius_error"]) for row in subset])),
                "mean_tail_abs_vr": float(np.mean([float(row["tail_mean_abs_vr"]) for row in subset])),
                "mean_steps": float(np.mean([float(row["steps"]) for row in subset])),
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
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(values + [1.0]) * 0.01, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def success_grid(rows: Sequence[Dict[str, object]], variant_name: str, thrust: float = 10000.0, target_scale: float = 1.00) -> np.ndarray:
    subset = [
        row
        for row in rows
        if row["variant_name"] == variant_name
        and abs(float(row["thrust_scale"]) - thrust) < 1.0e-9
        and abs(float(row["target_radius_scale"]) - target_scale) < 1.0e-9
    ]
    grid = np.zeros((len(ANGLE_VALUES), len(R0_VALUES)), dtype=np.float64)
    for row in subset:
        xi = R0_VALUES.index(float(row["r0_over_target"]))
        yi = ANGLE_VALUES.index(float(row["initial_velocity_angle_deg"]))
        grid[yi, xi] = 1.0 if bool(row["success"]) else 0.0
    return grid


def format_success_map(ax: plt.Axes, grid: np.ndarray, title: str) -> None:
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


def save_maps(rows: Sequence[Dict[str, object]], best_variant_name: str) -> None:
    best_grid = success_grid(rows, best_variant_name)
    fig, ax = plt.subplots(figsize=(8.0, 5.1))
    image = format_success_map(ax, best_grid, f"{best_variant_name} success map (thrust=10000, target scale=1.00)")
    fig.colorbar(image, ax=ax, ticks=[0, 1], fraction=0.045, pad=0.03)
    fig.tight_layout()
    fig.savefig(BEST_MAP_PNG, dpi=220)
    plt.close(fig)

    original_grid = success_grid(rows, "original_ws1")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    image = format_success_map(axes[0], original_grid, "original_ws1")
    format_success_map(axes[1], best_grid, best_variant_name)
    fig.suptitle("Original WS-1 vs best refined WS variant")
    fig.colorbar(image, ax=axes.ravel().tolist(), ticks=[0, 1], fraction=0.035, pad=0.02)
    fig.savefig(ORIGINAL_VS_BEST_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_plots(rows: Sequence[Dict[str, object]], ranking: Sequence[Dict[str, object]]) -> None:
    save_variant_bar(ranking, "success_count", SUCCESS_PNG, "Strict success count by WS variant", "count over 270 regimes", "#1B9E77")
    save_variant_bar(ranking, "capture_count", CAPTURE_PNG, "CAPTURE count by WS variant", "count over 270 regimes", "#386CB0")
    save_variant_bar(ranking, "near_miss_count", NEAR_MISS_PNG, "Near-miss count by WS variant", "failed near-misses over 270 regimes", "#D95F02")
    if ranking:
        save_maps(rows, str(ranking[0]["variant_name"]))


def variant_case_key(row: Dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(row["r0_over_target"]),
        float(row["initial_velocity_angle_deg"]),
        float(row["thrust_scale"]),
        float(row["target_radius_scale"]),
    )


def knob_effect_lines(ranking: Sequence[Dict[str, object]]) -> tuple[str, List[str]]:
    original = next(row for row in ranking if row["variant_name"] == "original_ws1")
    groups = {"near_band_ratio": [], "reduced_retro_scale": [], "score_energy": []}
    for row in ranking:
        if row["knob"] in groups:
            groups[str(row["knob"])].append(row)

    effect_summaries: List[tuple[str, int, int, int]] = []
    lines: List[str] = []
    for knob, rows in groups.items():
        if not rows:
            continue
        success_values = [int(row["success_count"]) for row in rows] + [int(original["success_count"])]
        capture_values = [int(row["capture_count"]) for row in rows] + [int(original["capture_count"])]
        near_values = [int(row["near_miss_count"]) for row in rows] + [int(original["near_miss_count"])]
        success_range = max(success_values) - min(success_values)
        capture_range = max(capture_values) - min(capture_values)
        near_range = max(near_values) - min(near_values)
        effect_summaries.append((knob, success_range, capture_range, near_range))
        best = min(
            rows + [original],
            key=lambda row: (
                -int(row["success_count"]),
                -int(row["capture_count"]),
                int(row["near_miss_count"]),
                float(row["mean_minimum_abs_radius_error"]),
            ),
        )
        lines.append(
            f"- `{knob}`: success range `{success_range}`, capture range `{capture_range}`, near-miss range `{near_range}`; best one-factor setting `{best['variant_name']}`."
        )

    effect_summaries.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
    return effect_summaries[0][0], lines


def write_summary(rows: Sequence[Dict[str, object]], ranking: Sequence[Dict[str, object]]) -> None:
    original_rank = next(row for row in ranking if row["variant_name"] == "original_ws1")
    best = ranking[0]
    original_rows = {variant_case_key(row): row for row in rows if row["variant_name"] == "original_ws1"}
    best_rows = {variant_case_key(row): row for row in rows if row["variant_name"] == best["variant_name"]}
    original_success_keys = {key for key, row in original_rows.items() if bool(row["success"])}
    best_success_keys = {key for key, row in best_rows.items() if bool(row["success"])}
    gained = len(best_success_keys - original_success_keys)
    lost = len(original_success_keys - best_success_keys)
    retained = len(best_success_keys & original_success_keys)
    widened_or_shifted = "widened" if gained > 0 and lost == 0 else "shifted" if gained > 0 and lost > 0 else "unchanged"
    most_sensitive_knob, effect_lines = knob_effect_lines(ranking)

    success_delta = int(best["success_count"]) - int(original_rank["success_count"])
    capture_delta = int(best["capture_count"]) - int(original_rank["capture_count"])
    near_delta = int(best["near_miss_count"]) - int(original_rank["near_miss_count"])
    mean_min_delta = float(best["mean_minimum_abs_radius_error"]) - float(original_rank["mean_minimum_abs_radius_error"])

    lines = [
        "# Phase 6.6 WS-1 Refinement Summary",
        "",
        "## Setup",
        "",
        "- Scope: local 2D WS-1 parameter refinement only; no learning, PPO retraining, physics changes, 3D, or multi-orbit cases.",
        "- Variant family: original WS-1 plus one-knob-at-a-time changes for `near_band_ratio`, `reduced_retro_scale`, and `score_energy`.",
        f"- Total variants: `{len(ranking)}`; regimes per variant: `{len(R0_VALUES) * len(ANGLE_VALUES) * len(THRUST_VALUES) * len(TARGET_RADIUS_SCALES)}`.",
        "- Ranking order: success count, CAPTURE count, lower near-miss count, lower mean minimum radius error.",
        "",
        "## Result",
        "",
        f"- Original WS-1: success `{original_rank['success_count']}`, CAPTURE `{original_rank['capture_count']}`, near-miss `{original_rank['near_miss_count']}`, mean min |radius error| `{float(original_rank['mean_minimum_abs_radius_error']):.3e}`.",
        f"- Best variant: `{best['variant_name']}` with success `{best['success_count']}`, CAPTURE `{best['capture_count']}`, near-miss `{best['near_miss_count']}`, mean min |radius error| `{float(best['mean_minimum_abs_radius_error']):.3e}`.",
        f"- Delta vs original: success `{success_delta:+d}`, CAPTURE `{capture_delta:+d}`, near-miss `{near_delta:+d}`, mean min |radius error| `{mean_min_delta:+.3e}`.",
        "",
        "## Knob Sensitivity",
        "",
        *effect_lines,
        "",
        "## Answers",
        "",
        f"1. Small parameter refinement {'improved WS-1' if success_delta > 0 else 'did not improve WS-1 success'} on this grid. The best strict-success delta is `{success_delta:+d}`.",
        f"2. The most consequential knob by one-factor success/capture/near-miss spread is `{most_sensitive_knob}`.",
        f"3. The best variant changes success by `{success_delta:+d}`, CAPTURE by `{capture_delta:+d}`, and near-misses by `{near_delta:+d}`; this indicates {'a real CAPTURE/success change' if success_delta or capture_delta else 'mainly near-miss/quality movement'} rather than an inferred post-CAPTURE effect.",
        f"4. The success set is `{widened_or_shifted}` relative to original WS-1: retained `{retained}`, gained `{gained}`, lost `{lost}` regimes.",
        "5. Best next step: stay in this same 2D Python evaluator and run a second compact refinement centered only on the best one-factor setting, with at most a few adjacent values for the dominant knob and no changes to CAPTURE/LOCK physics.",
        "",
        "## Caution",
        "",
        "These results are local to the Phase 6.5 270-regime grid. The script intentionally avoids broad search and does not change the environment or learned policies.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def iter_cases(variants: Sequence[WSVariant]) -> Iterable[tuple[WSVariant, float, float, float, float]]:
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
        print(f"phase66 pending={len(pending)} completed={len(done)} total={total} workers={workers}")
        if workers == 1:
            for idx, (variant, r0, angle, thrust, target_scale) in pending:
                row = rollout_case(variant, r0, angle, thrust, target_scale)
                append_row(GRID_CSV_PATH, row)
                rows.append(row)
                done.add((variant.name, r0, angle, thrust, target_scale))
                print(
                    f"phase66 {idx}/{total} variant={variant.name} r0={r0:.5f} angle={angle:g} "
                    f"thrust={thrust:g} target={target_scale:.2f} success={row['success']} "
                    f"capture={row['capture_entered']} min_abs_rerr={row['minimum_abs_radius_error']:.3e}"
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
                        f"phase66 {idx}/{total} done_now={completed_now}/{len(pending)} variant={variant.name} "
                        f"r0={r0:.5f} angle={angle:g} thrust={thrust:g} target={target_scale:.2f} "
                        f"success={row['success']} capture={row['capture_entered']} "
                        f"min_abs_rerr={row['minimum_abs_radius_error']:.3e}"
                    )
    else:
        print(f"phase66 no pending rows; completed={len(done)} total={total}")

    rows = load_rows()
    ranking = rank_rows(rows, variants)
    write_ranking(ranking)
    save_plots(rows, ranking)
    write_summary(rows, ranking)
    print(f"Saved Phase 6.6 WS-1 refinement outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
