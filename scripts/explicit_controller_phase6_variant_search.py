from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig
from scripts.explicit_controller_analysis_utils import DT, MAX_STEPS, SEED, STRICT_CFG, TAIL_LEN, set_seed
from scripts.explicit_controller_phase4_regime_sweep import DEFAULT_TARGET_RADIUS


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase6_variant_search"
GRID_PATH = OUTPUT_DIR / "variant_grid.csv"
RANKING_PATH = OUTPUT_DIR / "variant_ranking.csv"
SUMMARY_PATH = OUTPUT_DIR / "phase6_summary.md"
TRACE_SUMMARY_PATH = OUTPUT_DIR / "top_variant_traces_summary.md"

SUCCESS_BY_VARIANT_PNG = OUTPUT_DIR / "success_by_variant.png"
CAPTURE_BY_VARIANT_PNG = OUTPUT_DIR / "capture_by_variant.png"
NEAR_MISS_BY_VARIANT_PNG = OUTPUT_DIR / "near_miss_by_variant.png"
TOP_VARIANT_MAP_PNG = OUTPUT_DIR / "top_variant_success_map.png"
BASELINE_VS_BEST_PNG = OUTPUT_DIR / "baseline_vs_best_variant.png"

R0_VALUES = [1.00002, 1.00004, 1.00005, 1.00006, 1.00008, 1.00010]
ANGLE_VALUES = [165.0, 168.0, 170.0, 172.0, 175.0]
THRUST_VALUES = [9000.0, 10000.0, 11000.0]
TARGET_RADIUS_SCALES = [0.99, 1.00, 1.01]

G = 6.67430e-11
M = 1.989e30
MU = G * M
MASS = 722.0
NEAR_MISS_REL = 3.0e-5

FIELDNAMES = [
    "variant_name",
    "variant_params_json",
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


@dataclass(frozen=True)
class VariantSpec:
    name: str
    params: Dict[str, float | str]


VARIANTS = [
    VariantSpec("baseline", {"retrograde_scale": 1.0, "radial_gain": 0.0, "near_scale": 1.0, "handoff_r": 0.0}),
    VariantSpec("retro_085", {"retrograde_scale": 0.85, "radial_gain": 0.0, "near_scale": 0.85, "handoff_r": 0.0}),
    VariantSpec("retro_070", {"retrograde_scale": 0.70, "radial_gain": 0.0, "near_scale": 0.70, "handoff_r": 0.0}),
    VariantSpec("retro_115", {"retrograde_scale": 1.15, "radial_gain": 0.0, "near_scale": 1.0, "handoff_r": 0.0}),
    VariantSpec("radial_005", {"retrograde_scale": 0.95, "radial_gain": 0.05, "near_scale": 0.85, "handoff_r": 0.0}),
    VariantSpec("radial_010", {"retrograde_scale": 0.90, "radial_gain": 0.10, "near_scale": 0.75, "handoff_r": 0.0}),
    VariantSpec("radial_015", {"retrograde_scale": 0.85, "radial_gain": 0.15, "near_scale": 0.70, "handoff_r": 0.0}),
    VariantSpec("scheduled_soft", {"retrograde_scale": 1.0, "radial_gain": 0.05, "near_scale": 0.60, "handoff_r": 0.0}),
    VariantSpec("scheduled_mid", {"retrograde_scale": 1.0, "radial_gain": 0.10, "near_scale": 0.50, "handoff_r": 0.0}),
    VariantSpec("scheduled_aggr", {"retrograde_scale": 1.10, "radial_gain": 0.12, "near_scale": 0.45, "handoff_r": 0.0}),
    VariantSpec("early_handoff_3e4", {"retrograde_scale": 0.95, "radial_gain": 0.05, "near_scale": 0.65, "handoff_r": 3.0e-4}),
    VariantSpec("early_handoff_6e4", {"retrograde_scale": 0.90, "radial_gain": 0.08, "near_scale": 0.55, "handoff_r": 6.0e-4}),
]


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


def obs_from_state(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    return np.asarray([pos[0], pos[1], vel[0], vel[1], compute_vr(pos, vel)], dtype=np.float32)


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


def env_equivalent_step(
    pos: np.ndarray,
    vel: np.ndarray,
    action: np.ndarray,
    thrust_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    action = np.clip(action, -1.0, 1.0).astype(np.float64)
    thrust = thrust_scale * action
    acc_thrust = thrust / MASS
    radius = norm2(pos)
    acc_gravity = -MU * pos / ((radius**3) + 1.0e-12)
    vel_next = vel + (acc_gravity + acc_thrust) * DT
    pos_next = pos + vel_next * DT
    return pos_next, vel_next


class DescentVariantController:
    def __init__(self, *, target_radius: float, spec: VariantSpec) -> None:
        self.target_radius = float(target_radius)
        self.v_circ = math.sqrt(MU / target_radius)
        self.spec = spec
        self.cfg = OrbitLockConfig()
        self.phase = "DESCENT"
        self.prev_real_r_error: float | None = None
        self.phase_transition_count = 0

    def set_phase(self, new_phase: str) -> None:
        if new_phase != self.phase:
            self.phase = new_phase
            self.phase_transition_count += 1

    def maybe_early_handoff(self, pos: np.ndarray, vel: np.ndarray) -> None:
        handoff_r = float(self.spec.params.get("handoff_r", 0.0))
        if self.phase != "DESCENT" or handoff_r <= 0.0:
            return
        radius = norm2(pos)
        r_error = radius - self.target_radius
        v_r = compute_vr(pos, vel)
        if abs(r_error / self.target_radius) < handoff_r and v_r < 0.0:
            self.set_phase("CAPTURE")

    def descent_action(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        radius = norm2(pos)
        r_hat = pos / (radius + 1.0e-12)
        v_norm = norm2(vel)
        retrograde = -vel / (v_norm + 1.0e-12)
        r_error_ratio = (radius - self.target_radius) / self.target_radius
        retro_scale = float(self.spec.params.get("retrograde_scale", 1.0))
        near_scale = float(self.spec.params.get("near_scale", retro_scale))
        radial_gain = float(self.spec.params.get("radial_gain", 0.0))
        schedule_width = 8.0e-5
        blend = clamp(abs(r_error_ratio) / schedule_width, 0.0, 1.0)
        active_retro = near_scale + (retro_scale - near_scale) * blend
        inward_cmd = -radial_gain * clamp(r_error_ratio / 1.0e-4, -1.0, 1.0) * r_hat
        action = active_retro * retrograde + inward_cmd
        action_norm = norm2(action)
        if action_norm > 1.0:
            action = action / action_norm
        return np.clip(action, -1.0, 1.0)

    def act(self, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, str]:
        self.maybe_early_handoff(pos, vel)
        radius = norm2(pos)
        real_r_error = radius - self.target_radius
        v_r = compute_vr(pos, vel)
        v_t_error = compute_vt(pos, vel) - self.v_circ
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

        phase = self.phase
        if phase == "DESCENT":
            if self.spec.name != "baseline":
                action = self.descent_action(pos, vel)
            else:
                v_norm = norm2(vel)
                action = np.array([-vel[0] / (v_norm + 1.0e-12), -vel[1] / (v_norm + 1.0e-12)], dtype=np.float64)
        else:
            r_hat = pos / (radius + 1.0e-12)
            t_hat = np.array([-r_hat[1], r_hat[0]], dtype=np.float64)
            if phase == "CAPTURE":
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
            action = radial_cmd * r_hat + tangential_cmd * t_hat
            action = np.clip(action, -1.0, 1.0)
        return action, phase


def rollout_case(spec: VariantSpec, r0: float, angle: float, thrust_scale: float, target_scale: float) -> Dict[str, object]:
    target_radius = DEFAULT_TARGET_RADIUS * target_scale
    pos, vel = initial_state(r0, angle, target_radius)
    controller = DescentVariantController(target_radius=target_radius, spec=spec)
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

    for step in range(1, MAX_STEPS + 1):
        action, phase = controller.act(pos, vel)
        capture_entered = capture_entered or phase == "CAPTURE"
        lock_entered = lock_entered or phase == "LOCK"
        pos, vel = env_equivalent_step(pos, vel, action, thrust_scale)
        radius = norm2(pos)
        r_error = radius - target_radius
        vr = compute_vr(pos, vel)
        radius_errors.append(r_error)
        vr_values.append(vr)
        if prev_r_error is not None and ((prev_r_error > 0.0 and r_error <= 0.0) or (prev_r_error < 0.0 and r_error >= 0.0)):
            radius_crossings_total += 1
            if first_crossing_step is None:
                first_crossing_step = step
        prev_r_error = r_error

        if inside_tolerance(pos, vel, target_radius):
            success_counter += 1
        else:
            success_counter = 0

        speed = norm2(vel)
        v_target = math.sqrt(MU / target_radius)
        r_err_now = abs(r_error) / target_radius
        ur_now = pos / (radius + 1.0e-12)
        uv_now = vel / (speed + 1.0e-12)
        ang_abs_now = abs(float(np.dot(ur_now, uv_now)))
        v_err_now = abs(speed - v_target) / v_target
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
        "variant_name": spec.name,
        "variant_params_json": json.dumps(spec.params, sort_keys=True),
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


def read_existing(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def completed_key(row: Dict[str, object]) -> tuple[str, float, float, float, float]:
    return (
        str(row["variant_name"]),
        float(row["r0_over_target"]),
        float(row["initial_velocity_angle_deg"]),
        float(row["thrust_scale"]),
        float(row["target_radius_scale"]),
    )


def append_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in FIELDNAMES})


def all_cases() -> List[tuple[VariantSpec, float, float, float, float]]:
    return [
        (variant, r0, angle, thrust, target_scale)
        for variant in VARIANTS
        for target_scale in TARGET_RADIUS_SCALES
        for thrust in THRUST_VALUES
        for angle in ANGLE_VALUES
        for r0 in R0_VALUES
    ]


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_grid_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in read_existing(GRID_PATH):
        rows.append(
            {
                **row,
                "r0_over_target": float(row["r0_over_target"]),
                "initial_velocity_angle_deg": float(row["initial_velocity_angle_deg"]),
                "thrust_scale": float(row["thrust_scale"]),
                "target_radius_scale": float(row["target_radius_scale"]),
                "success": as_bool(row["success"]),
                "capture_entered": as_bool(row["capture_entered"]),
                "lock_entered": as_bool(row["lock_entered"]),
                "crossing_occurs": as_bool(row["crossing_occurs"]),
                "minimum_abs_radius_error": float(row["minimum_abs_radius_error"]),
                "final_radius_error": float(row["final_radius_error"]),
                "tail_mean_abs_vr": float(row["tail_mean_abs_vr"]),
                "steps": int(float(row["steps"])),
            }
        )
    return rows


def write_ranking(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ranking: List[Dict[str, object]] = []
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant_name"] == variant.name]
        successes = [row for row in subset if bool(row["success"])]
        near_miss_count = sum(
            (not bool(row["success"])) and float(row["minimum_abs_radius_error"]) / (DEFAULT_TARGET_RADIUS * float(row["target_radius_scale"])) <= NEAR_MISS_REL
            for row in subset
        )
        ranking.append(
            {
                "variant_name": variant.name,
                "variant_params_json": json.dumps(variant.params, sort_keys=True),
                "total_runs": len(subset),
                "success_count": sum(bool(row["success"]) for row in subset),
                "capture_count": sum(bool(row["capture_entered"]) for row in subset),
                "lock_count": sum(bool(row["lock_entered"]) for row in subset),
                "mean_minimum_abs_radius_error": float(np.mean([float(row["minimum_abs_radius_error"]) for row in subset])) if subset else float("nan"),
                "mean_final_radius_error_success": float(np.mean([float(row["final_radius_error"]) for row in successes])) if successes else float("nan"),
                "near_miss_count": int(near_miss_count),
            }
        )
    ranking.sort(
        key=lambda row: (
            -int(row["success_count"]),
            -int(row["capture_count"]),
            float(row["mean_minimum_abs_radius_error"]),
        )
    )
    with RANKING_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking[0].keys()))
        writer.writeheader()
        writer.writerows(ranking)
    return ranking


def bar_plot(path: Path, ranking: List[Dict[str, object]], key: str, title: str, ylabel: str) -> None:
    ordered = list(reversed(ranking))
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.barh([row["variant_name"] for row in ordered], [float(row[key]) for row in ordered], color="#4C78A8")
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_top_variant_map(rows: List[Dict[str, object]], top_variant: str) -> None:
    subset = [
        row
        for row in rows
        if row["variant_name"] == top_variant
        and abs(float(row["thrust_scale"]) - 10000.0) < 1.0e-9
        and abs(float(row["target_radius_scale"]) - 1.0) < 1.0e-9
    ]
    x_values = R0_VALUES
    y_values = ANGLE_VALUES
    grid = np.zeros((len(y_values), len(x_values)), dtype=np.float64)
    for row in subset:
        xi = x_values.index(float(row["r0_over_target"]))
        yi = y_values.index(float(row["initial_velocity_angle_deg"]))
        grid[yi, xi] = 1.0 if bool(row["success"]) else 0.0
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    cmap = colors.ListedColormap(["#F2F2F2", "#1B9E77"])
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=colors.BoundaryNorm([-0.5, 0.5, 1.5], 2))
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{x:.5f}" for x in x_values], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([f"{y:g}" for y in y_values])
    ax.set_xlabel("r0_over_target")
    ax.set_ylabel("initial_velocity_angle_deg")
    ax.set_title(f"Top variant success map: {top_variant} (thrust=10000, target scale=1.00)")
    fig.colorbar(image, ax=ax, ticks=[0, 1], fraction=0.045, pad=0.03)
    fig.tight_layout()
    fig.savefig(TOP_VARIANT_MAP_PNG, dpi=220)
    plt.close(fig)


def save_baseline_vs_best(rows: List[Dict[str, object]], best_variant: str) -> None:
    variants = ["baseline", best_variant]
    metrics = ["success", "capture_entered", "lock_entered"]
    values = []
    for variant in variants:
        subset = [row for row in rows if row["variant_name"] == variant]
        values.append([sum(bool(row[metric]) for row in subset) for metric in metrics])
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(x - width / 2, values[0], width, label="baseline", color="#777777")
    ax.bar(x + width / 2, values[1], width, label=best_variant, color="#1B9E77")
    ax.set_xticks(x)
    ax.set_xticklabels(["success", "capture", "lock"])
    ax.set_ylabel("count over 270 regimes")
    ax.set_title("Baseline vs best Phase 6 variant")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(BASELINE_VS_BEST_PNG, dpi=220)
    plt.close(fig)


def save_plots(rows: List[Dict[str, object]], ranking: List[Dict[str, object]]) -> None:
    bar_plot(SUCCESS_BY_VARIANT_PNG, ranking, "success_count", "Strict success count by variant", "success count")
    bar_plot(CAPTURE_BY_VARIANT_PNG, ranking, "capture_count", "CAPTURE entry count by variant", "capture count")
    bar_plot(NEAR_MISS_BY_VARIANT_PNG, ranking, "near_miss_count", "Near-miss count by variant", "near-miss count")
    top_variant = str(ranking[0]["variant_name"])
    save_top_variant_map(rows, top_variant)
    save_baseline_vs_best(rows, top_variant)


def rollout_trace(spec: VariantSpec, r0: float, angle: float, thrust_scale: float, target_scale: float) -> Dict[str, np.ndarray | bool]:
    target_radius = DEFAULT_TARGET_RADIUS * target_scale
    pos, vel = initial_state(r0, angle, target_radius)
    controller = DescentVariantController(target_radius=target_radius, spec=spec)
    trace = {"time": [], "radius_error": [], "v_r": [], "energy": []}
    success_counter = 0
    for step in range(1, MAX_STEPS + 1):
        action, _ = controller.act(pos, vel)
        pos, vel = env_equivalent_step(pos, vel, action, thrust_scale)
        radius = norm2(pos)
        speed = norm2(vel)
        trace["time"].append(step * DT)
        trace["radius_error"].append(radius - target_radius)
        trace["v_r"].append(compute_vr(pos, vel))
        trace["energy"].append(0.5 * speed * speed - MU / (radius + 1.0e-12))
        success_counter = success_counter + 1 if inside_tolerance(pos, vel, target_radius) else 0
        if success_counter >= int(STRICT_CFG["success_threshold"]):
            break
        if radius > 2.5 * target_radius or speed > 1.90 * math.sqrt(MU / target_radius) or radius < 0.35 * target_radius:
            break
    return {key: np.asarray(value, dtype=np.float64) for key, value in trace.items()}


def write_trace_summary(rows: List[Dict[str, object]], ranking: List[Dict[str, object]]) -> None:
    top_names = ["baseline"] + [str(row["variant_name"]) for row in ranking if str(row["variant_name"]) != "baseline"][:3]
    variants_by_name = {variant.name: variant for variant in VARIANTS}
    representative = [
        (1.00005, 170.0, 10000.0, 1.00),
        (1.00002, 175.0, 10000.0, 1.00),
        (1.00008, 170.0, 10000.0, 1.01),
        (1.00010, 165.0, 11000.0, 1.00),
    ]
    lines = [
        "# Top Variant Trace Summary",
        "",
        f"Compared variants: `{', '.join(top_names)}`.",
        "",
        "| variant | r0 | angle | thrust | target_scale | final_radius_error | min_abs_radius_error | final_energy | steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in top_names:
        spec = variants_by_name[name]
        for r0, angle, thrust, target_scale in representative:
            trace = rollout_trace(spec, r0, angle, thrust, target_scale)
            rerr = trace["radius_error"]
            energy = trace["energy"]
            lines.append(
                f"| {name} | {r0:.5f} | {angle:g} | {thrust:g} | {target_scale:.2f} | "
                f"{abs(float(rerr[-1])):.3e} | {float(np.min(np.abs(rerr))):.3e} | {float(energy[-1]):.3e} | {len(rerr)} |"
            )
    lines.extend(
        [
            "",
            "Trace comparison is intentionally tabular here to avoid generating another large bundle of per-case plots. The primary run-level plots and CSV contain the full-grid evidence.",
        ]
    )
    TRACE_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_summary(rows: List[Dict[str, object]], ranking: List[Dict[str, object]]) -> None:
    baseline = next(row for row in ranking if row["variant_name"] == "baseline")
    best = ranking[0]
    best_name = str(best["variant_name"])
    baseline_success = int(baseline["success_count"])
    best_success = int(best["success_count"])
    baseline_capture = int(baseline["capture_count"])
    best_capture = int(best["capture_count"])
    best_params = json.loads(str(best["variant_params_json"]))
    family_counts = Counter()
    for row in ranking[:5]:
        name = str(row["variant_name"])
        if "handoff" in name:
            family_counts["early handoff"] += 1
        elif "scheduled" in name:
            family_counts["descent scheduling"] += 1
        elif "radial" in name:
            family_counts["radial damping"] += 1
        elif "retro" in name:
            family_counts["retrograde scale"] += 1
        else:
            family_counts["baseline"] += 1
    likely_knob = family_counts.most_common(1)[0][0] if family_counts else "unclear"
    lines = [
        "# Phase 6 Explicit DESCENT Variant Search",
        "",
        "## Setup",
        "",
        "- Scope: 2D explicit-controller family only.",
        "- Physics implementation: local Python evaluator mirrors the existing `OrbitEnv` Euler dynamics with action smoothing and orbit-capture assist disabled, matching the Phase 3-5 explicit evaluations.",
        "- Controller changes are limited to DESCENT-side action construction or early handoff; CAPTURE and LOCK gains are unchanged.",
        f"- Variants: `{len(VARIANTS)}`.",
        f"- Regimes per variant: `{len(R0_VALUES) * len(ANGLE_VALUES) * len(THRUST_VALUES) * len(TARGET_RADIUS_SCALES)}`.",
        f"- Total completed runs: `{len(rows)}`.",
        "",
        "## Best Variant",
        "",
        f"- Best variant: `{best_name}`.",
        f"- Parameters: `{json.dumps(best_params, sort_keys=True)}`.",
        f"- Baseline successes: `{baseline_success}`.",
        f"- Best successes: `{best_success}`.",
        f"- Baseline CAPTURE count: `{baseline_capture}`.",
        f"- Best CAPTURE count: `{best_capture}`.",
        "",
        "## Answers",
        "",
        f"1. Explicit reachability improved without learning: `{best_success > baseline_success}`. The best variant changes success count from `{baseline_success}` to `{best_success}` on the 270-regime grid.",
        f"2. The most important controller-side knob in this search appears to be `{likely_knob}`. This is inferred from the top-ranked variant family, not from a continuous sensitivity analysis.",
        f"3. The main gain comes from CAPTURE access if the CAPTURE count increased more than post-CAPTURE success conditional on CAPTURE. Here CAPTURE changes from `{baseline_capture}` to `{best_capture}`, while CAPTURE and LOCK remain closely coupled in the ranking data.",
        "4. The best variant should be read as shifting and locally widening the practical capture access region only if its success map adds neighboring cells around the baseline; otherwise it mainly shifts the pocket. See `top_variant_success_map.png` and `baseline_vs_best_variant.png`.",
        "5. Best next step: refine the winning DESCENT knob family with a smaller local grid, then validate against the Phase 4 and Phase 5 representative failures before considering any broader architecture changes.",
        "",
        "## Caution",
        "",
        "This is an explicit variant search, not a new learned policy. The result should not be generalized beyond the sampled local 2D regimes.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_grid() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = read_existing(GRID_PATH)
    done = {completed_key(row) for row in existing}
    cases = all_cases()
    pending = [case for case in cases if completed_key({"variant_name": case[0].name, "r0_over_target": case[1], "initial_velocity_angle_deg": case[2], "thrust_scale": case[3], "target_radius_scale": case[4]}) not in done]
    print(f"phase6 total={len(cases)} completed={len(done)} pending={len(pending)}")
    buffer: List[Dict[str, object]] = []
    completed_now = 0
    for idx, (variant, r0, angle, thrust, target_scale) in enumerate(pending, start=1):
        row = rollout_case(variant, r0, angle, thrust, target_scale)
        buffer.append(row)
        completed_now += 1
        print(
            f"phase6 {idx}/{len(pending)} variant={variant.name} r0={r0:.5f} angle={angle:g} "
            f"thrust={thrust:g} target={target_scale:.2f} success={row['success']} capture={row['capture_entered']} "
            f"min_abs_rerr={row['minimum_abs_radius_error']:.3e}"
        )
        if len(buffer) >= 50:
            append_rows(GRID_PATH, buffer)
            buffer.clear()
            print(f"checkpoint wrote {completed_now} new runs")
    append_rows(GRID_PATH, buffer)


def main() -> None:
    set_seed(SEED)
    run_grid()
    rows = load_grid_rows()
    ranking = write_ranking(rows)
    save_plots(rows, ranking)
    write_trace_summary(rows, ranking)
    write_summary(rows, ranking)
    print(f"Saved Phase 6 variant search outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
