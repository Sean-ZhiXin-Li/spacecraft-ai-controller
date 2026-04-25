from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig
from scripts.explicit_controller_analysis_utils import DT, MAX_STEPS, SEED, STRICT_CFG, TAIL_LEN, set_seed
from scripts.explicit_controller_phase4_regime_sweep import DEFAULT_TARGET_RADIUS


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase65_window_seeking"
CSV_PATH = OUTPUT_DIR / "ws1_vs_baseline.csv"
CASE_SUMMARY_PATH = OUTPUT_DIR / "ws1_case_comparisons.md"
SUMMARY_PATH = OUTPUT_DIR / "phase65_summary.md"

SUCCESS_PNG = OUTPUT_DIR / "success_baseline_vs_ws1.png"
CAPTURE_PNG = OUTPUT_DIR / "capture_baseline_vs_ws1.png"
WS1_MAP_PNG = OUTPUT_DIR / "ws1_success_map.png"
WS1_NEAR_MISS_PNG = OUTPUT_DIR / "ws1_near_miss_map.png"

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
class ControllerSpec:
    name: str
    params: Dict[str, float | str]


BASELINE = ControllerSpec("baseline", {})
WS1 = ControllerSpec(
    "ws1",
    {
        "near_band_ratio": 8.0e-5,
        "inward_bias": 0.12,
        "reduced_retro_scale": 0.70,
        "score_r": 1.0,
        "score_vr": 0.65,
        "score_vt": 0.45,
        "score_energy": 0.30,
        "capture_bonus": 0.20,
        "inner_capture_bonus": 0.10,
    },
)
CONTROLLERS = [BASELINE, WS1]


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

    def act(self, pos: np.ndarray, vel: np.ndarray, thrust_scale: float) -> tuple[np.ndarray, str]:
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

        if self.phase == "DESCENT":
            if self.spec.name == "ws1" and abs(real_r_ratio) < float(self.spec.params["near_band_ratio"]):
                return self.window_seeking_action(pos, vel, thrust_scale), self.phase
            return self.baseline_descent_action(pos, vel), self.phase
        return self.capture_lock_action(pos, vel), self.phase


def rollout_case(spec: ControllerSpec, r0: float, angle: float, thrust_scale: float, target_scale: float) -> Dict[str, object]:
    target_radius = DEFAULT_TARGET_RADIUS * target_scale
    pos, vel = initial_state(r0, angle, target_radius)
    controller = ExplicitPhaseController(target_radius=target_radius, spec=spec)
    radius_errors: List[float] = []
    vr_values: List[float] = []
    energy_values: List[float] = []
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
        action, phase = controller.act(pos, vel, thrust_scale)
        capture_entered = capture_entered or phase == "CAPTURE"
        lock_entered = lock_entered or phase == "LOCK"
        pos, vel = env_equivalent_step(pos, vel, action, thrust_scale)
        radius = norm2(pos)
        r_error = radius - target_radius
        vr = compute_vr(pos, vel)
        energy = 0.5 * norm2(vel) ** 2 - MU / (radius + 1.0e-12)
        radius_errors.append(r_error)
        vr_values.append(vr)
        energy_values.append(energy)
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
        "controller_name": spec.name,
        "controller_params_json": json.dumps(spec.params, sort_keys=True),
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


def load_rows() -> List[Dict[str, object]]:
    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
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
                "success": str(row["success"]).strip().lower() == "true",
                "capture_entered": str(row["capture_entered"]).strip().lower() == "true",
                "lock_entered": str(row["lock_entered"]).strip().lower() == "true",
                "crossing_occurs": str(row["crossing_occurs"]).strip().lower() == "true",
                "minimum_abs_radius_error": float(row["minimum_abs_radius_error"]),
                "final_radius_error": float(row["final_radius_error"]),
                "tail_mean_abs_vr": float(row["tail_mean_abs_vr"]),
                "steps": int(float(row["steps"])),
            }
        )
    return rows


def save_pair_bars(rows: List[Dict[str, object]], metric: str, path: Path, title: str, ylabel: str) -> None:
    values = []
    for controller in CONTROLLERS:
        subset = [row for row in rows if row["controller_name"] == controller.name]
        values.append(sum(bool(row[metric]) for row in subset))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x = np.arange(len(CONTROLLERS))
    ax.bar(x, values, color=["#777777", "#1B9E77"])
    ax.set_xticks(x)
    ax.set_xticklabels([controller.name for controller in CONTROLLERS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for idx, value in enumerate(values):
        ax.text(idx, value + 2, str(value), ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_ws1_maps(rows: List[Dict[str, object]]) -> None:
    subset = [
        row
        for row in rows
        if row["controller_name"] == "ws1"
        and abs(float(row["thrust_scale"]) - 10000.0) < 1.0e-9
        and abs(float(row["target_radius_scale"]) - 1.0) < 1.0e-9
    ]
    x_values = R0_VALUES
    y_values = ANGLE_VALUES
    success_grid = np.zeros((len(y_values), len(x_values)), dtype=np.float64)
    near_grid = np.zeros((len(y_values), len(x_values)), dtype=np.float64)
    for row in subset:
        xi = x_values.index(float(row["r0_over_target"]))
        yi = y_values.index(float(row["initial_velocity_angle_deg"]))
        success_grid[yi, xi] = 1.0 if bool(row["success"]) else 0.0
        target_radius = DEFAULT_TARGET_RADIUS * float(row["target_radius_scale"])
        near_grid[yi, xi] = (
            1.0
            if (not bool(row["success"])) and float(row["minimum_abs_radius_error"]) / target_radius <= NEAR_MISS_REL
            else 0.0
        )

    for grid, path, title in [
        (success_grid, WS1_MAP_PNG, "WS-1 success map (thrust=10000, target scale=1.00)"),
        (near_grid, WS1_NEAR_MISS_PNG, "WS-1 near-miss map (thrust=10000, target scale=1.00)"),
    ]:
        fig, ax = plt.subplots(figsize=(8.0, 5.1))
        cmap = colors.ListedColormap(["#F2F2F2", "#1B9E77"])
        image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=colors.BoundaryNorm([-0.5, 0.5, 1.5], 2))
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels([f"{x:.5f}" for x in x_values], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_yticklabels([f"{y:g}" for y in y_values])
        ax.set_xlabel("r0_over_target")
        ax.set_ylabel("initial_velocity_angle_deg")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, ticks=[0, 1], fraction=0.045, pad=0.03)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)


def rollout_trace(spec: ControllerSpec, r0: float, angle: float, thrust_scale: float, target_scale: float) -> Dict[str, np.ndarray]:
    target_radius = DEFAULT_TARGET_RADIUS * target_scale
    pos, vel = initial_state(r0, angle, target_radius)
    controller = ExplicitPhaseController(target_radius=target_radius, spec=spec)
    trace = {"time": [], "radius_error": [], "v_r": [], "energy": []}
    success_counter = 0
    for step in range(1, MAX_STEPS + 1):
        action, _ = controller.act(pos, vel, thrust_scale)
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


def write_case_comparisons() -> None:
    cases = [
        ("validated_success", 1.00005, 170.0, 10000.0, 1.00, "validated baseline success"),
        ("near_miss", 1.00002, 175.0, 10000.0, 1.00, "Phase 4 near-miss"),
        ("energy_limited", 1.00002, 170.0, 10000.0, 1.02, "Phase 5 energy-limited failure"),
        ("geometry_miss", 1.00010, 175.0, 11000.0, 1.01, "geometry-miss style failure"),
    ]
    lines = [
        "# WS-1 Case Comparisons",
        "",
        "The table below compares baseline and WS-1 on four representative regimes using the same local 2D Euler rollout evaluator as the main WS-1 sweep.",
        "",
        "| case | controller | final_radius_error | min_abs_radius_error | final_v_r | final_energy | steps |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for case_name, r0, angle, thrust, target_scale, note in cases:
        for controller in CONTROLLERS:
            trace = rollout_trace(controller, r0, angle, thrust, target_scale)
            lines.append(
                f"| {case_name} ({note}) | {controller.name} | {abs(float(trace['radius_error'][-1])):.3e} | "
                f"{float(np.min(np.abs(trace['radius_error']))):.3e} | {float(trace['v_r'][-1]):.3e} | "
                f"{float(trace['energy'][-1]):.3e} | {len(trace['time'])} |"
            )
    lines.extend(
        [
            "",
            "WS-1 trace comparison is kept tabular here to stay focused on mechanism-level differences without producing another large bundle of per-case figure files.",
        ]
    )
    CASE_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_summary(rows: List[Dict[str, object]]) -> None:
    baseline = [row for row in rows if row["controller_name"] == "baseline"]
    ws1 = [row for row in rows if row["controller_name"] == "ws1"]
    baseline_success = sum(bool(row["success"]) for row in baseline)
    ws1_success = sum(bool(row["success"]) for row in ws1)
    baseline_capture = sum(bool(row["capture_entered"]) for row in baseline)
    ws1_capture = sum(bool(row["capture_entered"]) for row in ws1)
    baseline_lock = sum(bool(row["lock_entered"]) for row in baseline)
    ws1_lock = sum(bool(row["lock_entered"]) for row in ws1)
    baseline_near = sum((not bool(row["success"])) and float(row["minimum_abs_radius_error"]) / (DEFAULT_TARGET_RADIUS * float(row["target_radius_scale"])) <= NEAR_MISS_REL for row in baseline)
    ws1_near = sum((not bool(row["success"])) and float(row["minimum_abs_radius_error"]) / (DEFAULT_TARGET_RADIUS * float(row["target_radius_scale"])) <= NEAR_MISS_REL for row in ws1)
    lines = [
        "# Phase 6.5 Window-Seeking Summary",
        "",
        "## Setup",
        "",
        "- Compared controllers: baseline explicit controller and Window-Seeking v1 (`WS-1`).",
        "- WS-1 keeps CAPTURE and LOCK equations unchanged and only changes DESCENT behavior near the target-radius band.",
        "- Near-target band: `|radius error| / target_radius < 8e-5`.",
        "- Candidate actions in window-seeking mode: pure retrograde; retrograde plus small inward radial bias; reduced retrograde magnitude.",
        "- Candidate selection uses one-step explicit Euler lookahead with a simple score over radius error, radial velocity, tangential-speed error, and energy error, plus a small bonus for more capture-compatible states.",
        "",
        "## Results",
        "",
        f"- Baseline: success `{baseline_success}`, capture `{baseline_capture}`, lock `{baseline_lock}`, near-miss `{baseline_near}`.",
        f"- WS-1: success `{ws1_success}`, capture `{ws1_capture}`, lock `{ws1_lock}`, near-miss `{ws1_near}`.",
        "",
        "## Answers",
        "",
        f"1. WS-1 improves CAPTURE reachability relative to baseline: `{ws1_capture > baseline_capture}`. Strict success improves: `{ws1_success > baseline_success}`.",
        "2. Any gain is primarily from better window access if CAPTURE count rises more than post-CAPTURE success conditional on CAPTURE. Because CAPTURE and LOCK remain tightly coupled here, the main question is whether WS-1 gets into CAPTURE more often.",
        "3. One-step explicit lookahead helps only if the near-target candidate choice improves CAPTURE access or reduces near-miss count. The baseline-vs-WS1 counts and the WS-1 near-miss map should be read directly for that answer.",
        "4. The clearest failure-mode improvements, if present, should appear first in near-miss and energy-limited cases rather than in large geometry-miss cases. Geometry misses that stay far outside the near-target band are not expected to benefit much from WS-1.",
        "5. Best next step: keep the same explicit architecture and refine the WS-1 near-target band and scoring weights on the same local 2D grid, then test a small second window-seeking variant rather than broad parameter sweeps.",
        "",
        "## Caution",
        "",
        "WS-1 is still a local 2D explicit controller. Any improvement should be read as evidence about pre-CAPTURE reachability in this narrow regime, not as a broad generalization result.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    buffer: List[Dict[str, object]] = []
    cases = [
        (controller, r0, angle, thrust, target_scale)
        for controller in CONTROLLERS
        for target_scale in TARGET_RADIUS_SCALES
        for thrust in THRUST_VALUES
        for angle in ANGLE_VALUES
        for r0 in R0_VALUES
    ]
    for idx, (controller, r0, angle, thrust, target_scale) in enumerate(cases, start=1):
        row = rollout_case(controller, r0, angle, thrust, target_scale)
        rows.append(row)
        buffer.append(row)
        print(
            f"phase65 {idx}/{len(cases)} controller={controller.name} r0={r0:.5f} angle={angle:g} "
            f"thrust={thrust:g} target={target_scale:.2f} success={row['success']} capture={row['capture_entered']} "
            f"min_abs_rerr={row['minimum_abs_radius_error']:.3e}"
        )
        if len(buffer) >= 100:
            append_rows(CSV_PATH, buffer)
            buffer.clear()
    append_rows(CSV_PATH, buffer)
    rows = load_rows()
    save_pair_bars(rows, "success", SUCCESS_PNG, "Baseline vs WS-1 strict success count", "count over 270 regimes")
    save_pair_bars(rows, "capture_entered", CAPTURE_PNG, "Baseline vs WS-1 CAPTURE count", "count over 270 regimes")
    save_ws1_maps(rows)
    write_case_comparisons()
    write_summary(rows)
    print(f"Saved Phase 6.5 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
