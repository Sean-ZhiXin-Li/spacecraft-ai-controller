from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase32_direct_optimal_control"
MPLCONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.explicit_controller_phase21_orbital_transfer_planner import (
    DEFAULT_TARGET_RADIUS,
    DT,
    MASS,
    MU,
    RECOVERABLE_R_RATIO,
    RECOVERABLE_VR_RATIO,
    RECOVERABLE_VT_RATIO,
    TARGET_RADIUS_SCALE,
    bool_from_csv,
    orbital_diagnostics,
)


PHASE31_RESULTS = PROJECT_ROOT / "analysis" / "phase31_global_transfer_solver" / "phase31_results.csv"
RESULTS_CSV = OUTPUT_DIR / "phase32_results.csv"
TRAJECTORY_CSV = OUTPUT_DIR / "phase32_trajectory_dataset.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
RESEARCH_CONTEXT_MD = OUTPUT_DIR / "research_context.md"

PLOT_TRAJ = OUTPUT_DIR / "optimal_trajectory_examples.png"
PLOT_STATE = OUTPUT_DIR / "state_error_vs_time.png"
PLOT_CONTROL = OUTPUT_DIR / "control_profile.png"
PLOT_SYNC = OUTPUT_DIR / "sync_error_progression.png"
PLOT_PHASE31 = OUTPUT_DIR / "phase32_vs_phase31_comparison.png"
PLOT_MODES = OUTPUT_DIR / "objective_mode_comparison.png"

CONTROL_INTERVALS = 64
HOLD_STEPS = 8
MAX_ACTION_NORM = 0.42
SCIPY_MAXITER = 80
TARGET_RADIUS = DEFAULT_TARGET_RADIUS * TARGET_RADIUS_SCALE


OBJECTIVE_MODES = [
    "radius_only",
    "recoverability_target",
    "sync_error_minimization",
    "fuel_constrained_recoverability",
]

CASES = [
    (1.00, 150.0, 10000.0, "baseline_crossing_low_angle"),
    (1.00, 175.0, 10000.0, "baseline_crossing_high_angle"),
    (0.98, 165.0, 10000.0, "inner_dead_window"),
    (1.02, 170.0, 10000.0, "outer_dead_geometry"),
]

RESULT_FIELDS = [
    "solver_backend",
    "objective_mode",
    "case_label",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "target_radius_scale",
    "solve_success",
    "solver_status",
    "objective_value",
    "iterations",
    "horizon_steps",
    "control_intervals",
    "hold_steps",
    "crossing_occurs",
    "first_crossing_step",
    "best_sync_error",
    "best_distance_to_recoverable",
    "final_sync_error",
    "final_distance_to_recoverable",
    "final_r_error_ratio",
    "final_vr_ratio",
    "final_vt_error_ratio",
    "best_r_error_ratio",
    "best_vr_ratio",
    "best_vt_error_ratio",
    "recoverable_state",
    "near_recoverable_state",
    "recoverable_crossing",
    "near_recoverable_crossing",
    "capture_potential",
    "success_potential",
    "control_effort",
    "control_smoothness",
    "max_action_norm",
    "overspeed",
    "instability",
    "runtime_seconds",
]

TRAJ_FIELDS = [
    "objective_mode",
    "case_label",
    "step",
    "time_seconds",
    "x",
    "y",
    "vx",
    "vy",
    "radius_ratio",
    "r_error_ratio",
    "vr_ratio",
    "vt_error_ratio",
    "sync_error",
    "distance_to_recoverable",
    "control_x",
    "control_y",
    "control_norm",
]


@dataclass
class SolveResult:
    row: Dict[str, object]
    trajectory: List[Dict[str, object]]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: object, default: float = float("nan")) -> float:
    if value in {"", None}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def has_casadi() -> tuple[bool, str]:
    try:
        import casadi as ca  # noqa: F401
    except Exception as exc:
        return False, str(exc)
    return True, "available"


def state_from_case(r0: float, angle: float) -> np.ndarray:
    x = 0.0
    y = r0 * TARGET_RADIUS
    radius0 = math.hypot(x, y)
    speed = math.sqrt(MU / radius0)
    angle_rad = math.radians(angle)
    vx = speed * math.cos(angle_rad)
    vy = speed * math.sin(angle_rad)
    return np.asarray([x, y, vx, vy], dtype=np.float64)


def clip_action(action: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(action))
    if norm > MAX_ACTION_NORM:
        return action * (MAX_ACTION_NORM / (norm + 1.0e-12))
    return action


def step_state(state: np.ndarray, action: np.ndarray, thrust_scale: float) -> np.ndarray:
    ax_cmd, ay_cmd = clip_action(action)
    x, y, vx, vy = state
    radius = math.hypot(x, y)
    denom = radius**3 + 1.0e-12
    ax = -MU * x / denom + thrust_scale * ax_cmd / MASS
    ay = -MU * y / denom + thrust_scale * ay_cmd / MASS
    next_vx = vx + ax * DT
    next_vy = vy + ay * DT
    next_x = x + next_vx * DT
    next_y = y + next_vy * DT
    return np.asarray([next_x, next_y, next_vx, next_vy], dtype=np.float64)


def sync_error(r_ratio: float, vr_ratio: float, vt_ratio: float) -> float:
    return max(
        abs(r_ratio) / RECOVERABLE_R_RATIO,
        abs(vr_ratio) / RECOVERABLE_VR_RATIO,
        abs(vt_ratio) / RECOVERABLE_VT_RATIO,
    )


def recoverability_distance(r_ratio: float, vr_ratio: float, vt_ratio: float) -> float:
    return math.sqrt(
        (abs(r_ratio) / RECOVERABLE_R_RATIO) ** 2
        + (abs(vr_ratio) / RECOVERABLE_VR_RATIO) ** 2
        + (abs(vt_ratio) / RECOVERABLE_VT_RATIO) ** 2
    )


def metrics_for_state(state: np.ndarray) -> Dict[str, float]:
    diag = orbital_diagnostics(float(state[0]), float(state[1]), float(state[2]), float(state[3]), TARGET_RADIUS)
    v_circ = math.sqrt(MU / TARGET_RADIUS)
    vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
    r_error_ratio = diag.r_error_ratio
    vt_error_ratio = diag.vt_error_ratio
    return {
        "radius_ratio": diag.radius / TARGET_RADIUS,
        "r_error_ratio": r_error_ratio,
        "vr_ratio": vr_ratio,
        "vt_error_ratio": vt_error_ratio,
        "sync_error": sync_error(r_error_ratio, vr_ratio, vt_error_ratio),
        "distance_to_recoverable": recoverability_distance(r_error_ratio, vr_ratio, vt_error_ratio),
        "speed_ratio": diag.speed / (v_circ + 1.0e-12),
    }


def rollout_controls(initial_state: np.ndarray, controls_flat: np.ndarray, thrust_scale: float) -> tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, float]]]:
    controls = controls_flat.reshape(CONTROL_INTERVALS, 2)
    state = initial_state.copy()
    states = [state.copy()]
    applied_controls: List[np.ndarray] = [np.zeros(2, dtype=np.float64)]
    metrics = [metrics_for_state(state)]
    for interval in range(CONTROL_INTERVALS):
        action = clip_action(controls[interval])
        for _ in range(HOLD_STEPS):
            state = step_state(state, action, thrust_scale)
            states.append(state.copy())
            applied_controls.append(action.copy())
            metrics.append(metrics_for_state(state))
    return states, applied_controls, metrics


def initial_guess(initial_state: np.ndarray) -> np.ndarray:
    vx = initial_state[2]
    vy = initial_state[3]
    speed = math.hypot(vx, vy)
    if speed <= 1.0e-12:
        base = np.zeros(2)
    else:
        base = -0.05 * np.asarray([vx, vy]) / speed
    guess = np.tile(base, (CONTROL_INTERVALS, 1))
    taper = np.linspace(1.0, 0.25, CONTROL_INTERVALS).reshape(-1, 1)
    return (guess * taper).reshape(-1)


def objective_value(mode: str, initial_state: np.ndarray, controls_flat: np.ndarray, thrust_scale: float) -> float:
    _states, applied, metrics = rollout_controls(initial_state, controls_flat, thrust_scale)
    final = metrics[-1]
    best = min(metrics, key=lambda item: item["distance_to_recoverable"])
    best_sync = min(item["sync_error"] for item in metrics)
    effort = sum(float(np.dot(action, action)) for action in applied) / max(1, len(applied))
    smooth = 0.0
    for idx in range(1, CONTROL_INTERVALS):
        prev = controls_flat.reshape(CONTROL_INTERVALS, 2)[idx - 1]
        cur = controls_flat.reshape(CONTROL_INTERVALS, 2)[idx]
        smooth += float(np.sum((cur - prev) ** 2))
    smooth /= max(1, CONTROL_INTERVALS - 1)

    if mode == "radius_only":
        return 1200.0 * final["r_error_ratio"] ** 2 + 12.0 * effort + 2.0 * smooth
    if mode == "recoverability_target":
        return (
            1.2 * final["distance_to_recoverable"] ** 2
            + 0.6 * best["distance_to_recoverable"] ** 2
            + 18.0 * effort
            + 3.0 * smooth
        )
    if mode == "sync_error_minimization":
        return 1.0 * best_sync**2 + 0.25 * final["sync_error"] ** 2 + 12.0 * effort + 2.0 * smooth
    if mode == "fuel_constrained_recoverability":
        return (
            1.0 * final["distance_to_recoverable"] ** 2
            + 0.8 * best["distance_to_recoverable"] ** 2
            + 120.0 * effort
            + 12.0 * smooth
        )
    raise ValueError(f"unknown objective mode: {mode}")


def solve_with_scipy(mode: str, r0: float, angle: float, thrust_scale: float, label: str) -> SolveResult:
    import time
    from scipy.optimize import minimize

    initial_state = state_from_case(r0, angle)
    x0 = initial_guess(initial_state)
    component_bound = MAX_ACTION_NORM / math.sqrt(2.0)
    bounds = [(-component_bound, component_bound)] * len(x0)
    start = time.time()
    result = minimize(
        lambda u: objective_value(mode, initial_state, np.asarray(u, dtype=np.float64), thrust_scale),
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": SCIPY_MAXITER, "ftol": 1.0e-7, "maxls": 20},
    )
    runtime = time.time() - start
    controls = np.asarray(result.x, dtype=np.float64)
    states, applied, metrics = rollout_controls(initial_state, controls, thrust_scale)
    return build_result_row(
        solver_backend="scipy_direct_shooting",
        mode=mode,
        case_label=label,
        r0=r0,
        angle=angle,
        thrust_scale=thrust_scale,
        solve_success=bool(result.success),
        solver_status=str(result.message),
        objective=float(result.fun),
        iterations=int(result.nit),
        runtime=runtime,
        states=states,
        applied_controls=applied,
        metrics=metrics,
    )


def solve_with_casadi(mode: str, r0: float, angle: float, thrust_scale: float, label: str) -> SolveResult:
    # Phase 32 is CasADi-first, but the checked runtime does not currently provide CasADi.
    # Keep the explicit failure path so missing solver support is reproducible in CSVs.
    available, status = has_casadi()
    if not available:
        raise RuntimeError(status)
    raise RuntimeError("CasADi backend is available but the Phase 32A prototype uses SciPy fallback in this environment")


def build_result_row(
    solver_backend: str,
    mode: str,
    case_label: str,
    r0: float,
    angle: float,
    thrust_scale: float,
    solve_success: bool,
    solver_status: str,
    objective: float,
    iterations: int,
    runtime: float,
    states: Sequence[np.ndarray],
    applied_controls: Sequence[np.ndarray],
    metrics: Sequence[Dict[str, float]],
) -> SolveResult:
    final = metrics[-1]
    best_idx = int(np.argmin([item["distance_to_recoverable"] for item in metrics]))
    best = metrics[best_idx]
    crossing_step = ""
    crossing_occurs = False
    for idx in range(1, len(metrics)):
        prev = metrics[idx - 1]["r_error_ratio"]
        cur = metrics[idx]["r_error_ratio"]
        if (prev > 0.0 and cur <= 0.0) or (prev < 0.0 and cur >= 0.0):
            crossing_step = idx
            crossing_occurs = True
            break
    recoverable_state = (
        abs(best["r_error_ratio"]) <= RECOVERABLE_R_RATIO
        and abs(best["vr_ratio"]) <= RECOVERABLE_VR_RATIO
        and abs(best["vt_error_ratio"]) <= RECOVERABLE_VT_RATIO
    )
    near_recoverable_state = best["sync_error"] <= 3.0 or best["distance_to_recoverable"] <= 3.0
    recoverable = bool(crossing_occurs and recoverable_state)
    near_recoverable = bool(crossing_occurs and near_recoverable_state)
    effort = sum(float(np.dot(action, action)) for action in applied_controls) / max(1, len(applied_controls))
    smooth = 0.0
    interval_controls = np.asarray(applied_controls[::HOLD_STEPS][1:], dtype=np.float64)
    if len(interval_controls) > 1:
        smooth = float(np.mean(np.sum(np.diff(interval_controls, axis=0) ** 2, axis=1)))
    max_action = max(float(np.linalg.norm(action)) for action in applied_controls)
    overspeed = any(item["speed_ratio"] > 1.90 for item in metrics)
    instability = any(item["radius_ratio"] < 0.35 or item["radius_ratio"] > 2.5 for item in metrics)
    row = {
        "solver_backend": solver_backend,
        "objective_mode": mode,
        "case_label": case_label,
        "r0_over_target": r0,
        "initial_velocity_angle_deg": angle,
        "thrust_scale": thrust_scale,
        "target_radius_scale": TARGET_RADIUS_SCALE,
        "solve_success": solve_success,
        "solver_status": solver_status,
        "objective_value": objective,
        "iterations": iterations,
        "horizon_steps": CONTROL_INTERVALS * HOLD_STEPS,
        "control_intervals": CONTROL_INTERVALS,
        "hold_steps": HOLD_STEPS,
        "crossing_occurs": crossing_occurs,
        "first_crossing_step": crossing_step,
        "best_sync_error": best["sync_error"],
        "best_distance_to_recoverable": best["distance_to_recoverable"],
        "final_sync_error": final["sync_error"],
        "final_distance_to_recoverable": final["distance_to_recoverable"],
        "final_r_error_ratio": final["r_error_ratio"],
        "final_vr_ratio": final["vr_ratio"],
        "final_vt_error_ratio": final["vt_error_ratio"],
        "best_r_error_ratio": best["r_error_ratio"],
        "best_vr_ratio": best["vr_ratio"],
        "best_vt_error_ratio": best["vt_error_ratio"],
        "recoverable_state": bool(recoverable_state),
        "near_recoverable_state": bool(near_recoverable_state),
        "recoverable_crossing": bool(recoverable),
        "near_recoverable_crossing": bool(near_recoverable),
        "capture_potential": bool(recoverable or near_recoverable),
        "success_potential": bool(recoverable and final["sync_error"] <= 1.0),
        "control_effort": effort,
        "control_smoothness": smooth,
        "max_action_norm": max_action,
        "overspeed": bool(overspeed),
        "instability": bool(instability),
        "runtime_seconds": runtime,
    }
    traj_rows: List[Dict[str, object]] = []
    for idx, (state, action, item) in enumerate(zip(states, applied_controls, metrics)):
        traj_rows.append(
            {
                "objective_mode": mode,
                "case_label": case_label,
                "step": idx,
                "time_seconds": idx * DT,
                "x": state[0],
                "y": state[1],
                "vx": state[2],
                "vy": state[3],
                "radius_ratio": item["radius_ratio"],
                "r_error_ratio": item["r_error_ratio"],
                "vr_ratio": item["vr_ratio"],
                "vt_error_ratio": item["vt_error_ratio"],
                "sync_error": item["sync_error"],
                "distance_to_recoverable": item["distance_to_recoverable"],
                "control_x": action[0],
                "control_y": action[1],
                "control_norm": float(np.linalg.norm(action)),
            }
        )
    return SolveResult(row=row, trajectory=traj_rows)


def write_research_context(casadi_status: str) -> None:
    lines = [
        "# Phase 32 Research Context",
        "",
        "## What Previous Layers Failed",
        "",
        "- Phase 25 showed existing crossings are near recoverable under relaxed thresholds, with tangential velocity as the dominant blocker.",
        "- Phase 28 showed window-producing and crossing-producing families are mostly disjoint.",
        "- Phase 29 Burn-A family selection did not improve crossings, recoverability, CAPTURE, or success.",
        "- Phase 30 Burn-A endpoint search mapped endpoint manifolds but did not change crossing-state structure.",
        "- Phase 31 bounded global transfer families still did not improve recoverable crossings, CAPTURE, or success.",
        "",
        "## Why Heuristics May Be Insufficient",
        "",
        "- Local, staged, endpoint, and named transfer-family approaches all encode limited structure.",
        "- They choose behaviors or coarse families, not a continuous state-control trajectory optimized against recoverability.",
        "- A negative result across these layers means the next question is theoretical reachability under the current physics.",
        "",
        "## Why Optimal Control Is Necessary",
        "",
        "- Direct optimal control can optimize all controls over a finite horizon at once.",
        "- It can target radius, radial velocity, tangential velocity, sync error, and effort simultaneously.",
        "- It provides an upper-bound style baseline rather than another production controller heuristic.",
        "",
        "## Architecture Comparison",
        "",
        "- Heuristic control: immediate feedback rules.",
        "- Staged transfer: fixed Burn A, coast, Burn B, handoff.",
        "- Family search: bounded named transfer templates.",
        "- Global transfer: architecture-level burn/coast schedules.",
        "- Continuous optimization: direct solve over a full control sequence under the same dynamics.",
        "",
        "## Optimal Under Current Physics",
        "",
        "- State is `[x, y, vx, vy]`.",
        "- Control is bounded normalized thrust `[u_x, u_y]`.",
        "- Dynamics reuse the project gravitational acceleration and thrust scaling.",
        "- No recoverability, CAPTURE, LOCK, reward, or physics threshold is changed.",
        f"- CasADi availability in this run: `{casadi_status}`.",
    ]
    RESEARCH_CONTEXT_MD.write_text("\n".join(lines), encoding="utf-8")


def run_phase32(prefer_casadi: bool) -> tuple[List[Dict[str, object]], List[Dict[str, object]], str]:
    available, casadi_status = has_casadi()
    backend = "casadi_direct_collocation" if prefer_casadi and available else "scipy_direct_shooting"
    if prefer_casadi and not available:
        casadi_status = f"unavailable, fallback used: {casadi_status}"
    rows: List[Dict[str, object]] = []
    traj_rows: List[Dict[str, object]] = []
    jobs = [(mode, *case) for mode in OBJECTIVE_MODES for case in CASES]
    for idx, (mode, r0, angle, thrust_scale, label) in enumerate(jobs, start=1):
        print(f"phase32 {idx}/{len(jobs)} mode={mode} case={label} backend={backend}")
        try:
            if backend == "casadi_direct_collocation":
                solved = solve_with_casadi(mode, r0, angle, thrust_scale, label)
            else:
                solved = solve_with_scipy(mode, r0, angle, thrust_scale, label)
        except Exception as exc:
            initial = state_from_case(r0, angle)
            controls = np.zeros(CONTROL_INTERVALS * 2, dtype=np.float64)
            states, applied, metrics = rollout_controls(initial, controls, thrust_scale)
            solved = build_result_row(
                solver_backend=backend,
                mode=mode,
                case_label=label,
                r0=r0,
                angle=angle,
                thrust_scale=thrust_scale,
                solve_success=False,
                solver_status=f"solve_failed: {exc}",
                objective=float("nan"),
                iterations=0,
                runtime=0.0,
                states=states,
                applied_controls=applied,
                metrics=metrics,
            )
        rows.append(solved.row)
        traj_rows.extend(solved.trajectory)
    return rows, traj_rows, casadi_status


def write_csvs(rows: Sequence[Dict[str, object]], traj_rows: Sequence[Dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with TRAJECTORY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRAJ_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in traj_rows:
            writer.writerow(row)


def aggregate(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for mode in OBJECTIVE_MODES:
        subset = [row for row in rows if row["objective_mode"] == mode]
        if not subset:
            continue
        stats[mode] = {
            "cases": len(subset),
            "solve_success": sum(bool_from_csv(row["solve_success"]) for row in subset),
            "crossings": sum(bool_from_csv(row["crossing_occurs"]) for row in subset),
            "near_recoverable": sum(bool_from_csv(row["near_recoverable_crossing"]) for row in subset),
            "recoverable": sum(bool_from_csv(row["recoverable_crossing"]) for row in subset),
            "near_recoverable_state": sum(bool_from_csv(row["near_recoverable_state"]) for row in subset),
            "recoverable_state": sum(bool_from_csv(row["recoverable_state"]) for row in subset),
            "capture_potential": sum(bool_from_csv(row["capture_potential"]) for row in subset),
            "success_potential": sum(bool_from_csv(row["success_potential"]) for row in subset),
            "mean_best_sync": float(np.mean([as_float(row["best_sync_error"]) for row in subset])),
            "mean_best_distance": float(np.mean([as_float(row["best_distance_to_recoverable"]) for row in subset])),
            "mean_effort": float(np.mean([as_float(row["control_effort"]) for row in subset])),
        }
    return stats


def phase31_reference() -> Dict[str, float]:
    rows = read_csv(PHASE31_RESULTS)
    subset = [row for row in rows if row.get("controller_name") == "phase31_phase22_baseline"]
    if not subset:
        return {"crossings": 12, "near_recoverable": 5, "recoverable": 0, "capture": 12, "success": 12}
    return {
        "crossings": sum(bool_from_csv(row.get("crossing_occurs")) for row in subset),
        "near_recoverable": sum(bool_from_csv(row.get("near_recoverable_crossing")) for row in subset),
        "recoverable": sum(bool_from_csv(row.get("recoverable_crossing")) for row in subset),
        "capture": sum(bool_from_csv(row.get("capture_entered")) for row in subset),
        "success": sum(bool_from_csv(row.get("success")) for row in subset),
    }


def save_plots(rows: Sequence[Dict[str, object]], traj_rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]]) -> None:
    best_row = min(rows, key=lambda row: as_float(row["best_distance_to_recoverable"], 1.0e9))
    best_mode = str(best_row["objective_mode"])
    best_case = str(best_row["case_label"])
    best_traj = [row for row in traj_rows if row["objective_mode"] == best_mode and row["case_label"] == best_case]

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    for mode in OBJECTIVE_MODES:
        mode_rows = [row for row in rows if row["objective_mode"] == mode]
        if not mode_rows:
            continue
        best = min(mode_rows, key=lambda row: as_float(row["best_distance_to_recoverable"], 1.0e9))
        traj = [row for row in traj_rows if row["objective_mode"] == mode and row["case_label"] == best["case_label"]]
        ax.plot([as_float(row["x"]) / TARGET_RADIUS for row in traj], [as_float(row["y"]) / TARGET_RADIUS for row in traj], label=mode, linewidth=1.2)
    theta = np.linspace(0, 2 * np.pi, 240)
    ax.plot(np.cos(theta), np.sin(theta), color="#333333", linestyle="--", linewidth=0.9, label="target radius")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Phase 32 optimal trajectory examples")
    ax.set_xlabel("x / target")
    ax.set_ylabel("y / target")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT_TRAJ, dpi=220)
    plt.close(fig)

    time_hours = [as_float(row["time_seconds"]) / 3600.0 for row in best_traj]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(time_hours, [abs(as_float(row["r_error_ratio"])) for row in best_traj], label="|r error|")
    ax.plot(time_hours, [abs(as_float(row["vr_ratio"])) for row in best_traj], label="|vr ratio|")
    ax.plot(time_hours, [abs(as_float(row["vt_error_ratio"])) for row in best_traj], label="|vt error|")
    ax.set_title(f"State error vs time: {best_mode}, {best_case}")
    ax.set_xlabel("time [hours]")
    ax.set_ylabel("absolute ratio")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_STATE, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(time_hours, [as_float(row["control_x"]) for row in best_traj], label="u_x")
    ax.plot(time_hours, [as_float(row["control_y"]) for row in best_traj], label="u_y")
    ax.plot(time_hours, [as_float(row["control_norm"]) for row in best_traj], label="|u|")
    ax.axhline(MAX_ACTION_NORM, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_title(f"Control profile: {best_mode}, {best_case}")
    ax.set_xlabel("time [hours]")
    ax.set_ylabel("normalized thrust")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_CONTROL, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(time_hours, [as_float(row["sync_error"]) for row in best_traj], label="sync error")
    ax.plot(time_hours, [as_float(row["distance_to_recoverable"]) for row in best_traj], label="distance")
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=0.8, label="recoverable boundary")
    ax.set_title(f"Sync error progression: {best_mode}, {best_case}")
    ax.set_xlabel("time [hours]")
    ax.set_ylabel("scaled error")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_SYNC, dpi=220)
    plt.close(fig)

    ref = phase31_reference()
    best_stats = min(stats.items(), key=lambda item: item[1]["mean_best_distance"])[1]
    labels = ["crossings", "near_recoverable", "recoverable", "capture/success potential"]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, [ref["crossings"], ref["near_recoverable"], ref["recoverable"], ref["capture"]], 0.36, label="Phase 31 baseline", color="#9D755D")
    ax.bar(x + 0.18, [best_stats["crossings"], best_stats["near_recoverable"], best_stats["recoverable"], best_stats["capture_potential"]], 0.36, label="Phase 32 best mode", color="#72B7B2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("cases")
    ax.set_title("Phase 32 vs Phase 31 comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PHASE31, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    labels = list(stats)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [stats[label]["mean_best_sync"] for label in labels], 0.4, label="mean best sync")
    ax.bar(x + 0.2, [stats[label]["mean_best_distance"] for label in labels], 0.4, label="mean best distance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Objective mode comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_MODES, dpi=220)
    plt.close(fig)


def write_summary(rows: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, object]], casadi_status: str) -> None:
    best_mode = min(stats, key=lambda mode: stats[mode]["mean_best_distance"])
    best = stats[best_mode]
    any_recoverable = any(item["recoverable"] > 0 for item in stats.values())
    any_near = any(item["near_recoverable"] > 0 for item in stats.values())
    phase31 = phase31_reference()
    phase31_improved = bool(best["recoverable"] > phase31["recoverable"] or best["capture_potential"] > phase31["capture"])
    lines = [
        "# Phase 32 Direct Optimal Control Baseline",
        "",
        "## Scope",
        "",
        "- Phase 32A coarse finite-horizon optimal-control prototype.",
        "- CasADi-first dependency detection; SciPy direct-shooting fallback used when CasADi is unavailable.",
        "- Dynamics preserve the project 2D gravity and thrust physics.",
        "- No CAPTURE/LOCK, reward, threshold, or physics rewrite.",
        f"- CasADi status: `{casadi_status}`.",
        f"- Horizon: `{CONTROL_INTERVALS * HOLD_STEPS}` physics steps with `{CONTROL_INTERVALS}` control intervals.",
        "",
        "## Objective Mode Results",
        "",
        "| Objective mode | Solves | Crossings | Near recoverable crossing | Recoverable crossing | Recoverable state | Capture potential | Mean best sync | Mean best distance | Mean effort |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, item in stats.items():
        lines.append(
            f"| `{mode}` | {item['solve_success']} / {item['cases']} | {item['crossings']} | {item['near_recoverable']} | "
            f"{item['recoverable']} | {item['recoverable_state']} | {item['capture_potential']} | {item['mean_best_sync']:.4f} | {item['mean_best_distance']:.4f} | {item['mean_effort']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Research Answers",
            "",
            f"1. Can optimal control outperform heuristic architectures? `{'yes' if phase31_improved else 'not on this Phase 32A prototype'}`.",
            f"2. Can recoverable crossing be reached? `{'yes' if any_recoverable else 'no'}`.",
            "3. Can CAPTURE improve? `not directly evaluated as a production CAPTURE rollout`; capture potential is reported instead.",
            f"4. Which objective formulation works best? `{best_mode}` by mean best recoverability distance.",
            f"5. Is recoverability theoretically reachable? `{'yes as a state in this coarse solve' if any(item['recoverable_state'] > 0 for item in stats.values()) else 'not demonstrated'}`.",
            f"6. Are prior failures mainly architecture failures? `{'plausibly' if any_recoverable else 'not proven'}`.",
            f"7. Does the current benchmark appear physically feasible? `{'partially' if any_near else 'not demonstrated by this finite horizon solve'}`.",
            "8. What should Phase 33 test? A real CasADi/IPOPT direct-collocation run after installing the declared CasADi dependency, then longer horizon or receding-horizon MPC.",
            "",
            "## Success Criteria",
            "",
            f"- Minimum, stable optimal-control solve runs: `{'met' if any(item['solve_success'] for item in stats.values()) else 'not met'}`.",
            f"- Moderate, better sync / lower recoverability distance: `{'met' if any_near else 'not met'}`.",
            f"- Strong, near recoverable crossing: `{'met' if any_near else 'not met'}`.",
            f"- Major, first recoverable crossing: `{'met' if any_recoverable else 'not met'}`.",
            f"- Breakthrough, CAPTURE/success improvement: `{'not evaluated'}`.",
            "",
            "## Honest Interpretation",
            "",
            "- This is an upper-bound prototype, not a production controller.",
            "- CasADi was declared in repo environment files but unavailable in the checked runtime, so the run used SciPy direct shooting.",
            "- If direct continuous optimization still cannot reach recoverability after a proper CasADi collocation run, the benchmark or thresholds may be structurally too hard.",
            "",
            "## Artifacts",
            "",
            "- `research_context.md`",
            "- `phase32_results.csv`",
            "- `phase32_trajectory_dataset.csv`",
            "- `optimal_trajectory_examples.png`",
            "- `state_error_vs_time.png`",
            "- `control_profile.png`",
            "- `sync_error_progression.png`",
            "- `phase32_vs_phase31_comparison.png`",
            "- `objective_mode_comparison.png`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 32 direct optimal control baseline.")
    parser.add_argument("--prefer-scipy", action="store_true", help="Skip CasADi attempt and use SciPy direct shooting.")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, traj_rows, casadi_status = run_phase32(prefer_casadi=not args.prefer_scipy)
    write_research_context(casadi_status)
    write_csvs(rows, traj_rows)
    stats = aggregate(rows)
    if not args.skip_plots:
        save_plots(rows, traj_rows, stats)
    write_summary(rows, stats, casadi_status)
    print(f"Saved Phase 32 direct optimal control outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
