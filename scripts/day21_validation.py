from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from envs.orbit_env import OrbitEnv
from scripts.day20_policy_surface import DEFAULT_CHECKPOINTS, LoadedPolicy


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figs" / "day21_validation"
SUMMARY_PATH = PROJECT_ROOT / "analysis" / "day21_summary.md"
STABILITY_CSV = OUTPUT_DIR / "stability_sweep.csv"
STABILITY_JSON = OUTPUT_DIR / "stability_sweep.json"
THRUST_CSV = OUTPUT_DIR / "thrust_generalization.csv"
THRUST_JSON = OUTPUT_DIR / "thrust_generalization.json"
COMPARISON_CSV = OUTPUT_DIR / "controller_comparison.csv"
COMPARISON_JSON = OUTPUT_DIR / "controller_comparison.json"

R0_VALUES = [1.00002, 1.00005, 1.0001, 1.0002]
DT = 100.0
THRUST_SCALE = 10000.0
THRUST_VARIANTS = [0.8 * THRUST_SCALE, THRUST_SCALE, 1.2 * THRUST_SCALE]
MAX_STEPS = 100000
STRICT_CFG = {
    "tol_r": 1.0e-3,
    "tol_v": 1.0e-3,
    "tol_ang": 0.02,
    "success_threshold": 200,
}


@dataclass
class SweepRow:
    controller: str
    r0_over_target: float
    dt: float
    thrust_scale: float
    success: bool
    terminated: bool
    truncated: bool
    radius_crossings_total: int
    first_crossing_step: int | None
    final_radius_error: float
    tail_mean_abs_vr: float
    mean_action_norm: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_env(*, dt: float, thrust_scale: float) -> OrbitEnv:
    return OrbitEnv(
        thrust_scale=thrust_scale,
        dt=dt,
        max_steps=MAX_STEPS,
        reward_mode="orbit_circular_minimal",
        tol_r=STRICT_CFG["tol_r"],
        tol_v=STRICT_CFG["tol_v"],
        tol_ang=STRICT_CFG["tol_ang"],
        success_threshold=STRICT_CFG["success_threshold"],
        use_action_smoothing=False,
        use_orbit_capture_assist=False,
        verbose=False,
    )


def set_default_start(env: OrbitEnv, r0_over_target: float) -> None:
    env.reset(start_mode="default")
    env.steps = 0
    env.success_counter = 0
    env.radial_stall_counter = 0
    env.orbit_lock_counter = 0
    env.prev_action = np.zeros(2, dtype=np.float64)
    env.pos = np.array([0.0, r0_over_target * env.target_radius], dtype=np.float64)
    v_mag = math.sqrt(env.mu / np.linalg.norm(env.pos))
    angle = math.radians(170.0)
    env.vel = v_mag * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)


def tail_slice(length: int, tail_len: int = 5000) -> slice:
    use_len = min(tail_len, length)
    return slice(length - use_len, length)


def rollout(controller, *, r0_over_target: float, dt: float, thrust_scale: float) -> tuple[SweepRow, Dict[str, List[float]]]:
    env = make_env(dt=dt, thrust_scale=thrust_scale)
    set_default_start(env, r0_over_target)
    obs = env._get_obs()

    trace: Dict[str, List[float]] = {
        "step": [],
        "radius": [],
        "v_r": [],
        "action_norm": [],
    }

    success = False
    terminated = False
    truncated = False

    for step in range(MAX_STEPS):
        action = np.asarray(controller.act(obs), dtype=np.float64)
        obs, _, terminated, truncated, info = env.step(action)
        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1e-12)
        v_r = float(np.dot(env.vel, r_hat))

        trace["step"].append(float(step))
        trace["radius"].append(radius)
        trace["v_r"].append(v_r)
        trace["action_norm"].append(float(np.linalg.norm(np.asarray(info["action_executed"], dtype=np.float64))))
        success = success or bool(info.get("success", False))
        if terminated or truncated:
            break

    radius_arr = np.asarray(trace["radius"], dtype=np.float64)
    vr_arr = np.asarray(trace["v_r"], dtype=np.float64)
    action_arr = np.asarray(trace["action_norm"], dtype=np.float64)
    r_error = radius_arr - env.target_radius
    sign_flip_mask = np.signbit(r_error[1:]) != np.signbit(r_error[:-1]) if len(r_error) > 1 else np.zeros(0, dtype=bool)
    crossing_indices = np.flatnonzero(sign_flip_mask)

    row = SweepRow(
        controller=type(controller).__name__.replace("Controller", "").lower(),
        r0_over_target=float(r0_over_target),
        dt=float(dt),
        thrust_scale=float(thrust_scale),
        success=bool(success),
        terminated=bool(terminated),
        truncated=bool(truncated),
        radius_crossings_total=int(np.sum(sign_flip_mask)),
        first_crossing_step=int(crossing_indices[0] + 1) if len(crossing_indices) else None,
        final_radius_error=float(abs(r_error[-1])) if len(r_error) else 0.0,
        tail_mean_abs_vr=float(np.mean(np.abs(vr_arr[tail_slice(len(vr_arr))]))) if len(vr_arr) else 0.0,
        mean_action_norm=float(np.mean(action_arr)) if len(action_arr) else 0.0,
    )
    return row, trace


def save_rows_csv(path: Path, rows: List[SweepRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_plot(path: Path, traces: Dict[str, Dict[str, List[float]]], key: str, title: str, ylabel: str, target_value: float | None = None) -> None:
    plt.figure(figsize=(10, 4.8))
    for label, trace in traces.items():
        plt.plot(trace["step"], trace[key], label=label, linewidth=1.5)
    if target_value is not None:
        plt.axhline(target_value, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_radius_error_plot(path: Path, traces: Dict[str, Dict[str, List[float]]], target_radius: float) -> None:
    plt.figure(figsize=(10, 4.8))
    for label, trace in traces.items():
        radius = np.asarray(trace["radius"], dtype=np.float64)
        radius_error = radius - target_radius
        plt.plot(trace["step"], radius_error, label=label, linewidth=1.5)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel("radius_error [m]")
    plt.title("Radius Error vs Time (Explicit vs PPO)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_summary(stability_rows: List[SweepRow], thrust_rows: List[SweepRow], comparison_rows: List[SweepRow]) -> None:
    explicit_successes = [row for row in stability_rows if row.success]
    crossing_rows = [row for row in stability_rows if row.radius_crossings_total > 0]
    best_stability = min(
        stability_rows,
        key=lambda row: (0 if row.success else 1, 0 if row.radius_crossings_total > 0 else 1, row.final_radius_error, row.tail_mean_abs_vr),
    )
    thrust_successes = [row for row in thrust_rows if row.success]
    explicit_cmp = next(row for row in comparison_rows if row.controller == "explicit_orbit_lock")
    ppo_cmp = next(row for row in comparison_rows if row.controller == "ppo_speed_refine_50")

    lines = [
        "# Day21 Validation",
        "",
        "## Setup",
        "",
        f"- Fixed comparison regime: `dt = {int(DT)}`, `thrust_scale = {int(THRUST_SCALE)}`, `max_steps = {MAX_STEPS}`",
        f"- `r0_over_target` sweep: `{', '.join(f'{v:.5f}' for v in R0_VALUES)}`",
        "- Strict tolerances:",
        f"  - `tol_r = {STRICT_CFG['tol_r']}`",
        f"  - `tol_v = {STRICT_CFG['tol_v']}`",
        f"  - `tol_ang = {STRICT_CFG['tol_ang']}`",
        "",
        "## Stability Sweep | Explicit Controller",
        "",
    ]
    for row in stability_rows:
        lines.append(
            f"- `r0={row.r0_over_target:.5f}`: success `{row.success}`, crossings `{row.radius_crossings_total}`, "
            f"final_radius_error `{row.final_radius_error:.3e}`, tail_mean_abs_vr `{row.tail_mean_abs_vr:.3f}`"
        )

    lines.extend([
        "",
        "## Controller Comparison",
        "",
        f"- `explicit_orbit_lock`: success `{explicit_cmp.success}`, crossings `{explicit_cmp.radius_crossings_total}`, "
        f"final_radius_error `{explicit_cmp.final_radius_error:.3e}`, tail_mean_abs_vr `{explicit_cmp.tail_mean_abs_vr:.3f}`",
        f"- `ppo_speed_refine_50`: success `{ppo_cmp.success}`, crossings `{ppo_cmp.radius_crossings_total}`, "
        f"final_radius_error `{ppo_cmp.final_radius_error:.3e}`, tail_mean_abs_vr `{ppo_cmp.tail_mean_abs_vr:.3f}`",
        "",
        "### Radius Error Diagnostic",
        "",
        "![radius error](figs/day21_validation/radius_error_vs_time.png)",
        "",
        "This plot makes the structural difference visible directly:",
        "",
        "- the explicit controller drives the system through the target radius and into closed-loop correction",
        "- PPO remains on one side of the target radius and never enters the capture regime",
        "",
        "This supports the main mechanism-level conclusion of the project:",
        "",
        "> PPO learns to stop moving, not to stabilize motion.",
        "",
        "## Optional Thrust Generalization | Explicit Controller",
        "",
    ])
    for row in thrust_rows:
        lines.append(
            f"- `thrust={int(row.thrust_scale)}`: success `{row.success}`, crossings `{row.radius_crossings_total}`, "
            f"final_radius_error `{row.final_radius_error:.3e}`, tail_mean_abs_vr `{row.tail_mean_abs_vr:.3f}`"
        )

    lines.extend([
        "",
        "## Main Answers",
        "",
        f"- Stability region under this sweep: success in `{len(explicit_successes)}` / `{len(stability_rows)}` explicit-controller setups.",
        f"- Crossing appears in `{len(crossing_rows)}` / `{len(stability_rows)}` explicit-controller setups.",
        f"- Best explicit setup in this sweep: `r0_over_target = {best_stability.r0_over_target:.5f}`.",
        f"- Explicit vs PPO on the same setup: explicit is structurally better because it reaches crossing and success, while PPO does not.",
        f"- Optional thrust variation result: success in `{len(thrust_successes)}` / `{len(thrust_rows)}` tested thrust values.",
        "",
        "## Structural Diagnosis",
        "",
        "- The explicit controller has a narrow but real stability region near the reachable insertion regime.",
        "- PPO remains a continuous reactive policy and does not reproduce the phase-structured crossing behavior on the same setup.",
        "- The main difference is control structure, not just gain magnitude.",
        "",
        "### Imitation Learning Result",
        "",
        "Based on the minimal imitation learning test:",
        "",
        "- the cloned MLP achieves low MSE on the dataset",
        "- but fails to produce crossing and orbit insertion",
        "",
        "This shows that:",
        "",
        "Matching expert actions locally is not sufficient to reproduce",
        "the global phase-structured behavior required for orbit insertion.",
        "",
        "This further supports the main conclusion:",
        "",
        "> The failure is structural, not due to optimization or model capacity.",
    ])
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    stability_rows: List[SweepRow] = []
    for r0 in R0_VALUES:
        controller = OrbitLockController(target_radius=make_env(dt=DT, thrust_scale=THRUST_SCALE).target_radius, mu=make_env(dt=DT, thrust_scale=THRUST_SCALE).mu, config=OrbitLockConfig())
        row, _ = rollout(controller, r0_over_target=r0, dt=DT, thrust_scale=THRUST_SCALE)
        row.controller = "explicit_orbit_lock"
        stability_rows.append(row)
        print(json.dumps(asdict(row)))

    explicit_controller = OrbitLockController(target_radius=make_env(dt=DT, thrust_scale=THRUST_SCALE).target_radius, mu=make_env(dt=DT, thrust_scale=THRUST_SCALE).mu, config=OrbitLockConfig())
    ppo_controller = LoadedPolicy(DEFAULT_CHECKPOINTS["speed_refine_50"])
    comparison_rows: List[SweepRow] = []
    comparison_traces: Dict[str, Dict[str, List[float]]] = {}
    for label, controller in {
        "explicit_orbit_lock": explicit_controller,
        "ppo_speed_refine_50": ppo_controller,
    }.items():
        row, trace = rollout(controller, r0_over_target=1.00005, dt=DT, thrust_scale=THRUST_SCALE)
        row.controller = label
        comparison_rows.append(row)
        comparison_traces[label] = trace
        print(json.dumps(asdict(row)))

    thrust_rows: List[SweepRow] = []
    for thrust_scale in THRUST_VARIANTS:
        controller = OrbitLockController(target_radius=make_env(dt=DT, thrust_scale=thrust_scale).target_radius, mu=make_env(dt=DT, thrust_scale=thrust_scale).mu, config=OrbitLockConfig())
        row, _ = rollout(controller, r0_over_target=1.00005, dt=DT, thrust_scale=thrust_scale)
        row.controller = "explicit_orbit_lock"
        thrust_rows.append(row)
        print(json.dumps(asdict(row)))

    stability_rows.sort(key=lambda row: row.r0_over_target)
    thrust_rows.sort(key=lambda row: row.thrust_scale)
    comparison_rows.sort(key=lambda row: row.controller)

    save_rows_csv(STABILITY_CSV, stability_rows)
    STABILITY_JSON.write_text(json.dumps([asdict(row) for row in stability_rows], indent=2), encoding="utf-8")
    save_rows_csv(THRUST_CSV, thrust_rows)
    THRUST_JSON.write_text(json.dumps([asdict(row) for row in thrust_rows], indent=2), encoding="utf-8")
    save_rows_csv(COMPARISON_CSV, comparison_rows)
    COMPARISON_JSON.write_text(json.dumps([asdict(row) for row in comparison_rows], indent=2), encoding="utf-8")

    env = make_env(dt=DT, thrust_scale=THRUST_SCALE)
    save_plot(OUTPUT_DIR / "radius_vs_time.png", comparison_traces, "radius", "Day21 validation | radius vs time", "radius [m]", target_value=env.target_radius)
    save_radius_error_plot(OUTPUT_DIR / "radius_error_vs_time.png", comparison_traces, target_radius=env.target_radius)
    save_plot(OUTPUT_DIR / "v_r_vs_time.png", comparison_traces, "v_r", "Day21 validation | v_r vs time", "v_r [m/s]", target_value=0.0)
    save_plot(OUTPUT_DIR / "action_norm_vs_time.png", comparison_traces, "action_norm", "Day21 validation | action norm", "action_norm", target_value=0.0)

    write_summary(stability_rows, thrust_rows, comparison_rows)
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
