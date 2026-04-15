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


GENERALIZATION_CSV = PROJECT_ROOT / "analysis" / "figs" / "orbit_lock_generalization" / "aggregate_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figs" / "orbit_lock_benchmark"
CSV_PATH = OUTPUT_DIR / "aggregate_results.csv"
JSON_PATH = OUTPUT_DIR / "aggregate_results.json"
SUMMARY_PATH = PROJECT_ROOT / "analysis" / "orbit_lock_benchmark.md"

STRICT_CFG = {
    "tol_r": 1.0e-3,
    "tol_v": 1.0e-3,
    "tol_ang": 0.02,
    "success_threshold": 200,
}
MAX_STEPS = 100000


@dataclass
class Setup:
    name: str
    r0_over_target: float
    dt: float
    thrust_scale: float


@dataclass
class BenchmarkRow:
    setup_name: str
    controller: str
    success: bool
    terminated: bool
    truncated: bool
    radius_crossings_total: int
    tail_crosses_target_radius: bool
    first_crossing_step: int | None
    final_radius_error: float
    tail_mean_abs_vr: float
    tail_radius_std: float
    tail_speed_cv: float
    amplitude_shrinks: bool
    sustained_crossing_score: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_env(dt: float, thrust_scale: float) -> OrbitEnv:
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


class MaxRetrogradeController:
    def act(self, obs: Sequence[float]) -> List[float]:
        vel = np.asarray(obs[2:4], dtype=np.float64)
        v = np.linalg.norm(vel)
        if v < 1e-12:
            return [0.0, 0.0]
        action = -(vel / v)
        return [float(action[0]), float(action[1])]


def tail_slice(length: int, tail_len: int = 5000) -> slice:
    use_len = min(tail_len, length)
    return slice(length - use_len, length)


def load_representative_setups() -> List[Setup]:
    rows: List[Dict[str, str]] = []
    with GENERALIZATION_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    success_rows = [row for row in rows if row["success"] == "True"]
    crossing_rows = [row for row in rows if int(row["radius_crossings_total"]) > 0]
    failure_rows = [row for row in rows if int(row["radius_crossings_total"]) == 0]

    best_success = min(success_rows, key=lambda row: float(row["final_radius_error"]))
    fastest_cross = min(crossing_rows, key=lambda row: int(row["first_crossing_step"]))
    representative_failure = min(failure_rows, key=lambda row: float(row["final_radius_error"]))

    selected = [
        ("best_success", best_success),
        ("fastest_cross", fastest_cross),
        ("representative_failure", representative_failure),
    ]
    dedup: Dict[tuple[float, float, float], Setup] = {}
    for name, row in selected:
        key = (float(row["r0_over_target"]), float(row["dt"]), float(row["thrust_scale"]))
        if key not in dedup:
            dedup[key] = Setup(name=name, r0_over_target=key[0], dt=key[1], thrust_scale=key[2])
    return list(dedup.values())


def rollout_one(setup: Setup, controller_name: str, controller) -> tuple[BenchmarkRow, Dict[str, List[float]]]:
    env = make_env(setup.dt, setup.thrust_scale)
    set_default_start(env, setup.r0_over_target)
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
    speed_arr = np.diff(radius_arr, prepend=radius_arr[0])
    re = radius_arr - env.target_radius
    sign_flip_mask = np.signbit(re[1:]) != np.signbit(re[:-1]) if len(re) > 1 else np.zeros(0, dtype=bool)
    crossing_indices = np.flatnonzero(sign_flip_mask)
    first_crossing_step = int(crossing_indices[0] + 1) if len(crossing_indices) else None
    ts = tail_slice(len(radius_arr))
    re_tail = re[ts]
    vr_tail = vr_arr[ts]
    tail_cross_mask = np.signbit(re_tail[1:]) != np.signbit(re_tail[:-1]) if len(re_tail) > 1 else np.zeros(0, dtype=bool)
    first_half = re[: max(1, len(re) // 2)]
    second_half = re[len(re) // 2 :]

    row = BenchmarkRow(
        setup_name=setup.name,
        controller=controller_name,
        success=bool(success),
        terminated=bool(terminated),
        truncated=bool(truncated),
        radius_crossings_total=int(np.sum(sign_flip_mask)),
        tail_crosses_target_radius=bool(np.any(tail_cross_mask)),
        first_crossing_step=first_crossing_step,
        final_radius_error=float(abs(re[-1])) if len(re) else 0.0,
        tail_mean_abs_vr=float(np.mean(np.abs(vr_tail))) if len(vr_tail) else 0.0,
        tail_radius_std=float(np.std(radius_arr[ts])) if len(radius_arr) else 0.0,
        tail_speed_cv=float(np.std(speed_arr[ts]) / (np.mean(np.abs(speed_arr[ts])) + 1e-12)) if len(speed_arr) else 0.0,
        amplitude_shrinks=bool(np.std(second_half) < np.std(first_half)) if len(re) >= 20 else False,
        sustained_crossing_score=int(np.sum(tail_cross_mask)),
    )
    return row, trace


def save_plot(path: Path, traces: Dict[str, Dict[str, List[float]]], key: str, title: str, ylabel: str, target_value: float | None = None) -> None:
    plt.figure(figsize=(9.5, 4.8))
    for label, trace in traces.items():
        plt.plot(trace["step"], trace[key], label=label, linewidth=1.3)
    if target_value is not None:
        plt.axhline(target_value, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_csv(rows: List[BenchmarkRow]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(setups: List[Setup], rows: List[BenchmarkRow]) -> None:
    lines = [
        "# Orbit Lock Benchmark",
        "",
        "## Representative Setups",
        "",
    ]
    for setup in setups:
        lines.append(
            f"- `{setup.name}`: `r0_over_target={setup.r0_over_target:.5f}`, `dt={int(setup.dt)}`, `thrust_scale={int(setup.thrust_scale)}`"
        )

    lines.extend(["", "## Results", ""])
    for row in rows:
        lines.append(
            f"- `{row.setup_name}` / `{row.controller}`: crossings `{row.radius_crossings_total}`, "
            f"tail_crosses `{row.tail_crosses_target_radius}`, success `{row.success}`, "
            f"final_radius_error `{row.final_radius_error:.3e}`, tail_mean_abs_vr `{row.tail_mean_abs_vr:.3f}`"
        )

    explicit_rows = [row for row in rows if row.controller == "explicit_orbit_lock"]
    probe_rows = [row for row in rows if row.controller == "probe_max_retrograde"]
    ppo_rows = [row for row in rows if row.controller == "ppo_speed_refine_50"]
    explicit_successes = sum(1 for row in explicit_rows if row.success)
    probe_successes = sum(1 for row in probe_rows if row.success)
    ppo_successes = sum(1 for row in ppo_rows if row.success)
    lines.extend([
        "",
        "## Main Answers",
        "",
        f"- Explicit controller successes: `{explicit_successes}` / `{len(explicit_rows)}`",
        f"- Probe controller successes: `{probe_successes}` / `{len(probe_rows)}`",
        f"- PPO successes: `{ppo_successes}` / `{len(ppo_rows)}`",
        "- The explicit controller should be compared as a structured insertion controller, not as a generic baseline.",
    ])
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    setups = load_representative_setups()
    rows: List[BenchmarkRow] = []
    traces_by_setup: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for setup in setups:
        traces_by_setup[setup.name] = {}
        for controller_name, controller in {
            "probe_max_retrograde": MaxRetrogradeController(),
            "explicit_orbit_lock": OrbitLockController(target_radius=make_env(setup.dt, setup.thrust_scale).target_radius, mu=make_env(setup.dt, setup.thrust_scale).mu, config=OrbitLockConfig()),
            "ppo_speed_refine_50": LoadedPolicy(DEFAULT_CHECKPOINTS["speed_refine_50"]),
        }.items():
            row, trace = rollout_one(setup, controller_name, controller)
            rows.append(row)
            traces_by_setup[setup.name][controller_name] = trace
            print(json.dumps(asdict(row)))

    rows.sort(key=lambda row: (row.setup_name, row.controller))
    write_csv(rows)
    JSON_PATH.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    for setup in setups:
        env = make_env(setup.dt, setup.thrust_scale)
        save_plot(OUTPUT_DIR / f"{setup.name}_radius.png", traces_by_setup[setup.name], "radius", f"radius | {setup.name}", "radius [m]", target_value=env.target_radius)
        save_plot(OUTPUT_DIR / f"{setup.name}_v_r.png", traces_by_setup[setup.name], "v_r", f"v_r | {setup.name}", "v_r [m/s]", target_value=0.0)
        save_plot(OUTPUT_DIR / f"{setup.name}_action_norm.png", traces_by_setup[setup.name], "action_norm", f"action_norm | {setup.name}", "action_norm", target_value=0.0)
    write_summary(setups, rows)
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
