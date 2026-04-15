from __future__ import annotations

import csv
import json
import math
import sys
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


BENCHMARK_CSV = PROJECT_ROOT / "analysis" / "figs" / "orbit_lock_benchmark" / "aggregate_results.csv"
GENERALIZATION_MD = PROJECT_ROOT / "analysis" / "orbit_lock_generalization.md"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figs" / "final_project"
SUMMARY_JSON = OUTPUT_DIR / "final_project_plot_summary.json"

STRICT_CFG = {
    "tol_r": 1.0e-3,
    "tol_v": 1.0e-3,
    "tol_ang": 0.02,
    "success_threshold": 200,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_benchmark_rows() -> List[Dict[str, str]]:
    with BENCHMARK_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def get_best_success_setup() -> Dict[str, float]:
    rows = load_benchmark_rows()
    explicit_success = [row for row in rows if row["controller"] == "explicit_orbit_lock" and row["setup_name"] == "best_success"]
    if not explicit_success:
        raise RuntimeError("Missing best_success explicit row in benchmark output.")
    # The benchmark summary already tells us the setup identity.
    return {
        "name": "best_success",
        "r0_over_target": 1.00005,
        "dt": 100.0,
        "thrust_scale": 10000.0,
        "max_steps": 100000,
    }


def make_env(dt: float, thrust_scale: float, max_steps: int) -> OrbitEnv:
    return OrbitEnv(
        thrust_scale=thrust_scale,
        dt=dt,
        max_steps=max_steps,
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


def rollout_trace(controller, setup: Dict[str, float]) -> Dict[str, List[float]]:
    env = make_env(setup["dt"], setup["thrust_scale"], setup["max_steps"])
    set_default_start(env, setup["r0_over_target"])
    obs = env._get_obs()

    trace: Dict[str, List[float]] = {
        "step": [],
        "radius": [],
        "v_r": [],
        "action_norm": [],
    }
    terminated = False
    truncated = False
    for step in range(int(setup["max_steps"])):
        action = np.asarray(controller.act(obs), dtype=np.float64)
        obs, _, terminated, truncated, info = env.step(action)
        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1e-12)
        v_r = float(np.dot(env.vel, r_hat))
        trace["step"].append(float(step))
        trace["radius"].append(radius)
        trace["v_r"].append(v_r)
        trace["action_norm"].append(float(np.linalg.norm(np.asarray(info["action_executed"], dtype=np.float64))))
        if terminated or truncated:
            break
    trace["target_radius"] = [env.target_radius] * len(trace["step"])
    return trace


def save_success_plot(rows: List[Dict[str, str]]) -> None:
    controllers = ["ppo_speed_refine_50", "probe_max_retrograde", "explicit_orbit_lock"]
    success_counts = []
    for controller in controllers:
        sub = [row for row in rows if row["controller"] == controller]
        success_counts.append(sum(1 for row in sub if row["success"] == "True"))

    plt.figure(figsize=(7.5, 4.8))
    plt.bar(controllers, success_counts, color=["#c44e52", "#4c72b0", "#55a868"])
    plt.ylabel("success count")
    plt.title("Success Comparison Across Representative Setups")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "success_comparison.png", dpi=180)
    plt.close()


def save_trace_plot(path: Path, traces: Dict[str, Dict[str, List[float]]], key: str, title: str, ylabel: str, target_value: float | None = None) -> None:
    plt.figure(figsize=(10, 4.8))
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


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    benchmark_rows = load_benchmark_rows()
    save_success_plot(benchmark_rows)

    setup = get_best_success_setup()
    traces = {
        "PPO": rollout_trace(LoadedPolicy(DEFAULT_CHECKPOINTS["speed_refine_50"]), setup),
        "Probe": rollout_trace(MaxRetrogradeController(), setup),
        "Explicit": rollout_trace(
            OrbitLockController(
                target_radius=make_env(setup["dt"], setup["thrust_scale"], setup["max_steps"]).target_radius,
                mu=make_env(setup["dt"], setup["thrust_scale"], setup["max_steps"]).mu,
                config=OrbitLockConfig(),
            ),
            setup,
        ),
    }
    target_radius = traces["Explicit"]["target_radius"][0]
    save_trace_plot(OUTPUT_DIR / "radius_vs_time.png", traces, "radius", "Radius vs Time | Representative Successful Setup", "radius [m]", target_value=target_radius)
    save_trace_plot(OUTPUT_DIR / "v_r_vs_time.png", traces, "v_r", "Radial Velocity vs Time | Representative Successful Setup", "v_r [m/s]", target_value=0.0)
    save_trace_plot(OUTPUT_DIR / "action_norm_vs_time.png", traces, "action_norm", "Action Norm vs Time | Representative Successful Setup", "action_norm", target_value=0.0)

    summary = {
        "representative_setup": setup,
        "benchmark_source": BENCHMARK_CSV.as_posix(),
        "generalization_summary": GENERALIZATION_MD.as_posix(),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved plots to: {OUTPUT_DIR}")
    print(f"Saved plot summary to: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
