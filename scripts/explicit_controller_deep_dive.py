from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from envs.orbit_env import OrbitEnv


OUTPUT_DIR = PROJECT_ROOT / "analysis"
LOCAL_STABILITY_CSV = OUTPUT_DIR / "local_stability_map.csv"
LOCAL_STABILITY_PNG = OUTPUT_DIR / "local_stability_map.png"
DT_SENSITIVITY_CSV = OUTPUT_DIR / "dt_sensitivity.csv"
DT_SENSITIVITY_PNG = OUTPUT_DIR / "dt_sensitivity.png"
ROBUSTNESS_CSV = OUTPUT_DIR / "robustness_results.csv"
ROBUSTNESS_MD = OUTPUT_DIR / "robustness_summary.md"
ENERGY_PNG = OUTPUT_DIR / "energy_vs_time.png"
ANGULAR_MOMENTUM_PNG = OUTPUT_DIR / "angular_momentum.png"
PHASE_STATS_JSON = OUTPUT_DIR / "phase_statistics.json"

SEED = 7
DT = 100.0
THRUST_SCALE = 10000.0
MAX_STEPS = 100000
R0_OVER_TARGET = 1.00005
STRICT_CFG = {
    "tol_r": 1.0e-3,
    "tol_v": 1.0e-3,
    "tol_ang": 0.02,
    "success_threshold": 200,
}

LOCAL_R0_VALUES = [1.00001, 1.00005, 1.0001, 1.001, 1.005, 1.01]
DT_VALUES = [50.0, 100.0, 200.0, 500.0]
ROBUSTNESS_TRIALS = 20
TAIL_LEN = 5000


@dataclass
class RolloutMetrics:
    success: bool
    crossing_occurs: bool
    radius_crossings_total: int
    first_crossing_step: int | None
    final_radius_error: float
    tail_mean_abs_vr: float
    steps: int
    terminated: bool
    truncated: bool
    phase_transition_count: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_env(dt: float = DT) -> OrbitEnv:
    return OrbitEnv(
        thrust_scale=THRUST_SCALE,
        dt=float(dt),
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


def set_default_start(
    env: OrbitEnv,
    r0_over_target: float,
    *,
    rng: np.random.Generator | None = None,
    velocity_noise_pct: float = 0.0,
    position_perturb_frac: float = 0.0,
) -> None:
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

    if rng is not None and position_perturb_frac > 0.0:
        direction = rng.normal(size=2)
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1.0e-12:
            direction = direction / direction_norm
            magnitude = rng.uniform(-position_perturb_frac, position_perturb_frac) * env.target_radius
            env.pos = env.pos + magnitude * direction

    if rng is not None and velocity_noise_pct > 0.0:
        env.vel = env.vel * (1.0 + rng.uniform(-velocity_noise_pct, velocity_noise_pct, size=2))


def tail_slice(length: int) -> slice:
    use_len = min(TAIL_LEN, length)
    return slice(length - use_len, length)


def compute_specific_energy(mu: float, pos: np.ndarray, vel: np.ndarray) -> float:
    radius = float(np.linalg.norm(pos))
    speed = float(np.linalg.norm(vel))
    return 0.5 * speed * speed - mu / (radius + 1.0e-12)


def compute_angular_momentum(pos: np.ndarray, vel: np.ndarray) -> float:
    return float(pos[0] * vel[1] - pos[1] * vel[0])


def compute_metrics(trace: Dict[str, List[float]], target_radius: float, success: bool, terminated: bool, truncated: bool) -> RolloutMetrics:
    radius_arr = np.asarray(trace["radius"], dtype=np.float64)
    vr_arr = np.asarray(trace["v_r"], dtype=np.float64)
    r_error = radius_arr - target_radius
    sign_flip_mask = np.signbit(r_error[1:]) != np.signbit(r_error[:-1]) if len(r_error) > 1 else np.zeros(0, dtype=bool)
    crossing_indices = np.flatnonzero(sign_flip_mask)
    ts = tail_slice(len(vr_arr))
    return RolloutMetrics(
        success=bool(success),
        crossing_occurs=bool(np.any(sign_flip_mask)),
        radius_crossings_total=int(np.sum(sign_flip_mask)),
        first_crossing_step=int(crossing_indices[0] + 1) if len(crossing_indices) else None,
        final_radius_error=float(abs(r_error[-1])) if len(r_error) else 0.0,
        tail_mean_abs_vr=float(np.mean(np.abs(vr_arr[ts]))) if len(vr_arr) else 0.0,
        steps=int(len(radius_arr)),
        terminated=bool(terminated),
        truncated=bool(truncated),
        phase_transition_count=int(max(trace["phase_transition_count"]) if trace["phase_transition_count"] else 0),
    )


def rollout(
    *,
    dt: float = DT,
    r0_over_target: float = R0_OVER_TARGET,
    seed: int = SEED,
    velocity_noise_pct: float = 0.0,
    position_perturb_frac: float = 0.0,
    action_noise_std: float = 0.0,
    keep_trace: bool = True,
) -> tuple[RolloutMetrics, Dict[str, List[float]]]:
    rng = np.random.default_rng(seed)
    env = make_env(dt=dt)
    controller = OrbitLockController(target_radius=env.target_radius, mu=env.mu, config=OrbitLockConfig())
    set_default_start(
        env,
        r0_over_target,
        rng=rng,
        velocity_noise_pct=velocity_noise_pct,
        position_perturb_frac=position_perturb_frac,
    )
    obs = np.asarray(env._get_obs(), dtype=np.float32)

    trace: Dict[str, List[float]] = {
        "step": [],
        "time": [],
        "radius": [],
        "r_error": [],
        "v_r": [],
        "speed": [],
        "energy": [],
        "angular_momentum": [],
        "action_norm": [],
        "phase": [],
        "radial_cmd": [],
        "tangential_cmd": [],
        "phase_transition_count": [],
    }

    success = False
    terminated = False
    truncated = False
    while not (terminated or truncated):
        with contextlib.redirect_stdout(io.StringIO()):
            info = controller.act_with_info(obs)
        action = np.asarray(info["final_action"], dtype=np.float64)
        if action_noise_std > 0.0:
            action = np.clip(action + rng.normal(0.0, action_noise_std, size=2), -1.0, 1.0)
        next_obs, _, terminated, truncated, step_info = env.step(action)

        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1.0e-12)
        v_r = float(np.dot(env.vel, r_hat))
        speed = float(np.linalg.norm(env.vel))

        if keep_trace:
            trace["step"].append(float(env.steps))
            trace["time"].append(float(env.steps * env.dt))
            trace["radius"].append(radius)
            trace["r_error"].append(radius - env.target_radius)
            trace["v_r"].append(v_r)
            trace["speed"].append(speed)
            trace["energy"].append(compute_specific_energy(env.mu, env.pos, env.vel))
            trace["angular_momentum"].append(compute_angular_momentum(env.pos, env.vel))
            trace["action_norm"].append(float(np.linalg.norm(action)))
            trace["phase"].append(str(info.get("phase", controller.phase)))
            trace["radial_cmd"].append(float(info.get("radial_cmd", 0.0)))
            trace["tangential_cmd"].append(float(info.get("tangential_cmd", 0.0)))
            trace["phase_transition_count"].append(float(info.get("phase_transition_count", len(controller.phase_transitions))))
        else:
            trace["radius"].append(radius)
            trace["v_r"].append(v_r)
            trace["phase_transition_count"].append(float(info.get("phase_transition_count", len(controller.phase_transitions))))

        success = success or bool(step_info.get("success", False))
        obs = np.asarray(next_obs, dtype=np.float32)

    metrics = compute_metrics(trace, env.target_radius, success, terminated, truncated)
    return metrics, trace


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str] | None = None) -> None:
    keys = list(fieldnames) if fieldnames is not None else list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def save_local_stability_plot(rows: List[Dict[str, object]]) -> None:
    x = np.asarray([float(row["r0_over_target"]) for row in rows])
    final_error = np.asarray([float(row["final_radius_error"]) for row in rows])
    tail_vr = np.asarray([float(row["tail_mean_abs_vr"]) for row in rows])
    success = np.asarray([bool(row["success"]) for row in rows])

    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    ax1.plot(x, final_error, marker="o", label="final radius error", color="#1565C0")
    ax1.set_xlabel("r0_over_target")
    ax1.set_ylabel("final radius error [m]", color="#1565C0")
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.plot(x, tail_vr, marker="s", label="tail mean abs v_r", color="#C62828")
    ax2.set_ylabel("tail mean abs v_r [m/s]", color="#C62828")
    ax2.tick_params(axis="y", labelcolor="#C62828")
    for xi, ok in zip(x, success):
        ax1.scatter([xi], [final_error[list(x).index(xi)]], s=90, facecolors="none", edgecolors="#2E7D32" if ok else "#6D4C41", linewidths=2.0)
    ax1.set_title("Local stability basin scan")
    fig.tight_layout()
    fig.savefig(LOCAL_STABILITY_PNG, dpi=180)
    plt.close(fig)


def resample(values: List[float], count: int = 1000) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return np.zeros(count, dtype=np.float64)
    if len(arr) == 1:
        return np.full(count, arr[0], dtype=np.float64)
    src = np.linspace(0.0, 1.0, len(arr))
    dst = np.linspace(0.0, 1.0, count)
    return np.interp(dst, src, arr)


def save_dt_sensitivity_plot(rows: List[Dict[str, object]]) -> None:
    dt_values = [float(row["dt"]) for row in rows]
    final_error = [float(row["final_radius_error"]) for row in rows]
    radius_rmse = [float(row["radius_rmse_vs_dt100"]) for row in rows]

    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    ax1.plot(dt_values, final_error, marker="o", label="final radius error", color="#1565C0")
    ax1.set_xlabel("dt [s]")
    ax1.set_ylabel("final radius error [m]", color="#1565C0")
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.plot(dt_values, radius_rmse, marker="s", label="radius RMSE vs dt=100", color="#EF6C00")
    ax2.set_ylabel("trajectory radius RMSE [m]", color="#EF6C00")
    ax2.tick_params(axis="y", labelcolor="#EF6C00")
    ax2.set_yscale("log")
    ax1.set_title("Time step sensitivity")
    fig.tight_layout()
    fig.savefig(DT_SENSITIVITY_PNG, dpi=180)
    plt.close(fig)


def run_local_stability_scan() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for r0 in LOCAL_R0_VALUES:
        metrics, _ = rollout(r0_over_target=r0, keep_trace=False)
        row = {"r0_over_target": r0, **asdict(metrics)}
        rows.append(row)
        print(f"local_stability r0={r0}: success={metrics.success} crossings={metrics.radius_crossings_total}")
    write_csv(
        LOCAL_STABILITY_CSV,
        rows,
        [
            "r0_over_target",
            "success",
            "crossing_occurs",
            "radius_crossings_total",
            "first_crossing_step",
            "final_radius_error",
            "tail_mean_abs_vr",
            "steps",
            "terminated",
            "truncated",
            "phase_transition_count",
        ],
    )
    save_local_stability_plot(rows)
    return rows


def run_dt_sensitivity() -> tuple[List[Dict[str, object]], Dict[float, Dict[str, List[float]]]]:
    traces: Dict[float, Dict[str, List[float]]] = {}
    metric_by_dt: Dict[float, RolloutMetrics] = {}
    for dt in DT_VALUES:
        metrics, trace = rollout(dt=dt, keep_trace=True)
        traces[dt] = trace
        metric_by_dt[dt] = metrics
        print(f"dt_sensitivity dt={dt}: success={metrics.success} final_error={metrics.final_radius_error:.3e}")

    reference = resample(traces[100.0]["radius"])
    rows: List[Dict[str, object]] = []
    for dt in DT_VALUES:
        radius_curve = resample(traces[dt]["radius"])
        vr_curve = resample(traces[dt]["v_r"])
        ref_vr = resample(traces[100.0]["v_r"])
        radius_rmse = float(np.sqrt(np.mean((radius_curve - reference) ** 2)))
        vr_rmse = float(np.sqrt(np.mean((vr_curve - ref_vr) ** 2)))
        rows.append(
            {
                "dt": dt,
                **asdict(metric_by_dt[dt]),
                "radius_rmse_vs_dt100": radius_rmse,
                "vr_rmse_vs_dt100": vr_rmse,
            }
        )

    write_csv(
        DT_SENSITIVITY_CSV,
        rows,
        [
            "dt",
            "success",
            "crossing_occurs",
            "radius_crossings_total",
            "first_crossing_step",
            "final_radius_error",
            "tail_mean_abs_vr",
            "steps",
            "terminated",
            "truncated",
            "phase_transition_count",
            "radius_rmse_vs_dt100",
            "vr_rmse_vs_dt100",
        ],
    )
    save_dt_sensitivity_plot(rows)
    return rows, traces


def run_robustness() -> List[Dict[str, object]]:
    settings = [
        {
            "setting": "initial_velocity_noise_pm_1pct",
            "velocity_noise_pct": 0.01,
            "position_perturb_frac": 0.0,
            "action_noise_std": 0.0,
        },
        {
            "setting": "small_position_perturbation",
            "velocity_noise_pct": 0.0,
            "position_perturb_frac": 1.0e-5,
            "action_noise_std": 0.0,
        },
        {
            "setting": "small_action_noise",
            "velocity_noise_pct": 0.0,
            "position_perturb_frac": 0.0,
            "action_noise_std": 0.005,
        },
    ]

    rows: List[Dict[str, object]] = []
    for setting in settings:
        trial_metrics: List[RolloutMetrics] = []
        for trial in range(ROBUSTNESS_TRIALS):
            metrics, _ = rollout(
                seed=SEED + 1000 * (settings.index(setting) + 1) + trial,
                velocity_noise_pct=float(setting["velocity_noise_pct"]),
                position_perturb_frac=float(setting["position_perturb_frac"]),
                action_noise_std=float(setting["action_noise_std"]),
                keep_trace=False,
            )
            trial_metrics.append(metrics)
            print(f"robustness {setting['setting']} trial={trial}: success={metrics.success}")

        final_errors = np.asarray([m.final_radius_error for m in trial_metrics], dtype=np.float64)
        tail_vrs = np.asarray([m.tail_mean_abs_vr for m in trial_metrics], dtype=np.float64)
        successes = np.asarray([m.success for m in trial_metrics], dtype=bool)
        crossings = np.asarray([m.radius_crossings_total for m in trial_metrics], dtype=np.float64)
        rows.append(
            {
                "setting": setting["setting"],
                "trials": ROBUSTNESS_TRIALS,
                "velocity_noise_pct": setting["velocity_noise_pct"],
                "position_perturb_frac": setting["position_perturb_frac"],
                "action_noise_std": setting["action_noise_std"],
                "success_rate": float(np.mean(successes)),
                "crossing_rate": float(np.mean(crossings > 0)),
                "mean_final_radius_error": float(np.mean(final_errors)),
                "var_final_radius_error": float(np.var(final_errors)),
                "std_final_radius_error": float(np.std(final_errors)),
                "mean_tail_mean_abs_vr": float(np.mean(tail_vrs)),
                "var_tail_mean_abs_vr": float(np.var(tail_vrs)),
                "std_tail_mean_abs_vr": float(np.std(tail_vrs)),
                "min_final_radius_error": float(np.min(final_errors)),
                "max_final_radius_error": float(np.max(final_errors)),
            }
        )

    write_csv(
        ROBUSTNESS_CSV,
        rows,
        [
            "setting",
            "trials",
            "velocity_noise_pct",
            "position_perturb_frac",
            "action_noise_std",
            "success_rate",
            "crossing_rate",
            "mean_final_radius_error",
            "var_final_radius_error",
            "std_final_radius_error",
            "mean_tail_mean_abs_vr",
            "var_tail_mean_abs_vr",
            "std_tail_mean_abs_vr",
            "min_final_radius_error",
            "max_final_radius_error",
        ],
    )
    write_robustness_summary(rows)
    return rows


def write_robustness_summary(rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Robustness Summary",
        "",
        "## Setup",
        "",
        f"- trials per setting: `{ROBUSTNESS_TRIALS}`",
        f"- baseline: `dt={DT}`, `max_steps={MAX_STEPS}`, `r0_over_target={R0_OVER_TARGET}`, `thrust_scale={THRUST_SCALE}`",
        "- perturbations are applied without changing environment physics or controller gains",
        "",
        "## Results",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['setting']}`: success_rate `{row['success_rate']:.2f}`, crossing_rate `{row['crossing_rate']:.2f}`, "
            f"mean_final_radius_error `{row['mean_final_radius_error']:.3e}`, std_final_radius_error `{row['std_final_radius_error']:.3e}`, "
            f"mean_tail_mean_abs_vr `{row['mean_tail_mean_abs_vr']:.3e}`"
        )

    weakest = min(rows, key=lambda row: (float(row["success_rate"]), -float(row["mean_final_radius_error"])))
    strongest = max(rows, key=lambda row: (float(row["success_rate"]), -float(row["mean_final_radius_error"])))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Most robust setting in this scan: `{strongest['setting']}`.",
            f"- Weakest setting in this scan: `{weakest['setting']}`.",
            "- Treat these as local robustness checks around the validated 2D baseline, not as a global robustness proof.",
        ]
    )
    ROBUSTNESS_MD.write_text("\n".join(lines), encoding="utf-8")


def save_physics_plot(path: Path, trace: Dict[str, List[float]], key: str, title: str, ylabel: str) -> None:
    phase_colors = {"DESCENT": "#1565C0", "CAPTURE": "#EF6C00", "LOCK": "#00897B"}
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    time = np.asarray(trace["time"], dtype=np.float64)
    values = np.asarray(trace[key], dtype=np.float64)
    phases = np.asarray(trace["phase"], dtype=object)
    for phase, color in phase_colors.items():
        mask = phases == phase
        if np.any(mask):
            ax.scatter(time[mask], values[mask], s=2, color=color, label=phase, alpha=0.75)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compute_phase_statistics(trace: Dict[str, List[float]], metrics: RolloutMetrics) -> Dict[str, object]:
    phases = np.asarray(trace["phase"], dtype=object)
    stats: Dict[str, object] = {
        "baseline_setup": {
            "dt": DT,
            "max_steps": MAX_STEPS,
            "r0_over_target": R0_OVER_TARGET,
            "thrust_scale": THRUST_SCALE,
            **STRICT_CFG,
        },
        "metrics": asdict(metrics),
        "phase_order": ["DESCENT", "CAPTURE", "LOCK"],
        "phase_stats": {},
    }
    for phase in ["DESCENT", "CAPTURE", "LOCK"]:
        mask = phases == phase
        phase_count = int(np.sum(mask))
        phase_data: Dict[str, object] = {"count": phase_count}
        if phase_count > 0:
            for key in ["energy", "angular_momentum", "v_r", "r_error", "action_norm", "radial_cmd", "tangential_cmd"]:
                arr = np.asarray(trace[key], dtype=np.float64)[mask]
                phase_data[f"{key}_mean"] = float(np.mean(arr))
                phase_data[f"{key}_std"] = float(np.std(arr))
                phase_data[f"{key}_min"] = float(np.min(arr))
                phase_data[f"{key}_max"] = float(np.max(arr))
        stats["phase_stats"][phase] = phase_data
    return stats


def run_physics_analysis() -> tuple[RolloutMetrics, Dict[str, object]]:
    metrics, trace = rollout(keep_trace=True)
    save_physics_plot(ENERGY_PNG, trace, "energy", "Specific energy vs time by phase", "specific energy [J/kg]")
    save_physics_plot(ANGULAR_MOMENTUM_PNG, trace, "angular_momentum", "Angular momentum vs time by phase", "specific angular momentum [m^2/s]")
    stats = compute_phase_statistics(trace, metrics)
    PHASE_STATS_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return metrics, stats


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting local stability scan")
    run_local_stability_scan()
    print("Starting dt sensitivity scan")
    run_dt_sensitivity()
    print("Starting robustness scan")
    run_robustness()
    print("Starting physics analysis")
    metrics, _ = run_physics_analysis()
    print(json.dumps({"baseline_physics_metrics": asdict(metrics)}, indent=2))


if __name__ == "__main__":
    main()
