from __future__ import annotations

import contextlib
import csv
import io
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from envs.orbit_env import OrbitEnv


SEED = 7
DT = 100.0
THRUST_SCALE = 10000.0
MAX_STEPS = 100000
R0_OVER_TARGET = 1.00005
TAIL_LEN = 5000
STRICT_CFG = {
    "tol_r": 1.0e-3,
    "tol_v": 1.0e-3,
    "tol_ang": 0.02,
    "success_threshold": 200,
}


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


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def recommended_workers(limit: int | None = None) -> int:
    cpu_count = os.cpu_count() or 2
    workers = max(1, min(cpu_count - 1, 12))
    if limit is not None:
        workers = min(workers, max(1, int(limit)))
    return workers


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


def specific_energy(mu: float, pos: np.ndarray, vel: np.ndarray) -> float:
    radius = float(np.linalg.norm(pos))
    speed = float(np.linalg.norm(vel))
    return 0.5 * speed * speed - mu / (radius + 1.0e-12)


def angular_momentum(pos: np.ndarray, vel: np.ndarray) -> float:
    return float(pos[0] * vel[1] - pos[1] * vel[0])


def compute_metrics(
    radius_values: Sequence[float],
    vr_values: Sequence[float],
    target_radius: float,
    success: bool,
    terminated: bool,
    truncated: bool,
    phase_transition_count: int,
) -> RolloutMetrics:
    radius_arr = np.asarray(radius_values, dtype=np.float64)
    vr_arr = np.asarray(vr_values, dtype=np.float64)
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
        phase_transition_count=int(phase_transition_count),
    )


def rollout_metrics(
    *,
    dt: float = DT,
    r0_over_target: float = R0_OVER_TARGET,
    seed: int = SEED,
    velocity_noise_pct: float = 0.0,
    position_perturb_frac: float = 0.0,
    action_noise_std: float = 0.0,
) -> RolloutMetrics:
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
    radius_values: List[float] = []
    vr_values: List[float] = []
    success = False
    terminated = False
    truncated = False
    phase_transition_count = 0
    while not (terminated or truncated):
        with contextlib.redirect_stdout(io.StringIO()):
            info = controller.act_with_info(obs)
        action = np.asarray(info["final_action"], dtype=np.float64)
        if action_noise_std > 0.0:
            action = np.clip(action + rng.normal(0.0, action_noise_std, size=2), -1.0, 1.0)
        obs, _, terminated, truncated, step_info = env.step(action)
        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1.0e-12)
        radius_values.append(radius)
        vr_values.append(float(np.dot(env.vel, r_hat)))
        phase_transition_count = int(info.get("phase_transition_count", len(controller.phase_transitions)))
        success = success or bool(step_info.get("success", False))
        obs = np.asarray(obs, dtype=np.float32)
    return compute_metrics(
        radius_values,
        vr_values,
        env.target_radius,
        success,
        terminated,
        truncated,
        phase_transition_count,
    )


def rollout_trace(
    *,
    dt: float = DT,
    r0_over_target: float = R0_OVER_TARGET,
    seed: int = SEED,
    velocity_noise_pct: float = 0.0,
    position_perturb_frac: float = 0.0,
    action_noise_std: float = 0.0,
) -> tuple[RolloutMetrics, Dict[str, List[float | str]]]:
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
    trace: Dict[str, List[float | str]] = {
        "step": [],
        "time": [],
        "radius": [],
        "radius_error": [],
        "v_r": [],
        "action_norm": [],
        "energy": [],
        "angular_momentum": [],
        "phase": [],
    }
    success = False
    terminated = False
    truncated = False
    phase_transition_count = 0
    while not (terminated or truncated):
        with contextlib.redirect_stdout(io.StringIO()):
            info = controller.act_with_info(obs)
        action = np.asarray(info["final_action"], dtype=np.float64)
        if action_noise_std > 0.0:
            action = np.clip(action + rng.normal(0.0, action_noise_std, size=2), -1.0, 1.0)
        obs, _, terminated, truncated, step_info = env.step(action)
        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1.0e-12)
        trace["step"].append(float(env.steps))
        trace["time"].append(float(env.steps * env.dt))
        trace["radius"].append(radius)
        trace["radius_error"].append(radius - env.target_radius)
        trace["v_r"].append(float(np.dot(env.vel, r_hat)))
        trace["action_norm"].append(float(np.linalg.norm(action)))
        trace["energy"].append(specific_energy(env.mu, env.pos, env.vel))
        trace["angular_momentum"].append(angular_momentum(env.pos, env.vel))
        trace["phase"].append(str(info.get("phase", controller.phase)))
        phase_transition_count = int(info.get("phase_transition_count", len(controller.phase_transitions)))
        success = success or bool(step_info.get("success", False))
        obs = np.asarray(obs, dtype=np.float32)
    metrics = compute_metrics(
        [float(x) for x in trace["radius"]],
        [float(x) for x in trace["v_r"]],
        env.target_radius,
        success,
        terminated,
        truncated,
        phase_transition_count,
    )
    return metrics, trace


def metrics_row(metrics: RolloutMetrics) -> Dict[str, object]:
    return asdict(metrics)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(fieldnames) if fieldnames is not None else list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bool_from_csv(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def save_heatmap(
    path: Path,
    rows: List[Dict[str, object]],
    *,
    x_key: str,
    y_key: str,
    value_key: str,
    title: str,
    cbar_label: str,
    log_scale: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = sorted({float(row[x_key]) for row in rows})
    y_values = sorted({float(row[y_key]) for row in rows})
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    for row in rows:
        x_idx = x_values.index(float(row[x_key]))
        y_idx = y_values.index(float(row[y_key]))
        value = row[value_key]
        if isinstance(value, bool):
            numeric = 1.0 if value else 0.0
        else:
            numeric = float(value)
        grid[y_idx, x_idx] = numeric

    plot_grid = np.log10(np.maximum(grid, 1.0e-12)) if log_scale else grid
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    image = ax.imshow(plot_grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{x:.5f}" for x in x_values], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([f"{y:g}" for y in y_values])
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(f"log10({cbar_label})" if log_scale else cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def phase_durations(trace: Dict[str, List[float | str]]) -> Dict[str, int]:
    phases = [str(x) for x in trace["phase"]]
    return {phase: phases.count(phase) for phase in ["DESCENT", "CAPTURE", "LOCK"]}


def first_failed_seed_for_action_noise(
    *,
    dt: float,
    r0_over_target: float,
    action_noise_std: float,
    start_seed: int = SEED,
    max_tries: int = 100,
) -> int:
    for offset in range(max_tries):
        seed = start_seed + offset
        metrics = rollout_metrics(dt=dt, r0_over_target=r0_over_target, seed=seed, action_noise_std=action_noise_std)
        if not metrics.success:
            return seed
    return start_seed
