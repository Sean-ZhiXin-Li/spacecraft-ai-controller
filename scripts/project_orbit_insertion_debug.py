from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from envs.orbit_env import OrbitEnv
from scripts.day20_policy_surface import DEFAULT_CHECKPOINTS, LoadedPolicy


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figs" / "project_orbit_insertion_debug"
SUMMARY_JSON = OUTPUT_DIR / "debug_results.json"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def norm2(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def compute_vr(pos: Sequence[float], vel: Sequence[float]) -> float:
    r = norm2(pos[0], pos[1]) + 1e-12
    return (pos[0] * vel[0] + pos[1] * vel[1]) / r


def compute_vt(pos: Sequence[float], vel: Sequence[float]) -> float:
    r = norm2(pos[0], pos[1]) + 1e-12
    r_hat_x = pos[0] / r
    r_hat_y = pos[1] / r
    t_hat_x = -r_hat_y
    t_hat_y = r_hat_x
    return vel[0] * t_hat_x + vel[1] * t_hat_y


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


def specific_energy(mu: float, pos: Sequence[float], vel: Sequence[float]) -> float:
    r = norm2(pos[0], pos[1]) + 1e-12
    v2 = vel[0] * vel[0] + vel[1] * vel[1]
    return 0.5 * v2 - mu / r


@dataclass
class FeasibilityRow:
    r0_over_target: float
    steps: int
    horizon_seconds: float
    radius_gap_m: float
    required_avg_radial_speed_mps: float
    required_const_accel_mps2: float
    max_delta_v_mps: float
    max_radial_displacement_bound_m: float
    bound_can_cover_gap: bool


@dataclass
class RolloutRow:
    controller: str
    checkpoint: str
    r0_over_target: float
    steps_run: int
    terminated: bool
    truncated: bool
    success: bool
    radius_crossings_total: int
    min_radius_error_m: float
    final_radius_error_m: float
    mean_action_norm: float
    mean_acc_thrust_mps2: float
    cumulative_delta_v_mps: float
    tail_mean_abs_vr_mps: float
    mean_alignment: float
    energy_change_j_per_kg: float
    within_success_tolerance_any: bool


class MaxInwardController:
    def act(self, obs: Sequence[float]) -> List[float]:
        pos = np.asarray(obs[:2], dtype=np.float64)
        r = np.linalg.norm(pos) + 1e-12
        r_hat = pos / r
        action = -r_hat
        return [float(action[0]), float(action[1])]


class MaxRetrogradeController:
    def act(self, obs: Sequence[float]) -> List[float]:
        vel = np.asarray(obs[2:4], dtype=np.float64)
        v = np.linalg.norm(vel)
        if v < 1e-12:
            return [0.0, 0.0]
        action = -(vel / v)
        return [float(action[0]), float(action[1])]


def make_env(thrust_scale: float, max_steps: int) -> OrbitEnv:
    return OrbitEnv(
        thrust_scale=thrust_scale,
        dt=2.0,
        max_steps=max_steps,
        reward_mode="orbit_circular_minimal",
        use_action_smoothing=False,
        use_orbit_capture_assist=False,
        verbose=False,
    )


def rollout_controller(
    controller_name: str,
    controller,
    *,
    checkpoint_name: str,
    r0_over_target: float,
    max_steps: int,
    thrust_scale: float,
) -> RolloutRow:
    env = make_env(thrust_scale=thrust_scale, max_steps=max_steps)
    set_default_start(env, r0_over_target)
    obs = env._get_obs()

    action_norms: List[float] = []
    acc_norms: List[float] = []
    radii: List[float] = [float(np.linalg.norm(env.pos))]
    vrs: List[float] = []
    alignments: List[float] = []
    energies: List[float] = [specific_energy(env.mu, env.pos, env.vel)]
    within_tol_any = False
    success = False

    for _ in range(max_steps):
        if hasattr(controller, "act_with_info"):
            action = controller.act(obs)
        else:
            action = controller.act(obs)
        action = np.asarray(action, dtype=np.float64)
        obs, _, terminated, truncated, info = env.step(action)
        action_exec = np.asarray(info["action_executed"], dtype=np.float64)
        acc_thrust = np.asarray(info["acc_thrust"], dtype=np.float64)
        r = float(np.linalg.norm(env.pos))
        vr = compute_vr(env.pos, env.vel)
        speed = float(np.linalg.norm(env.vel)) + 1e-12

        action_norm = float(np.linalg.norm(action_exec))
        acc_norm = float(np.linalg.norm(acc_thrust))
        if action_norm > 1e-12:
            r_hat = env.pos / (r + 1e-12)
            a_hat = action_exec / action_norm
            alignment = abs(float(np.dot(a_hat, r_hat)))
        else:
            alignment = 0.0

        action_norms.append(action_norm)
        acc_norms.append(acc_norm)
        radii.append(r)
        vrs.append(vr)
        alignments.append(alignment)
        energies.append(specific_energy(env.mu, env.pos, env.vel))
        success = success or bool(info.get("success", False))
        within_tol_any = within_tol_any or env._inside_tolerance(env.pos, env.vel)

        if terminated or truncated:
            break

    radius_arr = np.asarray(radii, dtype=np.float64)
    crossings = int(np.sum(np.signbit((radius_arr - env.target_radius)[1:]) != np.signbit((radius_arr - env.target_radius)[:-1]))) if len(radius_arr) > 1 else 0
    tail_vrs = np.asarray(vrs[-min(500, len(vrs)) :], dtype=np.float64) if vrs else np.zeros(0, dtype=np.float64)
    return RolloutRow(
        controller=controller_name,
        checkpoint=checkpoint_name,
        r0_over_target=float(r0_over_target),
        steps_run=len(action_norms),
        terminated=bool(len(action_norms) < max_steps),
        truncated=bool(len(action_norms) >= max_steps),
        success=bool(success),
        radius_crossings_total=crossings,
        min_radius_error_m=float(np.min(np.abs(radius_arr - env.target_radius))) if len(radius_arr) else 0.0,
        final_radius_error_m=float(abs(radius_arr[-1] - env.target_radius)) if len(radius_arr) else 0.0,
        mean_action_norm=float(np.mean(action_norms)) if action_norms else 0.0,
        mean_acc_thrust_mps2=float(np.mean(acc_norms)) if acc_norms else 0.0,
        cumulative_delta_v_mps=float(np.sum(acc_norms) * env.dt) if acc_norms else 0.0,
        tail_mean_abs_vr_mps=float(np.mean(np.abs(tail_vrs))) if len(tail_vrs) else 0.0,
        mean_alignment=float(np.mean(alignments)) if alignments else 0.0,
        energy_change_j_per_kg=float(energies[-1] - energies[0]) if len(energies) >= 2 else 0.0,
        within_success_tolerance_any=bool(within_tol_any),
    )


def feasibility_rows(env: OrbitEnv, r0_values: Sequence[float], step_values: Sequence[int]) -> List[FeasibilityRow]:
    rows: List[FeasibilityRow] = []
    a_max = env.thrust_scale / env.mass
    for r0 in r0_values:
        radius_gap = abs(r0 - 1.0) * env.target_radius
        for steps in step_values:
            horizon = steps * env.dt
            req_vr = radius_gap / horizon
            req_acc = 2.0 * radius_gap / (horizon * horizon)
            max_dv = a_max * horizon
            max_disp = 0.5 * a_max * horizon * horizon
            rows.append(
                FeasibilityRow(
                    r0_over_target=float(r0),
                    steps=int(steps),
                    horizon_seconds=float(horizon),
                    radius_gap_m=float(radius_gap),
                    required_avg_radial_speed_mps=float(req_vr),
                    required_const_accel_mps2=float(req_acc),
                    max_delta_v_mps=float(max_dv),
                    max_radial_displacement_bound_m=float(max_disp),
                    bound_can_cover_gap=bool(max_disp >= radius_gap),
                )
            )
    return rows


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    env = make_env(thrust_scale=20000.0, max_steps=60000)
    orbital_speed = math.sqrt(env.mu / env.target_radius)
    orbital_period = 2.0 * math.pi * env.target_radius / orbital_speed
    a_max = env.thrust_scale / env.mass

    physics_summary = {
        "target_radius_m": env.target_radius,
        "mu_m3_s2": env.mu,
        "spacecraft_mass_kg": env.mass,
        "dt_s": env.dt,
        "thrust_scale_n": env.thrust_scale,
        "max_acceleration_mps2": a_max,
        "max_delta_v_per_step_mps": a_max * env.dt,
        "circular_speed_mps": orbital_speed,
        "orbital_period_s": orbital_period,
        "orbital_period_days": orbital_period / 86400.0,
        "orbital_period_years": orbital_period / (86400.0 * 365.25),
        "success_tolerance_radius_m": env.tol_r * env.target_radius,
    }

    feas_rows = feasibility_rows(env, r0_values=[1.01, 1.03, 1.05], step_values=[4000, 20000, 60000])

    controllers: List[tuple[str, str, object]] = [
        ("ppo", "speed_refine_50", LoadedPolicy(DEFAULT_CHECKPOINTS["speed_refine_50"])),
        ("ppo", "state_vr_nonlinear_100", LoadedPolicy(DEFAULT_CHECKPOINTS["state_vr_nonlinear_100"])),
        ("explicit", "orbit_lock_controller", OrbitLockController(target_radius=env.target_radius, mu=env.mu, config=OrbitLockConfig())),
        ("scripted", "max_inward", MaxInwardController()),
        ("scripted", "max_retrograde", MaxRetrogradeController()),
    ]

    rollout_rows: List[RolloutRow] = []
    for r0 in [1.01, 1.05]:
        for group, name, controller in controllers:
            row = rollout_controller(
                controller_name=f"{group}:{name}",
                controller=controller,
                checkpoint_name=name,
                r0_over_target=r0,
                max_steps=4000,
                thrust_scale=20000.0,
            )
            rollout_rows.append(row)

    long_horizon_rows: List[RolloutRow] = []
    for group, name, controller in [
        ("scripted", "max_inward", MaxInwardController()),
        ("scripted", "max_retrograde", MaxRetrogradeController()),
    ]:
        for steps in [20000, 60000]:
            row = rollout_controller(
                controller_name=f"{group}:{name}:{steps}",
                controller=controller,
                checkpoint_name=name,
                r0_over_target=1.05,
                max_steps=steps,
                thrust_scale=20000.0,
            )
            long_horizon_rows.append(row)

    output = {
        "physics_summary": physics_summary,
        "feasibility_rows": [asdict(row) for row in feas_rows],
        "rollout_rows": [asdict(row) for row in rollout_rows],
        "long_horizon_rows": [asdict(row) for row in long_horizon_rows],
    }
    SUMMARY_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("[Physics]")
    print(json.dumps(physics_summary, indent=2))
    print("\n[Feasibility]")
    for row in feas_rows:
        print(json.dumps(asdict(row)))
    print("\n[Rollouts]")
    for row in rollout_rows:
        print(json.dumps(asdict(row)))
    print("\n[LongHorizon]")
    for row in long_horizon_rows:
        print(json.dumps(asdict(row)))
    print(f"\nSaved debug results to: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
