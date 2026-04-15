from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.orbit_env import OrbitEnv


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figs" / "time_scaling_reachability"
CSV_PATH = OUTPUT_DIR / "aggregate_results.csv"
JSON_PATH = OUTPUT_DIR / "aggregate_results.json"
SUMMARY_PATH = PROJECT_ROOT / "analysis" / "time_scaling_reachability.md"

DT_VALUES = [20.0, 50.0, 100.0]
R0_VALUES = [1.0005, 1.0002]
MAX_STEPS = 20000


@dataclass
class ReachabilityRow:
    dt: float
    r0_over_target: float
    max_steps: int
    success_at_start: bool
    success: bool
    terminated: bool
    truncated: bool
    radius_crossings_total: int
    tail_crosses_target_radius: bool
    final_radius_error: float
    tail_mean_abs_vr: float
    tail_radius_std: float
    tail_speed_cv: float
    min_abs_radius_error: float
    mean_action_norm: float
    first_crossing_step: int | None
    simulated_time: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def make_env(dt: float, max_steps: int) -> OrbitEnv:
    return OrbitEnv(
        thrust_scale=20000.0,
        dt=dt,
        max_steps=max_steps,
        reward_mode="orbit_circular_minimal",
        tol_r=1.0e-3,
        tol_v=1.0e-3,
        tol_ang=0.02,
        success_threshold=200,
        use_action_smoothing=False,
        use_orbit_capture_assist=False,
        verbose=False,
    )


class MaxRetrogradeController:
    def act(self, obs: Sequence[float]) -> List[float]:
        vel = np.asarray(obs[2:4], dtype=np.float64)
        v = np.linalg.norm(vel)
        if v < 1e-12:
            return [0.0, 0.0]
        action = -(vel / v)
        return [float(action[0]), float(action[1])]


def rollout(dt: float, r0_over_target: float, max_steps: int) -> ReachabilityRow:
    env = make_env(dt, max_steps)
    controller = MaxRetrogradeController()
    set_default_start(env, r0_over_target)

    success_at_start = env._inside_tolerance(env.pos, env.vel)
    obs = env._get_obs()

    radii: List[float] = [float(np.linalg.norm(env.pos))]
    speeds: List[float] = [float(np.linalg.norm(env.vel))]
    vrs: List[float] = []
    action_norms: List[float] = []
    success = False
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = np.asarray(controller.act(obs), dtype=np.float64)
        obs, _, terminated, truncated, info = env.step(action)

        radius = float(np.linalg.norm(env.pos))
        speed = float(np.linalg.norm(env.vel))
        r_hat = env.pos / (radius + 1e-12)
        vr = float(np.dot(env.vel, r_hat))

        radii.append(radius)
        speeds.append(speed)
        vrs.append(vr)
        action_norms.append(float(np.linalg.norm(np.asarray(info["action_executed"], dtype=np.float64))))
        success = success or bool(info.get("success", False))

        if terminated or truncated:
            break

    radius_arr = np.asarray(radii, dtype=np.float64)
    speed_arr = np.asarray(speeds, dtype=np.float64)
    vr_arr = np.asarray(vrs, dtype=np.float64)
    re = radius_arr - env.target_radius
    sign_flip_mask = np.signbit(re[1:]) != np.signbit(re[:-1]) if len(re) > 1 else np.zeros(0, dtype=bool)
    crossing_indices = np.flatnonzero(sign_flip_mask)
    first_crossing_step = int(crossing_indices[0] + 1) if len(crossing_indices) else None
    tail_len = min(1000, len(radius_arr))
    tail_slice = slice(len(radius_arr) - tail_len, len(radius_arr))
    vr_tail = vr_arr[-min(1000, len(vr_arr)) :] if len(vr_arr) else np.zeros(0, dtype=np.float64)
    speed_tail = speed_arr[tail_slice]
    re_tail = re[tail_slice]

    return ReachabilityRow(
        dt=float(dt),
        r0_over_target=float(r0_over_target),
        max_steps=int(max_steps),
        success_at_start=bool(success_at_start),
        success=bool(success),
        terminated=bool(terminated),
        truncated=bool(truncated),
        radius_crossings_total=int(np.sum(sign_flip_mask)),
        tail_crosses_target_radius=bool(np.any(np.signbit(re_tail[1:]) != np.signbit(re_tail[:-1]))) if len(re_tail) > 1 else False,
        final_radius_error=float(abs(re[-1])) if len(re) else 0.0,
        tail_mean_abs_vr=float(np.mean(np.abs(vr_tail))) if len(vr_tail) else 0.0,
        tail_radius_std=float(np.std(radius_arr[tail_slice])) if len(radius_arr) else 0.0,
        tail_speed_cv=float(np.std(speed_tail) / (np.mean(speed_tail) + 1e-12)) if len(speed_tail) else 0.0,
        min_abs_radius_error=float(np.min(np.abs(re))) if len(re) else 0.0,
        mean_action_norm=float(np.mean(action_norms)) if action_norms else 0.0,
        first_crossing_step=first_crossing_step,
        simulated_time=float(dt * (first_crossing_step if first_crossing_step is not None else len(radius_arr) - 1)),
    )


def write_csv(rows: List[ReachabilityRow]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(rows: List[ReachabilityRow], tested_dt_values: List[float], tested_r0_values: List[float], tested_max_steps_values: List[int]) -> None:
    crossed = [row for row in rows if row.radius_crossings_total > 0]
    first_cross = min(crossed, key=lambda row: (row.dt, row.r0_over_target, row.first_crossing_step or row.max_steps)) if crossed else None

    if first_cross is not None:
        recommended = first_cross
    else:
        recommended = min(
            rows,
            key=lambda row: (row.final_radius_error, row.tail_mean_abs_vr, row.dt, row.r0_over_target),
        )

    lines = [
        "# Time Scaling Reachability",
        "",
        "## Test Setup",
        "",
        "- Controller: `probe_max_retrograde`",
        f"- `r0_over_target`: `{', '.join(f'{value:.5f}' for value in tested_r0_values)}`",
        f"- `max_steps` tested: `{', '.join(str(value) for value in tested_max_steps_values)}`",
        f"- `dt` tested: `{', '.join(str(int(value)) for value in tested_dt_values)}`",
        "- Strict tolerances:",
        "  - `tol_r = 0.001`",
        "  - `tol_v = 0.001`",
        "  - `tol_ang = 0.02`",
        "",
        "## Main Answers",
        "",
    ]

    if first_cross is None:
        lines.append("- Crossing did not occur for any tested `dt`.")
        lines.append("- No new physically reachable time scale was found in this sweep.")
    else:
        lines.append(
            f"- Crossing first became possible at `dt = {int(first_cross.dt)}` for `r0_over_target = {first_cross.r0_over_target:.5f}`."
        )
        lines.append(f"- First crossing step: `{first_cross.first_crossing_step}`")
        lines.append(f"- Total simulated time to first crossing: `{first_cross.simulated_time:.3e}` seconds")

    lines.extend([
        "",
        "## Per-Configuration Results",
        "",
    ])
    for row in rows:
        lines.append(
            f"- `dt={int(row.dt)}`, `r0={row.r0_over_target:.5f}`: crossings `{row.radius_crossings_total}`, "
            f"first_crossing_step `{row.first_crossing_step}`, simulated_time `{row.simulated_time:.3e}`, final_radius_error `{row.final_radius_error:.3e}`, "
            f"tail_mean_abs_vr `{row.tail_mean_abs_vr:.3f}`"
        )

    lines.extend([
        "",
        "## Recommended Baseline",
        "",
        f"- Recommended new `dt`: `{int(recommended.dt)}`",
        f"- Supporting configuration: `r0_over_target = {recommended.r0_over_target:.5f}`, `max_steps = {recommended.max_steps}`",
        f"- Metrics: crossings `{recommended.radius_crossings_total}`, first_crossing_step `{recommended.first_crossing_step}`, simulated_time `{recommended.simulated_time:.3e}`, final_radius_error `{recommended.final_radius_error:.3e}`",
    ])

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe-only dt/max_steps reachability sweep.")
    parser.add_argument("--configs", type=str, default=None, help="Semicolon-separated dt:max_steps pairs, e.g. 50:100000;100:200000")
    parser.add_argument("--dt-values", type=str, default=None, help="Comma-separated dt values.")
    parser.add_argument("--r0-values", type=str, default=None, help="Comma-separated r0_over_target values.")
    parser.add_argument("--max-steps-values", type=str, default=None, help="Comma-separated max_steps values.")
    args = parser.parse_args()

    dt_values = [float(x.strip()) for x in args.dt_values.split(",")] if args.dt_values else list(DT_VALUES)
    r0_values = [float(x.strip()) for x in args.r0_values.split(",")] if args.r0_values else list(R0_VALUES)
    max_steps_values = [int(x.strip()) for x in args.max_steps_values.split(",")] if args.max_steps_values else [MAX_STEPS]
    if args.configs:
        config_pairs = []
        for item in args.configs.split(";"):
            item = item.strip()
            if not item:
                continue
            dt_text, max_steps_text = item.split(":")
            config_pairs.append((float(dt_text.strip()), int(max_steps_text.strip())))
    else:
        config_pairs = [(dt, max_steps) for dt in dt_values for max_steps in max_steps_values]

    ensure_dir(OUTPUT_DIR)
    rows = [rollout(dt, r0, max_steps) for dt, max_steps in config_pairs for r0 in r0_values]
    rows = sorted(rows, key=lambda row: (row.dt, row.max_steps, row.r0_over_target))

    for row in rows:
        print(json.dumps(asdict(row)))

    write_csv(rows)
    JSON_PATH.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    write_summary(
        rows,
        sorted({dt for dt, _ in config_pairs}),
        sorted(set(r0_values)),
        sorted({max_steps for _, max_steps in config_pairs}),
    )
    print(f"Saved dt sweep outputs to: {OUTPUT_DIR}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
