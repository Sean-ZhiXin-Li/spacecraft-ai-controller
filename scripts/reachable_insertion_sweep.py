from __future__ import annotations

import csv
import json
import math
import sys
import argparse
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


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figs" / "reachable_insertion_sweep"
CSV_PATH = OUTPUT_DIR / "aggregate_results.csv"
JSON_PATH = OUTPUT_DIR / "aggregate_results.json"
SUMMARY_PATH = PROJECT_ROOT / "analysis" / "reachable_insertion_sweep_summary.md"

R0_VALUES = [1.0005, 1.0010, 1.0020]
MAX_STEPS_VALUES = [4000, 10000, 20000, 60000]


@dataclass
class SweepRow:
    controller: str
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
    tail_mean_alignment: float
    tail_speed_cv: float
    amplitude_shrinks: bool
    within_tolerance_any: bool
    min_abs_radius_error: float
    mean_action_norm: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def norm2(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def compute_vr(pos: Sequence[float], vel: Sequence[float]) -> float:
    r = norm2(pos[0], pos[1]) + 1e-12
    return (pos[0] * vel[0] + pos[1] * vel[1]) / r


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


def make_strict_env(max_steps: int) -> OrbitEnv:
    return OrbitEnv(
        thrust_scale=20000.0,
        dt=2.0,
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


def build_controllers(target_radius: float, mu: float) -> Dict[str, object]:
    return {
        "ppo_speed_refine_50": LoadedPolicy(DEFAULT_CHECKPOINTS["speed_refine_50"]),
        "explicit_orbit_lock": OrbitLockController(target_radius=target_radius, mu=mu, config=OrbitLockConfig()),
        "probe_max_retrograde": MaxRetrogradeController(),
    }


def rollout_one(controller_name: str, controller, r0_over_target: float, max_steps: int) -> SweepRow:
    env = make_strict_env(max_steps=max_steps)
    set_default_start(env, r0_over_target)

    success_at_start = env._inside_tolerance(env.pos, env.vel)
    obs = env._get_obs()

    radii: List[float] = [float(np.linalg.norm(env.pos))]
    vrs: List[float] = []
    action_norms: List[float] = []
    alignments: List[float] = []
    speeds: List[float] = [float(np.linalg.norm(env.vel))]
    within_tolerance_any = bool(success_at_start)
    success = False
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = np.asarray(controller.act(obs), dtype=np.float64)
        obs, _, terminated, truncated, info = env.step(action)

        action_exec = np.asarray(info["action_executed"], dtype=np.float64)
        radius = float(np.linalg.norm(env.pos))
        speed = float(np.linalg.norm(env.vel))
        vr = compute_vr(env.pos, env.vel)
        action_norm = float(np.linalg.norm(action_exec))

        if action_norm > 1e-12:
            r_hat = env.pos / (radius + 1e-12)
            a_hat = action_exec / action_norm
            alignment = abs(float(np.dot(a_hat, r_hat)))
        else:
            alignment = 0.0

        radii.append(radius)
        speeds.append(speed)
        vrs.append(vr)
        action_norms.append(action_norm)
        alignments.append(alignment)

        within_tolerance_now = env._inside_tolerance(env.pos, env.vel)
        within_tolerance_any = within_tolerance_any or within_tolerance_now
        success = success or bool(info.get("success", False))

        if terminated or truncated:
            break

    radius_arr = np.asarray(radii, dtype=np.float64)
    speed_arr = np.asarray(speeds, dtype=np.float64)
    vr_arr = np.asarray(vrs, dtype=np.float64)
    align_arr = np.asarray(alignments, dtype=np.float64)
    tail_len = min(1000, len(radius_arr))
    tail_slice = slice(len(radius_arr) - tail_len, len(radius_arr))
    re = radius_arr - env.target_radius
    re_tail = re[tail_slice]
    vr_tail = vr_arr[-min(1000, len(vr_arr)) :] if len(vr_arr) else np.zeros(0, dtype=np.float64)
    speed_tail = speed_arr[tail_slice]
    align_tail = align_arr[-min(1000, len(align_arr)) :] if len(align_arr) else np.zeros(0, dtype=np.float64)
    crossings = int(np.sum(np.signbit(re[1:]) != np.signbit(re[:-1]))) if len(re) > 1 else 0

    first_half = re[: max(1, len(re) // 2)]
    second_half = re[len(re) // 2 :]
    amplitude_shrinks = bool(np.std(second_half) < np.std(first_half)) if len(re) >= 20 else False

    return SweepRow(
        controller=controller_name,
        r0_over_target=float(r0_over_target),
        max_steps=int(max_steps),
        success_at_start=bool(success_at_start),
        success=bool(success),
        terminated=bool(terminated),
        truncated=bool(truncated),
        radius_crossings_total=crossings,
        tail_crosses_target_radius=bool(np.any(np.signbit(re_tail[1:]) != np.signbit(re_tail[:-1]))) if len(re_tail) > 1 else False,
        final_radius_error=float(abs(re[-1])) if len(re) else 0.0,
        tail_mean_abs_vr=float(np.mean(np.abs(vr_tail))) if len(vr_tail) else 0.0,
        tail_radius_std=float(np.std(radius_arr[tail_slice])) if len(radius_arr) else 0.0,
        tail_mean_alignment=float(np.mean(align_tail)) if len(align_tail) else 0.0,
        tail_speed_cv=float(np.std(speed_tail) / (np.mean(speed_tail) + 1e-12)) if len(speed_tail) else 0.0,
        amplitude_shrinks=amplitude_shrinks,
        within_tolerance_any=bool(within_tolerance_any),
        min_abs_radius_error=float(np.min(np.abs(re))) if len(re) else 0.0,
        mean_action_norm=float(np.mean(action_norms)) if action_norms else 0.0,
    )


def load_existing_rows() -> List[SweepRow]:
    if not JSON_PATH.exists():
        return []
    try:
        raw = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        return [SweepRow(**row) for row in raw]
    except Exception:
        return []


def write_csv(rows: List[SweepRow]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_heatmap(rows: List[SweepRow], controller: str, metric: str, title: str, out_name: str, r0_values: List[float], max_steps_values: List[int]) -> None:
    data = np.full((len(r0_values), len(max_steps_values)), np.nan, dtype=np.float64)
    for i, r0 in enumerate(r0_values):
        for j, steps in enumerate(max_steps_values):
            match = next((r for r in rows if r.controller == controller and abs(r.r0_over_target - r0) < 1e-12 and r.max_steps == steps), None)
            if match is not None:
                data[i, j] = float(getattr(match, metric))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(max_steps_values)), [str(v) for v in max_steps_values])
    ax.set_yticks(np.arange(len(r0_values)), [f"{v:.4f}" for v in r0_values])
    ax.set_xlabel("max_steps")
    ax.set_ylabel("r0_over_target")
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = "NA" if np.isnan(data[i, j]) else f"{data[i, j]:.3g}"
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=180)
    plt.close(fig)


def parse_float_list(text: str | None, default: List[float]) -> List[float]:
    if not text:
        return list(default)
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str | None, default: List[int]) -> List[int]:
    if not text:
        return list(default)
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reachable insertion sweep with strict success settings.")
    parser.add_argument("--r0-values", type=str, default=None, help="Comma-separated r0_over_target values.")
    parser.add_argument("--max-steps-values", type=str, default=None, help="Comma-separated max_steps values.")
    args = parser.parse_args()

    r0_values = parse_float_list(args.r0_values, R0_VALUES)
    max_steps_values = parse_int_list(args.max_steps_values, MAX_STEPS_VALUES)

    ensure_dir(OUTPUT_DIR)
    env = make_strict_env(max_steps=4000)
    controllers = build_controllers(target_radius=env.target_radius, mu=env.mu)

    existing = load_existing_rows()
    row_map: Dict[tuple[str, float, int], SweepRow] = {
        (row.controller, row.r0_over_target, row.max_steps): row for row in existing
    }

    for r0 in r0_values:
        for steps in max_steps_values:
            for controller_name, controller in controllers.items():
                if controller_name == "explicit_orbit_lock":
                    controller = OrbitLockController(target_radius=env.target_radius, mu=env.mu, config=OrbitLockConfig())
                row = rollout_one(controller_name, controller, r0, steps)
                row_map[(row.controller, row.r0_over_target, row.max_steps)] = row
                print(json.dumps(asdict(row)))

    rows = sorted(row_map.values(), key=lambda r: (r.r0_over_target, r.max_steps, r.controller))
    write_csv(rows)
    JSON_PATH.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")

    all_r0_values = sorted({row.r0_over_target for row in rows})
    all_max_steps_values = sorted({row.max_steps for row in rows})
    for controller_name in controllers.keys():
        save_heatmap(rows, controller_name, "radius_crossings_total", f"{controller_name} | radius_crossings_total", f"{controller_name}_crossings.png", all_r0_values, all_max_steps_values)
        save_heatmap(rows, controller_name, "final_radius_error", f"{controller_name} | final_radius_error", f"{controller_name}_final_radius_error.png", all_r0_values, all_max_steps_values)

    strict_cfg = {
        "tol_r": 1.0e-3,
        "tol_v": 1.0e-3,
        "tol_ang": 0.02,
        "success_threshold": 200,
        "dt": env.dt,
        "thrust_scale": env.thrust_scale,
        "target_radius": env.target_radius,
    }

    crossed_rows = [r for r in rows if r.radius_crossings_total > 0]
    reachable_probe_rows = [r for r in rows if r.controller == "probe_max_retrograde"]
    explicit_rows = [r for r in rows if r.controller == "explicit_orbit_lock"]
    ppo_rows = [r for r in rows if r.controller == "ppo_speed_refine_50"]

    reachable_pairs = [(r.r0_over_target, r.max_steps) for r in reachable_probe_rows if r.radius_crossings_total > 0]
    explicit_success_pairs = [(r.r0_over_target, r.max_steps) for r in explicit_rows if r.radius_crossings_total > 0 or r.success]

    lines: List[str] = [
        "# Reachable Insertion Sweep Summary",
        "",
        "## Strict Evaluation Regime",
        "",
    ]
    for k, v in strict_cfg.items():
        lines.append(f"- `{k}` = `{v}`")
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- `{CSV_PATH.as_posix()}`",
        f"- `{JSON_PATH.as_posix()}`",
    ])
    for controller_name in controllers.keys():
        lines.append(f"- `{(OUTPUT_DIR / f'{controller_name}_crossings.png').as_posix()}`")
        lines.append(f"- `{(OUTPUT_DIR / f'{controller_name}_final_radius_error.png').as_posix()}`")

    lines.extend([
        "",
        "## Main Answers",
        "",
        "### Which r0 values are physically reachable under the current thrust and horizon?",
        "",
    ])
    if reachable_pairs:
        for r0, steps in reachable_pairs:
            lines.append(f"- Reachability probe crossed for `r0_over_target={r0:.4f}`, `max_steps={steps}`")
    else:
        lines.append("- No tested `(r0_over_target, max_steps)` pair produced a real radius crossing even for the max-retrograde reachability probe.")

    lines.extend([
        "",
        "### Does increasing horizon help once the task is reachable?",
        "",
    ])
    for r0 in all_r0_values:
        probe_sub = [r for r in reachable_probe_rows if abs(r.r0_over_target - r0) < 1e-12]
        best = min(probe_sub, key=lambda r: r.final_radius_error)
        lines.append(
            f"- `r0={r0:.4f}`: best probe horizon was `{best.max_steps}` with final_radius_error `{best.final_radius_error:.3e}` and crossings `{best.radius_crossings_total}`"
        )

    lines.extend([
        "",
        "### What is the smallest setup that produces real orbit crossing / orbit lock?",
        "",
    ])
    if crossed_rows:
        best_cross = min(crossed_rows, key=lambda r: (r.max_steps, r.r0_over_target, 0 if r.controller == 'explicit_orbit_lock' else 1))
        lines.append(
            f"- Smallest crossing setup: controller `{best_cross.controller}`, `r0_over_target={best_cross.r0_over_target:.4f}`, `max_steps={best_cross.max_steps}`, crossings `{best_cross.radius_crossings_total}`"
        )
    else:
        lines.append("- None of the tested setups produced real target-radius crossing.")

    lines.extend([
        "",
        "### What exact configuration should become the new project baseline?",
        "",
    ])
    if crossed_rows:
        best_baseline = min(
            crossed_rows,
            key=lambda r: (
                r.max_steps,
                r.r0_over_target,
                0 if r.controller == "explicit_orbit_lock" else 1,
                r.final_radius_error,
                r.tail_mean_abs_vr,
            ),
        )
    else:
        best_baseline = min(
            explicit_rows + ppo_rows + reachable_probe_rows,
            key=lambda r: (
                0 if r.controller == "explicit_orbit_lock" else 1,
                r.final_radius_error,
                r.tail_mean_abs_vr,
                r.max_steps,
                r.r0_over_target,
            ),
        )
    lines.append(
        f"- Recommended baseline: controller `{best_baseline.controller}`, `r0_over_target={best_baseline.r0_over_target:.4f}`, `max_steps={best_baseline.max_steps}`, strict tolerances as above."
    )
    lines.append(
        f"  Metrics: final_radius_error `{best_baseline.final_radius_error:.3e}`, tail_mean_abs_vr `{best_baseline.tail_mean_abs_vr:.3f}`, crossings `{best_baseline.radius_crossings_total}`, success `{best_baseline.success}`"
    )

    lines.extend([
        "",
        "## Diagnosis",
        "",
    ])
    start_success_rows = [r for r in rows if r.success_at_start]
    if start_success_rows:
        lines.append("- Even under the stricter regime, some start states are already inside tolerance; this is now tracked explicitly via `success_at_start`.")
    else:
        lines.append("- Under the stricter regime, no tested start state is already inside the success tolerance.")
    lines.append("- The sweep is intended to find the smallest reachable insertion regime before any retraining.")
    lines.append("- If the probe controller cannot cross, the task is still unreachable for learned controllers at that setup.")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved sweep outputs to: {OUTPUT_DIR}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
