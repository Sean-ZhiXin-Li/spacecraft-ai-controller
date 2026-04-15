from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from envs.orbit_env import OrbitEnv


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figs" / "orbit_lock_generalization"
CSV_PATH = OUTPUT_DIR / "aggregate_results.csv"
JSON_PATH = OUTPUT_DIR / "aggregate_results.json"
SUMMARY_PATH = PROJECT_ROOT / "analysis" / "orbit_lock_generalization.md"

R0_VALUES = [1.00002, 1.00005, 1.0001, 1.0002]
DT_VALUES = [20.0, 50.0, 100.0]
THRUST_VALUES = [10000.0, 20000.0, 40000.0]
MAX_STEPS = 100000
STRICT_CFG = {
    "tol_r": 1.0e-3,
    "tol_v": 1.0e-3,
    "tol_ang": 0.02,
    "success_threshold": 200,
}


@dataclass
class GeneralizationRow:
    r0_over_target: float
    dt: float
    thrust_scale: float
    success_at_start: bool
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
    mean_action_norm: float
    min_abs_radius_error: float
    sustained_crossing_score: int
    phase_transition_count: int


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


def tail_slice(length: int, tail_len: int = 5000) -> slice:
    use_len = min(tail_len, length)
    return slice(length - use_len, length)


def rollout_one(r0_over_target: float, dt: float, thrust_scale: float) -> GeneralizationRow:
    env = make_env(dt, thrust_scale)
    set_default_start(env, r0_over_target)
    controller = OrbitLockController(target_radius=env.target_radius, mu=env.mu, config=OrbitLockConfig())
    success_at_start = env._inside_tolerance(env.pos, env.vel)
    obs = env._get_obs()

    radii: List[float] = []
    speeds: List[float] = []
    vrs: List[float] = []
    action_norms: List[float] = []
    success = False
    terminated = False
    truncated = False

    for _ in range(MAX_STEPS):
        action = np.asarray(controller.act(obs), dtype=np.float64)
        obs, _, terminated, truncated, info = env.step(action)
        radius = float(np.linalg.norm(env.pos))
        speed = float(np.linalg.norm(env.vel))
        r_hat = env.pos / (radius + 1e-12)
        v_r = float(np.dot(env.vel, r_hat))
        radii.append(radius)
        speeds.append(speed)
        vrs.append(v_r)
        action_norms.append(float(np.linalg.norm(np.asarray(info["action_executed"], dtype=np.float64))))
        success = success or bool(info.get("success", False))
        if terminated or truncated:
            break

    radius_arr = np.asarray(radii, dtype=np.float64)
    speed_arr = np.asarray(speeds, dtype=np.float64)
    vr_arr = np.asarray(vrs, dtype=np.float64)
    action_arr = np.asarray(action_norms, dtype=np.float64)
    re = radius_arr - env.target_radius
    sign_flip_mask = np.signbit(re[1:]) != np.signbit(re[:-1]) if len(re) > 1 else np.zeros(0, dtype=bool)
    crossing_indices = np.flatnonzero(sign_flip_mask)
    first_crossing_step = int(crossing_indices[0] + 1) if len(crossing_indices) else None

    ts = tail_slice(len(radius_arr))
    re_tail = re[ts]
    vr_tail = vr_arr[ts]
    speed_tail = speed_arr[ts]
    tail_cross_mask = np.signbit(re_tail[1:]) != np.signbit(re_tail[:-1]) if len(re_tail) > 1 else np.zeros(0, dtype=bool)
    first_half = re[: max(1, len(re) // 2)]
    second_half = re[len(re) // 2 :]

    return GeneralizationRow(
        r0_over_target=float(r0_over_target),
        dt=float(dt),
        thrust_scale=float(thrust_scale),
        success_at_start=bool(success_at_start),
        success=bool(success),
        terminated=bool(terminated),
        truncated=bool(truncated),
        radius_crossings_total=int(np.sum(sign_flip_mask)),
        tail_crosses_target_radius=bool(np.any(tail_cross_mask)),
        first_crossing_step=first_crossing_step,
        final_radius_error=float(abs(re[-1])) if len(re) else 0.0,
        tail_mean_abs_vr=float(np.mean(np.abs(vr_tail))) if len(vr_tail) else 0.0,
        tail_radius_std=float(np.std(radius_arr[ts])) if len(radius_arr) else 0.0,
        tail_speed_cv=float(np.std(speed_tail) / (np.mean(speed_tail) + 1e-12)) if len(speed_tail) else 0.0,
        amplitude_shrinks=bool(np.std(second_half) < np.std(first_half)) if len(re) >= 20 else False,
        mean_action_norm=float(np.mean(action_arr)) if len(action_arr) else 0.0,
        min_abs_radius_error=float(np.min(np.abs(re))) if len(re) else 0.0,
        sustained_crossing_score=int(np.sum(tail_cross_mask)),
        phase_transition_count=int(len(controller.phase_transitions)),
    )


def write_csv(rows: List[GeneralizationRow]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_heatmap(rows: List[GeneralizationRow], metric: str, r0_value: float, out_name: str, title: str) -> None:
    subset = [row for row in rows if abs(row.r0_over_target - r0_value) < 1e-12]
    data = np.zeros((len(DT_VALUES), len(THRUST_VALUES)), dtype=np.float64)
    for i, dt in enumerate(DT_VALUES):
        for j, thrust in enumerate(THRUST_VALUES):
            match = next(row for row in subset if row.dt == dt and row.thrust_scale == thrust)
            data[i, j] = float(getattr(match, metric))

    fig, ax = plt.subplots(figsize=(7, 4.8))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(THRUST_VALUES)), [str(int(v)) for v in THRUST_VALUES])
    ax.set_yticks(np.arange(len(DT_VALUES)), [str(int(v)) for v in DT_VALUES])
    ax.set_xlabel("thrust_scale")
    ax.set_ylabel("dt")
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3g}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=180)
    plt.close(fig)


def write_summary(rows: List[GeneralizationRow]) -> None:
    crossing_rows = [row for row in rows if row.radius_crossings_total > 0]
    sustained_rows = [row for row in rows if row.tail_crosses_target_radius]
    successful_rows = [row for row in rows if row.success]
    best_row = min(rows, key=lambda row: (0 if row.success else 1, 0 if row.tail_crosses_target_radius else 1, row.final_radius_error, row.tail_mean_abs_vr))

    lines = [
        "# Orbit Lock Generalization",
        "",
        "## Sweep",
        "",
        f"- `r0_over_target`: `{', '.join(f'{v:.5f}' for v in R0_VALUES)}`",
        f"- `dt`: `{', '.join(str(int(v)) for v in DT_VALUES)}`",
        f"- `thrust_scale`: `{', '.join(str(int(v)) for v in THRUST_VALUES)}`",
        f"- `max_steps = {MAX_STEPS}`",
        "- Controller: `explicit_orbit_lock`",
        "",
        "## Main Answers",
        "",
    ]
    if crossing_rows:
        lines.append(f"- Crossing generalizes to `{len(crossing_rows)}` / `{len(rows)}` tested setups.")
    else:
        lines.append("- No tested setup produced any crossing.")
    if sustained_rows:
        lines.append(f"- Tail crossings persist in `{len(sustained_rows)}` setups.")
    else:
        lines.append("- No setup maintains tail crossings.")
    if successful_rows:
        lines.append(f"- Strict success is reached in `{len(successful_rows)}` setups.")
    else:
        lines.append("- No setup reaches strict success.")

    lines.extend([
        "",
        "## Best Observed Regime",
        "",
        f"- `r0_over_target = {best_row.r0_over_target:.5f}`",
        f"- `dt = {int(best_row.dt)}`",
        f"- `thrust_scale = {int(best_row.thrust_scale)}`",
        f"- crossings `{best_row.radius_crossings_total}`",
        f"- tail_crosses_target_radius `{best_row.tail_crosses_target_radius}`",
        f"- success `{best_row.success}`",
        f"- final_radius_error `{best_row.final_radius_error:.3e}`",
        f"- tail_mean_abs_vr `{best_row.tail_mean_abs_vr:.3f}`",
        "",
        "## Interpretation",
        "",
        "- Crossing sensitivity is now a joint function of `r0`, `dt`, and thrust rather than a single-setup artifact.",
        "- The next benchmark should use one successful setup, one crossing-without-success setup if present, and one failure setup from this table.",
    ])
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    rows: List[GeneralizationRow] = []
    for r0 in R0_VALUES:
        for dt in DT_VALUES:
            for thrust in THRUST_VALUES:
                row = rollout_one(r0, dt, thrust)
                rows.append(row)
                print(json.dumps(asdict(row)))

    rows.sort(key=lambda row: (row.r0_over_target, row.dt, row.thrust_scale))
    write_csv(rows)
    JSON_PATH.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    for r0 in R0_VALUES:
        tag = f"r0_{r0:.5f}".replace(".", "p")
        save_heatmap(rows, "radius_crossings_total", r0, f"{tag}_crossings.png", f"crossings | r0={r0:.5f}")
        save_heatmap(rows, "final_radius_error", r0, f"{tag}_final_radius_error.png", f"final radius error | r0={r0:.5f}")
        save_heatmap(rows, "tail_mean_abs_vr", r0, f"{tag}_tail_vr.png", f"tail mean abs vr | r0={r0:.5f}")
    write_summary(rows)
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
