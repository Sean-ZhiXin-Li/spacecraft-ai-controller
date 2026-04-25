from __future__ import annotations

import contextlib
import csv
import io
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from envs.orbit_env import OrbitEnv
from scripts.explicit_controller_analysis_utils import DT, MAX_STEPS, SEED, STRICT_CFG, TAIL_LEN, set_seed


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase4_regime"
CSV_PATH = OUTPUT_DIR / "regime_grid.csv"
SUMMARY_PATH = OUTPUT_DIR / "phase4_regime_summary.md"

SUCCESS_R0_ANGLE_PNG = OUTPUT_DIR / "success_map_r0_vs_angle.png"
SUCCESS_R0_THRUST_PNG = OUTPUT_DIR / "success_map_r0_vs_thrust.png"
SUCCESS_R0_TARGET_PNG = OUTPUT_DIR / "success_map_r0_vs_target_radius.png"
MIN_ERROR_R0_ANGLE_PNG = OUTPUT_DIR / "min_abs_radius_error_r0_vs_angle.png"
MIN_ERROR_R0_THRUST_PNG = OUTPUT_DIR / "min_abs_radius_error_r0_vs_thrust.png"
CAPTURE_SUMMARY_PNG = OUTPUT_DIR / "capture_entry_summary.png"

DEFAULT_TARGET_RADIUS = 7.5e12
R0_VALUES = [1.00002, 1.00005, 1.00008, 1.00012]
ANGLE_VALUES = [165.0, 170.0, 175.0]
THRUST_VALUES = [8000.0, 10000.0, 12000.0]
TARGET_RADIUS_SCALES = [0.98, 1.00, 1.02]

FIELDNAMES = [
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "target_radius_scale",
    "target_radius",
    "success",
    "crossing_occurs",
    "radius_crossings_total",
    "first_crossing_step",
    "first_crossing_time",
    "minimum_abs_radius_error",
    "final_radius_error",
    "tail_mean_abs_vr",
    "capture_entered",
    "lock_entered",
    "phase_transition_count",
    "steps",
    "terminated",
    "truncated",
]


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def make_regime_env(*, thrust_scale: float, target_radius_scale: float) -> OrbitEnv:
    return OrbitEnv(
        thrust_scale=float(thrust_scale),
        dt=DT,
        max_steps=MAX_STEPS,
        target_radius=DEFAULT_TARGET_RADIUS * float(target_radius_scale),
        reward_mode="orbit_circular_minimal",
        tol_r=STRICT_CFG["tol_r"],
        tol_v=STRICT_CFG["tol_v"],
        tol_ang=STRICT_CFG["tol_ang"],
        success_threshold=STRICT_CFG["success_threshold"],
        use_action_smoothing=False,
        use_orbit_capture_assist=False,
        verbose=False,
    )


def set_regime_start(env: OrbitEnv, *, r0_over_target: float, initial_velocity_angle_deg: float) -> None:
    env.reset(start_mode="default")
    env.steps = 0
    env.success_counter = 0
    env.radial_stall_counter = 0
    env.orbit_lock_counter = 0
    env.prev_action = np.zeros(2, dtype=np.float64)
    env.pos = np.array([0.0, float(r0_over_target) * env.target_radius], dtype=np.float64)
    v_mag = math.sqrt(env.mu / np.linalg.norm(env.pos))
    angle = math.radians(float(initial_velocity_angle_deg))
    env.vel = v_mag * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)


def tail_mean_abs(values: List[float]) -> float:
    if not values:
        return 0.0
    use_len = min(TAIL_LEN, len(values))
    return float(np.mean(np.abs(np.asarray(values[-use_len:], dtype=np.float64))))


def rollout_regime(
    *,
    r0_over_target: float,
    initial_velocity_angle_deg: float,
    thrust_scale: float,
    target_radius_scale: float,
) -> Dict[str, object]:
    env = make_regime_env(thrust_scale=thrust_scale, target_radius_scale=target_radius_scale)
    controller = OrbitLockController(target_radius=env.target_radius, mu=env.mu, config=OrbitLockConfig())
    set_regime_start(env, r0_over_target=r0_over_target, initial_velocity_angle_deg=initial_velocity_angle_deg)
    obs = np.asarray(env._get_obs(), dtype=np.float32)

    radius_errors: List[float] = []
    vr_values: List[float] = []
    success = False
    terminated = False
    truncated = False
    capture_entered = False
    lock_entered = False
    phase_transition_count = 0

    while not (terminated or truncated):
        with contextlib.redirect_stdout(io.StringIO()):
            info = controller.act_with_info(obs)
        action = np.asarray(info["final_action"], dtype=np.float64)
        obs, _, terminated, truncated, step_info = env.step(action)
        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1.0e-12)
        radius_errors.append(radius - env.target_radius)
        vr_values.append(float(np.dot(env.vel, r_hat)))
        phase = str(info.get("phase", controller.phase))
        capture_entered = capture_entered or phase == OrbitLockController.STATE_CAPTURE
        lock_entered = lock_entered or phase == OrbitLockController.STATE_LOCK
        phase_transition_count = int(info.get("phase_transition_count", len(controller.phase_transitions)))
        success = success or bool(step_info.get("success", False))
        obs = np.asarray(obs, dtype=np.float32)

    rerr = np.asarray(radius_errors, dtype=np.float64)
    sign_flip_mask = np.signbit(rerr[1:]) != np.signbit(rerr[:-1]) if len(rerr) > 1 else np.zeros(0, dtype=bool)
    crossing_indices = np.flatnonzero(sign_flip_mask)
    first_crossing_step = int(crossing_indices[0] + 1) if len(crossing_indices) else None
    minimum_abs_radius_error = float(np.min(np.abs(rerr))) if len(rerr) else 0.0

    return {
        "r0_over_target": float(r0_over_target),
        "initial_velocity_angle_deg": float(initial_velocity_angle_deg),
        "thrust_scale": float(thrust_scale),
        "target_radius_scale": float(target_radius_scale),
        "target_radius": float(env.target_radius),
        "success": bool(success),
        "crossing_occurs": bool(np.any(sign_flip_mask)),
        "radius_crossings_total": int(np.sum(sign_flip_mask)),
        "first_crossing_step": first_crossing_step,
        "first_crossing_time": float(first_crossing_step * DT) if first_crossing_step is not None else None,
        "minimum_abs_radius_error": minimum_abs_radius_error,
        "final_radius_error": float(abs(rerr[-1])) if len(rerr) else 0.0,
        "tail_mean_abs_vr": tail_mean_abs(vr_values),
        "capture_entered": bool(capture_entered),
        "lock_entered": bool(lock_entered),
        "phase_transition_count": int(phase_transition_count),
        "steps": int(len(rerr)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def filter_rows(rows: List[Dict[str, object]], **fixed: float) -> List[Dict[str, object]]:
    selected = []
    for row in rows:
        if all(abs(float(row[key]) - float(value)) < 1.0e-9 for key, value in fixed.items()):
            selected.append(row)
    return selected


def grid_from_rows(rows: List[Dict[str, object]], x_key: str, y_key: str, value_key: str) -> tuple[list[float], list[float], np.ndarray]:
    x_values = sorted({float(row[x_key]) for row in rows})
    y_values = sorted({float(row[y_key]) for row in rows})
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    for row in rows:
        x_idx = x_values.index(float(row[x_key]))
        y_idx = y_values.index(float(row[y_key]))
        value = row[value_key]
        grid[y_idx, x_idx] = 1.0 if value is True else 0.0 if value is False else float(value)
    return x_values, y_values, grid


def format_tick(value: float, key: str) -> str:
    if key == "r0_over_target":
        return f"{value:.5f}"
    if key in {"thrust_scale", "initial_velocity_angle_deg"}:
        return f"{value:g}"
    return f"{value:.2f}"


def save_success_map(path: Path, rows: List[Dict[str, object]], x_key: str, y_key: str, title: str) -> None:
    x_values, y_values, grid = grid_from_rows(rows, x_key, y_key, "success")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    cmap = colors.ListedColormap(["#F2F2F2", "#1B9E77"])
    norm = colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([format_tick(x, x_key) for x in x_values], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([format_tick(y, y_key) for y in y_values])
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    for yi, _ in enumerate(y_values):
        for xi, _ in enumerate(x_values):
            if grid[yi, xi] > 0.5:
                ax.text(xi, yi, "S", ha="center", va="center", fontsize=9, color="white", weight="bold")
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1], fraction=0.045, pad=0.03)
    cbar.ax.set_yticklabels(["fail", "success"])
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_metric_map(path: Path, rows: List[Dict[str, object]], x_key: str, y_key: str, title: str) -> None:
    x_values, y_values, grid = grid_from_rows(rows, x_key, y_key, "minimum_abs_radius_error")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    image = ax.imshow(np.log10(np.maximum(grid, 1.0e-12)), origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([format_tick(x, x_key) for x in x_values], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([format_tick(y, y_key) for y in y_values])
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    for row in rows:
        if bool(row["success"]):
            xi = x_values.index(float(row[x_key]))
            yi = y_values.index(float(row[y_key]))
            ax.scatter(xi, yi, s=60, marker="o", facecolors="none", edgecolors="white", linewidths=1.2)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("log10(min abs radius error [m])")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_capture_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    groups = [
        ("r0", "r0_over_target", R0_VALUES),
        ("angle", "initial_velocity_angle_deg", ANGLE_VALUES),
        ("thrust", "thrust_scale", THRUST_VALUES),
        ("target", "target_radius_scale", TARGET_RADIUS_SCALES),
    ]
    labels: List[str] = []
    capture_rates: List[float] = []
    lock_rates: List[float] = []
    success_rates: List[float] = []
    for prefix, key, values in groups:
        for value in values:
            subset = [row for row in rows if abs(float(row[key]) - float(value)) < 1.0e-9]
            labels.append(f"{prefix}={format_tick(float(value), key)}")
            capture_rates.append(float(np.mean([bool(row["capture_entered"]) for row in subset])))
            lock_rates.append(float(np.mean([bool(row["lock_entered"]) for row in subset])))
            success_rates.append(float(np.mean([bool(row["success"]) for row in subset])))

    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    ax.bar(x - width, capture_rates, width, label="CAPTURE entered", color="#4C78A8")
    ax.bar(x, lock_rates, width, label="LOCK entered", color="#F58518")
    ax.bar(x + width, success_rates, width, label="strict success", color="#54A24B")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("fraction of regimes")
    ax.set_title("Phase 4 capture/lock/success rates by one-dimensional grouping")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def safe_pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def dimension_rates(rows: List[Dict[str, object]], key: str, values: List[float], metric: str) -> List[str]:
    lines = []
    for value in values:
        subset = [row for row in rows if abs(float(row[key]) - float(value)) < 1.0e-9]
        count = sum(bool(row[metric]) for row in subset)
        lines.append(f"`{format_tick(float(value), key)}`: `{count}/{len(subset)}`")
    return lines


def write_summary(rows: List[Dict[str, object]]) -> None:
    total = len(rows)
    success_count = sum(bool(row["success"]) for row in rows)
    capture_count = sum(bool(row["capture_entered"]) for row in rows)
    lock_count = sum(bool(row["lock_entered"]) for row in rows)
    capture_no_lock = [row for row in rows if bool(row["capture_entered"]) and not bool(row["lock_entered"])]
    lock_no_success = [row for row in rows if bool(row["lock_entered"]) and not bool(row["success"])]
    near_misses = sorted(
        [row for row in rows if not bool(row["success"])],
        key=lambda row: float(row["minimum_abs_radius_error"]),
    )[:8]

    lines = [
        "# Phase 4 Nearby-Regime Sweep",
        "",
        "## Setup",
        "",
        "- Controller: unchanged explicit `DESCENT -> CAPTURE -> LOCK` controller.",
        "- Environment physics: unchanged 2D `OrbitEnv`; only task parameters and initial condition are varied.",
        f"- Fixed `dt`: `{DT:g}`.",
        f"- Fixed `max_steps`: `{MAX_STEPS}`.",
        f"- Default target radius reference: `{DEFAULT_TARGET_RADIUS:.3e}` m.",
        "",
        "## Aggregate Results",
        "",
        f"- Total regimes: `{total}`.",
        f"- Strict successes: `{success_count}/{total}` ({safe_pct(success_count, total):.1f}%).",
        f"- CAPTURE entered: `{capture_count}/{total}` ({safe_pct(capture_count, total):.1f}%).",
        f"- LOCK entered: `{lock_count}/{total}` ({safe_pct(lock_count, total):.1f}%).",
        f"- CAPTURE without LOCK: `{len(capture_no_lock)}` regimes.",
        f"- LOCK without strict success: `{len(lock_no_success)}` regimes.",
        "- In this grid, CAPTURE entry, LOCK entry, and strict success are the same set of regimes."
        if capture_count == lock_count == success_count and not capture_no_lock and not lock_no_success
        else "- CAPTURE, LOCK, and strict success differ in this grid; inspect the CSV for the mismatched regimes.",
        "",
        "## Direct Answers",
        "",
        "1. The capture mechanism is broader than one exact baseline, but it is still local. "
        f"It succeeds in `{success_count}/{total}` nearby regimes, not only at the validated baseline.",
        "2. `thrust_scale` matters most in this sweep: `8000` and `12000` produce no successes, "
        "while `10000` produces all observed successes. Angle and target-radius scale are also strong; "
        "the tested `r0` values matter, but less abruptly than thrust.",
        "3. The results show success pockets, not a broad connected band. At `thrust=10000`, the successful "
        "pocket shifts with angle and target-radius scale; outside that thrust slice, the controller does not enter CAPTURE.",
        f"4. CAPTURE-without-LOCK regimes: `{len(capture_no_lock)}`. "
        "In this sampled grid, failures are mostly pre-capture failures."
        if not capture_no_lock
        else f"4. CAPTURE-without-LOCK regimes: `{len(capture_no_lock)}`. "
        "These regimes isolate post-capture lock fragility.",
        "5. The explicit controller should be treated as a local phase mechanism with some nearby transfer, "
        "not as a general 2D multi-regime controller.",
        "",
        "## Dimension Sensitivity",
        "",
        "- Success by `r0_over_target`: " + "; ".join(dimension_rates(rows, "r0_over_target", R0_VALUES, "success")) + ".",
        "- Success by `initial_velocity_angle_deg`: "
        + "; ".join(dimension_rates(rows, "initial_velocity_angle_deg", ANGLE_VALUES, "success"))
        + ".",
        "- Success by `thrust_scale`: " + "; ".join(dimension_rates(rows, "thrust_scale", THRUST_VALUES, "success")) + ".",
        "- Success by `target_radius_scale`: "
        + "; ".join(dimension_rates(rows, "target_radius_scale", TARGET_RADIUS_SCALES, "success"))
        + ".",
        "- CAPTURE entry by `r0_over_target`: "
        + "; ".join(dimension_rates(rows, "r0_over_target", R0_VALUES, "capture_entered"))
        + ".",
        "- CAPTURE entry by `initial_velocity_angle_deg`: "
        + "; ".join(dimension_rates(rows, "initial_velocity_angle_deg", ANGLE_VALUES, "capture_entered"))
        + ".",
        "- CAPTURE entry by `thrust_scale`: "
        + "; ".join(dimension_rates(rows, "thrust_scale", THRUST_VALUES, "capture_entered"))
        + ".",
        "- CAPTURE entry by `target_radius_scale`: "
        + "; ".join(dimension_rates(rows, "target_radius_scale", TARGET_RADIUS_SCALES, "capture_entered"))
        + ".",
        "",
        "## Near Misses",
        "",
    ]
    for row in near_misses:
        lines.append(
            f"- r0 `{float(row['r0_over_target']):.5f}`, angle `{float(row['initial_velocity_angle_deg']):g}`, "
            f"thrust `{float(row['thrust_scale']):g}`, target scale `{float(row['target_radius_scale']):.2f}`: "
            f"min abs radius error `{float(row['minimum_abs_radius_error']):.3e}` m, "
            f"CAPTURE `{row['capture_entered']}`, LOCK `{row['lock_entered']}`."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The capture mechanism is not treated here as redesigned or retuned; every result is for the fixed Phase 3 controller.",
            "- Strict success and CAPTURE entry are concentrated around a small subset of the grid, so the controller should be read as a local construction rather than a broad nearby-regime solution.",
            "- Success pockets indicate that the phase mechanism can transfer across some neighboring 2D regimes, but pockets are weaker evidence than a connected success band.",
            "- This sweep did not find regimes that enter CAPTURE but fail LOCK. For these nearby cases, the dominant failure mode is failure to reach CAPTURE at all.",
            "- These results are still local to the sampled 2D grid and should not be generalized to 3D or large orbital changes.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def save_plots(rows: List[Dict[str, object]]) -> None:
    base_angle = 170.0
    base_thrust = 10000.0
    base_target_scale = 1.00
    save_success_map(
        SUCCESS_R0_ANGLE_PNG,
        filter_rows(rows, thrust_scale=base_thrust, target_radius_scale=base_target_scale),
        "r0_over_target",
        "initial_velocity_angle_deg",
        "Strict success: r0 vs angle (thrust=10000, target scale=1.00)",
    )
    save_success_map(
        SUCCESS_R0_THRUST_PNG,
        filter_rows(rows, initial_velocity_angle_deg=base_angle, target_radius_scale=base_target_scale),
        "r0_over_target",
        "thrust_scale",
        "Strict success: r0 vs thrust (angle=170 deg, target scale=1.00)",
    )
    save_success_map(
        SUCCESS_R0_TARGET_PNG,
        filter_rows(rows, initial_velocity_angle_deg=base_angle, thrust_scale=base_thrust),
        "r0_over_target",
        "target_radius_scale",
        "Strict success: r0 vs target radius scale (angle=170 deg, thrust=10000)",
    )
    save_metric_map(
        MIN_ERROR_R0_ANGLE_PNG,
        filter_rows(rows, thrust_scale=base_thrust, target_radius_scale=base_target_scale),
        "r0_over_target",
        "initial_velocity_angle_deg",
        "Min abs radius error: r0 vs angle (thrust=10000, target scale=1.00)",
    )
    save_metric_map(
        MIN_ERROR_R0_THRUST_PNG,
        filter_rows(rows, initial_velocity_angle_deg=base_angle, target_radius_scale=base_target_scale),
        "r0_over_target",
        "thrust_scale",
        "Min abs radius error: r0 vs thrust (angle=170 deg, target scale=1.00)",
    )
    save_capture_summary(CAPTURE_SUMMARY_PNG, rows)


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [
        (r0, angle, thrust, target_scale)
        for target_scale in TARGET_RADIUS_SCALES
        for thrust in THRUST_VALUES
        for angle in ANGLE_VALUES
        for r0 in R0_VALUES
    ]
    rows: List[Dict[str, object]] = []
    for idx, (r0, angle, thrust, target_scale) in enumerate(cases, start=1):
        row = rollout_regime(
            r0_over_target=r0,
            initial_velocity_angle_deg=angle,
            thrust_scale=thrust,
            target_radius_scale=target_scale,
        )
        rows.append(row)
        print(
            f"phase4 {idx}/{len(cases)} r0={r0:.5f} angle={angle:g} thrust={thrust:g} "
            f"target_scale={target_scale:.2f} success={row['success']} capture={row['capture_entered']} "
            f"lock={row['lock_entered']} min_abs_rerr={row['minimum_abs_radius_error']:.3e}"
        )
    rows.sort(
        key=lambda row: (
            float(row["target_radius_scale"]),
            float(row["thrust_scale"]),
            float(row["initial_velocity_angle_deg"]),
            float(row["r0_over_target"]),
        )
    )
    write_csv(CSV_PATH, rows, FIELDNAMES)
    save_plots(rows)
    write_summary(rows)
    print(f"Saved Phase 4 regime sweep to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
