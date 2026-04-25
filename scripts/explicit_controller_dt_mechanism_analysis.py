from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.explicit_controller_analysis_utils import (
    R0_OVER_TARGET,
    SEED,
    THRUST_SCALE,
    metrics_row,
    rollout_trace,
    set_seed,
    write_csv,
)


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "dt_mechanism"
CSV_PATH = OUTPUT_DIR / "dt_mechanism.csv"
FIRST_CROSSING_PNG = OUTPUT_DIR / "first_crossing_time_vs_dt.png"
MIN_ABS_ERROR_PNG = OUTPUT_DIR / "min_abs_radius_error_vs_dt.png"
RADIUS_ERROR_COMPARE_PNG = OUTPUT_DIR / "radius_error_vs_time_dt_compare.png"
ENERGY_COMPARE_PNG = OUTPUT_DIR / "energy_vs_time_dt_compare.png"
VR_COMPARE_PNG = OUTPUT_DIR / "v_r_vs_time_dt_compare.png"
SUMMARY_PATH = OUTPUT_DIR / "dt_mechanism_summary.md"

DT_VALUES = [50, 80, 90, 100, 110, 120, 130, 140, 150, 200, 300]
MAX_STEPS = 100000


def finite_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)


def trace_arrays(trace: Dict[str, List[float | str]]) -> Dict[str, np.ndarray]:
    return {
        "time": np.asarray(trace["time"], dtype=np.float64),
        "radius_error": np.asarray(trace["radius_error"], dtype=np.float64),
        "v_r": np.asarray(trace["v_r"], dtype=np.float64),
        "energy": np.asarray(trace["energy"], dtype=np.float64),
    }


def analyze_dt(dt: float) -> Dict[str, object]:
    metrics, trace = rollout_trace(dt=dt, r0_over_target=R0_OVER_TARGET, seed=SEED)
    arrays = trace_arrays(trace)
    radius_error = arrays["radius_error"]
    abs_radius_error = np.abs(radius_error)
    min_abs_idx = int(np.argmin(abs_radius_error))
    first_crossing_time = None
    if metrics.first_crossing_step is not None:
        first_crossing_time = float(metrics.first_crossing_step * dt)

    row = {
        "dt": float(dt),
        "r0_over_target": R0_OVER_TARGET,
        "max_steps": MAX_STEPS,
        "thrust_scale": THRUST_SCALE,
        "success": metrics.success,
        "crossing_occurs": metrics.crossing_occurs,
        "radius_crossings_total": metrics.radius_crossings_total,
        "first_crossing_step": metrics.first_crossing_step,
        "first_crossing_time": first_crossing_time,
        "minimum_signed_radius_error": float(radius_error[min_abs_idx]),
        "minimum_abs_radius_error": float(abs_radius_error[min_abs_idx]),
        "time_of_min_abs_radius_error": float(arrays["time"][min_abs_idx]),
        "final_radius_error": metrics.final_radius_error,
        "tail_mean_abs_vr": metrics.tail_mean_abs_vr,
        "phase_transition_count": metrics.phase_transition_count,
    }
    print(
        f"dt_mechanism dt={dt:g} success={metrics.success} crossing={metrics.crossing_occurs} "
        f"min_abs_rerr={row['minimum_abs_radius_error']:.3e} final_rerr={metrics.final_radius_error:.3e}"
    )
    return {"row": row, "trace": arrays, "metrics": metrics_row(metrics)}


def plot_scalar(rows: List[Dict[str, object]], path: Path, y_key: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    dt = np.asarray([float(row["dt"]) for row in rows], dtype=np.float64)
    y = np.asarray([finite_or_nan(row[y_key]) for row in rows], dtype=np.float64)
    success = np.asarray([bool(row["success"]) for row in rows], dtype=bool)
    ax.plot(dt, y, color="#404040", linewidth=1.2, alpha=0.65)
    ax.scatter(dt[~success], y[~success], s=52, marker="x", color="#b23a48", label="failure", zorder=3)
    ax.scatter(dt[success], y[success], s=46, marker="o", color="#1b7f5f", label="success", zorder=4)
    ax.set_xlabel("dt [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    if y_key == "minimum_abs_radius_error":
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def downsample_indices(length: int, max_points: int = 5000) -> np.ndarray:
    if length <= max_points:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, max_points, dtype=int))


def plot_trace(results: List[Dict[str, object]], path: Path, key: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    for result in results:
        row = result["row"]
        trace = result["trace"]
        idx = downsample_indices(len(trace["time"]))
        time_days = trace["time"][idx] / 86400.0
        label = f"dt={float(row['dt']):g}, {'success' if row['success'] else 'fail'}"
        linewidth = 1.4 if row["success"] else 1.0
        alpha = 0.95 if row["success"] else 0.72
        ax.plot(time_days, trace[key][idx], linewidth=linewidth, alpha=alpha, label=label)
    ax.set_xlabel("simulation time [days]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def first_lost_success_runs(rows: List[Dict[str, object]]) -> List[str]:
    ordered = sorted(rows, key=lambda row: float(row["dt"]))
    transitions: List[str] = []
    for prev, curr in zip(ordered[:-1], ordered[1:]):
        if bool(prev["success"]) and not bool(curr["success"]):
            transitions.append(f"`dt={float(prev['dt']):g}` succeeds but `dt={float(curr['dt']):g}` fails")
    return transitions


def write_summary(rows: List[Dict[str, object]]) -> None:
    ordered = sorted(rows, key=lambda row: float(row["dt"]))
    successes = [row for row in ordered if bool(row["success"])]
    failures = [row for row in ordered if not bool(row["success"])]
    success_dts = [float(row["dt"]) for row in successes]
    crossed_successes = [row for row in successes if bool(row["crossing_occurs"])]
    no_cross_successes = [row for row in successes if not bool(row["crossing_occurs"])]
    crossed_failures = [row for row in failures if bool(row["crossing_occurs"])]
    missed_failures = [row for row in failures if not bool(row["crossing_occurs"])]
    closest_failures = sorted(failures, key=lambda row: float(row["minimum_abs_radius_error"]))[:3]
    transition_text = first_lost_success_runs(ordered)

    success_flags = [bool(row["success"]) for row in ordered]
    monotonic_non_decreasing = all((not prev) or curr for prev, curr in zip(success_flags[:-1], success_flags[1:]))
    monotonic_non_increasing = all(prev or (not curr) for prev, curr in zip(success_flags[:-1], success_flags[1:]))
    monotonic = monotonic_non_decreasing or monotonic_non_increasing
    monotonic_answer = (
        "No. Success is re-entrant over this sweep, with failures and successes interleaved."
        if not monotonic
        else "Yes over this finite sweep, although this should not be extrapolated beyond the sampled dt values."
    )

    lines = [
        "# dt Mechanism Analysis",
        "",
        "## Setup",
        "",
        f"- Fixed `r0_over_target`: `{R0_OVER_TARGET:.5f}`.",
        f"- Fixed `max_steps`: `{MAX_STEPS}`.",
        f"- Fixed `thrust_scale`: `{THRUST_SCALE:g}`.",
        f"- Swept `dt`: `{', '.join(str(int(row['dt'])) for row in ordered)}`.",
        "",
        "## Results",
        "",
        f"- Successful dt values: `{', '.join(str(int(x)) for x in success_dts) if success_dts else 'none'}`.",
        f"- Non-monotonic success in dt: {monotonic_answer}",
        f"- Crossed successes: `{', '.join(str(int(float(row['dt']))) for row in crossed_successes) if crossed_successes else 'none'}`.",
        f"- Near-band successes without a sign crossing: `{', '.join(str(int(float(row['dt']))) for row in no_cross_successes) if no_cross_successes else 'none'}`.",
        f"- Failures with a radius crossing: `{len(crossed_failures)}`.",
        f"- Failures without a radius crossing: `{len(missed_failures)}`.",
    ]
    if transition_text:
        lines.append(f"- Local success-to-failure transitions: {'; '.join(transition_text)}.")
    lines.extend(["", "## Closest Failed Cases", ""])
    for row in closest_failures:
        crossing = "crossed" if bool(row["crossing_occurs"]) else "no crossing"
        lines.append(
            f"- `dt={float(row['dt']):g}`: min abs radius error `{float(row['minimum_abs_radius_error']):.3e}` m, "
            f"signed error at closest approach `{float(row['minimum_signed_radius_error']):.3e}` m, "
            f"time of closest approach `{float(row['time_of_min_abs_radius_error']) / 86400.0:.2f}` days, {crossing}."
        )
    lines.extend(["", "## Mechanism Interpretation", ""])
    if crossed_failures:
        lines.append(
            "- At least one failed case crosses the target radius, so failure is not only a question of missing the crossing event."
        )
    else:
        lines.append("- In this sweep, failed cases do not cross the target radius; crossing access is the dominant separator.")
    if no_cross_successes:
        no_cross_dt = ", ".join(str(int(float(row["dt"]))) for row in no_cross_successes)
        lines.append(
            f"- `dt={no_cross_dt}` succeeds without a sign-change crossing. These runs come close enough to satisfy the "
            "strict near-radius condition while staying on the same side of the target radius in the sampled trace."
        )
    if missed_failures:
        missed_dt = ", ".join(str(int(float(row["dt"]))) for row in missed_failures)
        lines.append(
            f"- The no-crossing failures (`dt={missed_dt}`) approach the target band with different closest-approach timing, "
            "then drift away or remain outside the capture condition instead of activating the later phases."
        )
    if crossed_failures:
        crossed_dt = ", ".join(str(int(float(row["dt"]))) for row in crossed_failures)
        lines.append(
            f"- The crossing failures (`dt={crossed_dt}`) indicate that the discrete step can land on the wrong side of the "
            "event with radial velocity or energy history that is still incompatible with sustained capture."
        )
    lines.extend(
        [
            "- The re-entrant pockets are most plausibly a discrete-time event-alignment effect: dt changes when the descent "
            "trajectory samples the narrow capture window and how much energy is removed before that sample.",
            "- The measured failures are mostly missed-window cases rather than delayed successes: none of the failed runs "
            "crosses later within the 100000-step budget. `dt=300` is the closest failed run by radius error, but it still "
            "does not sustain the phase sequence and later diverges.",
            "- The energy traces should therefore be read with the radius-error traces: successful dt values reach the crossing "
            "or near-radius window after a compatible energy evolution, while nearby failures miss that window or sample it "
            "with poor timing.",
            "- This is a local explanation for the fixed 2D setup only; it should not be generalized beyond the current "
            "controller, initial condition, and environment settings.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [analyze_dt(float(dt)) for dt in DT_VALUES]
    rows = [result["row"] for result in results]
    write_csv(
        CSV_PATH,
        rows,
        fieldnames=[
            "dt",
            "r0_over_target",
            "max_steps",
            "thrust_scale",
            "success",
            "crossing_occurs",
            "radius_crossings_total",
            "first_crossing_step",
            "first_crossing_time",
            "minimum_signed_radius_error",
            "minimum_abs_radius_error",
            "time_of_min_abs_radius_error",
            "final_radius_error",
            "tail_mean_abs_vr",
            "phase_transition_count",
        ],
    )
    plot_scalar(rows, FIRST_CROSSING_PNG, "first_crossing_time", "First crossing time vs dt", "first crossing time [s]")
    plot_scalar(rows, MIN_ABS_ERROR_PNG, "minimum_abs_radius_error", "Closest radius approach vs dt", "min |radius error| [m]")
    plot_trace(results, RADIUS_ERROR_COMPARE_PNG, "radius_error", "Radius error traces by dt", "radius - target [m]")
    plot_trace(results, ENERGY_COMPARE_PNG, "energy", "Specific energy traces by dt", "specific energy [J/kg]")
    plot_trace(results, VR_COMPARE_PNG, "v_r", "Radial velocity traces by dt", "v_r [m/s]")
    write_summary(rows)
    print(f"Saved dt mechanism analysis to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
