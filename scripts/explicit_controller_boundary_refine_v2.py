from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.explicit_controller_analysis_utils import (
    SEED,
    bool_from_csv,
    metrics_row,
    read_csv_dicts,
    rollout_metrics,
    set_seed,
    write_csv,
)


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase_map_v2"
PHASE_MAP_CSV = OUTPUT_DIR / "phase_map_v2.csv"
CSV_PATH = OUTPUT_DIR / "boundary_refine_v2.csv"
SUCCESS_HEATMAP_PATH = OUTPUT_DIR / "boundary_refine_success_heatmap_v2.png"
FINAL_ERROR_HEATMAP_PATH = OUTPUT_DIR / "boundary_refine_final_error_heatmap_v2.png"
SUMMARY_PATH = OUTPUT_DIR / "boundary_refine_summary_v2.md"

FIELDNAMES = [
    "r0_over_target",
    "dt",
    "success",
    "crossing_occurs",
    "radius_crossings_total",
    "first_crossing_step",
    "final_radius_error",
    "tail_mean_abs_vr",
    "phase_transition_count",
    "steps",
    "terminated",
    "truncated",
]


def detect_frontier_grid() -> tuple[list[float], list[float], str]:
    rows = read_csv_dicts(PHASE_MAP_CSV)
    dt_values = sorted({float(row["dt"]) for row in rows})
    transition_dts: set[float] = set()
    transition_r0s: set[float] = set()
    dt100_transition_r0s: set[float] = set()
    notes = []
    for dt in dt_values:
        dt_rows = sorted([row for row in rows if float(row["dt"]) == dt], key=lambda row: float(row["r0_over_target"]))
        states = [bool_from_csv(row["success"]) for row in dt_rows]
        if any(states) and not all(states):
            transition_dts.add(dt)
            for idx in range(1, len(dt_rows)):
                if states[idx - 1] != states[idx]:
                    transition_r0s.add(float(dt_rows[idx - 1]["r0_over_target"]))
                    transition_r0s.add(float(dt_rows[idx]["r0_over_target"]))
                    if abs(dt - 100.0) < 1.0e-9:
                        dt100_transition_r0s.add(float(dt_rows[idx - 1]["r0_over_target"]))
                        dt100_transition_r0s.add(float(dt_rows[idx]["r0_over_target"]))
            notes.append(f"dt={dt:g} contains a success/failure transition")
    if not transition_dts:
        transition_dts = {90.0, 100.0, 130.0, 140.0}
        transition_r0s = {1.00003, 1.00005, 1.00006, 1.00008}
        notes.append("No automatic frontier found; used conservative fallback around known v1 transition.")

    focused_dts = {100.0}
    for dt in transition_dts:
        if 85.0 <= dt <= 155.0:
            focused_dts.update({dt - 5.0, dt, dt + 5.0})
    dt_grid = sorted(dt for dt in focused_dts if 80.0 <= dt <= 160.0)

    focused_r0s = dt100_transition_r0s or transition_r0s or {1.00005, 1.00006}
    r0_min = max(1.00001, min(focused_r0s) - 0.00002)
    r0_max = min(1.00020, max(focused_r0s) + 0.00004)
    r0_grid = [round(float(x), 8) for x in np.arange(r0_min, r0_max + 0.000001, 0.000005)]
    return r0_grid, dt_grid, "; ".join(notes)


def run_case(case: tuple[float, float]) -> Dict[str, object]:
    r0, dt = case
    metrics = rollout_metrics(dt=dt, r0_over_target=r0, seed=SEED)
    return {"r0_over_target": r0, "dt": dt, **metrics_row(metrics)}


def _grid(rows: List[Dict[str, object]], value_key: str) -> tuple[list[float], list[float], np.ndarray]:
    x_values = sorted({float(row["r0_over_target"]) for row in rows})
    y_values = sorted({float(row["dt"]) for row in rows})
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    for row in rows:
        x_idx = x_values.index(float(row["r0_over_target"]))
        y_idx = y_values.index(float(row["dt"]))
        value = row[value_key]
        grid[y_idx, x_idx] = 1.0 if value is True else 0.0 if value is False else float(value)
    return x_values, y_values, grid


def _style_axes(ax: plt.Axes, x_values: list[float], y_values: list[float]) -> None:
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{x:.5f}" for x in x_values], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([f"{y:g}" for y in y_values], fontsize=8)
    ax.set_xlabel("initial radius / target radius")
    ax.set_ylabel("time step dt [s]")


def save_success_heatmap(rows: List[Dict[str, object]]) -> None:
    x_values, y_values, grid = _grid(rows, "success")
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    cmap = colors.ListedColormap(["#F2F2F2", "#1B9E77"])
    norm = colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    _style_axes(ax, x_values, y_values)
    ax.set_title("Adaptive boundary refinement v2: strict success")
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1], fraction=0.03, pad=0.02)
    cbar.ax.set_yticklabels(["fail", "success"])
    fig.tight_layout()
    fig.savefig(SUCCESS_HEATMAP_PATH, dpi=220)
    plt.close(fig)


def save_final_error_heatmap(rows: List[Dict[str, object]]) -> None:
    x_values, y_values, grid = _grid(rows, "final_radius_error")
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    image = ax.imshow(np.log10(np.maximum(grid, 1.0e-12)), origin="lower", aspect="auto", cmap="magma")
    _style_axes(ax, x_values, y_values)
    ax.set_title("Adaptive boundary refinement v2: final radius error")
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log10(final radius error [m])")
    for row in rows:
        if bool(row["success"]):
            ax.scatter(x_values.index(float(row["r0_over_target"])), y_values.index(float(row["dt"])), s=38, facecolors="none", edgecolors="white", linewidths=1.0)
    fig.tight_layout()
    fig.savefig(FINAL_ERROR_HEATMAP_PATH, dpi=220)
    plt.close(fig)


def summarize(rows: List[Dict[str, object]], detection_note: str) -> None:
    dt100 = sorted([row for row in rows if abs(float(row["dt"]) - 100.0) < 1.0e-9], key=lambda row: float(row["r0_over_target"]))
    first_lost_dt100 = "No dt=100 success-to-failure transition found in the v2 refinement grid."
    seen_success = False
    for row in dt100:
        if bool(row["success"]):
            seen_success = True
        elif seen_success:
            first_lost_dt100 = f"On the validated dt=100 line, success is first lost at r0_over_target={float(row['r0_over_target']):.5f}."
            break

    by_dt = {}
    non_monotonic_dts = []
    isolated_success_dts = []
    for dt in sorted({float(row["dt"]) for row in rows}):
        dt_rows = sorted([row for row in rows if float(row["dt"]) == dt], key=lambda row: float(row["r0_over_target"]))
        states = [bool(row["success"]) for row in dt_rows]
        transitions = sum(1 for idx in range(1, len(states)) if states[idx] != states[idx - 1])
        if transitions > 1:
            non_monotonic_dts.append(dt)
        for idx, state in enumerate(states):
            left = states[idx - 1] if idx > 0 else False
            right = states[idx + 1] if idx < len(states) - 1 else False
            if state and not left and not right:
                isolated_success_dts.append(dt)
                break
        success_r0 = [float(row["r0_over_target"]) for row in dt_rows if bool(row["success"])]
        by_dt[dt] = success_r0

    r0_monotonic_by_dt = not non_monotonic_dts
    success_counts_by_dt = [len(values) for _, values in sorted(by_dt.items())]
    timestep_non_monotonic = any(
        (success_counts_by_dt[idx] == 0 and success_counts_by_dt[idx - 1] > 0 and success_counts_by_dt[idx + 1] > 0)
        or (success_counts_by_dt[idx] > 0 and success_counts_by_dt[idx - 1] == 0 and success_counts_by_dt[idx + 1] == 0)
        for idx in range(1, len(success_counts_by_dt) - 1)
    )
    pocket_text = (
        f"Isolated or re-entrant success pockets are present at dt values: {', '.join(f'{dt:g}' for dt in sorted(set(isolated_success_dts)))}."
        if isolated_success_dts
        else "No isolated single-cell success pockets were detected in the v2 refinement grid."
    )

    best = min([row for row in rows if bool(row["success"])], key=lambda row: float(row["final_radius_error"]), default=None)
    lines = [
        "# Boundary Refinement v2 Summary",
        "",
        "## Detection",
        "",
        f"- Automatic frontier detection from `phase_map_v2.csv`: {detection_note}",
        "",
        "## Validated dt=100 Boundary",
        "",
        f"- {first_lost_dt100}",
        "",
        "## Boundary Shape",
        "",
        f"- Is the r0 boundary monotonic within each tested dt row? `{'Yes' if r0_monotonic_by_dt else 'No'}`",
        f"- Is success monotonic across dt? `{'No' if timestep_non_monotonic else 'Yes'}`",
        f"- {pocket_text}",
    ]
    if non_monotonic_dts:
        lines.append(f"- Multiple success/failure transitions were detected at dt values: {', '.join(f'{dt:g}' for dt in non_monotonic_dts)}.")
    if best is not None:
        lines.append(
            f"- Best successful refined point by final radius error: dt `{float(best['dt']):g}`, "
            f"r0_over_target `{float(best['r0_over_target']):.5f}`, final_radius_error `{float(best['final_radius_error']):.3e}`."
        )
    lines.extend(["", "## Per-dt Successful r0 Values", ""])
    for dt, success_r0 in by_dt.items():
        if success_r0:
            lines.append(f"- dt `{dt:g}`: {', '.join(f'{x:.6f}' for x in success_r0)}")
        else:
            lines.append(f"- dt `{dt:g}`: none")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PHASE_MAP_CSV.exists():
        raise FileNotFoundError(f"Run explicit_controller_phase_map_v2.py first: {PHASE_MAP_CSV}")
    r0_values, dt_values, detection_note = detect_frontier_grid()
    cases = [(r0, dt) for dt in dt_values for r0 in r0_values]
    rows: List[Dict[str, object]] = []
    for idx, case in enumerate(cases, start=1):
        row = run_case(case)
        rows.append(row)
        print(
            f"boundary_v2 {idx}/{len(cases)} r0={row['r0_over_target']:.5f} "
            f"dt={row['dt']:g} success={row['success']}"
        )
    rows.sort(key=lambda row: (float(row["dt"]), float(row["r0_over_target"])))
    write_csv(CSV_PATH, rows, FIELDNAMES)
    save_success_heatmap(rows)
    save_final_error_heatmap(rows)
    summarize(rows, detection_note)
    print(f"Saved v2 boundary refinement to {CSV_PATH}")


if __name__ == "__main__":
    main()
