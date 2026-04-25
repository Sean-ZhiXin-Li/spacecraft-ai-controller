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
    metrics_row,
    rollout_metrics,
    set_seed,
    write_csv,
)


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase_map_v2"
CSV_PATH = OUTPUT_DIR / "phase_map_v2.csv"
SUCCESS_HEATMAP_PATH = OUTPUT_DIR / "success_heatmap_v2.png"
FINAL_ERROR_HEATMAP_PATH = OUTPUT_DIR / "final_radius_error_heatmap_v2.png"
TAIL_VR_HEATMAP_PATH = OUTPUT_DIR / "tail_mean_abs_vr_heatmap_v2.png"

R0_VALUES = [
    1.00001,
    1.00002,
    1.00003,
    1.00004,
    1.00005,
    1.00006,
    1.00007,
    1.00008,
    1.00009,
    1.00010,
    1.00011,
    1.00012,
    1.0002,
    1.0005,
    1.001,
]
DT_VALUES = [50.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 200.0, 300.0, 500.0]

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
    ax.set_xticklabels([f"{x:.5f}" for x in x_values], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([f"{y:g}" for y in y_values], fontsize=9)
    ax.set_xlabel("initial radius / target radius")
    ax.set_ylabel("time step dt [s]")
    ax.grid(False)


def save_success_heatmap(rows: List[Dict[str, object]]) -> None:
    x_values, y_values, grid = _grid(rows, "success")
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    cmap = colors.ListedColormap(["#F2F2F2", "#1B9E77"])
    norm = colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    _style_axes(ax, x_values, y_values)
    ax.set_title("Explicit controller v2 phase map: strict success")
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1], fraction=0.035, pad=0.02)
    cbar.ax.set_yticklabels(["fail", "success"])
    for yi, _ in enumerate(y_values):
        for xi, _ in enumerate(x_values):
            ax.text(xi, yi, "S" if grid[yi, xi] > 0.5 else "", ha="center", va="center", fontsize=7, color="white")
    fig.tight_layout()
    fig.savefig(SUCCESS_HEATMAP_PATH, dpi=220)
    plt.close(fig)


def save_metric_heatmap(path: Path, rows: List[Dict[str, object]], value_key: str, title: str, cbar_label: str) -> None:
    x_values, y_values, grid = _grid(rows, value_key)
    plot_grid = np.log10(np.maximum(grid, 1.0e-12))
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    image = ax.imshow(plot_grid, origin="lower", aspect="auto", cmap="magma")
    _style_axes(ax, x_values, y_values)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(f"log10({cbar_label})")
    success_lookup = {(float(row["r0_over_target"]), float(row["dt"])): bool(row["success"]) for row in rows}
    for yi, dt in enumerate(y_values):
        for xi, r0 in enumerate(x_values):
            if success_lookup.get((r0, dt), False):
                ax.scatter(xi, yi, s=48, marker="o", facecolors="none", edgecolors="white", linewidths=1.1)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [(r0, dt) for dt in DT_VALUES for r0 in R0_VALUES]
    rows: List[Dict[str, object]] = []
    for idx, case in enumerate(cases, start=1):
        row = run_case(case)
        rows.append(row)
        print(
            f"phase_map_v2 {idx}/{len(cases)} r0={row['r0_over_target']:.5f} "
            f"dt={row['dt']:g} success={row['success']} crossings={row['radius_crossings_total']}"
        )
    rows.sort(key=lambda row: (float(row["dt"]), float(row["r0_over_target"])))
    write_csv(CSV_PATH, rows, FIELDNAMES)
    save_success_heatmap(rows)
    save_metric_heatmap(
        FINAL_ERROR_HEATMAP_PATH,
        rows,
        "final_radius_error",
        "Explicit controller v2 phase map: final radius error",
        "final radius error [m]",
    )
    save_metric_heatmap(
        TAIL_VR_HEATMAP_PATH,
        rows,
        "tail_mean_abs_vr",
        "Explicit controller v2 phase map: tail radial velocity",
        "tail mean abs v_r [m/s]",
    )
    print(f"Saved v2 phase map to {CSV_PATH}")


if __name__ == "__main__":
    main()
