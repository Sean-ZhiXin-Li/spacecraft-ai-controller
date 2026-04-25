from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.explicit_controller_analysis_utils import (
    SEED,
    bool_from_csv,
    metrics_row,
    read_csv_dicts,
    recommended_workers,
    rollout_metrics,
    save_heatmap,
    set_seed,
    write_csv,
)


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase_map"
PHASE_MAP_CSV = OUTPUT_DIR / "phase_map.csv"
CSV_PATH = OUTPUT_DIR / "boundary_refine.csv"
SUCCESS_HEATMAP_PATH = OUTPUT_DIR / "boundary_refine_success_heatmap.png"
FINAL_ERROR_HEATMAP_PATH = OUTPUT_DIR / "boundary_refine_final_error_heatmap.png"
SUMMARY_PATH = OUTPUT_DIR / "boundary_refine_summary.md"

R0_VALUES = [1.00002, 1.00003, 1.00004, 1.00005, 1.00006, 1.00007, 1.00008, 1.00009, 1.00010, 1.00011, 1.00012]
DT_VALUES = [80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0]

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


def summarize_phase_map() -> str:
    if not PHASE_MAP_CSV.exists():
        return "Phase-map CSV was not present before refinement, so the dense grid used the predefined frontier range."
    rows = read_csv_dicts(PHASE_MAP_CSV)
    successes = [row for row in rows if bool_from_csv(row["success"])]
    failures = [row for row in rows if not bool_from_csv(row["success"])]
    if not successes or not failures:
        return "Phase-map CSV did not contain both successes and failures."
    max_success_r0 = max(float(row["r0_over_target"]) for row in successes)
    min_failure_r0 = min(float(row["r0_over_target"]) for row in failures if float(row["r0_over_target"]) > max_success_r0) if any(float(row["r0_over_target"]) > max_success_r0 for row in failures) else None
    if min_failure_r0 is None:
        return f"Coarse phase map found successes up to r0_over_target={max_success_r0:.5f}; no higher-r0 failure bound was directly above it."
    return f"Coarse phase map places the r0 frontier between {max_success_r0:.5f} and {min_failure_r0:.5f} among tested points."


def write_summary(rows: List[Dict[str, object]]) -> None:
    successes = [row for row in rows if bool(row["success"])]
    failures = [row for row in rows if not bool(row["success"])]
    by_dt: Dict[float, str] = {}
    for dt in DT_VALUES:
        dt_rows = sorted([row for row in rows if float(row["dt"]) == dt], key=lambda row: float(row["r0_over_target"]))
        success_r0 = [float(row["r0_over_target"]) for row in dt_rows if bool(row["success"])]
        failure_after = [float(row["r0_over_target"]) for row in dt_rows if (not bool(row["success"])) and (not success_r0 or float(row["r0_over_target"]) > max(success_r0))]
        if success_r0 and failure_after:
            by_dt[dt] = f"success through {max(success_r0):.5f}; first failure after success at {min(failure_after):.5f}"
        elif success_r0:
            by_dt[dt] = f"success through {max(success_r0):.5f}; no failure after success in dense range"
        else:
            by_dt[dt] = "no successful dense-grid point"

    first_lost = "No success/failure frontier found in dense range."
    if successes and failures:
        sorted_rows = sorted(rows, key=lambda row: (float(row["dt"]), float(row["r0_over_target"])))
        for dt in DT_VALUES:
            dt_rows = [row for row in sorted_rows if float(row["dt"]) == dt]
            seen_success = False
            for row in dt_rows:
                if bool(row["success"]):
                    seen_success = True
                elif seen_success:
                    first_lost = f"At dt={dt:g}, success is first lost at r0_over_target={float(row['r0_over_target']):.5f}."
                    break
            if first_lost.startswith("At dt="):
                break

    best_success = min(successes, key=lambda row: float(row["final_radius_error"])) if successes else None
    lines = [
        "# Boundary Refinement Summary",
        "",
        "## Coarse Boundary",
        "",
        f"- {summarize_phase_map()}",
        "",
        "## Dense Boundary",
        "",
        f"- {first_lost}",
    ]
    if best_success is not None:
        lines.append(
            f"- Lowest final-radius-error success: dt `{float(best_success['dt']):g}`, "
            f"r0_over_target `{float(best_success['r0_over_target']):.5f}`, "
            f"final_radius_error `{float(best_success['final_radius_error']):.3e}`."
        )
    lines.extend(["", "## Per-dt Frontier", ""])
    for dt, text in by_dt.items():
        lines.append(f"- dt `{dt:g}`: {text}")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [(r0, dt) for dt in DT_VALUES for r0 in R0_VALUES]
    rows: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=recommended_workers()) as executor:
        future_map = {executor.submit(run_case, case): case for case in cases}
        for future in as_completed(future_map):
            row = future.result()
            rows.append(row)
            print(
                f"boundary_refine r0={row['r0_over_target']:.5f} dt={row['dt']:g} "
                f"success={row['success']} crossings={row['radius_crossings_total']}"
            )
    rows.sort(key=lambda row: (float(row["dt"]), float(row["r0_over_target"])))
    write_csv(CSV_PATH, rows, FIELDNAMES)
    save_heatmap(SUCCESS_HEATMAP_PATH, rows, x_key="r0_over_target", y_key="dt", value_key="success", title="Boundary refinement success map", cbar_label="success")
    save_heatmap(FINAL_ERROR_HEATMAP_PATH, rows, x_key="r0_over_target", y_key="dt", value_key="final_radius_error", title="Boundary refinement final radius error", cbar_label="final radius error [m]", log_scale=True)
    write_summary(rows)
    print(f"Saved boundary refinement to {CSV_PATH}")


if __name__ == "__main__":
    main()
