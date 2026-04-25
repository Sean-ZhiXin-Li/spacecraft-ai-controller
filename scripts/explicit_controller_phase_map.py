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
    metrics_row,
    recommended_workers,
    rollout_metrics,
    save_heatmap,
    set_seed,
    write_csv,
)


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase_map"
CSV_PATH = OUTPUT_DIR / "phase_map.csv"
SUCCESS_HEATMAP_PATH = OUTPUT_DIR / "success_heatmap.png"
FINAL_ERROR_HEATMAP_PATH = OUTPUT_DIR / "final_radius_error_heatmap.png"
TAIL_VR_HEATMAP_PATH = OUTPUT_DIR / "tail_mean_abs_vr_heatmap.png"

R0_VALUES = [1.00001, 1.00002, 1.00003, 1.00005, 1.00007, 1.0001, 1.0002, 1.0005, 1.001, 1.002, 1.005, 1.01]
DT_VALUES = [50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0]

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
                f"phase_map r0={row['r0_over_target']:.5f} dt={row['dt']:g} "
                f"success={row['success']} crossings={row['radius_crossings_total']}"
            )
    rows.sort(key=lambda row: (float(row["dt"]), float(row["r0_over_target"])))
    write_csv(CSV_PATH, rows, FIELDNAMES)
    save_heatmap(SUCCESS_HEATMAP_PATH, rows, x_key="r0_over_target", y_key="dt", value_key="success", title="Explicit controller success phase map", cbar_label="success")
    save_heatmap(FINAL_ERROR_HEATMAP_PATH, rows, x_key="r0_over_target", y_key="dt", value_key="final_radius_error", title="Final radius error phase map", cbar_label="final radius error [m]", log_scale=True)
    save_heatmap(TAIL_VR_HEATMAP_PATH, rows, x_key="r0_over_target", y_key="dt", value_key="tail_mean_abs_vr", title="Tail mean abs v_r phase map", cbar_label="tail mean abs v_r [m/s]", log_scale=True)
    print(f"Saved phase map to {CSV_PATH}")


if __name__ == "__main__":
    main()
