from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.explicit_controller_analysis_utils import SEED, metrics_row, phase_durations, rollout_trace, set_seed


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "mechanism_compare_v2"
RADIUS_PNG = OUTPUT_DIR / "radius_v2.png"
RADIUS_ERROR_PNG = OUTPUT_DIR / "radius_error_v2.png"
VR_PNG = OUTPUT_DIR / "v_r_v2.png"
ACTION_NORM_PNG = OUTPUT_DIR / "action_norm_v2.png"
ENERGY_PNG = OUTPUT_DIR / "energy_v2.png"
ANGULAR_MOMENTUM_PNG = OUTPUT_DIR / "angular_momentum_v2.png"
PHASE_TABLE_PATH = OUTPUT_DIR / "phase_duration_table_v2.json"
SUMMARY_PATH = OUTPUT_DIR / "mechanism_compare_summary_v2.md"


CASES = [
    {
        "case": "validated_success",
        "label": "Validated success: r0=1.00005, dt=100",
        "r0_over_target": 1.00005,
        "dt": 100.0,
        "seed": SEED,
    },
    {
        "case": "nearest_dt100_boundary_failure",
        "label": "Nearest dt=100 failure: r0=1.00006, dt=100",
        "r0_over_target": 1.00006,
        "dt": 100.0,
        "seed": SEED,
    },
    {
        "case": "dt_induced_failure",
        "label": "dt-induced failure: r0=1.00005, dt=200",
        "r0_over_target": 1.00005,
        "dt": 200.0,
        "seed": SEED,
    },
]


def run_cases() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for spec in CASES:
        metrics, trace = rollout_trace(
            dt=float(spec["dt"]),
            r0_over_target=float(spec["r0_over_target"]),
            seed=int(spec["seed"]),
        )
        durations = phase_durations(trace)
        table_row = {
            "case": spec["case"],
            "label": spec["label"],
            "r0_over_target": spec["r0_over_target"],
            "dt": spec["dt"],
            "metrics": metrics_row(metrics),
            "phase_durations": durations,
            "capture_entered": durations.get("CAPTURE", 0) > 0,
            "lock_entered": durations.get("LOCK", 0) > 0,
        }
        rows.append({"spec": spec, "metrics": metrics_row(metrics), "trace": trace, "table_row": table_row})
        print(
            f"mechanism_v2 {spec['case']} success={metrics.success} "
            f"crossings={metrics.radius_crossings_total} capture={table_row['capture_entered']} lock={table_row['lock_entered']}"
        )
    return rows


def save_plot(path: Path, rows: List[Dict[str, object]], key: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    for row in rows:
        trace = row["trace"]
        time_days = [float(x) / 86400.0 for x in trace["time"]]
        ax.plot(time_days, trace[key], linewidth=1.2, label=row["spec"]["label"])
    ax.set_xlabel("simulation time [days]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_phase_table(rows: List[Dict[str, object]]) -> None:
    table = [row["table_row"] for row in rows]
    PHASE_TABLE_PATH.write_text(json.dumps(table, indent=2), encoding="utf-8")


def write_summary(rows: List[Dict[str, object]]) -> None:
    table = [row["table_row"] for row in rows]
    success = next(row for row in table if row["case"] == "validated_success")
    boundary_failure = next(row for row in table if row["case"] == "nearest_dt100_boundary_failure")
    dt_failure = next(row for row in table if row["case"] == "dt_induced_failure")
    lines = [
        "# Mechanism Compare v2",
        "",
        "## Cases",
        "",
    ]
    for row in table:
        metrics = row["metrics"]
        lines.append(
            f"- `{row['case']}`: r0 `{float(row['r0_over_target']):.5f}`, dt `{float(row['dt']):g}`, "
            f"success `{metrics['success']}`, first_crossing_step `{metrics['first_crossing_step']}`, "
            f"crossings `{metrics['radius_crossings_total']}`, CAPTURE entered `{row['capture_entered']}`, "
            f"LOCK entered `{row['lock_entered']}`, final_radius_error `{float(metrics['final_radius_error']):.3e}`."
        )
    lines.extend(
        [
            "",
            "## Technical Interpretation",
            "",
            f"- The validated success spends `{success['phase_durations']['DESCENT']}` steps in DESCENT, "
            f"`{success['phase_durations']['CAPTURE']}` in CAPTURE, and `{success['phase_durations']['LOCK']}` in LOCK.",
            "- Both failure cases spend the full budget in DESCENT and never enter CAPTURE or LOCK.",
            f"- The nearest dt=100 failure at r0 `{float(boundary_failure['r0_over_target']):.5f}` misses the crossing-triggered phase transition even though it differs from the validated start by only `1e-5` in r0_over_target.",
            f"- The dt-induced failure at dt `{float(dt_failure['dt']):g}` shows that the mechanism is coupled to numerical step size: changing dt can prevent the controller from reaching the crossing/capture event at the same initial radius.",
            "- The key mechanism is therefore event access, not lock tuning: if DESCENT does not reach the crossing event, CAPTURE and LOCK never become active.",
            "",
            "## Plot Notes",
            "",
            "- Time is normalized to simulation days on every plot.",
            "- Radius error and radial velocity expose whether descent is approaching the target band.",
            "- Energy and angular momentum show the long retrograde-removal phase versus failure trajectories that do not trigger capture.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_cases()
    save_plot(RADIUS_PNG, rows, "radius", "Mechanism compare v2: radius", "radius [m]")
    save_plot(RADIUS_ERROR_PNG, rows, "radius_error", "Mechanism compare v2: radius error", "radius - target [m]")
    save_plot(VR_PNG, rows, "v_r", "Mechanism compare v2: radial velocity", "v_r [m/s]")
    save_plot(ACTION_NORM_PNG, rows, "action_norm", "Mechanism compare v2: action norm", "||action||")
    save_plot(ENERGY_PNG, rows, "energy", "Mechanism compare v2: specific orbital energy", "specific energy [J/kg]")
    save_plot(
        ANGULAR_MOMENTUM_PNG,
        rows,
        "angular_momentum",
        "Mechanism compare v2: angular momentum",
        "specific angular momentum [m^2/s]",
    )
    write_phase_table(rows)
    write_summary(rows)
    print(f"Saved mechanism compare v2 to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
