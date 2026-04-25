from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.explicit_controller_analysis_utils import (
    SEED,
    metrics_row,
    phase_durations,
    rollout_trace,
    set_seed,
)


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "mechanism_compare"
RADIUS_PNG = OUTPUT_DIR / "success_vs_failure_radius.png"
VR_PNG = OUTPUT_DIR / "success_vs_failure_vr.png"
ENERGY_PNG = OUTPUT_DIR / "success_vs_failure_energy.png"
ACTION_NORM_PNG = OUTPUT_DIR / "success_vs_failure_action_norm.png"
SUMMARY_PATH = OUTPUT_DIR / "mechanism_compare_summary.md"
PHASE_TABLE_PATH = OUTPUT_DIR / "phase_duration_table.json"


def run_cases() -> List[Dict[str, object]]:
    case_specs = [
        {"case": "validated_success", "r0_over_target": 1.00005, "dt": 100.0, "seed": SEED},
        {"case": "near_boundary_failure", "r0_over_target": 1.00006, "dt": 100.0, "seed": SEED},
        {"case": "dt_induced_failure", "r0_over_target": 1.00005, "dt": 200.0, "seed": SEED},
    ]
    cases: List[Dict[str, object]] = []
    for spec in case_specs:
        metrics, trace = rollout_trace(
            dt=float(spec["dt"]),
            r0_over_target=float(spec["r0_over_target"]),
            seed=int(spec["seed"]),
        )
        cases.append({"spec": spec, "metrics": metrics_row(metrics), "trace": trace, "phase_durations": phase_durations(trace)})
        print(f"mechanism {spec['case']}: success={metrics.success} crossings={metrics.radius_crossings_total}")
    return cases


def save_compare_plot(path: Path, cases: List[Dict[str, object]], key: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    for case in cases:
        trace = case["trace"]
        ax.plot(trace["time"], trace[key], linewidth=1.1, label=str(case["spec"]["case"]))
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_phase_table(cases: List[Dict[str, object]]) -> None:
    table = []
    for case in cases:
        table.append(
            {
                "case": case["spec"]["case"],
                "spec": case["spec"],
                "metrics": case["metrics"],
                "phase_durations": case["phase_durations"],
            }
        )
    PHASE_TABLE_PATH.write_text(json.dumps(table, indent=2), encoding="utf-8")


def write_summary(cases: List[Dict[str, object]]) -> None:
    lines = [
        "# Mechanism Compare Summary",
        "",
        "## Cases",
        "",
    ]
    for case in cases:
        spec = case["spec"]
        metrics = case["metrics"]
        durations = case["phase_durations"]
        lines.append(
            f"- `{spec['case']}`: r0 `{float(spec['r0_over_target']):.5f}`, dt `{float(spec['dt']):g}`, "
            f"success `{metrics['success']}`, "
            f"crossings `{metrics['radius_crossings_total']}`, first_crossing_step `{metrics['first_crossing_step']}`, "
            f"final_radius_error `{float(metrics['final_radius_error']):.3e}`, phase_durations `{durations}`"
        )
    lines.extend(
        [
            "",
            "## Mechanism Findings",
            "",
            "- The successful rollout reaches the phase transition sequence and spends only a short time in capture before low-authority lock.",
            "- The near-boundary failure does not reach the same crossing/capture sequence within the budget, so the controller remains in the descent mechanism too long.",
            "- The dt-induced failure shows that the controller behavior is coupled to the numerical integration step, not only to the geometric initial condition.",
            "",
            "## Artifacts",
            "",
            f"- `{RADIUS_PNG.as_posix()}`",
            f"- `{VR_PNG.as_posix()}`",
            f"- `{ENERGY_PNG.as_posix()}`",
            f"- `{ACTION_NORM_PNG.as_posix()}`",
            f"- `{PHASE_TABLE_PATH.as_posix()}`",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = run_cases()
    save_compare_plot(RADIUS_PNG, cases, "radius", "Success vs failure | radius", "radius [m]")
    save_compare_plot(VR_PNG, cases, "v_r", "Success vs failure | radial velocity", "v_r [m/s]")
    save_compare_plot(ENERGY_PNG, cases, "energy", "Success vs failure | specific energy", "specific energy [J/kg]")
    save_compare_plot(ACTION_NORM_PNG, cases, "action_norm", "Success vs failure | action norm", "action norm")
    write_phase_table(cases)
    write_summary(cases)
    print(f"Saved mechanism comparison to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
