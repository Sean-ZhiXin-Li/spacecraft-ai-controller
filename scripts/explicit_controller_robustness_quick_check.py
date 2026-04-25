from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.explicit_controller_analysis_utils import SEED, R0_OVER_TARGET, DT, rollout_metrics, set_seed


OUTPUT_PATH = PROJECT_ROOT / "analysis" / "robustness_quick_check.md"
TRIALS = 5
SETTINGS = [
    {"name": "velocity_noise_plus_minus_1pct", "velocity_noise_pct": 0.01},
    {"name": "action_noise_std_0.001", "action_noise_std": 0.001},
    {"name": "small_position_perturbation", "position_perturb_frac": 1.0e-5},
]


def run_setting(setting: Dict[str, object], setting_idx: int) -> Dict[str, object]:
    metrics_rows = []
    for trial in range(TRIALS):
        metrics = rollout_metrics(
            dt=DT,
            r0_over_target=R0_OVER_TARGET,
            seed=SEED + setting_idx * 1000 + trial,
            velocity_noise_pct=float(setting.get("velocity_noise_pct", 0.0)),
            action_noise_std=float(setting.get("action_noise_std", 0.0)),
            position_perturb_frac=float(setting.get("position_perturb_frac", 0.0)),
        )
        metrics_rows.append(metrics)
        print(f"quick_robustness {setting['name']} trial={trial} success={metrics.success}")
    successes = np.asarray([m.success for m in metrics_rows], dtype=bool)
    crossings = np.asarray([m.crossing_occurs for m in metrics_rows], dtype=bool)
    final_errors = np.asarray([m.final_radius_error for m in metrics_rows], dtype=np.float64)
    tail_vrs = np.asarray([m.tail_mean_abs_vr for m in metrics_rows], dtype=np.float64)
    return {
        "name": setting["name"],
        "success_rate": float(np.mean(successes)),
        "crossing_rate": float(np.mean(crossings)),
        "mean_final_radius_error": float(np.mean(final_errors)),
        "std_final_radius_error": float(np.std(final_errors)),
        "mean_tail_mean_abs_vr": float(np.mean(tail_vrs)),
        "std_tail_mean_abs_vr": float(np.std(tail_vrs)),
        "trial_successes": [bool(m.success) for m in metrics_rows],
        "trial_crossings": [bool(m.crossing_occurs) for m in metrics_rows],
    }


def qualitative_behavior(row: Dict[str, object]) -> str:
    success_rate = float(row["success_rate"])
    crossing_rate = float(row["crossing_rate"])
    if success_rate == 1.0:
        return "all trials preserved strict success"
    if success_rate == 0.0 and crossing_rate == 0.0:
        return "all trials failed before reaching the crossing mechanism"
    if success_rate == 0.0:
        return "some trials may cross, but none preserved strict success"
    return "mixed behavior with partial success preservation"


def write_summary(rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Robustness Quick Check",
        "",
        "## Setup",
        "",
        f"- baseline: `dt={DT}`, `r0_over_target={R0_OVER_TARGET}`, `max_steps=100000`, `thrust_scale=10000`",
        f"- trials per perturbation: `{TRIALS}`",
        "- no PPO retraining, no controller redesign, no physics changes",
        "",
        "## Results",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['name']}`: success_rate `{row['success_rate']:.2f}`, crossing_rate `{row['crossing_rate']:.2f}`, "
            f"mean_final_radius_error `{row['mean_final_radius_error']:.3e}`, "
            f"mean_tail_mean_abs_vr `{row['mean_tail_mean_abs_vr']:.3e}`; {qualitative_behavior(row)}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a lightweight smoke robustness check, not a statistical robustness proof.",
            "- The controller is most credible as a narrow deterministic insertion solution; perturbation sensitivity remains a core limitation.",
        ]
    )
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    rows = [run_setting(setting, idx) for idx, setting in enumerate(SETTINGS)]
    write_summary(rows)
    print(f"Saved quick robustness summary to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
