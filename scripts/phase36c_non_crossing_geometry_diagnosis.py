"""Phase36C non-crossing geometry diagnosis.

This script does not run a new controller. It reads the Phase36B benchmark
CSV, isolates the baseline Phase34 non-crossing cases, compares how the
Phase36B transfer families behaved on those same cases, and writes a
planner-search preparation package.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE36B_DIR = ROOT / "analysis" / "phase36b_transfer_family_benchmark"
PHASE36B_RESULTS = PHASE36B_DIR / "phase36b_results.csv"

OUTPUT_DIR = ROOT / "analysis" / "phase36c_non_crossing_geometry_diagnosis"
NON_CROSSING_CASE_SET = OUTPUT_DIR / "non_crossing_case_set.csv"
FAMILY_BEHAVIOR = OUTPUT_DIR / "family_behavior_on_non_crossing_cases.csv"
FAMILY_DELTA = OUTPUT_DIR / "non_crossing_family_delta.csv"
PLANNER_SEARCH_SPACE = OUTPUT_DIR / "planner_search_space.md"
SUMMARY = OUTPUT_DIR / "summary.md"

BASELINE = "baseline_phase34"
EPS = 1e-12


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def parse_float(value: str, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def parse_int(value: str, default: int = -1) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(float(value))


def case_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        canonical_float(row["r0_over_target"]),
        canonical_float(row["initial_velocity_angle_deg"]),
        canonical_float(row["thrust_scale"]),
    )


def canonical_float(value: str) -> str:
    parsed = float(value)
    if parsed.is_integer():
        return str(int(parsed))
    return f"{parsed:.12g}"


def read_phase36b_rows() -> list[dict[str, str]]:
    if not PHASE36B_RESULTS.exists():
        raise FileNotFoundError(f"Missing Phase36B results: {PHASE36B_RESULTS}")
    with PHASE36B_RESULTS.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def bool_text(value: bool) -> str:
    return "True" if value else "False"


def collect_non_crossing_cases(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    baseline_rows = [r for r in rows if r["transfer_family"] == BASELINE]
    non_crossing = [r for r in baseline_rows if not parse_bool(r["crossing_occurs"])]
    non_crossing.sort(
        key=lambda r: (
            parse_float(r["initial_velocity_angle_deg"]),
            parse_float(r["thrust_scale"]),
            parse_float(r["r0_over_target"]),
        )
    )

    case_rows: list[dict[str, Any]] = []
    for row in non_crossing:
        case_rows.append(
            {
                "r0_over_target": row["r0_over_target"],
                "initial_velocity_angle_deg": row["initial_velocity_angle_deg"],
                "thrust_scale": row["thrust_scale"],
                "baseline_min_abs_radius_error_ratio": row["min_abs_radius_error_ratio"],
                "baseline_best_crossing_potential": row["best_crossing_potential"],
                "baseline_closest_approach_step": row["closest_approach_step"],
                "baseline_dominant_failure_label": row["dominant_failure_label"],
            }
        )
    return case_rows


def collect_family_behavior(
    rows: list[dict[str, str]],
    non_crossing_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = {
        (
            canonical_float(case["r0_over_target"]),
            canonical_float(case["initial_velocity_angle_deg"]),
            canonical_float(case["thrust_scale"]),
        )
        for case in non_crossing_cases
    }

    behavior_rows: list[dict[str, Any]] = []
    for row in rows:
        if case_key(row) not in keys:
            continue
        behavior_rows.append(
            {
                "transfer_family": row["transfer_family"],
                "r0_over_target": row["r0_over_target"],
                "initial_velocity_angle_deg": row["initial_velocity_angle_deg"],
                "thrust_scale": row["thrust_scale"],
                "crossing_occurs": row["crossing_occurs"],
                "phase34_compatible_crossing": row["phase34_compatible_crossing"],
                "recoverable_crossing": row["recoverable_crossing"],
                "min_abs_radius_error_ratio": row["min_abs_radius_error_ratio"],
                "best_crossing_potential": row["best_crossing_potential"],
                "closest_approach_step": row["closest_approach_step"],
                "crossing_vr_ratio": row["crossing_vr_ratio"],
                "crossing_vt_error_ratio": row["crossing_vt_error_ratio"],
                "crossing_sync_error": row["crossing_sync_error"],
                "dominant_failure_label": row["dominant_failure_label"],
                "termination_reason": row["termination_reason"],
                "simulator_success_label": row["simulator_success_label"],
            }
        )

    behavior_rows.sort(
        key=lambda r: (
            parse_float(r["initial_velocity_angle_deg"]),
            parse_float(r["thrust_scale"]),
            parse_float(r["r0_over_target"]),
            r["transfer_family"],
        )
    )
    return behavior_rows


def collect_family_deltas(
    rows: list[dict[str, str]],
    non_crossing_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_family_case = {(r["transfer_family"], case_key(r)): r for r in rows}
    delta_rows: list[dict[str, Any]] = []

    for case in non_crossing_cases:
        key = (
            canonical_float(case["r0_over_target"]),
            canonical_float(case["initial_velocity_angle_deg"]),
            canonical_float(case["thrust_scale"]),
        )
        baseline = by_family_case[(BASELINE, key)]
        baseline_min = parse_float(baseline["min_abs_radius_error_ratio"])
        baseline_potential = parse_float(baseline["best_crossing_potential"])
        baseline_step = parse_int(baseline["closest_approach_step"])
        baseline_label = baseline["dominant_failure_label"]

        for (family, row_key), row in sorted(by_family_case.items()):
            if family == BASELINE or row_key != key:
                continue

            min_ratio = parse_float(row["min_abs_radius_error_ratio"])
            potential = parse_float(row["best_crossing_potential"])
            step = parse_int(row["closest_approach_step"])
            delta_min = min_ratio - baseline_min
            delta_potential = potential - baseline_potential
            delta_step = step - baseline_step
            improved_closest = delta_min < -EPS
            improved_potential = delta_potential > EPS
            worsened_geometry = delta_min > EPS or delta_potential < -EPS

            delta_rows.append(
                {
                    "transfer_family": family,
                    "r0_over_target": baseline["r0_over_target"],
                    "initial_velocity_angle_deg": baseline["initial_velocity_angle_deg"],
                    "thrust_scale": baseline["thrust_scale"],
                    "delta_min_abs_radius_error_ratio": f"{delta_min:.12g}",
                    "delta_best_crossing_potential": f"{delta_potential:.12g}",
                    "delta_closest_approach_step": delta_step,
                    "changed_failure_label": bool_text(row["dominant_failure_label"] != baseline_label),
                    "improved_closest_approach": bool_text(improved_closest),
                    "improved_crossing_potential": bool_text(improved_potential),
                    "worsened_geometry": bool_text(worsened_geometry),
                }
            )

    return delta_rows


def count_by(rows: list[dict[str, Any]], field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(field, "")).strip() or "unlabeled"
        counts[value] += 1
    return counts


def write_planner_search_space(
    non_crossing_cases: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> None:
    baseline_labels = count_by(non_crossing_cases, "baseline_dominant_failure_label")
    improved_closest = sum(parse_bool(r["improved_closest_approach"]) for r in delta_rows)
    improved_potential = sum(parse_bool(r["improved_crossing_potential"]) for r in delta_rows)
    worsened = sum(parse_bool(r["worsened_geometry"]) for r in delta_rows)
    family_improvements: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in delta_rows:
        family = row["transfer_family"]
        if parse_bool(row["improved_closest_approach"]):
            family_improvements[family]["closest"] += 1
        if parse_bool(row["improved_crossing_potential"]):
            family_improvements[family]["potential"] += 1
        if parse_bool(row["worsened_geometry"]):
            family_improvements[family]["worsened"] += 1

    lines = [
        "# Phase36C Planner Search Space",
        "",
        "## Purpose",
        "",
        "Phase36C does not define a new controller and does not rerun the simulator. It diagnoses the `16 / 24` baseline Phase36B non-crossing cases and prepares a planner-level search space for upstream crossing-generation in the simplified 2D sandbox.",
        "",
        "Phase36B showed that manual transfer-family variants matched the baseline crossing count but did not expand it. The next step should therefore be parameterized trajectory search, not another manually named heuristic family.",
        "",
        "## Why Manual Family Invention Is No Longer Enough",
        "",
        f"The baseline non-crossing set contains `{len(non_crossing_cases)}` cases. Failure labels are: "
        + ", ".join(f"`{label}={count}`" for label, count in sorted(baseline_labels.items()))
        + ".",
        "",
        f"Across non-baseline families on those same cases, `{improved_closest}` family-case rows improved closest approach, `{improved_potential}` improved crossing potential, and `{worsened}` worsened at least one geometry metric. These mixed local changes did not create new target-radius crossings.",
        "",
        "This indicates that crossing-generation should be searched as a structured trajectory-geometry problem. Manual labels are useful for interpretation, but the next experiment needs explicit parameters that control timing, energy, angular momentum, and handoff geometry.",
        "",
        "## Candidate Search Variables",
        "",
        "- `coast_duration`: number of steps or time fraction before active crossing commitment.",
        "- `radial_push_timing`: when radial motion toward target radius is introduced.",
        "- `radial_push_magnitude`: bounded radial component during the shaping or commit phase.",
        "- `tangential_shaping_magnitude`: bounded tangential correction before crossing.",
        "- `crossing_commit_time`: planned transition from shaping to target-radius crossing attempt.",
        "- `angular_momentum_correction_weight`: weight on tangential velocity or angular momentum proxy alignment.",
        "- `energy_correction_weight`: weight on energy-like transfer proximity.",
        "- `max_action_norm`: cap on the pre-cross action vector.",
        "- `handoff_window_length`: allowed window for crossing and subsequent Phase34 compatibility check.",
        "",
        "## Candidate Objective Terms",
        "",
        "- minimize `min_abs_radius_error_ratio`",
        "- maximize `best_crossing_potential`",
        "- penalize `overspeed`",
        "- penalize `instability`",
        "- penalize bad tangential corridor entry",
        "- reward Phase34-compatible crossing",
        "- reward low crossing sync error",
        "- penalize crossings that occur only with poor Phase34 handoff quality",
        "",
        "## Candidate Search Methods",
        "",
        "- grid search",
        "- random search",
        "- coarse-to-fine search",
        "",
        "Do not use MPC-lite yet. The current need is to map which coarse transfer parameters generate crossings, not to build a receding-horizon controller.",
        "",
        "## Recommended First Search",
        "",
        "Start with a small coarse grid over only 3 to 4 variables:",
        "",
        "1. `coast_duration`",
        "2. `radial_push_timing`",
        "3. `radial_push_magnitude`",
        "4. `tangential_shaping_magnitude`",
        "",
        "Keep Phase34 as the fixed terminal controller. Evaluate only whether the candidate parameter set creates target-radius crossings and whether those crossings are Phase34-compatible. Do not tune after seeing the first result set; treat the first grid as a hypothesis test.",
        "",
        "## Search Discipline",
        "",
        "- Use the same 24-case reduced benchmark.",
        "- Preserve Phase34 post-cross synchronization unchanged.",
        "- Report `simulator_success_label`, not plain success.",
        "- Separate geometric crossing from recoverable crossing.",
        "- Keep near-crossing cases visible instead of hiding them.",
        "- Do not present planner search as real spacecraft validation.",
        "",
        "## Family Delta Signals",
        "",
        "| Transfer family | Improved closest approach rows | Improved crossing potential rows | Worsened geometry rows |",
        "|---|---:|---:|---:|",
    ]

    for family in sorted(family_improvements):
        counts = family_improvements[family]
        lines.append(
            f"| `{family}` | {counts['closest']} | {counts['potential']} | {counts['worsened']} |"
        )

    lines.extend(
        [
            "",
            "## Bottom Line",
            "",
            "The next technical step should be a small parameterized planner search over transfer timing and shaping variables. It should prepare the project for planner-level reasoning without escalating yet to MPC-lite.",
        ]
    )

    PLANNER_SEARCH_SPACE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    non_crossing_cases: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> None:
    baseline_labels = count_by(non_crossing_cases, "baseline_dominant_failure_label")
    behavior_labels = count_by(behavior_rows, "dominant_failure_label")
    improved_closest_rows = [r for r in delta_rows if parse_bool(r["improved_closest_approach"])]
    improved_potential_rows = [r for r in delta_rows if parse_bool(r["improved_crossing_potential"])]
    worsened_rows = [r for r in delta_rows if parse_bool(r["worsened_geometry"])]
    changed_label_rows = [r for r in delta_rows if parse_bool(r["changed_failure_label"])]

    family_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in delta_rows:
        family = row["transfer_family"]
        if parse_bool(row["improved_closest_approach"]):
            family_counts[family]["improved_closest"] += 1
        if parse_bool(row["improved_crossing_potential"]):
            family_counts[family]["improved_potential"] += 1
        if parse_bool(row["worsened_geometry"]):
            family_counts[family]["worsened"] += 1
        if parse_bool(row["changed_failure_label"]):
            family_counts[family]["changed_label"] += 1

    cases_by_label: defaultdict[str, list[str]] = defaultdict(list)
    for case in non_crossing_cases:
        label = case["baseline_dominant_failure_label"]
        descriptor = (
            f"r0={case['r0_over_target']}, "
            f"angle={case['initial_velocity_angle_deg']}, "
            f"thrust={case['thrust_scale']}"
        )
        cases_by_label[label].append(descriptor)

    lines = [
        "# Phase36C Non-Crossing Geometry Diagnosis",
        "",
        "## Scope",
        "",
        "This diagnostic package reads existing Phase36B CSV outputs. It does not run a new controller, change physics, change thresholds, tune family gains, or modify Phase34 terminal behavior.",
        "",
        "The analysis is scoped to the simplified 2D orbital-control sandbox.",
        "",
        "## Which Baseline Cases Failed To Cross?",
        "",
        f"`baseline_phase34` had `{len(non_crossing_cases)} / 24` non-crossing cases in Phase36B.",
        "",
    ]

    for label, descriptors in sorted(cases_by_label.items()):
        lines.append(f"### `{label}`")
        lines.append("")
        for descriptor in descriptors:
            lines.append(f"- {descriptor}")
        lines.append("")

    lines.extend(
        [
            "## Failure Label Distribution",
            "",
            "| Label source | Failure label | Count |",
            "|---|---|---:|",
        ]
    )
    for label, count in sorted(baseline_labels.items()):
        lines.append(f"| baseline non-crossing cases | `{label}` | {count} |")
    for label, count in sorted(behavior_labels.items()):
        lines.append(f"| all Phase36B families on baseline non-crossing set | `{label}` | {count} |")

    lines.extend(
        [
            "",
            "The baseline non-crossing set is split between near-crossing behavior and over-conservative transfer behavior. Near-crossing should be treated as useful information: the geometry approaches the target-radius event but fails to commit under the tested transfer families.",
            "",
            "## Family Delta Summary",
            "",
            "| Transfer family | Improved closest approach | Improved crossing potential | Worsened geometry | Changed failure label |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for family in sorted(family_counts):
        counts = family_counts[family]
        lines.append(
            f"| `{family}` | {counts['improved_closest']} | {counts['improved_potential']} | {counts['worsened']} | {counts['changed_label']} |"
        )

    lines.extend(
        [
            "",
            "## Required Questions",
            "",
            f"- Are the baseline failures mostly near-crossing or over-conservative? They are tied: "
            + ", ".join(f"`{label}={count}`" for label, count in sorted(baseline_labels.items()))
            + ".",
            f"- Did any non-baseline family improve closest approach without crossing? `{'yes' if improved_closest_rows else 'no'}`; `{len(improved_closest_rows)}` non-baseline family-case rows improved `min_abs_radius_error_ratio` relative to baseline.",
            f"- Did any family improve crossing potential without crossing? `{'yes' if improved_potential_rows else 'no'}`; `{len(improved_potential_rows)}` non-baseline family-case rows improved `best_crossing_potential` relative to baseline.",
            f"- Did any family worsen geometry? `{'yes' if worsened_rows else 'no'}`; `{len(worsened_rows)}` non-baseline family-case rows worsened either closest approach or crossing potential.",
            f"- Did any family change failure labels? `{'yes' if changed_label_rows else 'no'}`; `{len(changed_label_rows)}` non-baseline family-case rows changed the baseline diagnostic label.",
            "- What does this imply about crossing-generation? The remaining bottleneck is upstream crossing-generation, and the Phase36B family variants changed local geometry metrics without creating new target-radius crossings.",
            "- What exact planner-level search should be tried next? Run a small coarse grid over `coast_duration`, `radial_push_timing`, `radial_push_magnitude`, and `tangential_shaping_magnitude`, with Phase34 fixed as the terminal controller.",
            "- Why is this not yet MPC-lite? The immediate question is not online replanning; it is whether coarse transfer timing and shaping parameters can create crossings at all in the non-crossing cases.",
            "",
            "## Bottleneck Interpretation",
            "",
            "Phase36B already showed that post-cross recovery is not the limiting factor for crossing-producing cases. Phase36C shows that the non-crossing cases can move in closest-approach and crossing-potential metrics without producing new Phase34-compatible crossings. That points toward a planner-level trajectory search space rather than another manually named local family.",
            "",
            "## Artifacts",
            "",
            "- `non_crossing_case_set.csv`",
            "- `family_behavior_on_non_crossing_cases.csv`",
            "- `non_crossing_family_delta.csv`",
            "- `planner_search_space.md`",
        ]
    )

    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_phase36b_rows()
    non_crossing_cases = collect_non_crossing_cases(rows)
    behavior_rows = collect_family_behavior(rows, non_crossing_cases)
    delta_rows = collect_family_deltas(rows, non_crossing_cases)

    write_csv(
        NON_CROSSING_CASE_SET,
        [
            "r0_over_target",
            "initial_velocity_angle_deg",
            "thrust_scale",
            "baseline_min_abs_radius_error_ratio",
            "baseline_best_crossing_potential",
            "baseline_closest_approach_step",
            "baseline_dominant_failure_label",
        ],
        non_crossing_cases,
    )
    write_csv(
        FAMILY_BEHAVIOR,
        [
            "transfer_family",
            "r0_over_target",
            "initial_velocity_angle_deg",
            "thrust_scale",
            "crossing_occurs",
            "phase34_compatible_crossing",
            "recoverable_crossing",
            "min_abs_radius_error_ratio",
            "best_crossing_potential",
            "closest_approach_step",
            "crossing_vr_ratio",
            "crossing_vt_error_ratio",
            "crossing_sync_error",
            "dominant_failure_label",
            "termination_reason",
            "simulator_success_label",
        ],
        behavior_rows,
    )
    write_csv(
        FAMILY_DELTA,
        [
            "transfer_family",
            "r0_over_target",
            "initial_velocity_angle_deg",
            "thrust_scale",
            "delta_min_abs_radius_error_ratio",
            "delta_best_crossing_potential",
            "delta_closest_approach_step",
            "changed_failure_label",
            "improved_closest_approach",
            "improved_crossing_potential",
            "worsened_geometry",
        ],
        delta_rows,
    )
    write_planner_search_space(non_crossing_cases, behavior_rows, delta_rows)
    write_summary(non_crossing_cases, behavior_rows, delta_rows)

    print(f"Phase36C diagnosis written to {OUTPUT_DIR}")
    print(f"Baseline non-crossing cases: {len(non_crossing_cases)}")


if __name__ == "__main__":
    main()
