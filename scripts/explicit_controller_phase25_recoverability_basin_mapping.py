from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase25_recoverability_basin_mapping"
MPLCONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.explicit_controller_phase21_orbital_transfer_planner import (
    DEFAULT_TARGET_RADIUS,
    MU,
    RECOVERABLE_R_RATIO,
    RECOVERABLE_VR_RATIO,
    RECOVERABLE_VT_RATIO,
    bool_from_csv,
)


PHASE_INPUTS = [
    ("phase22", PROJECT_ROOT / "analysis" / "phase22_two_burn_transfer" / "phase22_results.csv"),
    ("phase23", PROJECT_ROOT / "analysis" / "phase23_windowed_insertion_solver" / "phase23_results.csv"),
    ("phase24", PROJECT_ROOT / "analysis" / "phase24_precision_insertion_geometry" / "phase24_results.csv"),
]

CROSSING_DATASET_CSV = OUTPUT_DIR / "phase25_crossing_dataset.csv"
THRESHOLD_ABLATION_CSV = OUTPUT_DIR / "phase25_threshold_ablation.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"

PLOT_R_VR = OUTPUT_DIR / "crossing_r_vs_vr.png"
PLOT_VR_VT = OUTPUT_DIR / "crossing_vr_vs_vt.png"
PLOT_DISTANCE_HIST = OUTPUT_DIR / "recoverability_distance_histogram.png"
PLOT_THRESHOLD_HEATMAP = OUTPUT_DIR / "threshold_ablation_heatmap.png"
PLOT_FAILURE_CLUSTERS = OUTPUT_DIR / "failure_mode_clusters.png"
PLOT_BASIN_BOUNDARY = OUTPUT_DIR / "basin_boundary_analysis.png"

THRESHOLD_FACTORS = [1.0, 1.5, 2.0, 3.0]

DATASET_FIELDNAMES = [
    "phase",
    "controller_name",
    "state_type",
    "case_id",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "target_radius_scale",
    "r_error_ratio",
    "vr_ratio",
    "vt_error_ratio",
    "energy_error_ratio",
    "angular_momentum_error_ratio",
    "crossing_direction",
    "overspeed",
    "handoff_entered",
    "capture_entered",
    "success",
    "recoverable_original",
    "distance_to_recoverable",
    "dominant_failure_variable",
    "failure_cluster",
    "source_note",
]

ABLATION_FIELDNAMES = [
    "r_factor",
    "vr_factor",
    "vt_factor",
    "recoverable_state_count",
    "recoverable_case_count",
    "potential_capture_case_count",
    "potential_capture_gain_vs_observed",
    "dominant_blocking_threshold",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: object, default: float = float("nan")) -> float:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def target_radius(row: Dict[str, str]) -> float:
    return DEFAULT_TARGET_RADIUS * as_float(row.get("target_radius_scale"), 1.0)


def case_id(row: Dict[str, str]) -> str:
    return (
        f"r0={as_float(row.get('r0_over_target')):.5g}|"
        f"angle={as_float(row.get('initial_velocity_angle_deg')):.5g}|"
        f"thrust={as_float(row.get('thrust_scale')):.5g}|"
        f"target={as_float(row.get('target_radius_scale'), 1.0):.5g}"
    )


def orbital_errors_from_crossing(row: Dict[str, str]) -> tuple[float, float]:
    radius = target_radius(row)
    v_circ = math.sqrt(MU / radius)
    vr_ratio = as_float(row.get("crossing_vr_ratio"))
    vt_error_ratio = as_float(row.get("crossing_vt_error_ratio"))
    if not math.isfinite(vr_ratio) or not math.isfinite(vt_error_ratio):
        return float("nan"), float("nan")
    vr = vr_ratio * v_circ
    vt = (1.0 + vt_error_ratio) * v_circ
    energy = 0.5 * (vr * vr + vt * vt) - MU / radius
    target_energy = -MU / (2.0 * radius)
    h = radius * vt
    h_target = math.sqrt(MU * radius)
    return (
        (energy - target_energy) / (abs(target_energy) + 1.0e-12),
        (h - h_target) / (abs(h_target) + 1.0e-12),
    )


def recoverability_distance(r_ratio: float, vr_ratio: float, vt_ratio: float) -> float:
    if not all(math.isfinite(value) for value in [r_ratio, vr_ratio, vt_ratio]):
        return float("nan")
    dr = abs(r_ratio) / RECOVERABLE_R_RATIO
    dvr = abs(vr_ratio) / RECOVERABLE_VR_RATIO
    dvt = abs(vt_ratio) / RECOVERABLE_VT_RATIO
    return math.sqrt(dr * dr + dvr * dvr + dvt * dvt)


def dominant_failure_variable(r_ratio: float, vr_ratio: float, vt_ratio: float, e_ratio: float, h_ratio: float) -> str:
    components = {
        "radius": abs(r_ratio) / RECOVERABLE_R_RATIO if math.isfinite(r_ratio) else -1.0,
        "radial_velocity": abs(vr_ratio) / RECOVERABLE_VR_RATIO if math.isfinite(vr_ratio) else -1.0,
        "tangential_velocity": abs(vt_ratio) / RECOVERABLE_VT_RATIO if math.isfinite(vt_ratio) else -1.0,
        "energy": abs(e_ratio) if math.isfinite(e_ratio) else -1.0,
        "angular_momentum": abs(h_ratio) if math.isfinite(h_ratio) else -1.0,
    }
    return max(components.items(), key=lambda item: item[1])[0]


def failure_cluster(r_ratio: float, vr_ratio: float, vt_ratio: float, e_ratio: float, h_ratio: float, overspeed: bool) -> str:
    if overspeed:
        return "overspeed_collapse"
    if not all(math.isfinite(value) for value in [r_ratio, vr_ratio, vt_ratio]):
        return "uninstrumented_window"
    radius_close = abs(r_ratio) <= 3.0 * RECOVERABLE_R_RATIO
    vr_good = abs(vr_ratio) <= 3.0 * RECOVERABLE_VR_RATIO
    vt_good = abs(vt_ratio) <= 3.0 * RECOVERABLE_VT_RATIO
    if radius_close and not vr_good:
        return "radius_close_but_vr_bad"
    if radius_close and not vt_good:
        return "radius_close_but_vt_bad"
    if vr_good and vt_good and not radius_close:
        return "speed_good_but_radius_bad"
    if math.isfinite(e_ratio) and abs(e_ratio) > 0.50:
        return "energy_mismatch"
    if math.isfinite(h_ratio) and abs(h_ratio) > 0.50:
        return "angular_momentum_mismatch"
    return "mixed_geometry_error"


def make_dataset_row(
    phase: str,
    row: Dict[str, str],
    state_type: str,
    r_ratio: float,
    vr_ratio: float,
    vt_ratio: float,
    energy_ratio: float,
    h_ratio: float,
    crossing_direction: str,
    source_note: str,
) -> Dict[str, object]:
    overspeed = bool_from_csv(row.get("overspeed"))
    distance = recoverability_distance(r_ratio, vr_ratio, vt_ratio)
    dom = dominant_failure_variable(r_ratio, vr_ratio, vt_ratio, energy_ratio, h_ratio)
    cluster = failure_cluster(r_ratio, vr_ratio, vt_ratio, energy_ratio, h_ratio, overspeed)
    return {
        "phase": phase,
        "controller_name": row.get("controller_name", ""),
        "state_type": state_type,
        "case_id": case_id(row),
        "r0_over_target": row.get("r0_over_target", ""),
        "initial_velocity_angle_deg": row.get("initial_velocity_angle_deg", ""),
        "thrust_scale": row.get("thrust_scale", ""),
        "target_radius_scale": row.get("target_radius_scale", ""),
        "r_error_ratio": r_ratio,
        "vr_ratio": vr_ratio,
        "vt_error_ratio": vt_ratio,
        "energy_error_ratio": energy_ratio,
        "angular_momentum_error_ratio": h_ratio,
        "crossing_direction": crossing_direction,
        "overspeed": overspeed,
        "handoff_entered": bool_from_csv(row.get("handoff_entered")),
        "capture_entered": bool_from_csv(row.get("capture_entered")),
        "success": bool_from_csv(row.get("success")),
        "recoverable_original": bool_from_csv(row.get("recoverable_crossing")),
        "distance_to_recoverable": distance,
        "dominant_failure_variable": dom,
        "failure_cluster": cluster,
        "source_note": source_note,
    }


def extract_dataset() -> List[Dict[str, object]]:
    dataset: List[Dict[str, object]] = []
    for phase, path in PHASE_INPUTS:
        rows = read_csv(path)
        controller_rows = [row for row in rows if str(row.get("controller_name", "")).startswith(phase)]
        for row in controller_rows:
            if bool_from_csv(row.get("crossing_occurs")):
                r_ratio = as_float(row.get("crossing_radius_error")) / target_radius(row)
                vr_ratio = as_float(row.get("crossing_vr_ratio"))
                vt_ratio = as_float(row.get("crossing_vt_error_ratio"))
                e_ratio, h_ratio = orbital_errors_from_crossing(row)
                direction = "inbound" if math.isfinite(vr_ratio) and vr_ratio < 0.0 else "outbound"
                dataset.append(
                    make_dataset_row(
                        phase,
                        row,
                        "radius_crossing",
                        r_ratio,
                        vr_ratio,
                        vt_ratio,
                        e_ratio,
                        h_ratio,
                        direction,
                        "crossing state reconstructed from crossing vr/vt/r fields",
                    )
                )

            insertion_windows = int(as_float(row.get("insertion_windows"), 0.0))
            if insertion_windows > 0:
                pre_payload = str(row.get("pre_burn_state_json") or "{}")
                post_payload = str(row.get("post_burn_state_json") or "{}")
                for state_type, payload in [("insertion_window_pre_burn", pre_payload), ("insertion_window_post_burn", post_payload)]:
                    try:
                        state = json.loads(payload)
                    except json.JSONDecodeError:
                        state = {}
                    if state:
                        r_ratio = as_float(state.get("r_error_ratio"))
                        vr_ratio = as_float(state.get("vr_ratio"))
                        vt_ratio = as_float(state.get("vt_error_ratio"))
                        e_ratio = as_float(state.get("energy_error_ratio"))
                        h_ratio = as_float(state.get("angular_momentum_error_ratio"))
                        direction = "inbound" if math.isfinite(vr_ratio) and vr_ratio < 0.0 else "outbound"
                        dataset.append(
                            make_dataset_row(
                                phase,
                                row,
                                state_type,
                                r_ratio,
                                vr_ratio,
                                vt_ratio,
                                e_ratio,
                                h_ratio,
                                direction,
                                "instrumented insertion geometry state",
                            )
                        )
                    else:
                        dataset.append(
                            make_dataset_row(
                                phase,
                                row,
                                "insertion_window_uninstrumented",
                                float("nan"),
                                float("nan"),
                                float("nan"),
                                float("nan"),
                                float("nan"),
                                "",
                                "prior phase recorded insertion-window count but not state geometry",
                            )
                        )
    return dataset


def numeric_dataset_rows(dataset: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for row in dataset:
        values = [as_float(row.get("r_error_ratio")), as_float(row.get("vr_ratio")), as_float(row.get("vt_error_ratio"))]
        if all(math.isfinite(value) for value in values):
            rows.append(row)
    return rows


def relaxed_recoverable(row: Dict[str, object], r_factor: float, vr_factor: float, vt_factor: float) -> bool:
    r_ratio = as_float(row.get("r_error_ratio"))
    vr_ratio = as_float(row.get("vr_ratio"))
    vt_ratio = as_float(row.get("vt_error_ratio"))
    overspeed = bool_from_csv(row.get("overspeed"))
    return (
        math.isfinite(r_ratio)
        and math.isfinite(vr_ratio)
        and math.isfinite(vt_ratio)
        and not overspeed
        and abs(r_ratio) <= RECOVERABLE_R_RATIO * r_factor
        and abs(vr_ratio) <= RECOVERABLE_VR_RATIO * vr_factor
        and abs(vt_ratio) <= RECOVERABLE_VT_RATIO * vt_factor
    )


def blocking_threshold_counts(rows: Sequence[Dict[str, object]], r_factor: float, vr_factor: float, vt_factor: float) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        r_ratio = abs(as_float(row.get("r_error_ratio"))) / (RECOVERABLE_R_RATIO * r_factor)
        vr_ratio = abs(as_float(row.get("vr_ratio"))) / (RECOVERABLE_VR_RATIO * vr_factor)
        vt_ratio = abs(as_float(row.get("vt_error_ratio"))) / (RECOVERABLE_VT_RATIO * vt_factor)
        counts[max([("radius", r_ratio), ("radial_velocity", vr_ratio), ("tangential_velocity", vt_ratio)], key=lambda item: item[1])[0]] += 1
    return counts


def run_threshold_ablation(dataset: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = numeric_dataset_rows(dataset)
    observed_capture_cases = {row["case_id"] for row in dataset if bool_from_csv(row.get("capture_entered"))}
    ablation_rows: List[Dict[str, object]] = []
    for r_factor in THRESHOLD_FACTORS:
        for vr_factor in THRESHOLD_FACTORS:
            for vt_factor in THRESHOLD_FACTORS:
                relaxed_rows = [row for row in rows if relaxed_recoverable(row, r_factor, vr_factor, vt_factor)]
                relaxed_cases = {row["case_id"] for row in relaxed_rows}
                potential_capture_cases = observed_capture_cases | relaxed_cases
                blockers = blocking_threshold_counts(rows, r_factor, vr_factor, vt_factor)
                dominant = blockers.most_common(1)[0][0] if blockers else ""
                ablation_rows.append(
                    {
                        "r_factor": r_factor,
                        "vr_factor": vr_factor,
                        "vt_factor": vt_factor,
                        "recoverable_state_count": len(relaxed_rows),
                        "recoverable_case_count": len(relaxed_cases),
                        "potential_capture_case_count": len(potential_capture_cases),
                        "potential_capture_gain_vs_observed": len(potential_capture_cases) - len(observed_capture_cases),
                        "dominant_blocking_threshold": dominant,
                    }
                )
    return ablation_rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for key in fieldnames:
                value = row.get(key, "")
                cleaned[key] = "" if isinstance(value, float) and not math.isfinite(value) else value
            writer.writerow(cleaned)


def values(rows: Sequence[Dict[str, object]], key: str) -> np.ndarray:
    data = [as_float(row.get(key)) for row in rows]
    return np.asarray([value for value in data if math.isfinite(value)], dtype=np.float64)


def save_scatter(path: Path, rows: Sequence[Dict[str, object]], x_key: str, y_key: str, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    phase_colors = {"phase22": "#4C78A8", "phase23": "#F58518", "phase24": "#54A24B"}
    for phase in ["phase22", "phase23", "phase24"]:
        subset = [row for row in rows if row.get("phase") == phase]
        x = values(subset, x_key)
        y = values(subset, y_key)
        if len(x) and len(y):
            ax.scatter(x, y, s=44, alpha=0.78, label=phase, color=phase_colors[phase])
    if x_key == "r_error_ratio":
        ax.axvline(RECOVERABLE_R_RATIO, color="#333333", linestyle="--", linewidth=0.8)
        ax.axvline(-RECOVERABLE_R_RATIO, color="#333333", linestyle="--", linewidth=0.8)
    if y_key == "vr_ratio":
        ax.axhline(RECOVERABLE_VR_RATIO, color="#333333", linestyle="--", linewidth=0.8)
        ax.axhline(-RECOVERABLE_VR_RATIO, color="#333333", linestyle="--", linewidth=0.8)
    if y_key == "vt_error_ratio":
        ax.axhline(RECOVERABLE_VT_RATIO, color="#333333", linestyle="--", linewidth=0.8)
        ax.axhline(-RECOVERABLE_VT_RATIO, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_plots(dataset: Sequence[Dict[str, object]], ablation_rows: Sequence[Dict[str, object]]) -> None:
    rows = numeric_dataset_rows(dataset)
    save_scatter(PLOT_R_VR, rows, "r_error_ratio", "vr_ratio", "Crossing-state radius vs radial velocity", "r error / target", "v_r / v_circ")
    save_scatter(PLOT_VR_VT, rows, "vr_ratio", "vt_error_ratio", "Crossing-state radial vs tangential velocity", "v_r / v_circ", "v_t error / v_circ")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    distances = values(rows, "distance_to_recoverable")
    if len(distances):
        ax.hist(distances, bins=24, color="#4C78A8", edgecolor="#FFFFFF")
    ax.axvline(math.sqrt(3.0), color="#E45756", linestyle="--", linewidth=1.0, label="all normalized terms <= 1")
    ax.set_title("Distance to recoverability")
    ax.set_xlabel("weighted distance")
    ax.set_ylabel("states")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DISTANCE_HIST, dpi=220)
    plt.close(fig)

    heat = np.zeros((len(THRESHOLD_FACTORS), len(THRESHOLD_FACTORS)), dtype=np.float64)
    for i, r_factor in enumerate(THRESHOLD_FACTORS):
        for j, vr_factor in enumerate(THRESHOLD_FACTORS):
            selected = [
                row
                for row in ablation_rows
                if float(row["r_factor"]) == r_factor and float(row["vr_factor"]) == vr_factor and float(row["vt_factor"]) == 3.0
            ]
            heat[i, j] = float(selected[0]["recoverable_state_count"]) if selected else 0.0
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(heat, cmap="viridis", origin="lower")
    ax.set_xticks(np.arange(len(THRESHOLD_FACTORS)))
    ax.set_yticks(np.arange(len(THRESHOLD_FACTORS)))
    ax.set_xticklabels([str(v) for v in THRESHOLD_FACTORS])
    ax.set_yticklabels([str(v) for v in THRESHOLD_FACTORS])
    ax.set_xlabel("v_r threshold factor")
    ax.set_ylabel("r threshold factor")
    ax.set_title("Threshold ablation heatmap (v_t factor = 3.0)")
    fig.colorbar(im, ax=ax, label="recoverable states")
    fig.tight_layout()
    fig.savefig(PLOT_THRESHOLD_HEATMAP, dpi=220)
    plt.close(fig)

    cluster_counts = Counter(str(row.get("failure_cluster")) for row in dataset)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    labels = list(cluster_counts.keys())
    ax.bar(labels, [cluster_counts[label] for label in labels], color="#F58518")
    ax.set_title("Failure mode clusters")
    ax.set_ylabel("states")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_FAILURE_CLUSTERS, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    x = values(rows, "energy_error_ratio")
    y = values(rows, "angular_momentum_error_ratio")
    if len(x) and len(y):
        ax.scatter(x, y, s=44, alpha=0.78, color="#54A24B")
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_title("Basin boundary analysis: energy vs angular momentum")
    ax.set_xlabel("energy error ratio")
    ax.set_ylabel("angular momentum error ratio")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_BASIN_BOUNDARY, dpi=220)
    plt.close(fig)


def summarize(dataset: Sequence[Dict[str, object]], ablation_rows: Sequence[Dict[str, object]]) -> None:
    numeric_rows = numeric_dataset_rows(dataset)
    distances = values(numeric_rows, "distance_to_recoverable")
    cluster_counts = Counter(str(row.get("failure_cluster")) for row in dataset)
    numeric_cluster_counts = Counter(str(row.get("failure_cluster")) for row in numeric_rows)
    variable_counts = Counter(str(row.get("dominant_failure_variable")) for row in numeric_rows)
    best_ablation = max(ablation_rows, key=lambda row: (int(row["recoverable_state_count"]), int(row["recoverable_case_count"]))) if ablation_rows else {}
    baseline_ablation = next(
        (
            row
            for row in ablation_rows
            if float(row["r_factor"]) == 1.0 and float(row["vr_factor"]) == 1.0 and float(row["vt_factor"]) == 1.0
        ),
        {},
    )
    threshold_produces_recoverable = bool(best_ablation and int(best_ablation["recoverable_state_count"]) > 0)
    threshold_capture_gain = int(best_ablation.get("potential_capture_gain_vs_observed", 0)) if best_ablation else 0
    if threshold_produces_recoverable:
        bottleneck = "recoverability basin width"
    else:
        bottleneck = "crossing geometry fundamentally outside tested basin"
    dominant_cluster = cluster_counts.most_common(1)[0][0] if cluster_counts else ""
    dominant_numeric_cluster = numeric_cluster_counts.most_common(1)[0][0] if numeric_cluster_counts else ""
    dominant_variable = variable_counts.most_common(1)[0][0] if variable_counts else ""

    lines = [
        "# Phase 25 Recoverability Basin Mapping and CAPTURE Threshold Ablation",
        "",
        "## Scope",
        "",
        "- CSV-first structural analysis using Phase 22, Phase 23, and Phase 24 outputs.",
        "- No trajectories are rerun and no controller, physics, Burn A, coast arc, or Burn B logic is changed.",
        "- Phase 22/23 insertion windows did not store full geometry; those rows are marked as `insertion_window_uninstrumented`.",
        "",
        "## Dataset",
        "",
        f"- Total extracted states: `{len(dataset)}`.",
        f"- Numeric geometry states: `{len(numeric_rows)}`.",
        f"- Original recoverable states: `{sum(bool_from_csv(row.get('recoverable_original')) for row in dataset)}`.",
        f"- Median distance to recoverability: `{float(np.median(distances)) if len(distances) else float('nan'):.4f}`.",
        f"- Minimum distance to recoverability: `{float(np.min(distances)) if len(distances) else float('nan'):.4f}`.",
        f"- Dominant failure variable: `{dominant_variable}`.",
        f"- Dominant failure cluster, all states: `{dominant_cluster}`.",
        f"- Dominant failure cluster, numeric states only: `{dominant_numeric_cluster}`.",
        "",
        "## Threshold Ablation",
        "",
        f"- Baseline threshold recoverable states: `{baseline_ablation.get('recoverable_state_count', 0)}`.",
        f"- Best relaxed threshold recoverable states: `{best_ablation.get('recoverable_state_count', 0)}`.",
        f"- Best relaxed threshold recoverable cases: `{best_ablation.get('recoverable_case_count', 0)}`.",
        f"- Best threshold factors: r `{best_ablation.get('r_factor', '')}`, vr `{best_ablation.get('vr_factor', '')}`, vt `{best_ablation.get('vt_factor', '')}`.",
        f"- Dominant blocking threshold at best setting: `{best_ablation.get('dominant_blocking_threshold', '')}`.",
        f"- Potential CAPTURE case gain from threshold-only reclassification: `{threshold_capture_gain}`.",
        "",
        "## Research Answers",
        "",
        f"1. Are any prior crossings near recoverable? `{'yes' if len(distances) and float(np.min(distances)) <= 3.0 else 'no'}`.",
        f"2. Which variable is most often failing? `{dominant_variable}`.",
        f"3. Which threshold blocks recoverability most? `{best_ablation.get('dominant_blocking_threshold', '')}` at the best relaxed setting.",
        f"4. Can CAPTURE improve via threshold architecture only? `{'yes' if threshold_capture_gain > 0 else 'no'}` under the tested 1.0-3.0 factor grid. Recoverability can improve: `{'yes' if threshold_produces_recoverable else 'no'}`.",
        f"5. Current bottleneck: `{bottleneck}`.",
        "",
        "## Success Criteria",
        "",
        f"- Minimum, identify dominant failure variable: `{'met' if dominant_variable else 'not met'}`.",
        f"- Moderate, threshold relaxation produces recoverability: `{'met' if threshold_produces_recoverable else 'not met'}`.",
        f"- Strong, isolate exact bottleneck threshold: `{'met' if best_ablation.get('dominant_blocking_threshold') else 'not met'}`.",
        f"- Major, prove CAPTURE architecture is bottleneck: `{'met' if threshold_capture_gain > 0 else 'not met'}`.",
        "",
        "## Honesty Note",
        "",
        "- Threshold ablation reclassifies existing states only; it does not rerun trajectories.",
        "- CAPTURE gain is reported as potential acceptance gain, not as a changed simulation result.",
        "- If no relaxed setting works, this script reports crossing geometry as invalid for the tested basin.",
        "",
        "## Artifacts",
        "",
        "- `phase25_crossing_dataset.csv`",
        "- `phase25_threshold_ablation.csv`",
        "- `crossing_r_vs_vr.png`",
        "- `crossing_vr_vs_vt.png`",
        "- `recoverability_distance_histogram.png`",
        "- `threshold_ablation_heatmap.png`",
        "- `failure_mode_clusters.png`",
        "- `basin_boundary_analysis.png`",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = extract_dataset()
    ablation_rows = run_threshold_ablation(dataset)
    write_csv(CROSSING_DATASET_CSV, dataset, DATASET_FIELDNAMES)
    write_csv(THRESHOLD_ABLATION_CSV, ablation_rows, ABLATION_FIELDNAMES)
    save_plots(dataset, ablation_rows)
    summarize(dataset, ablation_rows)
    print(f"Saved Phase 25 recoverability basin mapping outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
