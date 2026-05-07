from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase28_trajectory_family_mapping"
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
    TARGET_RADIUS_SCALE,
    bool_from_csv,
    orbital_diagnostics,
    orbit_crosses_target,
)


PHASE_INPUTS = [
    ("phase22", PROJECT_ROOT / "analysis" / "phase22_two_burn_transfer" / "phase22_results.csv"),
    ("phase23", PROJECT_ROOT / "analysis" / "phase23_windowed_insertion_solver" / "phase23_results.csv"),
    ("phase24", PROJECT_ROOT / "analysis" / "phase24_precision_insertion_geometry" / "phase24_results.csv"),
    ("phase26", PROJECT_ROOT / "analysis" / "phase26_tangential_velocity_corridor" / "phase26_results.csv"),
    ("phase27", PROJECT_ROOT / "analysis" / "phase27_timing_synchronization" / "phase27_results.csv"),
]
PHASE25_DATASET = PROJECT_ROOT / "analysis" / "phase25_recoverability_basin_mapping" / "phase25_crossing_dataset.csv"

DATASET_CSV = OUTPUT_DIR / "phase28_family_dataset.csv"
SUMMARY_CSV = OUTPUT_DIR / "phase28_family_summary.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"

PLOT_ENERGY_H = OUTPUT_DIR / "family_energy_vs_angular_momentum.png"
PLOT_ANGLE_R0 = OUTPUT_DIR / "family_initial_angle_vs_r0.png"
PLOT_THRUST = OUTPUT_DIR / "family_thrust_outcome.png"
PLOT_CROSSING_V = OUTPUT_DIR / "crossing_vt_vs_vr_by_family.png"
PLOT_SYNC_PRED = OUTPUT_DIR / "sync_error_vs_predicted_crossing.png"
PLOT_WINDOW_QUALITY = OUTPUT_DIR / "window_count_vs_quality.png"
PLOT_IMPORTANCE = OUTPUT_DIR / "feature_importance_or_correlation.png"

PHASE_CONTROLLER_FILTERS = {
    "phase22": {"baseline_soft_linear_3e4", "phase22_two_burn_transfer"},
    "phase23": {"phase23_windowed_insertion_solver"},
    "phase24": {"phase24_precision_insertion_geometry"},
    "phase26": {
        "phase26_vt_aware_scoring",
        "phase26_two_step_corridor",
        "phase26_burn_hold_burn",
        "phase26_stronger_tangential",
    },
    "phase27": {
        "phase27_predicted_cross_vt_targeting",
        "phase27_delayed_sync_burn",
        "phase27_split_phase_sync_burn",
        "phase27_adaptive_sync_corridor",
    },
}

FAMILY_ORDER = [
    "dead_geometry",
    "window_no_crossing",
    "crossing_bad_sync",
    "near_recoverable_crossing",
    "capture_success_existing",
]
FAMILY_COLORS = {
    "dead_geometry": "#9D755D",
    "window_no_crossing": "#F58518",
    "crossing_bad_sync": "#E45756",
    "near_recoverable_crossing": "#72B7B2",
    "capture_success_existing": "#54A24B",
}

DATASET_FIELDS = [
    "source_phase",
    "controller_name",
    "case_id",
    "family_label",
    "window_quality_label",
    "quality_score",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "target_radius_scale",
    "initial_radius_ratio",
    "initial_vr_ratio",
    "initial_vt_error_ratio",
    "initial_energy_error_ratio_reconstructed",
    "initial_angular_momentum_error_ratio_reconstructed",
    "initial_semi_major_axis_ratio",
    "initial_periapsis_ratio",
    "initial_apoapsis_ratio",
    "initial_eccentricity_proxy",
    "initial_orbit_crosses_target",
    "initial_predicted_crossing_step",
    "reported_initial_energy_error_ratio",
    "reported_initial_angular_momentum_error_ratio",
    "early_r_error_ratio",
    "early_vr_ratio",
    "early_vt_error_ratio",
    "early_energy_error_ratio",
    "early_angular_momentum_error_ratio",
    "early_feature_source",
    "insertion_windows",
    "successful_insertion_windows",
    "predicted_crossing_step",
    "first_crossing_step",
    "timing_offset_steps",
    "crossing_radius_error_ratio",
    "crossing_vr_ratio",
    "crossing_vt_error_ratio",
    "crossing_distance_to_recoverable",
    "sync_error_at_crossing",
    "min_distance_to_recoverable",
    "best_vt_band",
    "best_sync_band",
    "recoverable_crossing",
    "capture_entered",
    "success",
    "overspeed",
    "instability",
    "dominant_failure_variable",
    "failure_cluster",
    "missing_feature_notes",
]

SUMMARY_FIELDS = [
    "group_type",
    "group_name",
    "case_count",
    "crossing_cases",
    "window_cases",
    "good_window_cases",
    "dead_window_cases",
    "recoverable_crossings",
    "capture",
    "success",
    "mean_quality_score",
    "mean_sync_error_at_crossing",
    "mean_min_distance_to_recoverable",
    "dominant_family_label",
    "dominant_failure_variable",
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


def as_int(value: object, default: int = 0) -> int:
    value_f = as_float(value)
    if not math.isfinite(value_f):
        return default
    return int(value_f)


def case_id(row: Dict[str, object]) -> str:
    return (
        f"r0={as_float(row.get('r0_over_target')):.5g}|"
        f"angle={as_float(row.get('initial_velocity_angle_deg')):.5g}|"
        f"thrust={as_float(row.get('thrust_scale')):.5g}|"
        f"target={as_float(row.get('target_radius_scale'), 1.0):.5g}"
    )


def sync_error(r_ratio: float, vr_ratio: float, vt_ratio: float) -> float:
    if not all(math.isfinite(value) for value in [r_ratio, vr_ratio, vt_ratio]):
        return float("nan")
    return max(
        abs(r_ratio) / RECOVERABLE_R_RATIO,
        abs(vr_ratio) / RECOVERABLE_VR_RATIO,
        abs(vt_ratio) / RECOVERABLE_VT_RATIO,
    )


def recoverability_distance(r_ratio: float, vr_ratio: float, vt_ratio: float) -> float:
    if not all(math.isfinite(value) for value in [r_ratio, vr_ratio, vt_ratio]):
        return float("nan")
    dr = abs(r_ratio) / RECOVERABLE_R_RATIO
    dvr = abs(vr_ratio) / RECOVERABLE_VR_RATIO
    dvt = abs(vt_ratio) / RECOVERABLE_VT_RATIO
    return math.sqrt(dr * dr + dvr * dvr + dvt * dvt)


def estimate_crossing_steps(radius_error: float, radial_velocity: float) -> float:
    if radius_error * radial_velocity >= 0.0 or abs(radial_velocity) < 1.0e-12:
        return float("nan")
    return abs(radius_error) / (abs(radial_velocity) * 100.0 + 1.0e-12)


def initial_features(row: Dict[str, str]) -> Dict[str, object]:
    r0 = as_float(row.get("r0_over_target"))
    target_scale = as_float(row.get("target_radius_scale"), TARGET_RADIUS_SCALE)
    angle = as_float(row.get("initial_velocity_angle_deg"))
    if not all(math.isfinite(value) for value in [r0, target_scale, angle]):
        return {}
    target_radius = DEFAULT_TARGET_RADIUS * target_scale
    x = 0.0
    y = r0 * target_radius
    radius0 = math.sqrt(x * x + y * y)
    v_circ = math.sqrt(MU / target_radius)
    v_mag = math.sqrt(MU / radius0)
    angle_rad = math.radians(angle)
    vx = v_mag * math.cos(angle_rad)
    vy = v_mag * math.sin(angle_rad)
    diag = orbital_diagnostics(x, y, vx, vy, target_radius)
    vr_ratio = diag.radial_velocity / (v_circ + 1.0e-12)
    predicted = estimate_crossing_steps(diag.radius - target_radius, diag.radial_velocity)
    eccentricity = float("nan")
    if diag.periapsis is not None and diag.apoapsis is not None and (diag.apoapsis + diag.periapsis) > 0.0:
        eccentricity = (diag.apoapsis - diag.periapsis) / (diag.apoapsis + diag.periapsis)
    return {
        "initial_radius_ratio": diag.radius / target_radius,
        "initial_vr_ratio": vr_ratio,
        "initial_vt_error_ratio": diag.vt_error_ratio,
        "initial_energy_error_ratio_reconstructed": diag.energy_error_ratio,
        "initial_angular_momentum_error_ratio_reconstructed": diag.angular_momentum_error_ratio,
        "initial_semi_major_axis_ratio": diag.semi_major_axis / target_radius if diag.semi_major_axis is not None else "",
        "initial_periapsis_ratio": diag.periapsis / target_radius if diag.periapsis is not None else "",
        "initial_apoapsis_ratio": diag.apoapsis / target_radius if diag.apoapsis is not None else "",
        "initial_eccentricity_proxy": eccentricity if math.isfinite(eccentricity) else "",
        "initial_orbit_crosses_target": orbit_crosses_target(diag, target_radius),
        "initial_predicted_crossing_step": predicted if math.isfinite(predicted) else "",
    }


def load_phase25_enrichment() -> Dict[tuple[str, str], Dict[str, object]]:
    enrichment: Dict[tuple[str, str], Dict[str, object]] = {}
    for row in read_csv(PHASE25_DATASET):
        if row.get("state_type") != "radius_crossing":
            continue
        key = (str(row.get("phase", "")), str(row.get("case_id", "")))
        distance = as_float(row.get("distance_to_recoverable"))
        current = enrichment.get(key)
        if current is None or (math.isfinite(distance) and distance < as_float(current.get("distance_to_recoverable"))):
            enrichment[key] = {
                "distance_to_recoverable": distance if math.isfinite(distance) else "",
                "dominant_failure_variable": row.get("dominant_failure_variable", ""),
                "failure_cluster": row.get("failure_cluster", ""),
            }
    return enrichment


def parse_state_json(payload: object) -> Dict[str, float]:
    if payload in {None, ""}:
        return {}
    try:
        data = json.loads(str(payload))
    except json.JSONDecodeError:
        return {}
    out: Dict[str, float] = {}
    for key in [
        "r_error_ratio",
        "vr_ratio",
        "vt_error_ratio",
        "energy_error_ratio",
        "angular_momentum_error_ratio",
    ]:
        value = as_float(data.get(key))
        if math.isfinite(value):
            out[key] = value
    return out


def normalize_phase_row(source_phase: str, row: Dict[str, str], phase25: Dict[tuple[str, str], Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {key: "" for key in DATASET_FIELDS}
    cid = case_id(row)
    out.update(
        {
            "source_phase": source_phase,
            "controller_name": row.get("controller_name", ""),
            "case_id": cid,
            "r0_over_target": row.get("r0_over_target", ""),
            "initial_velocity_angle_deg": row.get("initial_velocity_angle_deg", ""),
            "thrust_scale": row.get("thrust_scale", ""),
            "target_radius_scale": row.get("target_radius_scale", ""),
            "reported_initial_energy_error_ratio": row.get("initial_energy_error_ratio", ""),
            "reported_initial_angular_momentum_error_ratio": row.get("initial_angular_momentum_error_ratio", ""),
            "insertion_windows": as_int(row.get("insertion_windows")),
            "successful_insertion_windows": as_int(row.get("successful_insertion_windows")),
            "predicted_crossing_step": row.get("predicted_crossing_step", ""),
            "first_crossing_step": row.get("first_crossing_step", ""),
            "timing_offset_steps": row.get("timing_offset_steps", ""),
            "crossing_vr_ratio": row.get("crossing_vr_ratio", ""),
            "crossing_vt_error_ratio": row.get("crossing_vt_error_ratio", ""),
            "sync_error_at_crossing": row.get("sync_error_at_crossing", ""),
            "min_distance_to_recoverable": row.get("min_distance_to_recoverable", ""),
            "best_vt_band": row.get("best_vt_band", ""),
            "best_sync_band": row.get("best_sync_band", ""),
            "recoverable_crossing": bool_from_csv(row.get("recoverable_crossing")),
            "capture_entered": bool_from_csv(row.get("capture_entered")),
            "success": bool_from_csv(row.get("success")),
            "overspeed": bool_from_csv(row.get("overspeed")),
            "instability": bool_from_csv(row.get("instability")) or str(row.get("termination_reason", "")) in {"overspeed", "out_range", "too_close", "radial_stall"},
        }
    )
    out.update(initial_features(row))

    target_radius = DEFAULT_TARGET_RADIUS * as_float(row.get("target_radius_scale"), 1.0)
    crossing_radius_error = as_float(row.get("crossing_radius_error"))
    crossing_r_ratio = abs(crossing_radius_error) / (target_radius + 1.0e-12) if math.isfinite(crossing_radius_error) else float("nan")
    if math.isfinite(crossing_r_ratio):
        out["crossing_radius_error_ratio"] = crossing_r_ratio
    vr_ratio = as_float(out["crossing_vr_ratio"])
    vt_ratio = as_float(out["crossing_vt_error_ratio"])
    if out["sync_error_at_crossing"] in {"", None} and math.isfinite(crossing_r_ratio) and math.isfinite(vr_ratio) and math.isfinite(vt_ratio):
        out["sync_error_at_crossing"] = sync_error(crossing_r_ratio, vr_ratio, vt_ratio)
    if out["min_distance_to_recoverable"] in {"", None} and math.isfinite(crossing_r_ratio) and math.isfinite(vr_ratio) and math.isfinite(vt_ratio):
        out["min_distance_to_recoverable"] = recoverability_distance(crossing_r_ratio, vr_ratio, vt_ratio)
    if math.isfinite(crossing_r_ratio) and math.isfinite(vr_ratio) and math.isfinite(vt_ratio):
        out["crossing_distance_to_recoverable"] = recoverability_distance(crossing_r_ratio, vr_ratio, vt_ratio)

    enrich = phase25.get((source_phase, cid), {})
    if enrich:
        if out["min_distance_to_recoverable"] in {"", None}:
            out["min_distance_to_recoverable"] = enrich.get("distance_to_recoverable", "")
        out["crossing_distance_to_recoverable"] = enrich.get("distance_to_recoverable", "")
        out["dominant_failure_variable"] = enrich.get("dominant_failure_variable", "")
        out["failure_cluster"] = enrich.get("failure_cluster", "")

    state = parse_state_json(row.get("pre_burn_state_json"))
    if state:
        out["early_r_error_ratio"] = state.get("r_error_ratio", "")
        out["early_vr_ratio"] = state.get("vr_ratio", "")
        out["early_vt_error_ratio"] = state.get("vt_error_ratio", "")
        out["early_energy_error_ratio"] = state.get("energy_error_ratio", "")
        out["early_angular_momentum_error_ratio"] = state.get("angular_momentum_error_ratio", "")
        out["early_feature_source"] = "pre_burn_state_json"
    else:
        out["early_feature_source"] = ""

    notes: List[str] = []
    if not state:
        notes.append("burn_a_end_or_pre_burn_state_missing")
    if out["sync_error_at_crossing"] in {"", None}:
        notes.append("sync_error_missing_without_crossing")
    if out["min_distance_to_recoverable"] in {"", None}:
        notes.append("distance_to_recoverable_missing_without_geometry")
    out["missing_feature_notes"] = ";".join(notes)
    out["family_label"] = family_label(out)
    out["window_quality_label"] = window_quality_label(out)
    out["quality_score"] = quality_score(out)
    return out


def is_near_recoverable(row: Dict[str, object]) -> bool:
    sync_value = as_float(row.get("sync_error_at_crossing"))
    distance = as_float(row.get("crossing_distance_to_recoverable"))
    return (math.isfinite(sync_value) and sync_value <= 3.0) or (math.isfinite(distance) and distance <= 3.0)


def family_label(row: Dict[str, object]) -> str:
    crossing = bool_from_csv(row.get("crossing_occurs")) or as_int(row.get("radius_crossings_total")) > 0 or row.get("first_crossing_step") not in {"", None}
    windows = as_int(row.get("insertion_windows")) > 0
    capture_success = bool_from_csv(row.get("capture_entered")) or bool_from_csv(row.get("success"))
    recoverable = bool_from_csv(row.get("recoverable_crossing"))
    if crossing and recoverable:
        return "near_recoverable_crossing"
    if crossing and is_near_recoverable(row):
        return "near_recoverable_crossing"
    if crossing:
        return "crossing_bad_sync"
    if capture_success:
        return "capture_success_existing"
    if windows and not crossing:
        return "window_no_crossing"
    return "dead_geometry"


def window_quality_label(row: Dict[str, object]) -> str:
    windows = as_int(row.get("insertion_windows")) > 0
    crossing = bool_from_csv(row.get("crossing_occurs")) or row.get("first_crossing_step") not in {"", None}
    if not windows:
        return "no_window"
    if crossing and is_near_recoverable(row):
        return "good_window"
    if crossing:
        return "dead_window_bad_crossing"
    return "dead_window_no_crossing"


def quality_score(row: Dict[str, object]) -> float:
    crossing = bool_from_csv(row.get("crossing_occurs")) or row.get("first_crossing_step") not in {"", None}
    recoverable = bool_from_csv(row.get("recoverable_crossing"))
    near = crossing and is_near_recoverable(row)
    score = 0.0
    score += 1.0 if crossing else 0.0
    score += 2.0 if near else 0.0
    score += 3.0 if recoverable else 0.0
    score += 4.0 if bool_from_csv(row.get("capture_entered")) else 0.0
    score += 5.0 if bool_from_csv(row.get("success")) else 0.0
    score -= 2.0 if bool_from_csv(row.get("overspeed")) else 0.0
    score -= 1.5 if bool_from_csv(row.get("instability")) else 0.0
    sync_value = as_float(row.get("sync_error_at_crossing"))
    if math.isfinite(sync_value):
        score -= min(4.0, max(0.0, sync_value - 1.0) * 0.5)
    return score


def load_family_dataset() -> List[Dict[str, object]]:
    phase25 = load_phase25_enrichment()
    rows: List[Dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for source_phase, path in PHASE_INPUTS:
        allowed = PHASE_CONTROLLER_FILTERS[source_phase]
        for raw in read_csv(path):
            controller = str(raw.get("controller_name", ""))
            if controller not in allowed:
                continue
            cid = case_id(raw)
            key = (source_phase, controller, cid)
            if key in seen:
                continue
            seen.add(key)
            rows.append(normalize_phase_row(source_phase, raw, phase25))
    return rows


def write_dataset(rows: Sequence[Dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with DATASET_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in DATASET_FIELDS})


def numeric_values(rows: Sequence[Dict[str, object]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = as_float(row.get(key))
        if math.isfinite(value):
            values.append(value)
    return values


def summarize_group(group_type: str, group_name: str, rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    family_counts = Counter(str(row.get("family_label", "")) for row in rows)
    failure_counts = Counter(str(row.get("dominant_failure_variable", "")) for row in rows if row.get("dominant_failure_variable"))
    return {
        "group_type": group_type,
        "group_name": group_name,
        "case_count": len(rows),
        "crossing_cases": sum(bool_from_csv(row.get("crossing_occurs")) or row.get("first_crossing_step") not in {"", None} for row in rows),
        "window_cases": sum(as_int(row.get("insertion_windows")) > 0 for row in rows),
        "good_window_cases": sum(row.get("window_quality_label") == "good_window" for row in rows),
        "dead_window_cases": sum(str(row.get("window_quality_label", "")).startswith("dead_window") for row in rows),
        "recoverable_crossings": sum(bool_from_csv(row.get("recoverable_crossing")) for row in rows),
        "capture": sum(bool_from_csv(row.get("capture_entered")) for row in rows),
        "success": sum(bool_from_csv(row.get("success")) for row in rows),
        "mean_quality_score": float(np.mean(numeric_values(rows, "quality_score"))) if numeric_values(rows, "quality_score") else "",
        "mean_sync_error_at_crossing": float(np.mean(numeric_values(rows, "sync_error_at_crossing"))) if numeric_values(rows, "sync_error_at_crossing") else "",
        "mean_min_distance_to_recoverable": float(np.mean(numeric_values(rows, "min_distance_to_recoverable"))) if numeric_values(rows, "min_distance_to_recoverable") else "",
        "dominant_family_label": family_counts.most_common(1)[0][0] if family_counts else "",
        "dominant_failure_variable": failure_counts.most_common(1)[0][0] if failure_counts else "",
    }


def write_summary_csv(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    summary_rows.append(summarize_group("all", "all", rows))
    for phase in sorted({str(row.get("source_phase")) for row in rows}):
        summary_rows.append(summarize_group("source_phase", phase, [row for row in rows if row.get("source_phase") == phase]))
    for controller in sorted({str(row.get("controller_name")) for row in rows}):
        summary_rows.append(summarize_group("controller", controller, [row for row in rows if row.get("controller_name") == controller]))
    for family in FAMILY_ORDER:
        summary_rows.append(summarize_group("family_label", family, [row for row in rows if row.get("family_label") == family]))
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})
    return summary_rows


def scatter_by_family(rows: Sequence[Dict[str, object]], x_key: str, y_key: str, path: Path, title: str, x_label: str, y_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    for family in FAMILY_ORDER:
        subset = [row for row in rows if row.get("family_label") == family]
        x = numeric_values(subset, x_key)
        y = numeric_values(subset, y_key)
        pairs = [
            (as_float(row.get(x_key)), as_float(row.get(y_key)))
            for row in subset
            if math.isfinite(as_float(row.get(x_key))) and math.isfinite(as_float(row.get(y_key)))
        ]
        if pairs:
            ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=42, alpha=0.78, color=FAMILY_COLORS[family], label=family)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_thrust_plot(rows: Sequence[Dict[str, object]]) -> None:
    thrusts = sorted({as_float(row.get("thrust_scale")) for row in rows if math.isfinite(as_float(row.get("thrust_scale")))})
    counts = {family: [] for family in FAMILY_ORDER}
    for thrust in thrusts:
        subset = [row for row in rows if as_float(row.get("thrust_scale")) == thrust]
        family_counts = Counter(str(row.get("family_label")) for row in subset)
        for family in FAMILY_ORDER:
            counts[family].append(family_counts.get(family, 0))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bottom = np.zeros(len(thrusts))
    for family in FAMILY_ORDER:
        values = np.asarray(counts[family])
        ax.bar([str(int(t)) for t in thrusts], values, bottom=bottom, color=FAMILY_COLORS[family], label=family)
        bottom += values
    ax.set_title("Thrust vs trajectory-family outcome")
    ax.set_xlabel("thrust scale")
    ax.set_ylabel("rows")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT_THRUST, dpi=220)
    plt.close(fig)


def save_sync_predicted_plot(rows: Sequence[Dict[str, object]]) -> None:
    scatter_by_family(
        rows,
        "predicted_crossing_step",
        "sync_error_at_crossing",
        PLOT_SYNC_PRED,
        "Sync error vs predicted crossing step",
        "predicted crossing step",
        "sync error at crossing",
    )


def save_window_quality_plot(rows: Sequence[Dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    pairs = [
        (as_int(row.get("insertion_windows")), as_float(row.get("quality_score")), str(row.get("window_quality_label")))
        for row in rows
        if math.isfinite(as_float(row.get("quality_score")))
    ]
    colors = {
        "no_window": "#9D755D",
        "dead_window_no_crossing": "#F58518",
        "dead_window_bad_crossing": "#E45756",
        "good_window": "#54A24B",
    }
    for label, color in colors.items():
        subset = [pair for pair in pairs if pair[2] == label]
        if subset:
            ax.scatter([pair[0] for pair in subset], [pair[1] for pair in subset], label=label, color=color, alpha=0.75, s=42)
    ax.set_title("Window count vs family quality")
    ax.set_xlabel("insertion window count")
    ax.set_ylabel("quality score")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_WINDOW_QUALITY, dpi=220)
    plt.close(fig)


def feature_correlations(rows: Sequence[Dict[str, object]]) -> List[tuple[str, float]]:
    features = [
        "r0_over_target",
        "initial_velocity_angle_deg",
        "thrust_scale",
        "initial_vr_ratio",
        "initial_vt_error_ratio",
        "initial_energy_error_ratio_reconstructed",
        "initial_angular_momentum_error_ratio_reconstructed",
        "initial_eccentricity_proxy",
        "reported_initial_energy_error_ratio",
        "reported_initial_angular_momentum_error_ratio",
        "insertion_windows",
        "predicted_crossing_step",
        "crossing_vr_ratio",
        "crossing_vt_error_ratio",
        "sync_error_at_crossing",
    ]
    scores = numeric_values(rows, "quality_score")
    results: List[tuple[str, float]] = []
    for feature in features:
        pairs = [
            (as_float(row.get(feature)), as_float(row.get("quality_score")))
            for row in rows
            if math.isfinite(as_float(row.get(feature))) and math.isfinite(as_float(row.get("quality_score")))
        ]
        if len(pairs) < 3:
            continue
        x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if math.isfinite(corr):
            results.append((feature, corr))
    return sorted(results, key=lambda item: abs(item[1]), reverse=True)


def save_feature_correlation_plot(rows: Sequence[Dict[str, object]]) -> List[tuple[str, float]]:
    correlations = feature_correlations(rows)
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    top = correlations[:10]
    if top:
        labels = [item[0] for item in top]
        values = [item[1] for item in top]
        colors = ["#54A24B" if value >= 0 else "#E45756" for value in values]
        ax.barh(labels[::-1], values[::-1], color=colors[::-1])
        ax.axvline(0.0, color="#333333", linewidth=0.8)
    else:
        ax.text(0.5, 0.5, "No stable numeric correlations available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Feature correlation with family quality score")
    ax.set_xlabel("Pearson correlation")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_IMPORTANCE, dpi=220)
    plt.close(fig)
    return correlations


def save_plots(rows: Sequence[Dict[str, object]]) -> List[tuple[str, float]]:
    scatter_by_family(
        rows,
        "initial_energy_error_ratio_reconstructed",
        "initial_angular_momentum_error_ratio_reconstructed",
        PLOT_ENERGY_H,
        "Initial energy vs angular momentum by family",
        "initial energy error ratio",
        "initial angular momentum error ratio",
    )
    scatter_by_family(
        rows,
        "r0_over_target",
        "initial_velocity_angle_deg",
        PLOT_ANGLE_R0,
        "Initial angle vs r0 by family",
        "r0 / target",
        "initial velocity angle [deg]",
    )
    save_thrust_plot(rows)
    scatter_by_family(
        rows,
        "crossing_vr_ratio",
        "crossing_vt_error_ratio",
        PLOT_CROSSING_V,
        "Crossing vt vs vr by family",
        "crossing vr ratio",
        "crossing vt error ratio",
    )
    save_sync_predicted_plot(rows)
    save_window_quality_plot(rows)
    return save_feature_correlation_plot(rows)


def best_controller_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_case: Dict[str, Dict[str, object]] = {}
    for row in rows:
        cid = str(row.get("case_id"))
        current = by_case.get(cid)
        if current is None or as_float(row.get("quality_score"), -999.0) > as_float(current.get("quality_score"), -999.0):
            by_case[cid] = row
    return list(by_case.values())


def format_float(value: object, digits: int = 4) -> str:
    value_f = as_float(value)
    return f"{value_f:.{digits}f}" if math.isfinite(value_f) else "nan"


def write_markdown(rows: Sequence[Dict[str, object]], summary_rows: Sequence[Dict[str, object]], correlations: Sequence[tuple[str, float]]) -> None:
    family_counts = Counter(str(row.get("family_label")) for row in rows)
    window_counts = Counter(str(row.get("window_quality_label")) for row in rows)
    window_only_counts = Counter(str(row.get("window_quality_label")) for row in rows if as_int(row.get("insertion_windows")) > 0)
    best_rows = best_controller_rows(rows)
    best_family_counts = Counter(str(row.get("family_label")) for row in best_rows)
    good_windows = [row for row in rows if row.get("window_quality_label") == "good_window"]
    dead_windows = [row for row in rows if str(row.get("window_quality_label", "")).startswith("dead_window")]
    crossing_rows = [row for row in rows if bool_from_csv(row.get("crossing_occurs")) or row.get("first_crossing_step") not in {"", None}]
    window_rows = [row for row in rows if as_int(row.get("insertion_windows")) > 0]
    best_corr = correlations[0] if correlations else ("none", float("nan"))
    useful_crossing_rows = [row for row in rows if row.get("family_label") in {"near_recoverable_crossing", "capture_success_existing"}]
    candidate_rule = "not supported"
    if useful_crossing_rows:
        angles = numeric_values(useful_crossing_rows, "initial_velocity_angle_deg")
        r0s = numeric_values(useful_crossing_rows, "r0_over_target")
        if angles and r0s:
            candidate_rule = f"angle {min(angles):.1f}-{max(angles):.1f}, r0 {min(r0s):.3f}-{max(r0s):.3f}"

    rows_by_controller = {row["group_name"]: row for row in summary_rows if row.get("group_type") == "controller"}
    lines = [
        "# Phase 28 Trajectory Family Quality Mapping",
        "",
        "## Scope",
        "",
        "- CSV-first analysis using Phase 22, 23, 24, 25, 26, and 27 outputs.",
        "- No controller reruns and no changes to physics, thresholds, CAPTURE, LOCK, reward, or prior outputs.",
        "- Burn-A-end state is only filled when prior phases recorded a compatible pre-burn state JSON; otherwise it is left blank.",
        "",
        "## Dataset",
        "",
        f"- Normalized rows: `{len(rows)}`.",
        f"- Unique initial cases: `{len({row.get('case_id') for row in rows})}`.",
        f"- Crossing rows: `{len(crossing_rows)}`.",
        f"- Window rows: `{len(window_rows)}`.",
        f"- Family label counts: `{json.dumps(dict(family_counts), sort_keys=True)}`.",
        f"- Window quality counts: `{json.dumps(dict(window_counts), sort_keys=True)}`.",
        f"- Best-controller family counts by case: `{json.dumps(dict(best_family_counts), sort_keys=True)}`.",
        "",
        "## Controller Summary",
        "",
        "| Controller | Rows | Crossings | Windows | Good windows | Dead windows | Mean quality | Mean sync |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for controller in sorted(rows_by_controller):
        item = rows_by_controller[controller]
        lines.append(
            f"| `{controller}` | {item['case_count']} | {item['crossing_cases']} | {item['window_cases']} | "
            f"{item['good_window_cases']} | {item['dead_window_cases']} | {format_float(item['mean_quality_score'])} | "
            f"{format_float(item['mean_sync_error_at_crossing'])} |"
        )
    lines.extend(
        [
            "",
            "## Research Answers",
            "",
            f"1. Are window-producing cases and crossing-producing cases the same family? `{'yes' if good_windows else 'no'}` under current data; window-producing rows are mostly `{window_only_counts.most_common(1)[0][0] if window_only_counts else 'none'}`.",
            f"2. Which early trajectory features best predict useful crossing? Best numeric correlation with quality is `{best_corr[0]}` at `{best_corr[1]:.3f}`; this is correlation, not proof.",
            f"3. Do insertion windows mostly represent good windows or dead windows? `{'good windows' if len(good_windows) > len(dead_windows) else 'dead windows'}`: good `{len(good_windows)}`, dead `{len(dead_windows)}`.",
            f"4. Is Burn A selecting the wrong trajectory family? `likely yes` for window-producing transfer rows because windows are usually not paired with useful crossings.",
            f"5. Is the current architecture near a 2D explicit-control ceiling? `plausibly yes`; Phase 27 shows Burn B timing has no leverage on crossing-producing cases.",
            f"6. What should Phase 29 test? A Burn-A family selector that targets crossing-producing initial manifolds before window generation, with instrumentation at Burn A end.",
            "",
            "## Candidate Family",
            "",
            f"- Candidate useful-family rule from observed useful rows: `{candidate_rule}`.",
            "- This is a descriptive range from historical rows, not a validated controller rule.",
            "",
            "## Honest Limitations",
            "",
            "- Burn A end geometry is mostly missing from prior CSVs, so Phase 28 cannot prove a causal Burn A rule.",
            "- Phase 25 provides reconstructed crossing-state geometry for Phase 22-24, but Phase 22/23 insertion-window states remain uninstrumented.",
            "- The strongest conclusion is family separation: window existence is not equivalent to useful crossing geometry.",
            "",
            "## Artifacts",
            "",
            "- `phase28_family_dataset.csv`",
            "- `phase28_family_summary.csv`",
            "- `family_energy_vs_angular_momentum.png`",
            "- `family_initial_angle_vs_r0.png`",
            "- `family_thrust_outcome.png`",
            "- `crossing_vt_vs_vr_by_family.png`",
            "- `sync_error_vs_predicted_crossing.png`",
            "- `window_count_vs_quality.png`",
            "- `feature_importance_or_correlation.png`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 28 trajectory family quality mapping.")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_family_dataset()
    write_dataset(rows)
    summary_rows = write_summary_csv(rows)
    correlations: List[tuple[str, float]] = []
    if not args.skip_plots:
        correlations = save_plots(rows)
    else:
        correlations = feature_correlations(rows)
    write_markdown(rows, summary_rows, correlations)
    print(f"Saved Phase 28 trajectory family mapping outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
