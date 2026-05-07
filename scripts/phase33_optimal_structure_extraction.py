from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase33_optimal_structure_extraction"
MPLCONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.explicit_controller_phase21_orbital_transfer_planner import (
    DT,
    RECOVERABLE_VR_RATIO,
    RECOVERABLE_VT_RATIO,
    RECOVERABLE_R_RATIO,
    TARGET_RADIUS_SCALE,
    bool_from_csv,
)
from scripts.explicit_controller_phase22_two_burn_transfer import rollout_phase22_case


PHASE32_DIR = PROJECT_ROOT / "analysis" / "phase32_direct_optimal_control"
PHASE31_DIR = PROJECT_ROOT / "analysis" / "phase31_global_transfer_solver"
PHASE32_RESULTS = PHASE32_DIR / "phase32_results.csv"
PHASE32_TRAJ = PHASE32_DIR / "phase32_trajectory_dataset.csv"
PHASE31_RESULTS = PHASE31_DIR / "phase31_results.csv"

BEST_CASE_MD = OUTPUT_DIR / "best_case_selection.md"
DECOMP_MD = OUTPUT_DIR / "structure_decomposition.md"
GAP_MD = OUTPUT_DIR / "architecture_gap_map.md"
SUMMARY_MD = OUTPUT_DIR / "phase33_summary.md"
METRICS_CSV = OUTPUT_DIR / "phase33_metrics.csv"

PLOT_RADIUS = OUTPUT_DIR / "radius_vs_time_overlay.png"
PLOT_VR = OUTPUT_DIR / "vr_vs_time_overlay.png"
PLOT_VT = OUTPUT_DIR / "vt_error_vs_time_overlay.png"
PLOT_SYNC = OUTPUT_DIR / "sync_error_overlay.png"
PLOT_THRUST = OUTPUT_DIR / "thrust_profile_overlay.png"
PLOT_SEG = OUTPUT_DIR / "control_phase_segmentation.png"


METRIC_FIELDS = [
    "objective_mode",
    "case_label",
    "selected_role",
    "crossing_step",
    "crossing_sync_error",
    "crossing_distance_to_recoverable",
    "crossing_vr_ratio",
    "crossing_vt_error_ratio",
    "best_sync_step",
    "best_sync_error",
    "best_distance_to_recoverable",
    "best_state_relation_to_crossing",
    "final_r_error_ratio",
    "final_vr_ratio",
    "final_vt_error_ratio",
    "vr_flip_count",
    "first_vr_flip_step",
    "vt_min_step",
    "vt_min_before_crossing",
    "mean_control_norm",
    "max_control_norm",
    "mean_control_smoothness",
    "motif_summary",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: object, default: float = float("nan")) -> float:
    if value in {"", None}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = -1) -> int:
    value_f = as_float(value)
    if not math.isfinite(value_f):
        return default
    return int(value_f)


def sign(value: float, eps: float = 1.0e-9) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def sync_from_row(row: Dict[str, object]) -> float:
    return max(
        abs(as_float(row.get("r_error_ratio"))) / RECOVERABLE_R_RATIO,
        abs(as_float(row.get("vr_ratio"))) / RECOVERABLE_VR_RATIO,
        abs(as_float(row.get("vt_error_ratio"))) / RECOVERABLE_VT_RATIO,
    )


def select_best_cases(results: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    numeric = [row for row in results if math.isfinite(as_float(row.get("best_distance_to_recoverable")))]
    best_distance = min(numeric, key=lambda row: as_float(row["best_distance_to_recoverable"]))
    recoverable_crossings = [row for row in numeric if bool_from_csv(row.get("recoverable_crossing"))]
    recoverable_states = [row for row in numeric if bool_from_csv(row.get("recoverable_state"))]
    recoverability_target = [row for row in numeric if row.get("objective_mode") == "recoverability_target"]
    crossing_near = [
        row
        for row in numeric
        if bool_from_csv(row.get("crossing_occurs"))
        and (bool_from_csv(row.get("near_recoverable_crossing")) or bool_from_csv(row.get("recoverable_crossing")))
    ]
    return {
        "lowest_distance": best_distance,
        "best_recoverable_crossing": min(recoverable_crossings, key=lambda row: as_float(row["best_distance_to_recoverable"])),
        "best_recoverable_state": min(recoverable_states, key=lambda row: as_float(row["best_distance_to_recoverable"])),
        "best_recoverability_target": min(recoverability_target, key=lambda row: as_float(row["best_distance_to_recoverable"])),
        "best_crossing_near": min(crossing_near, key=lambda row: as_float(row["best_distance_to_recoverable"])),
    }


def trajectory_for(traj_rows: Sequence[Dict[str, str]], objective_mode: str, case_label: str) -> List[Dict[str, object]]:
    rows = [dict(row) for row in traj_rows if row["objective_mode"] == objective_mode and row["case_label"] == case_label]
    rows.sort(key=lambda row: as_int(row["step"]))
    return rows


def vr_flip_steps(traj: Sequence[Dict[str, object]]) -> List[int]:
    flips: List[int] = []
    prev = sign(as_float(traj[0]["vr_ratio"]))
    for row in traj[1:]:
        current = sign(as_float(row["vr_ratio"]))
        if current and prev and current != prev:
            flips.append(as_int(row["step"]))
        if current:
            prev = current
    return flips


def best_sync_step(traj: Sequence[Dict[str, object]]) -> int:
    return as_int(min(traj, key=lambda row: as_float(row["sync_error"]))["step"])


def control_smoothness(traj: Sequence[Dict[str, object]]) -> float:
    controls = np.asarray([[as_float(row["control_x"]), as_float(row["control_y"])] for row in traj], dtype=np.float64)
    if len(controls) < 2:
        return 0.0
    return float(np.mean(np.sum(np.diff(controls, axis=0) ** 2, axis=1)))


def phase_segments(traj: Sequence[Dict[str, object]], crossing_step: int, best_step: int) -> List[tuple[str, int, int]]:
    n = as_int(traj[-1]["step"])
    flips = vr_flip_steps(traj)
    first_flip = flips[0] if flips else max(1, int(0.15 * n))
    if crossing_step >= 0:
        prep_start = max(first_flip + 1, crossing_step - 19)
        cross_end = min(n, crossing_step + 5)
        if best_step > cross_end:
            return [
                ("Initial geometry shaping", 0, min(first_flip, n)),
                ("Radial approach", min(first_flip + 1, n), min(prep_start - 1, n)),
                ("Crossing preparation", min(prep_start, n), min(crossing_step - 1, n)),
                ("Crossing", min(crossing_step, n), cross_end),
                ("Energy / tangential correction", min(cross_end + 1, n), min(best_step - 1, n)),
                ("Post-cross stabilization", min(best_step, n), n),
            ]
        return [
            ("Initial geometry shaping", 0, min(first_flip, n)),
            ("Energy / tangential correction", min(first_flip + 1, n), min(best_step, n)),
            ("Radial approach", min(best_step + 1, n), min(prep_start - 1, n)),
            ("Crossing preparation", min(prep_start, n), min(crossing_step - 1, n)),
            ("Crossing", min(crossing_step, n), cross_end),
            ("Post-cross stabilization", min(cross_end + 1, n), n),
        ]
    return [
        ("Initial geometry shaping", 0, min(first_flip, n)),
        ("Energy / tangential correction", min(first_flip + 1, n), min(best_step, n)),
        ("Radial approach", min(best_step + 1, n), max(min(best_step + int(0.20 * n), n), min(best_step + 1, n))),
        ("Crossing preparation", max(0, best_step - 20), max(0, best_step - 1)),
        ("Crossing", best_step, min(n, best_step + 5)),
        ("Post-cross stabilization", min(best_step + 6, n), n),
    ]


def summarize_segment(name: str, rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "phase": name,
            "radius_trend": "empty",
            "vr_sign": "empty",
            "vt_start": "",
            "vt_end": "",
            "mean_thrust": "",
            "mean_angle_deg": "",
            "smoothness": "",
        }
    r0 = as_float(rows[0]["radius_ratio"])
    r1 = as_float(rows[-1]["radius_ratio"])
    vt0 = as_float(rows[0]["vt_error_ratio"])
    vt1 = as_float(rows[-1]["vt_error_ratio"])
    controls = np.asarray([[as_float(row["control_x"]), as_float(row["control_y"])] for row in rows], dtype=np.float64)
    norms = np.linalg.norm(controls, axis=1)
    nonzero = controls[norms > 1.0e-8]
    if len(nonzero):
        angles = np.degrees(np.arctan2(nonzero[:, 1], nonzero[:, 0]))
        mean_angle = float(np.mean(angles))
    else:
        mean_angle = 0.0
    vr_signs = [sign(as_float(row["vr_ratio"])) for row in rows if sign(as_float(row["vr_ratio"])) != 0]
    if vr_signs:
        pos = vr_signs.count(1)
        neg = vr_signs.count(-1)
        vr_desc = "mostly outward" if pos > neg else "mostly inward"
    else:
        vr_desc = "near zero"
    return {
        "phase": name,
        "radius_trend": "increasing" if r1 > r0 else "decreasing" if r1 < r0 else "flat",
        "vr_sign": vr_desc,
        "vt_start": vt0,
        "vt_end": vt1,
        "mean_thrust": float(np.mean(norms)),
        "mean_angle_deg": mean_angle,
        "smoothness": control_smoothness(rows),
    }


def phase31_baseline_trajectory(best_case: Dict[str, str]) -> List[Dict[str, object]]:
    r0 = as_float(best_case["r0_over_target"])
    angle = as_float(best_case["initial_velocity_angle_deg"])
    thrust = as_float(best_case["thrust_scale"])
    row = rollout_phase22_case(r0, angle, thrust, TARGET_RADIUS_SCALE, record_trajectory=True)
    traj = row["trajectory"]
    out: List[Dict[str, object]] = []
    for idx, step in enumerate(traj["time_step"]):
        r_ratio = as_float(traj["radius"][idx]) / as_float(row["target_radius"])
        proxy_thrust = 0.0 if str(traj["stage"][idx]) in {"coast_arc", "abort_safe_coast"} else 1.0
        out.append(
            {
                "step": int(step),
                "time_seconds": int(step) * 100.0,
                "radius_ratio": r_ratio,
                "r_error_ratio": r_ratio - 1.0,
                "vr_ratio": as_float(traj["radial_velocity_ratio"][idx]),
                "vt_error_ratio": as_float(traj["vt_error_ratio"][idx]),
                "sync_error": sync_from_row(
                    {
                        "r_error_ratio": r_ratio - 1.0,
                        "vr_ratio": traj["radial_velocity_ratio"][idx],
                        "vt_error_ratio": traj["vt_error_ratio"][idx],
                    }
                ),
                "control_norm": proxy_thrust,
            }
        )
    return out


def plot_overlay(phase31: Sequence[Dict[str, object]], phase32: Sequence[Dict[str, object]], key: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot([as_float(row["time_seconds"]) / 3600.0 for row in phase31], [as_float(row[key]) for row in phase31], label="Phase31 baseline", linewidth=1.2)
    ax.plot([as_float(row["time_seconds"]) / 3600.0 for row in phase32], [as_float(row[key]) for row in phase32], label="Phase32 optimal", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("time [hours]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_phase_plots(phase31: Sequence[Dict[str, object]], phase32: Sequence[Dict[str, object]], segments: Sequence[tuple[str, int, int]]) -> None:
    plot_overlay(phase31, phase32, "radius_ratio", "radius / target", "Radius vs time overlay", PLOT_RADIUS)
    plot_overlay(phase31, phase32, "vr_ratio", "vr / circular speed", "Radial velocity vs time overlay", PLOT_VR)
    plot_overlay(phase31, phase32, "vt_error_ratio", "vt error / circular speed", "Tangential velocity error vs time overlay", PLOT_VT)
    plot_overlay(phase31, phase32, "sync_error", "sync error", "Sync error overlay", PLOT_SYNC)
    plot_overlay(phase31, phase32, "control_norm", "control norm / stage proxy", "Thrust profile overlay", PLOT_THRUST)

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.2), sharex=True)
    time_hours = [as_float(row["time_seconds"]) / 3600.0 for row in phase32]
    axes[0].plot(time_hours, [as_float(row["radius_ratio"]) for row in phase32], color="#4C78A8")
    axes[0].set_ylabel("r / target")
    axes[1].plot(time_hours, [as_float(row["vr_ratio"]) for row in phase32], color="#F58518")
    axes[1].set_ylabel("vr ratio")
    axes[2].plot(time_hours, [as_float(row["vt_error_ratio"]) for row in phase32], color="#54A24B")
    axes[2].set_ylabel("vt err")
    axes[3].plot(time_hours, [as_float(row["control_norm"]) for row in phase32], color="#E45756")
    axes[3].set_ylabel("|u|")
    axes[3].set_xlabel("time [hours]")
    colors = ["#E6F2FF", "#FFF0DE", "#EAF7EA", "#FDECEC", "#EFEAF7", "#EEF7F7"]
    for idx, (name, start, end) in enumerate(segments):
        x0 = start * 100.0 / 3600.0
        x1 = end * 100.0 / 3600.0
        for ax in axes:
            ax.axvspan(x0, x1, color=colors[idx % len(colors)], alpha=0.55)
        axes[0].text((x0 + x1) / 2.0, axes[0].get_ylim()[1], name.split()[0], ha="center", va="top", fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Phase32 control phase segmentation")
    fig.tight_layout()
    fig.savefig(PLOT_SEG, dpi=220)
    plt.close(fig)


def write_best_case_md(selected: Dict[str, Dict[str, str]]) -> None:
    lines = ["# Phase 33 Best Case Selection", ""]
    for role, row in selected.items():
        lines.extend(
            [
                f"## {role.replace('_', ' ').title()}",
                "",
                f"- Objective mode: `{row['objective_mode']}`",
                f"- Case label: `{row['case_label']}`",
                f"- Crossing step: `{row.get('first_crossing_step') or 'none'}`",
                f"- Best recoverability step: `{row.get('best_state_step', 'not evaluated here')}`",
                f"- Best sync: `{as_float(row['best_sync_error']):.6f}`",
                f"- Best distance: `{as_float(row['best_distance_to_recoverable']):.6f}`",
                f"- Crossing-state sync: `{as_float(row.get('crossing_sync_error')):.6f}`",
                f"- Crossing-state distance: `{as_float(row.get('crossing_distance_to_recoverable')):.6f}`",
                f"- Recoverable state: `{row.get('recoverable_state')}`",
                f"- Recoverable crossing: `{row.get('recoverable_crossing')}`",
                f"- Final r error ratio: `{as_float(row['final_r_error_ratio']):.8f}`",
                f"- Final vr ratio: `{as_float(row['final_vr_ratio']):.8f}`",
                f"- Final vt error ratio: `{as_float(row['final_vt_error_ratio']):.8f}`",
                "",
            ]
        )
    BEST_CASE_MD.write_text("\n".join(lines), encoding="utf-8")


def write_decomposition_md(best: Dict[str, str], segments: Sequence[tuple[str, int, int]], summaries: Sequence[Dict[str, object]], flips: Sequence[int], vt_min_step: int, crossing_step: int) -> None:
    best_step = as_int(best.get("best_state_step"), -1)
    lines = [
        "# Phase 33 Structural Trajectory Decomposition",
        "",
        f"Best trajectory: `{best['objective_mode']}` / `{best['case_label']}`.",
        "",
        "## Phase Segments",
        "",
        "| Phase | Step range | Radius trend | vr sign | vt start | vt end | Mean thrust | Mean thrust angle | Smoothness |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for (name, start, end), summary in zip(segments, summaries):
        lines.append(
            f"| {name} | {start}-{end} | {summary['radius_trend']} | {summary['vr_sign']} | "
            f"{as_float(summary['vt_start']):.6f} | {as_float(summary['vt_end']):.6f} | "
            f"{as_float(summary['mean_thrust']):.6f} | {as_float(summary['mean_angle_deg']):.2f} | {as_float(summary['smoothness']):.8f} |"
        )
    vt_relation = "before crossing" if crossing_step >= 0 and vt_min_step < crossing_step else "after crossing or no crossing"
    best_relation = "after first crossing" if crossing_step >= 0 and best_step > crossing_step else "at or before first crossing"
    lines.extend(
        [
            "",
            "## Timing Facts",
            "",
            f"- vr sign flip steps: `{list(flips)}`.",
            f"- vr flip count: `{len(flips)}`.",
            f"- crossing step: `{crossing_step if crossing_step >= 0 else 'none'}`.",
            f"- best recoverability step: `{best_step if best_step >= 0 else 'unknown'}`.",
            f"- best recoverability occurs `{best_relation}`.",
            f"- minimum |vt error| step: `{vt_min_step}`.",
            f"- vt is minimized `{vt_relation}`.",
            f"- crossing-state sync error: `{as_float(best.get('crossing_sync_error')):.6f}`.",
            f"- crossing-state distance to recoverable: `{as_float(best.get('crossing_distance_to_recoverable')):.6f}`.",
            "",
            "## Extracted Behavior",
            "",
            "- The optimal trajectory uses very small continuous thrust rather than large impulse-like burns.",
            "- It crosses target radius early, but the crossing state is still outside the true recoverability basin.",
            "- It then uses a long post-cross correction arc to bring vt, vr, and radius into simultaneous alignment.",
            "- The control law is smooth and low authority; the important structure is timing and phase alignment, not thrust magnitude.",
            "",
            "## Implementation Note",
            "",
            "- Phase 32 labels `recoverable_crossing` when a trajectory both crosses target radius and reaches a recoverable state somewhere on the horizon.",
            "- For this best case, the first crossing itself is not the best recoverability state; Phase 33 therefore treats the first crossing and late recoverable endpoint separately.",
        ]
    )
    DECOMP_MD.write_text("\n".join(lines), encoding="utf-8")


def write_gap_map() -> None:
    lines = [
        "# Phase 33 Architecture Gap Map",
        "",
        "## Phase 7.6",
        "",
        "- had: phase-structured pre-window shaping, window seeking, CAPTURE, and LOCK.",
        "- lacked: continuous trajectory-wide optimization of vt/vr/r synchronization.",
        "- lacked: explicit permission to let vt error temporarily grow if it improves late recoverability.",
        "",
        "## Phase 20",
        "",
        "- had: predictive local candidate scoring.",
        "- lacked: global state-control trajectory optimization.",
        "- lacked: a mechanism for smooth long-horizon control motifs.",
        "",
        "## Phase 31",
        "",
        "- had: named global transfer families with burn/coast schedules.",
        "- lacked: continuous adjustment of every control interval.",
        "- lacked: the low-thrust smooth arc discovered by Phase 32.",
        "",
        "## Phase 32",
        "",
        "- introduced: finite-horizon direct control optimization against recoverability.",
        "- introduced: smooth low-authority steering as an upper-bound behavior.",
        "- introduced: evidence that recoverability can be reached as a state and, in one representative case, as a trajectory that also crosses target radius.",
        "- did not prove that the first radius-crossing state itself is recoverable.",
    ]
    GAP_MD.write_text("\n".join(lines), encoding="utf-8")


def write_summary(best: Dict[str, str], metrics: Dict[str, object]) -> None:
    lines = [
        "# Phase 33 Optimal Structure Extraction Summary",
        "",
        "## Findings",
        "",
        "1. What structural behavior made optimal better? Smooth low-thrust synchronization of radius, vr, and vt over the full horizon, rather than discrete Burn A/B corrections.",
        "2. Was recoverability mainly timing, geometry, or control smoothness? Primarily geometry-time synchronization; smoothness is the mechanism that prevents corrections from destroying another state component.",
        "3. What architecture limit blocked prior phases? Prior phases treated crossing and insertion as staged events, while the optimal trajectory keeps steering after the first crossing until the full recoverability state aligns.",
        "4. Which missing structure matters most? A continuous low-authority post-cross steering arc that can trade temporary crossing quality for late basin entry.",
        "5. Can this be approximated heuristically? Partially, with an MPC-lite or imitation controller trained on optimal trajectories; a hand heuristic would be fragile.",
        "6. Should Phase 34 be imitation controller, MPC-lite, CasADi full collocation, or hybrid heuristic-optimal? `hybrid heuristic-optimal`: first imitate Phase 32 motifs, then validate with MPC-lite or CasADi collocation.",
        "",
        "## Best Case",
        "",
        f"- Objective mode: `{best['objective_mode']}`",
        f"- Case label: `{best['case_label']}`",
        f"- Crossing step: `{best.get('first_crossing_step')}`",
        f"- Best sync: `{as_float(best['best_sync_error']):.6f}`",
        f"- Best distance: `{as_float(best['best_distance_to_recoverable']):.6f}`",
        "",
        "## Honesty Note",
        "",
        "- Phase 32 used SciPy direct shooting because CasADi was unavailable in the checked runtime.",
        "- The optimal advantage is structural for the representative recoverable case, but it is not yet a production-controller result.",
        "- The best Phase 32 row is crossing-labeled because it both crosses and later reaches recoverability; the first crossing state remains outside the recoverability basin.",
        "- Phase31 baseline thrust profile was not logged; the overlay uses the Phase22/31 baseline stage activity as a control proxy.",
        "",
        "## Phase 34 Blueprint",
        "",
        "- Extract state-control pairs from Phase 32 trajectories.",
        "- Train or hand-fit a low-thrust smooth steering policy around recoverability-targeted state errors.",
        "- Add an MPC-lite layer that optimizes short receding windows for sync error, not crossing alone.",
        "- Validate against Phase31 baseline on the reduced grid before attempting broader generalization.",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def write_metrics_csv(rows: Sequence[Dict[str, object]]) -> None:
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = read_csv(PHASE32_RESULTS)
    traj_rows = read_csv(PHASE32_TRAJ)
    selected = select_best_cases(results)
    best = selected["best_recoverable_crossing"]
    best_traj = trajectory_for(traj_rows, best["objective_mode"], best["case_label"])
    crossing_step = as_int(best.get("first_crossing_step"), -1)
    best_step = best_sync_step(best_traj)
    best["best_state_step"] = str(best_step)
    crossing_row = next((row for row in best_traj if as_int(row["step"]) == crossing_step), None)
    if crossing_row:
        best["crossing_sync_error"] = str(as_float(crossing_row["sync_error"]))
        best["crossing_distance_to_recoverable"] = str(as_float(crossing_row["distance_to_recoverable"]))
        best["crossing_vr_ratio"] = str(as_float(crossing_row["vr_ratio"]))
        best["crossing_vt_error_ratio"] = str(as_float(crossing_row["vt_error_ratio"]))
    else:
        best["crossing_sync_error"] = ""
        best["crossing_distance_to_recoverable"] = ""
        best["crossing_vr_ratio"] = ""
        best["crossing_vt_error_ratio"] = ""
    flips = vr_flip_steps(best_traj)
    vt_min_step = as_int(min(best_traj, key=lambda row: abs(as_float(row["vt_error_ratio"])))["step"])
    segments = phase_segments(best_traj, crossing_step, best_step)
    segment_summaries = [
        summarize_segment(name, [row for row in best_traj if start <= as_int(row["step"]) <= end])
        for name, start, end in segments
    ]
    phase31 = phase31_baseline_trajectory(best)
    save_phase_plots(phase31, best_traj, segments)
    write_best_case_md(selected)
    write_decomposition_md(best, segments, segment_summaries, flips, vt_min_step, crossing_step)
    write_gap_map()
    metrics_row = {
        "objective_mode": best["objective_mode"],
        "case_label": best["case_label"],
        "selected_role": "best_recoverable_crossing",
        "crossing_step": crossing_step,
        "crossing_sync_error": best["crossing_sync_error"],
        "crossing_distance_to_recoverable": best["crossing_distance_to_recoverable"],
        "crossing_vr_ratio": best["crossing_vr_ratio"],
        "crossing_vt_error_ratio": best["crossing_vt_error_ratio"],
        "best_sync_step": best_step,
        "best_sync_error": best["best_sync_error"],
        "best_distance_to_recoverable": best["best_distance_to_recoverable"],
        "best_state_relation_to_crossing": "after_first_crossing" if crossing_step >= 0 and best_step > crossing_step else "at_or_before_first_crossing",
        "final_r_error_ratio": best["final_r_error_ratio"],
        "final_vr_ratio": best["final_vr_ratio"],
        "final_vt_error_ratio": best["final_vt_error_ratio"],
        "vr_flip_count": len(flips),
        "first_vr_flip_step": flips[0] if flips else "",
        "vt_min_step": vt_min_step,
        "vt_min_before_crossing": bool(crossing_step >= 0 and vt_min_step < crossing_step),
        "mean_control_norm": float(np.mean([as_float(row["control_norm"]) for row in best_traj])),
        "max_control_norm": float(np.max([as_float(row["control_norm"]) for row in best_traj])),
        "mean_control_smoothness": control_smoothness(best_traj),
        "motif_summary": "smooth low-thrust full-horizon synchronization",
    }
    write_metrics_csv([metrics_row])
    write_summary(best, metrics_row)
    print(f"Saved Phase 33 optimal structure extraction outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
