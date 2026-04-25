from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from scripts.explicit_controller_analysis_utils import DT, MAX_STEPS, SEED, angular_momentum, set_seed, specific_energy
from scripts.explicit_controller_phase4_regime_sweep import (
    DEFAULT_TARGET_RADIUS,
    make_regime_env,
    set_regime_start,
)


PHASE4_CSV = PROJECT_ROOT / "analysis" / "phase4_regime" / "regime_grid.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase5_reachability"
TRACE_DIR = OUTPUT_DIR / "traces"
FAILURE_MODES_PATH = OUTPUT_DIR / "failure_modes.json"
SUMMARY_PATH = PROJECT_ROOT / "analysis" / "phase5_reachability_summary.md"

ENERGY_PNG = OUTPUT_DIR / "energy_vs_time_cases.png"
RADIUS_ERROR_PNG = OUTPUT_DIR / "radius_error_vs_time_cases.png"
VR_PNG = OUTPUT_DIR / "v_r_vs_time_cases.png"
CLASSIFICATION_PNG = OUTPUT_DIR / "reachability_classification.png"
PIE_PNG = OUTPUT_DIR / "failure_modes_pie.png"

TRACE_KEYS = [
    "time",
    "radius_error",
    "v_r",
    "energy",
    "angular_momentum",
    "thrust_magnitude",
    "dE_dt",
    "phase_code",
]
PHASE_TO_CODE = {"DESCENT": 0, "CAPTURE": 1, "LOCK": 2}


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def phase4_class(row: Dict[str, str]) -> str:
    if as_bool(row["success"]):
        return "success"
    target_radius = as_float(row, "target_radius")
    min_abs = as_float(row, "minimum_abs_radius_error")
    if min_abs / target_radius <= 3.0e-5:
        return "near_miss"
    return "clear_failure"


def case_id(index: int, row: Dict[str, str]) -> str:
    return (
        f"case_{index:02d}_r0_{as_float(row, 'r0_over_target'):.5f}"
        f"_a_{as_float(row, 'initial_velocity_angle_deg'):.0f}"
        f"_th_{as_float(row, 'thrust_scale'):.0f}"
        f"_ts_{as_float(row, 'target_radius_scale'):.2f}"
    ).replace(".", "p")


def pick_diverse(rows: List[Dict[str, str]], count: int, *, prefer_low_error: bool) -> List[Dict[str, str]]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda row: as_float(row, "minimum_abs_radius_error"), reverse=not prefer_low_error)
    picked: List[Dict[str, str]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for row in sorted_rows:
        key = (row["initial_velocity_angle_deg"], row["thrust_scale"], row["target_radius_scale"])
        if key in seen_keys and len(picked) < max(1, count // 2):
            continue
        picked.append(row)
        seen_keys.add(key)
        if len(picked) >= count:
            return picked
    for row in sorted_rows:
        if row not in picked:
            picked.append(row)
        if len(picked) >= count:
            return picked
    return picked[:count]


def select_cases(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    for row in rows:
        row["phase5_source_class"] = phase4_class(row)

    successes = [row for row in rows if row["phase5_source_class"] == "success"]
    near_misses = [row for row in rows if row["phase5_source_class"] == "near_miss"]
    clear_failures = [row for row in rows if row["phase5_source_class"] == "clear_failure"]

    baseline = [
        row
        for row in successes
        if abs(as_float(row, "r0_over_target") - 1.00005) < 1.0e-9
        and abs(as_float(row, "initial_velocity_angle_deg") - 170.0) < 1.0e-9
        and abs(as_float(row, "thrust_scale") - 10000.0) < 1.0e-9
        and abs(as_float(row, "target_radius_scale") - 1.0) < 1.0e-9
    ]

    selected: List[Dict[str, str]] = []
    if baseline:
        selected.extend(baseline[:1])
    for row in pick_diverse([row for row in successes if row not in selected], 5, prefer_low_error=True):
        if row not in selected:
            selected.append(row)

    for row in pick_diverse(near_misses, 8, prefer_low_error=True):
        if row not in selected:
            selected.append(row)

    for row in pick_diverse(clear_failures, 6, prefer_low_error=False):
        if row not in selected:
            selected.append(row)

    if len(selected) < 20:
        for row in sorted(rows, key=lambda item: as_float(item, "minimum_abs_radius_error")):
            if row not in selected:
                selected.append(row)
            if len(selected) >= 20:
                break

    return selected[:20]


def rollout_trace(row: Dict[str, str]) -> Dict[str, object]:
    r0 = as_float(row, "r0_over_target")
    angle = as_float(row, "initial_velocity_angle_deg")
    thrust_scale = as_float(row, "thrust_scale")
    target_radius_scale = as_float(row, "target_radius_scale")

    env = make_regime_env(thrust_scale=thrust_scale, target_radius_scale=target_radius_scale)
    controller = OrbitLockController(target_radius=env.target_radius, mu=env.mu, config=OrbitLockConfig())
    set_regime_start(env, r0_over_target=r0, initial_velocity_angle_deg=angle)
    obs = np.asarray(env._get_obs(), dtype=np.float32)

    trace: Dict[str, List[float]] = {key: [] for key in TRACE_KEYS}
    phase_names: List[str] = []
    success = False
    terminated = False
    truncated = False
    capture_entered = False
    lock_entered = False

    while not (terminated or truncated):
        with contextlib.redirect_stdout(io.StringIO()):
            info = controller.act_with_info(obs)
        action = np.asarray(info["final_action"], dtype=np.float64)
        obs, _, terminated, truncated, step_info = env.step(action)
        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1.0e-12)
        phase = str(info.get("phase", controller.phase))
        phase_names.append(phase)
        capture_entered = capture_entered or phase == OrbitLockController.STATE_CAPTURE
        lock_entered = lock_entered or phase == OrbitLockController.STATE_LOCK
        trace["time"].append(float(env.steps * env.dt))
        trace["radius_error"].append(radius - env.target_radius)
        trace["v_r"].append(float(np.dot(env.vel, r_hat)))
        trace["energy"].append(specific_energy(env.mu, env.pos, env.vel))
        trace["angular_momentum"].append(angular_momentum(env.pos, env.vel))
        trace["thrust_magnitude"].append(float(np.linalg.norm(action) * env.thrust_scale))
        trace["phase_code"].append(float(PHASE_TO_CODE.get(phase, -1)))
        success = success or bool(step_info.get("success", False))
        obs = np.asarray(obs, dtype=np.float32)

    time = np.asarray(trace["time"], dtype=np.float64)
    energy = np.asarray(trace["energy"], dtype=np.float64)
    if len(time) > 1:
        dE_dt = np.gradient(energy, time)
    else:
        dE_dt = np.zeros_like(energy)
    trace["dE_dt"] = [float(x) for x in dE_dt]

    arrays = {key: np.asarray(trace[key], dtype=np.float64) for key in TRACE_KEYS}
    radius_error = arrays["radius_error"]
    abs_radius_error = np.abs(radius_error)
    min_idx = int(np.argmin(abs_radius_error)) if len(abs_radius_error) else 0
    target_energy = -float(env.mu) / (2.0 * float(env.target_radius))
    energy_error = np.abs(energy - target_energy)
    min_energy_error = float(np.min(energy_error)) if len(energy_error) else 0.0
    final_energy_error = float(abs(energy[-1] - target_energy)) if len(energy) else 0.0

    meta = {
        "r0_over_target": r0,
        "initial_velocity_angle_deg": angle,
        "thrust_scale": thrust_scale,
        "target_radius_scale": target_radius_scale,
        "target_radius": float(env.target_radius),
        "target_energy": target_energy,
        "source_class": row["phase5_source_class"],
        "success": bool(success),
        "capture_entered": bool(capture_entered),
        "lock_entered": bool(lock_entered),
        "steps": int(len(time)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "minimum_abs_radius_error": float(abs_radius_error[min_idx]) if len(abs_radius_error) else 0.0,
        "minimum_signed_radius_error": float(radius_error[min_idx]) if len(radius_error) else 0.0,
        "time_of_min_abs_radius_error": float(time[min_idx]) if len(time) else 0.0,
        "minimum_energy_error": min_energy_error,
        "final_energy_error": final_energy_error,
        "initial_energy_error": float(abs(energy[0] - target_energy)) if len(energy) else 0.0,
        "min_energy": float(np.min(energy)) if len(energy) else 0.0,
        "max_energy": float(np.max(energy)) if len(energy) else 0.0,
        "final_energy": float(energy[-1]) if len(energy) else 0.0,
        "mean_abs_dE_dt": float(np.mean(np.abs(dE_dt))) if len(dE_dt) else 0.0,
    }
    return {"arrays": arrays, "meta": meta, "phase_names": phase_names}


def classify_failure(meta: Dict[str, object], arrays: Dict[str, np.ndarray]) -> str:
    if bool(meta["success"]):
        return "success"
    target_radius = float(meta["target_radius"])
    target_energy = float(meta["target_energy"])
    radius_rel = float(meta["minimum_abs_radius_error"]) / target_radius
    initial_energy_error = max(float(meta["initial_energy_error"]), 1.0e-12)
    min_energy_error_ratio = float(meta["minimum_energy_error"]) / max(abs(target_energy), 1.0e-12)
    final_energy = float(meta["final_energy"])
    min_energy = float(meta["min_energy"])

    if radius_rel > 6.0e-5:
        return "geometry_miss"
    if min_energy_error_ratio > 2.0e-4 and float(meta["minimum_energy_error"]) > 0.45 * initial_energy_error:
        return "energy_limited"
    if min_energy < target_energy and final_energy < target_energy and np.nanmin(arrays["dE_dt"]) < -1.0e-3:
        return "overshoot_over_energy"
    return "timing_miss"


def save_trace_npz(path: Path, result: Dict[str, object], meta: Dict[str, object], failure_mode: str) -> None:
    arrays = result["arrays"]
    serial_meta = dict(meta)
    serial_meta["failure_mode"] = failure_mode
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(serial_meta, sort_keys=True), dtype=np.str_),
    )


def downsample(arr: np.ndarray, max_points: int = 4000) -> np.ndarray:
    if len(arr) <= max_points:
        return np.arange(len(arr))
    return np.unique(np.linspace(0, len(arr) - 1, max_points, dtype=int))


def label_for(meta: Dict[str, object]) -> str:
    return (
        f"{meta['failure_mode']}: r0={float(meta['r0_over_target']):.5f}, "
        f"a={float(meta['initial_velocity_angle_deg']):.0f}, th={float(meta['thrust_scale']):.0f}, "
        f"ts={float(meta['target_radius_scale']):.2f}"
    )


def plot_cases(path: Path, cases: List[Dict[str, object]], key: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    for case in cases:
        arrays = case["arrays"]
        meta = case["meta"]
        idx = downsample(arrays["time"])
        style = "-" if meta["failure_mode"] == "success" else "--"
        ax.plot(arrays["time"][idx] / 86400.0, arrays[key][idx], linestyle=style, linewidth=1.0, alpha=0.78, label=label_for(meta))
    ax.set_xlabel("simulation time [days]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_classification_plot(path: Path, records: List[Dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    colors = {
        "success": "#1B9E77",
        "energy_limited": "#7570B3",
        "overshoot_over_energy": "#D95F02",
        "geometry_miss": "#666666",
        "timing_miss": "#E7298A",
    }
    for mode in sorted({str(record["failure_mode"]) for record in records}):
        subset = [record for record in records if record["failure_mode"] == mode]
        ax.scatter(
            [float(record["minimum_abs_radius_error"]) for record in subset],
            [float(record["minimum_energy_error"]) for record in subset],
            s=58,
            label=mode,
            color=colors.get(mode, "#333333"),
            alpha=0.9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("minimum abs radius error [m]")
    ax.set_ylabel("minimum abs energy error [J/kg]")
    ax.set_title("Reachability classification of selected Phase 5 cases")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_failure_pie(path: Path, records: List[Dict[str, object]]) -> None:
    failures = [record for record in records if record["failure_mode"] != "success"]
    counts = Counter(str(record["failure_mode"]) for record in failures)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    if counts:
        ax.pie(counts.values(), labels=counts.keys(), autopct="%1.0f%%", startangle=90)
    else:
        ax.pie([1], labels=["no failures"], startangle=90)
    ax.set_title("Failure modes among selected failed cases")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_failure_modes(records: List[Dict[str, object]], selected: List[Dict[str, str]]) -> None:
    output = {
        "source_csv": str(PHASE4_CSV.relative_to(PROJECT_ROOT)),
        "selection_count": len(records),
        "selected_source_classes": Counter(row["phase5_source_class"] for row in selected),
        "failure_mode_counts": Counter(str(record["failure_mode"]) for record in records),
        "cases": records,
    }
    FAILURE_MODES_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")


def write_summary(records: List[Dict[str, object]]) -> None:
    failures = [record for record in records if record["failure_mode"] != "success"]
    counts = Counter(str(record["failure_mode"]) for record in failures)
    dominant_mode, dominant_count = counts.most_common(1)[0] if counts else ("none", 0)
    timing = counts.get("timing_miss", 0)
    geometry = counts.get("geometry_miss", 0)
    energy_limited = counts.get("energy_limited", 0)
    overshoot = counts.get("overshoot_over_energy", 0)
    capture_count = sum(bool(record["capture_entered"]) for record in records)
    success_count = sum(str(record["failure_mode"]) == "success" for record in records)

    near_failures = sorted(failures, key=lambda record: float(record["minimum_abs_radius_error"]))[:5]
    lines = [
        "# Phase 5 Reachability Analysis",
        "",
        "## Setup",
        "",
        "- Source data: `analysis/phase4_regime/regime_grid.csv`.",
        f"- Selected representative cases: `{len(records)}`.",
        "- Detailed traces saved under `analysis/phase5_reachability/traces/`.",
        "- Controller, environment physics, PPO, and learning experiments were not modified.",
        "",
        "## Classification Counts",
        "",
        f"- Success cases in selected set: `{success_count}`.",
        f"- CAPTURE entered in selected set: `{capture_count}`.",
        f"- `timing_miss`: `{timing}`.",
        f"- `geometry_miss`: `{geometry}`.",
        f"- `energy_limited`: `{energy_limited}`.",
        f"- `overshoot_over_energy`: `{overshoot}`.",
        "",
        "## Answers",
        "",
        f"1. Dominant failure mode: `{dominant_mode}` (`{dominant_count}` selected failures).",
        "2. Most selected failures fail before CAPTURE because the DESCENT phase does not deliver the spacecraft into the narrow radius/energy/velocity window needed by the phase transition. The controller applies a fixed retrograde descent law, so parameter changes alter closest-approach timing and energy state without any adaptive targeting.",
        "3. The capture window is defined operationally by reaching the target-radius neighborhood with compatible radial velocity and orbital energy. In this implementation, CAPTURE itself is triggered by a radius-error sign crossing, while strict success also requires sustained radius, speed, and angle tolerances.",
        "4. The missing capability is regime-adaptive descent targeting: the controller lacks a way to modulate energy removal and crossing timing across thrust, initial velocity angle, radius offset, and target-radius scale. It can stabilize once the right window is reached, but it does not reliably steer into that window.",
        "",
        "## Closest Failed Cases",
        "",
    ]
    for record in near_failures:
        lines.append(
            f"- `{record['case_id']}`: mode `{record['failure_mode']}`, "
            f"min abs radius error `{float(record['minimum_abs_radius_error']):.3e}` m, "
            f"min energy error `{float(record['minimum_energy_error']):.3e}` J/kg, "
            f"CAPTURE `{record['capture_entered']}`, LOCK `{record['lock_entered']}`."
        )
    lines.extend(
        [
            "",
            "## Caution",
            "",
            "This is a selected-case reachability analysis, not a new controller validation sweep. It explains representative Phase 4 failures and should be read together with the full Phase 4 grid.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_csv_dicts(PHASE4_CSV)
    selected = select_cases(rows)
    detailed_cases: List[Dict[str, object]] = []
    records: List[Dict[str, object]] = []

    for idx, row in enumerate(selected, start=1):
        cid = case_id(idx, row)
        result = rollout_trace(row)
        meta = result["meta"]
        failure_mode = classify_failure(meta, result["arrays"])
        meta["failure_mode"] = failure_mode
        meta["case_id"] = cid
        trace_path = TRACE_DIR / f"{cid}.npz"
        save_trace_npz(trace_path, result, meta, failure_mode)
        detailed_cases.append({"arrays": result["arrays"], "meta": meta})
        record = dict(meta)
        record["trace_path"] = str(trace_path.relative_to(PROJECT_ROOT))
        records.append(record)
        print(
            f"phase5 {idx}/{len(selected)} {cid} source={meta['source_class']} mode={failure_mode} "
            f"success={meta['success']} capture={meta['capture_entered']} "
            f"min_abs_rerr={meta['minimum_abs_radius_error']:.3e}"
        )

    plot_cases(ENERGY_PNG, detailed_cases, "energy", "Specific energy vs time for selected Phase 5 cases", "specific energy [J/kg]")
    plot_cases(
        RADIUS_ERROR_PNG,
        detailed_cases,
        "radius_error",
        "Radius error vs time for selected Phase 5 cases",
        "radius - target [m]",
    )
    plot_cases(VR_PNG, detailed_cases, "v_r", "Radial velocity vs time for selected Phase 5 cases", "v_r [m/s]")
    save_classification_plot(CLASSIFICATION_PNG, records)
    save_failure_pie(PIE_PNG, records)
    write_failure_modes(records, selected)
    write_summary(records)
    print(f"Saved Phase 5 reachability outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
