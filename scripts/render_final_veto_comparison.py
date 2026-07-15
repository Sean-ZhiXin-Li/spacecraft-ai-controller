from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXPERIMENT_ID = "final_veto_overspeed_ablation_v0"
PRESERVATION_SUBSET_ID = "phase34_known_recoverable_preservation_v1"
STRESS_SUBSET_ID = "phase35_radial_energy_push_overspeed_stress_v0"
FORMAL_OUTPUT_DIRECTORY = PROJECT_ROOT / "analysis" / "final_veto_ablation_v0"
FORMAL_RESULTS = FORMAL_OUTPUT_DIRECTORY / "results.csv"
FORMAL_PAIRED_RESULTS = FORMAL_OUTPUT_DIRECTORY / "paired_results.csv"
FORMAL_COMPARISON = FORMAL_OUTPUT_DIRECTORY / "comparison.png"
FORMAL_REQUIRED_BEFORE_PLOT = (
    FORMAL_OUTPUT_DIRECTORY / "manifest.json",
    FORMAL_RESULTS,
    FORMAL_PAIRED_RESULTS,
    FORMAL_OUTPUT_DIRECTORY / "decision_log.jsonl",
    FORMAL_OUTPUT_DIRECTORY / "summary.md",
)
PROTECTED_HISTORICAL_DIRECTORIES = (
    PROJECT_ROOT / "analysis" / "phase34_post_cross_sync",
    PROJECT_ROOT / "analysis" / "phase35_crossing_basin_expansion",
    PROJECT_ROOT / "analysis" / "phase36b_transfer_family_benchmark",
    PROJECT_ROOT / "analysis" / "phase36c_non_crossing_geometry_diagnosis",
    PROJECT_ROOT / "analysis" / "phase37a_radial_commit_timing",
    PROJECT_ROOT / "analysis" / "phase37b_weak_tangential_subset",
)
FIGURE_WIDTH_INCHES = 16.0
FIGURE_HEIGHT_INCHES = 10.0
FIGURE_DPI = 150
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

ARM_REQUIRED_FIELDS = frozenset(
    {
        "experiment_id",
        "run_id",
        "paired_run_id",
        "subset_id",
        "case_id",
        "arm_id",
        "r0_over_target",
        "initial_velocity_angle_deg",
        "thrust_scale",
        "crossed_target_radius",
        "recoverable_crossing",
        "final_simulator_success",
        "overspeed",
        "invalid_simulation",
        "terminal_label",
        "termination_reason",
        "monitor_evaluation_count",
        "veto_count",
        "intervention_rate",
        "longest_consecutive_veto_steps",
        "steps",
    }
)
PAIR_REQUIRED_FIELDS = frozenset(
    {
        "experiment_id",
        "paired_run_id",
        "case_id",
        "subset_id",
        "off_run_id",
        "on_run_id",
        "pair_complete",
        "pair_valid",
        "avoided_failure",
        "task_recovered_after_hazard_avoidance",
        "terminal_outcome_transition",
        "step_count_delta",
    }
)


class ComparisonRenderError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ArmPlotRecord:
    experiment_id: str
    run_id: str
    paired_run_id: str
    subset_id: str
    case_id: str
    arm_id: str
    r0_over_target: float
    initial_velocity_angle_deg: float
    thrust_scale: float
    crossed_target_radius: bool
    recoverable_crossing: bool
    final_simulator_success: bool
    overspeed: bool
    invalid_simulation: bool
    terminal_label: str
    termination_reason: str
    monitor_evaluation_count: int
    veto_count: int
    intervention_rate: float
    longest_consecutive_veto_steps: int
    steps: int


@dataclass(frozen=True, slots=True)
class PairPlotRecord:
    experiment_id: str
    paired_run_id: str
    case_id: str
    subset_id: str
    off_run_id: str
    on_run_id: str
    pair_complete: bool
    pair_valid: bool
    avoided_failure: bool
    task_recovered_after_hazard_avoidance: bool
    terminal_outcome_transition: str
    step_count_delta: int


@dataclass(frozen=True, slots=True)
class MonitorOnCase:
    case_id: str
    subset_id: str
    label: str
    intervention_rate: float
    veto_count: int
    evaluation_count: int


@dataclass(frozen=True, slots=True)
class StressTransition:
    case_id: str
    label: str
    terminal_transition: str
    hazard_avoided: bool
    task_recovered: bool
    step_count_delta: int
    longest_veto_streak: int


@dataclass(frozen=True, slots=True)
class ComparisonData:
    arm_rows: tuple[ArmPlotRecord, ...]
    pair_rows: tuple[PairPlotRecord, ...]
    case_order: tuple[str, ...]
    monitor_on_cases: tuple[MonitorOnCase, ...]
    stress_transitions: tuple[StressTransition, ...]
    hazard_counts: Mapping[str, tuple[int, int]]
    preservation_counts: Mapping[str, tuple[int, int]]
    total_monitor_evaluations: int
    total_vetoes: int
    overall_intervention_rate: float


@dataclass(frozen=True, slots=True)
class PngMetadata:
    width: int
    height: int
    bit_depth: int
    color_type: int


def _read_csv(path: Path, required_fields: frozenset[str], label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise ComparisonRenderError(f"missing {label} CSV: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames is None:
                raise ComparisonRenderError(f"{label} CSV has no header")
            missing = sorted(required_fields - set(reader.fieldnames))
            if missing:
                raise ComparisonRenderError(f"{label} CSV is missing fields: {missing}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ComparisonRenderError(f"malformed {label} CSV: {exc}") from exc
    if not rows:
        raise ComparisonRenderError(f"{label} CSV contains no rows")
    return rows


def _boolean(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ComparisonRenderError(f"{field} must be true or false")


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ComparisonRenderError(f"{field} must be an integer")
    try:
        converted = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ComparisonRenderError(f"{field} must be an integer") from exc
    return converted


def _number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ComparisonRenderError(f"{field} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonRenderError(f"{field} must be finite") from exc
    if not math.isfinite(converted):
        raise ComparisonRenderError(f"{field} must be finite")
    return converted


def _parse_arm(row: Mapping[str, object]) -> ArmPlotRecord:
    return ArmPlotRecord(
        experiment_id=str(row["experiment_id"]),
        run_id=str(row["run_id"]),
        paired_run_id=str(row["paired_run_id"]),
        subset_id=str(row["subset_id"]),
        case_id=str(row["case_id"]),
        arm_id=str(row["arm_id"]),
        r0_over_target=_number(row["r0_over_target"], "r0_over_target"),
        initial_velocity_angle_deg=_number(
            row["initial_velocity_angle_deg"], "initial_velocity_angle_deg"
        ),
        thrust_scale=_number(row["thrust_scale"], "thrust_scale"),
        crossed_target_radius=_boolean(
            row["crossed_target_radius"], "crossed_target_radius"
        ),
        recoverable_crossing=_boolean(
            row["recoverable_crossing"], "recoverable_crossing"
        ),
        final_simulator_success=_boolean(
            row["final_simulator_success"], "final_simulator_success"
        ),
        overspeed=_boolean(row["overspeed"], "overspeed"),
        invalid_simulation=_boolean(row["invalid_simulation"], "invalid_simulation"),
        terminal_label=str(row["terminal_label"]),
        termination_reason=str(row["termination_reason"]),
        monitor_evaluation_count=_integer(
            row["monitor_evaluation_count"], "monitor_evaluation_count"
        ),
        veto_count=_integer(row["veto_count"], "veto_count"),
        intervention_rate=_number(row["intervention_rate"], "intervention_rate"),
        longest_consecutive_veto_steps=_integer(
            row["longest_consecutive_veto_steps"],
            "longest_consecutive_veto_steps",
        ),
        steps=_integer(row["steps"], "steps"),
    )


def _parse_pair(row: Mapping[str, object]) -> PairPlotRecord:
    return PairPlotRecord(
        experiment_id=str(row["experiment_id"]),
        paired_run_id=str(row["paired_run_id"]),
        case_id=str(row["case_id"]),
        subset_id=str(row["subset_id"]),
        off_run_id=str(row["off_run_id"]),
        on_run_id=str(row["on_run_id"]),
        pair_complete=_boolean(row["pair_complete"], "pair_complete"),
        pair_valid=_boolean(row["pair_valid"], "pair_valid"),
        avoided_failure=_boolean(row["avoided_failure"], "avoided_failure"),
        task_recovered_after_hazard_avoidance=_boolean(
            row["task_recovered_after_hazard_avoidance"],
            "task_recovered_after_hazard_avoidance",
        ),
        terminal_outcome_transition=str(row["terminal_outcome_transition"]),
        step_count_delta=_integer(row["step_count_delta"], "step_count_delta"),
    )


def _case_sort_key(record: ArmPlotRecord) -> tuple[object, ...]:
    subset_order = {
        PRESERVATION_SUBSET_ID: 0,
        STRESS_SUBSET_ID: 1,
    }
    return (
        subset_order[record.subset_id],
        record.r0_over_target,
        record.thrust_scale,
        record.initial_velocity_angle_deg,
        record.case_id,
    )


def _case_label(record: ArmPlotRecord) -> str:
    prefix = "P" if record.subset_id == PRESERVATION_SUBSET_ID else "S"
    thrust = f"{record.thrust_scale / 1000:g}k"
    return f"{prefix}{record.initial_velocity_angle_deg:g}/{thrust}"


def load_comparison_data(results_path: Path, paired_results_path: Path) -> ComparisonData:
    raw_arms = _read_csv(results_path, ARM_REQUIRED_FIELDS, "arm results")
    raw_pairs = _read_csv(paired_results_path, PAIR_REQUIRED_FIELDS, "paired results")
    if len(raw_arms) != 26:
        raise ComparisonRenderError(f"expected exactly 26 arm rows, found {len(raw_arms)}")
    if len(raw_pairs) != 13:
        raise ComparisonRenderError(f"expected exactly 13 pair rows, found {len(raw_pairs)}")

    arms = tuple(_parse_arm(row) for row in raw_arms)
    pairs = tuple(_parse_pair(row) for row in raw_pairs)
    if {row.experiment_id for row in arms + pairs} != {EXPECTED_EXPERIMENT_ID}:
        raise ComparisonRenderError(
            f"all rows must use experiment_id={EXPECTED_EXPERIMENT_ID}"
        )
    if any(row.subset_id not in {PRESERVATION_SUBSET_ID, STRESS_SUBSET_ID} for row in arms):
        raise ComparisonRenderError("arm results contain an undeclared subset")
    if any(row.subset_id not in {PRESERVATION_SUBSET_ID, STRESS_SUBSET_ID} for row in pairs):
        raise ComparisonRenderError("paired results contain an undeclared subset")

    by_run: dict[str, ArmPlotRecord] = {}
    grouped: dict[str, list[ArmPlotRecord]] = {}
    for row in arms:
        if row.run_id in by_run:
            raise ComparisonRenderError(f"duplicate run_id: {row.run_id}")
        by_run[row.run_id] = row
        grouped.setdefault(row.paired_run_id, []).append(row)
    if set(grouped) != {row.paired_run_id for row in pairs}:
        raise ComparisonRenderError("pair IDs do not match arm results")

    pair_by_id: dict[str, PairPlotRecord] = {}
    for pair in pairs:
        if pair.paired_run_id in pair_by_id:
            raise ComparisonRenderError(f"duplicate paired_run_id: {pair.paired_run_id}")
        pair_by_id[pair.paired_run_id] = pair
        if not pair.pair_complete or not pair.pair_valid:
            raise ComparisonRenderError(f"incomplete or invalid pair: {pair.paired_run_id}")
        rows = grouped.get(pair.paired_run_id, [])
        if len(rows) != 2 or {row.arm_id for row in rows} != {"monitor_off", "monitor_on"}:
            raise ComparisonRenderError(
                f"pair requires exactly one monitor_off and monitor_on arm: {pair.paired_run_id}"
            )
        off = by_run.get(pair.off_run_id)
        on = by_run.get(pair.on_run_id)
        if off is None or on is None or off.arm_id != "monitor_off" or on.arm_id != "monitor_on":
            raise ComparisonRenderError(f"pair arm references are invalid: {pair.paired_run_id}")
        if (
            off.case_id != pair.case_id
            or on.case_id != pair.case_id
            or off.subset_id != pair.subset_id
            or on.subset_id != pair.subset_id
        ):
            raise ComparisonRenderError(f"pair identity differs from arm rows: {pair.paired_run_id}")
        observed_transition = f"{off.termination_reason} -> {on.termination_reason}"
        if pair.terminal_outcome_transition != observed_transition:
            raise ComparisonRenderError(
                f"terminal transition disagrees with arm rows: {pair.paired_run_id}"
            )
        if pair.step_count_delta != on.steps - off.steps:
            raise ComparisonRenderError(f"step-count delta disagrees with arm rows: {pair.paired_run_id}")

    preservation_pairs = [row for row in pairs if row.subset_id == PRESERVATION_SUBSET_ID]
    stress_pairs = [row for row in pairs if row.subset_id == STRESS_SUBSET_ID]
    if len(preservation_pairs) != 8 or len(stress_pairs) != 5:
        raise ComparisonRenderError("expected exactly 8 preservation and 5 stress pairs")

    on_rows = sorted((row for row in arms if row.arm_id == "monitor_on"), key=_case_sort_key)
    monitor_cases = tuple(
        MonitorOnCase(
            case_id=row.case_id,
            subset_id=row.subset_id,
            label=_case_label(row),
            intervention_rate=(
                row.veto_count / row.monitor_evaluation_count
                if row.monitor_evaluation_count
                else 0.0
            ),
            veto_count=row.veto_count,
            evaluation_count=row.monitor_evaluation_count,
        )
        for row in on_rows
    )
    for case, row in zip(monitor_cases, on_rows):
        if not math.isclose(case.intervention_rate, row.intervention_rate, abs_tol=1.0e-12):
            raise ComparisonRenderError(f"intervention rate disagrees with counts: {row.run_id}")

    stress_transitions = tuple(
        StressTransition(
            case_id=row.case_id,
            label=_case_label(by_run[row.on_run_id]),
            terminal_transition=row.terminal_outcome_transition,
            hazard_avoided=row.avoided_failure,
            task_recovered=row.task_recovered_after_hazard_avoidance,
            step_count_delta=row.step_count_delta,
            longest_veto_streak=by_run[row.on_run_id].longest_consecutive_veto_steps,
        )
        for row in sorted(
            stress_pairs,
            key=lambda pair: _case_sort_key(by_run[pair.on_run_id]),
        )
    )

    def arm_count(subset: str, arm_id: str, field: str) -> int:
        return sum(
            bool(getattr(row, field))
            for row in arms
            if row.subset_id == subset and row.arm_id == arm_id
        )

    hazard_counts = {
        "preservation": (
            arm_count(PRESERVATION_SUBSET_ID, "monitor_off", "overspeed"),
            arm_count(PRESERVATION_SUBSET_ID, "monitor_on", "overspeed"),
        ),
        "diagnostic stress": (
            arm_count(STRESS_SUBSET_ID, "monitor_off", "overspeed"),
            arm_count(STRESS_SUBSET_ID, "monitor_on", "overspeed"),
        ),
    }
    preservation_counts = {
        "crossing": (
            arm_count(PRESERVATION_SUBSET_ID, "monitor_off", "crossed_target_radius"),
            arm_count(PRESERVATION_SUBSET_ID, "monitor_on", "crossed_target_radius"),
        ),
        "recoverable crossing": (
            arm_count(PRESERVATION_SUBSET_ID, "monitor_off", "recoverable_crossing"),
            arm_count(PRESERVATION_SUBSET_ID, "monitor_on", "recoverable_crossing"),
        ),
        "simulator success": (
            arm_count(PRESERVATION_SUBSET_ID, "monitor_off", "final_simulator_success"),
            arm_count(PRESERVATION_SUBSET_ID, "monitor_on", "final_simulator_success"),
        ),
    }
    total_evaluations = sum(row.monitor_evaluation_count for row in on_rows)
    total_vetoes = sum(row.veto_count for row in on_rows)
    return ComparisonData(
        arm_rows=arms,
        pair_rows=pairs,
        case_order=tuple(row.case_id for row in on_rows),
        monitor_on_cases=monitor_cases,
        stress_transitions=stress_transitions,
        hazard_counts=hazard_counts,
        preservation_counts=preservation_counts,
        total_monitor_evaluations=total_evaluations,
        total_vetoes=total_vetoes,
        overall_intervention_rate=(total_vetoes / total_evaluations),
    )


def inspect_png(path: Path) -> PngMetadata:
    if not path.is_file():
        raise ComparisonRenderError(f"PNG does not exist: {path}")
    payload = path.read_bytes()
    if not payload.startswith(PNG_SIGNATURE):
        raise ComparisonRenderError("file does not have a valid PNG signature")
    offset = len(PNG_SIGNATURE)
    ihdr: tuple[int, int, int, int, int] | None = None
    compressed = bytearray()
    saw_iend = False
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ComparisonRenderError("PNG contains a truncated chunk")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(payload):
            raise ComparisonRenderError("PNG contains a truncated chunk payload")
        chunk_data = payload[data_start:data_end]
        expected_crc = struct.unpack(">I", payload[data_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ComparisonRenderError("PNG chunk CRC validation failed")
        if chunk_type == b"IHDR":
            if ihdr is not None or length != 13:
                raise ComparisonRenderError("PNG has an invalid IHDR chunk")
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if width <= 0 or height <= 0 or compression != 0 or filtering != 0 or interlace != 0:
                raise ComparisonRenderError("PNG IHDR uses unsupported dimensions or encoding")
            ihdr = (width, height, bit_depth, color_type, interlace)
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            if length != 0:
                raise ComparisonRenderError("PNG has an invalid IEND chunk")
            saw_iend = True
            offset = crc_end
            break
        offset = crc_end
    if ihdr is None or not compressed or not saw_iend or offset != len(payload):
        raise ComparisonRenderError("PNG lacks a complete IHDR/IDAT/IEND structure")
    width, height, bit_depth, color_type, _ = ihdr
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None or bit_depth not in {1, 2, 4, 8, 16}:
        raise ComparisonRenderError("PNG color type or bit depth is unsupported")
    row_bytes = (width * channels * bit_depth + 7) // 8
    try:
        pixels = zlib.decompress(bytes(compressed))
    except zlib.error as exc:
        raise ComparisonRenderError("PNG image stream cannot be decompressed") from exc
    expected_length = height * (row_bytes + 1)
    if len(pixels) != expected_length:
        raise ComparisonRenderError("PNG decompressed image length is inconsistent")
    if any(pixels[row * (row_bytes + 1)] > 4 for row in range(height)):
        raise ComparisonRenderError("PNG contains an invalid scanline filter")
    return PngMetadata(width, height, bit_depth, color_type)


def _draw_figure(data: ComparisonData, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {
        "off": "#C43C39",
        "on": "#198C8C",
        "preservation": "#3268A8",
        "stress": "#D28A22",
        "grid": "#D9DEE5",
        "text": "#20252B",
    }
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "#7B838C",
            "axes.labelcolor": colors["text"],
            "text.color": colors["text"],
            "xtick.color": colors["text"],
            "ytick.color": colors["text"],
        }
    ):
        figure = plt.figure(
            figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
            dpi=FIGURE_DPI,
            facecolor="white",
        )
        grid = figure.add_gridspec(2, 2, hspace=0.42, wspace=0.27)
        hazard_axis = figure.add_subplot(grid[0, 0])
        preservation_axis = figure.add_subplot(grid[0, 1])
        burden_axis = figure.add_subplot(grid[1, 0])
        transition_axis = figure.add_subplot(grid[1, 1])

        categories = list(data.hazard_counts)
        positions = np.arange(len(categories))
        width = 0.34
        off_values = [data.hazard_counts[name][0] for name in categories]
        on_values = [data.hazard_counts[name][1] for name in categories]
        off_bars = hazard_axis.bar(
            positions - width / 2,
            off_values,
            width,
            label="monitor_off",
            color=colors["off"],
        )
        on_bars = hazard_axis.bar(
            positions + width / 2,
            on_values,
            width,
            label="monitor_on",
            color=colors["on"],
        )
        hazard_axis.bar_label(off_bars, padding=3)
        hazard_axis.bar_label(on_bars, padding=3)
        hazard_axis.set_xticks(positions, categories)
        hazard_axis.set_ylim(0, 8.8)
        hazard_axis.set_ylabel("Overspeed outcomes (count)")
        hazard_axis.set_title("1. Declared Overspeed Hazard Outcomes", loc="left", fontweight="bold")
        hazard_axis.legend(frameon=False, ncols=2, loc="upper left")
        hazard_axis.grid(axis="y", color=colors["grid"], linewidth=0.8)
        hazard_axis.set_axisbelow(True)

        metrics = list(data.preservation_counts)
        metric_positions = np.arange(len(metrics))
        off_metrics = [data.preservation_counts[name][0] for name in metrics]
        on_metrics = [data.preservation_counts[name][1] for name in metrics]
        off_bars = preservation_axis.bar(
            metric_positions - width / 2,
            off_metrics,
            width,
            label="monitor_off",
            color=colors["off"],
        )
        on_bars = preservation_axis.bar(
            metric_positions + width / 2,
            on_metrics,
            width,
            label="monitor_on",
            color=colors["on"],
        )
        preservation_axis.bar_label(off_bars, padding=3)
        preservation_axis.bar_label(on_bars, padding=3)
        preservation_axis.set_xticks(metric_positions, metrics)
        preservation_axis.set_ylim(0, 10.4)
        preservation_axis.set_ylabel("Preservation cases (count of 8)")
        preservation_axis.set_title("2. Protected Preservation Outcomes", loc="left", fontweight="bold")
        preservation_axis.legend(frameon=False, ncols=2, loc="upper left")
        preservation_axis.grid(axis="y", color=colors["grid"], linewidth=0.8)
        preservation_axis.set_axisbelow(True)

        labels = [case.label for case in data.monitor_on_cases]
        rates = [case.intervention_rate for case in data.monitor_on_cases]
        bar_colors = [
            colors["preservation"]
            if case.subset_id == PRESERVATION_SUBSET_ID
            else colors["stress"]
            for case in data.monitor_on_cases
        ]
        burden_axis.bar(
            np.arange(len(labels)),
            rates,
            color=bar_colors,
            width=0.72,
        )
        burden_axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        burden_axis.set_ylim(0, 1.05)
        burden_axis.set_ylabel("Vetoes / monitor evaluations")
        burden_axis.set_xlabel(
            "P = preservation; S = diagnostic stress | "
            f"Overall: {data.total_vetoes:,} / {data.total_monitor_evaluations:,} "
            f"= {data.overall_intervention_rate:.6f}"
        )
        burden_axis.set_title("3. Monitor-On Intervention Burden by Case", loc="left", fontweight="bold")
        burden_axis.grid(axis="y", color=colors["grid"], linewidth=0.8)
        burden_axis.set_axisbelow(True)

        transition_axis.set_axis_off()
        transition_axis.set_title(
            "4. Diagnostic Stress Terminal Transitions",
            loc="left",
            fontweight="bold",
            pad=12,
        )
        columns = (0.00, 0.20, 0.60, 0.78, 0.92)
        headers = (
            "Case",
            "Terminal transition",
            "Hazard\navoided",
            "Task\nrecovered",
            "Delta\nsteps",
        )
        for x, header in zip(columns, headers):
            transition_axis.text(
                x,
                0.91,
                header,
                transform=transition_axis.transAxes,
                fontweight="bold",
                fontsize=9,
                linespacing=1.05,
            )
        transition_axis.plot([0, 1], [0.88, 0.88], transform=transition_axis.transAxes, color=colors["grid"])
        for index, transition in enumerate(data.stress_transitions):
            y = 0.79 - index * 0.14
            values = (
                transition.label,
                transition.terminal_transition,
                "yes" if transition.hazard_avoided else "no",
                "yes" if transition.task_recovered else "no",
                f"{transition.step_count_delta:+,}",
            )
            for x, value in zip(columns, values):
                transition_axis.text(x, y, value, transform=transition_axis.transAxes, fontsize=9)
        transition_axis.text(
            0.0,
            0.04,
            (
                "Declared hazard avoided: "
                f"{sum(item.hazard_avoided for item in data.stress_transitions)}/5; "
                "task recovered: "
                f"{sum(item.task_recovered for item in data.stress_transitions)}/5."
            ),
            transform=transition_axis.transAxes,
            fontsize=9,
            fontweight="bold",
        )

        figure.suptitle(
            "Final Veto Overspeed Ablation v0: Paired Simulator Evidence",
            fontsize=17,
            fontweight="bold",
            y=0.985,
        )
        figure.text(
            0.5,
            0.016,
            (
                "Simulator-level paired ablation. Overspeed-hazard avoidance does not imply task recovery, "
                "formal safety, hardware readiness, or deployment readiness."
            ),
            ha="center",
            fontsize=9,
            color="#41464D",
        )
        figure.savefig(
            destination,
            format="png",
            dpi=FIGURE_DPI,
            facecolor="white",
            metadata={
                "Software": "spacecraft-ai-controller deterministic renderer",
                "Title": "Final Veto Overspeed Ablation v0 Comparison",
                "Description": "Simulator-level paired ablation; hazard avoidance and task recovery are separate.",
            },
        )
        plt.close(figure)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_path(output_path: Path) -> Path:
    if ".." in output_path.parts:
        raise ComparisonRenderError("output path traversal is not allowed")
    if output_path.is_absolute():
        destination = output_path.resolve()
    else:
        destination = (Path.cwd() / output_path).resolve()
    for protected in PROTECTED_HISTORICAL_DIRECTORIES:
        protected_resolved = protected.resolve()
        if destination == protected_resolved or _is_relative_to(destination, protected_resolved):
            raise ComparisonRenderError(
                f"output path overlaps protected historical directory: {protected}"
            )
    return destination


def render_comparison_plot(
    results_path: Path,
    paired_results_path: Path,
    output_path: Path,
) -> tuple[ComparisonData, PngMetadata]:
    destination = validate_output_path(output_path)
    if destination.exists():
        raise ComparisonRenderError(f"refusing to overwrite existing plot: {destination}")
    data = load_comparison_data(results_path, paired_results_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _draw_figure(data, temporary)
        metadata = inspect_png(temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return data, metadata


def _run_required_check(command: list[str], label: str) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).strip()
        raise ComparisonRenderError(f"{label} failed: {detail}")


def validate_formal_plot_creation_preconditions(
    results_path: Path,
    paired_results_path: Path,
    output_path: Path,
) -> None:
    if results_path.resolve() != FORMAL_RESULTS.resolve():
        raise ComparisonRenderError("formal plot creation requires the frozen results.csv path")
    if paired_results_path.resolve() != FORMAL_PAIRED_RESULTS.resolve():
        raise ComparisonRenderError("formal plot creation requires the frozen paired_results.csv path")
    if output_path.resolve() != FORMAL_COMPARISON.resolve():
        raise ComparisonRenderError("formal plot creation requires the frozen comparison.png path")
    if output_path.exists():
        raise ComparisonRenderError("formal comparison plot already exists")
    for path in FORMAL_REQUIRED_BEFORE_PLOT:
        if not path.is_file() or path.stat().st_size <= 0:
            raise ComparisonRenderError(f"required formal artifact is missing or empty: {path}")
    load_comparison_data(results_path, paired_results_path)
    _run_required_check(
        [sys.executable, "scripts/check_final_veto_manifest.py"],
        "frozen manifest validation",
    )
    _run_required_check(
        [sys.executable, "scripts/check_phase_results.py"],
        "protected historical regression guard",
    )
    _run_required_check(
        [
            sys.executable,
            "scripts/check_final_veto_results.py",
            "--results",
            str(results_path),
            "--paired-results",
            str(paired_results_path),
            "--decision-log",
            str(FORMAL_OUTPUT_DIRECTORY / "decision_log.jsonl"),
            "--formal",
            "--formal-plot-pending",
        ],
        "formal result validation with plot pending",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the deterministic Final Veto paired-ablation comparison plot."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--paired-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-create-missing-formal-plot",
        action="store_true",
        help="Create only the missing frozen formal plot after all read-only guards pass.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.allow_create_missing_formal_plot:
            validate_formal_plot_creation_preconditions(
                args.results,
                args.paired_results,
                args.output,
            )
        data, metadata = render_comparison_plot(
            args.results,
            args.paired_results,
            args.output,
        )
    except (ComparisonRenderError, OSError) as exc:
        print(f"FAIL {exc}")
        return 1
    print(
        "COMPARISON_PLOT_CREATED "
        f"arms={len(data.arm_rows)} pairs={len(data.pair_rows)} "
        f"pixels={metadata.width}x{metadata.height} dpi={FIGURE_DPI}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
