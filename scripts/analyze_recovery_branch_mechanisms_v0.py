from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ANALYSIS_SCHEMA_VERSION = "recovery_branch_mechanism_diagnosis_v0"
EXPERIMENT_ID = "recovery_action_branching_nonformal_v0"
RESULT_COMMIT = "5f31c3fd74dbf8e8ea5a60d70d7b88f5a9def7c8"
IMPLEMENTATION_COMMIT = "2e1fffbb00789c185256d0b13dff65150f21ba50"
UNAVAILABLE = "not_available_in_published_artifacts"

BRANCH_ORDER = (
    "zero_action_reference_v0",
    "velocity_opposed_thrust_v0",
    "tangential_error_correction_v0",
    "explicit_abort_v0",
)
PHYSICAL_BRANCHES = BRANCH_ORDER[:3]
CHECKPOINT_STEPS = (1, 10, 100, 1000, 2500, 5000, 7500, 10000)
EXPLORATORY_NEAR_THRESHOLD_HEADROOM = 0.05

SOURCE_FILENAMES = (
    "manifest.json",
    "branch_state.json",
    "results.csv",
    "decision_log.jsonl",
    "summary.md",
    "comparison.png",
)
FROZEN_SOURCE_HASHES = {
    "manifest.json": "c317f94f937412f4fb5ac826fd97b21002c47cfc9bba6d9ad4eda6c4be1a921b",
    "branch_state.json": "b9fbcdd3544f527c7431d1b3bc5795ea755935a4973a18cbb8d8b710685d64fc",
    "results.csv": "c13abb4e15a6f04a9322c6c7955553464b9815d1b4d5c58374a3100eb4ccc668",
    "decision_log.jsonl": "43cfc05100648b6d0d652a8ac1d9a35f7179ebec78ff0138eaa5e7ab846096b4",
    "summary.md": "3e93152ef22e05a58650561111d4d3d96206391ea324a497993398eab3f8e8c0",
    "comparison.png": "d18310b4e15a9eb26bcc7884e8b56d9d6a90a4b231a999e7b8ed251dc4d902cb",
}

EXPECTED_EVENT_COUNTS = {
    "zero_action_reference_v0": 10000,
    "velocity_opposed_thrust_v0": 10000,
    "tangential_error_correction_v0": 10000,
    "explicit_abort_v0": 1,
}

FIELD_CLASSIFICATIONS = (
    "directly_measured",
    "derivable_from_measured_fields",
    "available_only_at_branch_point",
    "available_only_as_final_summary",
    UNAVAILABLE,
)

RESULT_REQUIRED_FIELDS = {
    "experiment_id",
    "branch_id",
    "branch_state_hash",
    "manifest_hash",
    "implementation_commit",
    "seed",
    "branch_step",
    "nominal_prefix_transition_count",
    "recovery_transition_count",
    "total_transition_count",
    "terminal_reason",
    "overspeed_status",
    "instability_status",
    "unsafe_state_status",
    "invalid_simulation_status",
    "crossed_target_radius",
    "phase34_compatible_recoverable_crossing",
    "recovery_success",
    "final_simulator_success",
    "monitor_evaluation_count",
    "allow_count",
    "veto_count",
    "normalized_control_effort",
    "delta_v_proxy",
    "final_radius_error",
    "final_radial_velocity_error",
    "final_tangential_velocity_error",
}

EVENT_REQUIRED_FIELDS = {
    "branch_id",
    "branch_state_hash",
    "case_id",
    "current_state_hash",
    "event_index",
    "post_branch_step",
    "total_transition_count",
    "proposed_action",
    "executed_action",
    "final_veto_decision",
    "transition_occurred",
    "next_state_hash",
    "predicted_speed_ratio",
    "realized_speed_ratio",
    "hazard_threshold",
    "hazard_comparator",
    "evaluator_statuses",
    "terminal_reason",
    "triggered_stop_condition",
}

BRANCH_METRIC_FIELDS = (
    "branch_id",
    "frozen_branch_order",
    "branch_state_hash",
    "manifest_hash",
    "implementation_commit",
    "seed",
    "branch_step",
    "nominal_prefix_transition_count",
    "terminal_reason",
    "recovery_transition_count",
    "total_transition_count",
    "overspeed_status",
    "instability_status",
    "unsafe_state_status",
    "invalid_simulation_status",
    "target_radius_crossing",
    "recoverable_crossing",
    "recovery_success_v0",
    "final_simulator_success",
    "monitor_evaluation_count",
    "allow_count",
    "veto_count",
    "allow_rate",
    "intervention_rate",
    "first_veto_step",
    "longest_veto_streak",
    "action_count",
    "zero_action_count",
    "mean_action_magnitude",
    "minimum_action_magnitude",
    "maximum_action_magnitude",
    "recomputed_cumulative_normalized_effort",
    "published_cumulative_normalized_effort",
    "action_direction_variation_radians",
    "action_direction_change_count",
    "action_direction_flip_count_over_90deg",
    "first_action",
    "final_action",
    "first_action_radial_component_at_branch_point",
    "first_action_tangential_component_at_branch_point",
    "trajectory_action_radial_component",
    "trajectory_action_tangential_component",
    "initial_post_branch_speed_ratio",
    "final_speed_ratio",
    "minimum_speed_ratio",
    "maximum_speed_ratio",
    "mean_speed_ratio",
    "minimum_overspeed_headroom",
    "final_overspeed_headroom",
    "closest_threshold_step",
    "speed_ratio_trend",
    "speed_ratio_increase_count",
    "speed_ratio_decrease_count",
    "exploratory_near_threshold_event_count",
    "exploratory_near_threshold_headroom_band",
    "maximum_prediction_realization_error",
    "initial_target_radius_error",
    "final_target_radius_error",
    "target_radius_gap_reduction",
    "target_radius_gap_reduction_fraction",
    "minimum_absolute_target_radius_error",
    "closest_target_approach_step",
    "initial_radial_velocity_error",
    "final_radial_velocity_error",
    "minimum_radial_velocity_error",
    "initial_tangential_velocity_error",
    "final_tangential_velocity_error",
    "minimum_tangential_velocity_error",
    "first_crossing_step",
    "first_recoverable_crossing_step",
    "initial_r_error_ratio",
    "final_r_error_ratio",
    "initial_vr_ratio",
    "final_vr_ratio",
    "initial_vt_error_ratio",
    "final_vt_error_ratio",
    "final_r_component_margin",
    "final_vr_component_margin",
    "final_vt_component_margin",
    "closest_recoverability_component_margins",
    "initial_specific_orbital_energy_j_per_kg",
    "final_specific_orbital_energy_j_per_kg",
    "specific_orbital_energy_change_j_per_kg",
    "target_circular_specific_energy_j_per_kg",
    "final_energy_difference_from_target_j_per_kg",
    "energy_derivation_basis",
    "published_delta_v_proxy",
)

CHECKPOINT_FIELDS = (
    "branch_id",
    "recovery_step",
    "total_transition_count",
    "current_state_hash",
    "next_state_hash",
    "action",
    "action_magnitude",
    "final_veto_decision",
    "predicted_speed_ratio",
    "realized_speed_ratio",
    "overspeed_headroom",
    "radius",
    "target_radius_error",
    "radial_velocity",
    "tangential_velocity",
    "recovery_evaluator_status",
    "stop_condition",
)


class MechanismAnalysisError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceBundle:
    source_directory: Path
    source_hashes: Mapping[str, str]
    manifest: Mapping[str, object]
    branch_state: Mapping[str, object]
    result_rows: tuple[Mapping[str, str], ...]
    events_by_branch: Mapping[str, tuple[Mapping[str, object], ...]]
    distinct_event_keys: tuple[str, ...]
    event_numeric_fields: tuple[str, ...]
    event_categorical_fields: tuple[str, ...]
    event_vector_fields: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MechanismAnalysisError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MechanismAnalysisError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MechanismAnalysisError(f"JSON artifact must be an object: {path}")
    return value


def load_results_csv(path: Path) -> tuple[Mapping[str, str], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not RESULT_REQUIRED_FIELDS.issubset(reader.fieldnames):
                missing = sorted(RESULT_REQUIRED_FIELDS - set(reader.fieldnames or ()))
                raise MechanismAnalysisError(
                    f"results.csv lacks required fields: {', '.join(missing)}"
                )
            rows = tuple(dict(row) for row in reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise MechanismAnalysisError(f"cannot read results CSV {path}: {exc}") from exc
    branch_ids = tuple(row.get("branch_id", "") for row in rows)
    if len(set(branch_ids)) != len(branch_ids):
        raise MechanismAnalysisError("results.csv contains a duplicate branch record")
    unknown = sorted(set(branch_ids) - set(BRANCH_ORDER))
    if unknown:
        raise MechanismAnalysisError(f"results.csv contains unknown branches: {unknown}")
    if branch_ids != BRANCH_ORDER:
        raise MechanismAnalysisError(
            f"results.csv branch order must be {BRANCH_ORDER}, got {branch_ids}"
        )
    return rows


def load_decision_events(
    path: Path,
) -> tuple[
    Mapping[str, tuple[Mapping[str, object], ...]],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    grouped: dict[str, list[Mapping[str, object]]] = {branch: [] for branch in BRANCH_ORDER}
    keys: set[str] = set()
    numeric: set[str] = set()
    categorical: set[str] = set()
    vectors: set[str] = set()
    encountered_groups: list[str] = []
    active_branch: str | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise MechanismAnalysisError(
                        f"decision log contains a blank line at {line_number}"
                    )
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MechanismAnalysisError(
                        f"malformed decision JSONL at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(event, dict):
                    raise MechanismAnalysisError(
                        f"decision event at line {line_number} is not an object"
                    )
                missing = EVENT_REQUIRED_FIELDS - set(event)
                if missing:
                    raise MechanismAnalysisError(
                        f"decision event at line {line_number} lacks fields: {sorted(missing)}"
                    )
                branch_id = event.get("branch_id")
                if branch_id not in grouped:
                    raise MechanismAnalysisError(
                        f"decision event at line {line_number} has unknown branch {branch_id!r}"
                    )
                if branch_id != active_branch:
                    if branch_id in encountered_groups:
                        raise MechanismAnalysisError(
                            f"decision events for {branch_id} are not contiguous"
                        )
                    encountered_groups.append(str(branch_id))
                    active_branch = str(branch_id)
                grouped[str(branch_id)].append(event)
                keys.update(event)
                for key, value in event.items():
                    if value is None:
                        continue
                    if isinstance(value, bool):
                        categorical.add(key)
                    elif isinstance(value, (int, float)):
                        if not math.isfinite(float(value)):
                            raise MechanismAnalysisError(
                                f"nonfinite event value {key} at line {line_number}"
                            )
                        numeric.add(key)
                    elif isinstance(value, str):
                        categorical.add(key)
                    elif isinstance(value, list) and key in {
                        "proposed_action",
                        "executed_action",
                    }:
                        vectors.add(key)
    except (OSError, UnicodeDecodeError) as exc:
        raise MechanismAnalysisError(f"cannot read decision log {path}: {exc}") from exc
    if tuple(encountered_groups) != BRANCH_ORDER:
        raise MechanismAnalysisError(
            f"decision-log branch order must be {BRANCH_ORDER}, got {tuple(encountered_groups)}"
        )
    return (
        {branch: tuple(grouped[branch]) for branch in BRANCH_ORDER},
        tuple(sorted(keys)),
        tuple(sorted(numeric)),
        tuple(sorted(categorical)),
        tuple(sorted(vectors)),
    )


def load_source_bundle(
    source_directory: Path,
    *,
    enforce_frozen_hashes: bool = True,
) -> SourceBundle:
    source = source_directory.resolve()
    if not source.is_dir():
        raise MechanismAnalysisError(f"source directory does not exist: {source}")
    missing = [name for name in SOURCE_FILENAMES if not (source / name).is_file()]
    if missing:
        raise MechanismAnalysisError(f"source artifacts are missing: {missing}")
    hashes = {name: sha256_file(source / name) for name in SOURCE_FILENAMES}
    if enforce_frozen_hashes:
        drift = {
            name: (FROZEN_SOURCE_HASHES[name], hashes[name])
            for name in SOURCE_FILENAMES
            if hashes[name] != FROZEN_SOURCE_HASHES[name]
        }
        if drift:
            raise MechanismAnalysisError(f"frozen source artifact hash drift: {drift}")
    manifest = _load_json(source / "manifest.json")
    branch_state = _load_json(source / "branch_state.json")
    rows = load_results_csv(source / "results.csv")
    events, keys, numeric, categorical, vectors = load_decision_events(
        source / "decision_log.jsonl"
    )
    return SourceBundle(
        source_directory=source,
        source_hashes=hashes,
        manifest=manifest,
        branch_state=branch_state,
        result_rows=rows,
        events_by_branch=events,
        distinct_event_keys=keys,
        event_numeric_fields=numeric,
        event_categorical_fields=categorical,
        event_vector_fields=vectors,
    )


def _parse_bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise MechanismAnalysisError(f"{field} must be an explicit boolean")


def _parse_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise MechanismAnalysisError(f"{field} must be an integer")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise MechanismAnalysisError(f"{field} must be an integer") from exc
    return parsed


def _parse_float(value: object, field: str) -> float:
    if value is None or value == "":
        raise MechanismAnalysisError(f"{field} must be available")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MechanismAnalysisError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise MechanismAnalysisError(f"{field} must be finite")
    return parsed


def _optional_float(value: object, field: str) -> float | None:
    if value is None or value == "":
        return None
    return _parse_float(value, field)


def _optional_int(value: object, field: str) -> int | None:
    if value is None or value == "":
        return None
    return _parse_int(value, field)


def _action(value: object, field: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise MechanismAnalysisError(f"{field} must be a two-component action or null")
    return (_parse_float(value[0], field), _parse_float(value[1], field))


def _branch_state_payload_hash(branch_state: Mapping[str, object]) -> str:
    payload = dict(branch_state)
    stored = payload.pop("canonical_branch_state_hash", None)
    if not isinstance(stored, str):
        raise MechanismAnalysisError("branch state lacks canonical_branch_state_hash")
    return canonical_sha256(payload)


def _branch_state_vector_hash(branch_state: Mapping[str, object]) -> str:
    state = branch_state.get("state")
    if not isinstance(state, dict):
        raise MechanismAnalysisError("branch state lacks state object")
    payload = {
        "position_x": _parse_float(state.get("position_x"), "state.position_x"),
        "position_y": _parse_float(state.get("position_y"), "state.position_y"),
        "velocity_x": _parse_float(state.get("velocity_x"), "state.velocity_x"),
        "velocity_y": _parse_float(state.get("velocity_y"), "state.velocity_y"),
    }
    return canonical_sha256(payload)


def structural_issues(bundle: SourceBundle) -> tuple[str, ...]:
    issues: list[str] = []
    manifest_branches = bundle.manifest.get("branches")
    manifest_order = tuple(
        item.get("branch_id")
        for item in manifest_branches
        if isinstance(item, dict)
    ) if isinstance(manifest_branches, list) else ()
    if manifest_order != BRANCH_ORDER:
        issues.append(f"manifest branch order differs: {manifest_order}")

    stored_branch_hash = bundle.branch_state.get("canonical_branch_state_hash")
    recomputed_branch_hash = _branch_state_payload_hash(bundle.branch_state)
    if stored_branch_hash != recomputed_branch_hash:
        issues.append("canonical branch-state hash does not recompute")
    manifest_hash = canonical_sha256(bundle.manifest)
    initial_state_hash = _branch_state_vector_hash(bundle.branch_state)

    common_branch_hashes = {row.get("branch_state_hash") for row in bundle.result_rows}
    common_manifest_hashes = {row.get("manifest_hash") for row in bundle.result_rows}
    common_implementations = {row.get("implementation_commit") for row in bundle.result_rows}
    if common_branch_hashes != {stored_branch_hash}:
        issues.append("result rows do not share the frozen branch-state hash")
    if common_manifest_hashes != {manifest_hash}:
        issues.append("result rows do not share the canonical manifest hash")
    if common_implementations != {IMPLEMENTATION_COMMIT}:
        issues.append("result rows do not record the frozen implementation commit")
    if any(row.get("experiment_id") != EXPERIMENT_ID for row in bundle.result_rows):
        issues.append("result rows contain an unexpected experiment ID")

    for row in bundle.result_rows:
        branch = row["branch_id"]
        events = bundle.events_by_branch[branch]
        expected = EXPECTED_EVENT_COUNTS[branch]
        if len(events) != expected:
            issues.append(
                f"event-count mismatch for {branch}: measured={len(events)} expected={expected}"
            )
        monitor_count = _parse_int(row["monitor_evaluation_count"], "monitor_evaluation_count")
        expected_monitor_count = len(events) if branch in PHYSICAL_BRANCHES else 0
        if monitor_count != expected_monitor_count:
            issues.append(
                f"monitor/event count mismatch for {branch}: {monitor_count}/{len(events)}"
            )
        recomputed_allows = sum(
            event.get("final_veto_decision") == "allow" for event in events
        )
        recomputed_vetoes = sum(
            event.get("final_veto_decision") == "veto" for event in events
        )
        if _parse_int(row["allow_count"], "allow_count") != recomputed_allows:
            issues.append(f"allow-count mismatch for {branch}")
        if _parse_int(row["veto_count"], "veto_count") != recomputed_vetoes:
            issues.append(f"veto-count mismatch for {branch}")
        for index, event in enumerate(events):
            if event.get("branch_state_hash") != stored_branch_hash:
                issues.append(f"event branch-state hash mismatch for {branch} at {index}")
                break
            if event.get("event_index") != index:
                issues.append(f"event index mismatch for {branch} at {index}")
                break
        if events and events[0].get("current_state_hash") != initial_state_hash:
            issues.append(f"{branch} does not start at the frozen Cartesian state")
        if branch == "explicit_abort_v0":
            if len(events) != 1 or events[0].get("transition_occurred") is not False:
                issues.append("explicit abort contains physical transition evidence")
            if _parse_int(row["recovery_transition_count"], "recovery_transition_count") != 0:
                issues.append("explicit abort reports a nonzero transition count")
        else:
            if any(event.get("transition_occurred") is not True for event in events):
                issues.append(f"physical branch {branch} contains a non-transition event")
            if events and events[-1].get("terminal_reason") != row.get("terminal_reason"):
                issues.append(f"terminal reason mismatch for {branch}")
    return tuple(issues)


def require_structural_validity(bundle: SourceBundle) -> None:
    issues = structural_issues(bundle)
    if issues:
        raise MechanismAnalysisError("source structural validation failed: " + "; ".join(issues))


def _availability_inventory() -> Mapping[str, str]:
    inventory = {
        "branch_point_state_vector": "available_only_at_branch_point",
        "per_step_state_vectors": UNAVAILABLE,
        "final_cartesian_state_vector": UNAVAILABLE,
        "per_step_action_vectors": "directly_measured",
        "action_magnitude": "derivable_from_measured_fields",
        "action_direction": "derivable_from_measured_fields",
        "branch_point_radius": "derivable_from_measured_fields",
        "per_step_radius": UNAVAILABLE,
        "final_radius_error": "available_only_as_final_summary",
        "branch_point_radial_velocity": "derivable_from_measured_fields",
        "per_step_radial_velocity": UNAVAILABLE,
        "final_radial_velocity": "available_only_as_final_summary",
        "branch_point_tangential_velocity": "derivable_from_measured_fields",
        "per_step_tangential_velocity": UNAVAILABLE,
        "final_tangential_velocity_error": "available_only_as_final_summary",
        "per_step_target_radius_error": UNAVAILABLE,
        "final_target_radius_error": "available_only_as_final_summary",
        "per_step_orbital_energy": UNAVAILABLE,
        "endpoint_specific_orbital_energy": "derivable_from_measured_fields",
        "predicted_speed_ratio": "directly_measured",
        "realized_speed_ratio": "directly_measured",
        "state_hashes": "directly_measured",
        "crossing_outcome": "available_only_as_final_summary",
        "first_crossing_step": "available_only_as_final_summary",
        "recoverable_crossing_outcome": "available_only_as_final_summary",
        "per_step_recoverability_components": UNAVAILABLE,
        "final_recoverability_components": "derivable_from_measured_fields",
        "closest_recoverability_component_margins": UNAVAILABLE,
        "final_state_orbital_summary": "available_only_as_final_summary",
        "minimum_target_radius_error": UNAVAILABLE,
        "closest_target_approach_step": UNAVAILABLE,
    }
    if not set(inventory.values()).issubset(FIELD_CLASSIFICATIONS):
        raise AssertionError("invalid availability classification")
    return inventory


def build_field_inventory(bundle: SourceBundle) -> Mapping[str, object]:
    event_counts = {
        branch: len(bundle.events_by_branch[branch]) for branch in BRANCH_ORDER
    }
    event_types: Counter[str] = Counter()
    branch_schemas: dict[str, object] = {}
    for branch in BRANCH_ORDER:
        branch_keys: set[str] = set()
        for event in bundle.events_by_branch[branch]:
            branch_keys.update(event)
            if event.get("transition_occurred"):
                event_type = (
                    "terminal_physical_transition"
                    if event.get("terminal_reason")
                    else "physical_transition"
                )
            else:
                event_type = "explicit_abort_terminal"
            event_types[event_type] += 1
        branch_schemas[branch] = {
            "event_count": len(bundle.events_by_branch[branch]),
            "keys": sorted(branch_keys),
        }
    availability = _availability_inventory()
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "published_recovery_evidence_analyzed_no_new_trajectory",
        "source_artifact_paths": {
            name: f"analysis/recovery_action_branching_nonformal_v0/{name}"
            for name in SOURCE_FILENAMES
        },
        "source_artifact_sha256": dict(sorted(bundle.source_hashes.items())),
        "result_commit": RESULT_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "result_row_count": len(bundle.result_rows),
        "decision_event_count": sum(event_counts.values()),
        "event_count_by_branch": event_counts,
        "expected_event_count_by_branch": EXPECTED_EVENT_COUNTS,
        "event_count_discrepancies": {
            branch: event_counts[branch] - EXPECTED_EVENT_COUNTS[branch]
            for branch in BRANCH_ORDER
            if event_counts[branch] != EXPECTED_EVENT_COUNTS[branch]
        },
        "distinct_event_types": dict(sorted(event_types.items())),
        "distinct_event_keys": list(bundle.distinct_event_keys),
        "event_schema_by_branch": branch_schemas,
        "numeric_fields": list(bundle.event_numeric_fields),
        "categorical_fields": list(bundle.event_categorical_fields),
        "numeric_vector_fields": list(bundle.event_vector_fields),
        "state_vector_availability": availability["per_step_state_vectors"],
        "action_vector_availability": availability["per_step_action_vectors"],
        "radius_availability": availability["per_step_radius"],
        "radial_velocity_availability": availability["per_step_radial_velocity"],
        "tangential_velocity_availability": availability["per_step_tangential_velocity"],
        "target_radius_error_availability": availability["per_step_target_radius_error"],
        "orbital_energy_availability": availability["per_step_orbital_energy"],
        "predicted_speed_ratio_availability": availability["predicted_speed_ratio"],
        "realized_speed_ratio_availability": availability["realized_speed_ratio"],
        "state_hash_availability": availability["state_hashes"],
        "crossing_event_availability": availability["crossing_outcome"],
        "recoverability_component_availability": availability[
            "per_step_recoverability_components"
        ],
        "final_state_availability": availability["final_state_orbital_summary"],
        "diagnostic_quantity_classification": availability,
        "limitations": [
            "State hashes establish exact identity or difference but reveal no physical distance or state components.",
            "Per-step Cartesian state, radius, radial velocity, tangential velocity, target-radius error, orbital energy, and recoverability components were not published.",
            "Endpoint orbital quantities can be derived from the branch-point state, final scalar summaries, and frozen constants; they are not trajectory extrema.",
            "No interpolation or trajectory reconstruction is performed.",
        ],
    }


def _branch_point_quantities(branch_state: Mapping[str, object]) -> Mapping[str, float]:
    state = branch_state.get("state")
    simulator = branch_state.get("simulator_configuration")
    if not isinstance(state, dict) or not isinstance(simulator, dict):
        raise MechanismAnalysisError("branch state lacks state or simulator configuration")
    constants = simulator.get("simulator_constants")
    if not isinstance(constants, dict):
        raise MechanismAnalysisError("branch state lacks simulator constants")
    x = _parse_float(state.get("position_x"), "position_x")
    y = _parse_float(state.get("position_y"), "position_y")
    vx = _parse_float(state.get("velocity_x"), "velocity_x")
    vy = _parse_float(state.get("velocity_y"), "velocity_y")
    target_radius = _parse_float(constants.get("target_radius"), "target_radius")
    target_speed = _parse_float(
        constants.get("target_circular_speed"), "target_circular_speed"
    )
    mu = _parse_float(constants.get("mu"), "mu")
    radius = math.hypot(x, y)
    if radius == 0.0:
        raise MechanismAnalysisError("branch-point radius is zero")
    radial_x, radial_y = x / radius, y / radius
    tangential_x, tangential_y = -radial_y, radial_x
    radial_velocity = vx * radial_x + vy * radial_y
    tangential_velocity = vx * tangential_x + vy * tangential_y
    speed = math.hypot(vx, vy)
    return {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "radius": radius,
        "radial_x": radial_x,
        "radial_y": radial_y,
        "tangential_x": tangential_x,
        "tangential_y": tangential_y,
        "radial_velocity": radial_velocity,
        "tangential_velocity": tangential_velocity,
        "target_radius": target_radius,
        "target_speed": target_speed,
        "mu": mu,
        "speed_ratio": speed / target_speed,
        "specific_energy": 0.5 * speed * speed - mu / radius,
        "target_specific_energy": -mu / (2.0 * target_radius),
    }


def _wrapped_angle_difference(first: float, second: float) -> float:
    return abs((second - first + math.pi) % (2.0 * math.pi) - math.pi)


def summarize_actions(
    events: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    actions = [
        action
        for action in (_action(event.get("executed_action"), "executed_action") for event in events)
        if action is not None
    ]
    if not actions:
        return {
            "count": 0,
            "zero_count": 0,
            "mean_magnitude": None,
            "minimum_magnitude": None,
            "maximum_magnitude": None,
            "effort": None,
            "direction_variation": None,
            "direction_change_count": 0,
            "direction_flip_count": 0,
            "first_action": None,
            "final_action": None,
        }
    magnitudes = [math.hypot(*action) for action in actions]
    angles = [
        math.atan2(action[1], action[0]) if magnitude > 0.0 else None
        for action, magnitude in zip(actions, magnitudes)
    ]
    differences = [
        _wrapped_angle_difference(first, second)
        for first, second in zip(angles, angles[1:])
        if first is not None and second is not None
    ]
    return {
        "count": len(actions),
        "zero_count": sum(magnitude == 0.0 for magnitude in magnitudes),
        "mean_magnitude": sum(magnitudes) / len(magnitudes),
        "minimum_magnitude": min(magnitudes),
        "maximum_magnitude": max(magnitudes),
        "effort": sum(magnitudes),
        "direction_variation": sum(differences) if differences else None,
        "direction_change_count": sum(value > 1.0e-12 for value in differences),
        "direction_flip_count": sum(value > math.pi / 2.0 for value in differences),
        "first_action": actions[0],
        "final_action": actions[-1],
    }


def summarize_speed_ratios(
    events: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    pairs = [
        (
            _parse_int(event.get("post_branch_step"), "post_branch_step"),
            _parse_float(event.get("realized_speed_ratio"), "realized_speed_ratio"),
            _parse_float(event.get("predicted_speed_ratio"), "predicted_speed_ratio"),
            _parse_float(event.get("hazard_threshold"), "hazard_threshold"),
        )
        for event in events
        if event.get("realized_speed_ratio") is not None
    ]
    if not pairs:
        return {
            "initial": None,
            "final": None,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "minimum_headroom": None,
            "final_headroom": None,
            "closest_threshold_step": None,
            "trend": UNAVAILABLE,
            "increase_count": 0,
            "decrease_count": 0,
            "near_threshold_count": 0,
            "prediction_realization_error": None,
        }
    values = [item[1] for item in pairs]
    increases = sum(second > first for first, second in zip(values, values[1:]))
    decreases = sum(second < first for first, second in zip(values, values[1:]))
    if increases == 0 and decreases > 0:
        trend = "strictly_decreasing"
    elif decreases == 0 and increases > 0:
        trend = "strictly_increasing"
    elif increases == 0 and decreases == 0:
        trend = "constant"
    else:
        trend = "non_monotonic"
    headrooms = [threshold - realized for _, realized, _, threshold in pairs]
    closest_index = min(range(len(headrooms)), key=headrooms.__getitem__)
    return {
        "initial": values[0],
        "final": values[-1],
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
        "minimum_headroom": min(headrooms),
        "final_headroom": headrooms[-1],
        "closest_threshold_step": pairs[closest_index][0],
        "trend": trend,
        "increase_count": increases,
        "decrease_count": decreases,
        "near_threshold_count": sum(
            0.0 <= headroom <= EXPLORATORY_NEAR_THRESHOLD_HEADROOM
            for headroom in headrooms
        ),
        "prediction_realization_error": max(
            abs(realized - predicted) for _, realized, predicted, _ in pairs
        ),
    }


def summarize_monitor_decisions(
    events: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    decisions = [
        (
            _parse_int(event.get("post_branch_step"), "post_branch_step"),
            event.get("final_veto_decision"),
        )
        for event in events
        if event.get("final_veto_decision") in {"allow", "veto"}
    ]
    veto_steps = [step for step, decision in decisions if decision == "veto"]
    longest = 0
    current = 0
    for _, decision in decisions:
        if decision == "veto":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    evaluations = len(decisions)
    allows = sum(decision == "allow" for _, decision in decisions)
    vetoes = len(veto_steps)
    return {
        "evaluation_count": evaluations,
        "allow_count": allows,
        "veto_count": vetoes,
        "allow_rate": allows / evaluations if evaluations else None,
        "intervention_rate": vetoes / evaluations if evaluations else None,
        "first_veto_step": veto_steps[0] if veto_steps else None,
        "longest_veto_streak": longest,
    }


def _json_action(value: object) -> str | None:
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def compute_branch_metrics(bundle: SourceBundle) -> tuple[Mapping[str, object], ...]:
    point = _branch_point_quantities(bundle.branch_state)
    target_radius = point["target_radius"]
    target_speed = point["target_speed"]
    initial_radius_error = point["radius"] - target_radius
    r_max, vr_max, vt_max = 0.0025, 0.02, 0.25
    output: list[Mapping[str, object]] = []
    for order, (row, branch) in enumerate(zip(bundle.result_rows, BRANCH_ORDER), start=1):
        events = bundle.events_by_branch[branch]
        actions = summarize_actions(events)
        speeds = summarize_speed_ratios(events)
        monitor = summarize_monitor_decisions(events)
        evaluations = int(monitor["evaluation_count"])
        allows = int(monitor["allow_count"])
        vetoes = int(monitor["veto_count"])
        final_radius_error = _parse_float(row["final_radius_error"], "final_radius_error")
        final_vr = _parse_float(
            row["final_radial_velocity_error"], "final_radial_velocity_error"
        )
        final_vt_error = _parse_float(
            row["final_tangential_velocity_error"], "final_tangential_velocity_error"
        )
        final_radius = target_radius + final_radius_error
        final_vt = target_speed + final_vt_error
        final_energy = 0.5 * (final_vr * final_vr + final_vt * final_vt) - point["mu"] / final_radius
        final_r_ratio = final_radius_error / target_radius
        final_vr_ratio = final_vr / target_speed
        final_vt_ratio = final_vt_error / target_speed
        first_action = actions["first_action"]
        first_radial = first_tangential = None
        if first_action is not None:
            first_radial = (
                first_action[0] * point["radial_x"]
                + first_action[1] * point["radial_y"]
            )
            first_tangential = (
                first_action[0] * point["tangential_x"]
                + first_action[1] * point["tangential_y"]
            )
        gap_reduction = abs(initial_radius_error) - abs(final_radius_error)
        metric = {
            "branch_id": branch,
            "frozen_branch_order": order,
            "branch_state_hash": row["branch_state_hash"],
            "manifest_hash": row["manifest_hash"],
            "implementation_commit": row["implementation_commit"],
            "seed": _parse_int(row["seed"], "seed"),
            "branch_step": _parse_int(row["branch_step"], "branch_step"),
            "nominal_prefix_transition_count": _parse_int(
                row["nominal_prefix_transition_count"], "nominal_prefix_transition_count"
            ),
            "terminal_reason": row["terminal_reason"],
            "recovery_transition_count": _parse_int(
                row["recovery_transition_count"], "recovery_transition_count"
            ),
            "total_transition_count": _parse_int(
                row["total_transition_count"], "total_transition_count"
            ),
            "overspeed_status": row["overspeed_status"],
            "instability_status": row["instability_status"],
            "unsafe_state_status": row["unsafe_state_status"],
            "invalid_simulation_status": row["invalid_simulation_status"],
            "target_radius_crossing": _parse_bool(
                row["crossed_target_radius"], "crossed_target_radius"
            ),
            "recoverable_crossing": _parse_bool(
                row["phase34_compatible_recoverable_crossing"],
                "phase34_compatible_recoverable_crossing",
            ),
            "recovery_success_v0": _parse_bool(row["recovery_success"], "recovery_success"),
            "final_simulator_success": _parse_bool(
                row["final_simulator_success"], "final_simulator_success"
            ),
            "monitor_evaluation_count": evaluations,
            "allow_count": allows,
            "veto_count": vetoes,
            "allow_rate": monitor["allow_rate"],
            "intervention_rate": monitor["intervention_rate"],
            "first_veto_step": monitor["first_veto_step"],
            "longest_veto_streak": monitor["longest_veto_streak"],
            "action_count": actions["count"],
            "zero_action_count": actions["zero_count"],
            "mean_action_magnitude": actions["mean_magnitude"],
            "minimum_action_magnitude": actions["minimum_magnitude"],
            "maximum_action_magnitude": actions["maximum_magnitude"],
            "recomputed_cumulative_normalized_effort": actions["effort"],
            "published_cumulative_normalized_effort": _optional_float(
                row["normalized_control_effort"], "normalized_control_effort"
            ),
            "action_direction_variation_radians": actions["direction_variation"],
            "action_direction_change_count": actions["direction_change_count"],
            "action_direction_flip_count_over_90deg": actions["direction_flip_count"],
            "first_action": _json_action(actions["first_action"]),
            "final_action": _json_action(actions["final_action"]),
            "first_action_radial_component_at_branch_point": first_radial,
            "first_action_tangential_component_at_branch_point": first_tangential,
            "trajectory_action_radial_component": None,
            "trajectory_action_tangential_component": None,
            "initial_post_branch_speed_ratio": speeds["initial"],
            "final_speed_ratio": speeds["final"],
            "minimum_speed_ratio": speeds["minimum"],
            "maximum_speed_ratio": speeds["maximum"],
            "mean_speed_ratio": speeds["mean"],
            "minimum_overspeed_headroom": speeds["minimum_headroom"],
            "final_overspeed_headroom": speeds["final_headroom"],
            "closest_threshold_step": speeds["closest_threshold_step"],
            "speed_ratio_trend": speeds["trend"],
            "speed_ratio_increase_count": speeds["increase_count"],
            "speed_ratio_decrease_count": speeds["decrease_count"],
            "exploratory_near_threshold_event_count": speeds["near_threshold_count"],
            "exploratory_near_threshold_headroom_band": EXPLORATORY_NEAR_THRESHOLD_HEADROOM,
            "maximum_prediction_realization_error": speeds[
                "prediction_realization_error"
            ],
            "initial_target_radius_error": initial_radius_error,
            "final_target_radius_error": final_radius_error,
            "target_radius_gap_reduction": gap_reduction,
            "target_radius_gap_reduction_fraction": (
                gap_reduction / abs(initial_radius_error)
                if initial_radius_error != 0.0
                else None
            ),
            "minimum_absolute_target_radius_error": None,
            "closest_target_approach_step": None,
            "initial_radial_velocity_error": point["radial_velocity"],
            "final_radial_velocity_error": final_vr,
            "minimum_radial_velocity_error": None,
            "initial_tangential_velocity_error": point["tangential_velocity"] - target_speed,
            "final_tangential_velocity_error": final_vt_error,
            "minimum_tangential_velocity_error": None,
            "first_crossing_step": _optional_int(row.get("first_crossing_step"), "first_crossing_step"),
            "first_recoverable_crossing_step": _optional_int(
                row.get("first_recoverable_crossing_step"),
                "first_recoverable_crossing_step",
            ),
            "initial_r_error_ratio": initial_radius_error / target_radius,
            "final_r_error_ratio": final_r_ratio,
            "initial_vr_ratio": point["radial_velocity"] / target_speed,
            "final_vr_ratio": final_vr_ratio,
            "initial_vt_error_ratio": (
                point["tangential_velocity"] - target_speed
            ) / target_speed,
            "final_vt_error_ratio": final_vt_ratio,
            "final_r_component_margin": r_max - abs(final_r_ratio),
            "final_vr_component_margin": vr_max - abs(final_vr_ratio),
            "final_vt_component_margin": vt_max - abs(final_vt_ratio),
            "closest_recoverability_component_margins": None,
            "initial_specific_orbital_energy_j_per_kg": point["specific_energy"],
            "final_specific_orbital_energy_j_per_kg": final_energy,
            "specific_orbital_energy_change_j_per_kg": final_energy - point["specific_energy"],
            "target_circular_specific_energy_j_per_kg": point["target_specific_energy"],
            "final_energy_difference_from_target_j_per_kg": final_energy
            - point["target_specific_energy"],
            "energy_derivation_basis": (
                "derived_endpoint_quantity_from_frozen_mu_target_radius_target_speed_"
                "branch_point_state_and_published_final_radial_tangential_summaries"
            ),
            "published_delta_v_proxy": _optional_float(row["delta_v_proxy"], "delta_v_proxy"),
        }
        output.append(metric)
    return tuple(output)


def extract_checkpoints(bundle: SourceBundle) -> tuple[Mapping[str, object], ...]:
    output: list[Mapping[str, object]] = []
    for branch in BRANCH_ORDER:
        events = bundle.events_by_branch[branch]
        desired = (0,) if branch == "explicit_abort_v0" else CHECKPOINT_STEPS
        by_step = {event.get("post_branch_step"): event for event in events}
        for step in desired:
            event = by_step.get(step)
            if event is None:
                continue
            action = _action(event.get("executed_action"), "executed_action")
            realized = _optional_float(event.get("realized_speed_ratio"), "realized_speed_ratio")
            threshold = _parse_float(event.get("hazard_threshold"), "hazard_threshold")
            statuses = event.get("evaluator_statuses")
            recovery_status = UNAVAILABLE
            if isinstance(statuses, list):
                for pair in statuses:
                    if isinstance(pair, list) and len(pair) == 2 and pair[0] == "recovery_success":
                        recovery_status = str(pair[1])
                        break
            output.append(
                {
                    "branch_id": branch,
                    "recovery_step": step,
                    "total_transition_count": event.get("total_transition_count"),
                    "current_state_hash": event.get("current_state_hash"),
                    "next_state_hash": event.get("next_state_hash") or UNAVAILABLE,
                    "action": _json_action(action) or UNAVAILABLE,
                    "action_magnitude": math.hypot(*action) if action is not None else None,
                    "final_veto_decision": event.get("final_veto_decision"),
                    "predicted_speed_ratio": event.get("predicted_speed_ratio"),
                    "realized_speed_ratio": event.get("realized_speed_ratio"),
                    "overspeed_headroom": threshold - realized if realized is not None else None,
                    "radius": None,
                    "target_radius_error": None,
                    "radial_velocity": None,
                    "tangential_velocity": None,
                    "recovery_evaluator_status": recovery_status,
                    "stop_condition": event.get("triggered_stop_condition") or "none",
                }
            )
    return tuple(output)


def analyze_trajectory_divergence(bundle: SourceBundle) -> Mapping[str, object]:
    events = {branch: bundle.events_by_branch[branch] for branch in PHYSICAL_BRANCHES}
    initial_hashes = {branch: events[branch][0]["current_state_hash"] for branch in PHYSICAL_BRANCHES}
    first_divergence: int | None = None
    for index in range(min(len(events[branch]) for branch in PHYSICAL_BRANCHES)):
        next_hashes = {events[branch][index].get("next_state_hash") for branch in PHYSICAL_BRANCHES}
        if len(next_hashes) > 1:
            first_divergence = index + 1
            break

    pairwise: dict[str, object] = {}
    for first_index, first in enumerate(PHYSICAL_BRANCHES):
        for second in PHYSICAL_BRANCHES[first_index + 1 :]:
            first_states = {
                event.get("next_state_hash")
                for event in events[first]
                if event.get("next_state_hash")
            }
            second_states = {
                event.get("next_state_hash")
                for event in events[second]
                if event.get("next_state_hash")
            }
            exact_later_matches = sorted(first_states & second_states)
            first_actions = [
                _action(event.get("executed_action"), "executed_action")
                for event in events[first]
            ]
            second_actions = [
                _action(event.get("executed_action"), "executed_action")
                for event in events[second]
            ]
            same_action = 0
            opposite_action = 0
            cosines: list[float] = []
            for action_a, action_b in zip(first_actions, second_actions):
                if action_a is None or action_b is None:
                    continue
                if action_a == action_b:
                    same_action += 1
                if all(abs(a + b) <= 1.0e-12 for a, b in zip(action_a, action_b)):
                    opposite_action += 1
                norm_a, norm_b = math.hypot(*action_a), math.hypot(*action_b)
                if norm_a > 0.0 and norm_b > 0.0:
                    cosines.append(
                        (action_a[0] * action_b[0] + action_a[1] * action_b[1])
                        / (norm_a * norm_b)
                    )
            pairwise[f"{first}__vs__{second}"] = {
                "exact_later_state_hash_match_count": len(exact_later_matches),
                "exact_later_state_hash_match_observed": bool(exact_later_matches),
                "same_step_exact_action_count": same_action,
                "same_step_opposite_action_count_tolerance_1e-12": opposite_action,
                "action_direction_cosine_minimum": min(cosines) if cosines else None,
                "action_direction_cosine_maximum": max(cosines) if cosines else None,
                "action_direction_cosine_mean": (
                    sum(cosines) / len(cosines) if cosines else None
                ),
            }
    return {
        "common_pre_transition_state_at_step_1": len(set(initial_hashes.values())) == 1,
        "initial_current_state_hashes": initial_hashes,
        "first_next_state_hash_divergence_step": first_divergence,
        "different_hash_interpretation": (
            "exact_state_identity_differs; physical distance is not derivable from hashes"
        ),
        "pairwise": pairwise,
    }


def classify_mechanisms(
    metrics: Sequence[Mapping[str, object]],
    divergence: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    by_branch = {str(metric["branch_id"]): metric for metric in metrics}
    zero = by_branch["zero_action_reference_v0"]
    velocity = by_branch["velocity_opposed_thrust_v0"]
    tangential = by_branch["tangential_error_correction_v0"]
    common_scope = (
        "one frozen branch state, three declared physical responses, magnitude 0.25 where "
        "applicable, 10000-transition horizon, current simulator"
    )
    return (
        {
            "mechanism_id": "A_hazard_only_correction",
            "status": "supported",
            "supporting_evidence": [
                "All 30000 physical proposals were allowed and all realized speed ratios remained below 1.90.",
                "All three physical branches ended without crossing, recoverable crossing, Recovery Success v0, or simulator success.",
            ],
            "counterevidence": [],
            "missing_evidence": [
                "No post-branch execution of the rejected nominal source action exists as a branch counterfactual."
            ],
            "confidence": "high",
            "scope": common_scope,
        },
        {
            "mechanism_id": "B_insufficient_radial_commitment",
            "status": "partially_supported",
            "supporting_evidence": [
                f"Velocity-opposed final radial velocity fell from {velocity['initial_radial_velocity_error']:.6f} to {velocity['final_radial_velocity_error']:.6f} m/s and reduced only {velocity['target_radius_gap_reduction_fraction']:.6%} of the initial radius gap.",
                f"Zero action and tangential correction reduced only {zero['target_radius_gap_reduction_fraction']:.6%} and {tangential['target_radius_gap_reduction_fraction']:.6%} of the endpoint radius gap.",
                "All final radius and radial-velocity recoverability component margins remained negative.",
            ],
            "counterevidence": [
                "Zero action and tangential correction retained positive outward endpoint radial velocity."
            ],
            "missing_evidence": [
                "Per-step radius and radial velocity are not available, so the closest radial approach and near-horizon radial trend cannot be evaluated."
            ],
            "confidence": "medium",
            "scope": common_scope,
        },
        {
            "mechanism_id": "C_excessive_braking_or_energy_removal",
            "status": "supported",
            "supporting_evidence": [
                f"Velocity-opposed speed ratio fell from {velocity['initial_post_branch_speed_ratio']:.6f} to {velocity['final_speed_ratio']:.6f}.",
                f"Its derived endpoint specific orbital energy changed by {velocity['specific_orbital_energy_change_j_per_kg']:.6f} J/kg and ended {velocity['final_energy_difference_from_target_j_per_kg']:.6f} J/kg below the target circular energy.",
                f"Endpoint radial velocity was {velocity['final_radial_velocity_error']:.6f} m/s and tangential error was {velocity['final_tangential_velocity_error']:.6f} m/s, showing suppression of both components.",
            ],
            "counterevidence": [],
            "missing_evidence": [
                "Per-step energy is unavailable; only branch-point and final energy are derivable."
            ],
            "confidence": "high",
            "scope": "velocity_opposed_thrust_v0 for this frozen state and horizon",
        },
        {
            "mechanism_id": "D_tangential_only_correction_limitation",
            "status": "supported",
            "supporting_evidence": [
                f"Tangential endpoint error changed from {tangential['initial_tangential_velocity_error']:.6f} to {tangential['final_tangential_velocity_error']:.6f} m/s while no crossing occurred.",
                f"Its final radial velocity remained {tangential['final_radial_velocity_error']:.6f} m/s and its final radius gap remained {abs(tangential['final_target_radius_error']):.6f} m.",
                "The frozen branch definition supplies tangential correction without a radial correction term.",
            ],
            "counterevidence": [],
            "missing_evidence": [
                "Per-step radial and tangential state components are unavailable."
            ],
            "confidence": "high",
            "scope": "tangential_error_correction_v0 for this frozen state and horizon",
        },
        {
            "mechanism_id": "E_static_action_limitation",
            "status": "not_supported",
            "supporting_evidence": [
                "The policies remain single-mode and fixed-magnitude where nonzero."
            ],
            "counterevidence": [
                f"Velocity-opposed and tangential actions changed recorded direction on {velocity['action_direction_change_count']} and {tangential['action_direction_change_count']} consecutive boundaries; they were recomputed from current state rather than repeating one fixed vector."
            ],
            "missing_evidence": [
                "No staged or phase-switching recovery policy was tested."
            ],
            "confidence": "high",
            "scope": common_scope,
        },
        {
            "mechanism_id": "F_horizon_limitation",
            "status": "not_evaluable",
            "supporting_evidence": [],
            "counterevidence": [
                "Recovery-horizon exhaustion alone does not establish that a longer horizon would recover."
            ],
            "missing_evidence": [
                "Per-step target-radius and recoverability-component trends near transition 10000 are unavailable."
            ],
            "confidence": "high",
            "scope": common_scope,
        },
        {
            "mechanism_id": "G_state_region_irrecoverability_under_tested_actions",
            "status": "consistent_with_evidence",
            "supporting_evidence": [
                "None of the three declared physical responses recovered from the identical branch state within 10000 transitions."
            ],
            "counterevidence": [
                "Only three simple response policies were evaluated; no adaptive, staged, or optimized policy was tested."
            ],
            "missing_evidence": [
                "No reachable-set or controller-complete irrecoverability analysis exists."
            ],
            "confidence": "medium",
            "scope": common_scope,
        },
        {
            "mechanism_id": "H_missing_phase_switching",
            "status": "consistent_with_evidence",
            "supporting_evidence": [
                "Each physical branch used one response rule for the entire horizon and none reached crossing.",
                "The active branches corrected different state components but did not jointly satisfy radius, radial-velocity, and tangential-velocity recoverability conditions."
            ],
            "counterevidence": [
                "No phase-switching policy was measured, so its benefit is untested."
            ],
            "missing_evidence": [
                "No logged policy-phase transitions or alternate staged trajectory exist."
            ],
            "confidence": "medium",
            "scope": "architectural hypothesis motivated by this one-case outcome",
        },
        {
            "mechanism_id": "I_insufficient_observability_in_published_artifacts",
            "status": "supported",
            "supporting_evidence": [
                "Decision events contain state hashes, actions, and speed ratios but no per-step state vectors.",
                "Per-step radius, radial velocity, tangential velocity, orbital energy, and recoverability components cannot be calculated from hashes.",
            ],
            "counterevidence": [
                "Branch-point and final scalar summaries permit endpoint comparisons."
            ],
            "missing_evidence": [
                "A compact checkpoint state-summary stream was not published."
            ],
            "confidence": "high",
            "scope": "physical mechanism diagnosis from the published v0 artifact schema",
        },
    )


def _csv_value(value: object) -> object:
    if value is None:
        return UNAVAILABLE
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def csv_bytes(rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    return buffer.getvalue().encode("utf-8")


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _fmt(value: object) -> str:
    if value is None:
        return UNAVAILABLE
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def render_summary(
    bundle: SourceBundle,
    metrics: Sequence[Mapping[str, object]],
    divergence: Mapping[str, object],
    mechanisms: Sequence[Mapping[str, object]],
) -> bytes:
    by_branch = {str(metric["branch_id"]): metric for metric in metrics}
    lines = [
        "# Recovery Branch Mechanism Diagnosis v0",
        "",
        "## Status",
        "",
        "Published recovery evidence analyzed; no new trajectory executed.",
        "",
        "Completed: 2026-07-25",
        "",
        "## Source Evidence",
        "",
        f"- Result commit: `{RESULT_COMMIT}`",
        f"- Implementation commit: `{IMPLEMENTATION_COMMIT}`",
        f"- Branch-state canonical hash: `{bundle.branch_state['canonical_branch_state_hash']}`",
        f"- Manifest canonical hash: `{canonical_sha256(bundle.manifest)}`",
        f"- Branch records: `{len(bundle.result_rows)}`",
        f"- Decision events: `{sum(len(bundle.events_by_branch[b]) for b in BRANCH_ORDER)}`",
    ]
    for name in SOURCE_FILENAMES:
        lines.append(f"- `{name}` SHA-256: `{bundle.source_hashes[name]}`")
    lines += [
        "",
        "## Structural Validation",
        "",
        "- Four frozen branches: confirmed in frozen order.",
        "- Common branch-point Cartesian state: confirmed by the recovery-step 1 current-state hash.",
        "- Event counts: 10,000 zero-action, 10,000 velocity-opposed, 10,000 tangential-correction, and one explicit-abort event.",
        "- Source artifacts were read only. No rollout, interpolation, state reconstruction, or branch execution was performed.",
        "",
        "## Branch-By-Branch Findings",
        "",
        "### Zero Action",
        "",
    ]
    zero = by_branch[BRANCH_ORDER[0]]
    velocity = by_branch[BRANCH_ORDER[1]]
    tangential = by_branch[BRANCH_ORDER[2]]
    abort = by_branch[BRANCH_ORDER[3]]
    lines += [
        f"Zero action remained numerically ballistic at the endpoint: derived specific orbital energy changed by only `{_fmt(zero['specific_orbital_energy_change_j_per_kg'])}` J/kg and remained `{zero['final_energy_difference_from_target_j_per_kg']:.6f}` J/kg above the target circular energy. Radius moved toward the outer target by `{_fmt(zero['target_radius_gap_reduction'])}` m, only `{zero['target_radius_gap_reduction_fraction']:.6%}` of the initial gap, while the final radius gap remained `{abs(zero['final_target_radius_error']):.6f}` m. Speed ratio declined slightly from `{zero['initial_post_branch_speed_ratio']:.9f}` to `{zero['final_speed_ratio']:.9f}` and remained within the exploratory 0.05 headroom band for all 10,000 events. The artifact does not contain per-step radius, so it cannot establish closest approach, a stable stalled region, or eventual recovery under a longer horizon.",
        "",
        "### Velocity-Opposed Thrust",
        "",
        f"Velocity-opposed thrust reduced speed ratio from `{velocity['initial_post_branch_speed_ratio']:.9f}` to `{velocity['final_speed_ratio']:.9f}`. It reduced endpoint radial velocity from `{velocity['initial_radial_velocity_error']:.6f}` to `{velocity['final_radial_velocity_error']:.6f}` m/s and tangential velocity error from `{velocity['initial_tangential_velocity_error']:.6f}` to `{velocity['final_tangential_velocity_error']:.6f}` m/s. Derived endpoint specific orbital energy ended `{velocity['final_energy_difference_from_target_j_per_kg']:.6f}` J/kg below the target circular energy. It therefore suppressed useful and hazardous motion together without restoring target geometry in this case.",
        "",
        "### Tangential Correction",
        "",
        f"Tangential correction changed endpoint tangential error from `{tangential['initial_tangential_velocity_error']:.6f}` to `{tangential['final_tangential_velocity_error']:.6f}` m/s, but final radial velocity remained `{tangential['final_radial_velocity_error']:.6f}` m/s and `{abs(tangential['final_target_radius_error']):.6f}` m of radius gap remained. Its speed ratio declined from `{tangential['initial_post_branch_speed_ratio']:.9f}` to `{tangential['final_speed_ratio']:.9f}`, while endpoint specific orbital energy remained `{tangential['final_energy_difference_from_target_j_per_kg']:.6f}` J/kg above the target circular value. This supports tangential-component correction without task-geometry recovery, not a claim about all tangential policies.",
        "",
        "### Explicit Abort",
        "",
        f"Explicit abort executed `{abort['recovery_transition_count']}` recovery transitions and terminated as `{abort['terminal_reason']}`. It prevented further exposure through termination and did not provide task recovery.",
        "",
        "## Cross-Branch Findings",
        "",
        f"- The three physical branches shared the exact recovery-step 1 current-state hash and first diverged in next-state hash at recovery step `{divergence['first_next_state_hash_divergence_step']}`.",
        "- No exact recorded state-hash convergence occurred between physical branches after divergence. Different hashes establish nonidentity, not physical distance.",
        "- Velocity-opposed and tangential correction each used magnitude 0.25 for 10,000 transitions. Their equal effort of 2,500 and equal delta-v proxy follow from the same norm and duration; their action directions and state hashes were distinct.",
        "- The two active branches never proposed exactly equal or exactly opposite recorded actions at the same step. Their state-dependent directions changed at every consecutive boundary, so the experiment tested persistent single-mode rules, not fixed inertial action vectors.",
        f"- Velocity-opposed final speed ratio was `{velocity['final_speed_ratio']:.9f}`; tangential-correction final speed ratio was `{tangential['final_speed_ratio']:.9f}`. Equal scalar cost did not produce equivalent trajectories.",
        f"- Velocity-opposed improved the endpoint radial-velocity ratio to `{velocity['final_vr_ratio']:.9f}` but degraded tangential error ratio to `{velocity['final_vt_error_ratio']:.9f}`; tangential correction improved tangential error ratio to `{tangential['final_vt_error_ratio']:.9f}` while radial-velocity ratio remained `{tangential['final_vr_ratio']:.9f}`. All endpoint radius and radial-velocity margins remained outside the Phase34-compatible limits.",
        "- Final Veto allowed all 30,000 physical proposals. Post-branch stalling was not caused by repeated veto intervention.",
        "- Endpoint radius summaries show limited net progress, but per-step target geometry is unavailable. No closest-approach or longer-horizon recovery claim is supported.",
        "",
        "## Mechanism Diagnosis",
        "",
        "### Directly Supported",
        "",
    ]
    for item in mechanisms:
        if item["status"] == "supported":
            lines.append(f"- `{item['mechanism_id']}`: {item['supporting_evidence'][0]}")
    lines += [
        "",
        "### Consistent Or Partial",
        "",
    ]
    for item in mechanisms:
        if item["status"] in {"partially_supported", "consistent_with_evidence"}:
            lines.append(
                f"- `{item['mechanism_id']}` (`{item['status']}`): {item['supporting_evidence'][0]}"
            )
    lines += [
        "",
        "### Unevaluable Or Unsupported",
        "",
    ]
    for item in mechanisms:
        if item["status"] in {"not_evaluable", "not_supported"}:
            evidence = item["counterevidence"] or item["missing_evidence"]
            lines.append(f"- `{item['mechanism_id']}` (`{item['status']}`): {evidence[0]}")
    lines += [
        "",
        "## Strongest Supported Conclusion",
        "",
        "For this frozen one-case diagnostic, the three tested physical responses prevented realized overspeed but did not restore target crossing or Phase34-compatible recoverability within 10,000 transitions. Zero action retained the branch-point energy and made limited endpoint radius progress; velocity-opposed thrust over-suppressed radial and tangential motion; tangential correction improved the endpoint tangential component without resolving the radius and radial-velocity components. These are endpoint and logged-speed findings, not a proof that the state is irrecoverable under other policies.",
        "",
        "## Next Architecture Requirement",
        "",
        "The next recovery policy should separate hazard arrest from task recovery and use state-dependent staged decisions. At minimum it should monitor radius progress, radial velocity, tangential error, orbital-energy change, and the Phase34 recoverability component vector; stop a single-mode response when progress stalls; and switch deliberately among hazard arrest, radial recommitment, tangential alignment, crossing, retreat, and termination. This is a design requirement inferred from the diagnosed failure modes, not a validated policy.",
        "",
        "## Evidence Limitations",
        "",
        f"Per-step Cartesian state, radius, radial velocity, tangential velocity, target-radius error, orbital energy, and recoverability-component margins are `{UNAVAILABLE}`. State hashes cannot be inverted into those quantities. Only branch-point and final endpoint orbital summaries support physical derivation.",
        "",
        "## Claim Restrictions",
        "",
        "This analysis does not establish branch optimality, universal failure of velocity-opposed thrust, universal failure of tangential correction, state irrecoverability under all controllers, formal safety, hardware validity, benchmark-wide recovery performance, recovery under a longer horizon, or success of any proposed future policy.",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def analysis_text_artifacts(bundle: SourceBundle) -> Mapping[str, bytes]:
    require_structural_validity(bundle)
    inventory = build_field_inventory(bundle)
    metrics = compute_branch_metrics(bundle)
    checkpoints = extract_checkpoints(bundle)
    divergence = analyze_trajectory_divergence(bundle)
    mechanisms = classify_mechanisms(metrics, divergence)
    evidence = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "trajectory_divergence": divergence,
        "mechanisms": list(mechanisms),
        "interpretation_rule": (
            "measured and endpoint-derived evidence is separated from hypotheses; "
            "unavailable quantities remain explicit"
        ),
    }
    return {
        "field_inventory.json": json_bytes(inventory),
        "branch_metrics.csv": csv_bytes(metrics, BRANCH_METRIC_FIELDS),
        "checkpoint_comparison.csv": csv_bytes(checkpoints, CHECKPOINT_FIELDS),
        "mechanism_evidence.json": json_bytes(evidence),
        "summary.md": render_summary(bundle, metrics, divergence, mechanisms),
    }


def _configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 100,
            "savefig.dpi": 120,
        }
    )
    return plt


def write_speed_ratio_plot(path: Path, bundle: SourceBundle) -> None:
    plt = _configure_matplotlib()
    colors = ("#4C78A8", "#E45756", "#54A24B")
    fig, axis = plt.subplots(figsize=(10.0, 5.5), constrained_layout=True)
    try:
        for branch, color in zip(PHYSICAL_BRANCHES, colors):
            events = bundle.events_by_branch[branch]
            steps = [int(event["post_branch_step"]) for event in events]
            values = [float(event["realized_speed_ratio"]) for event in events]
            axis.plot(steps, values, color=color, linewidth=1.1, label=branch)
        axis.axhline(1.9, color="#222222", linestyle="--", linewidth=1.0, label="overspeed threshold 1.90")
        axis.set_title("Measured speed ratio, frozen one-case recovery diagnostic")
        axis.set_xlabel("Recovery transition")
        axis.set_ylabel("Realized speed ratio")
        axis.set_xlim(1, 10000)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        fig.savefig(
            path,
            format="png",
            dpi=120,
            metadata={"Software": "recovery-branch-mechanism-diagnosis-v0"},
        )
    finally:
        plt.close(fig)


def write_action_geometry_plot(path: Path, bundle: SourceBundle) -> None:
    plt = _configure_matplotlib()
    colors = ("#4C78A8", "#E45756", "#54A24B")
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True, constrained_layout=True)
    try:
        for branch, color in zip(PHYSICAL_BRANCHES, colors):
            events = bundle.events_by_branch[branch]
            steps = [int(event["post_branch_step"]) for event in events]
            actions = [_action(event.get("executed_action"), "executed_action") for event in events]
            axes[0].plot(steps, [action[0] for action in actions if action], color=color, linewidth=0.8, label=branch)
            axes[1].plot(steps, [action[1] for action in actions if action], color=color, linewidth=0.8, label=branch)
        axes[0].set_title("Measured inertial action components, frozen one-case diagnostic")
        axes[0].set_ylabel("Normalized action x")
        axes[1].set_ylabel("Normalized action y")
        axes[1].set_xlabel("Recovery transition")
        for axis in axes:
            axis.set_xlim(1, 10000)
            axis.set_ylim(-0.27, 0.27)
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best")
        fig.savefig(
            path,
            format="png",
            dpi=120,
            metadata={"Software": "recovery-branch-mechanism-diagnosis-v0"},
        )
    finally:
        plt.close(fig)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_directory(
    output_directory: Path,
    *,
    repository_root: Path,
    source_directory: Path,
) -> Path:
    root = repository_root.resolve()
    output = output_directory.resolve()
    source = source_directory.resolve()
    protected = (
        source,
        root / "analysis" / "final_veto_ablation_v0",
        root / "analysis" / "phase34_post_cross_sync",
        root / "analysis" / "phase35_crossing_basin_expansion",
        root / "analysis" / "phase36b_transfer_family_benchmark",
        root / "analysis" / "phase36c_non_crossing_geometry_diagnosis",
        root / "analysis" / "phase37a_radial_commit_timing",
        root / "analysis" / "phase37b_weak_tangential_subset",
    )
    if not _is_within(output, root):
        raise MechanismAnalysisError("analysis output must remain inside the repository")
    for protected_path in protected:
        protected_resolved = protected_path.resolve()
        if output == protected_resolved or _is_within(output, protected_resolved):
            raise MechanismAnalysisError(
                f"analysis output overlaps a frozen or protected directory: {protected_path}"
            )
    if output.exists():
        raise MechanismAnalysisError(f"refusing to overwrite existing analysis output: {output}")
    if not output.parent.is_dir():
        raise MechanismAnalysisError(f"analysis output parent does not exist: {output.parent}")
    return output


def publish_analysis(
    bundle: SourceBundle,
    output_directory: Path,
    *,
    repository_root: Path,
) -> Mapping[str, str]:
    target = validate_output_directory(
        output_directory,
        repository_root=repository_root,
        source_directory=bundle.source_directory,
    )
    text_artifacts = analysis_text_artifacts(bundle)
    staging = Path(tempfile.mkdtemp(prefix=".recovery-mechanism-", dir=target.parent))
    try:
        for name, payload in text_artifacts.items():
            (staging / name).write_bytes(payload)
        write_speed_ratio_plot(staging / "speed_ratio_trajectory.png", bundle)
        write_action_geometry_plot(staging / "action_geometry_trajectory.png", bundle)
        expected = set(text_artifacts) | {
            "speed_ratio_trajectory.png",
            "action_geometry_trajectory.png",
        }
        actual = {path.name for path in staging.iterdir() if path.is_file()}
        if actual != expected:
            raise MechanismAnalysisError(
                f"staged diagnosis bundle is incomplete: expected={sorted(expected)} actual={sorted(actual)}"
            )
        for path in staging.iterdir():
            if path.is_file() and path.stat().st_size == 0:
                raise MechanismAnalysisError(f"staged analysis artifact is empty: {path.name}")
        staging.rename(target)
        return {
            path.name: sha256_file(path)
            for path in sorted(target.iterdir(), key=lambda item: item.name)
            if path.is_file()
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Analyze frozen recovery-branch evidence without executing a trajectory."
    )
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=root / "analysis" / "recovery_action_branching_nonformal_v0",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=root / "analysis" / "recovery_branch_mechanism_diagnosis_v0",
    )
    parser.add_argument(
        "--allow-create",
        action="store_true",
        help="Create the missing diagnosis directory; existing output is never overwritten.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    bundle = load_source_bundle(args.source_directory, enforce_frozen_hashes=True)
    require_structural_validity(bundle)
    if not args.allow_create:
        print("SOURCE_VALIDATION PASS")
        print("No outputs written. Use --allow-create to create the separate diagnosis bundle.")
        return 0
    hashes = publish_analysis(bundle, args.output_directory, repository_root=root)
    print("SOURCE_VALIDATION PASS")
    print(f"ANALYSIS_OUTPUT {args.output_directory.resolve()}")
    for name, digest in hashes.items():
        print(f"ARTIFACT {name} {digest}")
    print("NO_TRAJECTORY_EXECUTED true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
