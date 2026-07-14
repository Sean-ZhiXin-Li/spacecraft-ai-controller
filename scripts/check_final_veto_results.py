from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_final_veto_manifest import (  # noqa: E402
    MANIFEST_RELATIVE_PATH,
    ManifestValidationError,
    load_manifest,
    validate_manifest_data,
)
from scripts.final_veto_artifacts import (  # noqa: E402
    ARM_DIAGNOSTIC_FIELDNAMES,
    ARM_JSON_LIST_FIELDS,
    ARM_SCHEMA_VERSION,
    COMPACT_DECISION_EVENT_FIELDNAMES,
    DECISION_FIELDNAMES,
    DECISION_SEGMENT_FIELDNAMES,
    DECISION_SCHEMA_VERSION,
    LEGACY_ARM_FIELDNAMES,
    LEGACY_PAIR_FIELDNAMES,
    PAIR_DIAGNOSTIC_FIELDNAMES,
    PAIR_SCHEMA_VERSION,
    TERMINAL_TRANSITION_FIELDNAMES,
    ArtifactWriteError,
    validate_output_directory,
)
from scripts.final_veto_compact_log import (  # noqa: E402
    LOG_MODE_COMPACT,
    LOG_MODE_FULL_TRACE,
    MAX_COMPACT_RECORDS_PER_RUN,
    VALID_DECISION_LOG_MODES,
    exceptional_event_reasons,
)
from scripts.run_final_veto_ablation import (  # noqa: E402
    RunnerContractError,
    build_pair_record,
    build_planned_jobs,
)


PRESERVATION_SUBSET_ID = "phase34_known_recoverable_preservation_v1"
STRESS_SUBSET_ID = "phase35_radial_energy_push_overspeed_stress_v0"
CONTROLLED_TERMINAL_LABELS = frozenset(
    {
        "success",
        "no_crossing",
        "crossing_unrecoverable",
        "recoverable_crossing_failed_late",
        "overspeed",
        "instability",
        "timeout",
        "resource_depletion",
        "unsafe_state",
        "invalid_simulation",
        "unknown",
    }
)
BOOLEAN_ARM_FIELDS = frozenset(
    {
        "monitor_enabled",
        "crossed_target_radius",
        "recoverable_crossing",
        "final_simulator_success",
        "overspeed",
        "instability",
        "unsafe_state",
        "invalid_simulation",
        "is_full_benchmark",
        "known_phase34_recoverable_case",
        "accepted_as_progress",
        "is_formal_experiment",
    }
)
INTEGER_ARM_FIELDS = frozenset(
    {
        "seed",
        "monitor_evaluation_count",
        "allow_count",
        "veto_count",
        "fallback_count",
        "false_negative_count",
        "fallback_failure_count",
        "invalid_monitor_evaluation_count",
        "nominal_actions_unchanged_count",
        "steps",
    }
)
FLOAT_ARM_FIELDS = frozenset(
    {
        "hazard_threshold",
        "r0_over_target",
        "initial_velocity_angle_deg",
        "thrust_scale",
    }
)
INTEGER_ARM_DIAGNOSTIC_FIELDS = frozenset(
    {
        "decision_stream_event_count",
        "compact_decision_record_count",
        "longest_consecutive_veto_steps",
        "longest_consecutive_allow_steps",
        "veto_segment_count",
        "allow_segment_count",
    }
)
OPTIONAL_INTEGER_ARM_DIAGNOSTIC_FIELDS = frozenset(
    {"first_veto_step", "last_veto_step"}
)
OPTIONAL_FLOAT_ARM_DIAGNOSTIC_FIELDS = frozenset(
    {"intervention_rate", "allow_rate", "fallback_rate"}
)
BOOLEAN_PAIR_FIELDS = frozenset(
    {
        "pair_complete",
        "pair_valid",
        "off_overspeed",
        "on_overspeed",
        "off_crossed_target_radius",
        "on_crossed_target_radius",
        "off_recoverable_crossing",
        "on_recoverable_crossing",
        "off_final_simulator_success",
        "on_final_simulator_success",
        "off_invalid_simulation",
        "on_invalid_simulation",
        "avoided_failure",
        "blocked_success",
        "unnecessary_veto",
        "claim_eligible",
        "is_formal_experiment",
    }
)
INTEGER_PAIR_FIELDS = frozenset(
    {
        "seed",
        "on_monitor_evaluation_count",
        "on_allow_count",
        "on_veto_count",
        "on_fallback_count",
        "on_false_negative_count",
        "on_fallback_failure_count",
    }
)
BOOLEAN_PAIR_DIAGNOSTIC_FIELDS = frozenset(
    {
        "terminal_label_changed",
        "task_outcome_preserved",
        "declared_hazard_avoided",
        "task_recovered_after_hazard_avoidance",
    }
)
INTEGER_PAIR_DIAGNOSTIC_FIELDS = frozenset(
    {"step_count_delta", "monitor_induced_horizon_extension"}
)


@dataclass(frozen=True, slots=True)
class ResultValidationReport:
    structural_valid: bool
    pair_complete: bool
    preservation_acceptance: str
    stress_hazard_exercised: str
    positive_claim_eligible: bool
    errors: tuple[str, ...]
    observations: tuple[str, ...]
    metrics: Mapping[str, int]


def _parse_bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise ValueError(f"{field} must be true or false")


def _parse_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        parsed = int(value)
        if str(parsed) == value.strip() or value.strip().lstrip("+").isdigit():
            return parsed
    raise ValueError(f"{field} must be an integer")


def _parse_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc


def _parse_optional_int(value: object, field: str) -> int | None:
    if value is None or value == "":
        return None
    return _parse_int(value, field)


def _parse_optional_float(value: object, field: str) -> float | None:
    if value is None or value == "":
        return None
    return _parse_float(value, field)


def _parse_json_list(value: object, field: str) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must contain valid compact JSON") from exc
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"{field} must be a JSON array")


def _normalize_arm_record(record: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(record)
    for field in BOOLEAN_ARM_FIELDS:
        normalized[field] = _parse_bool(record.get(field), field)
    for field in INTEGER_ARM_FIELDS:
        normalized[field] = _parse_int(record.get(field), field)
    for field in FLOAT_ARM_FIELDS:
        normalized[field] = _parse_float(record.get(field), field)
    for field in ARM_JSON_LIST_FIELDS:
        normalized[field] = _parse_json_list(record.get(field), field)
    if all(field in record for field in ARM_DIAGNOSTIC_FIELDNAMES):
        for field in INTEGER_ARM_DIAGNOSTIC_FIELDS:
            normalized[field] = _parse_int(record.get(field), field)
        for field in OPTIONAL_INTEGER_ARM_DIAGNOSTIC_FIELDS:
            normalized[field] = _parse_optional_int(record.get(field), field)
        for field in OPTIONAL_FLOAT_ARM_DIAGNOSTIC_FIELDS:
            normalized[field] = _parse_optional_float(record.get(field), field)
        normalized["_has_diagnostic_extension"] = True
    else:
        normalized["_has_diagnostic_extension"] = False
    return normalized


def _normalize_pair_record(record: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(record)
    for field in BOOLEAN_PAIR_FIELDS:
        normalized[field] = _parse_bool(record.get(field), field)
    for field in INTEGER_PAIR_FIELDS:
        normalized[field] = _parse_int(record.get(field), field)
    performance = record.get("performance_cost_summary")
    if isinstance(performance, str):
        try:
            performance = json.loads(performance)
        except json.JSONDecodeError as exc:
            raise ValueError("performance_cost_summary must be valid JSON") from exc
    if not isinstance(performance, dict):
        raise ValueError("performance_cost_summary must be a JSON object")
    normalized["performance_cost_summary"] = performance
    if all(field in record for field in PAIR_DIAGNOSTIC_FIELDNAMES):
        for field in BOOLEAN_PAIR_DIAGNOSTIC_FIELDS:
            normalized[field] = _parse_bool(record.get(field), field)
        for field in INTEGER_PAIR_DIAGNOSTIC_FIELDS:
            normalized[field] = _parse_int(record.get(field), field)
        normalized["_has_diagnostic_extension"] = True
    else:
        normalized["_has_diagnostic_extension"] = False
    return normalized


def _missing_fields(record: Mapping[str, object], required: Sequence[str]) -> list[str]:
    return [field for field in required if field not in record]


def _claim_contains_formal_safety(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            is_empty_or_prohibited = (
                child is False
                or child is None
                or child == ""
                or child == "prohibited"
            )
            if lowered in {"formal_safety_claim", "formal_safety"} and not is_empty_or_prohibited:
                return True
            if _claim_contains_formal_safety(child):
                return True
    elif isinstance(value, list):
        return any(_claim_contains_formal_safety(item) for item in value)
    return False


def _decision_int(
    event: Mapping[str, object],
    field: str,
    prefix: str,
    errors: list[str],
) -> int:
    try:
        return int(event.get(field, 0))
    except (TypeError, ValueError):
        errors.append(f"{prefix} has invalid integer field {field}")
        return 0


def _validate_decision_events_with_counts(
    events: Iterable[Mapping[str, object]],
) -> tuple[list[str], dict[str, dict[str, object]]]:
    errors: list[str] = []
    decision_ids: set[str] = set()
    per_run: dict[str, dict[str, object]] = {}
    active_run_id: str | None = None
    completed_run_ids: set[str] = set()
    for index, event in enumerate(events, start=1):
        prefix = f"decision event {index}"
        event_kind = event.get("event_kind")
        if event_kind is None:
            required = DECISION_FIELDNAMES
        elif event_kind == "decision_segment":
            required = DECISION_SEGMENT_FIELDNAMES
        elif event_kind == "decision_event":
            required = COMPACT_DECISION_EVENT_FIELDNAMES
        elif event_kind == "terminal_transition":
            required = TERMINAL_TRANSITION_FIELDNAMES
        else:
            errors.append(f"{prefix} has unsupported event_kind: {event_kind}")
            continue
        missing = _missing_fields(event, required)
        if missing:
            errors.append(f"{prefix} missing fields: {missing}")
            continue
        if event.get("decision_schema_version") != DECISION_SCHEMA_VERSION:
            errors.append(f"{prefix} has wrong decision schema version")
        if event_kind in {None, "decision_event"}:
            decision_id = str(event.get("decision_id"))
            if decision_id in decision_ids:
                errors.append(f"duplicate decision_id: {decision_id}")
            decision_ids.add(decision_id)
        if event.get("arm_id") != "monitor_on":
            errors.append(f"{prefix} creates a runtime-assurance event for monitor_off")
        if event_kind != "terminal_transition" and (
            event.get("hazard_threshold") != 1.90
            or event.get("hazard_comparator") != ">"
        ):
            errors.append(f"{prefix} drifts from strict > 1.90")
        if event_kind is None and bool(event.get("is_formal_experiment")):
            errors.append(f"{prefix} uses unbounded per-step logging for a formal event")
        if event_kind in {"decision_segment", "decision_event"} and not bool(
            event.get("invalid_evaluation")
        ):
            expected_mapping = {
                "continue": (
                    "predicted_nominal_within_threshold",
                    "allow",
                ),
                "veto_action": (
                    "predicted_nominal_exceeds_overspeed_threshold",
                    "veto",
                ),
            }
            decision_type = str(event.get("decision_type", ""))
            if decision_type not in expected_mapping:
                errors.append(f"{prefix} has an invalid compact decision type")
            else:
                expected_reason, expected_status = expected_mapping[decision_type]
                if event.get("decision_reason") != expected_reason:
                    errors.append(f"{prefix} has an invalid explicit decision reason")
                if event.get("veto_status") != expected_status:
                    errors.append(f"{prefix} has an invalid decision/veto-status mapping")
        if event_kind in {None, "decision_event"} and not bool(
            event.get("invalid_evaluation")
        ):
            try:
                realized_ratio = float(event.get("realized_executed_speed_ratio"))
                threshold = float(event.get("hazard_threshold"))
            except (TypeError, ValueError):
                errors.append(f"{prefix} has no realized hazard classification")
            else:
                expected_false_negative = (
                    event.get("decision_type") == "continue"
                    and realized_ratio > threshold
                )
                expected_fallback_failure = (
                    bool(event.get("fallback_executed"))
                    and realized_ratio > threshold
                )
                if bool(event.get("false_negative")) != expected_false_negative:
                    errors.append(f"{prefix} misclassifies a false negative")
                if bool(event.get("fallback_failure")) != expected_fallback_failure:
                    errors.append(f"{prefix} misclassifies a fallback failure")
        if event_kind == "decision_event":
            provided_reasons = event.get("exception_reasons")
            try:
                expected_reasons = list(exceptional_event_reasons(event))
            except (TypeError, ValueError):
                expected_reasons = []
                errors.append(f"{prefix} has invalid exceptional-event quantities")
            if not isinstance(provided_reasons, list) or provided_reasons != expected_reasons:
                errors.append(
                    f"{prefix} exceptional reasons do not match its logical flags"
                )
        if event_kind == "terminal_transition" and not bool(event.get("terminal_state")):
            errors.append(f"{prefix} does not mark the terminal state")
        run_id = str(event.get("run_id", ""))
        if not run_id:
            errors.append(f"{prefix} has no run_id")
        if active_run_id is None:
            active_run_id = run_id
        elif run_id != active_run_id:
            completed_run_ids.add(active_run_id)
            if run_id in completed_run_ids:
                errors.append(f"{prefix} resumes a completed run out of order")
            active_run_id = run_id
        identity = (
            event.get("experiment_id"),
            event.get("paired_run_id"),
            event.get("case_id"),
            event.get("subset_id"),
            event.get("arm_id"),
            event.get("monitor_id"),
            event.get("is_formal_experiment"),
        )
        counts = per_run.setdefault(
            run_id,
            {
                "record_count": 0,
                "event_count": 0,
                "allow_count": 0,
                "veto_count": 0,
                "fallback_count": 0,
                "invalid_evaluation_count": 0,
                "false_negative_count": 0,
                "fallback_failure_count": 0,
                "first_veto_step": None,
                "last_veto_step": None,
                "longest_consecutive_veto_steps": 0,
                "longest_consecutive_allow_steps": 0,
                "veto_segment_count": 0,
                "allow_segment_count": 0,
                "current_decision_status": None,
                "current_decision_streak": 0,
                "previous_legacy_behavior": None,
                "terminal_count": 0,
                "terminal_step": None,
                "terminal_label": None,
                "termination_reason": None,
                "identity": identity,
                "last_logical_step": None,
                "terminal_seen": False,
                "has_compact_decisions": False,
            },
        )
        if counts["identity"] != identity:
            errors.append(f"{prefix} changes run identity or formality metadata")
        if bool(counts["terminal_seen"]):
            errors.append(f"{prefix} appears after the run terminal transition")
        counts["record_count"] = int(counts["record_count"]) + 1
        logical_start: int | None = None
        logical_end: int | None = None
        if event_kind == "decision_segment":
            counts["has_compact_decisions"] = True
            step_count = _decision_int(event, "step_count", prefix, errors)
            start_step = _decision_int(event, "start_step", prefix, errors)
            end_step = _decision_int(event, "end_step", prefix, errors)
            if step_count <= 0 or end_step - start_step + 1 != step_count:
                errors.append(f"{prefix} has inconsistent segment bounds")
            if (
                bool(event.get("invalid_evaluation"))
                or bool(event.get("fallback_failure"))
                or bool(event.get("terminal_state"))
                or _decision_int(event, "fallback_failure_count", prefix, errors) != 0
                or _decision_int(event, "false_negative_count", prefix, errors) != 0
            ):
                errors.append(f"{prefix} hides a mandatory exceptional event in a segment")
            for action_field in (
                "first_nominal_action",
                "last_nominal_action",
                "first_executed_action",
                "last_executed_action",
            ):
                if not isinstance(event.get(action_field), list):
                    errors.append(f"{prefix} has invalid {action_field}")
            for quantity in (
                "predicted_nominal_speed_ratio",
                "realized_executed_speed_ratio",
            ):
                range_fields = (
                    f"first_{quantity}",
                    f"last_{quantity}",
                    f"minimum_{quantity}",
                    f"maximum_{quantity}",
                )
                try:
                    first, last, minimum, maximum = (
                        float(event[field]) for field in range_fields
                    )
                except (TypeError, ValueError):
                    errors.append(f"{prefix} has unavailable {quantity} aggregates")
                else:
                    if not all(math.isfinite(value) for value in (first, last, minimum, maximum)):
                        errors.append(f"{prefix} has non-finite {quantity} aggregates")
                    elif not minimum <= first <= maximum or not minimum <= last <= maximum:
                        errors.append(f"{prefix} has inconsistent {quantity} aggregates")
            fallback_minimum = event.get("minimum_predicted_fallback_speed_ratio")
            fallback_maximum = event.get("maximum_predicted_fallback_speed_ratio")
            if bool(event.get("fallback_executed")):
                try:
                    fallback_minimum = float(fallback_minimum)
                    fallback_maximum = float(fallback_maximum)
                except (TypeError, ValueError):
                    errors.append(f"{prefix} omits fallback prediction aggregates")
                else:
                    if (
                        not math.isfinite(fallback_minimum)
                        or not math.isfinite(fallback_maximum)
                        or fallback_minimum > fallback_maximum
                    ):
                        errors.append(f"{prefix} has invalid fallback prediction aggregates")
                    else:
                        try:
                            realized_minimum = float(
                                event.get("minimum_realized_executed_speed_ratio")
                            )
                            realized_maximum = float(
                                event.get("maximum_realized_executed_speed_ratio")
                            )
                            threshold = float(event.get("hazard_threshold"))
                        except (TypeError, ValueError):
                            pass
                        else:
                            predicted_classes = {
                                fallback_minimum > threshold,
                                fallback_maximum > threshold,
                            }
                            realized_classes = {
                                realized_minimum > threshold,
                                realized_maximum > threshold,
                            }
                            if predicted_classes != realized_classes:
                                errors.append(
                                    f"{prefix} hides a fallback classification mismatch"
                                )
            elif fallback_minimum is not None or fallback_maximum is not None:
                errors.append(f"{prefix} must use explicit null fallback aggregates")
            if event.get("decision_type") == "continue" and bool(
                event.get("fallback_executed")
            ):
                errors.append(f"{prefix} executes fallback for an allow segment")
            if event.get("decision_type") == "veto_action" and not bool(
                event.get("fallback_executed")
            ):
                errors.append(f"{prefix} omits fallback execution for a veto segment")
            try:
                realized_maximum = float(
                    event.get("maximum_realized_executed_speed_ratio")
                )
                threshold = float(event.get("hazard_threshold"))
            except (TypeError, ValueError):
                pass
            else:
                if event.get("decision_type") == "continue" and realized_maximum > threshold:
                    errors.append(f"{prefix} hides a false negative in an allow segment")
                if bool(event.get("fallback_executed")) and realized_maximum > threshold:
                    errors.append(f"{prefix} hides a fallback failure in a veto segment")
            logical_count = max(step_count, 0)
            logical_start = start_step
            logical_end = end_step
            counts["event_count"] = int(counts["event_count"]) + logical_count
            if event.get("decision_type") == "continue":
                counts["allow_count"] = int(counts["allow_count"]) + logical_count
            elif event.get("decision_type") == "veto_action":
                counts["veto_count"] = int(counts["veto_count"]) + logical_count
            if bool(event.get("fallback_executed")):
                counts["fallback_count"] = int(counts["fallback_count"]) + logical_count
            if bool(event.get("invalid_evaluation")):
                counts["invalid_evaluation_count"] = (
                    int(counts["invalid_evaluation_count"]) + logical_count
                )
            counts["false_negative_count"] = int(counts["false_negative_count"]) + int(
                _decision_int(event, "false_negative_count", prefix, errors)
            )
            counts["fallback_failure_count"] = int(
                counts["fallback_failure_count"]
            ) + _decision_int(event, "fallback_failure_count", prefix, errors)
        elif event_kind != "terminal_transition":
            if event_kind == "decision_event":
                counts["has_compact_decisions"] = True
            logical_start = _decision_int(event, "step", prefix, errors)
            logical_end = logical_start
            counts["event_count"] = int(counts["event_count"]) + 1
            if event.get("decision_type") == "continue" and not bool(
                event.get("invalid_evaluation")
            ):
                counts["allow_count"] = int(counts["allow_count"]) + 1
            elif event.get("decision_type") == "veto_action" and not bool(
                event.get("invalid_evaluation")
            ):
                counts["veto_count"] = int(counts["veto_count"]) + 1
            counts["fallback_count"] = int(counts["fallback_count"]) + int(
                bool(event.get("fallback_executed"))
            )
            counts["invalid_evaluation_count"] = int(
                counts["invalid_evaluation_count"]
            ) + int(
                bool(event.get("invalid_evaluation"))
            )
            counts["false_negative_count"] = int(counts["false_negative_count"]) + int(
                bool(event.get("false_negative"))
            )
            counts["fallback_failure_count"] = int(
                counts["fallback_failure_count"]
            ) + int(
                bool(event.get("fallback_failure"))
            )
        else:
            counts["terminal_count"] = int(counts["terminal_count"]) + 1
            terminal_step = _decision_int(event, "step", prefix, errors)
            if (
                counts["last_logical_step"] is not None
                and terminal_step < int(counts["last_logical_step"])
            ):
                errors.append(f"{prefix} precedes the last logical decision step")
            counts["terminal_step"] = terminal_step
            counts["terminal_label"] = event.get("terminal_label")
            counts["termination_reason"] = event.get("termination_reason")
            counts["terminal_seen"] = True
        if logical_start is not None and logical_end is not None:
            previous_end = counts["last_logical_step"]
            expected_start = 1 if previous_end is None else int(previous_end) + 1
            if logical_start != expected_start:
                errors.append(
                    f"{prefix} is not contiguous with the logical decision stream"
                )
            invalid = bool(event.get("invalid_evaluation"))
            status = None
            if not invalid and event.get("decision_type") == "continue":
                status = "allow"
            elif not invalid and event.get("decision_type") == "veto_action":
                status = "veto"
            span = max(logical_end - logical_start + 1, 0)
            contiguous = previous_end is not None and logical_start == int(previous_end) + 1
            if status is None:
                counts["current_decision_status"] = None
                counts["current_decision_streak"] = 0
                counts["previous_legacy_behavior"] = None
            else:
                if event_kind in {"decision_segment", "decision_event"}:
                    begins_decision_segment = True
                else:
                    legacy_behavior = (
                        event.get("phase"),
                        event.get("active_stage"),
                        event.get("decision_type"),
                        event.get("decision_reason"),
                        event.get("veto_status"),
                        bool(event.get("fallback_executed")),
                        invalid,
                        bool(event.get("fallback_failure")),
                        bool(event.get("false_negative")),
                        exceptional_event_reasons(event),
                    )
                    begins_decision_segment = (
                        not contiguous
                        or legacy_behavior != counts["previous_legacy_behavior"]
                    )
                    counts["previous_legacy_behavior"] = legacy_behavior
                if begins_decision_segment:
                    segment_field = f"{status}_segment_count"
                    counts[segment_field] = int(counts[segment_field]) + 1
                if counts["current_decision_status"] == status and contiguous:
                    streak = int(counts["current_decision_streak"]) + span
                else:
                    streak = span
                counts["current_decision_status"] = status
                counts["current_decision_streak"] = streak
                longest_field = f"longest_consecutive_{status}_steps"
                counts[longest_field] = max(int(counts[longest_field]), streak)
                if status == "veto":
                    if counts["first_veto_step"] is None:
                        counts["first_veto_step"] = logical_start
                    counts["last_veto_step"] = logical_end
            counts["last_logical_step"] = logical_end
        if any(key in event for key in ("avoided_failure", "blocked_success", "unnecessary_veto")):
            errors.append(f"{prefix} contains pair-level counterfactual claims")
        if _claim_contains_formal_safety(event):
            errors.append(f"{prefix} contains a formal-safety claim")
    for run_id, counts in per_run.items():
        if bool(counts["has_compact_decisions"]):
            if int(counts["terminal_count"]) != 1:
                errors.append(
                    f"{run_id} compact decision stream requires exactly one terminal transition"
                )
            if int(counts["record_count"]) > MAX_COMPACT_RECORDS_PER_RUN:
                errors.append(
                    f"{run_id} exceeds the compact record cap of {MAX_COMPACT_RECORDS_PER_RUN}"
                )
    return errors, per_run


def validate_decision_events(events: Iterable[Mapping[str, object]]) -> list[str]:
    errors, _ = _validate_decision_events_with_counts(events)
    return errors


def validate_result_records(
    arm_records: Iterable[Mapping[str, object]],
    pair_records: Iterable[Mapping[str, object]],
    manifest: Mapping[str, Any],
    *,
    decision_events: Iterable[Mapping[str, object]] | None = None,
    output_directory: Path | None = None,
    repository_root: Path = PROJECT_ROOT,
    require_formal_acceptance: bool = False,
) -> ResultValidationReport:
    errors: list[str] = []
    observations: list[str] = []
    manifest_valid = True
    try:
        validate_manifest_data(dict(manifest))
    except ManifestValidationError as exc:
        manifest_valid = False
        errors.extend(f"frozen manifest: {error}" for error in exc.errors)

    expected_jobs: dict[tuple[str, str], object] = {}
    if manifest_valid:
        try:
            expected_jobs = {
                (job.case_id, job.arm_id): job
                for job in build_planned_jobs(manifest)
            }
        except RunnerContractError as exc:
            errors.append(f"frozen job plan: {exc}")

    if output_directory is not None:
        try:
            validate_output_directory(
                output_directory,
                repository_root=repository_root,
                protected_paths=manifest.get("protected_paths", []),
            )
        except ArtifactWriteError as exc:
            errors.append(str(exc))

    raw_arms = list(arm_records)
    raw_pairs = list(pair_records)
    arms: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for index, raw in enumerate(raw_arms, start=1):
        missing = _missing_fields(raw, LEGACY_ARM_FIELDNAMES)
        if missing:
            errors.append(f"arm row {index} missing fields: {missing}")
            continue
        extension_presence = [field in raw for field in ARM_DIAGNOSTIC_FIELDNAMES]
        if any(extension_presence) and not all(extension_presence):
            errors.append(f"arm row {index} has a partial compact-logging extension")
            continue
        try:
            formal_hint = _parse_bool(raw.get("is_formal_experiment"), "is_formal_experiment")
        except ValueError as exc:
            errors.append(f"arm row {index}: {exc}")
            continue
        if formal_hint and not all(extension_presence):
            errors.append(f"arm row {index} formal result lacks compact-logging diagnostics")
            continue
        try:
            row = _normalize_arm_record(raw)
        except (TypeError, ValueError) as exc:
            errors.append(f"arm row {index}: {exc}")
            continue
        arms.append(row)

    for index, raw in enumerate(raw_pairs, start=1):
        missing = _missing_fields(raw, LEGACY_PAIR_FIELDNAMES)
        if missing:
            errors.append(f"pair row {index} missing fields: {missing}")
            continue
        extension_presence = [field in raw for field in PAIR_DIAGNOSTIC_FIELDNAMES]
        if any(extension_presence) and not all(extension_presence):
            errors.append(f"pair row {index} has a partial intervention-diagnostic extension")
            continue
        try:
            formal_hint = _parse_bool(raw.get("is_formal_experiment"), "is_formal_experiment")
        except ValueError as exc:
            errors.append(f"pair row {index}: {exc}")
            continue
        if formal_hint and not all(extension_presence):
            errors.append(f"pair row {index} formal result lacks intervention diagnostics")
            continue
        try:
            row = _normalize_pair_record(raw)
        except (TypeError, ValueError) as exc:
            errors.append(f"pair row {index}: {exc}")
            continue
        pairs.append(row)

    run_ids: set[str] = set()
    pair_arms: set[tuple[str, str]] = set()
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in arms:
        run_id = str(row["run_id"])
        pair_arm = (str(row["paired_run_id"]), str(row["arm_id"]))
        if run_id in run_ids:
            errors.append(f"duplicate run_id: {run_id}")
        if pair_arm in pair_arms:
            errors.append(f"duplicate pair/arm combination: {pair_arm}")
        run_ids.add(run_id)
        pair_arms.add(pair_arm)
        grouped.setdefault(str(row["paired_run_id"]), []).append(row)

        if row["schema_version"] != ARM_SCHEMA_VERSION:
            errors.append(f"{run_id} has wrong Result Schema version")
        if row["experiment_id"] != manifest.get("experiment_id"):
            errors.append(f"{run_id} has wrong experiment_id")
        expected_job = expected_jobs.get((str(row["case_id"]), str(row["arm_id"])))
        if expected_jobs and expected_job is None:
            errors.append(f"{run_id} is not a case/arm declared by the frozen manifest")
        elif expected_job is not None:
            expected_fields = {
                "paired_run_id": expected_job.paired_run_id,
                "subset_id": expected_job.subset_id,
                "seed": expected_job.seed,
                "controller_id": expected_job.controller_id,
                "case_config_hash": expected_job.case_config_hash,
                "run_config_hash": expected_job.run_config_hash,
                "r0_over_target": expected_job.r0_over_target,
                "initial_velocity_angle_deg": expected_job.initial_velocity_angle_deg,
                "thrust_scale": expected_job.thrust_scale,
            }
            drifted = [
                field
                for field, expected_value in expected_fields.items()
                if row.get(field) != expected_value
            ]
            if drifted:
                errors.append(f"{run_id} drifts from frozen job fields: {drifted}")
        if row["hazard_threshold"] != 1.90 or row["hazard_comparator"] != ">":
            errors.append(f"{run_id} drifts from strict > 1.90")
        if row["terminal_label"] not in CONTROLLED_TERMINAL_LABELS:
            errors.append(f"{run_id} has uncontrolled terminal_label")
        if row["terminal_label"] == "unknown" and not str(row["manual_audit_note"]).strip():
            errors.append(f"{run_id} uses unknown without a manual audit note")
        if row["subset_id"] not in {PRESERVATION_SUBSET_ID, STRESS_SUBSET_ID}:
            errors.append(f"{run_id} has an undeclared subset")
        expected_known = row["subset_id"] == PRESERVATION_SUBSET_ID
        if row["known_phase34_recoverable_case"] != expected_known:
            errors.append(f"{run_id} has inconsistent known-success membership")
        if row["is_full_benchmark"]:
            errors.append(f"{run_id} incorrectly marks a subset row as full benchmark")
        if row["accepted_as_progress"] and not row["is_formal_experiment"]:
            errors.append(f"{run_id} accepts progress from a smoke/nonformal row")
        if row["is_formal_experiment"] and row["experiment_status"] == "nonformal_smoke":
            errors.append(f"{run_id} marks a smoke status as formal")
        if _claim_contains_formal_safety(row):
            errors.append(f"{run_id} contains a formal-safety claim")

        if row["_has_diagnostic_extension"]:
            if not str(row.get("termination_reason", "")).strip():
                errors.append(f"{run_id} has no raw termination reason")
            mode = str(row.get("decision_log_mode", ""))
            if mode not in VALID_DECISION_LOG_MODES:
                errors.append(f"{run_id} has invalid decision_log_mode")
            if row["is_formal_experiment"] and mode != LOG_MODE_COMPACT:
                errors.append(f"{run_id} formal result does not use compact logging")
            digest = str(row.get("decision_stream_sha256", ""))
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                errors.append(f"{run_id} has invalid decision-stream SHA-256")
            expected_stream_events = (
                row["monitor_evaluation_count"] + row["invalid_monitor_evaluation_count"]
            )
            if row["decision_stream_event_count"] != expected_stream_events:
                errors.append(f"{run_id} decision-stream event count disagrees with monitor counts")
            if mode == LOG_MODE_FULL_TRACE and row["compact_decision_record_count"] != 0:
                errors.append(f"{run_id} full trace reports compact records")
            if row["arm_id"] == "monitor_off" and row["compact_decision_record_count"] != 0:
                errors.append(f"{run_id} monitor_off reports compact decision records")
            if (
                row["arm_id"] == "monitor_on"
                and mode == LOG_MODE_COMPACT
                and row["compact_decision_record_count"] < 1
            ):
                errors.append(f"{run_id} compact monitor_on log lacks a terminal record")
            evaluations = row["monitor_evaluation_count"]
            expected_rates = (
                (
                    row["veto_count"] / evaluations,
                    row["allow_count"] / evaluations,
                    row["fallback_count"] / evaluations,
                )
                if evaluations
                else ((0.0, 0.0, 0.0) if row["arm_id"] == "monitor_off" else (None, None, None))
            )
            actual_rates = (
                row["intervention_rate"],
                row["allow_rate"],
                row["fallback_rate"],
            )
            for field, actual, expected in zip(
                ("intervention_rate", "allow_rate", "fallback_rate"),
                actual_rates,
                expected_rates,
            ):
                if expected is None:
                    if actual is not None:
                        errors.append(f"{run_id} {field} must be unavailable")
                elif actual is None or not math.isclose(actual, expected, abs_tol=1e-12):
                    errors.append(f"{run_id} has inconsistent {field}")
            if row["veto_count"] == 0:
                if row["first_veto_step"] is not None or row["last_veto_step"] is not None:
                    errors.append(f"{run_id} reports veto steps without a veto")
                if row["longest_consecutive_veto_steps"] != 0 or row["veto_segment_count"] != 0:
                    errors.append(f"{run_id} reports veto burden without a veto")
            else:
                if row["first_veto_step"] is None or row["last_veto_step"] is None:
                    errors.append(f"{run_id} omits first or last veto step")
            if row["longest_consecutive_veto_steps"] > row["veto_count"]:
                errors.append(f"{run_id} longest veto streak exceeds veto count")
            if row["longest_consecutive_allow_steps"] > row["allow_count"]:
                errors.append(f"{run_id} longest allow streak exceeds allow count")

        counts = [
            row["monitor_evaluation_count"],
            row["allow_count"],
            row["veto_count"],
            row["fallback_count"],
            row["false_negative_count"],
            row["fallback_failure_count"],
            row["invalid_monitor_evaluation_count"],
        ]
        if any(int(value) < 0 for value in counts):
            errors.append(f"{run_id} has a negative monitor count")
        if row["arm_id"] == "monitor_off":
            if row["monitor_enabled"]:
                errors.append(f"{run_id} enables the monitor in monitor_off")
            if any(int(row[field]) for field in (
                "monitor_evaluation_count",
                "allow_count",
                "veto_count",
                "fallback_count",
                "false_negative_count",
                "fallback_failure_count",
                "invalid_monitor_evaluation_count",
            )):
                errors.append(f"{run_id} records monitor activity in monitor_off")
        elif row["arm_id"] == "monitor_on":
            if not row["monitor_enabled"]:
                errors.append(f"{run_id} disables the monitor in monitor_on")
            if row["allow_count"] + row["veto_count"] != row["monitor_evaluation_count"]:
                errors.append(f"{run_id} allow + veto does not equal evaluations")
            if row["fallback_count"] != row["veto_count"]:
                errors.append(f"{run_id} fallback count does not equal veto count")
        else:
            errors.append(f"{run_id} has invalid arm_id")

    pair_complete = True
    for pair_id, rows in grouped.items():
        if len(rows) != 2 or {str(row["arm_id"]) for row in rows} != {
            "monitor_off",
            "monitor_on",
        }:
            pair_complete = False
            errors.append(f"{pair_id} is missing or duplicates an arm")
            continue
        if len({str(row["case_config_hash"]) for row in rows}) != 1:
            pair_complete = False
            errors.append(f"{pair_id} has mismatched case hashes")

    pair_by_id: dict[str, dict[str, object]] = {}
    for row in pairs:
        pair_id = str(row["paired_run_id"])
        if pair_id in pair_by_id:
            errors.append(f"duplicate pair result: {pair_id}")
            pair_complete = False
        pair_by_id[pair_id] = row
        if row["pair_schema_version"] != PAIR_SCHEMA_VERSION:
            errors.append(f"{pair_id} has wrong pair schema version")
        if row["experiment_id"] != manifest.get("experiment_id"):
            errors.append(f"{pair_id} has wrong experiment_id")
        if row["avoided_failure"] and (
            not row["pair_complete"]
            or not row["pair_valid"]
            or not row["off_overspeed"]
            or row["on_overspeed"]
            or row["on_invalid_simulation"]
        ):
            errors.append(f"{pair_id} claims avoided failure without a valid hazard counterfactual")
        if row["blocked_success"] and not (
            (row["off_recoverable_crossing"] and not row["on_recoverable_crossing"])
            or (row["off_final_simulator_success"] and not row["on_final_simulator_success"])
        ):
            errors.append(f"{pair_id} claims blocked success without lost off-arm success")
        if row["claim_eligible"] and not row["is_formal_experiment"]:
            errors.append(f"{pair_id} is claim eligible despite being nonformal")
        if _claim_contains_formal_safety(row):
            errors.append(f"{pair_id} contains a formal-safety claim")
        if row["_has_diagnostic_extension"]:
            if row["declared_hazard_avoided"] != row["avoided_failure"]:
                errors.append(f"{pair_id} changes avoided-failure semantics")
            if row["monitor_induced_horizon_extension"] != row["step_count_delta"]:
                errors.append(f"{pair_id} has inconsistent horizon-extension diagnostics")
            if row["task_recovered_after_hazard_avoidance"] and not row[
                "declared_hazard_avoided"
            ]:
                errors.append(f"{pair_id} claims task recovery without hazard avoidance")
            expected_transition = (
                f"{row['termination_reason_without_monitor']} -> "
                f"{row['termination_reason_with_monitor']}"
            )
            if row["terminal_outcome_transition"] != expected_transition:
                errors.append(f"{pair_id} has inconsistent terminal outcome transition")

    if set(pair_by_id) != set(grouped):
        pair_complete = False
        missing = sorted(set(grouped) - set(pair_by_id))
        extra = sorted(set(pair_by_id) - set(grouped))
        if missing:
            errors.append(f"missing pair result rows: {missing}")
        if extra:
            errors.append(f"pair results have no matching arm rows: {extra}")

    for pair_id in sorted(set(pair_by_id) & set(grouped)):
        rows = grouped[pair_id]
        if len(rows) != 2:
            continue
        try:
            expected = build_pair_record(rows)
        except RunnerContractError as exc:
            errors.append(f"{pair_id} cannot be paired: {exc}")
            pair_complete = False
            continue
        actual = pair_by_id[pair_id]
        comparison_fields = (
            "case_config_hash",
            "pair_complete",
            "pair_valid",
            "off_overspeed",
            "on_overspeed",
            "avoided_failure",
            "blocked_success",
            "unnecessary_veto",
            "on_false_negative_count",
            "on_fallback_failure_count",
        )
        if actual.get("_has_diagnostic_extension"):
            comparison_fields += PAIR_DIAGNOSTIC_FIELDNAMES
        mismatches = [field for field in comparison_fields if actual[field] != expected[field]]
        if mismatches:
            errors.append(f"{pair_id} pair metrics disagree with arm rows: {mismatches}")

    decision_counts: dict[str, dict[str, object]] = {}
    if decision_events is not None:
        decision_errors, decision_counts = _validate_decision_events_with_counts(
            decision_events
        )
        errors.extend(decision_errors)
        monitor_on_by_run = {
            str(row["run_id"]): row
            for row in arms
            if row["arm_id"] == "monitor_on"
        }
        extra_run_ids = sorted(set(decision_counts) - set(monitor_on_by_run))
        missing_run_ids = sorted(set(monitor_on_by_run) - set(decision_counts))
        if extra_run_ids:
            errors.append(f"decision log has no matching monitor_on arm rows: {extra_run_ids}")
        if missing_run_ids:
            errors.append(f"monitor_on arm rows have no decision-log run: {missing_run_ids}")
        for run_id in sorted(set(decision_counts) & set(monitor_on_by_run)):
            row = monitor_on_by_run[run_id]
            counts = decision_counts[run_id]
            expected_identity = (
                row["experiment_id"],
                row["paired_run_id"],
                row["case_id"],
                row["subset_id"],
                row["arm_id"],
                row["monitor_id"],
                row["is_formal_experiment"],
            )
            if counts.get("identity") != expected_identity:
                errors.append(f"{run_id} decision-log identity disagrees with its arm row")
            if int(counts.get("terminal_count", 0)):
                if (
                    counts.get("terminal_step") != row["steps"]
                    or counts.get("terminal_label") != row["terminal_label"]
                ):
                    errors.append(
                        f"{run_id} terminal record disagrees with arm step or label"
                    )
                if (
                    row.get("_has_diagnostic_extension")
                    and counts.get("termination_reason") != row["termination_reason"]
                ):
                    errors.append(
                        f"{run_id} terminal record disagrees with arm termination reason"
                    )
            expected_counts = {
                "event_count": (
                    row["decision_stream_event_count"]
                    if row.get("_has_diagnostic_extension")
                    else row["monitor_evaluation_count"]
                    + row["invalid_monitor_evaluation_count"]
                ),
                "allow_count": row["allow_count"],
                "veto_count": row["veto_count"],
                "fallback_count": row["fallback_count"],
                "invalid_evaluation_count": row["invalid_monitor_evaluation_count"],
                "false_negative_count": row["false_negative_count"],
                "fallback_failure_count": row["fallback_failure_count"],
            }
            if row.get("_has_diagnostic_extension"):
                expected_counts.update(
                    {
                        "first_veto_step": row["first_veto_step"],
                        "last_veto_step": row["last_veto_step"],
                        "longest_consecutive_veto_steps": row[
                            "longest_consecutive_veto_steps"
                        ],
                        "longest_consecutive_allow_steps": row[
                            "longest_consecutive_allow_steps"
                        ],
                        "veto_segment_count": row["veto_segment_count"],
                        "allow_segment_count": row["allow_segment_count"],
                    }
                )
            if (
                row.get("_has_diagnostic_extension")
                and row.get("decision_log_mode") == LOG_MODE_COMPACT
            ):
                expected_counts["record_count"] = row["compact_decision_record_count"]
                if int(counts.get("terminal_count", 0)) != 1:
                    errors.append(
                        f"{run_id} compact monitor_on log requires one terminal record"
                    )
            drifted = [
                field
                for field, expected_value in expected_counts.items()
                if counts.get(field) != expected_value
            ]
            if drifted:
                errors.append(
                    f"{run_id} decision records disagree with arm counters: {drifted}"
                )

    formal_rows = bool(arms) and all(bool(row["is_formal_experiment"]) for row in arms)
    preservation_on = [
        row
        for row in arms
        if row["subset_id"] == PRESERVATION_SUBSET_ID and row["arm_id"] == "monitor_on"
    ]
    stress_off = [
        row
        for row in arms
        if row["subset_id"] == STRESS_SUBSET_ID and row["arm_id"] == "monitor_off"
    ]
    stress_on = [
        row
        for row in arms
        if row["subset_id"] == STRESS_SUBSET_ID and row["arm_id"] == "monitor_on"
    ]
    preservation_pairs = [
        row for row in pairs if row["subset_id"] == PRESERVATION_SUBSET_ID
    ]
    preservation_pass = (
        len(preservation_on) == 8
        and all(bool(row["crossed_target_radius"]) for row in preservation_on)
        and all(bool(row["recoverable_crossing"]) for row in preservation_on)
        and not any(bool(row["blocked_success"]) for row in preservation_pairs)
    )
    off_stress_overspeed = sum(bool(row["overspeed"]) for row in stress_off)
    on_stress_overspeed = sum(bool(row["overspeed"]) for row in stress_on)
    avoided_count = sum(bool(row["avoided_failure"]) for row in pairs)
    invalid_off = sum(
        bool(row["invalid_simulation"]) for row in arms if row["arm_id"] == "monitor_off"
    )
    invalid_on = sum(
        bool(row["invalid_simulation"]) for row in arms if row["arm_id"] == "monitor_on"
    )
    total_evaluations = sum(int(row["monitor_evaluation_count"]) for row in stress_on + preservation_on)
    total_vetoes = sum(int(row["veto_count"]) for row in stress_on + preservation_on)
    stress_exercised = len(stress_off) == 5 and off_stress_overspeed >= 1
    hazard_reduced = (
        len(stress_on) == 5 and on_stress_overspeed <= off_stress_overspeed - 1
    )
    nontrivial = total_evaluations > 0 and total_vetoes < total_evaluations

    if not formal_rows:
        preservation_status = "not_evaluated_nonformal"
        stress_status = "not_evaluated_nonformal"
    else:
        preservation_status = "pass" if preservation_pass else "fail"
        stress_status = "pass" if stress_exercised else "monitor_not_exercised"
    structural_valid = not errors
    positive_claim = (
        structural_valid
        and pair_complete
        and formal_rows
        and len(arms) == 26
        and len(pairs) == 13
        and preservation_pass
        and stress_exercised
        and hazard_reduced
        and avoided_count >= 1
        and invalid_on <= invalid_off
        and nontrivial
    )
    if require_formal_acceptance and not positive_claim:
        errors.append("formal positive-claim acceptance criteria are not satisfied")
        structural_valid = False

    observations.extend(
        [
            f"arm_rows={len(arms)} pair_rows={len(pairs)}",
            f"preservation_monitor_on={len(preservation_on)}",
            f"stress_monitor_off_overspeed={off_stress_overspeed}",
            f"stress_monitor_on_overspeed={on_stress_overspeed}",
            f"avoided_failures={avoided_count}",
            f"monitor_evaluations={total_evaluations} vetoes={total_vetoes}",
        ]
    )
    metrics = {
        "arm_rows": len(arms),
        "pair_rows": len(pairs),
        "preservation_monitor_on": len(preservation_on),
        "stress_monitor_off_overspeed": off_stress_overspeed,
        "stress_monitor_on_overspeed": on_stress_overspeed,
        "avoided_failures": avoided_count,
        "blocked_successes": sum(bool(row["blocked_success"]) for row in pairs),
        "unnecessary_vetoes": sum(bool(row["unnecessary_veto"]) for row in pairs),
        "false_negatives": sum(int(row["on_false_negative_count"]) for row in pairs),
        "fallback_failures": sum(int(row["on_fallback_failure_count"]) for row in pairs),
    }
    return ResultValidationReport(
        structural_valid=structural_valid,
        pair_complete=pair_complete,
        preservation_acceptance=preservation_status,
        stress_hazard_exercised=stress_status,
        positive_claim_eligible=positive_claim,
        errors=tuple(errors),
        observations=tuple(observations),
        metrics=metrics,
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            yield value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Final Veto arm and pair results.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--paired-results", type=Path, required=True)
    parser.add_argument("--decision-log", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--formal", action="store_true", help="Require formal positive-claim acceptance.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest_path = args.manifest or PROJECT_ROOT / MANIFEST_RELATIVE_PATH
        manifest = load_manifest(manifest_path)
        arm_rows = read_csv_rows(args.results)
        pair_rows = read_csv_rows(args.paired_results)
        events = read_jsonl(args.decision_log) if args.decision_log else None
        report = validate_result_records(
            arm_rows,
            pair_rows,
            manifest,
            decision_events=events,
            output_directory=args.results.parent,
            require_formal_acceptance=args.formal,
        )
    except (OSError, ValueError, ManifestValidationError) as exc:
        print(f"FAIL {exc}")
        return 1

    print(f"STRUCTURAL {'PASS' if report.structural_valid else 'FAIL'}")
    print(f"PAIR_COMPLETENESS {'PASS' if report.pair_complete else 'FAIL'}")
    print(f"PRESERVATION {report.preservation_acceptance}")
    print(f"STRESS_HAZARD_EXERCISE {report.stress_hazard_exercised}")
    print(f"POSITIVE_CLAIM_ELIGIBLE {str(report.positive_claim_eligible).lower()}")
    for observation in report.observations:
        print(f"INFO {observation}")
    for error in report.errors:
        print(f"FAIL {error}")
    return 0 if report.structural_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
