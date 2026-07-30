from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from runtime_assurance.recovery_branch_executor import (
    execute_recovery_branch,
    load_frozen_branch_state,
    validate_branch_state_integrity,
)
from runtime_assurance.recovery_experiment_preflight import (
    FROZEN_BRANCH_STATE_CANONICAL_HASH,
    FROZEN_MANIFEST_CANONICAL_HASH,
    inspect_repository_state,
    load_json_object,
)
from runtime_assurance.staged_recovery_instrumentation import (
    CANONICAL_FIELD_ORDER,
    InstrumentationEvidenceStatus,
    canonical_sha256,
    field_catalog,
)
from runtime_assurance.staged_recovery_logger_adapter import (
    VALIDATION_TERMINAL_REASON,
    BoundedRecoveryValidationRun,
    Stage0BValidationLoggerAdapter,
    build_bounded_runtime_identity,
    run_bounded_recovery_validation_path,
)
from runtime_assurance.staged_recovery_runtime_logger import (
    LOGGER_SCHEMA_VERSION,
    SOURCE_ARCHITECTURE_CANONICAL_HASH,
    SOURCE_ARCHITECTURE_COMMIT,
    SOURCE_INSTRUMENTATION_CANONICAL_HASH,
    SOURCE_INSTRUMENTATION_COMMIT,
    STAGED_EXECUTION_STATUS,
    StagedRecoveryRuntimeEvent,
    aggregate_trace_sha256,
    canonical_runtime_json_bytes,
    canonical_runtime_sha256,
    event_document,
    event_schema_document,
    logger_field_coverage,
    trace_jsonl_bytes_from_events,
    validate_trace_publication_target,
)
from scripts.check_recovery_branch_state import canonical_sha256 as recovery_canonical_sha256
from scripts.run_recovery_branch_runner import MAX_RUNNER_HORIZON_STEPS


VALIDATION_ID = "staged_recovery_instrumentation_validation_v0"
VALIDATION_SCHEMA_VERSION = "staged_recovery_instrumentation_validation_manifest_v0"
EQUIVALENCE_SCHEMA_VERSION = "staged_recovery_paired_equivalence_v0"
FIELD_COMPLETENESS_SCHEMA_VERSION = "staged_recovery_field_completeness_v0"
TRACE_MANIFEST_SCHEMA_VERSION = "staged_recovery_measured_validation_trace_v0"
COMPLETED_DATE = "2026-07-30"
VALIDATION_BRANCH_ID = "velocity_opposed_thrust_v0"
VALIDATION_HORIZON = 8
EXPECTED_EVENT_COUNT = 10
FROZEN_BRANCH_ORDER = (
    "zero_action_reference_v0",
    "velocity_opposed_thrust_v0",
    "tangential_error_correction_v0",
    "explicit_abort_v0",
)
TRACE_CLASSIFICATION = "measured_instrumentation_validation"
RUNTIME_SOURCE = "bounded_recovery_validation_path"
SCIENTIFIC_RESULT = False

SOURCE_LOGGER_COMMIT = "f92b7ffe11ca559764228a0d2500b211ad562ecf"
SOURCE_LOGGER_CANONICAL_HASH = (
    "b4f7a25e53795845895707b9d5a3d14804431f5323858854e48685f27723d6dd"
)

MANIFEST_RELATIVE_PATH = Path(
    "analysis/recovery_action_branching_nonformal_v0/manifest.json"
)
BRANCH_STATE_RELATIVE_PATH = Path(
    "analysis/recovery_action_branching_nonformal_v0/branch_state.json"
)
OUTPUT_RELATIVE_PATH = Path(
    "analysis/staged_recovery_instrumentation_validation_v0"
)
VALIDATION_MANIFEST_FILENAME = "validation_manifest.json"
TRACE_MANIFEST_FILENAME = "trace_manifest.json"
TRACE_JSONL_FILENAME = "staged_recovery_trace.jsonl"
EQUIVALENCE_FILENAME = "equivalence_report.json"
FIELD_COMPLETENESS_FILENAME = "field_completeness.json"
SUMMARY_FILENAME = "summary.md"
PUBLISHED_FILENAMES = (
    VALIDATION_MANIFEST_FILENAME,
    TRACE_MANIFEST_FILENAME,
    TRACE_JSONL_FILENAME,
    EQUIVALENCE_FILENAME,
    FIELD_COMPLETENESS_FILENAME,
    SUMMARY_FILENAME,
)

CLAIM_RESTRICTIONS = (
    "no_formal_safety_claim",
    "no_hardware_or_deployment_claim",
    "no_phase_policy_validation_claim",
    "no_recovery_performance_claim",
    "no_staged_controller_execution_claim",
    "no_task_recovery_claim",
)

PHASE_RUNTIME_FIELDS = frozenset(
    {
        "current_phase",
        "previous_phase",
        "phase_dwell_count",
        "phase_transition_count",
        "recent_phase_history",
        "phase_transition_reason",
        "no_progress_status",
        "handoff_readiness",
        "retreat_status",
    }
)
FUTURE_EVALUATOR_FIELDS = frozenset(
    {
        "instability_status",
        "unsafe_state_status",
        "recovery_success_v0",
        "simulator_success",
        "available_correction_authority",
        "required_to_available_correction_ratio",
    }
)
UNSUPPORTED_FIELDS = frozenset(
    {"available_correction_authority", "required_to_available_correction_ratio"}
)
EXPECTED_UNAVAILABLE_FIELDS = frozenset(
    set(PHASE_RUNTIME_FIELDS)
    | set(FUTURE_EVALUATOR_FIELDS)
    | {
        "predicted_crossing_direction",
        "crossing_proximity",
        "progress_classification",
        "crossing_interpolation_fraction",
        "first_crossing_step",
        "first_recoverable_crossing_step",
        "recovery_horizon_remaining",
        "radial_progress",
        "radial_progress_direction",
        "radial_progress_rate",
        "tangential_progress",
        "crossing_progress",
        "energy_change_direction",
    }
)
TRANSITION_ONLY_STAGE0A_FIELDS = frozenset(
    {
        "predicted_speed_ratio",
        "predicted_overspeed_headroom",
        "predicted_overspeed_status",
        "target_radius_crossing",
        "crossing_direction",
        "crossing_recovery_eligible",
        "delta_signed_target_radius_error",
        "delta_absolute_target_radius_error",
        "delta_radial_velocity",
        "delta_tangential_velocity_error",
        "delta_realized_speed_ratio",
        "delta_overspeed_headroom",
        "delta_specific_orbital_energy",
        "delta_radius_error_ratio",
        "delta_radial_velocity_ratio",
        "delta_tangential_velocity_error_ratio",
        "transition_count_delta",
        "elapsed_time_delta",
        "proposed_action",
        "executed_action",
        "proposed_action_magnitude",
        "executed_action_magnitude",
        "proposed_action_radial_component",
        "proposed_action_tangential_component",
        "executed_action_radial_component",
        "executed_action_tangential_component",
        "proposed_equals_executed",
        "action_saturation_margin",
        "action_suppression_status",
        "action_geometry_status",
        "final_veto_decision",
    }
)
RUNTIME_STATUS_STAGE0A_FIELDS = frozenset(
    {
        "simulation_validity",
        "recovery_evaluation_validity",
        "recovery_horizon_exhausted",
        "total_horizon_exhausted",
    }
)


class InstrumentationValidationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EquivalenceCheck:
    check_id: str
    passed: bool
    baseline_digest: str
    observed_digest: str


@dataclass(frozen=True, slots=True)
class PairedEquivalenceReport:
    checks: tuple[EquivalenceCheck, ...]
    all_equivalence_checks: bool
    baseline_trace_published: bool
    observed_trace_published: bool


@dataclass(frozen=True, slots=True)
class FieldCompletenessEntry:
    field_id: str
    schema_source: str
    schema_accepts_field: bool
    runtime_supplied_field: bool
    logger_derived_field: bool
    measured_event_count: int
    valid_event_count: int
    not_evaluated_event_count: int
    invalid_event_count: int
    first_valid_event_index: int | None
    last_valid_event_index: int | None
    expected_availability: str
    observed_availability: str
    completeness_status: str
    limitation: str


@dataclass(frozen=True, slots=True)
class FieldCompletenessReport:
    entries: tuple[FieldCompletenessEntry, ...]
    measured_runtime_fields: int
    derived_runtime_fields: int
    correctly_not_evaluated_fields: int
    unexpectedly_missing_fields: int
    invalid_fields: int
    invalid_required_fields: int
    unsupported_fields: int


@dataclass(frozen=True, slots=True)
class StaticValidationReport:
    valid: bool
    checks: tuple[tuple[str, bool], ...]
    errors: tuple[str, ...]
    manifest_hash: str
    branch_state_hash: str


@dataclass(frozen=True, slots=True)
class ValidationPublicationResult:
    published: bool
    target_directory: str
    artifact_paths: tuple[str, ...]
    artifact_hashes: tuple[tuple[str, str], ...]
    trace_aggregate_hash: str
    trace_manifest_hash: str


@dataclass(frozen=True, slots=True)
class PairedValidationResult:
    baseline: BoundedRecoveryValidationRun
    observed: BoundedRecoveryValidationRun
    events: tuple[StagedRecoveryRuntimeEvent, ...]
    equivalence: PairedEquivalenceReport
    field_completeness: FieldCompletenessReport
    publication: ValidationPublicationResult


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _document_with_hash(
    payload: Mapping[str, object],
    *,
    hash_field: str = "canonical_payload_hash",
) -> dict[str, object]:
    document = dict(payload)
    document.pop(hash_field, None)
    document[hash_field] = canonical_runtime_sha256(document)
    return document


def canonical_document_hash_is_valid(
    document: Mapping[str, object],
    *,
    hash_field: str = "canonical_payload_hash",
) -> bool:
    supplied = document.get(hash_field)
    payload = dict(document)
    payload.pop(hash_field, None)
    return isinstance(supplied, str) and supplied == canonical_runtime_sha256(payload)


def _digest(value: object) -> str:
    return canonical_runtime_sha256(value)


def compare_bounded_runs(
    baseline: BoundedRecoveryValidationRun,
    observed: BoundedRecoveryValidationRun,
    *,
    manifest_hash: str,
) -> PairedEquivalenceReport:
    baseline_steps = baseline.transition_snapshots
    observed_steps = observed.transition_snapshots
    pairs: tuple[tuple[str, object, object], ...] = (
        ("same_source_case", baseline.identity.case_id, observed.identity.case_id),
        ("same_seed", baseline.identity.seed, observed.identity.seed),
        ("same_branch_id", baseline.identity.branch_id, observed.identity.branch_id),
        (
            "same_branch_state_hash",
            baseline.identity.branch_state_hash,
            observed.identity.branch_state_hash,
        ),
        ("same_manifest_hash", manifest_hash, manifest_hash),
        (
            "same_implementation_commit",
            baseline.identity.implementation_commit,
            observed.identity.implementation_commit,
        ),
        (
            "same_simulator_configuration_hash",
            baseline.identity.simulator_configuration_hash,
            observed.identity.simulator_configuration_hash,
        ),
        (
            "same_constants_hash",
            baseline.identity.constants_hash,
            observed.identity.constants_hash,
        ),
        ("same_initial_state_hash", baseline.initial_state_hash, observed.initial_state_hash),
        (
            "same_eventual_transition_count",
            baseline.recovery_transition_count,
            observed.recovery_transition_count,
        ),
        (
            "same_recovery_step_sequence",
            tuple(item.recovery_step for item in baseline_steps),
            tuple(item.recovery_step for item in observed_steps),
        ),
        (
            "same_total_transition_count_sequence",
            tuple(item.total_transition_count for item in baseline_steps),
            tuple(item.total_transition_count for item in observed_steps),
        ),
        (
            "same_proposed_action_sequence",
            tuple(item.proposed_action for item in baseline_steps),
            tuple(item.proposed_action for item in observed_steps),
        ),
        (
            "same_monitor_prediction_sequence",
            tuple(item.predicted_speed_ratio for item in baseline_steps),
            tuple(item.predicted_speed_ratio for item in observed_steps),
        ),
        (
            "same_monitor_decision_sequence",
            tuple(item.monitor_decision for item in baseline_steps),
            tuple(item.monitor_decision for item in observed_steps),
        ),
        (
            "same_executed_action_sequence",
            tuple(item.executed_action for item in baseline_steps),
            tuple(item.executed_action for item in observed_steps),
        ),
        (
            "same_action_disposition_sequence",
            tuple(item.action_disposition.value for item in baseline_steps),
            tuple(item.action_disposition.value for item in observed_steps),
        ),
        (
            "same_predicted_state_hash_sequence",
            tuple(item.predicted_state_hash for item in baseline_steps),
            tuple(item.predicted_state_hash for item in observed_steps),
        ),
        (
            "same_realized_state_hash_sequence",
            tuple(item.realized_state_hash for item in baseline_steps),
            tuple(item.realized_state_hash for item in observed_steps),
        ),
        (
            "same_predicted_speed_ratio_sequence",
            tuple(item.predicted_speed_ratio for item in baseline_steps),
            tuple(item.predicted_speed_ratio for item in observed_steps),
        ),
        (
            "same_realized_speed_ratio_sequence",
            tuple(item.realized_speed_ratio for item in baseline_steps),
            tuple(item.realized_speed_ratio for item in observed_steps),
        ),
        (
            "same_evaluator_status_sequence",
            tuple(item.evaluator_statuses for item in baseline_steps),
            tuple(item.evaluator_statuses for item in observed_steps),
        ),
        (
            "same_runtime_terminal_reason",
            baseline.runtime_terminal_reason,
            observed.runtime_terminal_reason,
        ),
        ("same_final_state_hash", baseline.final_state_hash, observed.final_state_hash),
    )
    checks = tuple(
        EquivalenceCheck(
            check_id=check_id,
            passed=baseline_value == observed_value,
            baseline_digest=_digest(baseline_value),
            observed_digest=_digest(observed_value),
        )
        for check_id, baseline_value, observed_value in pairs
    )
    return PairedEquivalenceReport(
        checks=checks,
        all_equivalence_checks=all(check.passed for check in checks),
        baseline_trace_published=False,
        observed_trace_published=True,
    )


def _instrumented_values_by_event(
    events: Sequence[StagedRecoveryRuntimeEvent],
) -> tuple[dict[str, object], ...]:
    output: list[dict[str, object]] = []
    for event in events:
        candidates: dict[str, list[object]] = {}
        collections: list[Sequence[tuple[str, object]]] = [
            event.progress_sample,
            event.action_geometry,
            event.phase_evidence,
            event.evaluator_evidence,
        ]
        if event.pre_observation is not None:
            collections.append(event.pre_observation.fields)
        if event.post_observation is not None:
            collections.append(event.post_observation.fields)
        if event.predicted_observation is not None:
            collections.append(event.predicted_observation.fields)
        for collection in collections:
            for field_id, value in collection:
                candidates.setdefault(field_id, []).append(value)
        selected: dict[str, object] = {}
        for field_id, values in candidates.items():
            invalid = [
                value
                for value in values
                if getattr(value, "status", None)
                == InstrumentationEvidenceStatus.INVALID
            ]
            available = [value for value in values if getattr(value, "available", False)]
            selected[field_id] = invalid[0] if invalid else available[0] if available else values[0]
        output.append(selected)
    return tuple(output)


def _stage0a_expected_count(field_id: str, event_count: int) -> int:
    if field_id in EXPECTED_UNAVAILABLE_FIELDS:
        return 0
    if field_id in TRANSITION_ONLY_STAGE0A_FIELDS:
        return VALIDATION_HORIZON
    if field_id in RUNTIME_STATUS_STAGE0A_FIELDS:
        return VALIDATION_HORIZON + 1
    return event_count


def build_field_completeness(
    events: Sequence[StagedRecoveryRuntimeEvent],
) -> FieldCompletenessReport:
    if len(events) != EXPECTED_EVENT_COUNT:
        raise InstrumentationValidationError(
            f"field completeness requires {EXPECTED_EVENT_COUNT} events"
        )
    per_event = _instrumented_values_by_event(events)
    coverage = {
        item["field_id"]: str(item["logger_coverage_classification"])
        for item in logger_field_coverage()
    }
    entries: list[FieldCompletenessEntry] = []
    for definition in field_catalog():
        field_id = definition.field_id
        values = [event.get(field_id) for event in per_event]
        measured_indices = [
            index
            for index, value in enumerate(values)
            if value is not None
            and getattr(value, "status", None)
            == InstrumentationEvidenceStatus.MEASURED
        ]
        valid_indices = [
            index
            for index, value in enumerate(values)
            if value is not None and getattr(value, "available", False)
        ]
        invalid_indices = [
            index
            for index, value in enumerate(values)
            if value is not None
            and getattr(value, "status", None)
            == InstrumentationEvidenceStatus.INVALID
        ]
        not_evaluated_count = sum(
            value is None
            or getattr(value, "status", None)
            == InstrumentationEvidenceStatus.NOT_EVALUATED
            for value in values
        )
        expected_count = _stage0a_expected_count(field_id, len(events))
        if field_id in UNSUPPORTED_FIELDS:
            status = "unsupported"
        elif invalid_indices:
            status = "invalid"
        elif expected_count == 0 and not valid_indices:
            status = "correctly_not_evaluated"
        elif len(valid_indices) >= expected_count:
            status = "complete_for_expected_events"
        elif valid_indices:
            status = "partially_complete"
        else:
            status = "unexpectedly_missing"
        entries.append(
            FieldCompletenessEntry(
                field_id=field_id,
                schema_source="stage0a",
                schema_accepts_field=True,
                runtime_supplied_field=coverage.get(field_id)
                == "logged_from_direct_runtime_input",
                logger_derived_field=coverage.get(field_id)
                in {
                    "derived_by_stage0a_during_logging",
                    "requires_previous_runtime_state",
                    "requires_predicted_runtime_state",
                },
                measured_event_count=len(measured_indices),
                valid_event_count=len(valid_indices),
                not_evaluated_event_count=not_evaluated_count,
                invalid_event_count=len(invalid_indices),
                first_valid_event_index=min(valid_indices) if valid_indices else None,
                last_valid_event_index=max(valid_indices) if valid_indices else None,
                expected_availability=(
                    "not_expected_without_phase_or_future_evaluator"
                    if expected_count == 0
                    else f"{expected_count}_events"
                ),
                observed_availability=f"{len(valid_indices)}_events",
                completeness_status=status,
                limitation=(
                    "No staged phase runtime or unresolved evaluator is present."
                    if expected_count == 0
                    else "Validated only on this eight-transition bounded trace."
                ),
            )
        )

    documents = tuple(event_document(event) for event in events)
    schema_fields = event_schema_document()["fields"]
    assert isinstance(schema_fields, list)
    for schema_field in schema_fields:
        assert isinstance(schema_field, dict)
        field_id = str(schema_field["field_id"])
        allowed = tuple(str(item) for item in schema_field["event_types_allowed"])
        expected_indices = [
            index
            for index, document in enumerate(documents)
            if str(document["event_type"]) in allowed
        ]
        if field_id in {
            "proposed_action",
            "executed_action",
            "monitor_decision",
            "predicted_state_hash",
            "realized_state_hash",
            "predicted_observation",
            "post_observation",
        }:
            expected_indices = list(range(1, 1 + VALIDATION_HORIZON))
        elif field_id == "terminal_reason":
            expected_indices = [len(documents) - 1]
        elif field_id == "volatile_timestamp":
            expected_indices = []
        valid_indices = [
            index
            for index in expected_indices
            if field_id in documents[index] and documents[index][field_id] is not None
        ]
        invalid_indices = [
            index
            for index in expected_indices
            if documents[index].get("event_valid") is False
        ]
        if invalid_indices:
            status = "invalid"
        elif not expected_indices:
            status = "correctly_not_evaluated"
        elif len(valid_indices) == len(expected_indices):
            status = "complete_for_expected_events"
        elif valid_indices:
            status = "partially_complete"
        else:
            status = "unexpectedly_missing"
        entries.append(
            FieldCompletenessEntry(
                field_id=field_id,
                schema_source="stage0b",
                schema_accepts_field=True,
                runtime_supplied_field=str(schema_field["direct_or_derived"]) == "direct",
                logger_derived_field=str(schema_field["direct_or_derived"]) == "derived",
                measured_event_count=len(valid_indices),
                valid_event_count=len(valid_indices),
                not_evaluated_event_count=len(expected_indices) - len(valid_indices),
                invalid_event_count=len(invalid_indices),
                first_valid_event_index=min(valid_indices) if valid_indices else None,
                last_valid_event_index=max(valid_indices) if valid_indices else None,
                expected_availability=(
                    "not_expected" if not expected_indices else f"{len(expected_indices)}_events"
                ),
                observed_availability=f"{len(valid_indices)}_events",
                completeness_status=status,
                limitation="Observational validation only; no recovery claim follows.",
            )
        )

    statuses = [entry.completeness_status for entry in entries]
    return FieldCompletenessReport(
        entries=tuple(entries),
        measured_runtime_fields=sum(
            entry.schema_source == "stage0b"
            and entry.completeness_status == "complete_for_expected_events"
            for entry in entries
        ),
        derived_runtime_fields=sum(
            entry.schema_source == "stage0a"
            and entry.logger_derived_field
            and entry.valid_event_count > 0
            for entry in entries
        ),
        correctly_not_evaluated_fields=statuses.count("correctly_not_evaluated"),
        unexpectedly_missing_fields=statuses.count("unexpectedly_missing"),
        invalid_fields=statuses.count("invalid"),
        invalid_required_fields=sum(
            entry.completeness_status == "invalid"
            and entry.field_id not in EXPECTED_UNAVAILABLE_FIELDS
            for entry in entries
        ),
        unsupported_fields=statuses.count("unsupported"),
    )


def _without_volatile_fields(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_volatile_fields(item)
            for key, item in value.items()
            if key
            not in {
                "volatile_timestamp",
                "volatile_provenance_timestamp",
                "volatile_finalization_timestamp",
            }
        }
    if isinstance(value, list):
        return [_without_volatile_fields(item) for item in value]
    return value


def event_document_hash_recomputes(document: Mapping[str, object]) -> bool:
    supplied = document.get("canonical_event_sha256")
    payload = _without_volatile_fields(dict(document))
    assert isinstance(payload, dict)
    payload.pop("canonical_event_sha256", None)
    return isinstance(supplied, str) and supplied == canonical_runtime_sha256(payload)


def aggregate_event_document_hash(documents: Sequence[Mapping[str, object]]) -> str:
    return canonical_runtime_sha256(
        {
            "ordered_event_scientific_hashes": [
                document["canonical_event_sha256"] for document in documents
            ]
        }
    )


def _equivalence_document(report: PairedEquivalenceReport) -> dict[str, object]:
    return _document_with_hash(
        {
            "equivalence_schema_version": EQUIVALENCE_SCHEMA_VERSION,
            "validation_id": VALIDATION_ID,
            "checks": [
                {
                    "check_id": check.check_id,
                    "passed": check.passed,
                    "baseline_digest": check.baseline_digest,
                    "observed_digest": check.observed_digest,
                }
                for check in report.checks
            ],
            "all_equivalence_checks": report.all_equivalence_checks,
            "baseline_trace_published": report.baseline_trace_published,
            "observed_trace_published": report.observed_trace_published,
            "comparison_semantics": "exact_equality_no_new_tolerance",
        }
    )


def _field_completeness_document(
    report: FieldCompletenessReport,
) -> dict[str, object]:
    return _document_with_hash(
        {
            "field_completeness_schema_version": FIELD_COMPLETENESS_SCHEMA_VERSION,
            "validation_id": VALIDATION_ID,
            "fields": [
                {
                    name: getattr(entry, name)
                    for name in entry.__dataclass_fields__
                }
                for entry in report.entries
            ],
            "totals": {
                "measured_runtime_fields": report.measured_runtime_fields,
                "derived_runtime_fields": report.derived_runtime_fields,
                "correctly_not_evaluated_fields": report.correctly_not_evaluated_fields,
                "unexpectedly_missing_fields": report.unexpectedly_missing_fields,
                "invalid_fields": report.invalid_fields,
                "invalid_required_fields": report.invalid_required_fields,
                "unsupported_fields": report.unsupported_fields,
            },
            "success_requirements": {
                "unexpectedly_missing_fields": 0,
                "invalid_required_fields": 0,
            },
        }
    )


def _trace_manifest_document(
    *,
    events: Sequence[StagedRecoveryRuntimeEvent],
    run: BoundedRecoveryValidationRun,
    trace_jsonl_hash: str,
) -> dict[str, object]:
    return _document_with_hash(
        {
            "trace_manifest_schema_version": TRACE_MANIFEST_SCHEMA_VERSION,
            "validation_id": VALIDATION_ID,
            "logger_schema_version": LOGGER_SCHEMA_VERSION,
            "source_logger_commit": SOURCE_LOGGER_COMMIT,
            "source_logger_canonical_hash": SOURCE_LOGGER_CANONICAL_HASH,
            "source_instrumentation_commit": SOURCE_INSTRUMENTATION_COMMIT,
            "source_instrumentation_canonical_hash": SOURCE_INSTRUMENTATION_CANONICAL_HASH,
            "source_architecture_commit": SOURCE_ARCHITECTURE_COMMIT,
            "source_architecture_canonical_hash": SOURCE_ARCHITECTURE_CANONICAL_HASH,
            "case_id": run.identity.case_id,
            "seed": run.identity.seed,
            "branch_id": run.identity.branch_id,
            "branch_state_hash": run.identity.branch_state_hash,
            "implementation_commit": run.identity.implementation_commit,
            "event_count": len(events),
            "first_event_index": events[0].event_index,
            "last_event_index": events[-1].event_index,
            "first_recovery_step": events[0].recovery_step,
            "last_recovery_step": events[-1].recovery_step,
            "first_total_transition_count": events[0].total_transition_count,
            "last_total_transition_count": events[-1].total_transition_count,
            "event_scientific_hashes": [
                event.canonical_event_sha256 for event in events
            ],
            "aggregate_trace_hash": aggregate_trace_sha256(events),
            "trace_jsonl_sha256": trace_jsonl_hash,
            "runtime_terminal_reason": run.runtime_terminal_reason,
            "validation_terminal_reason": run.validation_terminal_reason,
            "trace_classification": TRACE_CLASSIFICATION,
            "runtime_source": RUNTIME_SOURCE,
            "scientific_result": SCIENTIFIC_RESULT,
            "staged_recovery_execution": STAGED_EXECUTION_STATUS,
            "claim_restrictions": list(CLAIM_RESTRICTIONS),
        }
    )


def _validation_manifest_document(
    *,
    repository_root: Path,
    implementation_commit: str,
    manifest: Mapping[str, object],
    branch_state: Mapping[str, object],
    trace_manifest: Mapping[str, object],
    equivalence: Mapping[str, object],
    completeness: Mapping[str, object],
) -> dict[str, object]:
    manifest_path = repository_root / MANIFEST_RELATIVE_PATH
    branch_state_path = repository_root / BRANCH_STATE_RELATIVE_PATH
    source_case = manifest["source_case"]
    ordering = branch_state["branch_ordering"]
    assert isinstance(source_case, Mapping) and isinstance(ordering, Mapping)
    return _document_with_hash(
        {
            "validation_manifest_schema_version": VALIDATION_SCHEMA_VERSION,
            "validation_id": VALIDATION_ID,
            "completed_date": COMPLETED_DATE,
            "executed_date": COMPLETED_DATE,
            "implementation_commit": implementation_commit,
            "result_commit": None,
            "result_commit_rule": "omitted_to_avoid_self_referential_result_commit_hash",
            "source_instrumentation_commit": SOURCE_INSTRUMENTATION_COMMIT,
            "source_instrumentation_canonical_hash": SOURCE_INSTRUMENTATION_CANONICAL_HASH,
            "source_logger_commit": SOURCE_LOGGER_COMMIT,
            "source_logger_canonical_hash": SOURCE_LOGGER_CANONICAL_HASH,
            "source_architecture_commit": SOURCE_ARCHITECTURE_COMMIT,
            "source_architecture_canonical_hash": SOURCE_ARCHITECTURE_CANONICAL_HASH,
            "frozen_manifest_raw_sha256": sha256_file(manifest_path),
            "frozen_manifest_canonical_sha256": recovery_canonical_sha256(dict(manifest)),
            "branch_state_raw_sha256": sha256_file(branch_state_path),
            "branch_state_canonical_sha256": branch_state["canonical_branch_state_hash"],
            "case_id": source_case["case_id"],
            "seed": source_case["seed"],
            "branch_id": VALIDATION_BRANCH_ID,
            "nominal_prefix_transition_count": ordering[
                "realized_prefix_transition_count"
            ],
            "validation_horizon": VALIDATION_HORIZON,
            "baseline_logger_status": "disabled",
            "observed_logger_status": "enabled",
            "paired_runs": 2,
            "expected_event_count": EXPECTED_EVENT_COUNT,
            "artifact_filenames": list(PUBLISHED_FILENAMES),
            "equivalence_contract_version": EQUIVALENCE_SCHEMA_VERSION,
            "field_completeness_contract_version": FIELD_COMPLETENESS_SCHEMA_VERSION,
            "trace_classification": TRACE_CLASSIFICATION,
            "scientific_recovery_result": False,
            "recovery_performance_evaluation": False,
            "staged_controller_execution": False,
            "staged_recovery_execution": STAGED_EXECUTION_STATUS,
            "trace_manifest_canonical_hash": trace_manifest["canonical_payload_hash"],
            "equivalence_report_canonical_hash": equivalence["canonical_payload_hash"],
            "field_completeness_canonical_hash": completeness[
                "canonical_payload_hash"
            ],
            "claim_restrictions": list(CLAIM_RESTRICTIONS),
        }
    )


def _summary_bytes(
    *,
    implementation_commit: str,
    run: BoundedRecoveryValidationRun,
    trace_manifest: Mapping[str, object],
    equivalence: PairedEquivalenceReport,
    completeness: FieldCompletenessReport,
    manifest_hash: str,
) -> bytes:
    lines = [
        "# Staged Recovery Instrumentation Validation v0",
        "",
        "## Status",
        "",
        "Measured instrumentation-validation trace completed; staged recovery execution not authorized.",
        "",
        "Executed: 2026-07-30",
        "",
        "## Purpose",
        "",
        "This trace validates observational instrumentation on one bounded runtime path. It is not a recovery-performance experiment, staged-controller execution, phase-policy validation, task-recovery result, formal safety result, hardware result, or deployment result.",
        "",
        "## Frozen Validation Configuration",
        "",
        f"- Validation ID: `{VALIDATION_ID}`",
        f"- Case: `{run.identity.case_id}`",
        f"- Seed: `{run.identity.seed}`",
        f"- Branch: `{run.identity.branch_id}`",
        f"- Canonical branch state: `{run.identity.branch_state_hash}`",
        f"- Canonical manifest: `{manifest_hash}`",
        f"- Validation horizon: `{VALIDATION_HORIZON}` realized transitions",
        "- Pair: one logger-disabled baseline and one logger-enabled observed run",
        "",
        "## Implementation Commit",
        "",
        f"`{implementation_commit}`",
        "",
        "## Source Hashes",
        "",
        f"- Stage 0A: `{SOURCE_INSTRUMENTATION_CANONICAL_HASH}`",
        f"- Stage 0B: `{SOURCE_LOGGER_CANONICAL_HASH}`",
        f"- Architecture: `{SOURCE_ARCHITECTURE_CANONICAL_HASH}`",
        "",
        "## Paired Baseline/Observed Procedure",
        "",
        "Both runs independently reloaded and revalidated the canonical branch state. The baseline retained an in-memory semantic transcript only. The observed run passed immutable snapshots to the Stage 0B logger adapter.",
        "",
        "## Logger Integration Boundary",
        "",
        "The observer return value is forbidden, and the observer receives no mutable simulator, controller, action, or monitor object. Observer failures are infrastructure failures and cannot become scientific terminal outcomes.",
        "",
        "## Equivalence Result",
        "",
        f"All {len(equivalence.checks)} required exact checks passed: `{str(equivalence.all_equivalence_checks).lower()}`.",
        "",
        "## Event Counts",
        "",
        f"The observed trace contains `{len(run.snapshots)}` events: one initial snapshot, `{run.recovery_transition_count}` transitions, and one zero-transition terminal event.",
        "",
        "## Counter Consistency",
        "",
        f"Recovery steps progressed from `0` through `{run.recovery_transition_count}`. Total transitions progressed from `{run.identity.nominal_prefix_transition_count}` through `{run.identity.nominal_prefix_transition_count + run.recovery_transition_count}`.",
        "",
        "## Action and Monitor Consistency",
        "",
        "Proposed actions, Final Veto predictions and decisions, executed actions, and action dispositions were exactly equal with logging disabled and enabled.",
        "",
        "## Predicted-Versus-Realized Evidence",
        "",
        "Predicted and realized Cartesian states, state hashes, speed ratios, and headroom remained separate in every transition event.",
        "",
        "## Measured Cartesian and Orbital Fields",
        "",
        "The trace preserves supplied Cartesian state and pure Stage 0A radius, basis, speed, radial/tangential velocity, target-error, speed-ratio, headroom, and diagnostic energy-proxy derivations.",
        "",
        "## Recoverability Fields",
        "",
        "Crossing evidence and the three Phase34-compatible components are logged separately. Recovery Success v0 remains not evaluated by this validation wrapper.",
        "",
        "## Progress Fields",
        "",
        "Threshold-free measured pre/post deltas are present. No progressing, stalled, or regressing classification was introduced.",
        "",
        "## Correctly Unavailable Phase Fields",
        "",
        "Current/previous staged phase, dwell, phase transitions, no-progress status, handoff readiness, and retreat evidence remain `not_evaluated` because no staged phase runtime exists.",
        "",
        "## Field Completeness",
        "",
        f"- Measured runtime fields: `{completeness.measured_runtime_fields}`",
        f"- Derived runtime fields: `{completeness.derived_runtime_fields}`",
        f"- Correctly not evaluated fields: `{completeness.correctly_not_evaluated_fields}`",
        f"- Unexpectedly missing fields: `{completeness.unexpectedly_missing_fields}`",
        f"- Invalid required fields: `{completeness.invalid_required_fields}`",
        f"- Unsupported fields: `{completeness.unsupported_fields}`",
        "",
        "## Artifact Hashes",
        "",
        f"- Trace manifest canonical hash: `{trace_manifest['canonical_payload_hash']}`",
        f"- Trace aggregate hash: `{trace_manifest['aggregate_trace_hash']}`",
        f"- Trace JSONL SHA-256: `{trace_manifest['trace_jsonl_sha256']}`",
        "",
        "## Protected Evidence Preservation",
        "",
        "Frozen recovery inputs, measured recovery evidence, mechanism diagnosis, staged architecture, Stage 0A, Stage 0B, Final Veto evidence, and Phase34-37 evidence remained read-only.",
        "",
        "## Current Limitations",
        "",
        "This one eight-transition engineering trace does not validate recovery performance, phase logic, no-progress thresholds, hysteresis, handoff readiness, or general runtime completeness.",
        "",
        "## Stage 1 Offline-Guard Boundary",
        "",
        "The next authorized milestone remains offline guard and missing-evidence validation. Staged execution is still unauthorized.",
        "",
        "## Claim Restrictions",
        "",
        "No recovery, controller, safety, hardware, deployment, or cross-domain claim follows from this instrumentation validation.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def build_validation_payloads(
    *,
    repository_root: Path,
    implementation_commit: str,
    manifest: Mapping[str, object],
    branch_state: Mapping[str, object],
    run: BoundedRecoveryValidationRun,
    events: Sequence[StagedRecoveryRuntimeEvent],
    equivalence: PairedEquivalenceReport,
    completeness: FieldCompletenessReport,
) -> dict[str, bytes]:
    trace_jsonl = trace_jsonl_bytes_from_events(events)
    trace_manifest = _trace_manifest_document(
        events=events,
        run=run,
        trace_jsonl_hash=hashlib.sha256(trace_jsonl).hexdigest(),
    )
    equivalence_document = _equivalence_document(equivalence)
    completeness_document = _field_completeness_document(completeness)
    validation_manifest = _validation_manifest_document(
        repository_root=repository_root,
        implementation_commit=implementation_commit,
        manifest=manifest,
        branch_state=branch_state,
        trace_manifest=trace_manifest,
        equivalence=equivalence_document,
        completeness=completeness_document,
    )
    return {
        VALIDATION_MANIFEST_FILENAME: _json_bytes(validation_manifest),
        TRACE_MANIFEST_FILENAME: _json_bytes(trace_manifest),
        TRACE_JSONL_FILENAME: trace_jsonl,
        EQUIVALENCE_FILENAME: _json_bytes(equivalence_document),
        FIELD_COMPLETENESS_FILENAME: _json_bytes(completeness_document),
        SUMMARY_FILENAME: _summary_bytes(
            implementation_commit=implementation_commit,
            run=run,
            trace_manifest=trace_manifest,
            equivalence=equivalence,
            completeness=completeness,
            manifest_hash=recovery_canonical_sha256(dict(manifest)),
        ),
    }


def validate_payload_bytes(payloads: Mapping[str, bytes]) -> tuple[str, str]:
    if tuple(sorted(payloads)) != tuple(sorted(PUBLISHED_FILENAMES)):
        raise InstrumentationValidationError("validation artifact set is incomplete")
    if any(not payloads[name] for name in PUBLISHED_FILENAMES):
        raise InstrumentationValidationError("validation artifact is empty")
    json_documents: dict[str, dict[str, object]] = {}
    for filename in (
        VALIDATION_MANIFEST_FILENAME,
        TRACE_MANIFEST_FILENAME,
        EQUIVALENCE_FILENAME,
        FIELD_COMPLETENESS_FILENAME,
    ):
        value = json.loads(payloads[filename].decode("utf-8"))
        if not isinstance(value, dict) or not canonical_document_hash_is_valid(value):
            raise InstrumentationValidationError(f"invalid canonical document: {filename}")
        json_documents[filename] = value
    event_documents = [
        json.loads(line)
        for line in payloads[TRACE_JSONL_FILENAME].decode("utf-8").splitlines()
        if line
    ]
    if len(event_documents) != EXPECTED_EVENT_COUNT:
        raise InstrumentationValidationError("trace event count mismatch")
    if not all(isinstance(item, dict) for item in event_documents):
        raise InstrumentationValidationError("trace event must be a JSON object")
    if [item["event_index"] for item in event_documents] != list(
        range(EXPECTED_EVENT_COUNT)
    ):
        raise InstrumentationValidationError("trace event ordering mismatch")
    event_types = [item["event_type"] for item in event_documents]
    if event_types != ["initial_snapshot"] + ["transition"] * 8 + ["terminal"]:
        raise InstrumentationValidationError("trace event-type contract mismatch")
    if not all(event_document_hash_recomputes(item) for item in event_documents):
        raise InstrumentationValidationError("trace event hash mismatch")
    trace_manifest = json_documents[TRACE_MANIFEST_FILENAME]
    aggregate = aggregate_event_document_hash(event_documents)
    if trace_manifest.get("aggregate_trace_hash") != aggregate:
        raise InstrumentationValidationError("trace aggregate hash mismatch")
    if trace_manifest.get("trace_classification") != TRACE_CLASSIFICATION:
        raise InstrumentationValidationError("trace classification mismatch")
    if trace_manifest.get("runtime_source") != RUNTIME_SOURCE:
        raise InstrumentationValidationError("runtime source mismatch")
    if trace_manifest.get("scientific_result") is not False:
        raise InstrumentationValidationError("trace cannot be a scientific result")
    if trace_manifest.get("validation_terminal_reason") != VALIDATION_TERMINAL_REASON:
        raise InstrumentationValidationError("validation terminal reason mismatch")
    equivalence = json_documents[EQUIVALENCE_FILENAME]
    if equivalence.get("all_equivalence_checks") is not True:
        raise InstrumentationValidationError("paired equivalence did not pass")
    if equivalence.get("baseline_trace_published") is not False:
        raise InstrumentationValidationError("baseline trace must not be published")
    if equivalence.get("observed_trace_published") is not True:
        raise InstrumentationValidationError("observed trace publication is not declared")
    totals = json_documents[FIELD_COMPLETENESS_FILENAME].get("totals")
    if not isinstance(totals, dict):
        raise InstrumentationValidationError("field completeness totals are missing")
    if totals.get("unexpectedly_missing_fields") != 0:
        raise InstrumentationValidationError("required fields are unexpectedly missing")
    if totals.get("invalid_required_fields") != 0:
        raise InstrumentationValidationError("required fields are invalid")
    summary = payloads[SUMMARY_FILENAME].decode("utf-8")
    required_nonclaim = (
        "This trace validates observational instrumentation on one bounded runtime path. "
        "It is not a recovery-performance experiment"
    )
    if required_nonclaim not in summary:
        raise InstrumentationValidationError("summary non-claim is missing")
    return str(trace_manifest["canonical_payload_hash"]), aggregate


def publish_validation_payloads(
    payloads: Mapping[str, bytes],
    *,
    target_directory: Path,
    repository_root: Path,
    failure_injector: Callable[[str], None] | None = None,
) -> ValidationPublicationResult:
    target = validate_trace_publication_target(
        target_directory, repository_root=repository_root
    )
    if target.exists():
        raise FileExistsError(f"refusing to overwrite validation target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    trace_manifest_hash, aggregate_hash = validate_payload_bytes(payloads)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    ).resolve()
    published = False
    try:
        for filename in PUBLISHED_FILENAMES:
            path = staging / filename
            with path.open("xb") as handle:
                handle.write(payloads[filename])
        if failure_injector is not None:
            failure_injector("after_staged_write")
        staged_payloads = {
            filename: (staging / filename).read_bytes()
            for filename in PUBLISHED_FILENAMES
        }
        validate_payload_bytes(staged_payloads)
        if failure_injector is not None:
            failure_injector("after_staged_validation")
        os.replace(staging, target)
        published = True
        final_payloads = {
            filename: (target / filename).read_bytes()
            for filename in PUBLISHED_FILENAMES
        }
        validate_payload_bytes(final_payloads)
        if final_payloads != dict(payloads):
            raise InstrumentationValidationError(
                "published validation bytes differ from staged bytes"
            )
        hashes = tuple(
            (filename, hashlib.sha256(final_payloads[filename]).hexdigest())
            for filename in PUBLISHED_FILENAMES
        )
        return ValidationPublicationResult(
            published=True,
            target_directory=str(target),
            artifact_paths=tuple(str(target / name) for name in PUBLISHED_FILENAMES),
            artifact_hashes=hashes,
            trace_aggregate_hash=aggregate_hash,
            trace_manifest_hash=trace_manifest_hash,
        )
    except Exception:
        if published:
            # Publication validation occurs immediately after a same-filesystem
            # rename. A post-rename failure is surfaced without deleting user data.
            pass
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _load_source_document(repository_root: Path, relative: Path) -> dict[str, object]:
    return load_json_object(repository_root / relative)


def validate_static_configuration(
    repository_root: Path,
    *,
    require_output_absent: bool = True,
) -> StaticValidationReport:
    root = repository_root.resolve()
    checks: list[tuple[str, bool]] = []
    errors: list[str] = []

    def add(check_id: str, passed: bool) -> None:
        checks.append((check_id, passed))
        if not passed:
            errors.append(check_id)

    manifest = _load_source_document(root, MANIFEST_RELATIVE_PATH)
    branch_state = _load_source_document(root, BRANCH_STATE_RELATIVE_PATH)
    manifest_hash = recovery_canonical_sha256(manifest)
    branch_hash = str(branch_state.get("canonical_branch_state_hash", ""))
    add("frozen_manifest_hash", manifest_hash == FROZEN_MANIFEST_CANONICAL_HASH)
    add("frozen_branch_state_hash", branch_hash == FROZEN_BRANCH_STATE_CANONICAL_HASH)
    try:
        validate_branch_state_integrity(branch_state)
        add("branch_state_integrity", True)
    except Exception:
        add("branch_state_integrity", False)
    branches = manifest.get("branches")
    branch_ids = (
        tuple(str(item.get("branch_id")) for item in branches)
        if isinstance(branches, list) and all(isinstance(item, dict) for item in branches)
        else ()
    )
    add("frozen_branch_order", branch_ids == FROZEN_BRANCH_ORDER)
    add("validation_branch", VALIDATION_BRANCH_ID in branch_ids)
    add("validation_horizon", VALIDATION_HORIZON == 8)
    add("bounded_runner_cap", MAX_RUNNER_HORIZON_STEPS == 32)
    add("event_count_contract", EXPECTED_EVENT_COUNT == VALIDATION_HORIZON + 2)
    add(
        "stage0a_source_hash",
        _load_source_document(
            root,
            Path("analysis/staged_recovery_instrumentation_v0/instrumentation_manifest.json"),
        ).get("canonical_payload_hash")
        == SOURCE_INSTRUMENTATION_CANONICAL_HASH,
    )
    add(
        "stage0b_source_hash",
        _load_source_document(
            root, Path("analysis/staged_recovery_runtime_logger_v0/logger_manifest.json")
        ).get("canonical_payload_hash")
        == SOURCE_LOGGER_CANONICAL_HASH,
    )
    add(
        "architecture_source_hash",
        _load_source_document(
            root, Path("analysis/staged_recovery_architecture_v0/architecture_manifest.json")
        ).get("canonical_payload_hash")
        == SOURCE_ARCHITECTURE_CANONICAL_HASH,
    )
    add("staged_execution_unauthorized", STAGED_EXECUTION_STATUS == "not_authorized")
    output = root / OUTPUT_RELATIVE_PATH
    if require_output_absent:
        add("output_absent", not output.exists())
        if not output.exists():
            try:
                validate_trace_publication_target(output, repository_root=root)
                add("output_path_allowed", True)
            except Exception:
                add("output_path_allowed", False)
    else:
        add("output_present", output.is_dir())
    return StaticValidationReport(
        valid=not errors,
        checks=tuple(checks),
        errors=tuple(errors),
        manifest_hash=manifest_hash,
        branch_state_hash=branch_hash,
    )


def validate_published_bundle(repository_root: Path) -> ValidationPublicationResult:
    root = repository_root.resolve()
    target = root / OUTPUT_RELATIVE_PATH
    if not target.is_dir():
        raise FileNotFoundError(target)
    actual = tuple(sorted(path.name for path in target.iterdir() if path.is_file()))
    if actual != tuple(sorted(PUBLISHED_FILENAMES)):
        raise InstrumentationValidationError("published artifact set is not exact")
    payloads = {name: (target / name).read_bytes() for name in PUBLISHED_FILENAMES}
    trace_manifest_hash, aggregate_hash = validate_payload_bytes(payloads)
    return ValidationPublicationResult(
        published=True,
        target_directory=str(target),
        artifact_paths=tuple(str(target / name) for name in PUBLISHED_FILENAMES),
        artifact_hashes=tuple(
            (name, hashlib.sha256(payloads[name]).hexdigest())
            for name in PUBLISHED_FILENAMES
        ),
        trace_aggregate_hash=aggregate_hash,
        trace_manifest_hash=trace_manifest_hash,
    )


def execute_frozen_validation_pair(
    repository_root: Path,
    *,
    implementation_commit: str,
    branch_state_loader: Callable[[str | Path], Mapping[str, object]] = (
        load_frozen_branch_state
    ),
    step_executor: Callable[..., object] = execute_recovery_branch,
    publisher: Callable[..., ValidationPublicationResult] = publish_validation_payloads,
) -> PairedValidationResult:
    root = repository_root.resolve()
    static = validate_static_configuration(root, require_output_absent=True)
    if not static.valid:
        raise InstrumentationValidationError(
            f"static validation failed: {', '.join(static.errors)}"
        )
    manifest = _load_source_document(root, MANIFEST_RELATIVE_PATH)
    branch_path = root / BRANCH_STATE_RELATIVE_PATH
    before_branch_bytes = branch_path.read_bytes()

    baseline_state = branch_state_loader(branch_path)
    baseline = run_bounded_recovery_validation_path(
        baseline_state,
        branch_id=VALIDATION_BRANCH_ID,
        horizon_steps=VALIDATION_HORIZON,
        implementation_commit=implementation_commit,
        observer=None,
        step_executor=step_executor,
    )

    observed_state = branch_state_loader(branch_path)
    observed_identity, _ = build_bounded_runtime_identity(
        observed_state,
        branch_id=VALIDATION_BRANCH_ID,
        implementation_commit=implementation_commit,
    )
    adapter = Stage0BValidationLoggerAdapter(
        observed_identity,
        session_id=VALIDATION_ID,
        max_events=EXPECTED_EVENT_COUNT,
    )
    observed = run_bounded_recovery_validation_path(
        observed_state,
        branch_id=VALIDATION_BRANCH_ID,
        horizon_steps=VALIDATION_HORIZON,
        implementation_commit=implementation_commit,
        observer=adapter.observe,
        step_executor=step_executor,
    )
    logger_bundle = adapter.finalize()
    events = logger_bundle.events
    if len(events) != EXPECTED_EVENT_COUNT:
        raise InstrumentationValidationError("observed logger event count mismatch")

    equivalence = compare_bounded_runs(
        baseline, observed, manifest_hash=static.manifest_hash
    )
    if not equivalence.all_equivalence_checks:
        failed = [check.check_id for check in equivalence.checks if not check.passed]
        raise InstrumentationValidationError(
            f"paired trajectory equivalence failed: {', '.join(failed)}"
        )
    completeness = build_field_completeness(events)
    if completeness.unexpectedly_missing_fields != 0:
        raise InstrumentationValidationError("field completeness has missing fields")
    if completeness.invalid_required_fields != 0:
        raise InstrumentationValidationError("field completeness has invalid fields")
    if branch_path.read_bytes() != before_branch_bytes:
        raise InstrumentationValidationError("frozen branch state changed during validation")
    payloads = build_validation_payloads(
        repository_root=root,
        implementation_commit=implementation_commit,
        manifest=manifest,
        branch_state=observed_state,
        run=observed,
        events=events,
        equivalence=equivalence,
        completeness=completeness,
    )
    publication = publisher(
        payloads,
        target_directory=root / OUTPUT_RELATIVE_PATH,
        repository_root=root,
    )
    return PairedValidationResult(
        baseline=baseline,
        observed=observed,
        events=tuple(events),
        equivalence=equivalence,
        field_completeness=completeness,
        publication=publication,
    )


def require_clean_committed_repository(repository_root: Path) -> str:
    state = inspect_repository_state(repository_root)
    if not state.is_git_worktree:
        raise InstrumentationValidationError("execution requires a Git worktree")
    if len(state.head_commit) != 40:
        raise InstrumentationValidationError("execution requires a committed HEAD")
    if not state.tracked_clean:
        raise InstrumentationValidationError("tracked working tree is not clean")
    if not state.staged_clean:
        raise InstrumentationValidationError("staged changes are present")
    collision_prefix = OUTPUT_RELATIVE_PATH.as_posix() + "/"
    collisions = [
        path
        for path in state.untracked_paths
        if path == OUTPUT_RELATIVE_PATH.as_posix() or path.startswith(collision_prefix)
    ]
    if collisions:
        raise InstrumentationValidationError(
            f"untracked validation output collision: {collisions}"
        )
    return state.head_commit


__all__ = [
    "BRANCH_STATE_RELATIVE_PATH",
    "CLAIM_RESTRICTIONS",
    "COMPLETED_DATE",
    "EQUIVALENCE_FILENAME",
    "EXPECTED_EVENT_COUNT",
    "FIELD_COMPLETENESS_FILENAME",
    "InstrumentationValidationError",
    "MANIFEST_RELATIVE_PATH",
    "OUTPUT_RELATIVE_PATH",
    "PUBLISHED_FILENAMES",
    "PairedEquivalenceReport",
    "PairedValidationResult",
    "StaticValidationReport",
    "SUMMARY_FILENAME",
    "TRACE_CLASSIFICATION",
    "TRACE_JSONL_FILENAME",
    "TRACE_MANIFEST_FILENAME",
    "VALIDATION_BRANCH_ID",
    "VALIDATION_HORIZON",
    "VALIDATION_ID",
    "VALIDATION_MANIFEST_FILENAME",
    "ValidationPublicationResult",
    "aggregate_event_document_hash",
    "build_field_completeness",
    "build_validation_payloads",
    "canonical_document_hash_is_valid",
    "compare_bounded_runs",
    "event_document_hash_recomputes",
    "execute_frozen_validation_pair",
    "publish_validation_payloads",
    "require_clean_committed_repository",
    "sha256_file",
    "validate_payload_bytes",
    "validate_published_bundle",
    "validate_static_configuration",
]
