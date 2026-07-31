from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


ANALYSIS_ID = "staged_recovery_guard_evidence_v0"
ANALYSIS_SCHEMA_VERSION = "staged_recovery_guard_evidence_manifest_v0"
SIGNAL_PROFILE_SCHEMA_VERSION = "staged_recovery_measured_signal_profile_v0"
GUARD_INVENTORY_SCHEMA_VERSION = "staged_recovery_guard_atom_inventory_v0"
GUARD_TRACE_SCHEMA_VERSION = "staged_recovery_guard_evaluation_trace_v0"
PHASE_MATRIX_SCHEMA_VERSION = "staged_recovery_phase_observability_matrix_v0"
NO_PROGRESS_SCHEMA_VERSION = "staged_recovery_no_progress_window_profile_v0"
TRACEABILITY_SCHEMA_VERSION = "staged_recovery_guard_evidence_traceability_v0"
COMPLETED_DATE = "2026-07-31"
ANALYZED_DATE = "2026-07-31"

SOURCE_STAGE0C_RESULT_COMMIT = "7844cc5824cf83dc84d8732e96d361d9f4b06aeb"
SOURCE_STAGE0C_VALIDATION_ID = "staged_recovery_instrumentation_validation_v0"
SOURCE_TRACE_MANIFEST_CANONICAL_HASH = (
    "23dd44711641eb2fcae9f1be81f405ee0660146862e8193bb8a0ebd871140680"
)
SOURCE_TRACE_AGGREGATE_HASH = (
    "4f3d700422c47abf4ece93c0dd54770be5f3109a49ef49708485c41ca67e962e"
)
SOURCE_EVENT_COUNT = 10
SOURCE_TRANSITION_COUNT = 8
SOURCE_BRANCH_ID = "velocity_opposed_thrust_v0"
SOURCE_SEED = 0
SOURCE_NOMINAL_PREFIX = 27

SOURCE_ARCHITECTURE_COMMIT = "0d416603027e8a27991baf4f89445f6f466b86e6"
SOURCE_ARCHITECTURE_CANONICAL_HASH = (
    "22fa7e0f01c7836ecb1f10838ef00c4cafa937d212bba579fffb25e2c8f11971"
)
SOURCE_INSTRUMENTATION_COMMIT = "ebc208aedecd11155c6ac9f03bb9b5e40bc69b10"
SOURCE_INSTRUMENTATION_CANONICAL_HASH = (
    "c4947e623e7f9a83de16163f58c5a0da7a3f7b10ee3b10ce88f4eae4805f122c"
)
SOURCE_LOGGER_COMMIT = "f92b7ffe11ca559764228a0d2500b211ad562ecf"
SOURCE_LOGGER_CANONICAL_HASH = (
    "b4f7a25e53795845895707b9d5a3d14804431f5323858854e48685f27723d6dd"
)

OVERSPEED_THRESHOLD = 1.90
RADIUS_ERROR_RATIO_MAX = 0.0025
RADIAL_VELOCITY_RATIO_MAX = 0.02
TANGENTIAL_VELOCITY_ERROR_RATIO_MAX = 0.25

SOURCE_STAGE0C_DIRECTORY = Path(
    "analysis/staged_recovery_instrumentation_validation_v0"
)
OUTPUT_RELATIVE_PATH = Path("analysis/staged_recovery_guard_evidence_v0")
ARCHITECTURE_MANIFEST_PATH = Path(
    "analysis/staged_recovery_architecture_v0/architecture_manifest.json"
)
INSTRUMENTATION_CATALOG_PATH = Path(
    "analysis/staged_recovery_instrumentation_v0/field_catalog.json"
)

ANALYSIS_MANIFEST_FILENAME = "analysis_manifest.json"
SIGNAL_PROFILE_FILENAME = "measured_signal_profile.json"
GUARD_INVENTORY_FILENAME = "guard_atom_inventory.json"
GUARD_TRACE_FILENAME = "guard_evaluation_trace.jsonl"
PHASE_MATRIX_FILENAME = "phase_observability_matrix.json"
NO_PROGRESS_FILENAME = "no_progress_window_profile.json"
TRACEABILITY_FILENAME = "evidence_traceability.json"
SUMMARY_FILENAME = "summary.md"

PUBLISHED_FILENAMES = (
    ANALYSIS_MANIFEST_FILENAME,
    SIGNAL_PROFILE_FILENAME,
    GUARD_INVENTORY_FILENAME,
    GUARD_TRACE_FILENAME,
    PHASE_MATRIX_FILENAME,
    NO_PROGRESS_FILENAME,
    TRACEABILITY_FILENAME,
    SUMMARY_FILENAME,
)

SOURCE_STAGE0C_FILENAMES = (
    "validation_manifest.json",
    "trace_manifest.json",
    "staged_recovery_trace.jsonl",
    "equivalence_report.json",
    "field_completeness.json",
    "summary.md",
)

PHASE_IDS = (
    "hazard_arrest",
    "stabilization_assessment",
    "radial_recommitment",
    "tangential_alignment",
    "crossing_preparation",
    "recoverability_verification",
    "nominal_handoff",
    "retreat",
    "explicit_abort",
)

UNRESOLVED_PARAMETER_IDS = (
    "NO_PROGRESS_WINDOW_LENGTH",
    "NO_PROGRESS_MIN_RADIUS_GAP_IMPROVEMENT",
    "NO_PROGRESS_MIN_RADIAL_COMPONENT_IMPROVEMENT",
    "NO_PROGRESS_MIN_TANGENTIAL_COMPONENT_IMPROVEMENT",
    "NO_PROGRESS_MIN_HEADROOM_IMPROVEMENT",
    "NO_PROGRESS_REQUIRED_COMPONENT_COUNT",
    "NO_PROGRESS_CONSECUTIVE_WINDOWS",
    "NO_PROGRESS_MIN_PHASE_DWELL",
    "NO_PROGRESS_COOLDOWN",
)

CLAIM_RESTRICTIONS = (
    "no_recovery_performance_claim",
    "no_phase_policy_validity_claim",
    "no_guard_false_positive_or_false_negative_claim",
    "no_general_noise_estimate_claim",
    "no_selected_no_progress_threshold",
    "no_selected_hysteresis_parameter",
    "no_staged_controller_claim",
    "no_formal_safety_claim",
    "no_hardware_or_deployment_claim",
)

PROTECTED_OUTPUT_PREFIXES = (
    Path("analysis/recovery_action_branching_nonformal_v0"),
    Path("analysis/recovery_branch_mechanism_diagnosis_v0"),
    Path("analysis/staged_recovery_architecture_v0"),
    Path("analysis/staged_recovery_instrumentation_v0"),
    Path("analysis/staged_recovery_runtime_logger_v0"),
    SOURCE_STAGE0C_DIRECTORY,
    Path("analysis/final_veto_ablation_v0"),
)


class GuardEvidenceStatus(str, Enum):
    TRUE = "true"
    FALSE = "false"
    NOT_EVALUATED = "not_evaluated"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"
    POLICY_UNRESOLVED = "policy_unresolved"


class GuardEvidenceLevel(str, Enum):
    MEASURED = "measured"
    DERIVED = "derived"
    EXTERNALLY_SUPPLIED = "externally_supplied"
    DIAGNOSTIC_PROXY = "diagnostic_proxy"
    NOT_EVALUATED = "not_evaluated"
    INVALID = "invalid"


class GuardEvidenceError(ValueError):
    pass


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if _is_finite_number(value):
        return True
    if isinstance(value, (tuple, list)):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_json(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _to_json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _to_json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {item[0]: _to_json_value(item[1]) for item in value}
        return [_to_json_value(item) for item in value]
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _to_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {item[0]: _thaw_json(item[1]) for item in value}
        return [_thaw_json(item) for item in value]
    return value


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _to_json_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _to_json_value(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def with_document_hash(document: Mapping[str, object]) -> dict[str, object]:
    payload = dict(document)
    payload.pop("canonical_payload_hash", None)
    payload["canonical_payload_hash"] = canonical_sha256(payload)
    return payload


def document_hash_recomputes(document: Mapping[str, object]) -> bool:
    payload = dict(document)
    supplied = payload.pop("canonical_payload_hash", None)
    return isinstance(supplied, str) and supplied == canonical_sha256(payload)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


@dataclass(frozen=True, slots=True)
class GuardAtomDefinition:
    guard_atom_id: str
    category: str
    description: str
    required_fields: tuple[str, ...]
    evidence_level: GuardEvidenceLevel
    comparator: str
    fixed_threshold: float | None
    threshold_source: str | None
    unresolved_parameter_ids: tuple[str, ...]
    evaluation_scope: str
    result_meaning: str
    non_meaning: str
    source_path: str
    source_symbol: str
    phase_relevance: tuple[str, ...]
    policy_authorization_status: str = "not_authorized"

    def __post_init__(self) -> None:
        if not self.guard_atom_id or not self.category or not self.description:
            raise GuardEvidenceError("guard atom identity, category, and description are required")
        if self.policy_authorization_status != "not_authorized":
            raise GuardEvidenceError("Stage 1A guard atoms cannot be authorized")
        if self.fixed_threshold is not None and not _is_finite_number(self.fixed_threshold):
            raise GuardEvidenceError("guard threshold must be finite")
        if len(set(self.required_fields)) != len(self.required_fields):
            raise GuardEvidenceError("guard required fields must be unique")
        if not set(self.phase_relevance).issubset(PHASE_IDS):
            raise GuardEvidenceError("guard references an unknown phase")
        if not set(self.unresolved_parameter_ids).issubset(UNRESOLVED_PARAMETER_IDS):
            raise GuardEvidenceError("guard references an unknown unresolved parameter")


@dataclass(frozen=True, slots=True)
class GuardAtomEvaluation:
    guard_atom_id: str
    status: GuardEvidenceStatus
    value: bool | None
    evidence_level: GuardEvidenceLevel
    raw_source_values: tuple[tuple[str, object], ...]
    comparator: str
    threshold_or_parameter_reference: str | float | None
    reason: str
    policy_authorization_status: str = "not_authorized"

    def __post_init__(self) -> None:
        if self.status == GuardEvidenceStatus.TRUE and self.value is not True:
            raise GuardEvidenceError("true guard status requires value true")
        if self.status == GuardEvidenceStatus.FALSE and self.value is not False:
            raise GuardEvidenceError("false guard status requires value false")
        if self.status not in {GuardEvidenceStatus.TRUE, GuardEvidenceStatus.FALSE} and self.value is not None:
            raise GuardEvidenceError("unknown, invalid, unsupported, and unresolved guards require null")
        if self.policy_authorization_status != "not_authorized":
            raise GuardEvidenceError("guard evaluation cannot authorize a phase transition")
        if tuple(sorted(self.raw_source_values)) != self.raw_source_values:
            raise GuardEvidenceError("guard raw source values must be sorted")
        if not _is_json_value(_to_json_value(self.raw_source_values)):
            raise GuardEvidenceError("guard raw evidence must be canonical JSON compatible")


@dataclass(frozen=True, slots=True)
class MeasuredSignalProfile:
    field_id: str
    source_event_path: str
    profile_kind: str
    units: str
    evidence_level: str
    expected_event_types: tuple[str, ...]
    statistics: tuple[tuple[str, object], ...]
    scientific_limitation: str


@dataclass(frozen=True, slots=True)
class WindowedProgressRecord:
    window_length: int
    start_recovery_step: int
    end_recovery_step: int
    start_event_index: int
    end_event_index: int
    component_changes: tuple[tuple[str, object], ...]
    component_labels: tuple[tuple[str, str], ...]
    directional_radial_commitment_count: int
    crossing_count: int
    valid_sample_count: int
    unavailable_sample_count: int
    invalid_sample_count: int
    descriptive_direction: str


@dataclass(frozen=True, slots=True)
class PhaseObservabilityEntry:
    phase_id: str
    possible_entry_evidence: tuple[str, ...]
    possible_stay_evidence: tuple[str, ...]
    possible_exit_evidence: tuple[str, ...]
    required_architecture_signals: tuple[str, ...]
    current_stage0a_schema_support: tuple[tuple[str, str], ...]
    stage0c_measured_availability: tuple[tuple[str, bool], ...]
    pure_derivation_availability: tuple[str, ...]
    previous_state_dependencies: tuple[str, ...]
    predicted_state_dependencies: tuple[str, ...]
    runtime_phase_dependencies: tuple[str, ...]
    future_evaluator_dependencies: tuple[str, ...]
    unsupported_dependencies: tuple[str, ...]
    available_guard_atoms: tuple[str, ...]
    unavailable_guard_atoms: tuple[str, ...]
    unresolved_numerical_parameters: tuple[str, ...]
    action_law_complete: bool
    current_observability_status: str
    policy_authorization: str
    strongest_permitted_interpretation: str


@dataclass(frozen=True, slots=True)
class GuardEvidenceValidationReport:
    valid: bool
    errors: tuple[str, ...]
    source_event_count: int
    source_transition_count: int
    source_trace_manifest_hash: str | None
    source_trace_aggregate_hash: str | None


@dataclass(frozen=True, slots=True)
class AnalysisPublicationResult:
    published: bool
    target_directory: str
    artifact_paths: tuple[str, ...]
    artifact_hashes: tuple[tuple[str, str], ...]
    analysis_manifest_hash: str
    guard_trace_aggregate_hash: str


@dataclass(frozen=True, slots=True)
class SourceBundle:
    validation_manifest: tuple[tuple[str, object], ...]
    trace_manifest: tuple[tuple[str, object], ...]
    equivalence_report: tuple[tuple[str, object], ...]
    field_completeness: tuple[tuple[str, object], ...]
    events: tuple[tuple[tuple[str, object], ...], ...]

    def document(self, name: str) -> dict[str, object]:
        value = _thaw_json(getattr(self, name))
        assert isinstance(value, dict)
        return value

    def event_documents(self) -> tuple[dict[str, object], ...]:
        output = []
        for event in self.events:
            value = _thaw_json(event)
            assert isinstance(value, dict)
            output.append(value)
        return tuple(output)


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardEvidenceError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GuardEvidenceError(f"JSON artifact must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    events: list[dict[str, object]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line:
                raise GuardEvidenceError(f"blank JSONL line at {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise GuardEvidenceError(f"JSONL event {line_number} is not an object")
            events.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardEvidenceError(f"cannot read valid JSONL from {path}: {exc}") from exc
    return tuple(events)


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


def source_event_hash_recomputes(event: Mapping[str, object]) -> bool:
    supplied = event.get("canonical_event_sha256")
    payload = _without_volatile_fields(dict(event))
    assert isinstance(payload, dict)
    payload.pop("canonical_event_sha256", None)
    return isinstance(supplied, str) and supplied == canonical_sha256(payload)


def aggregate_source_event_hashes(events: Sequence[Mapping[str, object]]) -> str:
    return canonical_sha256(
        {
            "ordered_event_scientific_hashes": [
                event.get("canonical_event_sha256") for event in events
            ]
        }
    )


def validate_source_bundle(repository_root: Path) -> GuardEvidenceValidationReport:
    root = repository_root.resolve()
    source = root / SOURCE_STAGE0C_DIRECTORY
    errors: list[str] = []
    missing = [name for name in SOURCE_STAGE0C_FILENAMES if not (source / name).is_file()]
    if missing:
        return GuardEvidenceValidationReport(
            False,
            tuple(f"missing_source_artifact:{name}" for name in missing),
            0,
            0,
            None,
            None,
        )
    try:
        validation = _load_json_object(source / "validation_manifest.json")
        trace_manifest = _load_json_object(source / "trace_manifest.json")
        equivalence = _load_json_object(source / "equivalence_report.json")
        completeness = _load_json_object(source / "field_completeness.json")
        events = _load_jsonl(source / "staged_recovery_trace.jsonl")
    except GuardEvidenceError as exc:
        return GuardEvidenceValidationReport(False, (str(exc),), 0, 0, None, None)

    for name, document in (
        ("validation_manifest", validation),
        ("trace_manifest", trace_manifest),
        ("equivalence_report", equivalence),
        ("field_completeness", completeness),
    ):
        if not document_hash_recomputes(document):
            errors.append(f"{name}_canonical_hash_mismatch")

    if validation.get("validation_id") != SOURCE_STAGE0C_VALIDATION_ID:
        errors.append("source_validation_id_mismatch")
    if validation.get("trace_classification") != "measured_instrumentation_validation":
        errors.append("source_trace_classification_mismatch")
    if validation.get("branch_id") != SOURCE_BRANCH_ID or validation.get("seed") != SOURCE_SEED:
        errors.append("source_branch_or_seed_mismatch")
    if validation.get("expected_event_count") != SOURCE_EVENT_COUNT:
        errors.append("source_expected_event_count_mismatch")
    if validation.get("validation_horizon") != SOURCE_TRANSITION_COUNT:
        errors.append("source_transition_count_mismatch")
    if validation.get("nominal_prefix_transition_count") != SOURCE_NOMINAL_PREFIX:
        errors.append("source_nominal_prefix_mismatch")
    if validation.get("source_instrumentation_commit") != SOURCE_INSTRUMENTATION_COMMIT:
        errors.append("source_stage0a_commit_mismatch")
    if validation.get("source_logger_commit") != SOURCE_LOGGER_COMMIT:
        errors.append("source_stage0b_commit_mismatch")
    if validation.get("source_architecture_commit") != SOURCE_ARCHITECTURE_COMMIT:
        errors.append("source_architecture_commit_mismatch")
    if validation.get("source_instrumentation_canonical_hash") != SOURCE_INSTRUMENTATION_CANONICAL_HASH:
        errors.append("source_stage0a_hash_mismatch")
    if validation.get("source_logger_canonical_hash") != SOURCE_LOGGER_CANONICAL_HASH:
        errors.append("source_stage0b_hash_mismatch")
    if validation.get("source_architecture_canonical_hash") != SOURCE_ARCHITECTURE_CANONICAL_HASH:
        errors.append("source_architecture_hash_mismatch")

    trace_hash = trace_manifest.get("canonical_payload_hash")
    aggregate_hash = trace_manifest.get("aggregate_trace_hash")
    if trace_hash != SOURCE_TRACE_MANIFEST_CANONICAL_HASH:
        errors.append("source_trace_manifest_identity_mismatch")
    if aggregate_hash != SOURCE_TRACE_AGGREGATE_HASH:
        errors.append("source_trace_aggregate_identity_mismatch")
    if trace_manifest.get("trace_classification") != "measured_instrumentation_validation":
        errors.append("source_trace_is_not_measured_validation")
    if trace_manifest.get("scientific_result") is not False:
        errors.append("source_trace_scientific_result_must_be_false")
    if trace_manifest.get("event_count") != len(events):
        errors.append("source_event_count_manifest_mismatch")
    if sha256_file(source / "staged_recovery_trace.jsonl") != trace_manifest.get("trace_jsonl_sha256"):
        errors.append("source_trace_jsonl_hash_mismatch")
    if len(events) != SOURCE_EVENT_COUNT:
        errors.append("source_event_count_mismatch")
    if [event.get("event_index") for event in events] != list(range(SOURCE_EVENT_COUNT)):
        errors.append("source_event_order_mismatch")
    expected_types = ["initial_snapshot"] + ["transition"] * SOURCE_TRANSITION_COUNT + ["terminal"]
    if [event.get("event_type") for event in events] != expected_types:
        errors.append("source_event_type_order_mismatch")
    if not all(source_event_hash_recomputes(event) for event in events):
        errors.append("source_event_hash_mismatch")
    recomputed_aggregate = aggregate_source_event_hashes(events)
    if recomputed_aggregate != aggregate_hash:
        errors.append("source_trace_aggregate_recomputation_mismatch")
    if equivalence.get("all_equivalence_checks") is not True:
        errors.append("source_equivalence_not_complete")
    checks = equivalence.get("checks")
    if not isinstance(checks, list) or len(checks) != 24 or not all(
        isinstance(item, dict) and item.get("passed") is True for item in checks
    ):
        errors.append("source_equivalence_checks_invalid")
    totals = completeness.get("totals")
    if not isinstance(totals, dict):
        errors.append("source_field_completeness_totals_missing")
    else:
        if totals.get("unexpectedly_missing_fields") != 0:
            errors.append("source_fields_unexpectedly_missing")
        if totals.get("invalid_fields") != 0 or totals.get("invalid_required_fields") != 0:
            errors.append("source_fields_invalid")

    transition_count = sum(event.get("event_type") == "transition" for event in events)
    return GuardEvidenceValidationReport(
        valid=not errors,
        errors=tuple(errors),
        source_event_count=len(events),
        source_transition_count=transition_count,
        source_trace_manifest_hash=str(trace_hash) if isinstance(trace_hash, str) else None,
        source_trace_aggregate_hash=(
            str(aggregate_hash) if isinstance(aggregate_hash, str) else None
        ),
    )


def load_validated_source_bundle(repository_root: Path) -> SourceBundle:
    report = validate_source_bundle(repository_root)
    if not report.valid:
        raise GuardEvidenceError("source Stage 0C validation failed: " + ",".join(report.errors))
    source = repository_root.resolve() / SOURCE_STAGE0C_DIRECTORY
    validation = _load_json_object(source / "validation_manifest.json")
    trace_manifest = _load_json_object(source / "trace_manifest.json")
    equivalence = _load_json_object(source / "equivalence_report.json")
    completeness = _load_json_object(source / "field_completeness.json")
    events = _load_jsonl(source / "staged_recovery_trace.jsonl")
    return SourceBundle(
        validation_manifest=_freeze_json(validation),
        trace_manifest=_freeze_json(trace_manifest),
        equivalence_report=_freeze_json(equivalence),
        field_completeness=_freeze_json(completeness),
        events=tuple(_freeze_json(event) for event in events),
    )


def _pairs_to_map(value: object) -> dict[str, dict[str, object]]:
    if isinstance(value, dict):
        return {
            str(key): item
            for key, item in value.items()
            if isinstance(key, str) and isinstance(item, dict)
        }
    if not isinstance(value, list):
        return {}
    output: dict[str, dict[str, object]] = {}
    for pair in value:
        if (
            isinstance(pair, list)
            and len(pair) == 2
            and isinstance(pair[0], str)
            and isinstance(pair[1], dict)
        ):
            output[pair[0]] = pair[1]
    return output


def _observation_fields(observation: object) -> dict[str, dict[str, object]]:
    if not isinstance(observation, dict):
        return {}
    return _pairs_to_map(observation.get("fields"))


def _current_observation(event: Mapping[str, object]) -> dict[str, object] | None:
    event_type = event.get("event_type")
    selected = event.get("post_observation") if event_type == "transition" else event.get("pre_observation")
    return selected if isinstance(selected, dict) else None


def _current_fields(event: Mapping[str, object]) -> dict[str, dict[str, object]]:
    return _observation_fields(_current_observation(event))


def _context_fields(event: Mapping[str, object], context: str) -> dict[str, dict[str, object]]:
    if context in {"pre_observation", "post_observation"}:
        return _observation_fields(event.get(context))
    if context == "predicted_observation":
        predicted = event.get(context)
        if not isinstance(predicted, dict):
            return {}
        return _pairs_to_map(predicted.get("fields"))
    return _pairs_to_map(event.get(context))


def _evidence_status(evidence: object) -> str:
    if not isinstance(evidence, dict):
        return "not_evaluated"
    status = evidence.get("status")
    return str(status) if isinstance(status, str) else "invalid"


def _evidence_value(evidence: object) -> object:
    return evidence.get("value") if isinstance(evidence, dict) else None


def _evidence_valid(evidence: object) -> bool:
    return isinstance(evidence, dict) and evidence.get("valid") is True


def _numeric_evidence(evidence: object) -> tuple[str, float | None]:
    status = _evidence_status(evidence)
    value = _evidence_value(evidence)
    if status == "invalid":
        return "invalid", None
    if status in {"not_evaluated", "unsupported"} or value is None:
        return status if status in {"not_evaluated", "unsupported"} else "not_evaluated", None
    if not _is_finite_number(value):
        return "invalid", None
    return "available", float(value)


def _boolean_evidence(evidence: object) -> tuple[str, bool | None]:
    status = _evidence_status(evidence)
    value = _evidence_value(evidence)
    if status == "invalid":
        return "invalid", None
    if status in {"not_evaluated", "unsupported"} or value is None:
        return status if status in {"not_evaluated", "unsupported"} else "not_evaluated", None
    if not isinstance(value, bool):
        return "invalid", None
    return "available", value


def _guard_result(
    definition: GuardAtomDefinition,
    *,
    status: GuardEvidenceStatus,
    value: bool | None,
    raw: Mapping[str, object] | None,
    reason: str,
    evidence_level: GuardEvidenceLevel | None = None,
) -> GuardAtomEvaluation:
    reference: str | float | None = definition.fixed_threshold
    if reference is None and definition.unresolved_parameter_ids:
        reference = ",".join(definition.unresolved_parameter_ids)
    frozen_raw = tuple(
        (key, _freeze_json(item)) for key, item in sorted((raw or {}).items())
    )
    return GuardAtomEvaluation(
        guard_atom_id=definition.guard_atom_id,
        status=status,
        value=value,
        evidence_level=evidence_level or definition.evidence_level,
        raw_source_values=frozen_raw,
        comparator=definition.comparator,
        threshold_or_parameter_reference=reference,
        reason=reason,
    )


def _evaluated_boolean(
    definition: GuardAtomDefinition,
    value: bool,
    raw: Mapping[str, object],
    reason: str,
    *,
    evidence_level: GuardEvidenceLevel | None = None,
) -> GuardAtomEvaluation:
    return _guard_result(
        definition,
        status=GuardEvidenceStatus.TRUE if value else GuardEvidenceStatus.FALSE,
        value=value,
        raw=raw,
        reason=reason,
        evidence_level=evidence_level,
    )


def _unavailable_guard(
    definition: GuardAtomDefinition,
    *,
    invalid: bool = False,
    unsupported: bool = False,
    policy_unresolved: bool = False,
    raw: Mapping[str, object] | None = None,
    reason: str,
) -> GuardAtomEvaluation:
    if invalid:
        status = GuardEvidenceStatus.INVALID
        level = GuardEvidenceLevel.INVALID
    elif unsupported:
        status = GuardEvidenceStatus.UNSUPPORTED
        level = GuardEvidenceLevel.NOT_EVALUATED
    elif policy_unresolved:
        status = GuardEvidenceStatus.POLICY_UNRESOLVED
        level = GuardEvidenceLevel.NOT_EVALUATED
    else:
        status = GuardEvidenceStatus.NOT_EVALUATED
        level = GuardEvidenceLevel.NOT_EVALUATED
    return _guard_result(
        definition,
        status=status,
        value=None,
        raw=raw,
        reason=reason,
        evidence_level=level,
    )


def _definition(
    guard_atom_id: str,
    category: str,
    description: str,
    required_fields: Sequence[str],
    comparator: str,
    *,
    evidence_level: GuardEvidenceLevel = GuardEvidenceLevel.DERIVED,
    fixed_threshold: float | None = None,
    threshold_source: str | None = None,
    unresolved: Sequence[str] = (),
    source_path: str,
    source_symbol: str,
    phases: Sequence[str],
    meaning: str,
    non_meaning: str,
) -> GuardAtomDefinition:
    return GuardAtomDefinition(
        guard_atom_id=guard_atom_id,
        category=category,
        description=description,
        required_fields=tuple(required_fields),
        evidence_level=evidence_level,
        comparator=comparator,
        fixed_threshold=fixed_threshold,
        threshold_source=threshold_source,
        unresolved_parameter_ids=tuple(unresolved),
        evaluation_scope="one checked-in eight-transition Stage 0C validation trace",
        result_meaning=meaning,
        non_meaning=non_meaning,
        source_path=source_path,
        source_symbol=source_symbol,
        phase_relevance=tuple(phases),
    )


EXACT_INHERITED_GUARD_ATOM_IDS = (
    "state_evidence_valid",
    "instrumentation_evaluation_valid",
    "recovery_evaluation_valid",
    "realized_overspeed",
    "realized_overspeed_clear",
    "predicted_overspeed",
    "predicted_overspeed_clear",
    "recoverability_radius_component_pass",
    "recoverability_radial_velocity_component_pass",
    "recoverability_tangential_velocity_component_pass",
    "phase34_compatible_recoverability_pass",
    "eligible_target_radius_crossing",
    "pre_branch_crossing_only",
    "no_eligible_crossing",
)


THRESHOLD_FREE_DIRECTIONAL_GUARD_ATOM_IDS = (
    "absolute_radius_gap_improving",
    "absolute_radius_gap_unchanged",
    "absolute_radius_gap_worsening",
    "radial_velocity_toward_target",
    "radial_velocity_away_from_target",
    "radial_velocity_no_directional_commitment",
    "absolute_tangential_error_improving",
    "absolute_tangential_error_unchanged",
    "absolute_tangential_error_worsening",
    "overspeed_headroom_improving",
    "overspeed_headroom_unchanged",
    "overspeed_headroom_worsening",
    "recoverability_radius_component_improving",
    "recoverability_radius_component_unchanged",
    "recoverability_radius_component_worsening",
    "recoverability_radial_component_improving",
    "recoverability_radial_component_unchanged",
    "recoverability_radial_component_worsening",
    "recoverability_tangential_component_improving",
    "recoverability_tangential_component_unchanged",
    "recoverability_tangential_component_worsening",
    "absolute_energy_proxy_error_improving",
    "absolute_energy_proxy_error_unchanged",
    "absolute_energy_proxy_error_worsening",
)


def guard_atom_definitions() -> tuple[GuardAtomDefinition, ...]:
    evaluator_source = "runtime_assurance/recovery_evaluators.py"
    instrumentation_source = "runtime_assurance/staged_recovery_instrumentation.py"
    trace_source = "analysis/staged_recovery_instrumentation_validation_v0/staged_recovery_trace.jsonl"
    definitions: list[GuardAtomDefinition] = []

    for atom_id, field_id, description in (
        ("state_evidence_valid", "event_valid", "The supplied runtime event and selected state evidence are structurally valid."),
        ("instrumentation_evaluation_valid", "selected_observation", "The selected instrumentation observation contains no invalid evidence."),
        ("recovery_evaluation_valid", "recovery_evaluation_validity", "The externally supplied recovery-evaluation validity evidence is true."),
    ):
        definitions.append(
            _definition(
                atom_id,
                "validity",
                description,
                (field_id,),
                "is_true",
                evidence_level=GuardEvidenceLevel.MEASURED,
                source_path=trace_source,
                source_symbol=field_id,
                phases=PHASE_IDS,
                meaning="This evidence item is valid at this event.",
                non_meaning="Validity alone does not establish stabilization, recovery, or phase readiness.",
            )
        )

    for atom_id, field_id, comparator, description in (
        ("realized_overspeed", "realized_speed_ratio", ">", "Realized speed ratio exceeds the frozen strict hazard threshold."),
        ("realized_overspeed_clear", "realized_speed_ratio", "<=", "Realized speed ratio does not exceed the frozen strict hazard threshold."),
        ("predicted_overspeed", "predicted_speed_ratio", ">", "One-step predicted speed ratio exceeds the frozen strict hazard threshold."),
        ("predicted_overspeed_clear", "predicted_speed_ratio", "<=", "One-step predicted speed ratio does not exceed the frozen strict hazard threshold."),
    ):
        definitions.append(
            _definition(
                atom_id,
                "hazard",
                description,
                (field_id,),
                comparator,
                fixed_threshold=OVERSPEED_THRESHOLD,
                threshold_source="analysis/recovery_action_branching_nonformal_v0/manifest.json:hazard_definition",
                source_path=instrumentation_source,
                source_symbol="derive_orbital_state/derive_predicted_hazard_state",
                phases=("hazard_arrest", "stabilization_assessment"),
                meaning="The selected realized or predicted ratio satisfies this one threshold comparison.",
                non_meaning="A clear atom is not a general safety or recovery claim; predicted and realized evidence remain separate.",
            )
        )

    component_specs = (
        ("recoverability_radius_component_pass", "radius_error_ratio", RADIUS_ERROR_RATIO_MAX, "PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX"),
        ("recoverability_radial_velocity_component_pass", "radial_velocity_ratio", RADIAL_VELOCITY_RATIO_MAX, "PHASE34_RECOVERABLE_VR_RATIO_MAX"),
        ("recoverability_tangential_velocity_component_pass", "tangential_velocity_error_ratio", TANGENTIAL_VELOCITY_ERROR_RATIO_MAX, "PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX"),
    )
    for atom_id, field_id, threshold, symbol in component_specs:
        definitions.append(
            _definition(
                atom_id,
                "recoverability",
                f"Absolute {field_id} satisfies the existing inclusive Phase34-compatible bound.",
                (field_id,),
                "absolute_value_<=",
                fixed_threshold=threshold,
                threshold_source=f"{evaluator_source}:{symbol}",
                source_path=evaluator_source,
                source_symbol="evaluate_phase34_compatible_recoverability",
                phases=("crossing_preparation", "recoverability_verification", "nominal_handoff"),
                meaning="This single Phase34-compatible component is inside its inherited inclusive bound.",
                non_meaning="One passing component does not establish combined recoverability or handoff readiness.",
            )
        )
    definitions.append(
        _definition(
            "phase34_compatible_recoverability_pass",
            "recoverability",
            "All three absolute Phase34-compatible component bounds pass inclusively.",
            ("radius_error_ratio", "radial_velocity_ratio", "tangential_velocity_error_ratio"),
            "all_inclusive_absolute_component_bounds",
            source_path=evaluator_source,
            source_symbol="evaluate_phase34_compatible_recoverability",
            phases=("recoverability_verification", "nominal_handoff"),
            meaning="The measured state satisfies the existing component predicate.",
            non_meaning="This is not crossing, Recovery Success v0, nominal handoff readiness, or simulator success.",
        )
    )

    for atom_id, description in (
        ("eligible_target_radius_crossing", "A measured crossing occurred at or after the branch boundary."),
        ("pre_branch_crossing_only", "Crossing evidence exists but is ineligible because it predates the branch boundary."),
        ("no_eligible_crossing", "The evaluated measured state pair contains no eligible crossing."),
    ):
        definitions.append(
            _definition(
                atom_id,
                "crossing",
                description,
                ("target_radius_crossing", "crossing_recovery_eligible"),
                "phase34_signed_radius_error_transition_rule",
                source_path=instrumentation_source,
                source_symbol="derive_crossing_event",
                phases=("crossing_preparation", "recoverability_verification"),
                meaning="This describes discrete measured crossing evidence for one transition.",
                non_meaning="No interpolation, future crossing, or recovery outcome is inferred.",
            )
        )

    directional_specs = (
        ("absolute_radius_gap", "absolute_target_radius_error", "kinematic_direction", ("radial_recommitment", "crossing_preparation"), GuardEvidenceLevel.DERIVED),
        ("absolute_tangential_error", "tangential_velocity_error", "kinematic_direction", ("tangential_alignment", "crossing_preparation"), GuardEvidenceLevel.DERIVED),
        ("overspeed_headroom", "overspeed_headroom", "hazard", ("hazard_arrest", "stabilization_assessment"), GuardEvidenceLevel.DERIVED),
        ("recoverability_radius_component", "radius_error_ratio", "recoverability", ("crossing_preparation", "recoverability_verification"), GuardEvidenceLevel.DERIVED),
        ("recoverability_radial_component", "radial_velocity_ratio", "recoverability", ("radial_recommitment", "recoverability_verification"), GuardEvidenceLevel.DERIVED),
        ("recoverability_tangential_component", "tangential_velocity_error_ratio", "recoverability", ("tangential_alignment", "recoverability_verification"), GuardEvidenceLevel.DERIVED),
        ("absolute_energy_proxy_error", "specific_energy_error", "progress", ("radial_recommitment", "tangential_alignment"), GuardEvidenceLevel.DIAGNOSTIC_PROXY),
    )
    for prefix, field_id, category, phases, level in directional_specs:
        target_is_higher = field_id == "overspeed_headroom"
        for suffix, comparator in (
            ("improving", ">" if target_is_higher else "absolute_current_< _absolute_previous"),
            ("unchanged", "exact_equal"),
            ("worsening", "<" if target_is_higher else "absolute_current_>_absolute_previous"),
        ):
            atom_id = f"{prefix}_{suffix}"
            definitions.append(
                _definition(
                    atom_id,
                    category,
                    f"Consecutive measured {field_id} evidence is {suffix} under its explicit target direction.",
                    (f"previous.{field_id}", f"current.{field_id}"),
                    comparator,
                    evidence_level=level,
                    source_path=trace_source,
                    source_symbol="threshold_free_consecutive_state_comparison",
                    phases=phases,
                    meaning=f"This component changed in the named direction over one measured transition.",
                    non_meaning="One component direction does not establish adequate progress, recovery, or a phase transition.",
                )
            )

    for atom_id, comparator, description in (
        ("radial_velocity_toward_target", "signed_radius_error*radial_velocity<0", "The signed radius error and radial velocity point toward the target radius."),
        ("radial_velocity_away_from_target", "signed_radius_error*radial_velocity>0", "The signed radius error and radial velocity point away from the target radius."),
        ("radial_velocity_no_directional_commitment", "signed_radius_error*radial_velocity==0", "The sign product provides no directional commitment."),
    ):
        definitions.append(
            _definition(
                atom_id,
                "kinematic_direction",
                description,
                ("signed_target_radius_error", "radial_velocity"),
                comparator,
                source_path=trace_source,
                source_symbol="threshold_free_radial_direction_product",
                phases=("radial_recommitment", "crossing_preparation"),
                meaning="The current measured radial motion has the stated direction relative to target radius.",
                non_meaning="The sign condition does not establish adequate magnitude, recoverability, or future crossing.",
            )
        )

    for atom_id, category, required, comparator, meaning, non_meaning in (
        ("final_veto_allow", "action", ("final_veto_decision",), "equals_allow", "The externally supplied monitor decision allowed this proposal.", "This does not establish action quality or task progress."),
        ("action_executed_unchanged", "action", ("proposed_equals_executed",), "is_true", "The supplied proposed and executed vectors are exactly equal.", "Equality does not establish action optimality or recovery."),
        ("action_not_rejected", "action", ("action_rejection_status",), "is_false", "The supplied action-rejection status is false.", "No rejection does not authorize phase progression."),
        ("phase_runtime_available", "phase_runtime", ("current_phase",), "evidence_available", "A compatible staged phase identity was externally supplied.", "Availability would not validate phase policy."),
        ("handoff_readiness_available", "handoff", ("handoff_readiness",), "evidence_available", "A handoff-readiness evaluator supplied evidence.", "Availability alone would not authorize handoff."),
        ("correction_authority_available", "authority", ("available_correction_authority",), "evidence_available", "A correction-authority evaluator supplied evidence.", "Availability alone would not establish sufficient control authority."),
        ("no_progress_policy_evaluable", "progress", ("no_progress_status",), "requires_frozen_parameters", "A complete frozen no-progress policy could be evaluated.", "Component samples do not select a no-progress policy."),
        ("nominal_handoff_authorized", "handoff", ("recovery_success_v0", "handoff_readiness"), "requires_frozen_policy", "A complete policy authorized nominal handoff.", "No Stage 1A evidence authorizes handoff."),
    ):
        unresolved = ()
        if atom_id == "no_progress_policy_evaluable":
            unresolved = UNRESOLVED_PARAMETER_IDS
        definitions.append(
            _definition(
                atom_id,
                category,
                meaning,
                required,
                comparator,
                evidence_level=GuardEvidenceLevel.EXTERNALLY_SUPPLIED,
                unresolved=unresolved,
                source_path=trace_source,
                source_symbol=required[0],
                phases=PHASE_IDS if category == "progress" else ("nominal_handoff",) if category == "handoff" else PHASE_IDS,
                meaning=meaning,
                non_meaning=non_meaning,
            )
        )

    definitions.append(
        _definition(
            "explicit_abort_requested",
            "phase_runtime",
            "The externally supplied explicit-abort request is true.",
            ("explicit_abort_requested",),
            "is_true",
            evidence_level=GuardEvidenceLevel.EXTERNALLY_SUPPLIED,
            source_path=instrumentation_source,
            source_symbol="build_instrumentation_record.explicit_abort_requested",
            phases=("explicit_abort",),
            meaning="The runtime evidence explicitly records an abort request at this event.",
            non_meaning="A false or unavailable value does not define an autonomous abort policy.",
        )
    )

    ordered = tuple(definitions)
    ids = tuple(item.guard_atom_id for item in ordered)
    if len(set(ids)) != len(ids):
        raise GuardEvidenceError("guard atom IDs must be unique")
    if not set(EXACT_INHERITED_GUARD_ATOM_IDS).issubset(ids):
        raise GuardEvidenceError("exact inherited guard inventory is incomplete")
    if not set(THRESHOLD_FREE_DIRECTIONAL_GUARD_ATOM_IDS).issubset(ids):
        raise GuardEvidenceError("threshold-free directional guard inventory is incomplete")
    return ordered


def _compare_triplet(
    current: float,
    previous: float,
    *,
    target_higher: bool,
    absolute: bool,
) -> tuple[bool, bool, bool]:
    current_value = abs(current) if absolute else current
    previous_value = abs(previous) if absolute else previous
    if current_value == previous_value:
        return False, True, False
    improved = current_value > previous_value if target_higher else current_value < previous_value
    return improved, False, not improved


def evaluate_guard_atoms_for_event(
    event: Mapping[str, object],
    previous_state_event: Mapping[str, object] | None,
    definitions: Sequence[GuardAtomDefinition] | None = None,
) -> tuple[GuardAtomEvaluation, ...]:
    definition_items = tuple(definitions or guard_atom_definitions())
    by_id = {item.guard_atom_id: item for item in definition_items}
    current = _current_fields(event)
    previous = _current_fields(previous_state_event) if previous_state_event is not None else {}
    predicted = _context_fields(event, "predicted_observation")
    action = _context_fields(event, "action_geometry")
    phase = _context_fields(event, "phase_evidence")
    evaluator = _context_fields(event, "evaluator_evidence")
    evaluations: dict[str, GuardAtomEvaluation] = {}

    state_valid_def = by_id["state_evidence_valid"]
    state_valid = event.get("event_valid")
    if isinstance(state_valid, bool):
        evaluations[state_valid_def.guard_atom_id] = _evaluated_boolean(
            state_valid_def, state_valid, {"event_valid": state_valid}, "supplied_event_validity"
        )
    else:
        evaluations[state_valid_def.guard_atom_id] = _unavailable_guard(
            state_valid_def, invalid=True, raw={"event_valid": state_valid}, reason="event_valid_is_malformed"
        )

    instrumentation_def = by_id["instrumentation_evaluation_valid"]
    invalid_fields = sorted(field_id for field_id, evidence in current.items() if _evidence_status(evidence) == "invalid")
    evaluations[instrumentation_def.guard_atom_id] = _evaluated_boolean(
        instrumentation_def,
        not invalid_fields,
        {"invalid_field_ids": invalid_fields},
        "selected_observation_contains_no_invalid_fields" if not invalid_fields else "selected_observation_contains_invalid_fields",
    )

    recovery_def = by_id["recovery_evaluation_valid"]
    recovery_evidence = evaluator.get("recovery_evaluation_validity") or current.get("recovery_evaluation_validity")
    recovery_status, recovery_value = _boolean_evidence(recovery_evidence)
    if recovery_status == "available":
        evaluations[recovery_def.guard_atom_id] = _evaluated_boolean(
            recovery_def, bool(recovery_value), {"recovery_evaluation_validity": recovery_value}, "externally_supplied_recovery_evaluation_validity"
        )
    else:
        evaluations[recovery_def.guard_atom_id] = _unavailable_guard(
            recovery_def,
            invalid=recovery_status == "invalid",
            raw={"status": recovery_status},
            reason="recovery_evaluation_validity_unavailable_or_invalid",
        )

    for atom_id, source_map, field_id in (
        ("realized_overspeed", current, "realized_speed_ratio"),
        ("realized_overspeed_clear", current, "realized_speed_ratio"),
        ("predicted_overspeed", predicted, "predicted_speed_ratio"),
        ("predicted_overspeed_clear", predicted, "predicted_speed_ratio"),
    ):
        definition = by_id[atom_id]
        status, value = _numeric_evidence(source_map.get(field_id))
        if status == "available" and value is not None:
            result = value > OVERSPEED_THRESHOLD if atom_id.endswith("overspeed") else value <= OVERSPEED_THRESHOLD
            level = GuardEvidenceLevel.EXTERNALLY_SUPPLIED if atom_id.startswith("predicted") else GuardEvidenceLevel.DERIVED
            evaluations[atom_id] = _evaluated_boolean(
                definition,
                result,
                {field_id: value},
                "strict_overspeed_comparator" if atom_id.endswith("overspeed") else "strict_hazard_clear_complement",
                evidence_level=level,
            )
        else:
            evaluations[atom_id] = _unavailable_guard(
                definition,
                invalid=status == "invalid",
                raw={field_id: _evidence_value(source_map.get(field_id)), "status": status},
                reason=f"{field_id}_unavailable_or_invalid",
            )

    component_atoms = (
        ("recoverability_radius_component_pass", "radius_error_ratio", RADIUS_ERROR_RATIO_MAX),
        ("recoverability_radial_velocity_component_pass", "radial_velocity_ratio", RADIAL_VELOCITY_RATIO_MAX),
        ("recoverability_tangential_velocity_component_pass", "tangential_velocity_error_ratio", TANGENTIAL_VELOCITY_ERROR_RATIO_MAX),
    )
    component_results: list[bool] = []
    component_problem: str | None = None
    for atom_id, field_id, threshold in component_atoms:
        definition = by_id[atom_id]
        status, value = _numeric_evidence(current.get(field_id))
        if status == "available" and value is not None:
            passed = abs(value) <= threshold
            component_results.append(passed)
            evaluations[atom_id] = _evaluated_boolean(
                definition, passed, {field_id: value}, "inclusive_absolute_phase34_component_bound"
            )
        else:
            component_problem = "invalid" if status == "invalid" else "missing"
            evaluations[atom_id] = _unavailable_guard(
                definition,
                invalid=status == "invalid",
                raw={field_id: _evidence_value(current.get(field_id)), "status": status},
                reason=f"{field_id}_unavailable_or_invalid",
            )
    combined_def = by_id["phase34_compatible_recoverability_pass"]
    if component_problem:
        evaluations[combined_def.guard_atom_id] = _unavailable_guard(
            combined_def,
            invalid=component_problem == "invalid",
            reason="combined_recoverability_requires_all_valid_components",
        )
    else:
        values = {
            field_id: _evidence_value(current.get(field_id))
            for _, field_id, _ in component_atoms
        }
        evaluations[combined_def.guard_atom_id] = _evaluated_boolean(
            combined_def,
            all(component_results),
            values,
            "all_inclusive_absolute_phase34_component_bounds",
        )

    crossing = current.get("target_radius_crossing")
    eligible = current.get("crossing_recovery_eligible")
    crossing_status, crossing_value = _boolean_evidence(crossing)
    eligible_status, eligible_value = _boolean_evidence(eligible)
    crossing_evaluable = event.get("event_type") == "transition"
    for atom_id in ("eligible_target_radius_crossing", "pre_branch_crossing_only", "no_eligible_crossing"):
        definition = by_id[atom_id]
        if not crossing_evaluable:
            evaluations[atom_id] = _unavailable_guard(
                definition, reason="crossing_requires_a_measured_transition_pair"
            )
        elif crossing_status == "invalid" or eligible_status == "invalid":
            evaluations[atom_id] = _unavailable_guard(
                definition, invalid=True, reason="crossing_evidence_is_invalid"
            )
        elif crossing_status != "available" or eligible_status != "available":
            evaluations[atom_id] = _unavailable_guard(
                definition, reason="crossing_evidence_is_unavailable"
            )
        else:
            result = (
                bool(crossing_value and eligible_value)
                if atom_id == "eligible_target_radius_crossing"
                else bool(crossing_value and not eligible_value)
                if atom_id == "pre_branch_crossing_only"
                else bool(not (crossing_value and eligible_value))
            )
            evaluations[atom_id] = _evaluated_boolean(
                definition,
                result,
                {"target_radius_crossing": crossing_value, "crossing_recovery_eligible": eligible_value},
                "discrete_measured_crossing_evidence",
            )

    directional_groups = (
        ("absolute_radius_gap", "absolute_target_radius_error", False, True),
        ("absolute_tangential_error", "tangential_velocity_error", False, True),
        ("overspeed_headroom", "overspeed_headroom", True, False),
        ("recoverability_radius_component", "radius_error_ratio", False, True),
        ("recoverability_radial_component", "radial_velocity_ratio", False, True),
        ("recoverability_tangential_component", "tangential_velocity_error_ratio", False, True),
        ("absolute_energy_proxy_error", "specific_energy_error", False, True),
    )
    for prefix, field_id, target_higher, absolute in directional_groups:
        ids = (f"{prefix}_improving", f"{prefix}_unchanged", f"{prefix}_worsening")
        current_status, current_value = _numeric_evidence(current.get(field_id))
        previous_status, previous_value = _numeric_evidence(previous.get(field_id))
        if event.get("event_type") != "transition" or previous_state_event is None:
            for atom_id in ids:
                evaluations[atom_id] = _unavailable_guard(
                    by_id[atom_id], reason="directional_atom_requires_consecutive_measured_transition_states"
                )
        elif "invalid" in {current_status, previous_status}:
            for atom_id in ids:
                evaluations[atom_id] = _unavailable_guard(
                    by_id[atom_id], invalid=True, reason="directional_source_evidence_is_invalid"
                )
        elif current_status != "available" or previous_status != "available" or current_value is None or previous_value is None:
            for atom_id in ids:
                evaluations[atom_id] = _unavailable_guard(
                    by_id[atom_id], reason="directional_source_evidence_is_unavailable"
                )
        else:
            triplet = _compare_triplet(current_value, previous_value, target_higher=target_higher, absolute=absolute)
            level = GuardEvidenceLevel.DIAGNOSTIC_PROXY if prefix == "absolute_energy_proxy_error" else GuardEvidenceLevel.DERIVED
            for atom_id, result in zip(ids, triplet):
                evaluations[atom_id] = _evaluated_boolean(
                    by_id[atom_id],
                    result,
                    {f"current.{field_id}": current_value, f"previous.{field_id}": previous_value},
                    "exact_threshold_free_consecutive_comparison",
                    evidence_level=level,
                )

    radial_ids = (
        "radial_velocity_toward_target",
        "radial_velocity_away_from_target",
        "radial_velocity_no_directional_commitment",
    )
    gap_status, gap_value = _numeric_evidence(current.get("signed_target_radius_error"))
    radial_status, radial_value = _numeric_evidence(current.get("radial_velocity"))
    if gap_status == "invalid" or radial_status == "invalid":
        for atom_id in radial_ids:
            evaluations[atom_id] = _unavailable_guard(
                by_id[atom_id], invalid=True, reason="radial_direction_evidence_is_invalid"
            )
    elif gap_status != "available" or radial_status != "available" or gap_value is None or radial_value is None:
        for atom_id in radial_ids:
            evaluations[atom_id] = _unavailable_guard(
                by_id[atom_id], reason="radial_direction_evidence_is_unavailable"
            )
    else:
        product = gap_value * radial_value
        results = (product < 0.0, product > 0.0, product == 0.0)
        for atom_id, result in zip(radial_ids, results):
            evaluations[atom_id] = _evaluated_boolean(
                by_id[atom_id],
                result,
                {"signed_target_radius_error": gap_value, "radial_velocity": radial_value, "sign_product": product},
                "exact_signed_radius_error_radial_velocity_product",
            )

    action_atoms = (
        ("final_veto_allow", event.get("monitor_decision"), "allow"),
        ("action_executed_unchanged", _evidence_value(action.get("proposed_equals_executed")), True),
        ("action_not_rejected", _evidence_value(evaluator.get("action_rejection_status") or current.get("action_rejection_status")), False),
    )
    for atom_id, supplied, expected in action_atoms:
        definition = by_id[atom_id]
        if event.get("event_type") != "transition":
            evaluations[atom_id] = _unavailable_guard(definition, reason="action_atom_requires_transition_event")
        elif isinstance(expected, bool) and not isinstance(supplied, bool):
            evaluations[atom_id] = _unavailable_guard(definition, invalid=supplied is not None, reason="action_boolean_evidence_unavailable_or_invalid")
        elif isinstance(expected, str) and not isinstance(supplied, str):
            evaluations[atom_id] = _unavailable_guard(definition, invalid=supplied is not None, reason="monitor_decision_unavailable_or_invalid")
        else:
            evaluations[atom_id] = _evaluated_boolean(
                definition,
                supplied == expected,
                {definition.required_fields[0]: supplied},
                "externally_supplied_action_or_monitor_evidence",
                evidence_level=GuardEvidenceLevel.EXTERNALLY_SUPPLIED,
            )

    abort_def = by_id["explicit_abort_requested"]
    abort_evidence = current.get("explicit_abort_requested")
    abort_status = _evidence_status(abort_evidence)
    abort_value = _evidence_value(abort_evidence)
    if abort_status == "invalid":
        evaluations[abort_def.guard_atom_id] = _unavailable_guard(
            abort_def,
            invalid=True,
            raw={"explicit_abort_requested": abort_value},
            reason="explicit_abort_evidence_is_invalid",
        )
    elif _evidence_valid(abort_evidence) and isinstance(abort_value, bool):
        evaluations[abort_def.guard_atom_id] = _evaluated_boolean(
            abort_def,
            abort_value,
            {"explicit_abort_requested": abort_value},
            "externally_supplied_explicit_abort_evidence",
            evidence_level=GuardEvidenceLevel.EXTERNALLY_SUPPLIED,
        )
    else:
        evaluations[abort_def.guard_atom_id] = _unavailable_guard(
            abort_def,
            raw={"explicit_abort_requested": abort_value},
            reason="explicit_abort_evidence_is_unavailable",
        )

    phase_def = by_id["phase_runtime_available"]
    phase_evidence = phase.get("current_phase") or current.get("current_phase")
    if _evidence_valid(phase_evidence) and isinstance(_evidence_value(phase_evidence), str):
        evaluations[phase_def.guard_atom_id] = _evaluated_boolean(
            phase_def, True, {"current_phase": _evidence_value(phase_evidence)}, "externally_supplied_phase_runtime"
        )
    else:
        evaluations[phase_def.guard_atom_id] = _unavailable_guard(
            phase_def, invalid=_evidence_status(phase_evidence) == "invalid", reason="staged_phase_runtime_not_integrated"
        )

    handoff_def = by_id["handoff_readiness_available"]
    handoff = phase.get("handoff_readiness") or current.get("handoff_readiness")
    if _evidence_valid(handoff) and isinstance(_evidence_value(handoff), bool):
        evaluations[handoff_def.guard_atom_id] = _evaluated_boolean(
            handoff_def, True, {"handoff_readiness": _evidence_value(handoff)}, "externally_supplied_handoff_evaluator"
        )
    else:
        evaluations[handoff_def.guard_atom_id] = _unavailable_guard(
            handoff_def, invalid=_evidence_status(handoff) == "invalid", reason="handoff_readiness_evaluator_not_implemented"
        )

    authority_def = by_id["correction_authority_available"]
    evaluations[authority_def.guard_atom_id] = _unavailable_guard(
        authority_def,
        unsupported=True,
        raw={"available_correction_authority": _evidence_value(current.get("available_correction_authority"))},
        reason="available_correction_authority_is_declared_unsupported",
    )

    for atom_id, reason in (
        ("no_progress_policy_evaluable", "no_progress_parameters_are_not_frozen"),
        ("nominal_handoff_authorized", "handoff_policy_and_evaluator_are_not_frozen"),
    ):
        evaluations[atom_id] = _unavailable_guard(
            by_id[atom_id], policy_unresolved=True, reason=reason
        )

    return tuple(evaluations[item.guard_atom_id] for item in definition_items)


def _missing_evidence(reason: str = "field_not_present_for_event") -> dict[str, object]:
    return {
        "value": None,
        "status": "not_evaluated",
        "reason": reason,
        "units": "not_applicable",
        "source_id": "offline_guard_evidence_profile",
        "source_step": None,
        "valid": False,
        "input_source_ids": [],
    }


def _direct_evidence(value: object, units: str = "categorical") -> dict[str, object]:
    if value is None:
        return _missing_evidence("direct_event_field_is_null")
    valid = _is_json_value(value)
    return {
        "value": value if valid else None,
        "status": "measured" if valid else "invalid",
        "reason": "direct_runtime_event_field" if valid else "direct_runtime_event_field_is_invalid",
        "units": units,
        "source_id": "stage0c.runtime_event",
        "source_step": None,
        "valid": valid,
        "input_source_ids": [],
    }


def _event_profile_evidence(event: Mapping[str, object]) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    top_units = {
        "event_index": "event",
        "recovery_step": "transition",
        "total_transition_count": "transition",
        "simulation_time": "s",
        "transition_executed": "boolean",
        "terminal": "boolean",
        "event_valid": "boolean",
        "seed": "integer",
    }
    omitted = {
        "pre_observation",
        "post_observation",
        "predicted_observation",
        "progress_sample",
        "action_geometry",
        "phase_evidence",
        "evaluator_evidence",
        "prediction_diagnostics",
        "canonical_event_sha256",
        "volatile_timestamp",
    }
    for key, value in event.items():
        if key in omitted:
            continue
        output[f"event.{key}"] = _direct_evidence(value, top_units.get(key, "categorical"))

    contexts = (
        ("realized", _current_fields(event)),
        ("pre", _context_fields(event, "pre_observation")),
        ("post", _context_fields(event, "post_observation")),
        ("predicted", _context_fields(event, "predicted_observation")),
        ("progress", _context_fields(event, "progress_sample")),
        ("action", _context_fields(event, "action_geometry")),
        ("phase", _context_fields(event, "phase_evidence")),
        ("evaluator", _context_fields(event, "evaluator_evidence")),
        ("prediction_diagnostic", _context_fields(event, "prediction_diagnostics")),
    )
    for prefix, values in contexts:
        for field_id, evidence in values.items():
            output[f"{prefix}.{field_id}"] = evidence
    return output


def _evidence_level_for_profile(path: str, samples: Sequence[dict[str, object]]) -> str:
    if "specific_energy" in path or "orbital_energy" in path:
        return GuardEvidenceLevel.DIAGNOSTIC_PROXY.value
    statuses = {
        str(sample.get("status"))
        for sample in samples
        if sample.get("valid") is True and sample.get("value") is not None
    }
    if not statuses:
        return (
            GuardEvidenceLevel.INVALID.value
            if any(sample.get("status") == "invalid" for sample in samples)
            else GuardEvidenceLevel.NOT_EVALUATED.value
        )
    if statuses == {"measured"}:
        return GuardEvidenceLevel.MEASURED.value
    if statuses.issubset({"derived"}):
        return GuardEvidenceLevel.DERIVED.value
    if statuses & {"one_step_predicted", "multi_step_predicted", "heuristic"}:
        return GuardEvidenceLevel.EXTERNALLY_SUPPLIED.value
    return GuardEvidenceLevel.DERIVED.value


def _profile_kind_from_samples(samples: Sequence[dict[str, object]]) -> str:
    values = [sample.get("value") for sample in samples if sample.get("valid") is True]
    if any(isinstance(value, bool) for value in values):
        return "boolean"
    if any(_is_finite_number(value) for value in values):
        return "numeric"
    units = next(
        (str(sample.get("units")) for sample in samples if sample.get("units") is not None),
        "categorical",
    )
    if units == "boolean":
        return "boolean"
    if units not in {"categorical", "sha256", "not_applicable"}:
        return "numeric"
    return "categorical"


def _expand_profile_series(
    path: str,
    samples: Sequence[dict[str, object]],
) -> tuple[tuple[str, tuple[dict[str, object], ...]], ...]:
    valid_values = [sample.get("value") for sample in samples if sample.get("valid") is True]
    exemplar = next((value for value in valid_values if value is not None), None)
    component_names: tuple[str, ...] = ()
    pair_vector = False
    if isinstance(exemplar, list) and exemplar and all(_is_finite_number(item) for item in exemplar):
        component_names = tuple(f"component_{index}" for index in range(len(exemplar)))
    elif (
        isinstance(exemplar, list)
        and exemplar
        and all(
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and _is_finite_number(item[1])
            for item in exemplar
        )
    ):
        component_names = tuple(str(item[0]) for item in exemplar)
        pair_vector = True
    if not component_names:
        return ((path, tuple(samples)),)

    output: list[tuple[str, tuple[dict[str, object], ...]]] = []
    for component_index, component_name in enumerate(component_names):
        component_samples: list[dict[str, object]] = []
        for sample in samples:
            copied = dict(sample)
            value = sample.get("value")
            if sample.get("valid") is not True or value is None:
                copied["value"] = None
            elif not isinstance(value, list) or component_index >= len(value):
                copied.update(value=None, status="invalid", valid=False, reason="vector_component_is_missing")
            else:
                candidate = value[component_index][1] if pair_vector else value[component_index]
                if _is_finite_number(candidate):
                    copied["value"] = candidate
                else:
                    copied.update(value=None, status="invalid", valid=False, reason="vector_component_is_invalid")
            component_samples.append(copied)
        output.append((f"{path}.{component_name}", tuple(component_samples)))

    if not pair_vector:
        magnitude_samples: list[dict[str, object]] = []
        for sample in samples:
            copied = dict(sample)
            value = sample.get("value")
            if sample.get("valid") is not True or value is None:
                copied["value"] = None
            elif isinstance(value, list) and all(_is_finite_number(item) for item in value):
                copied["value"] = math.sqrt(sum(float(item) ** 2 for item in value))
            else:
                copied.update(value=None, status="invalid", valid=False, reason="vector_magnitude_is_invalid")
            magnitude_samples.append(copied)
        output.append((f"{path}.magnitude", tuple(magnitude_samples)))
    return tuple(output)


def _sign(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def _profile_statistics(
    kind: str,
    samples: Sequence[dict[str, object]],
    event_indices: Sequence[int],
) -> tuple[tuple[str, object], ...]:
    valid_pairs = [
        (index, sample.get("value"))
        for index, sample in zip(event_indices, samples)
        if sample.get("valid") is True and sample.get("value") is not None
    ]
    invalid_count = sum(sample.get("status") == "invalid" for sample in samples)
    not_evaluated_count = sum(
        sample.get("status") in {"not_evaluated", "unsupported"} or sample.get("value") is None
        for sample in samples
        if sample.get("status") != "invalid"
    )
    base: dict[str, object] = {
        "valid_count": len(valid_pairs),
        "not_evaluated_count": not_evaluated_count,
        "invalid_count": invalid_count,
        "first_valid_event_index": valid_pairs[0][0] if valid_pairs else None,
        "last_valid_event_index": valid_pairs[-1][0] if valid_pairs else None,
    }
    if kind == "numeric":
        numeric_pairs = [(index, float(value)) for index, value in valid_pairs if _is_finite_number(value)]
        values = [value for _, value in numeric_pairs]
        adjacent_deltas = [
            current_value - previous_value
            for (previous_index, previous_value), (current_index, current_value) in zip(numeric_pairs, numeric_pairs[1:])
            if current_index == previous_index + 1
        ]
        nonzero = [abs(delta) for delta in adjacent_deltas if delta != 0.0]
        signs = [_sign(value) for value in values]
        sign_change_count = sum(
            left != 0 and right != 0 and left != right for left, right in zip(signs, signs[1:])
        )
        base.update(
            first_valid_value=values[0] if values else None,
            last_valid_value=values[-1] if values else None,
            minimum=min(values) if values else None,
            maximum=max(values) if values else None,
            arithmetic_change_first_to_last=(values[-1] - values[0]) if values else None,
            absolute_change=abs(values[-1] - values[0]) if values else None,
            observed_min_nonzero_adjacent_delta=min(nonzero) if nonzero else None,
            observed_max_adjacent_delta=max((abs(delta) for delta in adjacent_deltas), default=None),
            positive_delta_count=sum(delta > 0.0 for delta in adjacent_deltas),
            negative_delta_count=sum(delta < 0.0 for delta in adjacent_deltas),
            zero_delta_count=sum(delta == 0.0 for delta in adjacent_deltas),
            sign_change_count=sign_change_count,
            monotonic_nondecreasing=(all(delta >= 0.0 for delta in adjacent_deltas) if adjacent_deltas else None),
            monotonic_nonincreasing=(all(delta <= 0.0 for delta in adjacent_deltas) if adjacent_deltas else None),
            constant=(all(delta == 0.0 for delta in adjacent_deltas) if adjacent_deltas else None),
        )
    elif kind == "boolean":
        boolean_pairs = [(index, value) for index, value in valid_pairs if isinstance(value, bool)]
        changes = [
            current_index
            for (previous_index, previous_value), (current_index, current_value) in zip(boolean_pairs, boolean_pairs[1:])
            if current_index == previous_index + 1 and current_value != previous_value
        ]
        base.update(
            true_count=sum(value is True for _, value in boolean_pairs),
            false_count=sum(value is False for _, value in boolean_pairs),
            first_transition_index=changes[0] if changes else None,
            last_transition_index=changes[-1] if changes else None,
            value_change_count=len(changes),
        )
    else:
        categorical_pairs = [(index, value) for index, value in valid_pairs]
        changes = sum(
            current != previous
            for (_, previous), (_, current) in zip(categorical_pairs, categorical_pairs[1:])
        )
        base.update(
            first_valid_value=categorical_pairs[0][1] if categorical_pairs else None,
            last_valid_value=categorical_pairs[-1][1] if categorical_pairs else None,
            value_change_count=changes,
        )
    return tuple((key, _freeze_json(value)) for key, value in sorted(base.items()))


def build_measured_signal_profiles(
    events: Sequence[Mapping[str, object]],
) -> tuple[MeasuredSignalProfile, ...]:
    per_event = tuple(_event_profile_evidence(event) for event in events)
    all_paths = sorted({path for event_values in per_event for path in event_values})
    event_indices = [int(event["event_index"]) for event in events]
    event_types = tuple(dict.fromkeys(str(event["event_type"]) for event in events))
    profiles: list[MeasuredSignalProfile] = []
    for path in all_paths:
        samples = tuple(event_values.get(path, _missing_evidence()) for event_values in per_event)
        for expanded_path, expanded_samples in _expand_profile_series(path, samples):
            kind = _profile_kind_from_samples(expanded_samples)
            units = next(
                (
                    str(sample.get("units"))
                    for sample in expanded_samples
                    if isinstance(sample.get("units"), str)
                    and sample.get("units") != "not_applicable"
                ),
                "not_applicable",
            )
            expected = tuple(
                event_type
                for event_type in event_types
                if any(
                    event.get("event_type") == event_type
                    and event_values.get(path, _missing_evidence()).get("status") != "not_evaluated"
                    for event, event_values in zip(events, per_event)
                )
            )
            limitation = (
                "This quantity is an observed deterministic trace resolution statistic, not an estimate of sensor noise, process noise, numerical uncertainty, or a recommended guard threshold."
                if kind == "numeric"
                else "This profile reports observed availability and changes only; it does not authorize a guard or phase transition."
            )
            profiles.append(
                MeasuredSignalProfile(
                    field_id=expanded_path,
                    source_event_path=path,
                    profile_kind=kind,
                    units=units,
                    evidence_level=_evidence_level_for_profile(path, expanded_samples),
                    expected_event_types=expected,
                    statistics=_profile_statistics(kind, expanded_samples, event_indices),
                    scientific_limitation=limitation,
                )
            )
    return tuple(sorted(profiles, key=lambda item: item.field_id))


def signal_profile_document(
    events: Sequence[Mapping[str, object]],
    profiles: Sequence[MeasuredSignalProfile],
) -> dict[str, object]:
    profile_documents = [_to_json_value(profile) for profile in profiles]
    numeric_count = sum(profile.profile_kind == "numeric" for profile in profiles)
    boolean_count = sum(profile.profile_kind == "boolean" for profile in profiles)
    categorical_count = sum(profile.profile_kind == "categorical" for profile in profiles)
    valid_profile_count = sum(dict(profile.statistics).get("valid_count", 0) > 0 for profile in profiles)
    not_evaluated_profile_count = sum(
        dict(profile.statistics).get("valid_count", 0) == 0
        and dict(profile.statistics).get("invalid_count", 0) == 0
        for profile in profiles
    )
    invalid_profile_count = sum(
        dict(profile.statistics).get("invalid_count", 0) > 0 for profile in profiles
    )
    return with_document_hash(
        {
            "schema_version": SIGNAL_PROFILE_SCHEMA_VERSION,
            "analysis_id": ANALYSIS_ID,
            "source_event_count": len(events),
            "profile_count": len(profiles),
            "profile_counts": {
                "numeric": numeric_count,
                "boolean": boolean_count,
                "categorical": categorical_count,
                "with_valid_evidence": valid_profile_count,
                "all_not_evaluated": not_evaluated_profile_count,
                "with_invalid_evidence": invalid_profile_count,
            },
            "profiles": profile_documents,
            "resolution_statistic_limitation": (
                "This quantity is an observed deterministic trace resolution statistic, not an estimate of sensor noise, process noise, numerical uncertainty, or a recommended guard threshold."
            ),
        }
    )


def guard_inventory_document(
    definitions: Sequence[GuardAtomDefinition],
) -> dict[str, object]:
    categories: dict[str, int] = {}
    for definition in definitions:
        categories[definition.category] = categories.get(definition.category, 0) + 1
    parameters = unresolved_parameter_inventory()
    return with_document_hash(
        {
            "schema_version": GUARD_INVENTORY_SCHEMA_VERSION,
            "analysis_id": ANALYSIS_ID,
            "artifact_classification": "offline_guard_evidence_analysis",
            "guard_atom_count": len(definitions),
            "exact_inherited_guard_atom_count": len(EXACT_INHERITED_GUARD_ATOM_IDS),
            "threshold_free_directional_guard_atom_count": len(
                THRESHOLD_FREE_DIRECTIONAL_GUARD_ATOM_IDS
            ),
            "category_counts": dict(sorted(categories.items())),
            "exact_inherited_guard_atom_ids": list(EXACT_INHERITED_GUARD_ATOM_IDS),
            "threshold_free_directional_guard_atom_ids": list(
                THRESHOLD_FREE_DIRECTIONAL_GUARD_ATOM_IDS
            ),
            "guard_atoms": [_to_json_value(item) for item in definitions],
            "unresolved_parameter_count": len(parameters),
            "unresolved_parameters": list(parameters),
            "all_candidate_guard_authorizations": "not_authorized",
            "phase_transition_policy": "not_frozen",
        }
    )


def _guard_evaluation_document(
    event: Mapping[str, object],
    evaluations: Sequence[GuardAtomEvaluation],
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": GUARD_TRACE_SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "artifact_classification": "offline_guard_evidence_analysis",
        "event_index": event.get("event_index"),
        "event_type": event.get("event_type"),
        "recovery_step": event.get("recovery_step"),
        "total_transition_count": event.get("total_transition_count"),
        "source_event_scientific_hash": event.get("canonical_event_sha256"),
        "source_trace_aggregate_hash": SOURCE_TRACE_AGGREGATE_HASH,
        "guard_atoms": [_to_json_value(item) for item in evaluations],
        "action_generated": False,
        "phase_selected": False,
        "stop_condition_selected": False,
        "policy_authorization": "not_authorized",
    }
    document["canonical_guard_evaluation_event_hash"] = canonical_sha256(document)
    return document


def guard_evaluation_event_hash_recomputes(document: Mapping[str, object]) -> bool:
    payload = dict(document)
    supplied = payload.pop("canonical_guard_evaluation_event_hash", None)
    return isinstance(supplied, str) and supplied == canonical_sha256(payload)


def aggregate_guard_evaluation_hashes(events: Sequence[Mapping[str, object]]) -> str:
    return canonical_sha256(
        {
            "ordered_guard_evaluation_event_hashes": [
                event.get("canonical_guard_evaluation_event_hash") for event in events
            ]
        }
    )


def build_guard_evaluation_trace(
    events: Sequence[Mapping[str, object]],
    definitions: Sequence[GuardAtomDefinition],
) -> tuple[dict[str, object], ...]:
    output: list[dict[str, object]] = []
    previous_state_event: Mapping[str, object] | None = None
    for event in events:
        event_type = event.get("event_type")
        previous = previous_state_event if event_type == "transition" else None
        evaluations = evaluate_guard_atoms_for_event(event, previous, definitions)
        output.append(_guard_evaluation_document(event, evaluations))
        if event_type in {"initial_snapshot", "transition"}:
            previous_state_event = event
    return tuple(output)


def guard_trace_jsonl_bytes(events: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(event) + b"\n" for event in events)


def unresolved_parameter_inventory() -> tuple[dict[str, object], ...]:
    common = {
        "current_status": "unresolved",
        "available_stage0c_evidence": (
            "eight consecutive measured state transitions permit structural window evaluation only"
        ),
        "authorization_status": "not_authorized",
    }
    details = {
        "NO_PROGRESS_WINDOW_LENGTH": (
            "RecoveryNoProgressContract.observation_window",
            ["radius gap", "radial velocity", "tangential error", "headroom", "energy proxy"],
            "one eight-transition trace cannot estimate general temporal variability or false-trigger behavior",
            "Stage 1B multi-trace guard calibration dataset",
        ),
        "NO_PROGRESS_MIN_RADIUS_GAP_IMPROVEMENT": (
            "RecoveryNoProgressContract.minimum_meaningful_improvement",
            ["absolute target-radius error"],
            "observed adjacent changes are not a noise floor or transferable threshold",
            "multi-trace radius-progress calibration with repeated cases",
        ),
        "NO_PROGRESS_MIN_RADIAL_COMPONENT_IMPROVEMENT": (
            "RecoveryNoProgressContract.minimum_meaningful_improvement",
            ["radial velocity", "radial velocity ratio", "radial direction product"],
            "one response mode cannot establish adequate radial commitment magnitude",
            "multi-trace radial-commitment observational set",
        ),
        "NO_PROGRESS_MIN_TANGENTIAL_COMPONENT_IMPROVEMENT": (
            "RecoveryNoProgressContract.minimum_meaningful_improvement",
            ["absolute tangential velocity error", "tangential velocity error ratio"],
            "the trace crosses zero and then worsens in magnitude, but provides no general threshold distribution",
            "multi-trace tangential-alignment observational set",
        ),
        "NO_PROGRESS_MIN_HEADROOM_IMPROVEMENT": (
            "RecoveryNoProgressContract.minimum_meaningful_improvement",
            ["realized overspeed headroom", "predicted overspeed headroom"],
            "one monotone trace cannot establish a meaningful minimum improvement",
            "hazard-arrest/stabilization observational trace set",
        ),
        "NO_PROGRESS_REQUIRED_COMPONENT_COUNT": (
            "RecoveryNoProgressContract.monitored_progress_signal",
            ["separate component directions"],
            "component signals must not be silently combined and no validated aggregation rule exists",
            "predeclared multi-component guard study",
        ),
        "NO_PROGRESS_CONSECUTIVE_WINDOWS": (
            "RecoveryHysteresisContract.consecutive_evidence_count",
            ["all window endpoints for lengths one through eight"],
            "one short sequence cannot calibrate repeated-window evidence",
            "multi-trace repeated-window calibration",
        ),
        "NO_PROGRESS_MIN_PHASE_DWELL": (
            "RecoveryNoProgressContract.phase_specific_dwell_limit",
            ["runtime counters only"],
            "no staged phase runtime or measured phase dwell exists",
            "bounded phase-runtime instrumentation validation",
        ),
        "NO_PROGRESS_COOLDOWN": (
            "RecoveryHysteresisContract.transition_cooldown",
            ["event ordering only"],
            "no staged phase transition has been executed or observed",
            "bounded phase-runtime anti-chatter validation",
        ),
    }
    output = []
    for parameter_id in UNRESOLVED_PARAMETER_IDS:
        contract, signals, missing, next_experiment = details[parameter_id]
        output.append(
            {
                "parameter_id": parameter_id,
                "required_by_architecture_contract": contract,
                **common,
                "observable_supporting_signals": signals,
                "missing_evidence": missing,
                "why_eight_transitions_are_structural_only": (
                    "all integer windows can be enumerated, but this single path does not support parameter selection"
                ),
                "next_experiment_needed": next_experiment,
                "selected_value": None,
                "observed_ranges_are_recommendations": False,
            }
        )
    return tuple(output)


def _state_sample_events(events: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    samples = [event for event in events if event.get("event_type") in {"initial_snapshot", "transition"}]
    if len(samples) != SOURCE_TRANSITION_COUNT + 1:
        raise GuardEvidenceError("state sample sequence must contain boundary plus eight transitions")
    return tuple(samples)


def _state_numeric(event: Mapping[str, object], field_id: str) -> tuple[str, float | None]:
    return _numeric_evidence(_current_fields(event).get(field_id))


def _state_boolean(event: Mapping[str, object], field_id: str) -> tuple[str, bool | None]:
    return _boolean_evidence(_current_fields(event).get(field_id))


def _window_change(
    start: Mapping[str, object],
    end: Mapping[str, object],
    field_id: str,
    *,
    absolute: bool = False,
) -> tuple[str, float | None]:
    start_status, start_value = _state_numeric(start, field_id)
    end_status, end_value = _state_numeric(end, field_id)
    if "invalid" in {start_status, end_status}:
        return "invalid", None
    if start_status != "available" or end_status != "available" or start_value is None or end_value is None:
        return "not_evaluated", None
    if absolute:
        return "available", abs(end_value) - abs(start_value)
    return "available", end_value - start_value


def _component_label(change: float | None, *, higher_is_improvement: bool) -> str:
    if change is None:
        return "not_evaluated"
    if change == 0.0:
        return "component_unchanged"
    improved = change > 0.0 if higher_is_improvement else change < 0.0
    return "component_improved" if improved else "component_worsened"


def build_windowed_progress_records(
    events: Sequence[Mapping[str, object]],
) -> tuple[WindowedProgressRecord, ...]:
    samples = _state_sample_events(events)
    output: list[WindowedProgressRecord] = []
    change_specs = (
        ("absolute_radius_gap_change", "absolute_target_radius_error", True, False, True),
        ("signed_radius_gap_change", "signed_target_radius_error", False, True, False),
        ("radial_velocity_change", "radial_velocity", False, False, False),
        ("absolute_tangential_error_change", "tangential_velocity_error", True, False, True),
        ("speed_ratio_change", "realized_speed_ratio", False, False, False),
        ("overspeed_headroom_change", "overspeed_headroom", False, True, True),
        ("absolute_energy_proxy_error_change", "specific_energy_error", True, False, True),
        ("recoverability_radius_component_change", "radius_error_ratio", True, False, True),
        ("recoverability_radial_component_change", "radial_velocity_ratio", True, False, True),
        ("recoverability_tangential_component_change", "tangential_velocity_error_ratio", True, False, True),
    )
    for length in range(1, SOURCE_TRANSITION_COUNT + 1):
        for start_index in range(0, len(samples) - length):
            end_index = start_index + length
            start = samples[start_index]
            end = samples[end_index]
            changes: dict[str, object] = {}
            labels: dict[str, str] = {}
            invalid_count = 0
            unavailable_count = 0
            valid_count = 0
            for output_id, field_id, absolute, higher_is_improvement, direction_defined in change_specs:
                status, change = _window_change(start, end, field_id, absolute=absolute)
                changes[output_id] = change
                changes[f"{output_id}_status"] = status
                if status == "available":
                    valid_count += 1
                    labels[output_id] = (
                        _component_label(change, higher_is_improvement=higher_is_improvement)
                        if direction_defined
                        else "not_evaluated"
                    )
                elif status == "invalid":
                    invalid_count += 1
                    labels[output_id] = "invalid"
                else:
                    unavailable_count += 1
                    labels[output_id] = "not_evaluated"

            directional_count = 0
            crossing_count = 0
            for sample in samples[start_index + 1 : end_index + 1]:
                gap_status, gap = _state_numeric(sample, "signed_target_radius_error")
                radial_status, radial = _state_numeric(sample, "radial_velocity")
                if gap_status == "available" and radial_status == "available" and gap is not None and radial is not None:
                    directional_count += gap * radial < 0.0
                crossing_status, crossing = _state_boolean(sample, "target_radius_crossing")
                if crossing_status == "available":
                    crossing_count += crossing is True

            available_labels = [label for label in labels.values() if label.startswith("component_")]
            if available_labels and all(label == "component_improved" for label in available_labels):
                descriptive = "component_improved"
            elif available_labels and all(label == "component_unchanged" for label in available_labels):
                descriptive = "component_unchanged"
            elif available_labels and all(label == "component_worsened" for label in available_labels):
                descriptive = "component_worsened"
            else:
                descriptive = "mixed_component_direction"
            output.append(
                WindowedProgressRecord(
                    window_length=length,
                    start_recovery_step=int(start["recovery_step"]),
                    end_recovery_step=int(end["recovery_step"]),
                    start_event_index=int(start["event_index"]),
                    end_event_index=int(end["event_index"]),
                    component_changes=tuple((key, _freeze_json(value)) for key, value in sorted(changes.items())),
                    component_labels=tuple(sorted(labels.items())),
                    directional_radial_commitment_count=directional_count,
                    crossing_count=crossing_count,
                    valid_sample_count=valid_count,
                    unavailable_sample_count=unavailable_count,
                    invalid_sample_count=invalid_count,
                    descriptive_direction=descriptive,
                )
            )
    return tuple(output)


def no_progress_window_document(
    records: Sequence[WindowedProgressRecord],
) -> dict[str, object]:
    lengths: list[dict[str, object]] = []
    for length in range(1, SOURCE_TRANSITION_COUNT + 1):
        selected = [record for record in records if record.window_length == length]
        component_ids = sorted({key for record in selected for key, _ in record.component_labels})
        component_summary = {}
        for component_id in component_ids:
            labels = [dict(record.component_labels)[component_id] for record in selected]
            component_summary[component_id] = {
                "all_improving_count": sum(label == "component_improved" for label in labels),
                "no_change_count": sum(label == "component_unchanged" for label in labels),
                "worsening_count": sum(label == "component_worsened" for label in labels),
                "unavailable_or_invalid_count": sum(label in {"not_evaluated", "invalid"} for label in labels),
            }
        lengths.append(
            {
                "window_length": length,
                "number_of_possible_windows": len(selected),
                "earliest_endpoint": min(record.end_recovery_step for record in selected),
                "latest_endpoint": max(record.end_recovery_step for record in selected),
                "mixed_direction_count": sum(record.descriptive_direction == "mixed_component_direction" for record in selected),
                "component_summaries": component_summary,
                "windows": [_to_json_value(record) for record in selected],
            }
        )
    return with_document_hash(
        {
            "schema_version": NO_PROGRESS_SCHEMA_VERSION,
            "analysis_id": ANALYSIS_ID,
            "window_length_range": [1, SOURCE_TRANSITION_COUNT],
            "preferred_window_length": None,
            "minimum_improvement_threshold": None,
            "combined_progress_score": None,
            "stall_progress_regression_policy_classification": None,
            "window_lengths": lengths,
            "unresolved_parameters": list(unresolved_parameter_inventory()),
            "required_statement": "No window length or minimum-improvement threshold is selected or authorized by this analysis.",
        }
    )


def _first_stage0a_completeness_entries(document: Mapping[str, object]) -> dict[str, dict[str, object]]:
    fields_value = document.get("fields")
    if not isinstance(fields_value, list):
        raise GuardEvidenceError("Stage 0C field completeness entries are missing")
    output: dict[str, dict[str, object]] = {}
    for item in fields_value:
        if isinstance(item, dict) and isinstance(item.get("field_id"), str):
            output.setdefault(str(item["field_id"]), item)
    return output


def _catalog_map(document: Mapping[str, object]) -> dict[str, dict[str, object]]:
    fields_value = document.get("fields")
    if not isinstance(fields_value, list):
        raise GuardEvidenceError("Stage 0A field catalog entries are missing")
    return {
        str(item["field_id"]): item
        for item in fields_value
        if isinstance(item, dict) and isinstance(item.get("field_id"), str)
    }


def _phase_guard_candidates() -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        "hazard_arrest": {
            "entry": ("predicted_overspeed", "realized_overspeed"),
            "stay": ("predicted_overspeed", "realized_overspeed", "state_evidence_valid"),
            "exit": ("predicted_overspeed_clear", "realized_overspeed_clear"),
        },
        "stabilization_assessment": {
            "entry": ("realized_overspeed_clear", "state_evidence_valid"),
            "stay": ("instrumentation_evaluation_valid", "recovery_evaluation_valid"),
            "exit": ("radial_velocity_toward_target",),
        },
        "radial_recommitment": {
            "entry": ("radial_velocity_away_from_target", "absolute_radius_gap_worsening"),
            "stay": ("radial_velocity_toward_target", "absolute_radius_gap_improving"),
            "exit": ("recoverability_radial_component_pass",),
        },
        "tangential_alignment": {
            "entry": ("recoverability_tangential_velocity_component_pass",),
            "stay": ("absolute_tangential_error_improving", "overspeed_headroom_improving"),
            "exit": ("recoverability_tangential_velocity_component_pass",),
        },
        "crossing_preparation": {
            "entry": ("recoverability_radius_component_improving", "radial_velocity_toward_target"),
            "stay": ("no_eligible_crossing", "realized_overspeed_clear"),
            "exit": ("eligible_target_radius_crossing",),
        },
        "recoverability_verification": {
            "entry": ("eligible_target_radius_crossing",),
            "stay": ("phase34_compatible_recoverability_pass",),
            "exit": ("phase34_compatible_recoverability_pass",),
        },
        "nominal_handoff": {
            "entry": ("phase34_compatible_recoverability_pass", "handoff_readiness_available"),
            "stay": ("state_evidence_valid",),
            "exit": (),
        },
        "retreat": {
            "entry": ("no_progress_policy_evaluable", "correction_authority_available"),
            "stay": ("state_evidence_valid",),
            "exit": (),
        },
        "explicit_abort": {
            "entry": ("explicit_abort_requested",),
            "stay": (),
            "exit": (),
        },
    }


def _phase_unresolved_parameters(phase_id: str) -> tuple[str, ...]:
    mapping = {
        "hazard_arrest": ("NO_PROGRESS_CONSECUTIVE_WINDOWS", "NO_PROGRESS_MIN_PHASE_DWELL", "NO_PROGRESS_COOLDOWN"),
        "stabilization_assessment": ("NO_PROGRESS_WINDOW_LENGTH", "NO_PROGRESS_CONSECUTIVE_WINDOWS", "NO_PROGRESS_MIN_PHASE_DWELL"),
        "radial_recommitment": ("NO_PROGRESS_WINDOW_LENGTH", "NO_PROGRESS_MIN_RADIUS_GAP_IMPROVEMENT", "NO_PROGRESS_MIN_RADIAL_COMPONENT_IMPROVEMENT", "NO_PROGRESS_MIN_PHASE_DWELL"),
        "tangential_alignment": ("NO_PROGRESS_WINDOW_LENGTH", "NO_PROGRESS_MIN_TANGENTIAL_COMPONENT_IMPROVEMENT", "NO_PROGRESS_MIN_PHASE_DWELL"),
        "crossing_preparation": ("NO_PROGRESS_WINDOW_LENGTH", "NO_PROGRESS_REQUIRED_COMPONENT_COUNT", "NO_PROGRESS_CONSECUTIVE_WINDOWS"),
        "recoverability_verification": ("NO_PROGRESS_CONSECUTIVE_WINDOWS", "NO_PROGRESS_MIN_PHASE_DWELL"),
        "nominal_handoff": ("NO_PROGRESS_CONSECUTIVE_WINDOWS",),
        "retreat": ("NO_PROGRESS_WINDOW_LENGTH", "NO_PROGRESS_REQUIRED_COMPONENT_COUNT", "NO_PROGRESS_COOLDOWN"),
        "explicit_abort": (),
    }
    return mapping[phase_id]


def build_phase_observability_entries(
    architecture_manifest: Mapping[str, object],
    field_catalog_document: Mapping[str, object],
    field_completeness_document: Mapping[str, object],
    definitions: Sequence[GuardAtomDefinition],
    guard_trace: Sequence[Mapping[str, object]],
) -> tuple[PhaseObservabilityEntry, ...]:
    phase_contracts_value = architecture_manifest.get("phase_contracts")
    if not isinstance(phase_contracts_value, list):
        raise GuardEvidenceError("architecture phase contracts are missing")
    phase_contracts = {
        str(item["phase_id"]): item
        for item in phase_contracts_value
        if isinstance(item, dict) and isinstance(item.get("phase_id"), str)
    }
    if tuple(phase_contracts) != PHASE_IDS:
        raise GuardEvidenceError("architecture phase IDs or ordering drifted")
    catalog = _catalog_map(field_catalog_document)
    completeness = _first_stage0a_completeness_entries(field_completeness_document)
    candidates = _phase_guard_candidates()

    evaluated_by_atom: dict[str, bool] = {}
    for trace_event in guard_trace:
        atoms = trace_event.get("guard_atoms")
        if not isinstance(atoms, list):
            continue
        for atom in atoms:
            if isinstance(atom, dict) and isinstance(atom.get("guard_atom_id"), str):
                evaluated_by_atom[str(atom["guard_atom_id"])] = evaluated_by_atom.get(str(atom["guard_atom_id"]), False) or atom.get("status") in {"true", "false"}

    observability_status = {
        "hazard_arrest": "partially_observable",
        "stabilization_assessment": "future_evaluator_required",
        "radial_recommitment": "partially_observable",
        "tangential_alignment": "partially_observable",
        "crossing_preparation": "partially_observable",
        "recoverability_verification": "partially_observable",
        "nominal_handoff": "future_evaluator_required",
        "retreat": "future_evaluator_required",
        "explicit_abort": "partially_observable",
    }
    interpretations = {
        "hazard_arrest": "Hazard ratios and monitor evidence are observable, but clear dwell, hysteresis, and an action law are unresolved.",
        "stabilization_assessment": "Validity and kinematics are visible, while instability, unsafe-state, and stabilization criteria require future evaluators.",
        "radial_recommitment": "Radius gap and radial direction are observable; adequate commitment magnitude, authority, dwell, and action law are unresolved.",
        "tangential_alignment": "Tangential component value and trend are observable; allowed radial degradation, guards, and action law are unresolved.",
        "crossing_preparation": "Measured gap and crossing evidence are visible; reliable future crossing prediction and correction authority are unavailable.",
        "recoverability_verification": "Existing component values, pass atoms, and crossing evidence are observable; verification dwell and handoff compatibility are unresolved.",
        "nominal_handoff": "Existing recoverability evidence is insufficient because handoff readiness and nominal-controller acceptance are not implemented.",
        "retreat": "Some adverse and progress inputs are observable, but retreat authority, target, action, and success evaluator are unavailable.",
        "explicit_abort": "Externally supplied abort evidence can be recorded, but Stage 1A defines no autonomous abort policy.",
    }

    output: list[PhaseObservabilityEntry] = []
    for phase_id in PHASE_IDS:
        contract = phase_contracts[phase_id]
        required_value = contract.get("required_signal_ids")
        required = tuple(str(item) for item in required_value) if isinstance(required_value, list) else ()
        support = tuple(
            (
                signal_id,
                str(catalog.get(signal_id, {}).get("support_classification", "not_yet_supported")),
            )
            for signal_id in required
        )
        measured = tuple(
            (
                signal_id,
                bool(completeness.get(signal_id, {}).get("valid_event_count", 0)),
            )
            for signal_id in required
        )
        pure = tuple(
            signal_id
            for signal_id in required
            if catalog.get(signal_id, {}).get("support_classification")
            in {"pure_derivation_supported", "requires_previous_state", "requires_predicted_state"}
        )
        previous = tuple(
            signal_id
            for signal_id in required
            if catalog.get(signal_id, {}).get("support_classification") == "requires_previous_state"
        )
        predicted = tuple(
            signal_id
            for signal_id in required
            if catalog.get(signal_id, {}).get("support_classification") == "requires_predicted_state"
        )
        runtime_phase = tuple(
            signal_id
            for signal_id in required
            if catalog.get(signal_id, {}).get("support_classification") == "requires_runtime_phase_integration"
        )
        future = tuple(
            signal_id
            for signal_id in required
            if catalog.get(signal_id, {}).get("support_classification") == "requires_future_evaluator"
        )
        unsupported = tuple(
            signal_id
            for signal_id in required
            if catalog.get(signal_id, {}).get("support_classification") == "not_yet_supported"
        )
        phase_candidates = candidates[phase_id]
        all_candidates = tuple(
            dict.fromkeys(
                phase_candidates["entry"] + phase_candidates["stay"] + phase_candidates["exit"]
            )
        )
        available_atoms = tuple(atom_id for atom_id in all_candidates if evaluated_by_atom.get(atom_id, False))
        unavailable_atoms = tuple(atom_id for atom_id in all_candidates if atom_id not in available_atoms)
        output.append(
            PhaseObservabilityEntry(
                phase_id=phase_id,
                possible_entry_evidence=phase_candidates["entry"],
                possible_stay_evidence=phase_candidates["stay"],
                possible_exit_evidence=phase_candidates["exit"],
                required_architecture_signals=required,
                current_stage0a_schema_support=support,
                stage0c_measured_availability=measured,
                pure_derivation_availability=pure,
                previous_state_dependencies=previous,
                predicted_state_dependencies=predicted,
                runtime_phase_dependencies=runtime_phase,
                future_evaluator_dependencies=future,
                unsupported_dependencies=unsupported,
                available_guard_atoms=available_atoms,
                unavailable_guard_atoms=unavailable_atoms,
                unresolved_numerical_parameters=_phase_unresolved_parameters(phase_id),
                action_law_complete=False,
                current_observability_status=observability_status[phase_id],
                policy_authorization="not_authorized",
                strongest_permitted_interpretation=interpretations[phase_id],
            )
        )
    return tuple(output)


def phase_observability_document(
    entries: Sequence[PhaseObservabilityEntry],
) -> dict[str, object]:
    statuses = [entry.current_observability_status for entry in entries]
    unsupported_phase_count = sum(bool(entry.unsupported_dependencies) for entry in entries)
    return with_document_hash(
        {
            "schema_version": PHASE_MATRIX_SCHEMA_VERSION,
            "analysis_id": ANALYSIS_ID,
            "phase_count": len(entries),
            "totals": {
                "fully_observable_phases": statuses.count("fully_observable_from_current_schema"),
                "partially_observable_phases": statuses.count("partially_observable"),
                "not_observable_phases": statuses.count("not_observable"),
                "future_evaluator_required_phases": statuses.count("future_evaluator_required"),
                "phases_with_unsupported_dependencies": unsupported_phase_count,
                "phases_with_implemented_action_laws": sum(entry.action_law_complete for entry in entries),
                "phases_with_authorized_executable_guards": sum(entry.policy_authorization != "not_authorized" for entry in entries),
            },
            "phases": [_to_json_value(entry) for entry in entries],
            "guard_atom_true_does_not_authorize_transition": True,
            "existing_branch_actions_reused_as_staged_actions": False,
        }
    )


def evidence_traceability_document(
    profiles: Sequence[MeasuredSignalProfile],
    definitions: Sequence[GuardAtomDefinition],
    guard_trace: Sequence[Mapping[str, object]],
    source_events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    source_hash_by_index = {
        int(event["event_index"]): str(event["canonical_event_sha256"])
        for event in source_events
    }
    items: list[dict[str, object]] = []
    for profile in profiles:
        statistics = dict(profile.statistics)
        first = statistics.get("first_valid_event_index")
        last = statistics.get("last_valid_event_index")
        indices = (
            list(range(int(first), int(last) + 1))
            if isinstance(first, int) and isinstance(last, int)
            else []
        )
        items.append(
            {
                "analysis_item_id": f"signal_profile.{profile.field_id}",
                "item_kind": "signal_profile",
                "source_artifact": (
                    "analysis/staged_recovery_instrumentation_validation_v0/staged_recovery_trace.jsonl"
                ),
                "source_json_path": profile.source_event_path,
                "source_event_types": list(profile.expected_event_types),
                "source_event_indices": indices,
                "source_event_hashes": [source_hash_by_index[index] for index in indices],
                "formula_or_comparator": "deterministic per-field availability and adjacent-delta profile",
                "evidence_level": profile.evidence_level,
                "inherited_threshold_source": None,
                "unresolved_parameter_dependency": [],
                "tests": ["test_signal_profiles", "test_profile_determinism"],
                "limitations": profile.scientific_limitation,
                "source_classification": "measured Stage 0C evidence" if profile.evidence_level == "measured" else "derived Stage 0C evidence",
            }
        )

    trace_atoms_by_id: dict[str, list[tuple[int, str, str]]] = {}
    for trace_event in guard_trace:
        event_index = int(trace_event["event_index"])
        source_hash = str(trace_event["source_event_scientific_hash"])
        atoms = trace_event.get("guard_atoms")
        if not isinstance(atoms, list):
            continue
        for atom in atoms:
            if isinstance(atom, dict) and isinstance(atom.get("guard_atom_id"), str):
                trace_atoms_by_id.setdefault(str(atom["guard_atom_id"]), []).append(
                    (event_index, source_hash, str(atom.get("status")))
                )
    for definition in definitions:
        evidence = trace_atoms_by_id.get(definition.guard_atom_id, [])
        items.append(
            {
                "analysis_item_id": f"guard_atom.{definition.guard_atom_id}",
                "item_kind": "guard_atom",
                "source_artifact": definition.source_path,
                "source_json_path": list(definition.required_fields),
                "source_event_types": ["initial_snapshot", "transition", "terminal"],
                "source_event_indices": [index for index, _, _ in evidence],
                "source_event_hashes": [digest for _, digest, _ in evidence],
                "formula_or_comparator": definition.comparator,
                "evidence_level": definition.evidence_level.value,
                "inherited_threshold_source": definition.threshold_source,
                "unresolved_parameter_dependency": list(definition.unresolved_parameter_ids),
                "tests": [f"test_guard_{definition.guard_atom_id}", "test_guard_trace_hash_determinism"],
                "limitations": definition.non_meaning,
                "source_classification": (
                    "architecture requirement"
                    if definition.unresolved_parameter_ids
                    else "derived Stage 0C evidence"
                ),
                "observed_statuses": sorted({status for _, _, status in evidence}),
            }
        )
    return with_document_hash(
        {
            "schema_version": TRACEABILITY_SCHEMA_VERSION,
            "analysis_id": ANALYSIS_ID,
            "item_count": len(items),
            "items": sorted(items, key=lambda item: str(item["analysis_item_id"])),
            "historical_context": {
                "source": "analysis/recovery_branch_mechanism_diagnosis_v0/summary.md",
                "classification": "historical summary evidence",
                "use": "qualitative motivation only; no unlogged historical state was reconstructed",
            },
        }
    )


def _guard_status_counts(guard_trace: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {status.value: 0 for status in GuardEvidenceStatus}
    for event in guard_trace:
        atoms = event.get("guard_atoms")
        if not isinstance(atoms, list):
            continue
        for atom in atoms:
            if isinstance(atom, dict) and atom.get("status") in counts:
                counts[str(atom["status"])] += 1
    return counts


def _guard_true_count(guard_trace: Sequence[Mapping[str, object]], atom_id: str) -> int:
    count = 0
    for event in guard_trace:
        atoms = event.get("guard_atoms")
        if not isinstance(atoms, list):
            continue
        count += any(
            isinstance(atom, dict)
            and atom.get("guard_atom_id") == atom_id
            and atom.get("status") == "true"
            for atom in atoms
        )
    return count


def _analysis_manifest_document(
    *,
    implementation_commit: str,
    source_bundle: SourceBundle,
    profiles: Sequence[MeasuredSignalProfile],
    definitions: Sequence[GuardAtomDefinition],
    guard_trace: Sequence[Mapping[str, object]],
    phase_document: Mapping[str, object],
    signal_document: Mapping[str, object],
    inventory_document: Mapping[str, object],
    no_progress_document: Mapping[str, object],
    traceability_document: Mapping[str, object],
) -> dict[str, object]:
    validation = source_bundle.document("validation_manifest")
    trace_manifest = source_bundle.document("trace_manifest")
    guard_aggregate = aggregate_guard_evaluation_hashes(guard_trace)
    return with_document_hash(
        {
            "analysis_id": ANALYSIS_ID,
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "completed_date": COMPLETED_DATE,
            "analyzed_date": ANALYZED_DATE,
            "implementation_commit": implementation_commit,
            "stage0c_result_commit": SOURCE_STAGE0C_RESULT_COMMIT,
            "source_validation_id": validation.get("validation_id"),
            "source_trace_manifest_canonical_hash": trace_manifest.get("canonical_payload_hash"),
            "source_trace_aggregate_hash": trace_manifest.get("aggregate_trace_hash"),
            "source_event_count": trace_manifest.get("event_count"),
            "source_transition_count": SOURCE_TRANSITION_COUNT,
            "source_branch": trace_manifest.get("branch_id"),
            "source_seed": trace_manifest.get("seed"),
            "source_nominal_prefix": validation.get("nominal_prefix_transition_count"),
            "source_stage0a": {
                "commit": SOURCE_INSTRUMENTATION_COMMIT,
                "canonical_hash": SOURCE_INSTRUMENTATION_CANONICAL_HASH,
            },
            "source_stage0b": {
                "commit": SOURCE_LOGGER_COMMIT,
                "canonical_hash": SOURCE_LOGGER_CANONICAL_HASH,
            },
            "source_architecture": {
                "commit": SOURCE_ARCHITECTURE_COMMIT,
                "canonical_hash": SOURCE_ARCHITECTURE_CANONICAL_HASH,
            },
            "exact_inherited_thresholds": {
                "overspeed": {"threshold": OVERSPEED_THRESHOLD, "comparator": ">"},
                "phase34_radius_error_ratio": {"threshold": RADIUS_ERROR_RATIO_MAX, "comparator": "inclusive_absolute_<="},
                "phase34_radial_velocity_ratio": {"threshold": RADIAL_VELOCITY_RATIO_MAX, "comparator": "inclusive_absolute_<="},
                "phase34_tangential_velocity_error_ratio": {"threshold": TANGENTIAL_VELOCITY_ERROR_RATIO_MAX, "comparator": "inclusive_absolute_<="},
            },
            "threshold_free_guard_atom_ids": list(THRESHOLD_FREE_DIRECTIONAL_GUARD_ATOM_IDS),
            "exact_inherited_guard_atom_ids": list(EXACT_INHERITED_GUARD_ATOM_IDS),
            "guard_atom_count": len(definitions),
            "guard_evaluation_event_count": len(guard_trace),
            "guard_evaluation_trace_aggregate_hash": guard_aggregate,
            "unresolved_parameter_ids": list(UNRESOLVED_PARAMETER_IDS),
            "phase_observability_summary": phase_document.get("totals"),
            "signal_profile_summary": signal_document.get("profile_counts"),
            "guard_status_counts": _guard_status_counts(guard_trace),
            "no_progress_window_range": [1, SOURCE_TRANSITION_COUNT],
            "generated_artifact_filenames": list(PUBLISHED_FILENAMES),
            "artifact_canonical_hashes": {
                SIGNAL_PROFILE_FILENAME: signal_document.get("canonical_payload_hash"),
                GUARD_INVENTORY_FILENAME: inventory_document.get("canonical_payload_hash"),
                PHASE_MATRIX_FILENAME: phase_document.get("canonical_payload_hash"),
                NO_PROGRESS_FILENAME: no_progress_document.get("canonical_payload_hash"),
                TRACEABILITY_FILENAME: traceability_document.get("canonical_payload_hash"),
            },
            "analysis_classification": "offline_guard_evidence_analysis",
            "new_runtime_execution": False,
            "new_measured_trace": False,
            "phase_guard_policy": "not_frozen",
            "phase_actions": "not_implemented",
            "staged_recovery_execution": "not_authorized",
            "scientific_claim_restrictions": list(CLAIM_RESTRICTIONS),
        }
    )


def _summary_bytes(
    manifest: Mapping[str, object],
    signal_document: Mapping[str, object],
    guard_trace: Sequence[Mapping[str, object]],
    phase_document: Mapping[str, object],
    no_progress_document: Mapping[str, object],
) -> bytes:
    signal_counts = signal_document["profile_counts"]
    phase_totals = phase_document["totals"]
    window_counts = {
        int(item["window_length"]): int(item["number_of_possible_windows"])
        for item in no_progress_document["window_lengths"]
    }
    lines = [
        "# Staged Recovery Guard Evidence Analysis v0",
        "",
        "## Status",
        "",
        "Stage 1A offline guard-evidence analysis completed; staged recovery execution remains unauthorized.",
        "",
        "Analyzed: 2026-07-31",
        "",
        "## Source Evidence",
        "",
        f"- Stage 0C result commit: `{SOURCE_STAGE0C_RESULT_COMMIT}`",
        f"- Validation ID: `{SOURCE_STAGE0C_VALIDATION_ID}`",
        f"- Trace-manifest canonical hash: `{SOURCE_TRACE_MANIFEST_CANONICAL_HASH}`",
        f"- Trace aggregate hash: `{SOURCE_TRACE_AGGREGATE_HASH}`",
        f"- Source events/transitions: `{SOURCE_EVENT_COUNT}` / `{SOURCE_TRANSITION_COUNT}`",
        f"- Branch/seed/prefix: `{SOURCE_BRANCH_ID}` / `{SOURCE_SEED}` / `{SOURCE_NOMINAL_PREFIX}`",
        "",
        "All source event hashes, ordering, equivalence evidence, field-completeness evidence, and aggregate trace identity validated before analysis.",
        "",
        "## Signal Availability",
        "",
        f"The deterministic profile contains `{signal_counts['numeric']}` numeric, `{signal_counts['boolean']}` boolean, and `{signal_counts['categorical']}` categorical/component profiles. `{signal_counts['with_valid_evidence']}` profiles contain valid evidence, `{signal_counts['all_not_evaluated']}` remain entirely unavailable, and `{signal_counts['with_invalid_evidence']}` contain invalid evidence.",
        "",
        "Observed minimum nonzero adjacent deltas are trace-resolution statistics only. They are not sensor-noise, process-noise, numerical-uncertainty, or guard-threshold estimates.",
        "",
        "## Exact Inherited Guard Atoms",
        "",
        f"The analysis evaluates `{len(EXACT_INHERITED_GUARD_ATOM_IDS)}` exact inherited atoms. Realized overspeed uses strict `> 1.90`; clear uses `<= 1.90`. Phase34-compatible component checks preserve inclusive absolute bounds `0.0025`, `0.02`, and `0.25`.",
        "",
        f"Realized overspeed clear is true on `{_guard_true_count(guard_trace, 'realized_overspeed_clear')}` events and realized overspeed is true on `{_guard_true_count(guard_trace, 'realized_overspeed')}` events. Predicted clear and realized clear remain separate.",
        "",
        f"The tangential recoverability component passes on `{_guard_true_count(guard_trace, 'recoverability_tangential_velocity_component_pass')}` events. Radius and radial-velocity component pass counts are `{_guard_true_count(guard_trace, 'recoverability_radius_component_pass')}` and `{_guard_true_count(guard_trace, 'recoverability_radial_velocity_component_pass')}`; combined Phase34-compatible pass count is `{_guard_true_count(guard_trace, 'phase34_compatible_recoverability_pass')}`.",
        "",
        f"Eligible crossing is observed on `{_guard_true_count(guard_trace, 'eligible_target_radius_crossing')}` events; no eligible crossing is true on `{_guard_true_count(guard_trace, 'no_eligible_crossing')}` transition events.",
        "",
        "## Threshold-Free Directional Evidence",
        "",
        f"The analysis evaluates `{len(THRESHOLD_FREE_DIRECTIONAL_GUARD_ATOM_IDS)}` threshold-free directional/component atoms. Radius-gap improvement and target-directed radial motion are visible in the measured transitions. Tangential absolute error initially improves and later worsens after crossing zero. Overspeed headroom and the diagnostic absolute energy-proxy error improve over the checked path.",
        "",
        "These atoms report direction only. They do not establish adequate magnitude, future crossing, recoverability, or phase readiness.",
        "",
        "## Windowed Progress Evidence",
        "",
        "All integer window lengths from one through eight realized transitions were evaluated. Window counts are: "
        + ", ".join(f"`{length}:{count}`" for length, count in sorted(window_counts.items()))
        + ".",
        "",
        "Radius gap, radial recoverability magnitude, overspeed headroom, and diagnostic energy-proxy error improve across the full eight-transition window. Tangential-error direction is window-dependent because the signed error crosses zero. Crossing count remains zero.",
        "",
        "No combined progress score, preferred window, minimum improvement, or stalled/progressing/regressing policy classification was created.",
        "",
        "## Phase Observability",
        "",
        f"Fully observable phases: `{phase_totals['fully_observable_phases']}`. Partially observable phases: `{phase_totals['partially_observable_phases']}`. Future-evaluator-required phases: `{phase_totals['future_evaluator_required_phases']}`. Implemented staged action laws: `{phase_totals['phases_with_implemented_action_laws']}`. Authorized executable guards: `{phase_totals['phases_with_authorized_executable_guards']}`.",
        "",
        "Hazard, radial, tangential, crossing, and recoverability evidence is structurally available to varying degrees. Instability, unsafe-state, handoff readiness, phase-runtime metadata, and correction authority remain unavailable or unsupported.",
        "",
        "## Unresolved Parameters",
        "",
        f"All `{len(UNRESOLVED_PARAMETER_IDS)}` no-progress and anti-chatter parameters remain unresolved: "
        + ", ".join(f"`{item}`" for item in UNRESOLVED_PARAMETER_IDS)
        + ".",
        "",
        "No window length or minimum-improvement threshold is selected or authorized by this analysis.",
        "",
        "## Strongest Supported Conclusion",
        "",
        "The existing measured validation trace is sufficient to demonstrate that several hazard, kinematic-direction, recoverability, crossing, action, and component-wise progress guard atoms are deterministically observable and evaluable offline. It is not sufficient to select general numerical phase guards, no-progress thresholds, hysteresis parameters, action laws, or handoff criteria.",
        "",
        "## Next Smallest Milestone",
        "",
        "The next smallest evidence milestone is a predeclared Stage 1B hazard-arrest/stabilization observational trace set spanning repeated boundary conditions. It should estimate signal variability and evaluator availability without implementing a phase action or authorizing phase transitions.",
        "",
        "## Claim Restrictions",
        "",
        "This analysis does not establish recovery performance, phase-policy validity, false-positive or false-negative rates, general noise, safe hysteresis, optimal thresholds, controller superiority, formal safety, hardware validity, or deployment readiness.",
        "",
        f"Analysis-manifest canonical hash: `{manifest['canonical_payload_hash']}`.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def build_analysis_payloads(
    repository_root: Path,
    *,
    implementation_commit: str,
) -> dict[str, bytes]:
    if len(implementation_commit) != 40 or any(character not in "0123456789abcdef" for character in implementation_commit):
        raise GuardEvidenceError("implementation commit must be a full lowercase Git SHA")
    root = repository_root.resolve()
    source_bundle = load_validated_source_bundle(root)
    events = source_bundle.event_documents()
    definitions = guard_atom_definitions()
    profiles = build_measured_signal_profiles(events)
    signal_document = signal_profile_document(events, profiles)
    inventory_document = guard_inventory_document(definitions)
    guard_trace = build_guard_evaluation_trace(events, definitions)
    window_records = build_windowed_progress_records(events)
    no_progress_document = no_progress_window_document(window_records)
    architecture = _load_json_object(root / ARCHITECTURE_MANIFEST_PATH)
    catalog = _load_json_object(root / INSTRUMENTATION_CATALOG_PATH)
    completeness = source_bundle.document("field_completeness")
    phase_entries = build_phase_observability_entries(
        architecture, catalog, completeness, definitions, guard_trace
    )
    phase_document = phase_observability_document(phase_entries)
    traceability_document = evidence_traceability_document(
        profiles, definitions, guard_trace, events
    )
    manifest = _analysis_manifest_document(
        implementation_commit=implementation_commit,
        source_bundle=source_bundle,
        profiles=profiles,
        definitions=definitions,
        guard_trace=guard_trace,
        phase_document=phase_document,
        signal_document=signal_document,
        inventory_document=inventory_document,
        no_progress_document=no_progress_document,
        traceability_document=traceability_document,
    )
    payloads = {
        ANALYSIS_MANIFEST_FILENAME: pretty_json_bytes(manifest),
        SIGNAL_PROFILE_FILENAME: pretty_json_bytes(signal_document),
        GUARD_INVENTORY_FILENAME: pretty_json_bytes(inventory_document),
        GUARD_TRACE_FILENAME: guard_trace_jsonl_bytes(guard_trace),
        PHASE_MATRIX_FILENAME: pretty_json_bytes(phase_document),
        NO_PROGRESS_FILENAME: pretty_json_bytes(no_progress_document),
        TRACEABILITY_FILENAME: pretty_json_bytes(traceability_document),
        SUMMARY_FILENAME: _summary_bytes(
            manifest, signal_document, guard_trace, phase_document, no_progress_document
        ),
    }
    validate_analysis_payloads(payloads)
    return payloads


def validate_analysis_payloads(payloads: Mapping[str, bytes]) -> tuple[str, str]:
    if tuple(sorted(payloads)) != tuple(sorted(PUBLISHED_FILENAMES)):
        raise GuardEvidenceError("analysis payload set is not exact")
    documents: dict[str, dict[str, object]] = {}
    for filename in (
        ANALYSIS_MANIFEST_FILENAME,
        SIGNAL_PROFILE_FILENAME,
        GUARD_INVENTORY_FILENAME,
        PHASE_MATRIX_FILENAME,
        NO_PROGRESS_FILENAME,
        TRACEABILITY_FILENAME,
    ):
        try:
            document = json.loads(payloads[filename].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GuardEvidenceError(f"invalid JSON payload: {filename}") from exc
        if not isinstance(document, dict) or not document_hash_recomputes(document):
            raise GuardEvidenceError(f"invalid canonical document: {filename}")
        documents[filename] = document
    guard_events: list[dict[str, object]] = []
    for line in payloads[GUARD_TRACE_FILENAME].decode("utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise GuardEvidenceError("guard trace event must be an object")
        guard_events.append(value)
    if len(guard_events) != SOURCE_EVENT_COUNT:
        raise GuardEvidenceError("guard evaluation trace event count mismatch")
    if [event.get("event_index") for event in guard_events] != list(range(SOURCE_EVENT_COUNT)):
        raise GuardEvidenceError("guard evaluation trace ordering mismatch")
    if not all(guard_evaluation_event_hash_recomputes(event) for event in guard_events):
        raise GuardEvidenceError("guard evaluation event hash mismatch")
    if any(event.get("source_trace_aggregate_hash") != SOURCE_TRACE_AGGREGATE_HASH for event in guard_events):
        raise GuardEvidenceError("guard trace source identity mismatch")
    guard_aggregate = aggregate_guard_evaluation_hashes(guard_events)
    manifest = documents[ANALYSIS_MANIFEST_FILENAME]
    if manifest.get("guard_evaluation_trace_aggregate_hash") != guard_aggregate:
        raise GuardEvidenceError("guard trace aggregate hash mismatch")
    if manifest.get("source_trace_manifest_canonical_hash") != SOURCE_TRACE_MANIFEST_CANONICAL_HASH:
        raise GuardEvidenceError("analysis manifest source trace hash mismatch")
    if manifest.get("source_trace_aggregate_hash") != SOURCE_TRACE_AGGREGATE_HASH:
        raise GuardEvidenceError("analysis manifest source aggregate mismatch")
    if manifest.get("analysis_classification") != "offline_guard_evidence_analysis":
        raise GuardEvidenceError("analysis classification mismatch")
    if manifest.get("new_runtime_execution") is not False or manifest.get("new_measured_trace") is not False:
        raise GuardEvidenceError("analysis must not claim runtime execution or a new trace")
    if manifest.get("phase_guard_policy") != "not_frozen" or manifest.get("phase_actions") != "not_implemented":
        raise GuardEvidenceError("phase policy or action status drifted")
    if manifest.get("staged_recovery_execution") != "not_authorized":
        raise GuardEvidenceError("staged recovery execution must remain unauthorized")

    inventory = documents[GUARD_INVENTORY_FILENAME]
    guards = inventory.get("guard_atoms")
    if not isinstance(guards, list) or any(
        not isinstance(item, dict) or item.get("policy_authorization_status") != "not_authorized"
        for item in guards
    ):
        raise GuardEvidenceError("guard inventory contains an authorized or malformed guard")
    parameters = inventory.get("unresolved_parameters")
    if not isinstance(parameters, list) or len(parameters) != len(UNRESOLVED_PARAMETER_IDS):
        raise GuardEvidenceError("unresolved parameter inventory mismatch")
    if any(
        not isinstance(item, dict)
        or item.get("current_status") != "unresolved"
        or item.get("selected_value") is not None
        for item in parameters
    ):
        raise GuardEvidenceError("an unresolved parameter was selected")

    windows = documents[NO_PROGRESS_FILENAME]
    length_items = windows.get("window_lengths")
    if not isinstance(length_items, list) or [item.get("window_length") for item in length_items if isinstance(item, dict)] != list(range(1, 9)):
        raise GuardEvidenceError("window lengths must be exactly one through eight")
    if windows.get("preferred_window_length") is not None or windows.get("minimum_improvement_threshold") is not None:
        raise GuardEvidenceError("no-progress parameter was selected")
    if windows.get("combined_progress_score") is not None or windows.get("stall_progress_regression_policy_classification") is not None:
        raise GuardEvidenceError("unsupported combined progress policy was created")

    phases = documents[PHASE_MATRIX_FILENAME]
    phase_items = phases.get("phases")
    if not isinstance(phase_items, list) or [item.get("phase_id") for item in phase_items if isinstance(item, dict)] != list(PHASE_IDS):
        raise GuardEvidenceError("phase observability matrix mismatch")
    totals = phases.get("totals")
    if not isinstance(totals, dict) or totals.get("phases_with_implemented_action_laws") != 0 or totals.get("phases_with_authorized_executable_guards") != 0:
        raise GuardEvidenceError("phase action or executable guard was introduced")
    if "does not establish recovery performance" not in payloads[SUMMARY_FILENAME].decode("utf-8"):
        raise GuardEvidenceError("summary claim restrictions are incomplete")
    return str(manifest["canonical_payload_hash"]), guard_aggregate


def _target_is_allowed(repository_root: Path, target_directory: Path) -> bool:
    root = repository_root.resolve()
    target = target_directory.absolute()
    try:
        target.relative_to(root)
    except ValueError:
        return False
    if target == root or target == (root / "analysis").resolve():
        return False
    for relative in PROTECTED_OUTPUT_PREFIXES:
        protected = (root / relative).resolve()
        try:
            target.relative_to(protected)
            return False
        except ValueError:
            pass
        try:
            protected.relative_to(target)
            return False
        except ValueError:
            pass
    return target == (root / OUTPUT_RELATIVE_PATH).absolute()


def publish_analysis_payloads(
    repository_root: Path,
    payloads: Mapping[str, bytes],
    *,
    target_directory: Path | None = None,
    writer: object | None = None,
    validator: object | None = None,
) -> AnalysisPublicationResult:
    root = repository_root.resolve()
    target = (target_directory or (root / OUTPUT_RELATIVE_PATH)).absolute()
    if not _target_is_allowed(root, target):
        raise GuardEvidenceError("analysis output target is not allowed")
    if target.exists() or target.is_symlink():
        raise GuardEvidenceError("analysis output target already exists")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise GuardEvidenceError("analysis output parent must be an existing real directory")
    validate = validator if callable(validator) else validate_analysis_payloads
    manifest_hash, guard_hash = validate(payloads)
    staging = target.parent / f".{target.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir()
    published = False
    write = writer if callable(writer) else lambda path, payload: path.write_bytes(payload)
    try:
        for filename in PUBLISHED_FILENAMES:
            write(staging / filename, payloads[filename])
        staged = {filename: (staging / filename).read_bytes() for filename in PUBLISHED_FILENAMES}
        staged_manifest_hash, staged_guard_hash = validate(staged)
        if staged_manifest_hash != manifest_hash or staged_guard_hash != guard_hash:
            raise GuardEvidenceError("staged validation hashes differ from in-memory hashes")
        os.replace(staging, target)
        published = True
        final = {filename: (target / filename).read_bytes() for filename in PUBLISHED_FILENAMES}
        if final != staged:
            raise GuardEvidenceError("published analysis bytes differ from staged bytes")
        final_manifest_hash, final_guard_hash = validate(final)
        artifact_hashes = tuple(
            (filename, sha256_bytes(final[filename])) for filename in PUBLISHED_FILENAMES
        )
        return AnalysisPublicationResult(
            published=True,
            target_directory=str(target),
            artifact_paths=tuple(str(target / filename) for filename in PUBLISHED_FILENAMES),
            artifact_hashes=artifact_hashes,
            analysis_manifest_hash=final_manifest_hash,
            guard_trace_aggregate_hash=final_guard_hash,
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def validate_published_analysis(repository_root: Path) -> AnalysisPublicationResult:
    root = repository_root.resolve()
    target = root / OUTPUT_RELATIVE_PATH
    if not target.is_dir():
        raise GuardEvidenceError("published analysis directory is missing")
    actual = tuple(sorted(path.name for path in target.iterdir() if path.is_file()))
    if actual != tuple(sorted(PUBLISHED_FILENAMES)):
        raise GuardEvidenceError("published analysis artifact set is not exact")
    payloads = {filename: (target / filename).read_bytes() for filename in PUBLISHED_FILENAMES}
    manifest_hash, guard_hash = validate_analysis_payloads(payloads)
    return AnalysisPublicationResult(
        published=True,
        target_directory=str(target),
        artifact_paths=tuple(str(target / filename) for filename in PUBLISHED_FILENAMES),
        artifact_hashes=tuple((filename, sha256_bytes(payloads[filename])) for filename in PUBLISHED_FILENAMES),
        analysis_manifest_hash=manifest_hash,
        guard_trace_aggregate_hash=guard_hash,
    )


def validate_static_contract(
    repository_root: Path,
    *,
    require_output_absent: bool,
) -> GuardEvidenceValidationReport:
    root = repository_root.resolve()
    source_report = validate_source_bundle(root)
    errors = list(source_report.errors)
    try:
        architecture = _load_json_object(root / ARCHITECTURE_MANIFEST_PATH)
        catalog = _load_json_object(root / INSTRUMENTATION_CATALOG_PATH)
        definitions = guard_atom_definitions()
        if architecture.get("canonical_payload_hash") != SOURCE_ARCHITECTURE_CANONICAL_HASH or not document_hash_recomputes(architecture):
            errors.append("source_architecture_manifest_hash_mismatch")
        if architecture.get("phase_ids") != list(PHASE_IDS):
            errors.append("phase_identity_mismatch")
        if not document_hash_recomputes(catalog):
            errors.append("source_instrumentation_catalog_hash_mismatch")
        if any(item.policy_authorization_status != "not_authorized" for item in definitions):
            errors.append("guard_policy_is_authorized")
        if any(item.get("selected_value") is not None for item in unresolved_parameter_inventory()):
            errors.append("unresolved_parameter_has_selected_value")
    except (GuardEvidenceError, OSError) as exc:
        errors.append(str(exc))
    if require_output_absent and (root / OUTPUT_RELATIVE_PATH).exists():
        errors.append("analysis_output_already_exists")
    return GuardEvidenceValidationReport(
        valid=not errors,
        errors=tuple(errors),
        source_event_count=source_report.source_event_count,
        source_transition_count=source_report.source_transition_count,
        source_trace_manifest_hash=source_report.source_trace_manifest_hash,
        source_trace_aggregate_hash=source_report.source_trace_aggregate_hash,
    )


def require_clean_committed_repository(repository_root: Path) -> str:
    root = repository_root.resolve()
    try:
        inside = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "--is-inside-work-tree"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "diff", "--quiet"],
            cwd=root,
            check=False,
        ).returncode
        staged = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "diff", "--cached", "--quiet"],
            cwd=root,
            check=False,
        ).returncode
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GuardEvidenceError(f"cannot inspect Git repository state: {exc}") from exc
    if inside != "true" or len(head) != 40:
        raise GuardEvidenceError("analysis requires a committed Git worktree")
    if tracked != 0 or staged != 0:
        raise GuardEvidenceError("analysis requires a clean tracked tree and staging area")
    return head


__all__ = [
    "ANALYSIS_ID",
    "ANALYSIS_MANIFEST_FILENAME",
    "AnalysisPublicationResult",
    "EXACT_INHERITED_GUARD_ATOM_IDS",
    "GuardAtomDefinition",
    "GuardAtomEvaluation",
    "GuardEvidenceError",
    "GuardEvidenceLevel",
    "GuardEvidenceStatus",
    "GuardEvidenceValidationReport",
    "MeasuredSignalProfile",
    "NO_PROGRESS_FILENAME",
    "OUTPUT_RELATIVE_PATH",
    "PHASE_IDS",
    "PUBLISHED_FILENAMES",
    "SOURCE_STAGE0C_RESULT_COMMIT",
    "SOURCE_TRACE_AGGREGATE_HASH",
    "SOURCE_TRACE_MANIFEST_CANONICAL_HASH",
    "THRESHOLD_FREE_DIRECTIONAL_GUARD_ATOM_IDS",
    "UNRESOLVED_PARAMETER_IDS",
    "WindowedProgressRecord",
    "aggregate_guard_evaluation_hashes",
    "build_analysis_payloads",
    "build_guard_evaluation_trace",
    "build_measured_signal_profiles",
    "build_phase_observability_entries",
    "build_windowed_progress_records",
    "canonical_json_bytes",
    "canonical_sha256",
    "document_hash_recomputes",
    "evaluate_guard_atoms_for_event",
    "guard_atom_definitions",
    "guard_evaluation_event_hash_recomputes",
    "load_validated_source_bundle",
    "publish_analysis_payloads",
    "require_clean_committed_repository",
    "source_event_hash_recomputes",
    "unresolved_parameter_inventory",
    "validate_analysis_payloads",
    "validate_published_analysis",
    "validate_source_bundle",
    "validate_static_contract",
]
