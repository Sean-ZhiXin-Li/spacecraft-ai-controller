from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, fields, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Sequence

from runtime_assurance.staged_recovery_instrumentation import (
    INSTRUMENTATION_SCHEMA_VERSION,
    SOURCE_ARCHITECTURE_ARTIFACT_HASHES,
    CartesianState2D,
    InstrumentationContractError,
    InstrumentationEvidenceStatus,
    InstrumentedValue,
    OrbitalConfiguration,
    StagedRecoveryInstrumentationRecord,
    architecture_signal_coverage,
    build_instrumentation_record,
    canonical_sha256,
    derive_action_geometry,
    derive_orbital_basis,
    derive_orbital_state,
    derive_predicted_hazard_state,
    derive_progress_sample,
    derived_value,
    field_catalog,
    measured_value,
    not_evaluated_value,
)


LOGGER_ID = "staged_recovery_runtime_logger_v0"
LOGGER_SCHEMA_VERSION = "staged_recovery_runtime_logger_event_v0"
TRACE_MANIFEST_SCHEMA_VERSION = "staged_recovery_runtime_trace_manifest_v0"
LOGGER_MANIFEST_SCHEMA_VERSION = "staged_recovery_runtime_logger_manifest_v0"
EVENT_SCHEMA_VERSION = "staged_recovery_runtime_event_schema_v0"
INTEGRATION_CONTRACT_SCHEMA_VERSION = (
    "staged_recovery_runtime_logger_integration_contract_v0"
)
FIELD_COVERAGE_SCHEMA_VERSION = "staged_recovery_runtime_logger_field_coverage_v0"
COMPLETED_DATE = "2026-07-29"

SOURCE_INSTRUMENTATION_COMMIT = "ebc208aedecd11155c6ac9f03bb9b5e40bc69b10"
SOURCE_INSTRUMENTATION_CANONICAL_HASH = (
    "c4947e623e7f9a83de16163f58c5a0da7a3f7b10ee3b10ce88f4eae4805f122c"
)
SOURCE_ARCHITECTURE_COMMIT = "0d416603027e8a27991baf4f89445f6f466b86e6"
SOURCE_ARCHITECTURE_CANONICAL_HASH = (
    "22fa7e0f01c7836ecb1f10838ef00c4cafa937d212bba579fffb25e2c8f11971"
)
ARCHITECTURE_VERSION = "staged_recovery_architecture_v0"

RUNTIME_LOGGING_BOUNDARY_STATUS = "implemented"
SYNTHETIC_TRACE_VALIDATION_STATUS = "implemented"
REAL_RUNNER_INTEGRATION_STATUS = "not_implemented"
REAL_TRACE_VALIDATION_STATUS = "not_performed"
STAGED_EXECUTION_STATUS = "not_authorized"
EXECUTION_NOT_AUTHORIZED_REASON = (
    "the logger can record and validate explicitly supplied runtime events, but it "
    "has not been connected to an authorized runner, validated on a measured "
    "trajectory, or supplied with frozen phase actions, numerical guards, "
    "no-progress thresholds, hysteresis parameters, or handoff-readiness logic"
)

TRACE_MANIFEST_FILENAME = "trace_manifest.json"
TRACE_JSONL_FILENAME = "staged_recovery_trace.jsonl"
TRACE_BUNDLE_FILENAMES = (TRACE_MANIFEST_FILENAME, TRACE_JSONL_FILENAME)

TRACE_CLASSIFICATION = "synthetic"
RUNTIME_SOURCE = "dependency_injected_fixture"
SCIENTIFIC_RESULT = False

CLAIM_RESTRICTIONS = (
    "no_formal_safety_claim",
    "no_measured_trace_claim",
    "no_recovery_performance_claim",
    "no_runtime_completeness_claim",
    "no_staged_execution_claim",
)

SOURCE_INSTRUMENTATION_ARTIFACT_HASHES = (
    (
        "runtime_assurance/staged_recovery_instrumentation.py",
        "1ab0928858de0ed62f37593d2f104b403f7879452c9973194113fa53e939bed4",
    ),
    (
        "analysis/staged_recovery_instrumentation_v0/instrumentation_manifest.json",
        "542f82174737aa97f86ca46f2423621fa9cf0bd4f3e21dba103a25e27453464f",
    ),
    (
        "analysis/staged_recovery_instrumentation_v0/field_catalog.json",
        "aaaf93ef49cd812db0e658a71b3018458dd9eb1b298a4683d8b4d4f827f56646",
    ),
    (
        "analysis/staged_recovery_instrumentation_v0/derivation_traceability.json",
        "1a3aad1a9d2fb6604dd2090503e35f64bde90e2258a2a84a5b40f93f7ba3f4dd",
    ),
    (
        "docs/architecture/staged_recovery_instrumentation_v0.md",
        "ba6347902f0fbc6b1d2959c87b4732b7eda4ef1b039910b1ececea21755d970d",
    ),
    (
        "analysis/staged_recovery_instrumentation_v0/summary.md",
        "28fdaf9e721de3e4a83919731dcf996481e43b328c3c956f045701e8b53620c9",
    ),
    (
        "scripts/check_staged_recovery_instrumentation.py",
        "48fe3da8cd121fc987bc35055640897a1659b8f54339c77c70eccb7cc7f85c88",
    ),
    (
        "Tests/test_staged_recovery_instrumentation.py",
        "1a70b4fecf4057b1e9b7891c41f0afd2d23191cf45a3dec1d69ac6c451347dc2",
    ),
)

LOGGING_SEMANTIC_SOURCES = (
    (
        "docs/architecture/decision_log_schema_v0.md",
        "0c7ad9ee535355116eb10366c296187216052c5aeb8215bde45ad7b0bc52bdd1",
    ),
    (
        "runtime_assurance/recovery_experiment_artifacts.py",
        "00d0b6bd8db02e06cf27a64d1fcc8c21adf64ee765313a13f09559f2362c75e3",
    ),
    (
        "scripts/final_veto_artifacts.py",
        "a5e1369f8b0b62a9b24dd5ce167bd56d3122868b5df9342a9d3a504d36987400",
    ),
)


class RuntimeLoggerContractError(ValueError):
    pass


class RuntimeEventType(str, Enum):
    INITIAL_SNAPSHOT = "initial_snapshot"
    TRANSITION = "transition"
    TERMINAL = "terminal"


class LoggerSessionState(str, Enum):
    CREATED = "created"
    STARTED = "started"
    TERMINAL = "terminal"
    FINALIZED = "finalized"


class ActionDisposition(str, Enum):
    EXECUTED_UNCHANGED = "executed_unchanged"
    EXECUTED_MODIFIED = "executed_modified"
    SUPPRESSED = "suppressed"
    REJECTED = "rejected"
    ZERO_ACTION_EXECUTED = "zero_action_executed"
    NO_ACTION = "no_action"
    NOT_EVALUATED = "not_evaluated"
    INVALID = "invalid"


EVENT_TYPE_ORDER = tuple(item.value for item in RuntimeEventType)
SESSION_STATE_ORDER = tuple(item.value for item in LoggerSessionState)
ACTION_DISPOSITION_VOCABULARY = tuple(item.value for item in ActionDisposition)

PHASE_FIELD_IDS = (
    "current_phase",
    "previous_phase",
    "phase_dwell_count",
    "phase_transition_count",
    "recent_phase_history",
    "phase_transition_reason",
    "no_progress_status",
    "handoff_readiness",
    "retreat_status",
)

EVALUATOR_FIELD_IDS = (
    "simulation_validity",
    "recovery_evaluation_validity",
    "overspeed_status",
    "instability_status",
    "unsafe_state_status",
    "action_rejection_status",
    "explicit_abort_requested",
    "recovery_success_v0",
    "recovery_horizon_exhausted",
    "total_horizon_exhausted",
    "simulator_success",
)

MONITOR_FIELD_IDS = ("final_veto_decision",)

RUNTIME_EVIDENCE_INPUT_FIELD_IDS = tuple(
    sorted(
        set(PHASE_FIELD_IDS)
        | set(EVALUATOR_FIELD_IDS)
        | set(MONITOR_FIELD_IDS)
        | {
            "recovery_horizon_remaining",
            "recovery_horizon_exhausted",
            "total_horizon_remaining",
            "total_horizon_exhausted",
        }
    )
)

PROTECTED_STATIC_RELATIVE_PATHS = (
    "analysis/final_veto_ablation_v0",
    "analysis/recovery_action_branching_nonformal_v0",
    "analysis/recovery_branch_mechanism_diagnosis_v0",
    "analysis/staged_recovery_architecture_v0",
    "analysis/staged_recovery_instrumentation_v0",
    "analysis/staged_recovery_runtime_logger_v0",
    "controller",
    "controllers",
    "runtime_assurance",
    "simulator",
)


def _is_nonnegative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _require_nonempty(value: object, field_id: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeLoggerContractError(f"{field_id} must be a nonempty string")


def _require_sha256(value: object, field_id: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeLoggerContractError(f"{field_id} must be lowercase SHA-256")


def _to_json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {
            item.name: _to_json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _to_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def canonical_runtime_json_bytes(value: object) -> bytes:
    return json.dumps(
        _to_json_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_runtime_sha256(value: object) -> str:
    return hashlib.sha256(canonical_runtime_json_bytes(value)).hexdigest()


def _freeze_strings(values: Sequence[str], field_id: str) -> tuple[str, ...]:
    frozen = tuple(values)
    if any(not isinstance(value, str) or not value for value in frozen):
        raise RuntimeLoggerContractError(f"{field_id} must contain nonempty strings")
    if frozen != tuple(sorted(set(frozen))):
        raise RuntimeLoggerContractError(f"{field_id} must be unique and sorted")
    return frozen


def _freeze_evidence_fields(
    values: Mapping[str, InstrumentedValue]
    | Sequence[tuple[str, InstrumentedValue]]
    | None,
    *,
    allow_logger_fields: bool = False,
) -> tuple[tuple[str, InstrumentedValue], ...]:
    if values is None:
        return ()
    items = tuple(values.items()) if isinstance(values, Mapping) else tuple(values)
    if any(
        not isinstance(key, str) or not isinstance(value, InstrumentedValue)
        for key, value in items
    ):
        raise RuntimeLoggerContractError(
            "runtime evidence must map field IDs to InstrumentedValue instances"
        )
    ordered = tuple(sorted(items, key=lambda item: item[0]))
    if len({key for key, _ in ordered}) != len(ordered):
        raise RuntimeLoggerContractError("runtime evidence field IDs must be unique")
    catalog_ids = {definition.field_id for definition in field_catalog()}
    unknown = sorted({key for key, _ in ordered} - catalog_ids)
    if allow_logger_fields:
        unknown = []
    if unknown:
        raise RuntimeLoggerContractError(f"unknown instrumentation fields: {unknown}")
    return ordered


def _freeze_runtime_input_fields(
    values: Mapping[str, InstrumentedValue]
    | Sequence[tuple[str, InstrumentedValue]]
    | None,
) -> tuple[tuple[str, InstrumentedValue], ...]:
    frozen = _freeze_evidence_fields(values)
    disallowed = sorted(
        field_id
        for field_id, _ in frozen
        if field_id not in RUNTIME_EVIDENCE_INPUT_FIELD_IDS
    )
    if disallowed:
        raise RuntimeLoggerContractError(
            "runtime evidence cannot override explicit state, action, or pure-derived "
            f"instrumentation fields: {disallowed}"
        )
    return frozen


def _evidence_map(
    values: tuple[tuple[str, InstrumentedValue], ...],
) -> dict[str, InstrumentedValue]:
    return dict(values)


def _action_tuple(action: object, field_id: str) -> tuple[float, float] | None:
    if action is None:
        return None
    if (
        not isinstance(action, (tuple, list))
        or len(action) != 2
        or not all(_is_finite_number(item) for item in action)
    ):
        raise RuntimeLoggerContractError(
            f"{field_id} must be a finite two-component action or None"
        )
    return float(action[0]), float(action[1])


def canonical_state_sha256(state: CartesianState2D) -> str:
    if not isinstance(state, CartesianState2D):
        raise RuntimeLoggerContractError("state must be CartesianState2D")
    values = (state.x, state.y, state.vx, state.vy)
    if not all(_is_finite_number(value) for value in values):
        raise RuntimeLoggerContractError("state components must be finite")
    return canonical_runtime_sha256(
        {"x": float(state.x), "y": float(state.y), "vx": float(state.vx), "vy": float(state.vy)}
    )


def _validated_state_hash(
    state: CartesianState2D | None,
    supplied_hash: str | None,
    field_id: str,
) -> str | None:
    if state is None:
        if supplied_hash is not None:
            raise RuntimeLoggerContractError(
                f"{field_id} cannot substitute for a missing Cartesian state"
            )
        return None
    computed = canonical_state_sha256(state)
    if supplied_hash is not None and supplied_hash != computed:
        raise RuntimeLoggerContractError(f"{field_id} does not match supplied state")
    return computed


@dataclass(frozen=True, slots=True)
class StagedRecoverySessionHeader:
    logger_schema_version: str
    instrumentation_schema_version: str
    architecture_version: str
    session_id: str
    case_id: str
    seed: int
    implementation_commit: str
    source_state_hash: str
    simulator_configuration_hash: str
    constants_hash: str
    max_events: int
    declared_output_purpose: str
    execution_authorization_status: str
    scientific_claim_restrictions: tuple[str, ...]
    trace_classification: str = TRACE_CLASSIFICATION
    runtime_source: str = RUNTIME_SOURCE
    scientific_result: bool = SCIENTIFIC_RESULT

    def __post_init__(self) -> None:
        for field_id in (
            "logger_schema_version",
            "instrumentation_schema_version",
            "architecture_version",
            "session_id",
            "case_id",
            "implementation_commit",
            "declared_output_purpose",
        ):
            _require_nonempty(getattr(self, field_id), field_id)
        if self.logger_schema_version != LOGGER_SCHEMA_VERSION:
            raise RuntimeLoggerContractError("unsupported logger schema version")
        if self.instrumentation_schema_version != INSTRUMENTATION_SCHEMA_VERSION:
            raise RuntimeLoggerContractError("unsupported instrumentation schema version")
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise RuntimeLoggerContractError("unsupported architecture version")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise RuntimeLoggerContractError("seed must be an integer")
        for field_id in (
            "source_state_hash",
            "simulator_configuration_hash",
            "constants_hash",
        ):
            _require_sha256(getattr(self, field_id), field_id)
        if not _is_positive_int(self.max_events):
            raise RuntimeLoggerContractError("max_events must be an explicit positive integer")
        object.__setattr__(
            self,
            "scientific_claim_restrictions",
            _freeze_strings(self.scientific_claim_restrictions, "claim restrictions"),
        )
        if self.execution_authorization_status != STAGED_EXECUTION_STATUS:
            raise RuntimeLoggerContractError("Stage 0B execution must remain not_authorized")
        if self.trace_classification != TRACE_CLASSIFICATION:
            raise RuntimeLoggerContractError("Stage 0B traces must be synthetic")
        if self.runtime_source != RUNTIME_SOURCE:
            raise RuntimeLoggerContractError(
                "Stage 0B runtime source must be dependency_injected_fixture"
            )
        if self.scientific_result is not False:
            raise RuntimeLoggerContractError("Stage 0B trace cannot be a scientific result")


@dataclass(frozen=True, slots=True)
class StagedRecoveryInitialSnapshot:
    event_index: int
    recovery_step: int
    total_transition_count: int
    state: CartesianState2D
    configuration: OrbitalConfiguration
    simulation_time: float | None = None
    state_hash: str | None = None
    runtime_evidence: tuple[tuple[str, InstrumentedValue], ...] = ()
    volatile_timestamp: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "runtime_evidence", _freeze_runtime_input_fields(self.runtime_evidence)
        )
        if not _is_nonnegative_int(self.event_index):
            raise RuntimeLoggerContractError("event_index must be nonnegative")
        if not _is_nonnegative_int(self.recovery_step):
            raise RuntimeLoggerContractError("recovery_step must be nonnegative")
        if not _is_nonnegative_int(self.total_transition_count):
            raise RuntimeLoggerContractError("total_transition_count must be nonnegative")
        if self.simulation_time is not None and not _is_finite_number(self.simulation_time):
            raise RuntimeLoggerContractError("simulation_time must be finite or None")
        _validated_state_hash(self.state, self.state_hash, "state_hash")


@dataclass(frozen=True, slots=True)
class StagedRecoveryTransitionInput:
    event_index: int
    recovery_step: int
    total_transition_count: int
    pre_state: CartesianState2D
    configuration: OrbitalConfiguration
    proposed_action: tuple[float, float] | None
    executed_action: tuple[float, float] | None
    action_disposition: ActionDisposition
    transition_executed: bool
    realized_next_state: CartesianState2D | None
    monitor_decision: str | None = None
    predicted_next_state: CartesianState2D | None = None
    simulation_time: float | None = None
    next_simulation_time: float | None = None
    pre_state_hash: str | None = None
    predicted_state_hash: str | None = None
    realized_state_hash: str | None = None
    branch_step: int | None = None
    runtime_evidence: tuple[tuple[str, InstrumentedValue], ...] = ()
    evidence_level: str = "measured_and_derived"
    volatile_timestamp: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "runtime_evidence", _freeze_runtime_input_fields(self.runtime_evidence)
        )
        object.__setattr__(self, "proposed_action", _action_tuple(self.proposed_action, "proposed_action"))
        object.__setattr__(self, "executed_action", _action_tuple(self.executed_action, "executed_action"))
        if not isinstance(self.action_disposition, ActionDisposition):
            raise RuntimeLoggerContractError("unsupported action disposition")
        if type(self.transition_executed) is not bool:
            raise RuntimeLoggerContractError("transition_executed must be boolean")
        for field_id in ("event_index", "recovery_step", "total_transition_count"):
            if not _is_nonnegative_int(getattr(self, field_id)):
                raise RuntimeLoggerContractError(f"{field_id} must be nonnegative")
        if self.branch_step is not None and not _is_nonnegative_int(self.branch_step):
            raise RuntimeLoggerContractError("branch_step must be nonnegative or None")
        for field_id in ("simulation_time", "next_simulation_time"):
            value = getattr(self, field_id)
            if value is not None and not _is_finite_number(value):
                raise RuntimeLoggerContractError(f"{field_id} must be finite or None")
        _require_nonempty(self.evidence_level, "evidence_level")
        _validated_state_hash(self.pre_state, self.pre_state_hash, "pre_state_hash")
        _validated_state_hash(
            self.predicted_next_state, self.predicted_state_hash, "predicted_state_hash"
        )
        _validated_state_hash(
            self.realized_next_state, self.realized_state_hash, "realized_state_hash"
        )


@dataclass(frozen=True, slots=True)
class StagedRecoveryTerminalInput:
    event_index: int
    recovery_step: int
    total_transition_count: int
    terminal_reason: str
    action_disposition: ActionDisposition
    current_state: CartesianState2D | None = None
    configuration: OrbitalConfiguration | None = None
    proposed_action: tuple[float, float] | None = None
    executed_action: tuple[float, float] | None = None
    current_state_hash: str | None = None
    simulation_time: float | None = None
    runtime_evidence: tuple[tuple[str, InstrumentedValue], ...] = ()
    evidence_level: str = "externally_supplied_terminal_evidence"
    volatile_timestamp: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "runtime_evidence", _freeze_runtime_input_fields(self.runtime_evidence)
        )
        object.__setattr__(self, "proposed_action", _action_tuple(self.proposed_action, "proposed_action"))
        object.__setattr__(self, "executed_action", _action_tuple(self.executed_action, "executed_action"))
        if not isinstance(self.action_disposition, ActionDisposition):
            raise RuntimeLoggerContractError("unsupported action disposition")
        for field_id in ("event_index", "recovery_step", "total_transition_count"):
            if not _is_nonnegative_int(getattr(self, field_id)):
                raise RuntimeLoggerContractError(f"{field_id} must be nonnegative")
        _require_nonempty(self.terminal_reason, "terminal_reason")
        _require_nonempty(self.evidence_level, "evidence_level")
        if self.simulation_time is not None and not _is_finite_number(self.simulation_time):
            raise RuntimeLoggerContractError("simulation_time must be finite or None")
        if (self.current_state is None) != (self.configuration is None):
            raise RuntimeLoggerContractError(
                "current_state and configuration must be supplied together"
            )
        _validated_state_hash(self.current_state, self.current_state_hash, "current_state_hash")


@dataclass(frozen=True, slots=True)
class StagedRecoveryPredictedObservation:
    state: CartesianState2D
    state_hash: str
    fields: tuple[tuple[str, InstrumentedValue], ...]

    def __post_init__(self) -> None:
        _require_sha256(self.state_hash, "predicted state hash")
        if self.state_hash != canonical_state_sha256(self.state):
            raise RuntimeLoggerContractError("predicted observation state hash mismatch")
        object.__setattr__(self, "fields", _freeze_evidence_fields(self.fields))


@dataclass(frozen=True, slots=True)
class StagedRecoveryRuntimeEvent:
    schema_version: str
    session_id: str
    event_index: int
    event_type: RuntimeEventType
    case_id: str
    seed: int
    implementation_commit: str
    source_state_hash: str
    simulator_configuration_hash: str
    constants_hash: str
    recovery_step: int
    total_transition_count: int
    simulation_time: float | None
    proposed_action: tuple[float, float] | None
    executed_action: tuple[float, float] | None
    action_disposition: ActionDisposition
    transition_executed: bool
    monitor_decision: str | None
    pre_state_hash: str | None
    predicted_state_hash: str | None
    realized_state_hash: str | None
    pre_observation: StagedRecoveryInstrumentationRecord | None
    predicted_observation: StagedRecoveryPredictedObservation | None
    post_observation: StagedRecoveryInstrumentationRecord | None
    progress_sample: tuple[tuple[str, InstrumentedValue], ...]
    action_geometry: tuple[tuple[str, InstrumentedValue], ...]
    prediction_diagnostics: tuple[tuple[str, InstrumentedValue], ...]
    phase_evidence: tuple[tuple[str, InstrumentedValue], ...]
    evaluator_evidence: tuple[tuple[str, InstrumentedValue], ...]
    terminal_reason: str | None
    terminal: bool
    evidence_level: str
    event_valid: bool
    invalid_reasons: tuple[str, ...]
    volatile_timestamp: str | None
    canonical_event_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != LOGGER_SCHEMA_VERSION:
            raise RuntimeLoggerContractError("unsupported runtime event schema")
        _require_nonempty(self.session_id, "session_id")
        _require_nonempty(self.case_id, "case_id")
        if not isinstance(self.event_type, RuntimeEventType):
            raise RuntimeLoggerContractError("unsupported runtime event type")
        for field_id in ("event_index", "recovery_step", "total_transition_count"):
            if not _is_nonnegative_int(getattr(self, field_id)):
                raise RuntimeLoggerContractError(f"{field_id} must be nonnegative")
        if type(self.transition_executed) is not bool or type(self.terminal) is not bool:
            raise RuntimeLoggerContractError("event flags must be booleans")
        if type(self.event_valid) is not bool:
            raise RuntimeLoggerContractError("event_valid must be boolean")
        if self.simulation_time is not None and not _is_finite_number(self.simulation_time):
            raise RuntimeLoggerContractError("event simulation time must be finite or None")
        object.__setattr__(self, "proposed_action", _action_tuple(self.proposed_action, "proposed_action"))
        object.__setattr__(self, "executed_action", _action_tuple(self.executed_action, "executed_action"))
        for field_id in ("source_state_hash", "simulator_configuration_hash", "constants_hash"):
            _require_sha256(getattr(self, field_id), field_id)
        for field_id in ("pre_state_hash", "predicted_state_hash", "realized_state_hash"):
            value = getattr(self, field_id)
            if value is not None:
                _require_sha256(value, field_id)
        object.__setattr__(self, "progress_sample", _freeze_evidence_fields(self.progress_sample))
        object.__setattr__(self, "action_geometry", _freeze_evidence_fields(self.action_geometry))
        object.__setattr__(
            self,
            "prediction_diagnostics",
            _freeze_evidence_fields(
                self.prediction_diagnostics, allow_logger_fields=True
            ),
        )
        object.__setattr__(self, "phase_evidence", _freeze_evidence_fields(self.phase_evidence))
        object.__setattr__(
            self, "evaluator_evidence", _freeze_evidence_fields(self.evaluator_evidence)
        )
        object.__setattr__(self, "invalid_reasons", _freeze_strings(self.invalid_reasons, "invalid reasons") if self.invalid_reasons else ())
        if self.event_valid == bool(self.invalid_reasons):
            raise RuntimeLoggerContractError(
                "event_valid must be false exactly when invalid reasons are present"
            )
        if self.terminal != (self.event_type == RuntimeEventType.TERMINAL):
            raise RuntimeLoggerContractError("terminal flag must match terminal event type")
        if self.terminal and self.terminal_reason is None:
            raise RuntimeLoggerContractError("terminal event requires terminal_reason")
        if not self.terminal and self.terminal_reason is not None:
            raise RuntimeLoggerContractError("nonterminal event cannot have terminal_reason")
        if self.canonical_event_sha256:
            _require_sha256(self.canonical_event_sha256, "canonical_event_sha256")


@dataclass(frozen=True, slots=True)
class TraceFileHash:
    filename: str
    sha256: str
    hash_scope: str

    def __post_init__(self) -> None:
        if self.filename not in TRACE_BUNDLE_FILENAMES:
            raise RuntimeLoggerContractError("unsupported trace bundle filename")
        _require_sha256(self.sha256, "trace file hash")
        _require_nonempty(self.hash_scope, "trace file hash scope")


@dataclass(frozen=True, slots=True)
class StagedRecoveryTraceManifest:
    trace_manifest_schema_version: str
    logger_schema_version: str
    instrumentation_schema_version: str
    architecture_version: str
    source_instrumentation_commit: str
    source_instrumentation_canonical_hash: str
    source_architecture_commit: str
    source_architecture_canonical_hash: str
    session_id: str
    case_id: str
    seed: int
    implementation_commit: str
    source_state_hash: str
    simulator_configuration_hash: str
    constants_hash: str
    event_count: int
    declared_maximum_event_count: int
    first_event_index: int
    last_event_index: int
    first_recovery_step: int
    last_recovery_step: int
    first_total_transition_count: int
    last_total_transition_count: int
    terminal_status: str
    event_scientific_hashes: tuple[str, ...]
    aggregate_trace_hash: str
    bundle_file_hashes: tuple[TraceFileHash, ...]
    trace_classification: str
    runtime_source: str
    scientific_result: bool
    runtime_integration_classification: str
    claim_restrictions: tuple[str, ...]
    volatile_finalization_timestamp: str | None
    canonical_manifest_payload_sha256: str

    def __post_init__(self) -> None:
        if self.trace_manifest_schema_version != TRACE_MANIFEST_SCHEMA_VERSION:
            raise RuntimeLoggerContractError("unsupported trace manifest schema")
        if self.logger_schema_version != LOGGER_SCHEMA_VERSION:
            raise RuntimeLoggerContractError("trace logger schema mismatch")
        if self.instrumentation_schema_version != INSTRUMENTATION_SCHEMA_VERSION:
            raise RuntimeLoggerContractError("trace instrumentation schema mismatch")
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise RuntimeLoggerContractError("trace architecture version mismatch")
        for field_id in (
            "source_instrumentation_canonical_hash",
            "source_architecture_canonical_hash",
            "source_state_hash",
            "simulator_configuration_hash",
            "constants_hash",
            "aggregate_trace_hash",
        ):
            _require_sha256(getattr(self, field_id), field_id)
        if self.canonical_manifest_payload_sha256:
            _require_sha256(
                self.canonical_manifest_payload_sha256,
                "canonical_manifest_payload_sha256",
            )
        if not _is_positive_int(self.event_count):
            raise RuntimeLoggerContractError("trace must contain at least one event")
        if not _is_positive_int(self.declared_maximum_event_count):
            raise RuntimeLoggerContractError("trace maximum event count must be positive")
        if self.event_count > self.declared_maximum_event_count:
            raise RuntimeLoggerContractError("trace exceeds declared event capacity")
        if len(self.event_scientific_hashes) != self.event_count:
            raise RuntimeLoggerContractError("event hash count mismatch")
        for digest in self.event_scientific_hashes:
            _require_sha256(digest, "event scientific hash")
        object.__setattr__(
            self, "claim_restrictions", _freeze_strings(self.claim_restrictions, "claim restrictions")
        )
        if self.trace_classification != TRACE_CLASSIFICATION:
            raise RuntimeLoggerContractError("trace classification must remain synthetic")
        if self.runtime_source != RUNTIME_SOURCE:
            raise RuntimeLoggerContractError("runtime source must remain dependency injected")
        if self.scientific_result is not False:
            raise RuntimeLoggerContractError("synthetic trace is not a scientific result")
        if self.runtime_integration_classification != REAL_RUNNER_INTEGRATION_STATUS:
            raise RuntimeLoggerContractError("real runner integration must remain not_implemented")


@dataclass(frozen=True, slots=True)
class StagedRecoveryTraceBundle:
    header: StagedRecoverySessionHeader
    events: tuple[StagedRecoveryRuntimeEvent, ...]
    manifest: StagedRecoveryTraceManifest

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))


@dataclass(frozen=True, slots=True)
class StagedRecoveryTraceWriteResult:
    published: bool
    target_directory: str
    artifact_paths: tuple[str, ...]
    artifact_hashes: tuple[tuple[str, str], ...]
    event_count: int
    aggregate_trace_hash: str


@dataclass(frozen=True, slots=True)
class StagedRecoveryLoggerValidationReport:
    valid: bool
    errors: tuple[str, ...]
    event_count: int
    aggregate_trace_hash: str | None


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


def _event_scientific_payload(event: StagedRecoveryRuntimeEvent) -> dict[str, object]:
    payload = _without_volatile_fields(_to_json_value(event))
    assert isinstance(payload, dict)
    payload.pop("canonical_event_sha256", None)
    return payload


def canonical_event_sha256(event: StagedRecoveryRuntimeEvent) -> str:
    return canonical_runtime_sha256(_event_scientific_payload(event))


def with_canonical_event_hash(
    event: StagedRecoveryRuntimeEvent,
) -> StagedRecoveryRuntimeEvent:
    unhashed = replace(event, canonical_event_sha256="")
    return replace(unhashed, canonical_event_sha256=canonical_event_sha256(unhashed))


def event_hash_recomputes(event: StagedRecoveryRuntimeEvent) -> bool:
    return event.canonical_event_sha256 == canonical_event_sha256(event)


def aggregate_trace_sha256(events: Sequence[StagedRecoveryRuntimeEvent]) -> str:
    payload = {
        "ordered_event_scientific_hashes": [event.canonical_event_sha256 for event in events]
    }
    return canonical_runtime_sha256(payload)


def _trace_manifest_scientific_payload(
    manifest: StagedRecoveryTraceManifest,
) -> dict[str, object]:
    payload = _to_json_value(manifest)
    assert isinstance(payload, dict)
    payload.pop("volatile_finalization_timestamp", None)
    payload.pop("canonical_manifest_payload_sha256", None)
    payload.pop("bundle_file_hashes", None)
    return payload


def canonical_trace_manifest_sha256(manifest: StagedRecoveryTraceManifest) -> str:
    return canonical_runtime_sha256(_trace_manifest_scientific_payload(manifest))


def trace_manifest_hash_recomputes(manifest: StagedRecoveryTraceManifest) -> bool:
    return (
        manifest.canonical_manifest_payload_sha256
        == canonical_trace_manifest_sha256(manifest)
    )


def _action_flags(disposition: ActionDisposition) -> tuple[bool, bool]:
    return (
        disposition == ActionDisposition.REJECTED,
        False,
    )


def _validate_action_disposition(
    *,
    proposed_action: tuple[float, float] | None,
    executed_action: tuple[float, float] | None,
    disposition: ActionDisposition,
    transition_executed: bool,
    terminal_reason: str | None = None,
) -> None:
    equal = proposed_action is not None and proposed_action == executed_action
    both_zero = proposed_action == (0.0, 0.0) and executed_action == (0.0, 0.0)
    if disposition == ActionDisposition.EXECUTED_UNCHANGED:
        valid = transition_executed and equal and not both_zero
    elif disposition in {ActionDisposition.EXECUTED_MODIFIED, ActionDisposition.SUPPRESSED}:
        valid = (
            transition_executed
            and proposed_action is not None
            and executed_action is not None
            and proposed_action != executed_action
        )
    elif disposition == ActionDisposition.ZERO_ACTION_EXECUTED:
        valid = transition_executed and both_zero
    elif disposition == ActionDisposition.REJECTED:
        valid = (
            not transition_executed
            and proposed_action is not None
            and executed_action is None
        )
    elif disposition == ActionDisposition.NO_ACTION:
        valid = (
            not transition_executed
            and proposed_action is None
            and executed_action is None
        )
    elif disposition in {ActionDisposition.NOT_EVALUATED, ActionDisposition.INVALID}:
        valid = (
            not transition_executed
            and proposed_action is None
            and executed_action is None
        )
    else:
        valid = False
    if not valid:
        raise RuntimeLoggerContractError(
            f"action disposition {disposition.value} contradicts supplied action evidence"
        )
    if terminal_reason in {"explicit_abort", "explicit_recovery_abort"} and (
        disposition != ActionDisposition.NO_ACTION
        or proposed_action is not None
        or executed_action is not None
        or transition_executed
    ):
        raise RuntimeLoggerContractError(
            "explicit abort must record no action and zero physical transition"
        )
    if (
        terminal_reason in {"action_rejected", "recovery_action_rejected"}
        and disposition != ActionDisposition.REJECTED
    ):
        raise RuntimeLoggerContractError(
            "action-rejection terminal reason requires rejected disposition"
        )


def _not_evaluated_progress(source_step: int) -> tuple[tuple[str, InstrumentedValue], ...]:
    return derive_progress_sample(None, None, source_step=source_step).values


_PREDICTION_DIAGNOSTIC_UNITS = {
    "speed_ratio_prediction_error": "dimensionless",
    "position_x_prediction_error": "m",
    "position_y_prediction_error": "m",
    "velocity_x_prediction_error": "m/s",
    "velocity_y_prediction_error": "m/s",
    "predicted_state_hash_matches_realized_state_hash": "boolean",
}


def _prediction_diagnostics(
    predicted_state: CartesianState2D | None,
    realized_state: CartesianState2D | None,
    predicted_fields: Mapping[str, InstrumentedValue] | None,
    post_observation: StagedRecoveryInstrumentationRecord | None,
    *,
    source_step: int,
) -> tuple[tuple[str, InstrumentedValue], ...]:
    values: dict[str, InstrumentedValue] = {}
    if predicted_state is None or realized_state is None:
        return tuple(
            (
                field_id,
                not_evaluated_value(
                    reason="predicted_and_realized_state_are_both_required",
                    units=units,
                    source_id="staged_recovery_runtime_logger.prediction_diagnostics",
                    source_step=source_step,
                ),
            )
            for field_id, units in sorted(_PREDICTION_DIAGNOSTIC_UNITS.items())
        )
    for field_id, attribute, units in (
        ("position_x_prediction_error", "x", "m"),
        ("position_y_prediction_error", "y", "m"),
        ("velocity_x_prediction_error", "vx", "m/s"),
        ("velocity_y_prediction_error", "vy", "m/s"),
    ):
        values[field_id] = derived_value(
            float(getattr(realized_state, attribute))
            - float(getattr(predicted_state, attribute)),
            units=units,
            source_id=f"staged_recovery_runtime_logger.{field_id}",
            source_step=source_step,
            reason="realized_minus_one_step_predicted_component",
            input_source_ids=(
                f"predicted_state.{attribute}",
                f"realized_state.{attribute}",
            ),
        )
    predicted_ratio = (
        predicted_fields.get("predicted_speed_ratio") if predicted_fields else None
    )
    realized_ratio = (
        post_observation.field("realized_speed_ratio") if post_observation else None
    )
    if (
        predicted_ratio is not None
        and realized_ratio is not None
        and predicted_ratio.available
        and realized_ratio.available
        and _is_finite_number(predicted_ratio.value)
        and _is_finite_number(realized_ratio.value)
    ):
        values["speed_ratio_prediction_error"] = derived_value(
            float(realized_ratio.value) - float(predicted_ratio.value),
            units="dimensionless",
            source_id="staged_recovery_runtime_logger.speed_ratio_prediction_error",
            source_step=source_step,
            reason="realized_minus_one_step_predicted_speed_ratio",
            input_source_ids=("predicted_speed_ratio", "realized_speed_ratio"),
        )
    else:
        values["speed_ratio_prediction_error"] = not_evaluated_value(
            reason="predicted_or_realized_speed_ratio_is_unavailable",
            units="dimensionless",
            source_id="staged_recovery_runtime_logger.speed_ratio_prediction_error",
            source_step=source_step,
        )
    values["predicted_state_hash_matches_realized_state_hash"] = derived_value(
        canonical_state_sha256(predicted_state) == canonical_state_sha256(realized_state),
        units="boolean",
        source_id="staged_recovery_runtime_logger.state_hash_identity",
        source_step=source_step,
        reason="exact_canonical_state_identity_only_not_physical_distance",
        input_source_ids=("predicted_state_hash", "realized_state_hash"),
    )
    return tuple(sorted(values.items()))


def _event_invalid_reasons(
    runtime_evidence: tuple[tuple[str, InstrumentedValue], ...],
    *,
    disposition: ActionDisposition,
    terminal_reason: str | None,
) -> tuple[str, ...]:
    evidence = dict(runtime_evidence)
    reasons = {
        f"invalid_runtime_evidence:{field_id}"
        for field_id, value in runtime_evidence
        if value.status == InstrumentationEvidenceStatus.INVALID
    }
    recovery_success = evidence.get("recovery_success_v0")
    success_true = (
        recovery_success is not None
        and recovery_success.available
        and recovery_success.value is True
    )
    if success_true:
        for field_id in ("simulation_validity", "recovery_evaluation_validity"):
            value = evidence.get(field_id)
            if value is not None and (
                value.status == InstrumentationEvidenceStatus.INVALID
                or (value.available and value.value is False)
            ):
                reasons.add(f"recovery_success_contradicts:{field_id}")
        for field_id in (
            "overspeed_status",
            "instability_status",
            "unsafe_state_status",
            "action_rejection_status",
            "explicit_abort_requested",
        ):
            value = evidence.get(field_id)
            if value is not None and value.available and value.value is True:
                reasons.add(f"recovery_success_contradicts:{field_id}")
        if disposition == ActionDisposition.REJECTED:
            reasons.add("recovery_success_contradicts_terminal_action_disposition")
        if terminal_reason in {"explicit_abort", "explicit_recovery_abort"}:
            reasons.add("explicit_abort_cannot_be_recovery_success")
    return tuple(sorted(reasons))


def _select_evidence(
    runtime_evidence: tuple[tuple[str, InstrumentedValue], ...],
    field_ids: Sequence[str],
) -> tuple[tuple[str, InstrumentedValue], ...]:
    allowed = set(field_ids)
    return tuple((key, value) for key, value in runtime_evidence if key in allowed)


def _observation_external_fields(
    runtime_evidence: tuple[tuple[str, InstrumentedValue], ...],
    *,
    monitor_decision: str | None,
    action_rejected: bool,
    explicit_abort: bool,
    source_step: int,
) -> dict[str, InstrumentedValue]:
    values = dict(runtime_evidence)
    if monitor_decision is not None:
        supplied = values.get("final_veto_decision")
        measured = measured_value(
            monitor_decision,
            units="categorical",
            source_id="explicit.monitor_decision",
            source_step=source_step,
        )
        if supplied is not None and (
            not supplied.available or supplied.value != monitor_decision
        ):
            raise RuntimeLoggerContractError(
                "monitor_decision contradicts final_veto_decision evidence"
            )
        values["final_veto_decision"] = measured
    for field_id, flag in (
        ("action_rejection_status", action_rejected),
        ("explicit_abort_requested", explicit_abort),
    ):
        supplied = values.get(field_id)
        measured = measured_value(
            flag,
            units="boolean",
            source_id=f"explicit.{field_id}",
            source_step=source_step,
        )
        if supplied is not None and (
            not supplied.available or supplied.value is not flag
        ):
            raise RuntimeLoggerContractError(f"{field_id} contradicts action disposition")
        values[field_id] = measured
    return values


def _build_observation(
    *,
    header: StagedRecoverySessionHeader,
    state: CartesianState2D,
    configuration: OrbitalConfiguration,
    recovery_step: int,
    total_transition_count: int,
    simulation_time: float | None,
    previous_state: CartesianState2D | None,
    previous_step: int | None,
    predicted_state: CartesianState2D | None,
    proposed_action: tuple[float, float] | None,
    executed_action: tuple[float, float] | None,
    branch_step: int | None,
    action_rejected: bool,
    explicit_abort: bool,
    runtime_evidence: tuple[tuple[str, InstrumentedValue], ...],
    monitor_decision: str | None,
    volatile_timestamp: str | None,
) -> StagedRecoveryInstrumentationRecord:
    external = _observation_external_fields(
        runtime_evidence,
        monitor_decision=monitor_decision,
        action_rejected=action_rejected,
        explicit_abort=explicit_abort,
        source_step=recovery_step,
    )
    try:
        return build_instrumentation_record(
            state=state,
            configuration=configuration,
            case_id=header.case_id,
            seed=header.seed,
            implementation_commit=header.implementation_commit,
            branch_state_hash=header.source_state_hash,
            simulator_configuration_hash=header.simulator_configuration_hash,
            constants_hash=header.constants_hash,
            recovery_step=recovery_step,
            total_transition_count=total_transition_count,
            simulation_time=simulation_time,
            previous_state=previous_state,
            previous_step=previous_step,
            predicted_state=predicted_state,
            proposed_action=proposed_action,
            executed_action=executed_action,
            branch_step=branch_step,
            action_rejected=action_rejected,
            explicit_abort=explicit_abort,
            external_fields=external,
            volatile_provenance_timestamp=volatile_timestamp,
        )
    except InstrumentationContractError as exc:
        raise RuntimeLoggerContractError(str(exc)) from exc


def _empty_action_geometry(
    state: CartesianState2D,
    configuration: OrbitalConfiguration,
    *,
    source_step: int,
) -> tuple[tuple[str, InstrumentedValue], ...]:
    basis = derive_orbital_basis(state, source_step=source_step)
    return derive_action_geometry(
        None,
        None,
        basis,
        action_component_limit=configuration.action_component_limit,
        source_step=source_step,
    ).values


def _build_runtime_event(
    *,
    header: StagedRecoverySessionHeader,
    event_index: int,
    event_type: RuntimeEventType,
    recovery_step: int,
    total_transition_count: int,
    simulation_time: float | None,
    proposed_action: tuple[float, float] | None,
    executed_action: tuple[float, float] | None,
    action_disposition: ActionDisposition,
    transition_executed: bool,
    monitor_decision: str | None,
    pre_state_hash: str | None,
    predicted_state_hash: str | None,
    realized_state_hash: str | None,
    pre_observation: StagedRecoveryInstrumentationRecord | None,
    predicted_observation: StagedRecoveryPredictedObservation | None,
    post_observation: StagedRecoveryInstrumentationRecord | None,
    progress_sample: tuple[tuple[str, InstrumentedValue], ...],
    action_geometry: tuple[tuple[str, InstrumentedValue], ...],
    prediction_diagnostics: tuple[tuple[str, InstrumentedValue], ...],
    runtime_evidence: tuple[tuple[str, InstrumentedValue], ...],
    terminal_reason: str | None,
    evidence_level: str,
    volatile_timestamp: str | None,
) -> StagedRecoveryRuntimeEvent:
    invalid_reasons = _event_invalid_reasons(
        runtime_evidence,
        disposition=action_disposition,
        terminal_reason=terminal_reason,
    )
    event = StagedRecoveryRuntimeEvent(
        schema_version=LOGGER_SCHEMA_VERSION,
        session_id=header.session_id,
        event_index=event_index,
        event_type=event_type,
        case_id=header.case_id,
        seed=header.seed,
        implementation_commit=header.implementation_commit,
        source_state_hash=header.source_state_hash,
        simulator_configuration_hash=header.simulator_configuration_hash,
        constants_hash=header.constants_hash,
        recovery_step=recovery_step,
        total_transition_count=total_transition_count,
        simulation_time=simulation_time,
        proposed_action=proposed_action,
        executed_action=executed_action,
        action_disposition=action_disposition,
        transition_executed=transition_executed,
        monitor_decision=monitor_decision,
        pre_state_hash=pre_state_hash,
        predicted_state_hash=predicted_state_hash,
        realized_state_hash=realized_state_hash,
        pre_observation=pre_observation,
        predicted_observation=predicted_observation,
        post_observation=post_observation,
        progress_sample=progress_sample,
        action_geometry=action_geometry,
        prediction_diagnostics=prediction_diagnostics,
        phase_evidence=_select_evidence(runtime_evidence, PHASE_FIELD_IDS),
        evaluator_evidence=_select_evidence(runtime_evidence, EVALUATOR_FIELD_IDS),
        terminal_reason=terminal_reason,
        terminal=event_type == RuntimeEventType.TERMINAL,
        evidence_level=evidence_level,
        event_valid=not invalid_reasons,
        invalid_reasons=invalid_reasons,
        volatile_timestamp=volatile_timestamp,
        canonical_event_sha256="",
    )
    return with_canonical_event_hash(event)


class StagedRecoveryRuntimeLoggerSession:
    def __init__(self, header: StagedRecoverySessionHeader) -> None:
        if not isinstance(header, StagedRecoverySessionHeader):
            raise RuntimeLoggerContractError("header must be StagedRecoverySessionHeader")
        self._header = header
        self._state = LoggerSessionState.CREATED
        self._events: list[StagedRecoveryRuntimeEvent] = []
        self._last_state: CartesianState2D | None = None
        self._last_configuration: OrbitalConfiguration | None = None
        self._last_state_hash: str | None = None
        self._last_recovery_step: int | None = None
        self._last_total_transition_count: int | None = None
        self._capacity_exhausted = False

    @property
    def header(self) -> StagedRecoverySessionHeader:
        return self._header

    @property
    def state(self) -> LoggerSessionState:
        return self._state

    @property
    def events(self) -> tuple[StagedRecoveryRuntimeEvent, ...]:
        return tuple(self._events)

    @property
    def capacity_exhausted(self) -> bool:
        return self._capacity_exhausted

    def _require_capacity(self) -> None:
        if len(self._events) >= self._header.max_events:
            self._capacity_exhausted = True
            raise RuntimeLoggerContractError(
                "explicit logger event capacity exhausted before append"
            )

    def _require_event_index(self, event_index: int) -> None:
        if event_index != len(self._events):
            raise RuntimeLoggerContractError(
                f"event_index must be sequential; expected {len(self._events)}"
            )

    def record_initial_snapshot(
        self, snapshot: StagedRecoveryInitialSnapshot
    ) -> StagedRecoveryRuntimeEvent:
        if self._state != LoggerSessionState.CREATED:
            raise RuntimeLoggerContractError("initial snapshot is allowed only once")
        self._require_capacity()
        self._require_event_index(snapshot.event_index)
        if snapshot.event_index != 0:
            raise RuntimeLoggerContractError("initial snapshot event_index must be zero")
        state_hash = _validated_state_hash(snapshot.state, snapshot.state_hash, "state_hash")
        assert state_hash is not None
        observation = _build_observation(
            header=self._header,
            state=snapshot.state,
            configuration=snapshot.configuration,
            recovery_step=snapshot.recovery_step,
            total_transition_count=snapshot.total_transition_count,
            simulation_time=snapshot.simulation_time,
            previous_state=None,
            previous_step=None,
            predicted_state=None,
            proposed_action=None,
            executed_action=None,
            branch_step=None,
            action_rejected=False,
            explicit_abort=False,
            runtime_evidence=snapshot.runtime_evidence,
            monitor_decision=None,
            volatile_timestamp=snapshot.volatile_timestamp,
        )
        event = _build_runtime_event(
            header=self._header,
            event_index=snapshot.event_index,
            event_type=RuntimeEventType.INITIAL_SNAPSHOT,
            recovery_step=snapshot.recovery_step,
            total_transition_count=snapshot.total_transition_count,
            simulation_time=snapshot.simulation_time,
            proposed_action=None,
            executed_action=None,
            action_disposition=ActionDisposition.NO_ACTION,
            transition_executed=False,
            monitor_decision=None,
            pre_state_hash=state_hash,
            predicted_state_hash=None,
            realized_state_hash=None,
            pre_observation=observation,
            predicted_observation=None,
            post_observation=None,
            progress_sample=_not_evaluated_progress(snapshot.recovery_step),
            action_geometry=_empty_action_geometry(
                snapshot.state, snapshot.configuration, source_step=snapshot.recovery_step
            ),
            prediction_diagnostics=_prediction_diagnostics(
                None,
                None,
                None,
                None,
                source_step=snapshot.recovery_step,
            ),
            runtime_evidence=snapshot.runtime_evidence,
            terminal_reason=None,
            evidence_level="measured_initial_snapshot",
            volatile_timestamp=snapshot.volatile_timestamp,
        )
        self._events.append(event)
        self._state = LoggerSessionState.STARTED
        self._last_state = snapshot.state
        self._last_configuration = snapshot.configuration
        self._last_state_hash = state_hash
        self._last_recovery_step = snapshot.recovery_step
        self._last_total_transition_count = snapshot.total_transition_count
        return event

    def record_transition(
        self, transition: StagedRecoveryTransitionInput
    ) -> StagedRecoveryRuntimeEvent:
        if self._state != LoggerSessionState.STARTED:
            raise RuntimeLoggerContractError(
                "transition requires an initial snapshot and cannot follow terminal/finalize"
            )
        self._require_capacity()
        self._require_event_index(transition.event_index)
        assert self._last_state is not None
        assert self._last_configuration is not None
        assert self._last_state_hash is not None
        assert self._last_recovery_step is not None
        assert self._last_total_transition_count is not None
        if canonical_state_sha256(transition.pre_state) != self._last_state_hash:
            raise RuntimeLoggerContractError(
                "transition pre-state is not the last measured session state"
            )
        if transition.configuration != self._last_configuration:
            raise RuntimeLoggerContractError(
                "orbital configuration changed within logger session"
            )
        expected_delta = 1 if transition.transition_executed else 0
        if transition.recovery_step != self._last_recovery_step + expected_delta:
            raise RuntimeLoggerContractError(
                "recovery_step must increment exactly once for a realized transition"
            )
        if (
            transition.total_transition_count
            != self._last_total_transition_count + expected_delta
        ):
            raise RuntimeLoggerContractError(
                "total_transition_count must increment exactly once for a realized transition"
            )
        if transition.transition_executed and transition.realized_next_state is None:
            raise RuntimeLoggerContractError(
                "executed transition requires a measured realized next state"
            )
        if not transition.transition_executed and transition.realized_next_state is not None:
            raise RuntimeLoggerContractError(
                "nonexecuted transition forbids a realized next state"
            )
        _validate_action_disposition(
            proposed_action=transition.proposed_action,
            executed_action=transition.executed_action,
            disposition=transition.action_disposition,
            transition_executed=transition.transition_executed,
        )
        if transition.monitor_decision is not None:
            _require_nonempty(transition.monitor_decision, "monitor_decision")
        pre_hash = _validated_state_hash(
            transition.pre_state, transition.pre_state_hash, "pre_state_hash"
        )
        predicted_hash = _validated_state_hash(
            transition.predicted_next_state,
            transition.predicted_state_hash,
            "predicted_state_hash",
        )
        realized_hash = _validated_state_hash(
            transition.realized_next_state,
            transition.realized_state_hash,
            "realized_state_hash",
        )
        action_rejected, explicit_abort = _action_flags(transition.action_disposition)
        pre_observation = _build_observation(
            header=self._header,
            state=transition.pre_state,
            configuration=transition.configuration,
            recovery_step=self._last_recovery_step,
            total_transition_count=self._last_total_transition_count,
            simulation_time=transition.simulation_time,
            previous_state=None,
            previous_step=None,
            predicted_state=transition.predicted_next_state,
            proposed_action=transition.proposed_action,
            executed_action=transition.executed_action,
            branch_step=transition.branch_step,
            action_rejected=action_rejected,
            explicit_abort=explicit_abort,
            runtime_evidence=transition.runtime_evidence,
            monitor_decision=transition.monitor_decision,
            volatile_timestamp=transition.volatile_timestamp,
        )
        predicted_observation = None
        predicted_fields: tuple[tuple[str, InstrumentedValue], ...] | None = None
        if transition.predicted_next_state is not None:
            assert predicted_hash is not None
            predicted_fields = derive_predicted_hazard_state(
                transition.predicted_next_state,
                transition.configuration,
                source_step=self._last_recovery_step,
            )
            predicted_observation = StagedRecoveryPredictedObservation(
                state=transition.predicted_next_state,
                state_hash=predicted_hash,
                fields=predicted_fields,
            )
        post_observation = None
        progress_sample = _not_evaluated_progress(transition.recovery_step)
        if transition.transition_executed:
            assert transition.realized_next_state is not None
            post_observation = _build_observation(
                header=self._header,
                state=transition.realized_next_state,
                configuration=transition.configuration,
                recovery_step=transition.recovery_step,
                total_transition_count=transition.total_transition_count,
                simulation_time=transition.next_simulation_time,
                previous_state=transition.pre_state,
                previous_step=self._last_recovery_step,
                predicted_state=None,
                proposed_action=None,
                executed_action=None,
                branch_step=transition.branch_step,
                action_rejected=False,
                explicit_abort=False,
                runtime_evidence=transition.runtime_evidence,
                monitor_decision=transition.monitor_decision,
                volatile_timestamp=transition.volatile_timestamp,
            )
            previous_orbital = derive_orbital_state(
                transition.pre_state,
                transition.configuration,
                source_step=self._last_recovery_step,
            )
            current_orbital = derive_orbital_state(
                transition.realized_next_state,
                transition.configuration,
                source_step=transition.recovery_step,
            )
            progress_sample = derive_progress_sample(
                previous_orbital,
                current_orbital,
                previous_transition_count=self._last_total_transition_count,
                current_transition_count=transition.total_transition_count,
                previous_time=transition.simulation_time,
                current_time=transition.next_simulation_time,
                source_step=transition.recovery_step,
            ).values
        action_geometry = derive_action_geometry(
            transition.proposed_action,
            transition.executed_action,
            derive_orbital_basis(
                transition.pre_state, source_step=self._last_recovery_step
            ),
            action_component_limit=transition.configuration.action_component_limit,
            action_rejected=action_rejected,
            explicit_abort=explicit_abort,
            source_step=self._last_recovery_step,
        ).values
        diagnostics = _prediction_diagnostics(
            transition.predicted_next_state,
            transition.realized_next_state,
            dict(predicted_fields) if predicted_fields is not None else None,
            post_observation,
            source_step=transition.recovery_step,
        )
        event = _build_runtime_event(
            header=self._header,
            event_index=transition.event_index,
            event_type=RuntimeEventType.TRANSITION,
            recovery_step=transition.recovery_step,
            total_transition_count=transition.total_transition_count,
            simulation_time=(
                transition.next_simulation_time
                if transition.transition_executed
                else transition.simulation_time
            ),
            proposed_action=transition.proposed_action,
            executed_action=transition.executed_action,
            action_disposition=transition.action_disposition,
            transition_executed=transition.transition_executed,
            monitor_decision=transition.monitor_decision,
            pre_state_hash=pre_hash,
            predicted_state_hash=predicted_hash,
            realized_state_hash=realized_hash,
            pre_observation=pre_observation,
            predicted_observation=predicted_observation,
            post_observation=post_observation,
            progress_sample=progress_sample,
            action_geometry=action_geometry,
            prediction_diagnostics=diagnostics,
            runtime_evidence=transition.runtime_evidence,
            terminal_reason=None,
            evidence_level=transition.evidence_level,
            volatile_timestamp=transition.volatile_timestamp,
        )
        self._events.append(event)
        if transition.transition_executed:
            assert transition.realized_next_state is not None
            assert realized_hash is not None
            self._last_state = transition.realized_next_state
            self._last_state_hash = realized_hash
        self._last_recovery_step = transition.recovery_step
        self._last_total_transition_count = transition.total_transition_count
        return event

    def record_terminal(
        self, terminal: StagedRecoveryTerminalInput
    ) -> StagedRecoveryRuntimeEvent:
        if self._state not in {LoggerSessionState.CREATED, LoggerSessionState.STARTED}:
            raise RuntimeLoggerContractError(
                "terminal event cannot follow another terminal or finalization"
            )
        self._require_capacity()
        self._require_event_index(terminal.event_index)
        if terminal.executed_action is not None:
            raise RuntimeLoggerContractError(
                "terminal event cannot contain an executed physical action"
            )
        _validate_action_disposition(
            proposed_action=terminal.proposed_action,
            executed_action=terminal.executed_action,
            disposition=terminal.action_disposition,
            transition_executed=False,
            terminal_reason=terminal.terminal_reason,
        )
        if self._state == LoggerSessionState.CREATED:
            if terminal.current_state is None or terminal.configuration is None:
                raise RuntimeLoggerContractError(
                    "zero-transition terminal session must preserve its initial state"
                )
            state = terminal.current_state
            configuration = terminal.configuration
            state_hash = _validated_state_hash(
                state, terminal.current_state_hash, "current_state_hash"
            )
        else:
            assert self._last_state is not None
            assert self._last_configuration is not None
            assert self._last_state_hash is not None
            assert self._last_recovery_step is not None
            assert self._last_total_transition_count is not None
            state = terminal.current_state or self._last_state
            configuration = terminal.configuration or self._last_configuration
            state_hash = _validated_state_hash(
                state, terminal.current_state_hash, "current_state_hash"
            )
            if state_hash != self._last_state_hash:
                raise RuntimeLoggerContractError(
                    "zero-transition terminal state must equal the last measured state"
                )
            if terminal.recovery_step != self._last_recovery_step:
                raise RuntimeLoggerContractError(
                    "terminal event without transition must retain recovery_step"
                )
            if terminal.total_transition_count != self._last_total_transition_count:
                raise RuntimeLoggerContractError(
                    "terminal event without transition must retain total_transition_count"
                )
        assert state_hash is not None
        action_rejected = terminal.action_disposition == ActionDisposition.REJECTED
        explicit_abort = terminal.terminal_reason in {
            "explicit_abort",
            "explicit_recovery_abort",
        }
        observation = _build_observation(
            header=self._header,
            state=state,
            configuration=configuration,
            recovery_step=terminal.recovery_step,
            total_transition_count=terminal.total_transition_count,
            simulation_time=terminal.simulation_time,
            previous_state=None,
            previous_step=None,
            predicted_state=None,
            proposed_action=terminal.proposed_action,
            executed_action=None,
            branch_step=None,
            action_rejected=action_rejected,
            explicit_abort=explicit_abort,
            runtime_evidence=terminal.runtime_evidence,
            monitor_decision=None,
            volatile_timestamp=terminal.volatile_timestamp,
        )
        action_geometry = derive_action_geometry(
            terminal.proposed_action,
            None,
            derive_orbital_basis(state, source_step=terminal.recovery_step),
            action_component_limit=configuration.action_component_limit,
            action_rejected=action_rejected,
            explicit_abort=explicit_abort,
            source_step=terminal.recovery_step,
        ).values
        event = _build_runtime_event(
            header=self._header,
            event_index=terminal.event_index,
            event_type=RuntimeEventType.TERMINAL,
            recovery_step=terminal.recovery_step,
            total_transition_count=terminal.total_transition_count,
            simulation_time=terminal.simulation_time,
            proposed_action=terminal.proposed_action,
            executed_action=None,
            action_disposition=terminal.action_disposition,
            transition_executed=False,
            monitor_decision=None,
            pre_state_hash=state_hash,
            predicted_state_hash=None,
            realized_state_hash=None,
            pre_observation=observation,
            predicted_observation=None,
            post_observation=None,
            progress_sample=_not_evaluated_progress(terminal.recovery_step),
            action_geometry=action_geometry,
            prediction_diagnostics=_prediction_diagnostics(
                None, None, None, None, source_step=terminal.recovery_step
            ),
            runtime_evidence=terminal.runtime_evidence,
            terminal_reason=terminal.terminal_reason,
            evidence_level=terminal.evidence_level,
            volatile_timestamp=terminal.volatile_timestamp,
        )
        self._events.append(event)
        self._state = LoggerSessionState.TERMINAL
        self._last_state = state
        self._last_configuration = configuration
        self._last_state_hash = state_hash
        self._last_recovery_step = terminal.recovery_step
        self._last_total_transition_count = terminal.total_transition_count
        return event

    def finalize(
        self, *, volatile_finalization_timestamp: str | None = None
    ) -> StagedRecoveryTraceBundle:
        if self._state == LoggerSessionState.FINALIZED:
            raise RuntimeLoggerContractError("logger session cannot be finalized twice")
        if self._state != LoggerSessionState.TERMINAL and not self._capacity_exhausted:
            raise RuntimeLoggerContractError(
                "complete trace finalization requires terminal event"
            )
        if not self._events:
            raise RuntimeLoggerContractError("empty logger session cannot be finalized")
        events = tuple(self._events)
        aggregate_hash = aggregate_trace_sha256(events)
        jsonl_hash = hashlib.sha256(trace_jsonl_bytes_from_events(events)).hexdigest()
        terminal_status = (
            events[-1].terminal_reason
            if events[-1].event_type == RuntimeEventType.TERMINAL
            else "logger_capacity_exhausted_incomplete"
        )
        manifest = StagedRecoveryTraceManifest(
            trace_manifest_schema_version=TRACE_MANIFEST_SCHEMA_VERSION,
            logger_schema_version=LOGGER_SCHEMA_VERSION,
            instrumentation_schema_version=INSTRUMENTATION_SCHEMA_VERSION,
            architecture_version=ARCHITECTURE_VERSION,
            source_instrumentation_commit=SOURCE_INSTRUMENTATION_COMMIT,
            source_instrumentation_canonical_hash=SOURCE_INSTRUMENTATION_CANONICAL_HASH,
            source_architecture_commit=SOURCE_ARCHITECTURE_COMMIT,
            source_architecture_canonical_hash=SOURCE_ARCHITECTURE_CANONICAL_HASH,
            session_id=self._header.session_id,
            case_id=self._header.case_id,
            seed=self._header.seed,
            implementation_commit=self._header.implementation_commit,
            source_state_hash=self._header.source_state_hash,
            simulator_configuration_hash=self._header.simulator_configuration_hash,
            constants_hash=self._header.constants_hash,
            event_count=len(events),
            declared_maximum_event_count=self._header.max_events,
            first_event_index=events[0].event_index,
            last_event_index=events[-1].event_index,
            first_recovery_step=events[0].recovery_step,
            last_recovery_step=events[-1].recovery_step,
            first_total_transition_count=events[0].total_transition_count,
            last_total_transition_count=events[-1].total_transition_count,
            terminal_status=terminal_status or "terminal_reason_missing",
            event_scientific_hashes=tuple(
                event.canonical_event_sha256 for event in events
            ),
            aggregate_trace_hash=aggregate_hash,
            bundle_file_hashes=(),
            trace_classification=self._header.trace_classification,
            runtime_source=self._header.runtime_source,
            scientific_result=self._header.scientific_result,
            runtime_integration_classification=REAL_RUNNER_INTEGRATION_STATUS,
            claim_restrictions=self._header.scientific_claim_restrictions,
            volatile_finalization_timestamp=volatile_finalization_timestamp,
            canonical_manifest_payload_sha256="",
        )
        manifest_hash = canonical_trace_manifest_sha256(manifest)
        manifest = replace(
            manifest,
            bundle_file_hashes=(
                TraceFileHash(
                    filename=TRACE_JSONL_FILENAME,
                    sha256=jsonl_hash,
                    hash_scope="complete_file_bytes",
                ),
                TraceFileHash(
                    filename=TRACE_MANIFEST_FILENAME,
                    sha256=manifest_hash,
                    hash_scope=(
                        "canonical_payload_excluding_volatile_timestamp_self_hash_"
                        "and_bundle_file_hashes"
                    ),
                ),
            ),
            canonical_manifest_payload_sha256=manifest_hash,
        )
        bundle = StagedRecoveryTraceBundle(
            header=self._header,
            events=events,
            manifest=manifest,
        )
        report = validate_trace_bundle(bundle, require_complete=False)
        if not report.valid:
            raise RuntimeLoggerContractError(
                f"finalized trace is invalid: {', '.join(report.errors)}"
            )
        self._state = LoggerSessionState.FINALIZED
        return bundle


def event_document(event: StagedRecoveryRuntimeEvent) -> dict[str, object]:
    if not event_hash_recomputes(event):
        raise RuntimeLoggerContractError("event canonical hash mismatch")
    document = _to_json_value(event)
    assert isinstance(document, dict)
    return document


def trace_jsonl_bytes_from_events(
    events: Sequence[StagedRecoveryRuntimeEvent],
) -> bytes:
    return b"".join(
        canonical_runtime_json_bytes(event_document(event)) + b"\n"
        for event in events
    )


def trace_jsonl_bytes(bundle: StagedRecoveryTraceBundle) -> bytes:
    report = validate_trace_bundle(bundle, require_complete=False)
    if not report.valid:
        raise RuntimeLoggerContractError(
            f"cannot serialize invalid trace: {', '.join(report.errors)}"
        )
    return trace_jsonl_bytes_from_events(bundle.events)


def trace_manifest_json_bytes(bundle: StagedRecoveryTraceBundle) -> bytes:
    report = validate_trace_bundle(bundle, require_complete=False)
    if not report.valid:
        raise RuntimeLoggerContractError(
            f"cannot serialize invalid trace manifest: {', '.join(report.errors)}"
        )
    return (
        json.dumps(
            _to_json_value(bundle.manifest),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _validate_event_sequence(
    events: Sequence[StagedRecoveryRuntimeEvent],
    *,
    require_complete: bool,
) -> list[str]:
    errors: list[str] = []
    if not events:
        return ["empty_trace"]
    if events[0].event_type not in {
        RuntimeEventType.INITIAL_SNAPSHOT,
        RuntimeEventType.TERMINAL,
    }:
        errors.append("trace_does_not_start_with_initial_or_terminal")
    if events[0].event_type == RuntimeEventType.TERMINAL and len(events) != 1:
        errors.append("zero_transition_terminal_must_be_only_event")
    terminal_indices = [
        index
        for index, event in enumerate(events)
        if event.event_type == RuntimeEventType.TERMINAL
    ]
    if require_complete and terminal_indices != [len(events) - 1]:
        errors.append("complete_trace_requires_one_final_terminal_event")
    elif terminal_indices and terminal_indices != [len(events) - 1]:
        errors.append("terminal_event_must_be_last")
    for index, event in enumerate(events):
        if event.event_index != index:
            errors.append(f"nonsequential_event_index:{index}")
        if not event_hash_recomputes(event):
            errors.append(f"event_hash_mismatch:{index}")
        if index:
            previous = events[index - 1]
            recovery_delta = event.recovery_step - previous.recovery_step
            total_delta = (
                event.total_transition_count - previous.total_transition_count
            )
            expected = 1 if event.transition_executed else 0
            if recovery_delta != expected:
                errors.append(f"recovery_counter_mismatch:{index}")
            if total_delta != expected:
                errors.append(f"total_counter_mismatch:{index}")
            if previous.terminal:
                errors.append(f"event_after_terminal:{index}")
        if event.transition_executed:
            if event.event_type != RuntimeEventType.TRANSITION:
                errors.append(f"nontransition_event_executes_transition:{index}")
            if event.post_observation is None or event.realized_state_hash is None:
                errors.append(f"executed_transition_missing_realized_state:{index}")
        elif event.post_observation is not None or event.realized_state_hash is not None:
            errors.append(f"nonexecuted_event_has_realized_state:{index}")
        if event.event_type == RuntimeEventType.INITIAL_SNAPSHOT:
            if index != 0 or event.transition_executed:
                errors.append("invalid_initial_snapshot")
            if event.executed_action is not None or event.proposed_action is not None:
                errors.append("initial_snapshot_has_action")
        if event.event_type == RuntimeEventType.TERMINAL and event.transition_executed:
            errors.append(f"terminal_event_executes_transition:{index}")
    return errors


def validate_trace_bundle(
    bundle: StagedRecoveryTraceBundle,
    *,
    require_complete: bool = True,
) -> StagedRecoveryLoggerValidationReport:
    errors: list[str] = []
    if not isinstance(bundle, StagedRecoveryTraceBundle):
        return StagedRecoveryLoggerValidationReport(
            valid=False,
            errors=("bundle_type_invalid",),
            event_count=0,
            aggregate_trace_hash=None,
        )
    events = bundle.events
    errors.extend(_validate_event_sequence(events, require_complete=require_complete))
    header = bundle.header
    manifest = bundle.manifest
    if manifest.event_count != len(events):
        errors.append("manifest_event_count_mismatch")
    if len(events) > header.max_events:
        errors.append("header_event_capacity_exceeded")
    if manifest.declared_maximum_event_count != header.max_events:
        errors.append("manifest_event_capacity_mismatch")
    identity_checks = (
        (manifest.session_id, header.session_id, "session_id"),
        (manifest.case_id, header.case_id, "case_id"),
        (manifest.seed, header.seed, "seed"),
        (
            manifest.implementation_commit,
            header.implementation_commit,
            "implementation_commit",
        ),
        (manifest.source_state_hash, header.source_state_hash, "source_state_hash"),
        (
            manifest.simulator_configuration_hash,
            header.simulator_configuration_hash,
            "simulator_configuration_hash",
        ),
        (manifest.constants_hash, header.constants_hash, "constants_hash"),
    )
    for actual, expected, field_id in identity_checks:
        if actual != expected:
            errors.append(f"manifest_header_mismatch:{field_id}")
    for index, event in enumerate(events):
        for actual, expected, field_id in (
            (event.session_id, header.session_id, "session_id"),
            (event.case_id, header.case_id, "case_id"),
            (event.seed, header.seed, "seed"),
            (
                event.implementation_commit,
                header.implementation_commit,
                "implementation_commit",
            ),
            (event.source_state_hash, header.source_state_hash, "source_state_hash"),
            (
                event.simulator_configuration_hash,
                header.simulator_configuration_hash,
                "simulator_configuration_hash",
            ),
            (event.constants_hash, header.constants_hash, "constants_hash"),
        ):
            if actual != expected:
                errors.append(f"event_identity_mismatch:{index}:{field_id}")
    expected_hashes = tuple(event.canonical_event_sha256 for event in events)
    if manifest.event_scientific_hashes != expected_hashes:
        errors.append("manifest_event_hashes_mismatch")
    aggregate = aggregate_trace_sha256(events) if events else None
    if aggregate != manifest.aggregate_trace_hash:
        errors.append("aggregate_trace_hash_mismatch")
    if not trace_manifest_hash_recomputes(manifest):
        errors.append("trace_manifest_canonical_hash_mismatch")
    if events:
        boundary_checks = (
            (manifest.first_event_index, events[0].event_index, "first_event_index"),
            (manifest.last_event_index, events[-1].event_index, "last_event_index"),
            (
                manifest.first_recovery_step,
                events[0].recovery_step,
                "first_recovery_step",
            ),
            (
                manifest.last_recovery_step,
                events[-1].recovery_step,
                "last_recovery_step",
            ),
            (
                manifest.first_total_transition_count,
                events[0].total_transition_count,
                "first_total_transition_count",
            ),
            (
                manifest.last_total_transition_count,
                events[-1].total_transition_count,
                "last_total_transition_count",
            ),
        )
        for actual, expected, field_id in boundary_checks:
            if actual != expected:
                errors.append(f"manifest_boundary_mismatch:{field_id}")
    file_hashes = {item.filename: item for item in manifest.bundle_file_hashes}
    if set(file_hashes) != set(TRACE_BUNDLE_FILENAMES):
        errors.append("trace_bundle_file_hash_set_mismatch")
    else:
        jsonl_hash = hashlib.sha256(trace_jsonl_bytes_from_events(events)).hexdigest()
        if (
            file_hashes[TRACE_JSONL_FILENAME].sha256 != jsonl_hash
            or file_hashes[TRACE_JSONL_FILENAME].hash_scope != "complete_file_bytes"
        ):
            errors.append("trace_jsonl_hash_mismatch")
        if (
            file_hashes[TRACE_MANIFEST_FILENAME].sha256
            != manifest.canonical_manifest_payload_sha256
        ):
            errors.append("trace_manifest_scoped_hash_mismatch")
    if require_complete and (
        not events or events[-1].event_type != RuntimeEventType.TERMINAL
    ):
        errors.append("trace_is_not_complete")
    return StagedRecoveryLoggerValidationReport(
        valid=not errors,
        errors=tuple(sorted(set(errors))),
        event_count=len(events),
        aggregate_trace_hash=aggregate,
    )


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _is_within(candidate: Path, parent: Path) -> bool:
    candidate_text = _normalized_path(candidate)
    parent_text = _normalized_path(parent)
    try:
        return os.path.commonpath((candidate_text, parent_text)) == parent_text
    except ValueError:
        return False


def protected_trace_paths(repository_root: Path) -> tuple[Path, ...]:
    root = repository_root.resolve()
    protected = {(root / relative).resolve() for relative in PROTECTED_STATIC_RELATIVE_PATHS}
    analysis = root / "analysis"
    if analysis.is_dir():
        for child in analysis.iterdir():
            if child.is_dir() and child.name.lower().startswith(
                ("phase34", "phase35", "phase36", "phase37")
            ):
                protected.add(child.resolve())
    return tuple(sorted(protected, key=lambda path: _normalized_path(path)))


def validate_trace_publication_target(
    target_directory: Path | str,
    *,
    repository_root: Path | str,
) -> Path:
    raw = Path(target_directory)
    if ".." in raw.parts:
        raise RuntimeLoggerContractError("publication target may not contain path traversal")
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise RuntimeLoggerContractError("repository root does not exist")
    target = raw.resolve()
    if target == root:
        raise RuntimeLoggerContractError("repository root is not a trace target")
    if target.exists():
        raise RuntimeLoggerContractError("trace target already exists; overwrite refused")
    if not target.parent.is_dir():
        raise RuntimeLoggerContractError("trace target parent must already exist")
    for ancestor in (target.parent, *target.parent.parents):
        if ancestor.exists() and ancestor.is_symlink():
            resolved = ancestor.resolve()
            if any(
                resolved == protected or _is_within(resolved, protected)
                for protected in protected_trace_paths(root)
            ):
                raise RuntimeLoggerContractError(
                    "publication symlink resolves into protected location"
                )
    for protected in protected_trace_paths(root):
        if target == protected or _is_within(target, protected):
            raise RuntimeLoggerContractError(
                f"trace target overlaps protected path: {protected}"
            )
    return target


def _write_new_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_staged_trace_directory(
    directory: Path,
    bundle: StagedRecoveryTraceBundle,
) -> tuple[tuple[str, str], ...]:
    names = tuple(sorted(path.name for path in directory.iterdir() if path.is_file()))
    if names != tuple(sorted(TRACE_BUNDLE_FILENAMES)):
        raise RuntimeLoggerContractError("staged trace bundle is incomplete")
    expected = {
        TRACE_MANIFEST_FILENAME: trace_manifest_json_bytes(bundle),
        TRACE_JSONL_FILENAME: trace_jsonl_bytes(bundle),
    }
    hashes: list[tuple[str, str]] = []
    for filename in TRACE_BUNDLE_FILENAMES:
        actual = (directory / filename).read_bytes()
        if actual != expected[filename]:
            raise RuntimeLoggerContractError(f"staged artifact bytes changed: {filename}")
        hashes.append((filename, hashlib.sha256(actual).hexdigest()))
    return tuple(hashes)


def publish_trace_bundle(
    bundle: StagedRecoveryTraceBundle,
    target_directory: Path | str,
    *,
    repository_root: Path | str,
    failure_injector: Callable[[str], None] | None = None,
) -> StagedRecoveryTraceWriteResult:
    report = validate_trace_bundle(bundle, require_complete=True)
    if not report.valid:
        raise RuntimeLoggerContractError(
            f"invalid complete trace bundle: {', '.join(report.errors)}"
        )
    target = validate_trace_publication_target(
        target_directory, repository_root=repository_root
    )
    staging: Path | None = None
    published_target = False
    try:
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
        )
        if failure_injector:
            failure_injector("before_write_manifest")
        _write_new_file(
            staging / TRACE_MANIFEST_FILENAME, trace_manifest_json_bytes(bundle)
        )
        if failure_injector:
            failure_injector("before_write_jsonl")
        _write_new_file(staging / TRACE_JSONL_FILENAME, trace_jsonl_bytes(bundle))
        if failure_injector:
            failure_injector("before_staged_validation")
        staged_hashes = _validate_staged_trace_directory(staging, bundle)
        if target.exists():
            raise RuntimeLoggerContractError("trace target appeared during staging")
        if failure_injector:
            failure_injector("before_atomic_publish")
        os.replace(staging, target)
        staging = None
        published_target = True
        final_hashes = tuple(
            (filename, hashlib.sha256((target / filename).read_bytes()).hexdigest())
            for filename in TRACE_BUNDLE_FILENAMES
        )
        if final_hashes != staged_hashes:
            raise RuntimeLoggerContractError("published trace hashes differ from staged hashes")
        return StagedRecoveryTraceWriteResult(
            published=True,
            target_directory=str(target),
            artifact_paths=tuple(str(target / name) for name in TRACE_BUNDLE_FILENAMES),
            artifact_hashes=final_hashes,
            event_count=len(bundle.events),
            aggregate_trace_hash=bundle.manifest.aggregate_trace_hash,
        )
    except Exception:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        if published_target and target.exists():
            shutil.rmtree(target)
        raise


LOGGER_COVERAGE_CLASSIFICATIONS = (
    "logged_from_direct_runtime_input",
    "derived_by_stage0a_during_logging",
    "requires_previous_runtime_state",
    "requires_predicted_runtime_state",
    "requires_phase_runtime",
    "requires_future_evaluator",
    "unsupported",
)

_STAGE0A_TO_LOGGER_COVERAGE = {
    "direct_input_supported": "logged_from_direct_runtime_input",
    "pure_derivation_supported": "derived_by_stage0a_during_logging",
    "requires_previous_state": "requires_previous_runtime_state",
    "requires_predicted_state": "requires_predicted_runtime_state",
    "requires_runtime_phase_integration": "requires_phase_runtime",
    "requires_future_evaluator": "requires_future_evaluator",
    "not_yet_supported": "unsupported",
}


def logger_field_coverage() -> tuple[dict[str, object], ...]:
    entries = []
    for definition in field_catalog():
        classification = _STAGE0A_TO_LOGGER_COVERAGE[
            definition.support_classification
        ]
        entries.append(
            {
                "field_id": definition.field_id,
                "stage0a_support_classification": definition.support_classification,
                "logger_coverage_classification": classification,
                "schema_accepts": True,
                "logger_derives": classification
                == "derived_by_stage0a_during_logging",
                "future_runtime_must_supply": classification
                in {
                    "logged_from_direct_runtime_input",
                    "requires_phase_runtime",
                    "requires_future_evaluator",
                },
                "real_trace_has_validated": False,
                "scientific_limitation": (
                    "Schema acceptance or pure derivability does not establish that "
                    "this field has been observed in a measured runtime trace."
                ),
            }
        )
    return tuple(entries)


def logger_coverage_counts() -> tuple[tuple[str, int], ...]:
    coverage = logger_field_coverage()
    return tuple(
        (
            classification,
            sum(
                entry["logger_coverage_classification"] == classification
                for entry in coverage
            ),
        )
        for classification in LOGGER_COVERAGE_CLASSIFICATIONS
    )


_EVENT_SCHEMA_FIELD_ROWS = (
    ("schema_version", EVENT_TYPE_ORDER, "string", "schema", "direct", "logger contract"),
    ("session_id", EVENT_TYPE_ORDER, "string", "identifier", "direct", "session header"),
    ("event_index", EVENT_TYPE_ORDER, "integer", "event", "direct", "caller sequence"),
    ("event_type", EVENT_TYPE_ORDER, "enum", "categorical", "direct", "caller API"),
    ("case_id", EVENT_TYPE_ORDER, "string", "identifier", "direct", "session header"),
    ("seed", EVENT_TYPE_ORDER, "integer", "seed", "direct", "session header"),
    ("implementation_commit", EVENT_TYPE_ORDER, "string", "git_commit", "direct", "session header"),
    ("source_state_hash", EVENT_TYPE_ORDER, "string", "sha256", "direct", "session header"),
    ("simulator_configuration_hash", EVENT_TYPE_ORDER, "string", "sha256", "direct", "session header"),
    ("constants_hash", EVENT_TYPE_ORDER, "string", "sha256", "direct", "session header"),
    ("recovery_step", EVENT_TYPE_ORDER, "integer", "transition", "direct", "caller counter"),
    ("total_transition_count", EVENT_TYPE_ORDER, "integer", "transition", "direct", "caller counter"),
    ("simulation_time", EVENT_TYPE_ORDER, "number_or_null", "s", "direct", "caller snapshot"),
    ("proposed_action", ("transition", "terminal"), "vector2_or_null", "normalized_action", "direct", "caller decision"),
    ("executed_action", ("transition", "terminal"), "vector2_or_null", "normalized_action", "direct", "caller execution evidence"),
    ("action_disposition", EVENT_TYPE_ORDER, "enum", "categorical", "direct", "caller disposition"),
    ("transition_executed", EVENT_TYPE_ORDER, "boolean", "boolean", "direct", "caller execution evidence"),
    ("monitor_decision", ("transition",), "string_or_null", "categorical", "direct", "caller monitor evidence"),
    ("pre_state_hash", EVENT_TYPE_ORDER, "string_or_null", "sha256", "derived", "explicit Cartesian state"),
    ("predicted_state_hash", ("transition",), "string_or_null", "sha256", "derived", "explicit predicted state"),
    ("realized_state_hash", ("transition",), "string_or_null", "sha256", "derived", "explicit realized state"),
    ("pre_observation", EVENT_TYPE_ORDER, "instrumentation_record_or_null", "mixed", "derived", "Stage 0A"),
    ("predicted_observation", ("transition",), "predicted_record_or_null", "mixed", "derived", "Stage 0A"),
    ("post_observation", ("transition",), "instrumentation_record_or_null", "mixed", "derived", "Stage 0A"),
    ("progress_sample", EVENT_TYPE_ORDER, "instrumented_value_map", "mixed", "derived", "Stage 0A measured pre/post only"),
    ("action_geometry", EVENT_TYPE_ORDER, "instrumented_value_map", "mixed", "derived", "Stage 0A explicit action and pre-state basis"),
    ("prediction_diagnostics", EVENT_TYPE_ORDER, "instrumented_value_map", "mixed", "derived", "predicted and measured states"),
    ("phase_evidence", EVENT_TYPE_ORDER, "instrumented_value_map", "mixed", "direct", "future phase runtime"),
    ("evaluator_evidence", EVENT_TYPE_ORDER, "instrumented_value_map", "mixed", "direct", "external evaluators"),
    ("terminal_reason", ("terminal",), "string_or_null", "categorical", "direct", "caller terminal evidence"),
    ("terminal", EVENT_TYPE_ORDER, "boolean", "boolean", "derived", "event type"),
    ("evidence_level", EVENT_TYPE_ORDER, "string", "categorical", "direct", "caller provenance"),
    ("event_valid", EVENT_TYPE_ORDER, "boolean", "boolean", "derived", "structural/evidence consistency"),
    ("invalid_reasons", EVENT_TYPE_ORDER, "string_array", "categorical", "derived", "consistency validation"),
    ("volatile_timestamp", EVENT_TYPE_ORDER, "string_or_null", "timestamp", "direct", "optional caller provenance"),
    ("canonical_event_sha256", EVENT_TYPE_ORDER, "string", "sha256", "derived", "canonical scientific payload"),
)


def _document_with_hash(payload: dict[str, object]) -> dict[str, object]:
    document = dict(payload)
    document["canonical_payload_hash"] = canonical_runtime_sha256(document)
    return document


def canonical_document_hash_is_valid(document: Mapping[str, object]) -> bool:
    payload = dict(document)
    stored = payload.pop("canonical_payload_hash", None)
    return isinstance(stored, str) and stored == canonical_runtime_sha256(payload)


def event_schema_document() -> dict[str, object]:
    fields_document = []
    for index, (field_id, event_types, data_type, units, ownership, source) in enumerate(
        _EVENT_SCHEMA_FIELD_ROWS
    ):
        fields_document.append(
            {
                "field_id": field_id,
                "event_types_allowed": list(event_types),
                "data_type": data_type,
                "units": units,
                "nullable_behavior": (
                    "null only when unavailable under the event contract; unknown is never zero or false"
                ),
                "evidence_status": (
                    "nested physical and evaluator fields use Stage 0A evidence statuses"
                ),
                "direct_or_derived": ownership,
                "required_source_input": source,
                "canonical_order": index,
                "missing_value_rule": "preserve null/not_evaluated; never infer",
                "invalid_value_rule": "preserve invalid reason; never repair silently",
                "scientific_limitation": (
                    "This observational field does not establish action correctness, "
                    "phase readiness, stop priority, recovery, or safety."
                ),
            }
        )
    return _document_with_hash(
        {
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "logger_id": LOGGER_ID,
            "completed_date": COMPLETED_DATE,
            "event_types": list(EVENT_TYPE_ORDER),
            "action_dispositions": list(ACTION_DISPOSITION_VOCABULARY),
            "logical_transition_order": [
                "pre_transition_measured_state",
                "proposed_action_or_explicit_no_action",
                "supplied_monitor_decision",
                "supplied_predicted_next_state_when_available",
                "supplied_executed_action_or_rejection_or_suppression",
                "supplied_transition_executed_flag",
                "supplied_realized_next_state_when_executed",
                "pure_stage0a_post_state_derivation",
                "supplied_evaluator_and_phase_provenance",
            ],
            "fields": fields_document,
        }
    )


def integration_contract_document() -> dict[str, object]:
    common_forbidden = [
        "missing runtime events",
        "controller actions",
        "fallback actions",
        "phase decisions",
        "stop-condition selection",
        "Recovery Success decisions",
        "physical state reconstructed from a hash",
    ]
    return _document_with_hash(
        {
            "integration_contract_schema_version": INTEGRATION_CONTRACT_SCHEMA_VERSION,
            "logger_id": LOGGER_ID,
            "completed_date": COMPLETED_DATE,
            "required_call_order": [
                "create_session",
                "record_initial_snapshot",
                "record_zero_or_more_transitions",
                "record_terminal_event",
                "finalize_bundle",
            ],
            "zero_transition_terminal_call_order": [
                "create_session",
                "record_terminal_event_with_initial_state",
                "finalize_bundle",
            ],
            "event_contracts": [
                {
                    "event_type": "initial_snapshot",
                    "caller_owned_fields": [
                        "measured current state",
                        "configuration",
                        "runtime counters",
                        "phase and evaluator evidence when available",
                    ],
                    "logger_derived_fields": [
                        "state hash",
                        "Stage 0A current observation",
                        "canonical event hash",
                    ],
                    "optional_fields": ["simulation time", "phase evidence", "evaluator evidence"],
                    "forbidden_inferred_fields": common_forbidden,
                    "missing_evidence_allowed": True,
                    "structurally_valid_when_missing": True,
                    "requires_stage0c_real_validation": True,
                },
                {
                    "event_type": "transition",
                    "caller_owned_fields": [
                        "measured pre-state",
                        "proposed action",
                        "monitor decision",
                        "predicted state when available",
                        "executed action or disposition",
                        "transition-executed flag",
                        "measured realized state when executed",
                        "runtime counters",
                        "phase and evaluator evidence",
                    ],
                    "logger_derived_fields": [
                        "pre observation",
                        "predicted hazard observation",
                        "post observation",
                        "threshold-free progress",
                        "action geometry",
                        "prediction diagnostics",
                        "canonical event hash",
                    ],
                    "optional_fields": ["predicted state", "simulation times", "phase evidence"],
                    "forbidden_inferred_fields": common_forbidden,
                    "missing_evidence_allowed": True,
                    "structurally_valid_when_missing": (
                        "only when the transition/action/counter contract does not require it"
                    ),
                    "requires_stage0c_real_validation": True,
                },
                {
                    "event_type": "terminal",
                    "caller_owned_fields": [
                        "terminal reason",
                        "final measured state when needed",
                        "unchanged runtime counters",
                        "terminal evaluator and phase evidence",
                    ],
                    "logger_derived_fields": [
                        "final observation",
                        "terminal structural validation",
                        "canonical event hash",
                    ],
                    "optional_fields": ["final measured state after an initialized session"],
                    "forbidden_inferred_fields": common_forbidden,
                    "missing_evidence_allowed": True,
                    "structurally_valid_when_missing": (
                        "yes for optional evaluator/phase evidence; no for terminal reason or required initial state"
                    ),
                    "requires_stage0c_real_validation": True,
                },
            ],
            "failure_behavior": {
                "ordering_or_counter_error": "reject before append",
                "capacity_error": "reject extra event and preserve accepted in-memory records",
                "contradictory_evaluator_evidence": "retain raw evidence and mark event invalid",
                "publication_error": "publish nothing and remove only task-owned staging directory",
                "missing_event": "do not infer or repair",
            },
            "future_candidate_hook_points": [
                "before action decision",
                "after prediction",
                "after action disposition",
                "after realized transition",
                "at terminal decision",
            ],
            "real_runner_integration": REAL_RUNNER_INTEGRATION_STATUS,
            "real_trace_validation": REAL_TRACE_VALIDATION_STATUS,
            "staged_controller_integration": "not_implemented",
            "staged_recovery_execution": STAGED_EXECUTION_STATUS,
        }
    )


def field_coverage_document() -> dict[str, object]:
    architecture = architecture_signal_coverage()
    return _document_with_hash(
        {
            "field_coverage_schema_version": FIELD_COVERAGE_SCHEMA_VERSION,
            "logger_id": LOGGER_ID,
            "completed_date": COMPLETED_DATE,
            "architecture_signal_count": len(architecture),
            "stage0a_field_count": len(field_catalog()),
            "logger_coverage_counts": [list(item) for item in logger_coverage_counts()],
            "architecture_signal_coverage": [
                {
                    "field_id": field_id,
                    "stage0a_support_classification": classification,
                    "logger_coverage_classification": _STAGE0A_TO_LOGGER_COVERAGE[
                        classification
                    ],
                    "real_trace_has_validated": False,
                }
                for field_id, classification in architecture
            ],
            "stage0a_field_coverage": list(logger_field_coverage()),
            "schema_accepts_count": len(field_catalog()),
            "real_trace_validated_count": 0,
            "real_trace_has_validated": False,
            "interpretation_boundary": (
                "Schema acceptance and pure derivability do not imply that a future "
                "runtime supplies the field or that it has appeared in measured evidence."
            ),
        }
    )


def logger_manifest_payload() -> dict[str, object]:
    return {
        "logger_id": LOGGER_ID,
        "schema_version": LOGGER_MANIFEST_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "trace_manifest_schema_version": TRACE_MANIFEST_SCHEMA_VERSION,
        "completed_date": COMPLETED_DATE,
        "source_stage0a": {
            "commit": SOURCE_INSTRUMENTATION_COMMIT,
            "instrumentation_schema_version": INSTRUMENTATION_SCHEMA_VERSION,
            "canonical_hash": SOURCE_INSTRUMENTATION_CANONICAL_HASH,
        },
        "source_architecture": {
            "commit": SOURCE_ARCHITECTURE_COMMIT,
            "architecture_version": ARCHITECTURE_VERSION,
            "canonical_hash": SOURCE_ARCHITECTURE_CANONICAL_HASH,
        },
        "source_artifact_hashes": [
            {"path": path, "sha256": digest}
            for path, digest in (
                SOURCE_INSTRUMENTATION_ARTIFACT_HASHES
                + SOURCE_ARCHITECTURE_ARTIFACT_HASHES
                + LOGGING_SEMANTIC_SOURCES
            )
        ],
        "event_types": list(EVENT_TYPE_ORDER),
        "event_ordering_states": list(SESSION_STATE_ORDER),
        "action_disposition_vocabulary": list(ACTION_DISPOSITION_VOCABULARY),
        "canonical_event_order": [
            row[0] for row in _EVENT_SCHEMA_FIELD_ROWS
        ],
        "trace_bundle_filenames": list(TRACE_BUNDLE_FILENAMES),
        "bounded_capacity_rule": (
            "caller must explicitly supply a positive max_events; no scientific horizon default exists"
        ),
        "atomic_publication_rule": (
            "validate complete bundle, stage both files in a sibling directory, then publish by same-filesystem directory rename"
        ),
        "protected_path_policy_version": "staged_recovery_trace_protected_paths_v0",
        "protected_path_classes": [
            "Phase34-37 evidence",
            "Final Veto evidence",
            "frozen and published recovery evidence",
            "mechanism diagnosis",
            "staged architecture",
            "Stage 0A instrumentation",
            "Stage 0B contract artifacts",
            "controller, simulator, and runtime code",
            "repository root",
        ],
        "field_coverage_summary": [list(item) for item in logger_coverage_counts()],
        "unresolved_runtime_dependencies": [
            "authorized real runner hook points",
            "measured trace validation",
            "phase actions",
            "numerical phase guards",
            "no-progress thresholds",
            "hysteresis parameters",
            "handoff-readiness evaluator",
            "available-correction-authority evaluator",
        ],
        "runtime_logging_boundary": RUNTIME_LOGGING_BOUNDARY_STATUS,
        "synthetic_trace_validation": SYNTHETIC_TRACE_VALIDATION_STATUS,
        "real_runner_integration": REAL_RUNNER_INTEGRATION_STATUS,
        "real_trace_validation": REAL_TRACE_VALIDATION_STATUS,
        "staged_recovery_execution": STAGED_EXECUTION_STATUS,
        "execution_not_authorized_reason": EXECUTION_NOT_AUTHORIZED_REASON,
        "claim_restrictions": list(CLAIM_RESTRICTIONS),
        "canonicalization": {
            "encoding": "UTF-8",
            "json_keys": "sorted",
            "json_separators": [",", ":"],
            "event_self_hash_excluded": True,
            "volatile_timestamp_excluded_from_scientific_hash": True,
        },
    }


def logger_manifest_document() -> dict[str, object]:
    return _document_with_hash(logger_manifest_payload())


def validate_logger_contract_documents(
    manifest: Mapping[str, object],
    event_schema: Mapping[str, object],
    integration_contract: Mapping[str, object],
    field_coverage: Mapping[str, object],
) -> StagedRecoveryLoggerValidationReport:
    errors: list[str] = []
    expected_documents = (
        ("manifest", dict(manifest), logger_manifest_document()),
        ("event_schema", dict(event_schema), event_schema_document()),
        (
            "integration_contract",
            dict(integration_contract),
            integration_contract_document(),
        ),
        ("field_coverage", dict(field_coverage), field_coverage_document()),
    )
    for name, supplied, expected in expected_documents:
        if not canonical_document_hash_is_valid(supplied):
            errors.append(f"{name}_canonical_hash_mismatch")
        if supplied != expected:
            errors.append(f"{name}_contract_drift")
    if manifest.get("real_runner_integration") != REAL_RUNNER_INTEGRATION_STATUS:
        errors.append("real_runner_integration_must_remain_not_implemented")
    if manifest.get("real_trace_validation") != REAL_TRACE_VALIDATION_STATUS:
        errors.append("real_trace_validation_must_remain_not_performed")
    if manifest.get("staged_recovery_execution") != STAGED_EXECUTION_STATUS:
        errors.append("staged_recovery_execution_must_remain_not_authorized")
    if field_coverage.get("real_trace_validated_count") != 0:
        errors.append("measured_trace_validation_claim_is_prohibited")
    return StagedRecoveryLoggerValidationReport(
        valid=not errors,
        errors=tuple(sorted(set(errors))),
        event_count=0,
        aggregate_trace_hash=None,
    )


__all__ = [
    "ACTION_DISPOSITION_VOCABULARY",
    "ARCHITECTURE_VERSION",
    "ActionDisposition",
    "COMPLETED_DATE",
    "EVENT_SCHEMA_VERSION",
    "EVENT_TYPE_ORDER",
    "EXECUTION_NOT_AUTHORIZED_REASON",
    "FIELD_COVERAGE_SCHEMA_VERSION",
    "INTEGRATION_CONTRACT_SCHEMA_VERSION",
    "LOGGER_ID",
    "LOGGER_MANIFEST_SCHEMA_VERSION",
    "LOGGER_SCHEMA_VERSION",
    "LoggerSessionState",
    "REAL_RUNNER_INTEGRATION_STATUS",
    "REAL_TRACE_VALIDATION_STATUS",
    "RuntimeEventType",
    "RuntimeLoggerContractError",
    "SOURCE_ARCHITECTURE_CANONICAL_HASH",
    "SOURCE_INSTRUMENTATION_CANONICAL_HASH",
    "STAGED_EXECUTION_STATUS",
    "StagedRecoveryInitialSnapshot",
    "StagedRecoveryLoggerValidationReport",
    "StagedRecoveryPredictedObservation",
    "StagedRecoveryRuntimeEvent",
    "StagedRecoveryRuntimeLoggerSession",
    "StagedRecoverySessionHeader",
    "StagedRecoveryTerminalInput",
    "StagedRecoveryTraceBundle",
    "StagedRecoveryTraceManifest",
    "StagedRecoveryTraceWriteResult",
    "StagedRecoveryTransitionInput",
    "TRACE_BUNDLE_FILENAMES",
    "TRACE_JSONL_FILENAME",
    "TRACE_MANIFEST_FILENAME",
    "TRACE_MANIFEST_SCHEMA_VERSION",
    "aggregate_trace_sha256",
    "canonical_document_hash_is_valid",
    "canonical_event_sha256",
    "canonical_runtime_json_bytes",
    "canonical_runtime_sha256",
    "canonical_state_sha256",
    "canonical_trace_manifest_sha256",
    "event_document",
    "event_hash_recomputes",
    "event_schema_document",
    "field_coverage_document",
    "integration_contract_document",
    "logger_coverage_counts",
    "logger_field_coverage",
    "logger_manifest_document",
    "protected_trace_paths",
    "publish_trace_bundle",
    "trace_jsonl_bytes",
    "trace_manifest_hash_recomputes",
    "trace_manifest_json_bytes",
    "validate_logger_contract_documents",
    "validate_trace_bundle",
    "validate_trace_publication_target",
    "with_canonical_event_hash",
]
