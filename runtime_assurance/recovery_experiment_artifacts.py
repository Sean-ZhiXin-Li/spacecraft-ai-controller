from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence


ARTIFACT_SCHEMA_VERSION = "recovery_experiment_artifacts_v0"
DECISION_EVENT_SCHEMA_VERSION = "recovery_decision_event_v0"
STOP_PRIORITY_VERSION = "recovery_stop_priority_v0"

RESULTS_FILENAME = "results.csv"
DECISION_LOG_FILENAME = "decision_log.jsonl"
SUMMARY_FILENAME = "summary.md"
COMPARISON_FILENAME = "comparison.png"
PUBLISHED_ARTIFACT_FILENAMES = (
    RESULTS_FILENAME,
    DECISION_LOG_FILENAME,
    SUMMARY_FILENAME,
    COMPARISON_FILENAME,
)

EVALUATION_STATUSES = frozenset(
    {"triggered", "clear", "not_evaluated", "invalid"}
)
STRUCTURALLY_COMPLETE_EVALUATION_STATUSES = frozenset({"triggered", "clear"})
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

EXPECTED_BRANCH_TERMINAL_LABELS = {
    "invalid_simulation": "invalid_simulation",
    "invalid_recovery_evaluation": "invalid_recovery_evaluation",
    "overspeed": "overspeed",
    "instability": "instability",
    "unsafe_state": "unsafe_state",
    "action_rejected": "recovery_action_rejected",
    "explicit_abort": "explicit_recovery_abort",
    "recovery_success": "recovery_success",
    "recovery_horizon_exhausted": "recovery_horizon_exhausted",
    "total_horizon_exhausted": "total_horizon_exhausted",
}

EXPECTED_CONTROLLED_TERMINAL_LABELS = {
    "invalid_simulation": "invalid_simulation",
    "invalid_recovery_evaluation": "unknown_with_manual_audit",
    "overspeed": "overspeed",
    "instability": "instability",
    "unsafe_state": "unsafe_state",
    "action_rejected": "unknown_with_manual_audit",
    "explicit_abort": "unknown_with_manual_audit",
    "recovery_success": "success",
    "recovery_horizon_exhausted": "timeout",
    "total_horizon_exhausted": "timeout",
}


class RecoveryArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryArtifactContract:
    experiment_id: str
    case_id: str
    seed: int
    branch_ids: tuple[str, ...]
    branch_state_hash: str
    manifest_hash: str
    implementation_commit: str
    case_configuration_hash: str
    simulator_configuration_hash: str
    simulator_constants_hash: str
    branch_step: int
    nominal_prefix_transition_count: int
    recovery_horizon: int
    total_horizon: int
    hazard_threshold: float
    hazard_comparator: str
    stop_priority: tuple[str, ...]
    output_directory: str
    output_filenames: tuple[str, ...]
    protected_paths: tuple[str, ...]
    prohibited_claims: tuple[str, ...]
    artifact_schema_version: str = ARTIFACT_SCHEMA_VERSION
    stop_priority_version: str = STOP_PRIORITY_VERSION


@dataclass(frozen=True, slots=True)
class RecoveryBranchExperimentRecord:
    artifact_schema_version: str
    experiment_id: str
    case_id: str
    seed: int
    branch_id: str
    branch_state_hash: str
    manifest_hash: str
    implementation_commit: str
    case_configuration_hash: str
    simulator_configuration_hash: str
    simulator_constants_hash: str
    branch_step: int
    nominal_prefix_transition_count: int
    recovery_transition_count: int
    total_transition_count: int
    recovery_horizon: int
    total_horizon: int
    hazard_threshold: float
    hazard_comparator: str
    stop_priority: tuple[str, ...]
    stop_priority_version: str
    terminal_reason: str
    branch_terminal_label: str
    controlled_terminal_label: str
    recovery_outcome: str
    valid: bool
    overspeed_status: str
    instability_status: str
    unsafe_state_status: str
    invalid_simulation_status: str
    invalid_recovery_evaluation_status: str
    crossed_target_radius: bool | None
    first_crossing_step: int | None
    phase34_compatible_recoverable_crossing: bool | None
    first_recoverable_crossing_step: int | None
    recovery_success_status: str
    recovery_success: bool | None
    recovery_success_step: int | None
    final_simulator_success: bool | None
    overspeed_headroom: float | None
    action_saturation_margin: float | None
    available_correction_authority: float | None
    required_to_available_correction_ratio: float | None
    normalized_control_effort: float | None
    delta_v_proxy: float | None
    recovery_steps: int | None
    crossing_delay: int | None
    final_radius_error: float | None
    final_radial_velocity_error: float | None
    final_tangential_velocity_error: float | None
    task_abandonment_status: bool | None
    monitor_evaluation_count: int | None
    intervention_count: int | None
    allow_count: int | None
    veto_count: int | None
    intervention_rate: float | None
    first_intervention_step: int | None
    last_intervention_step: int | None
    longest_intervention_streak: int | None
    veto_segment_count: int | None
    action_suppression_duration: int | None
    recovery_action_rejection_count: int | None
    action_rejected: bool
    rejected_action_executed: bool
    physical_transition_executed: bool
    evaluator_versions: tuple[str, ...]
    result_payload_hash: str


@dataclass(frozen=True, slots=True)
class RecoveryDecisionEvent:
    decision_event_schema_version: str
    experiment_id: str
    case_id: str
    seed: int
    branch_id: str
    branch_state_hash: str
    event_index: int
    post_branch_step: int
    total_transition_count: int
    proposed_action: tuple[float, float] | None
    final_veto_decision: str
    executed_action: tuple[float, float] | None
    transition_occurred: bool
    current_state_hash: str
    predicted_next_state_hash: str | None
    next_state_hash: str | None
    predicted_speed_ratio: float | None
    realized_speed_ratio: float | None
    hazard_threshold: float
    hazard_comparator: str
    triggered_stop_condition: str | None
    evaluator_statuses: tuple[tuple[str, str], ...]
    terminal_reason: str | None
    evidence_level: str


@dataclass(frozen=True, slots=True)
class RecoveryExperimentBundle:
    records: tuple[RecoveryBranchExperimentRecord, ...]
    decision_events: tuple[RecoveryDecisionEvent, ...]
    is_synthetic: bool
    comparison_scope: str = "one_case_nonformal_diagnostic"


@dataclass(frozen=True, slots=True)
class RecoveryBundleValidationReport:
    valid: bool
    errors: tuple[str, ...]
    record_count: int
    event_count: int


@dataclass(frozen=True, slots=True)
class RecoveryArtifactWriteResult:
    published: bool
    target_directory: str
    artifact_paths: tuple[str, ...]
    artifact_hashes: tuple[tuple[str, str], ...]
    synthetic: bool


RESULT_FIELDNAMES = tuple(
    name
    for name in RecoveryBranchExperimentRecord.__dataclass_fields__
)


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RecoveryArtifactError(
            f"value is not canonical-JSON serializable: {exc}"
        ) from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def record_payload(
    record: RecoveryBranchExperimentRecord,
    *,
    include_hash: bool = False,
) -> dict[str, object]:
    payload = asdict(record)
    if not include_hash:
        payload.pop("result_payload_hash", None)
    return payload


def recompute_record_payload_hash(record: RecoveryBranchExperimentRecord) -> str:
    return canonical_sha256(record_payload(record))


def with_record_payload_hash(
    record: RecoveryBranchExperimentRecord,
) -> RecoveryBranchExperimentRecord:
    return replace(record, result_payload_hash=recompute_record_payload_hash(record))


def event_payload(event: RecoveryDecisionEvent) -> dict[str, object]:
    return asdict(event)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_nonnegative_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _optional_nonnegative_integer(value: object) -> bool:
    return value is None or _is_nonnegative_integer(value)


def _optional_finite_number(value: object) -> bool:
    return value is None or (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_recovery_experiment_bundle(
    bundle: RecoveryExperimentBundle,
    contract: RecoveryArtifactContract,
) -> RecoveryBundleValidationReport:
    errors: list[str] = []
    records = bundle.records
    branch_ids = tuple(record.branch_id for record in records)
    if branch_ids != contract.branch_ids:
        errors.append(
            "branch records must use the frozen branch order exactly: "
            f"{contract.branch_ids!r}"
        )
    if len(branch_ids) != len(set(branch_ids)):
        errors.append("branch records contain a duplicate branch ID")
    undeclared = sorted(set(branch_ids) - set(contract.branch_ids))
    if undeclared:
        errors.append(f"branch records contain undeclared branches: {undeclared}")
    missing = sorted(set(contract.branch_ids) - set(branch_ids))
    if missing:
        errors.append(f"branch records are missing frozen branches: {missing}")

    status_fields = (
        "overspeed_status",
        "instability_status",
        "unsafe_state_status",
        "invalid_simulation_status",
        "invalid_recovery_evaluation_status",
        "recovery_success_status",
    )
    optional_integer_fields = (
        "first_crossing_step",
        "first_recoverable_crossing_step",
        "recovery_success_step",
        "recovery_steps",
        "crossing_delay",
        "monitor_evaluation_count",
        "intervention_count",
        "allow_count",
        "veto_count",
        "first_intervention_step",
        "last_intervention_step",
        "longest_intervention_streak",
        "veto_segment_count",
        "action_suppression_duration",
        "recovery_action_rejection_count",
    )
    optional_numeric_fields = (
        "overspeed_headroom",
        "action_saturation_margin",
        "available_correction_authority",
        "required_to_available_correction_ratio",
        "normalized_control_effort",
        "delta_v_proxy",
        "final_radius_error",
        "final_radial_velocity_error",
        "final_tangential_velocity_error",
        "intervention_rate",
    )

    for record in records:
        prefix = f"branch {record.branch_id!r}"
        common_checks = (
            (record.artifact_schema_version, contract.artifact_schema_version, "artifact schema"),
            (record.experiment_id, contract.experiment_id, "experiment ID"),
            (record.case_id, contract.case_id, "case ID"),
            (record.seed, contract.seed, "seed"),
            (record.branch_state_hash, contract.branch_state_hash, "branch-state hash"),
            (record.manifest_hash, contract.manifest_hash, "manifest hash"),
            (record.implementation_commit, contract.implementation_commit, "implementation commit"),
            (record.case_configuration_hash, contract.case_configuration_hash, "case configuration hash"),
            (record.simulator_configuration_hash, contract.simulator_configuration_hash, "simulator configuration hash"),
            (record.simulator_constants_hash, contract.simulator_constants_hash, "simulator constants hash"),
            (record.branch_step, contract.branch_step, "branch step"),
            (record.nominal_prefix_transition_count, contract.nominal_prefix_transition_count, "nominal-prefix transition count"),
            (record.recovery_horizon, contract.recovery_horizon, "recovery horizon"),
            (record.total_horizon, contract.total_horizon, "total horizon"),
            (record.hazard_threshold, contract.hazard_threshold, "hazard threshold"),
            (record.hazard_comparator, contract.hazard_comparator, "hazard comparator"),
            (record.stop_priority, contract.stop_priority, "stop priority"),
            (record.stop_priority_version, contract.stop_priority_version, "stop-priority version"),
        )
        for actual, expected, name in common_checks:
            if actual != expected:
                errors.append(f"{prefix} has inconsistent {name}")
        if not record.valid:
            errors.append(f"{prefix} is structurally invalid")
        if record.terminal_reason not in contract.stop_priority:
            errors.append(f"{prefix} has an undeclared terminal reason")
        if not record.branch_terminal_label:
            errors.append(f"{prefix} lacks a branch terminal label")
        if not record.controlled_terminal_label:
            errors.append(f"{prefix} lacks a controlled terminal label")
        if not record.recovery_outcome:
            errors.append(f"{prefix} lacks a recovery outcome")
        expected_branch_label = EXPECTED_BRANCH_TERMINAL_LABELS.get(
            record.terminal_reason
        )
        if record.branch_terminal_label != expected_branch_label:
            errors.append(f"{prefix} has an inconsistent branch terminal label")
        expected_controlled_label = EXPECTED_CONTROLLED_TERMINAL_LABELS.get(
            record.terminal_reason
        )
        if record.controlled_terminal_label != expected_controlled_label:
            errors.append(f"{prefix} has an inconsistent controlled terminal label")
        if not _is_nonnegative_integer(record.recovery_transition_count):
            errors.append(f"{prefix} has an invalid recovery transition count")
        elif record.recovery_transition_count > contract.recovery_horizon:
            errors.append(f"{prefix} exceeds the frozen recovery horizon")
        if not _is_nonnegative_integer(record.total_transition_count):
            errors.append(f"{prefix} has an invalid total transition count")
        elif record.total_transition_count > contract.total_horizon:
            errors.append(f"{prefix} exceeds the frozen total horizon")
        if (
            _is_nonnegative_integer(record.recovery_transition_count)
            and record.total_transition_count
            != record.nominal_prefix_transition_count
            + record.recovery_transition_count
        ):
            errors.append(f"{prefix} transition counts are not internally consistent")
        for field in status_fields:
            status = getattr(record, field)
            if status not in EVALUATION_STATUSES:
                errors.append(f"{prefix} has invalid {field}: {status!r}")
            elif status not in STRUCTURALLY_COMPLETE_EVALUATION_STATUSES:
                errors.append(f"{prefix} required evaluator {field} is unavailable")
        for field in optional_integer_fields:
            if not _optional_nonnegative_integer(getattr(record, field)):
                errors.append(f"{prefix} has malformed optional integer {field}")
        for field in optional_numeric_fields:
            if not _optional_finite_number(getattr(record, field)):
                errors.append(f"{prefix} has malformed optional numeric {field}")
        if record.intervention_rate is not None and not (
            0.0 <= record.intervention_rate <= 1.0
        ):
            errors.append(f"{prefix} intervention rate is outside [0, 1]")
        if (
            record.monitor_evaluation_count is not None
            and record.allow_count is not None
            and record.veto_count is not None
            and record.allow_count + record.veto_count
            != record.monitor_evaluation_count
        ):
            errors.append(f"{prefix} monitor counts are inconsistent")
        if (
            record.intervention_count is not None
            and record.veto_count is not None
            and record.intervention_count != record.veto_count
        ):
            errors.append(f"{prefix} intervention and veto counts differ")
        if record.recovery_success_status == "triggered" and record.recovery_success is not True:
            errors.append(f"{prefix} recovery success boolean conflicts with evaluator")
        if record.recovery_success_status == "clear" and record.recovery_success is not False:
            errors.append(f"{prefix} recovery success boolean conflicts with evaluator")
        if record.action_rejected and record.rejected_action_executed:
            errors.append(f"{prefix} reports execution of a rejected action")
        if record.action_rejected and record.physical_transition_executed:
            errors.append(f"{prefix} reports a transition after action rejection")
        if record.action_rejected and record.terminal_reason != "action_rejected":
            errors.append(f"{prefix} action rejection has the wrong terminal reason")
        if record.branch_id == "explicit_abort_v0":
            if record.recovery_transition_count != 0 or record.physical_transition_executed:
                errors.append("explicit_abort_v0 must report zero physical transitions")
            if record.terminal_reason != "explicit_abort":
                errors.append("explicit_abort_v0 must terminate as explicit_abort")
        if not _is_sha256(record.result_payload_hash):
            errors.append(f"{prefix} lacks a valid result payload hash")
        elif record.result_payload_hash != recompute_record_payload_hash(record):
            errors.append(f"{prefix} result payload hash does not recompute")

    event_keys: list[tuple[int, int, int]] = []
    events_by_branch = {branch_id: 0 for branch_id in contract.branch_ids}
    for event in bundle.decision_events:
        if event.branch_id not in contract.branch_ids:
            errors.append(f"decision event uses undeclared branch {event.branch_id!r}")
            continue
        branch_index = contract.branch_ids.index(event.branch_id)
        event_keys.append((branch_index, event.post_branch_step, event.event_index))
        events_by_branch[event.branch_id] += 1
        if event.decision_event_schema_version != DECISION_EVENT_SCHEMA_VERSION:
            errors.append(f"decision event for {event.branch_id!r} has schema drift")
        if event.experiment_id != contract.experiment_id or event.case_id != contract.case_id:
            errors.append(f"decision event for {event.branch_id!r} has identity drift")
        if event.seed != contract.seed or event.branch_state_hash != contract.branch_state_hash:
            errors.append(f"decision event for {event.branch_id!r} has provenance drift")
        if event.hazard_threshold != contract.hazard_threshold or event.hazard_comparator != contract.hazard_comparator:
            errors.append(f"decision event for {event.branch_id!r} has hazard-contract drift")
        if (
            not _is_nonnegative_integer(event.event_index)
            or not _is_nonnegative_integer(event.post_branch_step)
            or not _is_nonnegative_integer(event.total_transition_count)
        ):
            errors.append(f"decision event for {event.branch_id!r} has invalid ordering")
        if event.final_veto_decision not in {"allow", "veto", "not_applicable"}:
            errors.append(f"decision event for {event.branch_id!r} has invalid veto decision")
        if not event.current_state_hash or (
            event.transition_occurred and not event.next_state_hash
        ):
            errors.append(f"decision event for {event.branch_id!r} lacks state hashes")
        if event.final_veto_decision != "not_applicable" and not event.predicted_next_state_hash:
            errors.append(
                f"decision event for {event.branch_id!r} lacks a predicted state hash"
            )
        evaluator_keys = tuple(key for key, _ in event.evaluator_statuses)
        if evaluator_keys != tuple(sorted(evaluator_keys)) or len(evaluator_keys) != len(set(evaluator_keys)):
            errors.append(f"decision event for {event.branch_id!r} has nondeterministic evaluator ordering")
        for _, status in event.evaluator_statuses:
            if status not in EVALUATION_STATUSES:
                errors.append(f"decision event for {event.branch_id!r} has invalid evaluator status")
    if event_keys != sorted(event_keys) or len(event_keys) != len(set(event_keys)):
        errors.append("decision events are not in deterministic unique branch/step/index order")
    for branch_id, count in events_by_branch.items():
        if count == 0:
            errors.append(f"decision log has no event for branch {branch_id!r}")

    return RecoveryBundleValidationReport(
        valid=not errors,
        errors=tuple(errors),
        record_count=len(records),
        event_count=len(bundle.decision_events),
    )


def _encode_csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise RecoveryArtifactError("CSV values must be finite")
    return value


def results_csv_bytes(
    bundle: RecoveryExperimentBundle,
    contract: RecoveryArtifactContract,
) -> bytes:
    report = validate_recovery_experiment_bundle(bundle, contract)
    if not report.valid:
        raise RecoveryArtifactError("invalid recovery bundle: " + "; ".join(report.errors))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=RESULT_FIELDNAMES,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for record in bundle.records:
        writer.writerow(
            {
                field: _encode_csv_value(getattr(record, field))
                for field in RESULT_FIELDNAMES
            }
        )
    return stream.getvalue().encode("utf-8")


def decision_log_jsonl_bytes(
    bundle: RecoveryExperimentBundle,
    contract: RecoveryArtifactContract,
) -> bytes:
    report = validate_recovery_experiment_bundle(bundle, contract)
    if not report.valid:
        raise RecoveryArtifactError("invalid recovery bundle: " + "; ".join(report.errors))
    return b"".join(
        canonical_json_bytes(event_payload(event)) + b"\n"
        for event in bundle.decision_events
    )


def _status_count(
    records: Sequence[RecoveryBranchExperimentRecord], field: str, value: str
) -> int:
    return sum(getattr(record, field) == value for record in records)


def summary_markdown_bytes(
    bundle: RecoveryExperimentBundle,
    contract: RecoveryArtifactContract,
) -> bytes:
    report = validate_recovery_experiment_bundle(bundle, contract)
    if not report.valid:
        raise RecoveryArtifactError("invalid recovery bundle: " + "; ".join(report.errors))
    records = bundle.records
    lines = [
        "# Recovery Action Branching Nonformal v0 Summary",
        "",
        "## Structural Validity",
        "",
        "- Four-branch common-state bundle: valid",
        f"- Branch records: {len(records)}",
        f"- Decision events: {len(bundle.decision_events)}",
        f"- Synthetic fixture: {'yes' if bundle.is_synthetic else 'no'}",
        "- Scope: one-case nonformal diagnostic",
        f"- Common branch-state hash: `{contract.branch_state_hash}`",
        f"- Manifest hash: `{contract.manifest_hash}`",
        "",
        "## Hazard Outcomes",
        "",
        f"- Overspeed triggered: {_status_count(records, 'overspeed_status', 'triggered')}",
        f"- Instability triggered: {_status_count(records, 'instability_status', 'triggered')}",
        f"- Unsafe state triggered: {_status_count(records, 'unsafe_state_status', 'triggered')}",
        f"- Invalid simulation: {_status_count(records, 'invalid_simulation_status', 'triggered')}",
        f"- Invalid recovery evaluation: {_status_count(records, 'invalid_recovery_evaluation_status', 'triggered')}",
        "",
        "## State And Task Recovery",
        "",
        f"- Target-radius crossing: {sum(record.crossed_target_radius is True for record in records)}",
        f"- Phase34-compatible recoverable crossing: {sum(record.phase34_compatible_recoverable_crossing is True for record in records)}",
        f"- Recovery Success v0: {sum(record.recovery_success is True for record in records)}",
        f"- Final simulator success: {sum(record.final_simulator_success is True for record in records)}",
        "",
        "| Branch | Overspeed | Crossing | Recoverable crossing | Recovery Success v0 | Simulator success | Terminal | Recovery outcome |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| `{record.branch_id}` | `{record.overspeed_status}` | "
            f"`{record.crossed_target_radius}` | "
            f"`{record.phase34_compatible_recoverable_crossing}` | "
            f"`{record.recovery_success}` | `{record.final_simulator_success}` | "
            f"`{record.branch_terminal_label}` | `{record.recovery_outcome}` |"
        )
    lines.extend(
        [
        "",
        "## Cost And Intervention Burden",
        "",
        ]
    )
    for record in records:
        lines.append(
            f"- `{record.branch_id}`: recovery_steps={record.recovery_steps!r}, "
            f"control_effort={record.normalized_control_effort!r}, "
            f"delta_v_proxy={record.delta_v_proxy!r}, "
            f"evaluations={record.monitor_evaluation_count!r}, "
            f"allows={record.allow_count!r}, vetoes={record.veto_count!r}, "
            f"intervention_rate={record.intervention_rate!r}, "
            f"crossing_delay={record.crossing_delay!r}"
        )
    lines.extend(
        [
            "",
            "## Failure Mechanisms",
            "",
        ]
    )
    for record in records:
        lines.append(f"- `{record.branch_id}`: `{record.terminal_reason}`")
    lines.extend(
        [
            "",
            "## Non-Claims",
            "",
            "This one-case diagnostic does not establish formal safety, universal "
            "recovery, controller superiority, benchmark-wide effectiveness, "
            "cross-case generalization, hardware validity, deployment readiness, "
            "or cross-embodiment validation. A complete bundle is only structurally "
            "comparable and does not imply recovery success.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _atomic_write_new(path: Path, payload: bytes) -> Path:
    destination = path.resolve()
    if destination.exists():
        raise RecoveryArtifactError(f"refusing to overwrite artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_results_csv(
    path: Path,
    bundle: RecoveryExperimentBundle,
    contract: RecoveryArtifactContract,
) -> Path:
    return _atomic_write_new(path, results_csv_bytes(bundle, contract))


def write_decision_log_jsonl(
    path: Path,
    bundle: RecoveryExperimentBundle,
    contract: RecoveryArtifactContract,
) -> Path:
    return _atomic_write_new(path, decision_log_jsonl_bytes(bundle, contract))


def write_summary_markdown(
    path: Path,
    bundle: RecoveryExperimentBundle,
    contract: RecoveryArtifactContract,
) -> Path:
    return _atomic_write_new(path, summary_markdown_bytes(bundle, contract))


def write_comparison_png(
    path: Path,
    bundle: RecoveryExperimentBundle,
    contract: RecoveryArtifactContract,
) -> Path:
    report = validate_recovery_experiment_bundle(bundle, contract)
    if not report.valid:
        raise RecoveryArtifactError("invalid recovery bundle: " + "; ".join(report.errors))
    destination = path.resolve()
    if destination.exists():
        raise RecoveryArtifactError(f"refusing to overwrite artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp.png", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        labels = [record.branch_id.replace("_v0", "") for record in bundle.records]
        hazard = [record.overspeed_status == "triggered" for record in bundle.records]
        crossing = [record.crossed_target_radius is True for record in bundle.records]
        recoverable = [
            record.phase34_compatible_recoverable_crossing is True
            for record in bundle.records
        ]
        recovery = [record.recovery_success is True for record in bundle.records]
        interventions = [
            float("nan")
            if record.intervention_count is None
            else record.intervention_count
            for record in bundle.records
        ]
        rates = [
            float("nan") if record.intervention_rate is None else record.intervention_rate
            for record in bundle.records
        ]
        positions = list(range(len(labels)))

        figure = Figure(figsize=(12.0, 9.0), dpi=100, constrained_layout=True)
        canvas = FigureCanvasAgg(figure)
        hazard_axis, recovery_axis, burden_axis = figure.subplots(3, 1)
        hazard_axis.bar(positions, hazard, color="#c9493f")
        hazard_axis.set_title("Declared overspeed outcome")
        hazard_axis.set_ylabel("Triggered")
        hazard_axis.set_ylim(0, 1.15)
        width = 0.24
        recovery_axis.bar(
            [position - width for position in positions],
            crossing,
            width=width,
            color="#4c78a8",
            label="Crossing",
        )
        recovery_axis.bar(
            positions,
            recoverable,
            width=width,
            color="#59a14f",
            label="Recoverable crossing",
        )
        recovery_axis.bar(
            [position + width for position in positions],
            recovery,
            width=width,
            color="#2f6b45",
            label="Recovery Success v0",
        )
        recovery_axis.set_title("State and task recovery outcomes")
        recovery_axis.set_ylabel("Triggered")
        recovery_axis.set_ylim(0, 1.15)
        recovery_axis.legend(fontsize=8, loc="upper right")
        burden_axis.bar(positions, interventions, color="#4c78a8", label="Interventions")
        burden_axis.set_title("Intervention burden (component metrics)")
        burden_axis.set_ylabel("Count")
        rate_axis = burden_axis.twinx()
        rate_axis.plot(positions, rates, color="#f28e2b", marker="o", label="Rate")
        rate_axis.set_ylabel("Intervention rate")
        rate_axis.set_ylim(0, 1.05)
        for index, record in enumerate(bundle.records):
            if record.intervention_count is None:
                burden_axis.text(index, 0.02, "not evaluated", rotation=90, fontsize=7)
            if record.intervention_rate is None:
                rate_axis.text(index, 0.02, "not evaluated", rotation=90, fontsize=7)
        for axis in (hazard_axis, recovery_axis, burden_axis):
            axis.set_xticks(positions, labels, rotation=15, ha="right", fontsize=8)
            axis.grid(axis="y", alpha=0.25)
        figure.suptitle(
            "Synthetic fixture" if bundle.is_synthetic else "One-case nonformal diagnostic",
            fontsize=12,
        )
        canvas.print_png(
            str(temporary),
            metadata={"Software": "spacecraft-ai-controller"},
        )
        figure.clear()
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_publication_target(
    target_directory: Path,
    *,
    repository_root: Path,
    contract: RecoveryArtifactContract,
    synthetic: bool,
) -> Path:
    target = target_directory.resolve()
    root = repository_root.resolve()
    for relative in contract.protected_paths:
        protected = (root / relative.rstrip("/")).resolve()
        if target == protected or _is_within(target, protected):
            raise RecoveryArtifactError(
                f"publication target overlaps protected path: {relative}"
            )
    frozen_output = (root / contract.output_directory.rstrip("/")).resolve()
    if synthetic and target == frozen_output:
        raise RecoveryArtifactError(
            "synthetic artifacts may not use the frozen recovery output directory"
        )
    for filename in contract.output_filenames:
        if (target / filename).exists():
            raise RecoveryArtifactError(
                f"refusing partial or overwrite publication; artifact exists: {filename}"
            )
    return target


def _validate_png(path: Path) -> None:
    data = path.read_bytes()
    if len(data) < 24 or not data.startswith(PNG_SIGNATURE) or data[12:16] != b"IHDR":
        raise RecoveryArtifactError(f"invalid PNG artifact: {path}")


def validate_staged_artifacts(
    directory: Path,
    bundle: RecoveryExperimentBundle,
    contract: RecoveryArtifactContract,
) -> tuple[tuple[str, str], ...]:
    report = validate_recovery_experiment_bundle(bundle, contract)
    if not report.valid:
        raise RecoveryArtifactError("invalid recovery bundle: " + "; ".join(report.errors))
    if tuple(contract.output_filenames) != PUBLISHED_ARTIFACT_FILENAMES:
        raise RecoveryArtifactError("reserved artifact filenames do not match v0 contract")
    hashes: list[tuple[str, str]] = []
    for filename in contract.output_filenames:
        path = directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise RecoveryArtifactError(f"staged artifact is missing or empty: {filename}")
        hashes.append((filename, _sha256_bytes(path.read_bytes())))

    with (directory / RESULTS_FILENAME).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(contract.branch_ids):
        raise RecoveryArtifactError("results.csv must contain exactly one row per branch")
    if tuple(row.get("branch_id") for row in rows) != contract.branch_ids:
        raise RecoveryArtifactError("results.csv branch order does not match the frozen order")

    decision_lines = (directory / DECISION_LOG_FILENAME).read_text(
        encoding="utf-8"
    ).splitlines()
    if len(decision_lines) != len(bundle.decision_events):
        raise RecoveryArtifactError("decision_log.jsonl event count changed during staging")
    for line in decision_lines:
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RecoveryArtifactError("decision log lines must be JSON objects")

    summary = (directory / SUMMARY_FILENAME).read_text(encoding="utf-8")
    if "## Structural Validity" not in summary or "## Non-Claims" not in summary:
        raise RecoveryArtifactError("summary.md lacks required scientific boundaries")
    _validate_png(directory / COMPARISON_FILENAME)
    return tuple(hashes)


def publish_recovery_experiment_bundle(
    bundle: RecoveryExperimentBundle,
    contract: RecoveryArtifactContract,
    target_directory: Path,
    *,
    repository_root: Path,
    failure_injector: Callable[[str], None] | None = None,
) -> RecoveryArtifactWriteResult:
    report = validate_recovery_experiment_bundle(bundle, contract)
    if not report.valid:
        raise RecoveryArtifactError("invalid recovery bundle: " + "; ".join(report.errors))
    target = validate_publication_target(
        target_directory,
        repository_root=repository_root,
        contract=contract,
        synthetic=bundle.is_synthetic,
    )
    if not target.parent.is_dir():
        raise RecoveryArtifactError("publication target parent must already exist")

    staging = Path(
        tempfile.mkdtemp(prefix=".recovery-experiment-stage-", dir=target.parent)
    )
    published: list[Path] = []
    target_created = False
    try:
        write_results_csv(staging / RESULTS_FILENAME, bundle, contract)
        if failure_injector:
            failure_injector("after_results")
        write_decision_log_jsonl(staging / DECISION_LOG_FILENAME, bundle, contract)
        if failure_injector:
            failure_injector("after_decision_log")
        write_summary_markdown(staging / SUMMARY_FILENAME, bundle, contract)
        if failure_injector:
            failure_injector("after_summary")
        write_comparison_png(staging / COMPARISON_FILENAME, bundle, contract)
        if failure_injector:
            failure_injector("after_plot")
        artifact_hashes = validate_staged_artifacts(staging, bundle, contract)
        if failure_injector:
            failure_injector("after_staged_validation")

        validate_publication_target(
            target,
            repository_root=repository_root,
            contract=contract,
            synthetic=bundle.is_synthetic,
        )
        if not target.exists():
            target.mkdir()
            target_created = True
        for filename in contract.output_filenames:
            destination = target / filename
            if destination.exists():
                raise RecoveryArtifactError(
                    f"artifact appeared during publication: {destination}"
                )
            if failure_injector:
                failure_injector(f"before_publish:{filename}")
            os.replace(staging / filename, destination)
            published.append(destination)
        final_hashes = tuple(
            (path.name, _sha256_bytes(path.read_bytes())) for path in published
        )
        if final_hashes != artifact_hashes:
            raise RecoveryArtifactError("published artifact hashes differ from staged hashes")
        return RecoveryArtifactWriteResult(
            published=True,
            target_directory=str(target),
            artifact_paths=tuple(str(path) for path in published),
            artifact_hashes=final_hashes,
            synthetic=bundle.is_synthetic,
        )
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        if target_created:
            try:
                target.rmdir()
            except OSError:
                pass
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "COMPARISON_FILENAME",
    "DECISION_EVENT_SCHEMA_VERSION",
    "DECISION_LOG_FILENAME",
    "PUBLISHED_ARTIFACT_FILENAMES",
    "RESULT_FIELDNAMES",
    "RESULTS_FILENAME",
    "STOP_PRIORITY_VERSION",
    "SUMMARY_FILENAME",
    "RecoveryArtifactContract",
    "RecoveryArtifactError",
    "RecoveryArtifactWriteResult",
    "RecoveryBranchExperimentRecord",
    "RecoveryBundleValidationReport",
    "RecoveryDecisionEvent",
    "RecoveryExperimentBundle",
    "canonical_json_bytes",
    "canonical_sha256",
    "decision_log_jsonl_bytes",
    "event_payload",
    "publish_recovery_experiment_bundle",
    "record_payload",
    "recompute_record_payload_hash",
    "results_csv_bytes",
    "summary_markdown_bytes",
    "validate_publication_target",
    "validate_recovery_experiment_bundle",
    "validate_staged_artifacts",
    "with_record_payload_hash",
    "write_comparison_png",
    "write_decision_log_jsonl",
    "write_results_csv",
    "write_summary_markdown",
]
