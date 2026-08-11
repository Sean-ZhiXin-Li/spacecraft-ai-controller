from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Sequence

from runtime_assurance.recovery_branch_state_registry import (
    load_branch_state_registry,
    load_registered_branch_state,
)
from runtime_assurance.staged_recovery_contract import EXECUTION_NOT_AUTHORIZED, RecoveryPhase
from runtime_assurance.staged_recovery_guard_evidence import (
    GuardAtomEvaluation,
    GuardEvidenceLevel,
    GuardEvidenceStatus,
)
from runtime_assurance.staged_recovery_shadow_guard import (
    SHADOW_OUTPUT_CONSUMED_BY_PHYSICAL_RUNTIME,
    ShadowGuardParameters,
    ShadowGuardResolution,
    ShadowPhaseMachine,
    ShadowTransitionRecord,
    allowed_shadow_edges,
    architecture_phase_ids,
    canonical_sha256,
    resolve_shadow_phase,
)
from runtime_assurance.staged_recovery_shadow_runtime import (
    EXPECTED_REGISTRY_AGGREGATE_HASH,
    EXPECTED_REGISTRY_MANIFEST_HASH,
    MAXIMUM_RECOVERY_TRANSITIONS,
    ShadowObservationAdapter,
    build_registered_runtime_identity,
    compare_physical_runs,
    run_registered_bounded_shadow_path,
)


CALIBRATION_ID = "staged_recovery_shadow_calibration_v0"
SCHEMA_VERSION = "staged_recovery_shadow_calibration_v0"
COMPLETED_DATE = "2026-08-11"
CONFIG_PATH = Path("configs/staged_recovery_shadow_calibration_v0.json")
TRACE_SET_OUTPUT_PATH = Path("analysis/staged_recovery_shadow_calibration_trace_set_v0")
CALIBRATION_OUTPUT_PATH = Path("analysis/staged_recovery_shadow_calibration_v0")
SMOKE_MANIFEST_PATH = Path("analysis/staged_recovery_shadow_smoke_v0/manifest.json")
EXPECTED_SMOKE_MANIFEST_HASH = (
    "c304a6ffdc22b418c5fb156d1a34e0eb67fc82f259cd961e3df8cb42565ab658"
)
PHYSICAL_BRANCHES = (
    "zero_action_reference_v0",
    "velocity_opposed_thrust_v0",
    "tangential_error_correction_v0",
)
EXPLICIT_ABORT_BRANCH = "explicit_abort_v0"
EXPECTED_TRACE_COUNT = 13
EXPECTED_PHYSICAL_EXECUTION_COUNT = 26
EXPECTED_CANDIDATE_COUNT = 216
EXPECTED_OFFLINE_REPLAY_COUNT = 2808
STUCK_RECOMMENDATION_RUN_LENGTH = 4
NO_PROGRESS_COMPONENTS = (
    "radius_gap",
    "radial_component",
    "absolute_tangential_error",
    "overspeed_headroom",
)
CLAIM_RESTRICTIONS = (
    "engineering shadow calibration only",
    "no controller or recovery improvement claim",
    "no optimality, formal safety, active threshold, handoff, or deployment claim",
)


class ShadowCalibrationError(RuntimeError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ShadowCalibrationError(f"{path.as_posix()} must contain a JSON object")
    return value


def config_canonical_hash(config: Mapping[str, object]) -> str:
    return canonical_sha256(dict(config))


def load_and_validate_config(repository_root: Path) -> dict[str, object]:
    config = _load_json(repository_root / CONFIG_PATH)
    expected = {
        "calibration_id": CALIBRATION_ID,
        "completed_date": COMPLETED_DATE,
        "maximum_recovery_transitions": MAXIMUM_RECOVERY_TRANSITIONS,
        "physical_branches": list(PHYSICAL_BRANCHES),
        "registry_aggregate_hash": EXPECTED_REGISTRY_AGGREGATE_HASH,
        "registry_manifest_hash": EXPECTED_REGISTRY_MANIFEST_HASH,
        "schema_version": "staged_recovery_shadow_calibration_config_v0",
        "shadow_only": True,
        "smoke_manifest_hash": EXPECTED_SMOKE_MANIFEST_HASH,
        "staged_recovery_execution": EXECUTION_NOT_AUTHORIZED,
        "trace_pair_count": EXPECTED_TRACE_COUNT,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ShadowCalibrationError(f"calibration configuration {field} mismatch")
    grid = config.get("grid")
    expected_grid = {
        "hazard_clear_consecutive_steps": [1, 2, 3],
        "maximum_shadow_transitions_per_trace": [8],
        "minimum_phase_dwell_steps": [1, 2, 4],
        "no_progress_consecutive_windows": [1, 2],
        "no_progress_required_component_count": [2, 3],
        "no_progress_window_length": [2, 4, 8],
        "transition_cooldown_steps": [0, 2],
    }
    if grid != expected_grid:
        raise ShadowCalibrationError("calibration parameter grid changed")
    return config


@dataclass(frozen=True, slots=True)
class CalibrationCandidate:
    candidate_id: str
    hazard_clear_consecutive_steps: int
    minimum_phase_dwell_steps: int
    no_progress_window_length: int
    no_progress_required_component_count: int
    no_progress_consecutive_windows: int
    transition_cooldown_steps: int
    maximum_shadow_transitions_per_trace: int

    def as_document(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def candidate_id(
    hazard_clear: int,
    dwell: int,
    window: int,
    required: int,
    consecutive: int,
    cooldown: int,
    budget: int,
) -> str:
    return (
        f"shadow_candidate_hc{hazard_clear}_d{dwell}_w{window}_r{required}"
        f"_n{consecutive}_cd{cooldown}_tb{budget}"
    )


def calibration_candidates(config: Mapping[str, object]) -> tuple[CalibrationCandidate, ...]:
    grid = config["grid"]
    if not isinstance(grid, Mapping):
        raise ShadowCalibrationError("grid must be a mapping")
    keys = (
        "hazard_clear_consecutive_steps",
        "minimum_phase_dwell_steps",
        "no_progress_window_length",
        "no_progress_required_component_count",
        "no_progress_consecutive_windows",
        "transition_cooldown_steps",
        "maximum_shadow_transitions_per_trace",
    )
    candidates = []
    for values in itertools.product(*(grid[key] for key in keys)):
        hc, dwell, window, required, consecutive, cooldown, budget = values
        candidates.append(
            CalibrationCandidate(
                candidate_id=candidate_id(*values),
                hazard_clear_consecutive_steps=hc,
                minimum_phase_dwell_steps=dwell,
                no_progress_window_length=window,
                no_progress_required_component_count=required,
                no_progress_consecutive_windows=consecutive,
                transition_cooldown_steps=cooldown,
                maximum_shadow_transitions_per_trace=budget,
            )
        )
    result = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    if len(result) != EXPECTED_CANDIDATE_COUNT or len({item.candidate_id for item in result}) != len(result):
        raise ShadowCalibrationError("candidate grid must contain exactly 216 unique IDs")
    return result


@dataclass(frozen=True, slots=True)
class CalibrationTraceDefinition:
    trace_id: str
    registry_member_id: str
    case_id: str
    branch_id: str
    explicit_abort: bool

    def as_document(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def trace_definitions(repository_root: Path) -> tuple[CalibrationTraceDefinition, ...]:
    registry = load_branch_state_registry(repository_root)
    if registry.canonical_manifest_hash != EXPECTED_REGISTRY_MANIFEST_HASH or len(registry.members) != 4:
        raise ShadowCalibrationError("frozen four-member registry identity mismatch")
    definitions = [
        CalibrationTraceDefinition(
            trace_id=f"{member.registry_member_id}__{branch}",
            registry_member_id=member.registry_member_id,
            case_id=member.case_id,
            branch_id=branch,
            explicit_abort=False,
        )
        for member in registry.members
        for branch in PHYSICAL_BRANCHES
    ]
    legacy = next((member for member in registry.members if member.legacy_member), None)
    if legacy is None:
        raise ShadowCalibrationError("legacy member is required for explicit-abort trace")
    definitions.append(
        CalibrationTraceDefinition(
            trace_id=f"{legacy.registry_member_id}__{EXPLICIT_ABORT_BRANCH}",
            registry_member_id=legacy.registry_member_id,
            case_id=legacy.case_id,
            branch_id=EXPLICIT_ABORT_BRANCH,
            explicit_abort=True,
        )
    )
    result = tuple(sorted(definitions, key=lambda item: item.trace_id))
    if len(result) != EXPECTED_TRACE_COUNT or len({item.trace_id for item in result}) != len(result):
        raise ShadowCalibrationError("frozen trace matrix must contain 13 unique definitions")
    return result


def guard_evaluation_document(item: GuardAtomEvaluation) -> dict[str, object]:
    return {
        "guard_atom_id": item.guard_atom_id,
        "status": item.status.value,
        "value": item.value,
        "evidence_level": item.evidence_level.value,
        "raw_source_values": list(item.raw_source_values),
        "comparator": item.comparator,
        "threshold_or_parameter_reference": item.threshold_or_parameter_reference,
        "reason": item.reason,
        "policy_authorization_status": item.policy_authorization_status,
    }


def guard_evaluation_from_document(value: Mapping[str, object]) -> GuardAtomEvaluation:
    raw = value.get("raw_source_values")
    if not isinstance(raw, list):
        raise ShadowCalibrationError("guard raw source values must be a list")
    return GuardAtomEvaluation(
        guard_atom_id=str(value["guard_atom_id"]),
        status=GuardEvidenceStatus(str(value["status"])),
        value=value.get("value"),
        evidence_level=GuardEvidenceLevel(str(value["evidence_level"])),
        raw_source_values=tuple((str(pair[0]), pair[1]) for pair in raw),
        comparator=str(value["comparator"]),
        threshold_or_parameter_reference=value.get("threshold_or_parameter_reference"),
        reason=str(value["reason"]),
        policy_authorization_status=str(value["policy_authorization_status"]),
    )


@dataclass(frozen=True, slots=True)
class CapturedCalibrationTrace:
    definition: CalibrationTraceDefinition
    transition_count: int
    terminal_reason: str
    equivalence: Mapping[str, object]
    records: tuple[dict[str, object], ...]


def capture_trace_pair(
    repository_root: Path,
    definition: CalibrationTraceDefinition,
    *,
    implementation_commit: str,
) -> CapturedCalibrationTrace:
    baseline_state = load_registered_branch_state(repository_root, definition.registry_member_id)
    baseline = run_registered_bounded_shadow_path(
        baseline_state,
        implementation_commit=implementation_commit,
        branch_id=definition.branch_id,
    )
    observed_state = load_registered_branch_state(repository_root, definition.registry_member_id)
    identity, _ = build_registered_runtime_identity(
        observed_state,
        implementation_commit=implementation_commit,
        branch_id=definition.branch_id,
    )
    adapter = ShadowObservationAdapter(identity, trace_id=definition.trace_id)
    observed = run_registered_bounded_shadow_path(
        observed_state,
        implementation_commit=implementation_commit,
        branch_id=definition.branch_id,
        observer=adapter,
    )
    equivalence = compare_physical_runs(baseline, observed)
    if equivalence.get("all_equivalence_checks") is not True:
        raise ShadowCalibrationError(f"physical equivalence failed for {definition.trace_id}")
    if not (
        len(adapter.records)
        == len(adapter.source_documents)
        == len(adapter.guard_evaluations)
        == len(observed.snapshots)
    ):
        raise ShadowCalibrationError("observer evidence sequence length mismatch")
    records = []
    for source, evaluations, shadow in zip(
        adapter.source_documents, adapter.guard_evaluations, adapter.records
    ):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "trace_id": definition.trace_id,
            "registry_member_id": definition.registry_member_id,
            "branch_id": definition.branch_id,
            "source_event": source,
            "guard_evaluations": [guard_evaluation_document(item) for item in evaluations],
            "observed_shadow_record": shadow.as_document(),
            "shadow_output_consumed_by_physical_runtime": False,
        }
        payload["canonical_trace_record_hash"] = canonical_sha256(payload)
        records.append(payload)
    return CapturedCalibrationTrace(
        definition=definition,
        transition_count=observed.recovery_transition_count,
        terminal_reason=observed.runtime_terminal_reason,
        equivalence=equivalence,
        records=tuple(records),
    )


def trace_jsonl_bytes(trace: CapturedCalibrationTrace) -> bytes:
    return b"\n".join(canonical_json_bytes(item) for item in trace.records) + b"\n"


def _validate_smoke(repository_root: Path) -> None:
    smoke = _load_json(repository_root / SMOKE_MANIFEST_PATH)
    supplied = smoke.pop("canonical_manifest_hash", None)
    if supplied != EXPECTED_SMOKE_MANIFEST_HASH or canonical_sha256(smoke) != supplied:
        raise ShadowCalibrationError("protected Stage 1B-A smoke manifest identity mismatch")


def build_trace_set_payloads(
    repository_root: Path,
    traces: Sequence[CapturedCalibrationTrace],
    *,
    implementation_commit: str,
) -> dict[str, bytes]:
    _validate_smoke(repository_root)
    ordered = tuple(sorted(traces, key=lambda item: item.definition.trace_id))
    if len(ordered) != EXPECTED_TRACE_COUNT or any(
        trace.equivalence.get("all_equivalence_checks") is not True for trace in ordered
    ):
        raise ShadowCalibrationError("trace set requires 13 equivalent pairs")
    trace_payloads = {
        f"traces/{trace.definition.trace_id}.jsonl": trace_jsonl_bytes(trace)
        for trace in ordered
    }
    trace_hashes = [sha256_bytes(trace_payloads[key]) for key in sorted(trace_payloads)]
    aggregate_hash = canonical_sha256(trace_hashes)
    index = {
        "schema_version": SCHEMA_VERSION,
        "traces": [
            {
                **trace.definition.as_document(),
                "event_count": len(trace.records),
                "transition_count": trace.transition_count,
                "terminal_reason": trace.terminal_reason,
                "trace_path": f"traces/{trace.definition.trace_id}.jsonl",
                "trace_sha256": sha256_bytes(
                    trace_payloads[f"traces/{trace.definition.trace_id}.jsonl"]
                ),
            }
            for trace in ordered
        ],
    }
    equivalence = {
        "schema_version": SCHEMA_VERSION,
        "pairs": [
            {**trace.definition.as_document(), **dict(trace.equivalence)} for trace in ordered
        ],
        "pair_count": EXPECTED_TRACE_COUNT,
        "physical_equivalence_failures": 0,
        "unauthorized_physical_effects": 0,
        "all_pairs_equivalent": True,
    }
    manifest = {
        "trace_set_id": f"{CALIBRATION_ID}_trace_set",
        "schema_version": SCHEMA_VERSION,
        "completed_date": COMPLETED_DATE,
        "implementation_commit": implementation_commit,
        "config_hash": config_canonical_hash(load_and_validate_config(repository_root)),
        "registry_manifest_hash": EXPECTED_REGISTRY_MANIFEST_HASH,
        "registry_aggregate_hash": EXPECTED_REGISTRY_AGGREGATE_HASH,
        "smoke_manifest_hash": EXPECTED_SMOKE_MANIFEST_HASH,
        "trace_count": EXPECTED_TRACE_COUNT,
        "pair_count": EXPECTED_TRACE_COUNT,
        "bounded_execution_count": EXPECTED_PHYSICAL_EXECUTION_COUNT,
        "baseline_traces_published": 0,
        "observed_traces_published": EXPECTED_TRACE_COUNT,
        "physical_equivalence_failures": 0,
        "unauthorized_physical_effects": 0,
        "automatic_retry": False,
        "trace_set_aggregate_hash": aggregate_hash,
        "shadow_output_consumed_by_physical_runtime": False,
        "staged_recovery_execution": EXECUTION_NOT_AUTHORIZED,
        "claim_restrictions": list(CLAIM_RESTRICTIONS),
        "artifact_filenames": sorted(
            ["trace_set_manifest.json", "trace_index.json", "equivalence_report.json", *trace_payloads]
        ),
    }
    manifest["canonical_manifest_hash"] = canonical_sha256(manifest)
    return {
        "trace_set_manifest.json": json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n",
        "trace_index.json": json.dumps(index, indent=2, sort_keys=True).encode() + b"\n",
        "equivalence_report.json": json.dumps(equivalence, indent=2, sort_keys=True).encode() + b"\n",
        **trace_payloads,
    }


def validate_trace_set_payloads(payloads: Mapping[str, bytes]) -> None:
    expected_top = {"trace_set_manifest.json", "trace_index.json", "equivalence_report.json"}
    if set(key for key in payloads if not key.startswith("traces/")) != expected_top:
        raise ShadowCalibrationError("trace-set top-level artifact set mismatch")
    if sum(key.startswith("traces/") for key in payloads) != EXPECTED_TRACE_COUNT:
        raise ShadowCalibrationError("trace set must contain exactly 13 observed traces")
    manifest = json.loads(payloads["trace_set_manifest.json"])
    supplied = manifest.pop("canonical_manifest_hash", None)
    if supplied != canonical_sha256(manifest):
        raise ShadowCalibrationError("trace-set manifest canonical hash mismatch")
    equivalence = json.loads(payloads["equivalence_report.json"])
    if equivalence.get("all_pairs_equivalent") is not True or equivalence.get("physical_equivalence_failures") != 0:
        raise ShadowCalibrationError("trace-set physical equivalence failed")
    index = json.loads(payloads["trace_index.json"])
    traces = index.get("traces")
    if not isinstance(traces, list) or len(traces) != EXPECTED_TRACE_COUNT:
        raise ShadowCalibrationError("trace index count mismatch")
    for item in traces:
        path = item["trace_path"]
        if sha256_bytes(payloads[path]) != item["trace_sha256"]:
            raise ShadowCalibrationError("trace file hash mismatch")
        rows = [json.loads(line) for line in payloads[path].splitlines() if line]
        if len(rows) != item["event_count"]:
            raise ShadowCalibrationError("trace event count mismatch")
        for row in rows:
            supplied_hash = row.pop("canonical_trace_record_hash", None)
            if supplied_hash != canonical_sha256(row):
                raise ShadowCalibrationError("trace record hash mismatch")


def atomic_publish_new_directory(
    repository_root: Path,
    relative_target: Path,
    payloads: Mapping[str, bytes],
    validator: Callable[[Mapping[str, bytes]], None],
) -> Path:
    validator(payloads)
    target = (repository_root / relative_target).resolve()
    analysis_root = (repository_root / "analysis").resolve()
    try:
        target.relative_to(analysis_root)
    except ValueError as exc:
        raise ShadowCalibrationError("publication target must remain under analysis") from exc
    if target.exists():
        raise ShadowCalibrationError("publication target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for relative, data in sorted(payloads.items()):
            path = staging / Path(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        staged = {
            path.relative_to(staging).as_posix(): path.read_bytes()
            for path in staging.rglob("*") if path.is_file()
        }
        validator(staged)
        os.replace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def load_trace_set(repository_root: Path) -> tuple[dict[str, object], tuple[tuple[dict[str, object], ...], ...]]:
    source = repository_root / TRACE_SET_OUTPUT_PATH
    payloads = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*") if path.is_file()
    }
    validate_trace_set_payloads(payloads)
    index = json.loads(payloads["trace_index.json"])
    trace_rows = []
    for item in index["traces"]:
        rows = tuple(json.loads(line) for line in payloads[item["trace_path"]].splitlines() if line)
        trace_rows.append(rows)
    return index, tuple(trace_rows)


def _pairs_to_map(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, list):
        return {}
    return {
        pair[0]: pair[1]
        for pair in value
        if isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str) and isinstance(pair[1], dict)
    }


def _current_fields(event: Mapping[str, object]) -> dict[str, dict[str, object]]:
    observation = event.get("post_observation") if event.get("event_type") == "transition" else event.get("pre_observation")
    if not isinstance(observation, Mapping):
        return {}
    return _pairs_to_map(observation.get("fields"))


def _number(fields: Mapping[str, Mapping[str, object]], field_id: str) -> float | None:
    evidence = fields.get(field_id)
    if not isinstance(evidence, Mapping) or evidence.get("valid") is not True:
        return None
    value = evidence.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _atom_status(evaluations: Sequence[GuardAtomEvaluation], atom_id: str) -> GuardEvidenceStatus | None:
    return next((item.status for item in evaluations if item.guard_atom_id == atom_id), None)


def _resolution_conditions(evaluations: Sequence[GuardAtomEvaluation]) -> tuple[str, ...]:
    conditions = []
    true = {item.guard_atom_id for item in evaluations if item.status == GuardEvidenceStatus.TRUE}
    false = {item.guard_atom_id for item in evaluations if item.status == GuardEvidenceStatus.FALSE}
    invalid = {item.guard_atom_id for item in evaluations if item.status == GuardEvidenceStatus.INVALID}
    if "explicit_abort_requested" in true:
        conditions.append(RecoveryPhase.EXPLICIT_ABORT.value)
    if invalid or not {"state_evidence_valid", "instrumentation_evaluation_valid"}.isdisjoint(false):
        conditions.append(RecoveryPhase.RETREAT.value)
    if not {"realized_overspeed", "predicted_overspeed"}.isdisjoint(true):
        conditions.append(RecoveryPhase.HAZARD_ARREST.value)
    if "phase34_compatible_recoverability_pass" in true:
        conditions.append(RecoveryPhase.RECOVERABILITY_VERIFICATION.value)
    if not {"recoverability_radius_component_pass", "recoverability_radial_velocity_component_pass"}.isdisjoint(false):
        conditions.append(RecoveryPhase.RADIAL_RECOMMITMENT.value)
    if "recoverability_tangential_velocity_component_pass" in false:
        conditions.append(RecoveryPhase.TANGENTIAL_ALIGNMENT.value)
    if "no_eligible_crossing" in true:
        conditions.append(RecoveryPhase.CROSSING_PREPARATION.value)
    return tuple(dict.fromkeys(conditions))


@dataclass(frozen=True, slots=True)
class CandidateReplayResult:
    candidate: CalibrationCandidate
    metrics: Mapping[str, object]
    per_trace_metrics: tuple[dict[str, object], ...]
    transition_records: tuple[dict[str, object], ...]
    conflicts: tuple[dict[str, object], ...]
    disqualified: bool
    disqualification_reasons: tuple[str, ...]
    replay_hash: str


def replay_candidate(
    candidate: CalibrationCandidate,
    trace_index: Mapping[str, object],
    trace_rows: Sequence[Sequence[Mapping[str, object]]],
) -> CandidateReplayResult:
    index_items = trace_index.get("traces")
    if not isinstance(index_items, list) or len(index_items) != len(trace_rows):
        raise ShadowCalibrationError("trace replay index mismatch")
    aggregate = {
        "inter_phase_transitions": 0, "holds": 0, "graph_blocks": 0,
        "two_cycles": 0, "three_cycles": 0, "rapid_reversals": 0,
        "repeated_transition_reasons": 0, "transition_budget_exhaustions": 0,
        "stuck_trace_count": 0, "unavailable_guard_count": 0,
        "unavailable_evidence_block_count": 0, "invalid_guard_count": 0,
        "guard_conflict_count": 0, "nominal_handoff_recommendation_count": 0,
        "retreat_recommendation_count": 0, "explicit_abort_recommendation_count": 0,
    }
    phase_entries = {phase: 0 for phase in architecture_phase_ids()}
    per_trace = []
    all_records = []
    all_conflicts = []
    disqualifiers: set[str] = set()
    parameters = ShadowGuardParameters(
        minimum_phase_dwell_steps=candidate.minimum_phase_dwell_steps,
        transition_cooldown_steps=candidate.transition_cooldown_steps,
        maximum_shadow_transitions_per_trace=candidate.maximum_shadow_transitions_per_trace,
    )
    for index_item, rows in zip(index_items, trace_rows):
        machine = ShadowPhaseMachine(parameters)
        clear_count = 0
        no_progress_count = 0
        state_history: list[dict[str, dict[str, object]]] = []
        local = {key: 0 for key in aggregate if key != "stuck_trace_count"}
        local_phases: set[str] = set()
        pending_phase: str | None = None
        pending_count = 0
        max_pending = 0
        for row in rows:
            source = row.get("source_event")
            raw_evaluations = row.get("guard_evaluations")
            if not isinstance(source, Mapping) or not isinstance(raw_evaluations, list):
                raise ShadowCalibrationError("trace replay record is malformed")
            evaluations = tuple(guard_evaluation_from_document(item) for item in raw_evaluations)
            base = resolve_shadow_phase(evaluations)
            conditions = _resolution_conditions(evaluations)
            conflict = len(conditions) > 1
            if conflict:
                local["guard_conflict_count"] += 1
                all_conflicts.append({
                    "candidate_id": candidate.candidate_id,
                    "trace_id": index_item["trace_id"],
                    "event_index": source.get("event_index"),
                    "conditions": list(conditions),
                    "winning_phase": base.desired_phase,
                })
            local["unavailable_guard_count"] += len(base.unavailable_guard_atoms)
            local["invalid_guard_count"] += len(base.invalid_guard_atoms)

            emergency = base.desired_phase in {
                RecoveryPhase.EXPLICIT_ABORT.value,
                RecoveryPhase.RETREAT.value,
                RecoveryPhase.HAZARD_ARREST.value,
            }
            realized_clear = _atom_status(evaluations, "realized_overspeed_clear")
            predicted_clear = _atom_status(evaluations, "predicted_overspeed_clear")
            if realized_clear == GuardEvidenceStatus.TRUE and predicted_clear == GuardEvidenceStatus.TRUE:
                clear_count += 1
            else:
                clear_count = 0

            fields = _current_fields(source)
            if source.get("event_type") in {"initial_snapshot", "transition"} and fields:
                state_history.append(fields)
            no_progress_evaluable = False
            no_progress_triggered = False
            component_results: dict[str, bool] = {}
            window = candidate.no_progress_window_length
            if source.get("event_type") == "transition" and len(state_history) > window:
                start, end = state_history[-window - 1], state_history[-1]
                values = {
                    "radius_gap": (_number(start, "absolute_target_radius_error"), _number(end, "absolute_target_radius_error"), "decrease"),
                    "radial_component": (_number(start, "radial_velocity_ratio"), _number(end, "radial_velocity_ratio"), "abs_decrease"),
                    "absolute_tangential_error": (_number(start, "tangential_velocity_error"), _number(end, "tangential_velocity_error"), "abs_decrease"),
                    "overspeed_headroom": (_number(start, "overspeed_headroom"), _number(end, "overspeed_headroom"), "increase"),
                }
                no_progress_evaluable = all(first is not None and last is not None for first, last, _ in values.values())
                if no_progress_evaluable:
                    for key, (first, last, direction) in values.items():
                        if direction == "increase":
                            improvement = last - first
                        elif direction == "abs_decrease":
                            improvement = abs(first) - abs(last)
                        else:
                            improvement = first - last
                        component_results[key] = improvement > 0.0
                    improved = sum(component_results.values())
                    if improved < candidate.no_progress_required_component_count:
                        no_progress_count += 1
                    else:
                        no_progress_count = 0
                    no_progress_triggered = no_progress_count >= candidate.no_progress_consecutive_windows
                else:
                    no_progress_count = 0

            resolution = base
            if not emergency and clear_count < candidate.hazard_clear_consecutive_steps:
                resolution = replace(
                    base,
                    desired_phase=RecoveryPhase.STABILIZATION_ASSESSMENT.value,
                    transition_reason="candidate_hazard_clear_consecutive_evidence_incomplete",
                    priority_reason="candidate_hazard_clear_consecutive_evidence_incomplete",
                )
            elif not emergency and no_progress_triggered:
                resolution = replace(
                    base,
                    desired_phase=RecoveryPhase.RETREAT.value,
                    transition_reason="candidate_no_progress_consecutive_windows_triggered",
                    priority_reason="candidate_no_progress_consecutive_windows_triggered",
                )
            record = machine.step(
                resolution,
                event_index=int(source["event_index"]),
                recovery_step=int(source["recovery_step"]),
                source_event_hash=str(source["canonical_event_sha256"]),
                terminal_event=source.get("event_type") == "terminal",
            )
            record_doc = record.as_document()
            record_doc.update({
                "candidate_id": candidate.candidate_id,
                "trace_id": index_item["trace_id"],
                "hazard_clear_count": clear_count,
                "no_progress_consecutive_count": no_progress_count,
                "no_progress_evaluable": no_progress_evaluable,
                "no_progress_component_results": component_results,
            })
            record_doc["canonical_candidate_record_hash"] = canonical_sha256(record_doc)
            all_records.append(record_doc)
            if record.shadow_transition_executed:
                local_phases.add(record.resulting_shadow_phase)
                phase_entries[record.resulting_shadow_phase] += 1
                if record.current_shadow_phase != "unassigned":
                    local["inter_phase_transitions"] += 1
            else:
                local["holds"] += 1
            local["graph_blocks"] += record.block_reason == "graph_edge_not_allowed"
            local["two_cycles"] += record.two_cycle_detected
            local["three_cycles"] += record.three_cycle_detected
            local["rapid_reversals"] += record.rapid_reversal_detected
            local["repeated_transition_reasons"] += record.repeated_transition_reason
            local["transition_budget_exhaustions"] += record.block_reason == "transition_budget_exhausted"
            local["unavailable_evidence_block_count"] += (
                record.transition_blocked
                and resolution.priority_reason == "candidate_hazard_clear_consecutive_evidence_incomplete"
                and predicted_clear not in {GuardEvidenceStatus.TRUE, GuardEvidenceStatus.FALSE}
            )
            local["nominal_handoff_recommendation_count"] += record.nominal_handoff_recommended
            local["retreat_recommendation_count"] += resolution.desired_phase == RecoveryPhase.RETREAT.value
            local["explicit_abort_recommendation_count"] += resolution.desired_phase == RecoveryPhase.EXPLICIT_ABORT.value

            if record.transition_blocked and resolution.desired_phase != record.resulting_shadow_phase:
                if pending_phase == resolution.desired_phase:
                    pending_count += 1
                else:
                    pending_phase, pending_count = resolution.desired_phase, 1
                max_pending = max(max_pending, pending_count)
            else:
                pending_phase, pending_count = None, 0

            if record.nominal_handoff_recommended:
                disqualifiers.add("nominal_handoff_without_external_readiness")
            if record.shadow_transition_executed and record.current_shadow_phase != "unassigned":
                if (record.current_shadow_phase, record.resulting_shadow_phase) not in allowed_shadow_edges():
                    disqualifiers.add("forbidden_graph_transition")
                if record.invalid_guard_atoms:
                    disqualifiers.add("invalid_guard_used_as_transition_evidence")
                if resolution.priority_reason == "stabilization_fallback_or_incomplete_evidence":
                    disqualifiers.add("unavailable_guard_used_as_positive_evidence")
            if record.transition_budget_used > candidate.maximum_shadow_transitions_per_trace:
                disqualifiers.add("transition_budget_overrun")
            if record.resulting_shadow_phase not in architecture_phase_ids():
                disqualifiers.add("invalid_architecture_phase")
            if record.shadow_output_consumed_by_physical_runtime:
                disqualifiers.add("shadow_output_consumed_by_physical_runtime")

        stuck = max_pending >= STUCK_RECOMMENDATION_RUN_LENGTH
        if stuck:
            aggregate["stuck_trace_count"] += 1
        for key, value in local.items():
            aggregate[key] += int(value)
        per_trace.append({
            "trace_id": index_item["trace_id"],
            **local,
            "phase_coverage": len(local_phases - {RecoveryPhase.NOMINAL_HANDOFF.value}),
            "stuck": stuck,
            "maximum_blocked_recommendation_run": max_pending,
        })

    aggregate["phase_entries"] = phase_entries
    aggregate["phase_coverage"] = sum(value > 0 for phase, value in phase_entries.items() if phase != RecoveryPhase.NOMINAL_HANDOFF.value)
    aggregate["trace_count"] = len(trace_rows)
    aggregate["event_count"] = len(all_records)
    replay_hash = canonical_sha256(all_records)
    return CandidateReplayResult(
        candidate=candidate,
        metrics=aggregate,
        per_trace_metrics=tuple(per_trace),
        transition_records=tuple(all_records),
        conflicts=tuple(all_conflicts),
        disqualified=bool(disqualifiers),
        disqualification_reasons=tuple(sorted(disqualifiers)),
        replay_hash=replay_hash,
    )


def ranking_tuple(result: CandidateReplayResult) -> tuple[object, ...]:
    m = result.metrics
    return (
        m["two_cycles"] + m["three_cycles"],
        m["rapid_reversals"],
        m["transition_budget_exhaustions"],
        m["invalid_guard_count"],
        m["unavailable_evidence_block_count"],
        m["stuck_trace_count"],
        m["graph_blocks"],
        m["guard_conflict_count"],
        -m["phase_coverage"],
        m["inter_phase_transitions"],
        result.candidate.candidate_id,
    )


def analyze_candidates(
    config: Mapping[str, object],
    trace_index: Mapping[str, object],
    trace_rows: Sequence[Sequence[Mapping[str, object]]],
) -> tuple[tuple[CandidateReplayResult, ...], CandidateReplayResult]:
    results = tuple(replay_candidate(candidate, trace_index, trace_rows) for candidate in calibration_candidates(config))
    eligible = tuple(item for item in results if not item.disqualified)
    if not eligible:
        raise ShadowCalibrationError("all calibration candidates are disqualified")
    selected = min(eligible, key=ranking_tuple)
    if len(results) * len(trace_rows) != EXPECTED_OFFLINE_REPLAY_COUNT:
        raise ShadowCalibrationError("offline replay count must be exactly 2808")
    return results, selected


def build_calibration_payloads(
    repository_root: Path,
    config: Mapping[str, object],
    trace_index: Mapping[str, object],
    trace_rows: Sequence[Sequence[Mapping[str, object]]],
    results: Sequence[CandidateReplayResult],
    selected: CandidateReplayResult,
    *,
    implementation_commit: str,
    trace_set_commit: str,
) -> dict[str, bytes]:
    source = repository_root / TRACE_SET_OUTPUT_PATH
    source_payloads = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*") if path.is_file()
    }
    validate_trace_set_payloads(source_payloads)
    candidate_lines = []
    ordered_results = sorted(results, key=lambda item: item.candidate.candidate_id)
    ranked = sorted((item for item in ordered_results if not item.disqualified), key=ranking_tuple)
    rank_by_id = {item.candidate.candidate_id: index + 1 for index, item in enumerate(ranked)}
    for item in ordered_results:
        document = {
            "candidate": item.candidate.as_document(),
            "metrics": item.metrics,
            "per_trace_metrics": item.per_trace_metrics,
            "disqualified": item.disqualified,
            "disqualification_reasons": item.disqualification_reasons,
            "ranking_tuple": ranking_tuple(item),
            "rank": rank_by_id.get(item.candidate.candidate_id),
            "replay_hash": item.replay_hash,
        }
        document["canonical_candidate_metrics_hash"] = canonical_sha256(document)
        candidate_lines.append(canonical_json_bytes(document))
    metrics_bytes = b"\n".join(candidate_lines) + b"\n"
    ranking = {
        "schema_version": SCHEMA_VERSION,
        "candidate_count": len(results),
        "nondisqualified_candidate_count": len(ranked),
        "disqualified_candidate_count": len(results) - len(ranked),
        "ranking_contract": [
            "minimize two_cycles plus three_cycles", "minimize rapid_reversals",
            "minimize transition_budget_exhaustions", "minimize invalid_guard_count",
            "minimize unavailable_evidence_block_count", "minimize stuck_trace_count",
            "minimize graph_blocks", "minimize guard_conflict_count",
            "maximize nonhandoff phase_coverage", "minimize inter_phase_transitions",
            "lexical candidate_id",
        ],
        "ranked_candidate_ids": [item.candidate.candidate_id for item in ranked],
        "selected_candidate_id": selected.candidate.candidate_id,
    }
    candidate_document = {
        "candidate_id": "engineering_candidate_v0",
        "source_candidate_id": selected.candidate.candidate_id,
        "parameters": selected.candidate.as_document(),
        "rank": 1,
        "ranking_tuple": ranking_tuple(selected),
        "metrics": selected.metrics,
        "per_trace_metrics": selected.per_trace_metrics,
        "source_trace_set_commit": trace_set_commit,
        "source_trace_set_hash": json.loads(source_payloads["trace_set_manifest.json"])["trace_set_aggregate_hash"],
        "parameter_grid_hash": config_canonical_hash(config),
        "implementation_commit": implementation_commit,
        "shadow_only": True,
        "active_authority": False,
        "scientific_threshold_validation": "not_performed",
        "staged_recovery_execution": EXECUTION_NOT_AUTHORIZED,
        "claim_restrictions": list(CLAIM_RESTRICTIONS),
    }
    candidate_document["canonical_engineering_candidate_hash"] = canonical_sha256(candidate_document)
    matrix = []
    for source_phase in architecture_phase_ids():
        for destination in architecture_phase_ids():
            matching = [
                item for item in selected.transition_records
                if item["current_shadow_phase"] == source_phase and item["desired_shadow_phase"] == destination
            ]
            matrix.append({
                "source_phase": source_phase,
                "desired_phase": destination,
                "recommendation_count": len(matching),
                "executed_shadow_transition_count": sum(item["shadow_transition_executed"] for item in matching),
                "blocked_count": sum(item["transition_blocked"] for item in matching),
            })
    conflict_signatures: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for item in selected.conflicts:
        signature = tuple(item["conditions"])
        conflict_signatures.setdefault(signature, []).append(item)
    conflicts = {
        "selected_candidate_id": selected.candidate.candidate_id,
        "conflict_count": len(selected.conflicts),
        "conflicts": [
            {
                "signature": list(signature),
                "count": len(items),
                "events": items,
                "scientific_limitation": "priority resolution evidence only; not a validated phase policy",
            }
            for signature, items in sorted(conflict_signatures.items())
        ],
    }
    grid_document = {
        "schema_version": SCHEMA_VERSION,
        "grid": config["grid"],
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "candidate_ids": [item.candidate.candidate_id for item in ordered_results],
        "canonical_parameter_grid_hash": config_canonical_hash(config),
    }
    trace_set_manifest = json.loads(source_payloads["trace_set_manifest.json"])
    manifest = {
        "calibration_id": CALIBRATION_ID,
        "schema_version": SCHEMA_VERSION,
        "completed_date": COMPLETED_DATE,
        "implementation_commit": implementation_commit,
        "trace_set_commit": trace_set_commit,
        "registry_manifest_hash": EXPECTED_REGISTRY_MANIFEST_HASH,
        "registry_aggregate_hash": EXPECTED_REGISTRY_AGGREGATE_HASH,
        "smoke_manifest_hash": EXPECTED_SMOKE_MANIFEST_HASH,
        "trace_set_manifest_hash": trace_set_manifest["canonical_manifest_hash"],
        "trace_set_aggregate_hash": trace_set_manifest["trace_set_aggregate_hash"],
        "trace_count": EXPECTED_TRACE_COUNT,
        "physical_pair_count": EXPECTED_TRACE_COUNT,
        "physical_execution_count": EXPECTED_PHYSICAL_EXECUTION_COUNT,
        "physical_executions_during_calibration_ranking": 0,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "offline_replay_count": EXPECTED_OFFLINE_REPLAY_COUNT,
        "selected_candidate_id": selected.candidate.candidate_id,
        "engineering_candidate_hash": candidate_document["canonical_engineering_candidate_hash"],
        "parameter_grid_hash": config_canonical_hash(config),
        "physical_equivalence_failures": 0,
        "unauthorized_physical_effects": 0,
        "nominal_handoff_authorized": False,
        "shadow_only": True,
        "active_authority": False,
        "staged_recovery_execution": EXECUTION_NOT_AUTHORIZED,
        "claim_restrictions": list(CLAIM_RESTRICTIONS),
        "artifact_filenames": sorted([
            "calibration_manifest.json", "parameter_grid.json", "trace_set_manifest.json",
            "trace_index.json", "equivalence_report.json", "candidate_metrics.jsonl",
            "candidate_ranking.json", "engineering_candidate.json",
            "phase_transition_matrix.json", "guard_conflict_report.json", "summary.md",
            *[key for key in source_payloads if key.startswith("traces/")],
        ]),
    }
    manifest["canonical_manifest_hash"] = canonical_sha256(manifest)
    summary = (
        "# Staged Recovery Shadow Calibration v0\n\n"
        "Status: Thirteen-trace shadow calibration and engineering candidate freeze completed; active staged recovery remains unauthorized.\n\n"
        "Completed: 2026-08-11\n\n"
        "## Result\n\n"
        f"Exactly 216 candidates were replayed across 13 frozen observed traces ({EXPECTED_OFFLINE_REPLAY_COUNT} offline replays). "
        f"The selected engineering baseline is `{selected.candidate.candidate_id}`.\n\n"
        "Unavailable and invalid guard evidence remained explicit. No baseline trace was published, no candidate replay executed physics, "
        "and no shadow output controlled an action, state, Final Veto decision, terminal condition, or real phase.\n\n"
        "## Claims\n\n"
        "This is deterministic engineering ranking under a frozen lexicographic contract. It does not demonstrate controller or recovery improvement, "
        "optimality, formal safety, validated active thresholds, handoff readiness, or deployment readiness.\n"
    ).encode("utf-8")
    return {
        "calibration_manifest.json": json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n",
        "parameter_grid.json": json.dumps(grid_document, indent=2, sort_keys=True).encode() + b"\n",
        "trace_set_manifest.json": source_payloads["trace_set_manifest.json"],
        "trace_index.json": source_payloads["trace_index.json"],
        "equivalence_report.json": source_payloads["equivalence_report.json"],
        "candidate_metrics.jsonl": metrics_bytes,
        "candidate_ranking.json": json.dumps(ranking, indent=2, sort_keys=True).encode() + b"\n",
        "engineering_candidate.json": json.dumps(candidate_document, indent=2, sort_keys=True).encode() + b"\n",
        "phase_transition_matrix.json": json.dumps({"matrix": matrix}, indent=2, sort_keys=True).encode() + b"\n",
        "guard_conflict_report.json": json.dumps(conflicts, indent=2, sort_keys=True).encode() + b"\n",
        "summary.md": summary,
        **{key: value for key, value in source_payloads.items() if key.startswith("traces/")},
    }


def validate_calibration_payloads(payloads: Mapping[str, bytes]) -> None:
    top = {
        "calibration_manifest.json", "parameter_grid.json", "trace_set_manifest.json",
        "trace_index.json", "equivalence_report.json", "candidate_metrics.jsonl",
        "candidate_ranking.json", "engineering_candidate.json", "phase_transition_matrix.json",
        "guard_conflict_report.json", "summary.md",
    }
    if {key for key in payloads if not key.startswith("traces/")} != top:
        raise ShadowCalibrationError("calibration top-level artifact set mismatch")
    if sum(key.startswith("traces/") for key in payloads) != EXPECTED_TRACE_COUNT:
        raise ShadowCalibrationError("calibration requires exactly 13 trace files")
    manifest = json.loads(payloads["calibration_manifest.json"])
    supplied = manifest.pop("canonical_manifest_hash", None)
    if supplied != canonical_sha256(manifest):
        raise ShadowCalibrationError("calibration manifest canonical hash mismatch")
    if manifest.get("physical_executions_during_calibration_ranking") != 0:
        raise ShadowCalibrationError("offline calibration attempted physical execution")
    grid = json.loads(payloads["parameter_grid.json"])
    if grid.get("candidate_count") != EXPECTED_CANDIDATE_COUNT:
        raise ShadowCalibrationError("published candidate count mismatch")
    lines = [json.loads(line) for line in payloads["candidate_metrics.jsonl"].splitlines() if line]
    if len(lines) != EXPECTED_CANDIDATE_COUNT:
        raise ShadowCalibrationError("candidate metrics must contain 216 records")
    ranking = json.loads(payloads["candidate_ranking.json"])
    if ranking.get("nondisqualified_candidate_count", 0) < 1:
        raise ShadowCalibrationError("no nondisqualified engineering candidate")
    candidate = json.loads(payloads["engineering_candidate.json"])
    candidate_hash = candidate.pop("canonical_engineering_candidate_hash", None)
    if candidate_hash != canonical_sha256(candidate):
        raise ShadowCalibrationError("engineering candidate canonical hash mismatch")
    if candidate.get("shadow_only") is not True or candidate.get("active_authority") is not False:
        raise ShadowCalibrationError("engineering candidate authority boundary failed")
    trace_subset = {
        key: value for key, value in payloads.items()
        if key in {"trace_set_manifest.json", "trace_index.json", "equivalence_report.json"} or key.startswith("traces/")
    }
    validate_trace_set_payloads(trace_subset)


__all__ = [
    "CALIBRATION_ID", "CALIBRATION_OUTPUT_PATH", "CLAIM_RESTRICTIONS", "COMPLETED_DATE",
    "CONFIG_PATH", "EXPECTED_CANDIDATE_COUNT", "EXPECTED_OFFLINE_REPLAY_COUNT",
    "EXPECTED_PHYSICAL_EXECUTION_COUNT", "EXPECTED_SMOKE_MANIFEST_HASH", "EXPECTED_TRACE_COUNT",
    "EXPLICIT_ABORT_BRANCH", "NO_PROGRESS_COMPONENTS", "PHYSICAL_BRANCHES", "SCHEMA_VERSION",
    "TRACE_SET_OUTPUT_PATH", "CalibrationCandidate", "CalibrationTraceDefinition",
    "CandidateReplayResult", "CapturedCalibrationTrace", "ShadowCalibrationError",
    "analyze_candidates", "atomic_publish_new_directory", "build_calibration_payloads",
    "build_trace_set_payloads", "calibration_candidates", "capture_trace_pair", "candidate_id",
    "config_canonical_hash", "guard_evaluation_document", "guard_evaluation_from_document",
    "load_and_validate_config", "load_trace_set", "ranking_tuple", "replay_candidate",
    "sha256_bytes", "trace_definitions", "trace_jsonl_bytes", "validate_calibration_payloads",
    "validate_trace_set_payloads",
]
