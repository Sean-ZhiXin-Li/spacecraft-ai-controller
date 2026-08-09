from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from runtime_assurance.recovery_branch_executor import (
    RecoveryBranchExecutionResult,
    execute_registered_recovery_branch,
)
from runtime_assurance.recovery_branch_state_registry import (
    RegisteredBranchState,
    load_branch_state_registry,
    load_registered_branch_state,
)
from runtime_assurance.recovery_stop_conditions import (
    RecoveryStopConditionReport,
    evaluate_recovery_stop_conditions,
)
from runtime_assurance.staged_recovery_contract import EXECUTION_NOT_AUTHORIZED
from runtime_assurance.staged_recovery_guard_evidence import (
    evaluate_guard_atoms_for_event,
    guard_atom_definitions,
)
from runtime_assurance.staged_recovery_logger_adapter import (
    BoundedRecoveryValidationRun,
    BoundedRuntimeIdentity,
    BoundedRuntimeSnapshot,
    ObserverIntegrationError,
    RuntimeSnapshotType,
    Stage0BValidationLoggerAdapter,
    runtime_state_hash,
)
from runtime_assurance.staged_recovery_runtime_logger import (
    ActionDisposition,
    event_document,
)
from runtime_assurance.staged_recovery_shadow_guard import (
    SHADOW_OUTPUT_CONSUMED_BY_PHYSICAL_RUNTIME,
    SHADOW_POLICY_ID,
    ShadowPhaseMachine,
    ShadowTransitionRecord,
    architecture_phase_ids,
    canonical_sha256,
    resolve_shadow_phase,
)
from simulator.phase34_35_transition import CartesianState2D


SHADOW_RUNTIME_SCHEMA_VERSION = "staged_recovery_shadow_runtime_v0"
SHADOW_SMOKE_ID = "staged_recovery_shadow_smoke_v0"
COMPLETED_DATE = "2026-08-09"
PHYSICAL_BRANCH_ID = "velocity_opposed_thrust_v0"
MAXIMUM_RECOVERY_TRANSITIONS = 32
EXPECTED_REGISTRY_MANIFEST_HASH = (
    "b800735062af8426045e40117044056d545f5ff2ebbc3795fdbf52266ba8a980"
)
EXPECTED_REGISTRY_AGGREGATE_HASH = (
    "228dae3661e79178ed55e30ec15308ce1d86acc010877605fde61f2d5012a3bd"
)
OUTPUT_PATH = Path("analysis/staged_recovery_shadow_smoke_v0")
VALIDATION_TERMINAL_REASON = "shadow_smoke_complete"
CLAIM_RESTRICTIONS = (
    "observational shadow FSM engineering validation only",
    "no recovery performance improvement claim",
    "no validated phase threshold or optimal ordering claim",
    "no autonomous handoff, active staged execution, safety, or deployment claim",
)


class ShadowRuntimeError(RuntimeError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShadowRuntimeError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ShadowRuntimeError(f"{name} must be finite")
    return result


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ShadowRuntimeError(f"{name} must be a mapping")
    return value


def build_registered_runtime_identity(
    registered: RegisteredBranchState,
    *,
    implementation_commit: str,
) -> tuple[BoundedRuntimeIdentity, CartesianState2D]:
    if not implementation_commit:
        raise ShadowRuntimeError("implementation commit is required")
    document = registered.as_document()
    state_document = _mapping(document.get("state"), "state")
    configuration = _mapping(document.get("simulator_configuration"), "simulator_configuration")
    constants = _mapping(configuration.get("simulator_constants"), "simulator_constants")
    member = registered.member
    state = CartesianState2D(
        x=_finite(state_document.get("position_x"), "state.position_x"),
        y=_finite(state_document.get("position_y"), "state.position_y"),
        vx=_finite(state_document.get("velocity_x"), "state.velocity_x"),
        vy=_finite(state_document.get("velocity_y"), "state.velocity_y"),
    )
    identity = BoundedRuntimeIdentity(
        case_id=member.case_id,
        seed=member.seed,
        branch_id=PHYSICAL_BRANCH_ID,
        branch_state_hash=member.canonical_branch_state_hash,
        simulator_configuration_hash=member.simulator_configuration_hash,
        constants_hash=member.constants_hash,
        branch_step=member.branch_step,
        nominal_prefix_transition_count=member.nominal_prefix_transition_count,
        dt=_finite(constants.get("dt"), "constants.dt"),
        target_radius=_finite(constants.get("target_radius"), "constants.target_radius"),
        mu=_finite(constants.get("mu"), "constants.mu"),
        ratio_denominator_epsilon=_finite(
            constants.get("speed_ratio_denominator_epsilon"),
            "constants.speed_ratio_denominator_epsilon",
        ),
        action_component_limit=_finite(
            constants.get("action_component_max"), "constants.action_component_max"
        ),
        total_horizon=int(constants.get("max_steps")),
        overspeed_threshold=_finite(document.get("threshold"), "threshold"),
        implementation_commit=implementation_commit,
    )
    return identity, state


def _speed_ratio(state: CartesianState2D | None, identity: BoundedRuntimeIdentity) -> float | None:
    if state is None:
        return None
    target_speed = math.sqrt(identity.mu / identity.target_radius)
    return math.hypot(state.vx, state.vy) / (target_speed + identity.ratio_denominator_epsilon)


def _disposition(execution: RecoveryBranchExecutionResult) -> ActionDisposition:
    if execution.terminal_reason == "explicit_recovery_abort":
        return ActionDisposition.NO_ACTION
    if execution.terminal_reason == "recovery_action_rejected":
        return ActionDisposition.REJECTED
    if execution.executed and execution.action == (0.0, 0.0):
        return ActionDisposition.ZERO_ACTION_EXECUTED
    if execution.executed:
        monitor = execution.monitor_decision
        if monitor is not None and monitor.executed_action == execution.action:
            return ActionDisposition.EXECUTED_UNCHANGED
        return ActionDisposition.EXECUTED_MODIFIED
    return ActionDisposition.NOT_EVALUATED


RuntimeObserver = Callable[[BoundedRuntimeSnapshot], None]
RegisteredStepExecutor = Callable[..., RecoveryBranchExecutionResult]


def _notify(observer: RuntimeObserver | None, snapshot: BoundedRuntimeSnapshot, count: int) -> None:
    if observer is None:
        return
    try:
        result = observer(snapshot)
    except Exception as exc:
        raise ObserverIntegrationError(
            f"shadow observer failed at event {snapshot.event_index}: {type(exc).__name__}: {exc}",
            event_index=snapshot.event_index,
            recovery_step=snapshot.recovery_step,
            physical_transitions=count,
        ) from exc
    if result is not None:
        raise ObserverIntegrationError(
            "shadow observer return values are forbidden and cannot affect physical runtime",
            event_index=snapshot.event_index,
            recovery_step=snapshot.recovery_step,
            physical_transitions=count,
        )


def run_registered_bounded_shadow_path(
    registered: RegisteredBranchState,
    *,
    implementation_commit: str,
    observer: RuntimeObserver | None = None,
    horizon_steps: int = MAXIMUM_RECOVERY_TRANSITIONS,
    step_executor: RegisteredStepExecutor = execute_registered_recovery_branch,
) -> BoundedRecoveryValidationRun:
    if horizon_steps != MAXIMUM_RECOVERY_TRANSITIONS:
        raise ShadowRuntimeError("Stage 1B-A smoke horizon is frozen at 32 transitions")
    identity, initial_state = build_registered_runtime_identity(
        registered, implementation_commit=implementation_commit
    )
    initial_hash = runtime_state_hash(initial_state)
    initial = BoundedRuntimeSnapshot(
        snapshot_type=RuntimeSnapshotType.INITIAL,
        event_index=0,
        identity=identity,
        recovery_step=0,
        total_transition_count=identity.nominal_prefix_transition_count,
        pre_state=initial_state,
        pre_state_hash=initial_hash,
    )
    snapshots = [initial]
    transitions: list[BoundedRuntimeSnapshot] = []
    _notify(observer, initial, 0)
    current = initial_state
    count = 0
    terminal_reason: str | None = None
    stop_report: RecoveryStopConditionReport | None = None
    for attempted in range(1, horizon_steps + 1):
        execution = step_executor(
            registered.registry_member_id,
            PHYSICAL_BRANCH_ID,
            horizon_steps=1,
            current_state=current,
        )
        if not isinstance(execution, RecoveryBranchExecutionResult) or not execution.valid:
            raise ShadowRuntimeError(f"registered executor failed at attempted step {attempted}")
        if execution.previous_state != current or execution.previous_state_hash != runtime_state_hash(current):
            raise ShadowRuntimeError("registered executor did not use the current continuation state")
        before = current
        if execution.executed:
            if execution.transition_count != 1 or execution.next_state is None:
                raise ShadowRuntimeError("executed action did not realize exactly one transition")
            current = execution.next_state
            count += 1
        elif execution.transition_count != 0:
            raise ShadowRuntimeError("nonexecuted action incremented the physical transition count")
        realized_ratio = _speed_ratio(execution.next_state, identity)
        stop_report = evaluate_recovery_stop_conditions(
            execution_terminal_reason=execution.terminal_reason,
            next_state=execution.next_state,
            realized_speed_ratio=realized_ratio,
            overspeed_threshold=identity.overspeed_threshold,
            recovery_transition_count=count,
            recovery_horizon_steps=horizon_steps,
            total_transition_count=identity.nominal_prefix_transition_count + count,
            total_horizon_steps=identity.total_horizon,
        )
        terminal_reason = stop_report.terminal_reason
        monitor = execution.monitor_decision
        predicted = execution.predicted_nominal_state
        snapshot = BoundedRuntimeSnapshot(
            snapshot_type=RuntimeSnapshotType.TRANSITION,
            event_index=len(snapshots),
            identity=identity,
            recovery_step=count,
            total_transition_count=identity.nominal_prefix_transition_count + count,
            pre_state=before,
            pre_state_hash=execution.previous_state_hash,
            proposed_action=execution.action,
            executed_action=execution.action if execution.executed else None,
            action_disposition=_disposition(execution),
            transition_executed=execution.executed,
            monitor_id=monitor.monitor_id if monitor else None,
            monitor_decision=monitor.decision if monitor else None,
            monitor_reason=monitor.reason if monitor else None,
            monitor_threshold=monitor.threshold if monitor else None,
            monitor_comparator=monitor.comparator if monitor else None,
            predicted_next_state=predicted,
            predicted_state_hash=runtime_state_hash(predicted) if predicted else None,
            predicted_speed_ratio=monitor.predicted_nominal_speed_ratio if monitor else None,
            realized_next_state=execution.next_state,
            realized_state_hash=execution.next_state_hash,
            realized_speed_ratio=realized_ratio,
            evaluator_statuses=stop_report.statuses,
            runtime_terminal_reason=terminal_reason,
        )
        snapshots.append(snapshot)
        transitions.append(snapshot)
        _notify(observer, snapshot, count)
        if terminal_reason is not None:
            break
    if stop_report is None or terminal_reason is None:
        raise ShadowRuntimeError("bounded smoke ended without a physical terminal reason")
    terminal = BoundedRuntimeSnapshot(
        snapshot_type=RuntimeSnapshotType.TERMINAL,
        event_index=len(snapshots),
        identity=identity,
        recovery_step=count,
        total_transition_count=identity.nominal_prefix_transition_count + count,
        pre_state=current,
        pre_state_hash=runtime_state_hash(current),
        evaluator_statuses=stop_report.statuses,
        runtime_terminal_reason=terminal_reason,
        validation_terminal_reason=VALIDATION_TERMINAL_REASON,
    )
    snapshots.append(terminal)
    _notify(observer, terminal, count)
    return BoundedRecoveryValidationRun(
        identity=identity,
        initial_state=initial_state,
        initial_state_hash=initial_hash,
        snapshots=tuple(snapshots),
        transition_snapshots=tuple(transitions),
        recovery_transition_count=count,
        final_state=current,
        final_state_hash=runtime_state_hash(current),
        runtime_terminal_reason=terminal_reason,
        validation_terminal_reason=VALIDATION_TERMINAL_REASON,
    )


def physical_run_document(run: BoundedRecoveryValidationRun) -> dict[str, object]:
    transitions = run.transition_snapshots
    return {
        "case_id": run.identity.case_id,
        "seed": run.identity.seed,
        "branch_id": run.identity.branch_id,
        "branch_state_hash": run.identity.branch_state_hash,
        "initial_state_hash": run.initial_state_hash,
        "initial_state": (
            run.initial_state.x, run.initial_state.y, run.initial_state.vx, run.initial_state.vy
        ),
        "executed_action_sequence": [item.executed_action for item in transitions],
        "realized_state_hash_sequence": [item.realized_state_hash for item in transitions],
        "realized_state_sequence": [
            None if item.realized_next_state is None else (
                item.realized_next_state.x, item.realized_next_state.y,
                item.realized_next_state.vx, item.realized_next_state.vy,
            )
            for item in transitions
        ],
        "transition_count": run.recovery_transition_count,
        "terminal_reason": run.runtime_terminal_reason,
        "final_veto_decision_sequence": [
            (
                item.monitor_id, item.monitor_decision, item.monitor_reason,
                item.monitor_threshold, item.monitor_comparator, item.predicted_speed_ratio,
            )
            for item in transitions
        ],
        "evaluator_status_sequence": [item.evaluator_statuses for item in transitions],
        "final_state_hash": run.final_state_hash,
    }


def compare_physical_runs(
    baseline: BoundedRecoveryValidationRun,
    observed: BoundedRecoveryValidationRun,
) -> dict[str, object]:
    left = physical_run_document(baseline)
    right = physical_run_document(observed)
    checks = {f"same_{key}": left[key] == right[key] for key in left}
    return {
        "checks": checks,
        "all_equivalence_checks": all(checks.values()),
        "baseline_trace_published": False,
        "observed_trace_published": True,
    }


class ShadowObservationAdapter:
    def __init__(self, identity: BoundedRuntimeIdentity, *, trace_id: str) -> None:
        self._logger = Stage0BValidationLoggerAdapter(
            identity, session_id=f"{SHADOW_SMOKE_ID}__{trace_id}", max_events=34
        )
        self._machine = ShadowPhaseMachine()
        self._records: list[ShadowTransitionRecord] = []
        self._source_documents: list[dict[str, object]] = []
        self._definitions = guard_atom_definitions()

    @property
    def records(self) -> tuple[ShadowTransitionRecord, ...]:
        return tuple(self._records)

    @property
    def source_documents(self) -> tuple[dict[str, object], ...]:
        return tuple(copy.deepcopy(item) for item in self._source_documents)

    def __call__(self, snapshot: BoundedRuntimeSnapshot) -> None:
        self._logger.observe(snapshot)
        source = event_document(self._logger.events[-1])
        previous = self._source_documents[-1] if self._source_documents else None
        evaluations = evaluate_guard_atoms_for_event(source, previous, self._definitions)
        resolution = resolve_shadow_phase(evaluations)
        record = self._machine.step(
            resolution,
            event_index=snapshot.event_index,
            recovery_step=snapshot.recovery_step,
            source_event_hash=str(source["canonical_event_sha256"]),
            terminal_event=snapshot.snapshot_type == RuntimeSnapshotType.TERMINAL,
        )
        self._source_documents.append(source)
        self._records.append(record)
        if snapshot.snapshot_type == RuntimeSnapshotType.TERMINAL:
            self._logger.finalize()
        return None


@dataclass(frozen=True, slots=True)
class ShadowSmokeCaseResult:
    registry_member_id: str
    case_id: str
    baseline: BoundedRecoveryValidationRun
    observed: BoundedRecoveryValidationRun
    equivalence: Mapping[str, object]
    records: tuple[ShadowTransitionRecord, ...]


def run_shadow_smoke_case(
    repository_root: Path,
    registry_member_id: str,
    *,
    implementation_commit: str,
    loader: Callable[[Path, str], RegisteredBranchState] = load_registered_branch_state,
    runner: Callable[..., BoundedRecoveryValidationRun] = run_registered_bounded_shadow_path,
) -> ShadowSmokeCaseResult:
    baseline_state = loader(repository_root, registry_member_id)
    baseline = runner(baseline_state, implementation_commit=implementation_commit)
    observed_state = loader(repository_root, registry_member_id)
    identity, _ = build_registered_runtime_identity(
        observed_state, implementation_commit=implementation_commit
    )
    adapter = ShadowObservationAdapter(identity, trace_id=registry_member_id)
    observed = runner(
        observed_state, implementation_commit=implementation_commit, observer=adapter
    )
    equivalence = compare_physical_runs(baseline, observed)
    if not equivalence["all_equivalence_checks"]:
        raise ShadowRuntimeError(f"physical equivalence failed for {registry_member_id}")
    return ShadowSmokeCaseResult(
        registry_member_id=registry_member_id,
        case_id=observed.identity.case_id,
        baseline=baseline,
        observed=observed,
        equivalence=equivalence,
        records=adapter.records,
    )


def _trace_bytes(result: ShadowSmokeCaseResult) -> bytes:
    lines = []
    for record in result.records:
        document = record.as_document()
        document["trace_id"] = result.registry_member_id
        document["case_id"] = result.case_id
        lines.append(_canonical_bytes(document))
    return b"\n".join(lines) + b"\n"


def build_smoke_payloads(
    results: tuple[ShadowSmokeCaseResult, ...],
    *,
    implementation_commit: str,
) -> dict[str, bytes]:
    if len(results) != 4 or len({item.registry_member_id for item in results}) != 4:
        raise ShadowRuntimeError("smoke publication requires four distinct registry members")
    if any(not item.equivalence["all_equivalence_checks"] for item in results):
        raise ShadowRuntimeError("smoke publication requires complete physical equivalence")
    trace_payloads = {
        f"traces/{item.case_id}.jsonl": _trace_bytes(item) for item in results
    }
    trace_index = {
        "schema_version": SHADOW_RUNTIME_SCHEMA_VERSION,
        "traces": [
            {
                "registry_member_id": item.registry_member_id,
                "case_id": item.case_id,
                "transition_count": item.observed.recovery_transition_count,
                "event_count": len(item.records),
                "terminal_reason": item.observed.runtime_terminal_reason,
                "trace_path": f"traces/{item.case_id}.jsonl",
                "trace_sha256": _sha256_bytes(trace_payloads[f"traces/{item.case_id}.jsonl"]),
            }
            for item in results
        ],
        "baseline_traces_published": 0,
        "observed_traces_published": 4,
    }
    equivalence_report = {
        "schema_version": SHADOW_RUNTIME_SCHEMA_VERSION,
        "pairs": [
            {
                "registry_member_id": item.registry_member_id,
                "case_id": item.case_id,
                **dict(item.equivalence),
            }
            for item in results
        ],
        "physical_equivalence_failures": 0,
        "all_pairs_equivalent": True,
    }
    records = [record for item in results for record in item.records]
    matrix: list[dict[str, object]] = []
    for source in architecture_phase_ids():
        for destination in architecture_phase_ids():
            matching = [
                record for record in records
                if record.current_shadow_phase == source and record.desired_shadow_phase == destination
            ]
            matrix.append({
                "source_phase": source,
                "desired_phase": destination,
                "recommendation_count": len(matching),
                "executed_shadow_transition_count": sum(item.shadow_transition_executed for item in matching),
                "blocked_count": sum(item.transition_blocked for item in matching),
            })
    summary = {
        "schema_version": SHADOW_RUNTIME_SCHEMA_VERSION,
        "case_execution_status": "4/4 passed",
        "shadow_phase_entries": {
            phase: sum(
                record.shadow_transition_executed and record.resulting_shadow_phase == phase
                for record in records
            )
            for phase in architecture_phase_ids()
        },
        "shadow_transition_count": sum(
            record.shadow_transition_executed and record.current_shadow_phase != "unassigned"
            for record in records
        ),
        "graph_blocks": sum(record.block_reason == "graph_edge_not_allowed" for record in records),
        "two_cycles": sum(record.two_cycle_detected for record in records),
        "repeated_reasons": sum(record.repeated_transition_reason for record in records),
        "transition_budget_exhaustion": sum(
            record.block_reason == "transition_budget_exhausted" for record in records
        ),
        "unavailable_guards": sum(len(record.unavailable_guard_atoms) for record in records),
        "invalid_guards": sum(len(record.invalid_guard_atoms) for record in records),
        "nominal_handoff_recommendations": sum(record.nominal_handoff_recommended for record in records),
        "unauthorized_physical_effects": 0,
    }
    if summary["nominal_handoff_recommendations"] != 0:
        raise ShadowRuntimeError("nominal handoff recommendation is forbidden")
    repository_root = Path(__file__).resolve().parents[1]
    registry_manifest = json.loads(
        (repository_root / "analysis/recovery_branch_state_registry_v0/registry_manifest.json").read_text("utf-8")
    )
    manifest = {
        "smoke_id": SHADOW_SMOKE_ID,
        "schema_version": SHADOW_RUNTIME_SCHEMA_VERSION,
        "completed_date": COMPLETED_DATE,
        "implementation_commit": implementation_commit,
        "registry_manifest_hash": EXPECTED_REGISTRY_MANIFEST_HASH,
        "registry_aggregate_hash": EXPECTED_REGISTRY_AGGREGATE_HASH,
        "registry_member_count": 4,
        "pair_count": 4,
        "bounded_execution_count": 8,
        "physical_branch_id": PHYSICAL_BRANCH_ID,
        "maximum_recovery_transitions": MAXIMUM_RECOVERY_TRANSITIONS,
        "shadow_policy_id": SHADOW_POLICY_ID,
        "shadow_output_consumed_by_physical_runtime": SHADOW_OUTPUT_CONSUMED_BY_PHYSICAL_RUNTIME,
        "physical_equivalence_failures": 0,
        "unauthorized_physical_effects": 0,
        "nominal_handoff_authorized": False,
        "automatic_retry": False,
        "baseline_traces_published": 0,
        "protected_evidence_preserved": True,
        "protected_artifact_hashes": registry_manifest.get("protected_artifact_hashes", {}),
        "staged_recovery_execution": EXECUTION_NOT_AUTHORIZED,
        "scientific_claim_restrictions": list(CLAIM_RESTRICTIONS),
        "artifact_filenames": sorted([
            "manifest.json", "trace_index.json", "equivalence_report.json",
            "phase_transition_matrix.json", "shadow_guard_summary.json", "summary.md",
            *trace_payloads,
        ]),
    }
    manifest["canonical_manifest_hash"] = canonical_sha256(manifest)
    markdown = (
        "# Staged Recovery Shadow Smoke v0\n\n"
        "Status: Four-case observational shadow smoke completed; staged recovery execution remains unauthorized.\n\n"
        "Completed: 2026-08-09\n\n"
        "## Scope\n\n"
        "The shadow FSM observed four fresh bounded executions using the existing velocity-opposed branch. "
        "It did not control an action, state, Final Veto decision, terminal condition, or real phase.\n\n"
        "## Result\n\n"
        "All four baseline/shadow pairs were physically equivalent. Baseline traces were not published. "
        "The records are engineering shadow-policy evidence, not recovery-performance, safety, threshold-validation, or deployment evidence.\n"
    ).encode("utf-8")
    payloads = {
        "manifest.json": json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        "trace_index.json": json.dumps(trace_index, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        "equivalence_report.json": json.dumps(equivalence_report, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        "phase_transition_matrix.json": json.dumps({"matrix": matrix}, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        "shadow_guard_summary.json": json.dumps(summary, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        "summary.md": markdown,
        **trace_payloads,
    }
    return payloads


def validate_smoke_payloads(payloads: Mapping[str, bytes]) -> None:
    expected_top = {
        "manifest.json", "trace_index.json", "equivalence_report.json",
        "phase_transition_matrix.json", "shadow_guard_summary.json", "summary.md",
    }
    if not expected_top.issubset(payloads) or sum(key.startswith("traces/") for key in payloads) != 4:
        raise ShadowRuntimeError("smoke payload set is incomplete")
    manifest = json.loads(payloads["manifest.json"])
    supplied = manifest.pop("canonical_manifest_hash", None)
    if supplied != canonical_sha256(manifest):
        raise ShadowRuntimeError("smoke manifest hash mismatch")
    equivalence = json.loads(payloads["equivalence_report.json"])
    summary = json.loads(payloads["shadow_guard_summary.json"])
    if not equivalence.get("all_pairs_equivalent") or equivalence.get("physical_equivalence_failures") != 0:
        raise ShadowRuntimeError("published physical equivalence is incomplete")
    if summary.get("unauthorized_physical_effects") != 0 or summary.get("nominal_handoff_recommendations") != 0:
        raise ShadowRuntimeError("published shadow authority boundary failed")


def publish_smoke_payloads(
    repository_root: Path,
    payloads: Mapping[str, bytes],
    *,
    writer: Callable[[Path, bytes], None] | None = None,
) -> Path:
    validate_smoke_payloads(payloads)
    target = (repository_root / OUTPUT_PATH).resolve()
    root = repository_root.resolve()
    try:
        target.relative_to(root / "analysis")
    except ValueError as exc:
        raise ShadowRuntimeError("smoke output must remain under analysis") from exc
    if target.exists():
        raise ShadowRuntimeError("smoke result directory already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    write = writer or (lambda path, data: path.write_bytes(data))
    try:
        for relative, data in sorted(payloads.items()):
            path = staging / Path(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            write(path, data)
        staged = {
            path.relative_to(staging).as_posix(): path.read_bytes()
            for path in staging.rglob("*") if path.is_file()
        }
        validate_smoke_payloads(staged)
        os.replace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def validate_registry_source(repository_root: Path) -> tuple[str, ...]:
    registry = load_branch_state_registry(repository_root)
    if registry.canonical_manifest_hash != EXPECTED_REGISTRY_MANIFEST_HASH:
        raise ShadowRuntimeError("registry manifest identity changed")
    if len(registry.members) != 4:
        raise ShadowRuntimeError("Stage 1B-A requires exactly four frozen registry members")
    manifest = json.loads((repository_root / "analysis/recovery_branch_state_registry_v0/registry_manifest.json").read_text("utf-8"))
    if manifest.get("registry_aggregate_hash") != EXPECTED_REGISTRY_AGGREGATE_HASH:
        raise ShadowRuntimeError("registry aggregate identity changed")
    return tuple(member.registry_member_id for member in registry.members)


__all__ = [
    "CLAIM_RESTRICTIONS", "COMPLETED_DATE", "EXPECTED_REGISTRY_AGGREGATE_HASH",
    "EXPECTED_REGISTRY_MANIFEST_HASH", "MAXIMUM_RECOVERY_TRANSITIONS", "OUTPUT_PATH",
    "PHYSICAL_BRANCH_ID", "SHADOW_RUNTIME_SCHEMA_VERSION", "SHADOW_SMOKE_ID",
    "ShadowObservationAdapter", "ShadowRuntimeError", "ShadowSmokeCaseResult",
    "build_registered_runtime_identity", "build_smoke_payloads", "compare_physical_runs",
    "physical_run_document", "publish_smoke_payloads", "run_registered_bounded_shadow_path",
    "run_shadow_smoke_case", "validate_registry_source", "validate_smoke_payloads",
]
