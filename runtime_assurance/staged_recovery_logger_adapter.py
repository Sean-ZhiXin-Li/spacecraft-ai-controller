from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Protocol

from runtime_assurance.recovery_branch_executor import (
    RecoveryBranchExecutionResult,
    RecoveryBranchExecutorError,
    execute_recovery_branch,
    load_frozen_branch_state,
    validate_branch_state_integrity,
)
from runtime_assurance.recovery_stop_conditions import (
    ACTION_REJECTED,
    CLEAR,
    EXPLICIT_ABORT,
    INVALID_RECOVERY_EVALUATION,
    INVALID_SIMULATION,
    OVERSPEED,
    RECOVERY_HORIZON_EXHAUSTED,
    TOTAL_HORIZON_EXHAUSTED,
    TRIGGERED,
    RecoveryStopConditionReport,
    evaluate_recovery_stop_conditions,
)
from runtime_assurance.staged_recovery_instrumentation import (
    INSTRUMENTATION_SCHEMA_VERSION,
    CartesianState2D as InstrumentationState2D,
    OrbitalConfiguration,
    measured_value,
)
from runtime_assurance.staged_recovery_runtime_logger import (
    ARCHITECTURE_VERSION,
    CLAIM_RESTRICTIONS,
    LOGGER_SCHEMA_VERSION,
    STAGED_EXECUTION_STATUS,
    ActionDisposition,
    StagedRecoveryInitialSnapshot,
    StagedRecoveryRuntimeLoggerSession,
    StagedRecoverySessionHeader,
    StagedRecoveryTerminalInput,
    StagedRecoveryTraceBundle,
    StagedRecoveryTransitionInput,
)
from simulator.phase34_35_transition import CartesianState2D


ADAPTER_SCHEMA_VERSION = "staged_recovery_logger_adapter_v0"
MAX_BOUNDED_VALIDATION_HORIZON = 32
VALIDATION_TERMINAL_REASON = "instrumentation_validation_complete"


class StagedRecoveryLoggerAdapterError(ValueError):
    pass


class ObserverIntegrationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        event_index: int,
        recovery_step: int,
        physical_transitions: int,
    ) -> None:
        super().__init__(message)
        self.event_index = event_index
        self.recovery_step = recovery_step
        self.physical_transitions = physical_transitions


class RuntimeSnapshotType(str, Enum):
    INITIAL = "initial_snapshot"
    TRANSITION = "transition"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class BoundedRuntimeIdentity:
    case_id: str
    seed: int
    branch_id: str
    branch_state_hash: str
    simulator_configuration_hash: str
    constants_hash: str
    branch_step: int
    nominal_prefix_transition_count: int
    dt: float
    target_radius: float
    mu: float
    ratio_denominator_epsilon: float
    action_component_limit: float
    total_horizon: int
    overspeed_threshold: float
    implementation_commit: str


@dataclass(frozen=True, slots=True)
class BoundedRuntimeSnapshot:
    snapshot_type: RuntimeSnapshotType
    event_index: int
    identity: BoundedRuntimeIdentity
    recovery_step: int
    total_transition_count: int
    pre_state: CartesianState2D
    pre_state_hash: str
    proposed_action: tuple[float, float] | None = None
    executed_action: tuple[float, float] | None = None
    action_disposition: ActionDisposition = ActionDisposition.NO_ACTION
    transition_executed: bool = False
    monitor_id: str | None = None
    monitor_decision: str | None = None
    monitor_reason: str | None = None
    monitor_threshold: float | None = None
    monitor_comparator: str | None = None
    predicted_next_state: CartesianState2D | None = None
    predicted_state_hash: str | None = None
    predicted_speed_ratio: float | None = None
    realized_next_state: CartesianState2D | None = None
    realized_state_hash: str | None = None
    realized_speed_ratio: float | None = None
    evaluator_statuses: tuple[tuple[str, str], ...] = ()
    runtime_terminal_reason: str | None = None
    validation_terminal_reason: str | None = None


@dataclass(frozen=True, slots=True)
class BoundedRecoveryValidationRun:
    identity: BoundedRuntimeIdentity
    initial_state: CartesianState2D
    initial_state_hash: str
    snapshots: tuple[BoundedRuntimeSnapshot, ...]
    transition_snapshots: tuple[BoundedRuntimeSnapshot, ...]
    recovery_transition_count: int
    final_state: CartesianState2D
    final_state_hash: str
    runtime_terminal_reason: str
    validation_terminal_reason: str


class RuntimeObserver(Protocol):
    def __call__(self, snapshot: BoundedRuntimeSnapshot) -> None: ...


StepExecutor = Callable[..., RecoveryBranchExecutionResult]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def runtime_state_hash(state: CartesianState2D) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "position_x": state.x,
                "position_y": state.y,
                "velocity_x": state.vx,
                "velocity_y": state.vy,
            }
        )
    ).hexdigest()


def _require_mapping(value: object, field_id: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StagedRecoveryLoggerAdapterError(f"{field_id} must be a mapping")
    return value


def _require_finite(value: object, field_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StagedRecoveryLoggerAdapterError(f"{field_id} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise StagedRecoveryLoggerAdapterError(f"{field_id} must be finite")
    return result


def build_bounded_runtime_identity(
    branch_state: Mapping[str, object],
    *,
    branch_id: str,
    implementation_commit: str,
) -> tuple[BoundedRuntimeIdentity, CartesianState2D]:
    validate_branch_state_integrity(branch_state)
    state_document = _require_mapping(branch_state.get("state"), "state")
    configuration = _require_mapping(
        branch_state.get("simulator_configuration"), "simulator_configuration"
    )
    constants = _require_mapping(
        configuration.get("simulator_constants"), "simulator_constants"
    )
    ordering = _require_mapping(branch_state.get("branch_ordering"), "branch_ordering")
    state = CartesianState2D(
        x=_require_finite(state_document.get("position_x"), "state.position_x"),
        y=_require_finite(state_document.get("position_y"), "state.position_y"),
        vx=_require_finite(state_document.get("velocity_x"), "state.velocity_x"),
        vy=_require_finite(state_document.get("velocity_y"), "state.velocity_y"),
    )
    if not implementation_commit:
        raise StagedRecoveryLoggerAdapterError("implementation_commit is required")
    identity = BoundedRuntimeIdentity(
        case_id=str(branch_state.get("case_id")),
        seed=int(branch_state.get("seed")),
        branch_id=branch_id,
        branch_state_hash=str(branch_state.get("canonical_branch_state_hash")),
        simulator_configuration_hash=str(
            branch_state.get("simulator_configuration_hash")
        ),
        constants_hash=str(branch_state.get("simulator_constants_hash")),
        branch_step=int(branch_state.get("branch_step")),
        nominal_prefix_transition_count=int(
            ordering.get("realized_prefix_transition_count")
        ),
        dt=_require_finite(constants.get("dt"), "simulator_constants.dt"),
        target_radius=_require_finite(
            constants.get("target_radius"), "simulator_constants.target_radius"
        ),
        mu=_require_finite(constants.get("mu"), "simulator_constants.mu"),
        ratio_denominator_epsilon=_require_finite(
            constants.get("speed_ratio_denominator_epsilon"),
            "simulator_constants.speed_ratio_denominator_epsilon",
        ),
        action_component_limit=_require_finite(
            constants.get("action_component_max"),
            "simulator_constants.action_component_max",
        ),
        total_horizon=int(constants.get("max_steps")),
        overspeed_threshold=_require_finite(
            branch_state.get("threshold"), "threshold"
        ),
        implementation_commit=implementation_commit,
    )
    return identity, state


def _realized_speed_ratio(
    state: CartesianState2D | None,
    branch_state: Mapping[str, object],
) -> float | None:
    if state is None:
        return None
    configuration = _require_mapping(
        branch_state.get("simulator_configuration"), "simulator_configuration"
    )
    constants = _require_mapping(
        configuration.get("simulator_constants"), "simulator_constants"
    )
    target_speed = _require_finite(
        constants.get("target_circular_speed"), "target_circular_speed"
    )
    epsilon = _require_finite(
        constants.get("speed_ratio_denominator_epsilon"),
        "speed_ratio_denominator_epsilon",
    )
    return math.hypot(state.vx, state.vy) / (target_speed + epsilon)


def _action_disposition(
    execution: RecoveryBranchExecutionResult,
) -> ActionDisposition:
    if execution.terminal_reason == "explicit_recovery_abort":
        return ActionDisposition.NO_ACTION
    if execution.terminal_reason == "recovery_action_rejected":
        return ActionDisposition.REJECTED
    if execution.executed and execution.action == (0.0, 0.0):
        return ActionDisposition.ZERO_ACTION_EXECUTED
    if execution.executed:
        decision = execution.monitor_decision
        if decision is not None and decision.executed_action == execution.action:
            return ActionDisposition.EXECUTED_UNCHANGED
        return ActionDisposition.EXECUTED_MODIFIED
    return ActionDisposition.NOT_EVALUATED


def _notify_observer(
    observer: RuntimeObserver | None,
    snapshot: BoundedRuntimeSnapshot,
    *,
    physical_transitions: int,
) -> None:
    if observer is None:
        return
    try:
        result = observer(snapshot)
    except Exception as exc:
        raise ObserverIntegrationError(
            f"observer failed at event {snapshot.event_index}: "
            f"{type(exc).__name__}: {exc}",
            event_index=snapshot.event_index,
            recovery_step=snapshot.recovery_step,
            physical_transitions=physical_transitions,
        ) from exc
    if result is not None:
        raise ObserverIntegrationError(
            "observer return values are forbidden and cannot affect runtime control",
            event_index=snapshot.event_index,
            recovery_step=snapshot.recovery_step,
            physical_transitions=physical_transitions,
        )


def run_bounded_recovery_validation_path(
    branch_state: Mapping[str, object],
    *,
    branch_id: str,
    horizon_steps: int,
    implementation_commit: str,
    observer: RuntimeObserver | None = None,
    step_executor: StepExecutor = execute_recovery_branch,
) -> BoundedRecoveryValidationRun:
    if (
        isinstance(horizon_steps, bool)
        or not isinstance(horizon_steps, int)
        or horizon_steps < 1
        or horizon_steps > MAX_BOUNDED_VALIDATION_HORIZON
    ):
        raise StagedRecoveryLoggerAdapterError(
            "horizon_steps must be within the existing bounded range 1..32"
        )
    document = copy.deepcopy(dict(branch_state))
    identity, initial_state = build_bounded_runtime_identity(
        document,
        branch_id=branch_id,
        implementation_commit=implementation_commit,
    )
    initial_hash = runtime_state_hash(initial_state)
    initial_snapshot = BoundedRuntimeSnapshot(
        snapshot_type=RuntimeSnapshotType.INITIAL,
        event_index=0,
        identity=identity,
        recovery_step=0,
        total_transition_count=identity.nominal_prefix_transition_count,
        pre_state=initial_state,
        pre_state_hash=initial_hash,
    )
    snapshots: list[BoundedRuntimeSnapshot] = [initial_snapshot]
    transitions: list[BoundedRuntimeSnapshot] = []
    _notify_observer(observer, initial_snapshot, physical_transitions=0)

    current_state = initial_state
    recovery_count = 0
    runtime_terminal_reason: str | None = None
    final_stop_report: RecoveryStopConditionReport | None = None
    for attempted_step in range(1, horizon_steps + 1):
        try:
            execution = step_executor(
                document,
                branch_id,
                horizon_steps=1,
                current_state=current_state,
            )
        except Exception as exc:
            raise StagedRecoveryLoggerAdapterError(
                f"bounded executor failed at recovery step {attempted_step}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(execution, RecoveryBranchExecutionResult):
            raise StagedRecoveryLoggerAdapterError(
                "step executor must return RecoveryBranchExecutionResult"
            )
        if execution.branch_id != branch_id or not execution.valid:
            raise StagedRecoveryLoggerAdapterError("executor returned invalid branch identity")
        if (
            execution.previous_state != current_state
            or execution.previous_state_hash != runtime_state_hash(current_state)
        ):
            raise StagedRecoveryLoggerAdapterError(
                "executor did not use the current continuation state"
            )

        previous_state = current_state
        if execution.executed:
            if execution.transition_count != 1 or execution.next_state is None:
                raise StagedRecoveryLoggerAdapterError(
                    "executed proposal must realize exactly one transition"
                )
            current_state = execution.next_state
            recovery_count += 1
        elif execution.transition_count != 0:
            raise StagedRecoveryLoggerAdapterError(
                "nonexecuted proposal cannot increment transition count"
            )

        realized_ratio = _realized_speed_ratio(execution.next_state, document)
        final_stop_report = evaluate_recovery_stop_conditions(
            execution_terminal_reason=execution.terminal_reason,
            next_state=execution.next_state,
            realized_speed_ratio=realized_ratio,
            overspeed_threshold=identity.overspeed_threshold,
            recovery_transition_count=recovery_count,
            recovery_horizon_steps=horizon_steps,
            total_transition_count=(
                identity.nominal_prefix_transition_count + recovery_count
            ),
            total_horizon_steps=identity.total_horizon,
        )
        runtime_terminal_reason = final_stop_report.terminal_reason
        monitor = execution.monitor_decision
        predicted_state = execution.predicted_nominal_state
        disposition = _action_disposition(execution)
        transition_snapshot = BoundedRuntimeSnapshot(
            snapshot_type=RuntimeSnapshotType.TRANSITION,
            event_index=len(snapshots),
            identity=identity,
            recovery_step=recovery_count,
            total_transition_count=(
                identity.nominal_prefix_transition_count + recovery_count
            ),
            pre_state=previous_state,
            pre_state_hash=execution.previous_state_hash,
            proposed_action=execution.action,
            executed_action=execution.action if execution.executed else None,
            action_disposition=disposition,
            transition_executed=execution.executed,
            monitor_id=monitor.monitor_id if monitor is not None else None,
            monitor_decision=monitor.decision if monitor is not None else None,
            monitor_reason=monitor.reason if monitor is not None else None,
            monitor_threshold=monitor.threshold if monitor is not None else None,
            monitor_comparator=monitor.comparator if monitor is not None else None,
            predicted_next_state=predicted_state,
            predicted_state_hash=(
                runtime_state_hash(predicted_state) if predicted_state is not None else None
            ),
            predicted_speed_ratio=(
                monitor.predicted_nominal_speed_ratio if monitor is not None else None
            ),
            realized_next_state=execution.next_state,
            realized_state_hash=execution.next_state_hash,
            realized_speed_ratio=realized_ratio,
            evaluator_statuses=final_stop_report.statuses,
            runtime_terminal_reason=runtime_terminal_reason,
        )
        snapshots.append(transition_snapshot)
        transitions.append(transition_snapshot)
        _notify_observer(
            observer,
            transition_snapshot,
            physical_transitions=recovery_count,
        )
        if runtime_terminal_reason is not None:
            break

    if final_stop_report is None or runtime_terminal_reason is None:
        raise StagedRecoveryLoggerAdapterError(
            "bounded validation path ended without a terminal condition"
        )
    if recovery_count != horizon_steps:
        # Earlier scientific/runtime stops are valid for the bounded runtime but
        # do not satisfy the frozen Stage 0C eight-transition trace contract.
        raise StagedRecoveryLoggerAdapterError(
            f"bounded validation terminated after {recovery_count} transitions "
            f"with {runtime_terminal_reason!r}; expected {horizon_steps}"
        )
    terminal_snapshot = BoundedRuntimeSnapshot(
        snapshot_type=RuntimeSnapshotType.TERMINAL,
        event_index=len(snapshots),
        identity=identity,
        recovery_step=recovery_count,
        total_transition_count=identity.nominal_prefix_transition_count + recovery_count,
        pre_state=current_state,
        pre_state_hash=runtime_state_hash(current_state),
        action_disposition=ActionDisposition.NO_ACTION,
        evaluator_statuses=final_stop_report.statuses,
        runtime_terminal_reason=runtime_terminal_reason,
        validation_terminal_reason=VALIDATION_TERMINAL_REASON,
    )
    snapshots.append(terminal_snapshot)
    _notify_observer(
        observer,
        terminal_snapshot,
        physical_transitions=recovery_count,
    )
    validate_branch_state_integrity(document)
    return BoundedRecoveryValidationRun(
        identity=identity,
        initial_state=initial_state,
        initial_state_hash=initial_hash,
        snapshots=tuple(snapshots),
        transition_snapshots=tuple(transitions),
        recovery_transition_count=recovery_count,
        final_state=current_state,
        final_state_hash=runtime_state_hash(current_state),
        runtime_terminal_reason=runtime_terminal_reason,
        validation_terminal_reason=VALIDATION_TERMINAL_REASON,
    )


def load_and_run_bounded_recovery_validation_path(
    branch_state_path: str | Path,
    *,
    branch_id: str,
    horizon_steps: int,
    implementation_commit: str,
    observer: RuntimeObserver | None = None,
    branch_state_loader: Callable[[str | Path], Mapping[str, object]] = (
        load_frozen_branch_state
    ),
    step_executor: StepExecutor = execute_recovery_branch,
) -> BoundedRecoveryValidationRun:
    branch_state = branch_state_loader(branch_state_path)
    return run_bounded_recovery_validation_path(
        branch_state,
        branch_id=branch_id,
        horizon_steps=horizon_steps,
        implementation_commit=implementation_commit,
        observer=observer,
        step_executor=step_executor,
    )


def _instrumentation_state(state: CartesianState2D) -> InstrumentationState2D:
    return InstrumentationState2D(state.x, state.y, state.vx, state.vy)


def _orbital_configuration(identity: BoundedRuntimeIdentity) -> OrbitalConfiguration:
    return OrbitalConfiguration(
        mu=identity.mu,
        target_radius=identity.target_radius,
        ratio_denominator_epsilon=identity.ratio_denominator_epsilon,
        speed_ratio_denominator_epsilon=identity.ratio_denominator_epsilon,
        action_component_limit=identity.action_component_limit,
    )


def _status_map(snapshot: BoundedRuntimeSnapshot) -> dict[str, str]:
    return dict(snapshot.evaluator_statuses)


def _runtime_evidence(
    snapshot: BoundedRuntimeSnapshot,
) -> tuple[tuple[str, object], ...]:
    statuses = _status_map(snapshot)
    values: dict[str, object] = {
        "total_horizon_remaining": measured_value(
            snapshot.identity.total_horizon - snapshot.total_transition_count,
            units="transition",
            source_id="bounded_runtime.total_horizon_remaining",
            source_step=snapshot.recovery_step,
        )
    }
    if snapshot.monitor_decision is not None:
        values["final_veto_decision"] = measured_value(
            snapshot.monitor_decision,
            units="categorical",
            source_id="bounded_runtime.final_veto_decision",
            source_step=snapshot.recovery_step,
        )
    if statuses:
        values["simulation_validity"] = measured_value(
            statuses.get(INVALID_SIMULATION) == CLEAR,
            units="boolean",
            source_id="bounded_runtime.stop_evidence",
            source_step=snapshot.recovery_step,
        )
        values["recovery_evaluation_validity"] = measured_value(
            statuses.get(INVALID_RECOVERY_EVALUATION) == CLEAR,
            units="boolean",
            source_id="bounded_runtime.stop_evidence",
            source_step=snapshot.recovery_step,
        )
        if statuses.get(OVERSPEED) in {CLEAR, TRIGGERED}:
            values["overspeed_status"] = measured_value(
                statuses[OVERSPEED] == TRIGGERED,
                units="boolean",
                source_id="bounded_runtime.stop_evidence",
                source_step=snapshot.recovery_step,
            )
        values["action_rejection_status"] = measured_value(
            statuses.get(ACTION_REJECTED) == TRIGGERED,
            units="boolean",
            source_id="bounded_runtime.stop_evidence",
            source_step=snapshot.recovery_step,
        )
        values["explicit_abort_requested"] = measured_value(
            statuses.get(EXPLICIT_ABORT) == TRIGGERED,
            units="boolean",
            source_id="bounded_runtime.stop_evidence",
            source_step=snapshot.recovery_step,
        )
        values["recovery_horizon_exhausted"] = measured_value(
            statuses.get(RECOVERY_HORIZON_EXHAUSTED) == TRIGGERED,
            units="boolean",
            source_id="bounded_runtime.stop_evidence",
            source_step=snapshot.recovery_step,
        )
        values["total_horizon_exhausted"] = measured_value(
            statuses.get(TOTAL_HORIZON_EXHAUSTED) == TRIGGERED,
            units="boolean",
            source_id="bounded_runtime.stop_evidence",
            source_step=snapshot.recovery_step,
        )
    return tuple(sorted(values.items()))


class Stage0BValidationLoggerAdapter:
    """Map immutable bounded-runtime snapshots into the frozen Stage 0B logger."""

    def __init__(
        self,
        identity: BoundedRuntimeIdentity,
        *,
        session_id: str,
        max_events: int,
    ) -> None:
        self._identity = identity
        self._configuration = _orbital_configuration(identity)
        self._session = StagedRecoveryRuntimeLoggerSession(
            StagedRecoverySessionHeader(
                logger_schema_version=LOGGER_SCHEMA_VERSION,
                instrumentation_schema_version=INSTRUMENTATION_SCHEMA_VERSION,
                architecture_version=ARCHITECTURE_VERSION,
                session_id=session_id,
                case_id=identity.case_id,
                seed=identity.seed,
                implementation_commit=identity.implementation_commit,
                source_state_hash=identity.branch_state_hash,
                simulator_configuration_hash=identity.simulator_configuration_hash,
                constants_hash=identity.constants_hash,
                max_events=max_events,
                declared_output_purpose=(
                    "Stage 0C measured instrumentation validation event construction"
                ),
                execution_authorization_status=STAGED_EXECUTION_STATUS,
                scientific_claim_restrictions=CLAIM_RESTRICTIONS,
            )
        )
        self._terminal_runtime_reason: str | None = None
        self._terminal_validation_reason: str | None = None

    @property
    def events(self):
        return self._session.events

    @property
    def runtime_terminal_reason(self) -> str | None:
        return self._terminal_runtime_reason

    @property
    def validation_terminal_reason(self) -> str | None:
        return self._terminal_validation_reason

    def observe(self, snapshot: BoundedRuntimeSnapshot) -> None:
        if snapshot.identity != self._identity:
            raise StagedRecoveryLoggerAdapterError("snapshot identity changed")
        state = _instrumentation_state(snapshot.pre_state)
        simulation_time = snapshot.total_transition_count * self._identity.dt
        if snapshot.snapshot_type == RuntimeSnapshotType.INITIAL:
            self._session.record_initial_snapshot(
                StagedRecoveryInitialSnapshot(
                    event_index=snapshot.event_index,
                    recovery_step=snapshot.recovery_step,
                    total_transition_count=snapshot.total_transition_count,
                    state=state,
                    configuration=self._configuration,
                    simulation_time=simulation_time,
                )
            )
            return None
        if snapshot.snapshot_type == RuntimeSnapshotType.TRANSITION:
            delta = 1 if snapshot.transition_executed else 0
            pre_total = snapshot.total_transition_count - delta
            predicted = (
                _instrumentation_state(snapshot.predicted_next_state)
                if snapshot.predicted_next_state is not None
                else None
            )
            realized = (
                _instrumentation_state(snapshot.realized_next_state)
                if snapshot.realized_next_state is not None
                else None
            )
            self._session.record_transition(
                StagedRecoveryTransitionInput(
                    event_index=snapshot.event_index,
                    recovery_step=snapshot.recovery_step,
                    total_transition_count=snapshot.total_transition_count,
                    pre_state=state,
                    configuration=self._configuration,
                    proposed_action=snapshot.proposed_action,
                    executed_action=snapshot.executed_action,
                    action_disposition=snapshot.action_disposition,
                    transition_executed=snapshot.transition_executed,
                    realized_next_state=realized,
                    monitor_decision=snapshot.monitor_decision,
                    predicted_next_state=predicted,
                    simulation_time=pre_total * self._identity.dt,
                    next_simulation_time=simulation_time,
                    # Stage 0A crossing steps use the post-branch counter; the
                    # canonical branch point is recovery step zero.
                    branch_step=0,
                    runtime_evidence=_runtime_evidence(snapshot),
                    evidence_level="measured_bounded_runtime_and_pure_derivation",
                )
            )
            return None
        if snapshot.snapshot_type == RuntimeSnapshotType.TERMINAL:
            if snapshot.validation_terminal_reason is None:
                raise StagedRecoveryLoggerAdapterError(
                    "terminal snapshot lacks validation terminal reason"
                )
            self._terminal_runtime_reason = snapshot.runtime_terminal_reason
            self._terminal_validation_reason = snapshot.validation_terminal_reason
            self._session.record_terminal(
                StagedRecoveryTerminalInput(
                    event_index=snapshot.event_index,
                    recovery_step=snapshot.recovery_step,
                    total_transition_count=snapshot.total_transition_count,
                    terminal_reason=snapshot.validation_terminal_reason,
                    action_disposition=ActionDisposition.NO_ACTION,
                    current_state=state,
                    configuration=self._configuration,
                    simulation_time=simulation_time,
                    runtime_evidence=_runtime_evidence(snapshot),
                    evidence_level="externally_supplied_validation_terminal_evidence",
                )
            )
            return None
        raise StagedRecoveryLoggerAdapterError("unsupported runtime snapshot type")

    def finalize(self) -> StagedRecoveryTraceBundle:
        if self._terminal_runtime_reason is None or self._terminal_validation_reason is None:
            raise StagedRecoveryLoggerAdapterError(
                "logger adapter cannot finalize before terminal evidence"
            )
        return self._session.finalize()


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "MAX_BOUNDED_VALIDATION_HORIZON",
    "VALIDATION_TERMINAL_REASON",
    "BoundedRecoveryValidationRun",
    "BoundedRuntimeIdentity",
    "BoundedRuntimeSnapshot",
    "ObserverIntegrationError",
    "RuntimeObserver",
    "RuntimeSnapshotType",
    "Stage0BValidationLoggerAdapter",
    "StagedRecoveryLoggerAdapterError",
    "build_bounded_runtime_identity",
    "load_and_run_bounded_recovery_validation_path",
    "run_bounded_recovery_validation_path",
    "runtime_state_hash",
]
