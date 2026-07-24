from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from runtime_assurance.recovery_branch_executor import (
    RecoveryBranchExecutionResult,
    execute_recovery_branch,
    load_frozen_branch_state,
    validate_branch_state_integrity,
)
from runtime_assurance.recovery_evaluators import (
    EVALUATION_CLEAR,
    EVALUATION_INVALID,
    EVALUATION_TRIGGERED,
    INSTABILITY_EVALUATOR_ID,
    PHASE34_RECOVERABILITY_EVALUATOR_ID,
    RECOVERY_SUCCESS_EVALUATOR_ID,
    UNSAFE_STATE_EVALUATOR_ID,
    RecoveryEvaluationResult,
    evaluate_instability,
    evaluate_phase34_compatible_recoverability,
    evaluate_recovery_success_v0,
    evaluate_unsafe_state,
)
from runtime_assurance.recovery_experiment_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    DECISION_EVENT_SCHEMA_VERSION,
    RecoveryArtifactContract,
    RecoveryArtifactWriteResult,
    RecoveryBranchExperimentRecord,
    RecoveryDecisionEvent,
    RecoveryExperimentBundle,
    canonical_sha256,
    publish_recovery_experiment_bundle,
    validate_recovery_experiment_bundle,
    with_record_payload_hash,
)
from runtime_assurance.recovery_experiment_preflight import (
    BRANCH_STATE_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    RecoveryExperimentPreflightReport,
    find_repository_root,
    run_recovery_experiment_preflight,
)
from runtime_assurance.recovery_stop_conditions import (
    ACTION_REJECTED,
    EXPLICIT_ABORT,
    INSTABILITY,
    INVALID_RECOVERY_EVALUATION,
    INVALID_SIMULATION,
    OVERSPEED,
    RECOVERY_HORIZON_EXHAUSTED,
    RECOVERY_SUCCESS,
    TOTAL_HORIZON_EXHAUSTED,
    UNSAFE_STATE,
    RecoveryStopConditionReport,
    evaluate_recovery_stop_conditions,
)
from simulator.phase34_35_transition import CartesianState2D


RUNNER_ID = "recovery_experiment_runner_v0"

# Exact Phase21/34/35 simulator-success contract. This is evaluation state,
# not transition physics, and is kept separate from Recovery Success v0.
STRICT_RADIUS_ERROR_MAX = 1.0e-3
STRICT_SPEED_ERROR_MAX = 1.0e-3
STRICT_ALIGNMENT_ERROR_MAX = 0.02
STRICT_SUCCESS_PERSISTENCE_STEPS = 200

RADIAL_STALL_RADIUS_ERROR_MAX = 0.10
RADIAL_STALL_SPEED_ERROR_MAX = 0.12
RADIAL_STALL_ALIGNMENT_MIN = 0.85
RADIAL_STALL_PERSISTENCE_STEPS = 800
OUT_RANGE_RADIUS_MULTIPLIER = 2.5
TOO_CLOSE_RADIUS_MULTIPLIER = 0.35

EVALUATOR_VERSIONS = tuple(
    sorted(
        (
            INSTABILITY_EVALUATOR_ID,
            PHASE34_RECOVERABILITY_EVALUATOR_ID,
            RECOVERY_SUCCESS_EVALUATOR_ID,
            UNSAFE_STATE_EVALUATOR_ID,
        )
    )
)

_RUNTIME_TO_BRANCH_TERMINAL = {
    INVALID_SIMULATION: "invalid_simulation",
    INVALID_RECOVERY_EVALUATION: "invalid_recovery_evaluation",
    OVERSPEED: "overspeed",
    INSTABILITY: "instability",
    UNSAFE_STATE: "unsafe_state",
    ACTION_REJECTED: "recovery_action_rejected",
    EXPLICIT_ABORT: "explicit_recovery_abort",
    RECOVERY_SUCCESS: "recovery_success",
    RECOVERY_HORIZON_EXHAUSTED: "recovery_horizon_exhausted",
    TOTAL_HORIZON_EXHAUSTED: "total_horizon_exhausted",
}

_RUNTIME_TO_CONTROLLED_TERMINAL = {
    INVALID_SIMULATION: "invalid_simulation",
    INVALID_RECOVERY_EVALUATION: "unknown_with_manual_audit",
    OVERSPEED: "overspeed",
    INSTABILITY: "instability",
    UNSAFE_STATE: "unsafe_state",
    ACTION_REJECTED: "unknown_with_manual_audit",
    EXPLICIT_ABORT: "unknown_with_manual_audit",
    RECOVERY_SUCCESS: "success",
    RECOVERY_HORIZON_EXHAUSTED: "timeout",
    TOTAL_HORIZON_EXHAUSTED: "timeout",
}


class RecoveryExperimentExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        branch_id: str | None = None,
        attempted_recovery_step: int | None = None,
        realized_transitions: int = 0,
    ) -> None:
        super().__init__(message)
        self.branch_id = branch_id
        self.attempted_recovery_step = attempted_recovery_step
        self.realized_transitions = realized_transitions


@dataclass(frozen=True, slots=True)
class RecoveryExperimentExecutionResult:
    preflight_report: RecoveryExperimentPreflightReport
    bundle: RecoveryExperimentBundle
    publication: RecoveryArtifactWriteResult


@dataclass(frozen=True, slots=True)
class _OrbitalObservation:
    radius: float
    speed: float
    radial_velocity: float
    tangential_velocity: float
    r_error_ratio: float
    vr_ratio: float
    vt_error_ratio: float
    speed_ratio: float
    strict_speed_error_ratio: float
    alignment_error: float


@dataclass(frozen=True, slots=True)
class _BranchRun:
    record: RecoveryBranchExperimentRecord
    events: tuple[RecoveryDecisionEvent, ...]


StepExecutor = Callable[..., RecoveryBranchExecutionResult]
BranchStateLoader = Callable[[], dict[str, object]]


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RecoveryExperimentExecutionError(f"{name} must be a JSON object")
    return value


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveryExperimentExecutionError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise RecoveryExperimentExecutionError(f"{name} must be a finite number")
    return converted


def _state_from_branch_state(branch_state: Mapping[str, object]) -> CartesianState2D:
    state = _require_mapping(branch_state.get("state"), "branch_state.state")
    return CartesianState2D(
        x=_require_finite(state.get("position_x"), "state.position_x"),
        y=_require_finite(state.get("position_y"), "state.position_y"),
        vx=_require_finite(state.get("velocity_x"), "state.velocity_x"),
        vy=_require_finite(state.get("velocity_y"), "state.velocity_y"),
    )


def _state_hash(state: CartesianState2D) -> str:
    return canonical_sha256(
        {
            "position_x": state.x,
            "position_y": state.y,
            "velocity_x": state.vx,
            "velocity_y": state.vy,
        }
    )


def _simulator_values(
    branch_state: Mapping[str, object],
) -> tuple[float, float, float, float, float]:
    simulator = _require_mapping(
        branch_state.get("simulator_configuration"),
        "simulator_configuration",
    )
    constants = _require_mapping(
        simulator.get("simulator_constants"),
        "simulator_configuration.simulator_constants",
    )
    return (
        _require_finite(constants.get("target_radius"), "target_radius"),
        _require_finite(
            constants.get("target_circular_speed"), "target_circular_speed"
        ),
        _require_finite(
            constants.get("speed_ratio_denominator_epsilon"),
            "speed_ratio_denominator_epsilon",
        ),
        _require_finite(constants.get("dt"), "dt"),
        _require_finite(simulator.get("thrust_scale"), "thrust_scale")
        / _require_finite(constants.get("mass"), "mass"),
    )


def _observe(
    state: CartesianState2D,
    *,
    target_radius: float,
    target_circular_speed: float,
    speed_ratio_epsilon: float,
) -> _OrbitalObservation:
    values = (state.x, state.y, state.vx, state.vy)
    if not all(math.isfinite(value) for value in values):
        raise RecoveryExperimentExecutionError("continuation state is nonfinite")
    radius = math.hypot(state.x, state.y)
    speed = math.hypot(state.vx, state.vy)
    inverse_radius = 1.0 / (radius + 1.0e-12)
    radial_x = state.x * inverse_radius
    radial_y = state.y * inverse_radius
    tangential_x = -radial_y
    tangential_y = radial_x
    radial_velocity = state.vx * radial_x + state.vy * radial_y
    tangential_velocity = state.vx * tangential_x + state.vy * tangential_y
    r_error_ratio = (radius - target_radius) / target_radius
    vr_ratio = radial_velocity / (target_circular_speed + 1.0e-12)
    vt_error_ratio = (
        tangential_velocity - target_circular_speed
    ) / (target_circular_speed + 1.0e-12)
    speed_ratio = speed / (target_circular_speed + speed_ratio_epsilon)
    strict_speed_error_ratio = abs(
        speed - target_circular_speed
    ) / target_circular_speed
    unit_radius_x = state.x / (radius + 1.0e-8)
    unit_radius_y = state.y / (radius + 1.0e-8)
    unit_velocity_x = state.vx / (speed + 1.0e-8)
    unit_velocity_y = state.vy / (speed + 1.0e-8)
    alignment_error = abs(
        unit_radius_x * unit_velocity_x + unit_radius_y * unit_velocity_y
    )
    result = _OrbitalObservation(
        radius=radius,
        speed=speed,
        radial_velocity=radial_velocity,
        tangential_velocity=tangential_velocity,
        r_error_ratio=r_error_ratio,
        vr_ratio=vr_ratio,
        vt_error_ratio=vt_error_ratio,
        speed_ratio=speed_ratio,
        strict_speed_error_ratio=strict_speed_error_ratio,
        alignment_error=alignment_error,
    )
    if not all(
        math.isfinite(value)
        for value in (
            result.radius,
            result.speed,
            result.radial_velocity,
            result.tangential_velocity,
            result.r_error_ratio,
            result.vr_ratio,
            result.vt_error_ratio,
            result.speed_ratio,
            result.strict_speed_error_ratio,
            result.alignment_error,
        )
    ):
        raise RecoveryExperimentExecutionError("orbital observation is nonfinite")
    return result


def _strict_success_state(observation: _OrbitalObservation) -> bool:
    return (
        abs(observation.r_error_ratio) < STRICT_RADIUS_ERROR_MAX
        and observation.strict_speed_error_ratio < STRICT_SPEED_ERROR_MAX
        and observation.alignment_error < STRICT_ALIGNMENT_ERROR_MAX
    )


def _legacy_instability(
    observation: _OrbitalObservation,
    *,
    target_radius: float,
    radial_stall_count: int,
) -> tuple[bool, str | None, int]:
    radial_stall = (
        abs(observation.r_error_ratio) < RADIAL_STALL_RADIUS_ERROR_MAX
        and observation.strict_speed_error_ratio < RADIAL_STALL_SPEED_ERROR_MAX
        and observation.alignment_error > RADIAL_STALL_ALIGNMENT_MIN
    )
    next_stall_count = radial_stall_count + 1 if radial_stall else 0
    if observation.radius > OUT_RANGE_RADIUS_MULTIPLIER * target_radius:
        return True, "out_range", next_stall_count
    if observation.radius < TOO_CLOSE_RADIUS_MULTIPLIER * target_radius:
        return True, "too_close", next_stall_count
    if next_stall_count >= RADIAL_STALL_PERSISTENCE_STEPS:
        return True, "radial_stall", next_stall_count
    return False, None, next_stall_count


def _status_map(report: RecoveryStopConditionReport) -> dict[str, str]:
    return dict(report.statuses)


def _event_evaluator_statuses(
    report: RecoveryStopConditionReport,
) -> tuple[tuple[str, str], ...]:
    statuses = _status_map(report)
    return tuple(
        sorted(
            (
                ("instability", statuses[INSTABILITY]),
                ("invalid_recovery_evaluation", statuses[INVALID_RECOVERY_EVALUATION]),
                ("invalid_simulation", statuses[INVALID_SIMULATION]),
                ("overspeed", statuses[OVERSPEED]),
                ("recovery_success", statuses[RECOVERY_SUCCESS]),
                ("unsafe_state", statuses[UNSAFE_STATE]),
            )
        )
    )


def _recovery_outcome(terminal_reason: str, recovery_success: bool) -> str:
    if recovery_success:
        return "hazard_avoided_and_task_recovered"
    if terminal_reason == OVERSPEED:
        return "hazard_not_avoided"
    if terminal_reason == EXPLICIT_ABORT:
        return "hazard_avoided_through_termination"
    if terminal_reason == ACTION_REJECTED:
        return "recovery_action_rejected"
    if terminal_reason in {INVALID_SIMULATION, INVALID_RECOVERY_EVALUATION}:
        return "invalid_evaluation"
    if terminal_reason in {INSTABILITY, UNSAFE_STATE}:
        return "hazard_avoided_with_adverse_state_outcome"
    return "hazard_avoided_task_stalled"


def _evaluate_boundary(
    *,
    execution_terminal_reason: str,
    state: CartesianState2D,
    observation: _OrbitalObservation,
    contract: RecoveryArtifactContract,
    recovery_transition_count: int,
    crossed: bool,
    crossing_step: int | None,
    recoverable_crossing: bool,
    instability_flag: bool,
    instability_reason: str | None,
    simulator_success: bool,
    explicit_abort: bool,
    action_rejected: bool,
) -> tuple[
    RecoveryStopConditionReport,
    RecoveryEvaluationResult,
    RecoveryEvaluationResult,
    RecoveryEvaluationResult,
]:
    total_transition_count = (
        contract.nominal_prefix_transition_count + recovery_transition_count
    )
    recovery_evaluation = evaluate_recovery_success_v0(
        declared_overspeed_occurred=observation.speed_ratio > contract.hazard_threshold,
        simulation_valid=True,
        recovery_evaluation_valid=True,
        target_radius_crossing=crossed,
        phase34_compatible_recoverable_crossing=recoverable_crossing,
        branch_transition_count=recovery_transition_count,
        recovery_horizon=contract.recovery_horizon,
        branch_step=contract.branch_step,
        crossing_step=crossing_step,
        explicit_abort=explicit_abort,
        action_rejected=action_rejected,
        evaluated_step=total_transition_count,
        evidence_level="measured",
    )
    instability_evaluation = evaluate_instability(
        instability_flag=instability_flag,
        termination_reason=instability_reason,
        evaluated_step=total_transition_count,
        evidence_level="measured",
    )
    # Final Veto v0 already maps the same legacy instability evidence into its
    # explicit unsafe_state field. This preserves that existing scoped mapping.
    unsafe_evaluation = evaluate_unsafe_state(
        unsafe_state_flag=instability_flag,
        overspeed=observation.speed_ratio > contract.hazard_threshold,
        invalid_simulation=False,
        action_rejected=action_rejected,
        explicit_abort=explicit_abort,
        simulator_success=simulator_success,
        evaluated_step=total_transition_count,
        evidence_level="measured",
    )
    stop_report = evaluate_recovery_stop_conditions(
        execution_terminal_reason=execution_terminal_reason,
        next_state=state,
        realized_speed_ratio=observation.speed_ratio,
        overspeed_threshold=contract.hazard_threshold,
        recovery_transition_count=recovery_transition_count,
        recovery_horizon_steps=contract.recovery_horizon,
        total_transition_count=total_transition_count,
        total_horizon_steps=contract.total_horizon,
        recovery_success_evaluation=recovery_evaluation,
        instability_evaluation=instability_evaluation,
        unsafe_state_evaluation=unsafe_evaluation,
    )
    return (
        stop_report,
        recovery_evaluation,
        instability_evaluation,
        unsafe_evaluation,
    )


def _run_branch(
    *,
    branch_state: Mapping[str, object],
    branch_id: str,
    contract: RecoveryArtifactContract,
    step_executor: StepExecutor,
) -> _BranchRun:
    validate_branch_state_integrity(branch_state)
    if branch_state.get("canonical_branch_state_hash") != contract.branch_state_hash:
        raise RecoveryExperimentExecutionError(
            "branch-state hash differs from the frozen artifact contract",
            branch_id=branch_id,
        )
    current_state = _state_from_branch_state(branch_state)
    (
        target_radius,
        target_circular_speed,
        speed_ratio_epsilon,
        dt,
        thrust_acceleration_scale,
    ) = _simulator_values(branch_state)
    current_observation = _observe(
        current_state,
        target_radius=target_radius,
        target_circular_speed=target_circular_speed,
        speed_ratio_epsilon=speed_ratio_epsilon,
    )
    if _strict_success_state(current_observation):
        raise RecoveryExperimentExecutionError(
            "frozen branch point unexpectedly begins inside strict success",
            branch_id=branch_id,
        )

    recovery_count = 0
    events: list[RecoveryDecisionEvent] = []
    crossed = False
    first_crossing_step: int | None = None
    recoverable_crossing = False
    first_recoverable_step: int | None = None
    success_counter = 0
    simulator_success = False
    radial_stall_counter = 0
    normalized_effort = 0.0
    delta_v_proxy = 0.0
    max_action_component = 0.0
    monitor_evaluations = 0
    allow_count = 0
    veto_count = 0
    first_intervention_step: int | None = None
    last_intervention_step: int | None = None
    longest_veto_streak = 0
    veto_segment_count = 0
    action_rejection_count = 0
    terminal_reason: str | None = None
    final_stop_report: RecoveryStopConditionReport | None = None
    final_recovery_evaluation: RecoveryEvaluationResult | None = None

    while terminal_reason is None:
        attempted_step = recovery_count + 1
        if attempted_step > contract.recovery_horizon + 1:
            raise RecoveryExperimentExecutionError(
                "runner attempted a transition after the frozen recovery horizon",
                branch_id=branch_id,
                attempted_recovery_step=attempted_step,
                realized_transitions=recovery_count,
            )
        try:
            execution = step_executor(
                branch_state,
                branch_id,
                horizon_steps=1,
                current_state=current_state,
            )
        except Exception as exc:
            raise RecoveryExperimentExecutionError(
                f"branch execution failed: {type(exc).__name__}: {exc}",
                branch_id=branch_id,
                attempted_recovery_step=attempted_step,
                realized_transitions=recovery_count,
            ) from exc
        if execution.branch_id != branch_id or not execution.valid:
            raise RecoveryExperimentExecutionError(
                "one-step executor returned an invalid branch result",
                branch_id=branch_id,
                attempted_recovery_step=attempted_step,
                realized_transitions=recovery_count,
            )
        if (
            execution.previous_state != current_state
            or execution.previous_state_hash != _state_hash(current_state)
        ):
            raise RecoveryExperimentExecutionError(
                "one-step executor did not use the current continuation state",
                branch_id=branch_id,
                attempted_recovery_step=attempted_step,
                realized_transitions=recovery_count,
            )

        explicit_abort = execution.terminal_reason == "explicit_recovery_abort"
        action_rejected = execution.terminal_reason == "recovery_action_rejected"
        monitor_decision = execution.monitor_decision
        if not explicit_abort:
            if execution.action is None:
                raise RecoveryExperimentExecutionError(
                    "physical branch did not propose an action",
                    branch_id=branch_id,
                    attempted_recovery_step=attempted_step,
                    realized_transitions=recovery_count,
                )
            max_action_component = max(
                max_action_component,
                abs(execution.action[0]),
                abs(execution.action[1]),
            )
            monitor_evaluations += 1
            if monitor_decision is None:
                raise RecoveryExperimentExecutionError(
                    "physical branch lacks a Final Veto decision",
                    branch_id=branch_id,
                    attempted_recovery_step=attempted_step,
                    realized_transitions=recovery_count,
                )
            if monitor_decision.decision == "allow":
                if monitor_decision.predicted_nominal_speed_ratio > contract.hazard_threshold:
                    raise RecoveryExperimentExecutionError(
                        "Final Veto allowed a prediction above the frozen threshold",
                        branch_id=branch_id,
                        attempted_recovery_step=attempted_step,
                        realized_transitions=recovery_count,
                    )
                allow_count += 1
            elif monitor_decision.decision == "veto":
                if not (
                    monitor_decision.predicted_nominal_speed_ratio
                    > contract.hazard_threshold
                ):
                    raise RecoveryExperimentExecutionError(
                        "Final Veto rejected a prediction not above the frozen threshold",
                        branch_id=branch_id,
                        attempted_recovery_step=attempted_step,
                        realized_transitions=recovery_count,
                    )
                veto_count += 1
            else:
                raise RecoveryExperimentExecutionError(
                    "Final Veto returned an undeclared decision",
                    branch_id=branch_id,
                    attempted_recovery_step=attempted_step,
                    realized_transitions=recovery_count,
                )

        previous_state = current_state
        previous_observation = current_observation
        if action_rejected:
            action_rejection_count += 1
            first_intervention_step = attempted_step
            last_intervention_step = attempted_step
            longest_veto_streak = 1
            veto_segment_count = 1
            if execution.executed or execution.transition_count != 0:
                raise RecoveryExperimentExecutionError(
                    "rejected recovery action executed a physical transition",
                    branch_id=branch_id,
                    attempted_recovery_step=attempted_step,
                    realized_transitions=recovery_count,
                )
        elif explicit_abort:
            if execution.executed or execution.transition_count != 0:
                raise RecoveryExperimentExecutionError(
                    "explicit abort executed a physical transition",
                    branch_id=branch_id,
                    attempted_recovery_step=attempted_step,
                    realized_transitions=recovery_count,
                )
        else:
            if (
                not execution.executed
                or execution.transition_count != 1
                or execution.next_state is None
                or execution.action is None
            ):
                raise RecoveryExperimentExecutionError(
                    "allowed recovery proposal did not realize exactly one transition",
                    branch_id=branch_id,
                    attempted_recovery_step=attempted_step,
                    realized_transitions=recovery_count,
                )
            if monitor_decision is None or monitor_decision.decision != "allow":
                raise RecoveryExperimentExecutionError(
                    "transition occurred without an allow decision",
                    branch_id=branch_id,
                    attempted_recovery_step=attempted_step,
                    realized_transitions=recovery_count,
                )
            current_state = execution.next_state
            if execution.next_state_hash != _state_hash(current_state):
                raise RecoveryExperimentExecutionError(
                    "one-step executor next-state hash does not recompute",
                    branch_id=branch_id,
                    attempted_recovery_step=attempted_step,
                    realized_transitions=recovery_count,
                )
            if execution.predicted_nominal_state != current_state:
                raise RecoveryExperimentExecutionError(
                    "allowed prediction and realized transition differ",
                    branch_id=branch_id,
                    attempted_recovery_step=attempted_step,
                    realized_transitions=recovery_count,
                )
            recovery_count += 1
            current_observation = _observe(
                current_state,
                target_radius=target_radius,
                target_circular_speed=target_circular_speed,
                speed_ratio_epsilon=speed_ratio_epsilon,
            )
            action_norm = math.hypot(execution.action[0], execution.action[1])
            normalized_effort += action_norm
            delta_v_proxy += action_norm * thrust_acceleration_scale * dt
            current_total_step = contract.nominal_prefix_transition_count + recovery_count
            previous_error = previous_observation.radius - target_radius
            current_error = current_observation.radius - target_radius
            if not crossed and (
                (previous_error > 0.0 and current_error <= 0.0)
                or (previous_error < 0.0 and current_error >= 0.0)
            ):
                crossed = True
                first_crossing_step = current_total_step
            recoverability = evaluate_phase34_compatible_recoverability(
                r_error_ratio=current_observation.r_error_ratio,
                vr_ratio=current_observation.vr_ratio,
                vt_error_ratio=current_observation.vt_error_ratio,
                evaluated_step=current_total_step,
                evidence_level="measured",
            )
            if recoverability.status == EVALUATION_INVALID:
                execution = RecoveryBranchExecutionResult(
                    branch_id=branch_id,
                    executed=True,
                    action=execution.action,
                    previous_state_hash=execution.previous_state_hash,
                    next_state_hash=execution.next_state_hash,
                    terminal_reason="invalid_recovery_evaluation",
                    transition_count=1,
                    valid=True,
                    previous_state=execution.previous_state,
                    next_state=execution.next_state,
                    monitor_decision=execution.monitor_decision,
                    predicted_nominal_state=execution.predicted_nominal_state,
                    predicted_fallback_state=execution.predicted_fallback_state,
                )
            elif crossed and recoverability.status == EVALUATION_TRIGGERED:
                if not recoverable_crossing:
                    first_recoverable_step = current_total_step
                recoverable_crossing = True
            if _strict_success_state(current_observation):
                success_counter += 1
            else:
                success_counter = 0
            simulator_success = success_counter >= STRICT_SUCCESS_PERSISTENCE_STEPS

        instability_flag, instability_reason, radial_stall_counter = _legacy_instability(
            current_observation,
            target_radius=target_radius,
            radial_stall_count=radial_stall_counter,
        )
        (
            stop_report,
            recovery_evaluation,
            _instability_evaluation,
            _unsafe_evaluation,
        ) = _evaluate_boundary(
            execution_terminal_reason=execution.terminal_reason,
            state=current_state,
            observation=current_observation,
            contract=contract,
            recovery_transition_count=recovery_count,
            crossed=crossed,
            crossing_step=first_crossing_step,
            recoverable_crossing=recoverable_crossing,
            instability_flag=instability_flag,
            instability_reason=instability_reason,
            simulator_success=simulator_success,
            explicit_abort=explicit_abort,
            action_rejected=action_rejected,
        )
        terminal_reason = stop_report.terminal_reason
        final_stop_report = stop_report
        final_recovery_evaluation = recovery_evaluation

        predicted_state_hash = (
            _state_hash(execution.predicted_nominal_state)
            if execution.predicted_nominal_state is not None
            else None
        )
        realized_state_hash = (
            _state_hash(current_state) if execution.executed else None
        )
        predicted_ratio = (
            monitor_decision.predicted_nominal_speed_ratio
            if monitor_decision is not None
            else None
        )
        events.append(
            RecoveryDecisionEvent(
                decision_event_schema_version=DECISION_EVENT_SCHEMA_VERSION,
                experiment_id=contract.experiment_id,
                case_id=contract.case_id,
                seed=contract.seed,
                branch_id=branch_id,
                branch_state_hash=contract.branch_state_hash,
                event_index=len(events),
                post_branch_step=0 if explicit_abort else attempted_step,
                total_transition_count=(
                    contract.nominal_prefix_transition_count + recovery_count
                ),
                proposed_action=execution.action,
                final_veto_decision=(
                    "not_applicable"
                    if explicit_abort
                    else monitor_decision.decision
                    if monitor_decision is not None
                    else "not_applicable"
                ),
                executed_action=execution.action if execution.executed else None,
                transition_occurred=execution.executed,
                current_state_hash=execution.previous_state_hash,
                predicted_next_state_hash=predicted_state_hash,
                next_state_hash=realized_state_hash,
                predicted_speed_ratio=predicted_ratio,
                realized_speed_ratio=(
                    current_observation.speed_ratio if execution.executed else None
                ),
                hazard_threshold=contract.hazard_threshold,
                hazard_comparator=contract.hazard_comparator,
                triggered_stop_condition=terminal_reason,
                evaluator_statuses=_event_evaluator_statuses(stop_report),
                terminal_reason=terminal_reason,
                evidence_level="measured",
            )
        )

        if terminal_reason is None and recovery_count >= contract.recovery_horizon:
            raise RecoveryExperimentExecutionError(
                "frozen horizon was reached without a terminal stop report",
                branch_id=branch_id,
                attempted_recovery_step=attempted_step,
                realized_transitions=recovery_count,
            )

    if final_stop_report is None or final_recovery_evaluation is None:
        raise RecoveryExperimentExecutionError(
            "branch ended without evaluator evidence",
            branch_id=branch_id,
            realized_transitions=recovery_count,
        )
    statuses = _status_map(final_stop_report)
    recovery_success = final_recovery_evaluation.status == EVALUATION_TRIGGERED
    branch_terminal_label = _RUNTIME_TO_BRANCH_TERMINAL[terminal_reason]
    controlled_terminal_label = _RUNTIME_TO_CONTROLLED_TERMINAL[terminal_reason]
    intervention_rate = (
        veto_count / monitor_evaluations if monitor_evaluations else None
    )
    record = RecoveryBranchExperimentRecord(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        experiment_id=contract.experiment_id,
        case_id=contract.case_id,
        seed=contract.seed,
        branch_id=branch_id,
        branch_state_hash=contract.branch_state_hash,
        manifest_hash=contract.manifest_hash,
        implementation_commit=contract.implementation_commit,
        case_configuration_hash=contract.case_configuration_hash,
        simulator_configuration_hash=contract.simulator_configuration_hash,
        simulator_constants_hash=contract.simulator_constants_hash,
        branch_step=contract.branch_step,
        nominal_prefix_transition_count=contract.nominal_prefix_transition_count,
        recovery_transition_count=recovery_count,
        total_transition_count=contract.nominal_prefix_transition_count + recovery_count,
        recovery_horizon=contract.recovery_horizon,
        total_horizon=contract.total_horizon,
        hazard_threshold=contract.hazard_threshold,
        hazard_comparator=contract.hazard_comparator,
        stop_priority=contract.stop_priority,
        stop_priority_version=contract.stop_priority_version,
        terminal_reason=terminal_reason,
        branch_terminal_label=branch_terminal_label,
        controlled_terminal_label=controlled_terminal_label,
        recovery_outcome=_recovery_outcome(terminal_reason, recovery_success),
        valid=True,
        overspeed_status=statuses[OVERSPEED],
        instability_status=statuses[INSTABILITY],
        unsafe_state_status=statuses[UNSAFE_STATE],
        invalid_simulation_status=statuses[INVALID_SIMULATION],
        invalid_recovery_evaluation_status=statuses[INVALID_RECOVERY_EVALUATION],
        crossed_target_radius=crossed,
        first_crossing_step=first_crossing_step,
        phase34_compatible_recoverable_crossing=recoverable_crossing,
        first_recoverable_crossing_step=first_recoverable_step,
        recovery_success_status=statuses[RECOVERY_SUCCESS],
        recovery_success=recovery_success,
        recovery_success_step=(
            contract.nominal_prefix_transition_count + recovery_count
            if recovery_success
            else None
        ),
        final_simulator_success=simulator_success,
        overspeed_headroom=contract.hazard_threshold - current_observation.speed_ratio,
        action_saturation_margin=(
            None if monitor_evaluations == 0 else 1.0 - max_action_component
        ),
        available_correction_authority=None,
        required_to_available_correction_ratio=None,
        normalized_control_effort=(None if monitor_evaluations == 0 else normalized_effort),
        delta_v_proxy=(None if monitor_evaluations == 0 else delta_v_proxy),
        recovery_steps=recovery_count,
        crossing_delay=(
            first_crossing_step - contract.nominal_prefix_transition_count
            if first_crossing_step is not None
            else None
        ),
        final_radius_error=current_observation.radius - target_radius,
        final_radial_velocity_error=current_observation.radial_velocity,
        final_tangential_velocity_error=(
            current_observation.tangential_velocity - target_circular_speed
        ),
        task_abandonment_status=not recovery_success,
        monitor_evaluation_count=monitor_evaluations,
        intervention_count=veto_count,
        allow_count=allow_count,
        veto_count=veto_count,
        intervention_rate=intervention_rate,
        first_intervention_step=first_intervention_step,
        last_intervention_step=last_intervention_step,
        longest_intervention_streak=longest_veto_streak,
        veto_segment_count=veto_segment_count,
        action_suppression_duration=veto_count,
        recovery_action_rejection_count=action_rejection_count,
        action_rejected=terminal_reason == ACTION_REJECTED,
        rejected_action_executed=False,
        physical_transition_executed=recovery_count > 0,
        evaluator_versions=EVALUATOR_VERSIONS,
        result_payload_hash="",
    )
    return _BranchRun(
        record=with_record_payload_hash(record),
        events=tuple(events),
    )


def _build_frozen_recovery_bundle(
    *,
    contract: RecoveryArtifactContract,
    branch_state_loader: BranchStateLoader,
    step_executor: StepExecutor = execute_recovery_branch,
    is_synthetic: bool,
) -> RecoveryExperimentBundle:
    records: list[RecoveryBranchExperimentRecord] = []
    events: list[RecoveryDecisionEvent] = []
    for branch_id in contract.branch_ids:
        branch_state = branch_state_loader()
        validate_branch_state_integrity(branch_state)
        branch_run = _run_branch(
            branch_state=copy.deepcopy(branch_state),
            branch_id=branch_id,
            contract=contract,
            step_executor=step_executor,
        )
        records.append(branch_run.record)
        events.extend(branch_run.events)
    bundle = RecoveryExperimentBundle(
        records=tuple(records),
        decision_events=tuple(events),
        is_synthetic=is_synthetic,
    )
    report = validate_recovery_experiment_bundle(bundle, contract)
    if not report.valid:
        raise RecoveryExperimentExecutionError(
            "completed branch bundle failed validation: " + "; ".join(report.errors),
            realized_transitions=sum(record.recovery_transition_count for record in records),
        )
    return bundle


def run_frozen_recovery_experiment(
    *,
    manifest_path: str | Path,
    branch_state_path: str | Path,
    output_directory: str | Path,
) -> RecoveryExperimentExecutionResult:
    """Execute and atomically publish the one frozen nonformal experiment."""
    repository_root = find_repository_root(Path(manifest_path))
    expected_manifest = (repository_root / MANIFEST_RELATIVE_PATH).resolve()
    expected_branch_state = (repository_root / BRANCH_STATE_RELATIVE_PATH).resolve()
    supplied_manifest = Path(manifest_path).resolve()
    supplied_branch_state = Path(branch_state_path).resolve()
    if supplied_manifest != expected_manifest:
        raise RecoveryExperimentExecutionError("real execution requires the frozen manifest path")
    if supplied_branch_state != expected_branch_state:
        raise RecoveryExperimentExecutionError(
            "real execution requires the frozen branch-state path"
        )

    preflight, contract = run_recovery_experiment_preflight(
        repository_root=repository_root,
        require_clean_repository=True,
    )
    if not preflight.ready or contract is None:
        raise RecoveryExperimentExecutionError(
            "full recovery experiment preflight is not READY: "
            + ", ".join(preflight.failed_check_ids)
        )
    expected_output = (repository_root / contract.output_directory).resolve()
    if Path(output_directory).resolve() != expected_output:
        raise RecoveryExperimentExecutionError(
            "real execution requires the frozen output directory"
        )

    def loader() -> dict[str, object]:
        return load_frozen_branch_state(expected_branch_state)

    bundle = _build_frozen_recovery_bundle(
        contract=contract,
        branch_state_loader=loader,
        step_executor=execute_recovery_branch,
        is_synthetic=False,
    )
    publication = publish_recovery_experiment_bundle(
        bundle,
        contract,
        expected_output,
        repository_root=repository_root,
    )
    return RecoveryExperimentExecutionResult(
        preflight_report=preflight,
        bundle=bundle,
        publication=publication,
    )


__all__ = [
    "RUNNER_ID",
    "RecoveryExperimentExecutionError",
    "RecoveryExperimentExecutionResult",
    "run_frozen_recovery_experiment",
]
