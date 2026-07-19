from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

from runtime_assurance.final_veto_monitor import (
    MONITOR_ID,
    OVERSPEED_COMPARATOR,
    OVERSPEED_THRESHOLD,
    FinalVetoDecision,
    OneStepPrediction,
    evaluate_overspeed_veto,
)
from simulator.phase34_35_transition import (
    CartesianState2D,
    NormalizedAction2D,
    Phase3435DynamicsContext,
    Phase3435TransitionResult,
    step_phase34_35_transition,
)


BRANCH_STATE_SCHEMA_VERSION = "recovery_branch_state_v0"
FROZEN_SOURCE_CASE_ID = (
    "phase35_radial_energy_push_overspeed_stress_v0"
    "__r0_0p98__angle_150__thrust_8000"
)
DEFAULT_BRANCH_STATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "recovery_action_branching_nonformal_v0"
    / "branch_state.json"
)
ACTION_MAGNITUDE = 0.25
VECTOR_ZERO_TOLERANCE = 1.0e-12
SUPPORTED_BRANCH_IDS = frozenset(
    {
        "zero_action_reference_v0",
        "velocity_opposed_thrust_v0",
        "tangential_error_correction_v0",
        "explicit_abort_v0",
    }
)

Action2D = tuple[float, float]


class RecoveryBranchExecutorError(ValueError):
    pass


class BranchStateIntegrityError(RecoveryBranchExecutorError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryBranchExecutionResult:
    branch_id: str
    executed: bool
    action: Action2D | None
    previous_state_hash: str
    next_state_hash: str | None
    terminal_reason: str
    transition_count: int
    valid: bool
    previous_state: CartesianState2D
    next_state: CartesianState2D | None
    monitor_decision: FinalVetoDecision | None


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BranchStateIntegrityError(
            f"branch state is not canonical-JSON serializable: {exc}"
        ) from exc


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BranchStateIntegrityError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveryBranchExecutorError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise RecoveryBranchExecutorError(f"{name} must be a finite number")
    return converted


def _require_finite_state(state: CartesianState2D) -> None:
    for name, value in (
        ("state.x", state.x),
        ("state.y", state.y),
        ("state.vx", state.vx),
        ("state.vy", state.vy),
    ):
        _require_finite(value, name)


def load_frozen_branch_state(
    path: str | Path = DEFAULT_BRANCH_STATE_PATH,
) -> dict[str, object]:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"frozen branch state not found: {artifact_path}")
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BranchStateIntegrityError(
            f"frozen branch state is not valid UTF-8 JSON: {artifact_path}"
        ) from exc
    document = _require_mapping(data, "branch_state")
    validate_branch_state_integrity(document)
    return document


def validate_branch_state_integrity(branch_state: Mapping[str, object]) -> None:
    document = copy.deepcopy(dict(branch_state))
    supplied_hash = document.pop("canonical_branch_state_hash", None)
    if not isinstance(supplied_hash, str) or len(supplied_hash) != 64:
        raise BranchStateIntegrityError(
            "canonical_branch_state_hash must be a SHA-256 digest"
        )
    if supplied_hash != _canonical_hash(document):
        raise BranchStateIntegrityError("canonical branch-state hash mismatch")
    if branch_state.get("schema_version") != BRANCH_STATE_SCHEMA_VERSION:
        raise BranchStateIntegrityError("unsupported branch-state schema version")
    if branch_state.get("case_id") != FROZEN_SOURCE_CASE_ID:
        raise BranchStateIntegrityError("branch state does not use the frozen source case")
    if branch_state.get("threshold") != OVERSPEED_THRESHOLD:
        raise BranchStateIntegrityError("branch-state threshold has drifted from 1.90")
    if branch_state.get("comparator") != OVERSPEED_COMPARATOR:
        raise BranchStateIntegrityError("branch-state comparator has drifted from strict >")

    monitor_decision = _require_mapping(
        branch_state.get("monitor_decision"), "monitor_decision"
    )
    if (
        monitor_decision.get("monitor_id") != MONITOR_ID
        or monitor_decision.get("decision") != "veto"
        or monitor_decision.get("veto_applied") is not True
    ):
        raise BranchStateIntegrityError("branch state is not the frozen veto boundary")

    ordering = _require_mapping(branch_state.get("branch_ordering"), "branch_ordering")
    if (
        ordering.get("before_nominal_action_execution") is not True
        or ordering.get("before_final_veto_fallback_execution") is not True
        or ordering.get("nominal_action_executed") is not False
        or ordering.get("final_veto_fallback_executed") is not False
    ):
        raise BranchStateIntegrityError(
            "branch state is not before nominal and fallback action execution"
        )

    case_configuration = _require_mapping(
        branch_state.get("case_configuration"), "case_configuration"
    )
    if branch_state.get("case_configuration_hash") != _canonical_hash(
        case_configuration
    ):
        raise BranchStateIntegrityError("case configuration hash mismatch")
    simulator_configuration = _require_mapping(
        branch_state.get("simulator_configuration"), "simulator_configuration"
    )
    if branch_state.get("simulator_configuration_hash") != _canonical_hash(
        simulator_configuration
    ):
        raise BranchStateIntegrityError("simulator configuration hash mismatch")
    simulator_constants = _require_mapping(
        simulator_configuration.get("simulator_constants"),
        "simulator_configuration.simulator_constants",
    )
    if branch_state.get("simulator_constants_hash") != _canonical_hash(
        simulator_constants
    ):
        raise BranchStateIntegrityError("simulator constants hash mismatch")
    _state_from_document(branch_state)


def _state_from_document(branch_state: Mapping[str, object]) -> CartesianState2D:
    state = _require_mapping(branch_state.get("state"), "state")
    result = CartesianState2D(
        x=_require_finite(state.get("position_x"), "state.position_x"),
        y=_require_finite(state.get("position_y"), "state.position_y"),
        vx=_require_finite(state.get("velocity_x"), "state.velocity_x"),
        vy=_require_finite(state.get("velocity_y"), "state.velocity_y"),
    )
    _require_finite_state(result)
    return result


def _state_hash(state: CartesianState2D) -> str:
    return _canonical_hash(
        {
            "position_x": state.x,
            "position_y": state.y,
            "velocity_x": state.vx,
            "velocity_y": state.vy,
        }
    )


def _dynamics_from_document(
    branch_state: Mapping[str, object],
) -> tuple[Phase3435DynamicsContext, float, float]:
    simulator_configuration = _require_mapping(
        branch_state.get("simulator_configuration"), "simulator_configuration"
    )
    constants = _require_mapping(
        simulator_configuration.get("simulator_constants"),
        "simulator_configuration.simulator_constants",
    )
    context = Phase3435DynamicsContext(
        mu=_require_finite(constants.get("mu"), "simulator_constants.mu"),
        dt=_require_finite(constants.get("dt"), "simulator_constants.dt"),
        mass=_require_finite(constants.get("mass"), "simulator_constants.mass"),
        thrust_scale=_require_finite(
            simulator_configuration.get("thrust_scale"),
            "simulator_configuration.thrust_scale",
        ),
    )
    target_circular_speed = _require_finite(
        constants.get("target_circular_speed"),
        "simulator_constants.target_circular_speed",
    )
    speed_ratio_epsilon = _require_finite(
        constants.get("speed_ratio_denominator_epsilon"),
        "simulator_constants.speed_ratio_denominator_epsilon",
    )
    if context.mass <= 0.0 or context.dt <= 0.0 or target_circular_speed <= 0.0:
        raise RecoveryBranchExecutorError("dynamics context contains nonpositive values")
    if speed_ratio_epsilon < 0.0:
        raise RecoveryBranchExecutorError("speed-ratio epsilon must not be negative")
    return context, target_circular_speed, speed_ratio_epsilon


def generate_zero_action() -> Action2D:
    return (0.0, 0.0)


def generate_velocity_opposed_action(state: CartesianState2D) -> Action2D:
    _require_finite_state(state)
    velocity_norm = math.hypot(state.vx, state.vy)
    if not math.isfinite(velocity_norm):
        raise RecoveryBranchExecutorError("velocity norm must be finite")
    if velocity_norm <= VECTOR_ZERO_TOLERANCE:
        return generate_zero_action()
    action = (
        -ACTION_MAGNITUDE * state.vx / velocity_norm,
        -ACTION_MAGNITUDE * state.vy / velocity_norm,
    )
    _require_finite(action[0], "velocity_opposed_action[0]")
    _require_finite(action[1], "velocity_opposed_action[1]")
    return action


def generate_tangential_correction_action(
    state: CartesianState2D,
    target_circular_speed: float,
) -> Action2D:
    _require_finite_state(state)
    checked_target_speed = _require_finite(
        target_circular_speed, "target_circular_speed"
    )
    if checked_target_speed <= 0.0:
        raise RecoveryBranchExecutorError("target_circular_speed must be positive")
    position_norm = math.hypot(state.x, state.y)
    if not math.isfinite(position_norm) or position_norm == 0.0:
        raise RecoveryBranchExecutorError("position norm must be finite and nonzero")
    radial_x = state.x / position_norm
    radial_y = state.y / position_norm
    tangential_x = -radial_y
    tangential_y = radial_x
    tangential_speed = state.vx * tangential_x + state.vy * tangential_y
    tangential_error = tangential_speed - checked_target_speed
    _require_finite(tangential_x, "tangential_unit_vector[0]")
    _require_finite(tangential_y, "tangential_unit_vector[1]")
    _require_finite(tangential_error, "tangential_error")
    if abs(tangential_error) <= VECTOR_ZERO_TOLERANCE:
        return generate_zero_action()
    error_sign = 1.0 if tangential_error > 0.0 else -1.0
    action = (
        -ACTION_MAGNITUDE * error_sign * tangential_x,
        -ACTION_MAGNITUDE * error_sign * tangential_y,
    )
    _require_finite(action[0], "tangential_correction_action[0]")
    _require_finite(action[1], "tangential_correction_action[1]")
    return action


def generate_explicit_abort() -> None:
    return None


def _generate_action(
    branch_id: str,
    state: CartesianState2D,
    target_circular_speed: float,
) -> Action2D | None:
    if branch_id == "zero_action_reference_v0":
        return generate_zero_action()
    if branch_id == "velocity_opposed_thrust_v0":
        return generate_velocity_opposed_action(state)
    if branch_id == "tangential_error_correction_v0":
        return generate_tangential_correction_action(state, target_circular_speed)
    if branch_id == "explicit_abort_v0":
        return generate_explicit_abort()
    raise RecoveryBranchExecutorError(f"unsupported recovery branch: {branch_id!r}")


def execute_recovery_branch(
    branch_state: Mapping[str, object],
    branch_id: str,
    horizon_steps: int = 1,
) -> RecoveryBranchExecutionResult:
    if branch_id not in SUPPORTED_BRANCH_IDS:
        raise RecoveryBranchExecutorError(f"unsupported recovery branch: {branch_id!r}")
    if (
        isinstance(horizon_steps, bool)
        or not isinstance(horizon_steps, int)
        or horizon_steps != 1
    ):
        raise RecoveryBranchExecutorError(
            "Recovery Branch Executor v0 permits exactly one transition"
        )
    validate_branch_state_integrity(branch_state)
    previous_state = _state_from_document(branch_state)
    previous_state_hash = _state_hash(previous_state)
    dynamics, target_circular_speed, speed_ratio_epsilon = _dynamics_from_document(
        branch_state
    )
    action = _generate_action(branch_id, previous_state, target_circular_speed)

    if action is None:
        return RecoveryBranchExecutionResult(
            branch_id=branch_id,
            executed=False,
            action=None,
            previous_state_hash=previous_state_hash,
            next_state_hash=None,
            terminal_reason="explicit_recovery_abort",
            transition_count=0,
            valid=True,
            previous_state=previous_state,
            next_state=None,
            monitor_decision=None,
        )

    predicted_transition: Phase3435TransitionResult | None = None

    def predictor(state: CartesianState2D, proposed_action: Action2D):
        nonlocal predicted_transition
        transition = step_phase34_35_transition(
            state,
            NormalizedAction2D(proposed_action[0], proposed_action[1]),
            dynamics,
        )
        speed = math.hypot(transition.next_state.vx, transition.next_state.vy)
        speed_ratio = speed / (target_circular_speed + speed_ratio_epsilon)
        if state == previous_state and proposed_action == action:
            predicted_transition = transition
        return OneStepPrediction(
            next_state=transition.next_state,
            speed_ratio=speed_ratio,
        )

    decision = evaluate_overspeed_veto(
        previous_state,
        action,
        predictor,
        threshold=OVERSPEED_THRESHOLD,
    )
    if decision.decision == "veto":
        return RecoveryBranchExecutionResult(
            branch_id=branch_id,
            executed=False,
            action=action,
            previous_state_hash=previous_state_hash,
            next_state_hash=None,
            terminal_reason="recovery_action_rejected",
            transition_count=0,
            valid=True,
            previous_state=previous_state,
            next_state=None,
            monitor_decision=decision,
        )
    if decision.decision != "allow" or decision.executed_action != action:
        raise RecoveryBranchExecutorError(
            "Final Veto returned an invalid recovery-action decision"
        )
    if predicted_transition is None:
        raise RecoveryBranchExecutorError("nominal recovery prediction was not captured")

    realized_transition = step_phase34_35_transition(
        previous_state,
        NormalizedAction2D(action[0], action[1]),
        dynamics,
    )
    if realized_transition != predicted_transition:
        raise RecoveryBranchExecutorError(
            "recovery prediction diverged from the executed transition"
        )
    expected_executed_action = NormalizedAction2D(action[0], action[1])
    if realized_transition.executed_action != expected_executed_action:
        raise RecoveryBranchExecutorError(
            "recovery action changed under existing component clipping"
        )
    next_state = realized_transition.next_state
    _require_finite_state(next_state)
    return RecoveryBranchExecutionResult(
        branch_id=branch_id,
        executed=True,
        action=action,
        previous_state_hash=previous_state_hash,
        next_state_hash=_state_hash(next_state),
        terminal_reason="one_step_horizon_complete",
        transition_count=1,
        valid=True,
        previous_state=previous_state,
        next_state=next_state,
        monitor_decision=decision,
    )


__all__ = [
    "ACTION_MAGNITUDE",
    "DEFAULT_BRANCH_STATE_PATH",
    "SUPPORTED_BRANCH_IDS",
    "BranchStateIntegrityError",
    "RecoveryBranchExecutionResult",
    "RecoveryBranchExecutorError",
    "execute_recovery_branch",
    "generate_explicit_abort",
    "generate_tangential_correction_action",
    "generate_velocity_opposed_action",
    "generate_zero_action",
    "load_frozen_branch_state",
    "validate_branch_state_integrity",
]
