from __future__ import annotations

import math
from dataclasses import dataclass

from simulator.phase34_35_transition import CartesianState2D


RECOVERY_SUCCESS = "recovery_success"
OVERSPEED = "overspeed"
INSTABILITY = "instability"
UNSAFE_STATE = "unsafe_state"
INVALID_SIMULATION = "invalid_simulation"
INVALID_RECOVERY_EVALUATION = "invalid_recovery_evaluation"
ACTION_REJECTED = "action_rejected"
EXPLICIT_ABORT = "explicit_abort"
RECOVERY_HORIZON_EXHAUSTED = "recovery_horizon_exhausted"
TOTAL_HORIZON_EXHAUSTED = "total_horizon_exhausted"
NOT_EVALUATED = "not_evaluated"
CLEAR = "clear"
TRIGGERED = "triggered"

FROZEN_STOP_LABELS = (
    RECOVERY_SUCCESS,
    OVERSPEED,
    INSTABILITY,
    UNSAFE_STATE,
    INVALID_SIMULATION,
    INVALID_RECOVERY_EVALUATION,
    ACTION_REJECTED,
    EXPLICIT_ABORT,
    RECOVERY_HORIZON_EXHAUSTED,
    TOTAL_HORIZON_EXHAUSTED,
)

_TERMINATION_PRIORITY = (
    INVALID_SIMULATION,
    INVALID_RECOVERY_EVALUATION,
    OVERSPEED,
    INSTABILITY,
    UNSAFE_STATE,
    ACTION_REJECTED,
    EXPLICIT_ABORT,
    RECOVERY_SUCCESS,
    RECOVERY_HORIZON_EXHAUSTED,
    TOTAL_HORIZON_EXHAUSTED,
)


class RecoveryStopConditionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryStopConditionReport:
    terminal_reason: str | None
    statuses: tuple[tuple[str, str], ...]

    def status_for(self, label: str) -> str:
        for current_label, status in self.statuses:
            if current_label == label:
                return status
        raise KeyError(label)


def _state_is_finite(state: CartesianState2D | None) -> bool | None:
    if state is None:
        return None
    return all(math.isfinite(value) for value in (state.x, state.y, state.vx, state.vy))


def evaluate_recovery_stop_conditions(
    *,
    execution_terminal_reason: str,
    next_state: CartesianState2D | None,
    realized_speed_ratio: float | None,
    overspeed_threshold: float,
    recovery_transition_count: int,
    recovery_horizon_steps: int,
    total_transition_count: int,
    total_horizon_steps: int,
) -> RecoveryStopConditionReport:
    for value, name in (
        (recovery_transition_count, "recovery_transition_count"),
        (recovery_horizon_steps, "recovery_horizon_steps"),
        (total_transition_count, "total_transition_count"),
        (total_horizon_steps, "total_horizon_steps"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RecoveryStopConditionError(f"{name} must be a nonnegative integer")
    if recovery_horizon_steps < 1 or total_horizon_steps < 1:
        raise RecoveryStopConditionError("horizon limits must be positive")
    if not math.isfinite(overspeed_threshold):
        raise RecoveryStopConditionError("overspeed_threshold must be finite")

    statuses = {label: NOT_EVALUATED for label in FROZEN_STOP_LABELS}
    finite_state = _state_is_finite(next_state)
    if finite_state is False:
        statuses[INVALID_SIMULATION] = TRIGGERED
    elif finite_state is True:
        statuses[INVALID_SIMULATION] = CLEAR

    if realized_speed_ratio is not None:
        if not math.isfinite(realized_speed_ratio):
            statuses[INVALID_SIMULATION] = TRIGGERED
            statuses[OVERSPEED] = NOT_EVALUATED
        else:
            statuses[OVERSPEED] = (
                TRIGGERED if realized_speed_ratio > overspeed_threshold else CLEAR
            )

    if execution_terminal_reason == "recovery_action_rejected":
        statuses[ACTION_REJECTED] = TRIGGERED
    elif execution_terminal_reason == "explicit_recovery_abort":
        statuses[EXPLICIT_ABORT] = TRIGGERED
    elif execution_terminal_reason == "invalid_recovery_evaluation":
        statuses[INVALID_RECOVERY_EVALUATION] = TRIGGERED
    else:
        statuses[ACTION_REJECTED] = CLEAR
        statuses[EXPLICIT_ABORT] = CLEAR
        statuses[INVALID_RECOVERY_EVALUATION] = CLEAR

    statuses[RECOVERY_HORIZON_EXHAUSTED] = (
        TRIGGERED
        if recovery_transition_count >= recovery_horizon_steps
        else CLEAR
    )
    statuses[TOTAL_HORIZON_EXHAUSTED] = (
        TRIGGERED if total_transition_count >= total_horizon_steps else CLEAR
    )

    terminal_reason = next(
        (
            label
            for label in _TERMINATION_PRIORITY
            if statuses[label] == TRIGGERED
        ),
        None,
    )
    return RecoveryStopConditionReport(
        terminal_reason=terminal_reason,
        statuses=tuple((label, statuses[label]) for label in FROZEN_STOP_LABELS),
    )


__all__ = [
    "ACTION_REJECTED",
    "CLEAR",
    "EXPLICIT_ABORT",
    "FROZEN_STOP_LABELS",
    "INSTABILITY",
    "INVALID_RECOVERY_EVALUATION",
    "INVALID_SIMULATION",
    "NOT_EVALUATED",
    "OVERSPEED",
    "RECOVERY_HORIZON_EXHAUSTED",
    "RECOVERY_SUCCESS",
    "TOTAL_HORIZON_EXHAUSTED",
    "TRIGGERED",
    "UNSAFE_STATE",
    "RecoveryStopConditionError",
    "RecoveryStopConditionReport",
    "evaluate_recovery_stop_conditions",
]
