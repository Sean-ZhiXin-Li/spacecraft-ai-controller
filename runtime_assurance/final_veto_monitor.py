from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Callable, Generic, TypeVar, cast


MONITOR_ID = "one_step_overspeed_veto_v0"
OVERSPEED_THRESHOLD = 1.90
OVERSPEED_COMPARATOR = ">"
PREDICTION_HORIZON_STEPS = 1
FALLBACK_ACTION = (0.0, 0.0)
FALLBACK_PROVEN_SAFE = False
VALID_DECISIONS = frozenset({"allow", "veto"})

Action2D = tuple[float, float]
StateT = TypeVar("StateT")


class MonitorEvaluationError(ValueError):
    """Raised when a monitor evaluation cannot produce a valid decision."""


@dataclass(frozen=True, slots=True)
class OneStepPrediction(Generic[StateT]):
    next_state: StateT
    speed_ratio: float


@dataclass(frozen=True, slots=True)
class FinalVetoDecision:
    monitor_id: str
    decision: str
    reason: str
    threshold: float
    comparator: str
    nominal_action: Action2D
    executed_action: Action2D
    fallback_action: Action2D
    predicted_nominal_speed_ratio: float
    predicted_fallback_speed_ratio: float | None
    fallback_predicted_to_exceed_threshold: bool | None
    veto_applied: bool


Predictor = Callable[[StateT, Action2D], OneStepPrediction[StateT]]


def _require_finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MonitorEvaluationError(f"{name} must be a finite real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise MonitorEvaluationError(f"{name} must be a finite real number")
    return converted


def _validate_action(action: object, name: str) -> Action2D:
    if not isinstance(action, tuple) or len(action) != 2:
        raise MonitorEvaluationError(f"{name} must be a two-component tuple")
    _require_finite_real(action[0], f"{name}[0]")
    _require_finite_real(action[1], f"{name}[1]")
    return cast(Action2D, action)


def _predict(
    predictor: Predictor[StateT],
    state: StateT,
    action: Action2D,
    label: str,
) -> OneStepPrediction[StateT]:
    try:
        prediction = predictor(state, action)
    except Exception as exc:
        raise MonitorEvaluationError(f"{label} prediction failed") from exc
    if not isinstance(prediction, OneStepPrediction):
        raise MonitorEvaluationError(
            f"{label} predictor must return OneStepPrediction"
        )
    if prediction.next_state is None:
        raise MonitorEvaluationError(f"{label} predictor returned no next state")
    speed_ratio = _require_finite_real(
        prediction.speed_ratio,
        f"{label} predicted speed ratio",
    )
    return OneStepPrediction(
        next_state=prediction.next_state,
        speed_ratio=speed_ratio,
    )


def evaluate_overspeed_veto(
    current_state: StateT,
    nominal_action: Action2D,
    predict_next_state: Predictor[StateT],
    *,
    threshold: float = OVERSPEED_THRESHOLD,
    fallback_action: Action2D = FALLBACK_ACTION,
) -> FinalVetoDecision:
    """Evaluate one nominal action using an injected one-step predictor."""
    if current_state is None:
        raise MonitorEvaluationError("current_state must not be None")
    if not callable(predict_next_state):
        raise MonitorEvaluationError("predict_next_state must be callable")

    checked_threshold = _require_finite_real(threshold, "threshold")
    checked_nominal_action = _validate_action(nominal_action, "nominal_action")
    checked_fallback_action = _validate_action(fallback_action, "fallback_action")
    nominal_prediction = _predict(
        predict_next_state,
        current_state,
        checked_nominal_action,
        "nominal",
    )

    if nominal_prediction.speed_ratio > checked_threshold:
        fallback_prediction = _predict(
            predict_next_state,
            current_state,
            checked_fallback_action,
            "fallback",
        )
        return FinalVetoDecision(
            monitor_id=MONITOR_ID,
            decision="veto",
            reason="predicted_nominal_overspeed",
            threshold=checked_threshold,
            comparator=OVERSPEED_COMPARATOR,
            nominal_action=checked_nominal_action,
            executed_action=checked_fallback_action,
            fallback_action=checked_fallback_action,
            predicted_nominal_speed_ratio=nominal_prediction.speed_ratio,
            predicted_fallback_speed_ratio=fallback_prediction.speed_ratio,
            fallback_predicted_to_exceed_threshold=(
                fallback_prediction.speed_ratio > checked_threshold
            ),
            veto_applied=True,
        )

    return FinalVetoDecision(
        monitor_id=MONITOR_ID,
        decision="allow",
        reason="predicted_nominal_within_threshold",
        threshold=checked_threshold,
        comparator=OVERSPEED_COMPARATOR,
        nominal_action=checked_nominal_action,
        executed_action=checked_nominal_action,
        fallback_action=checked_fallback_action,
        predicted_nominal_speed_ratio=nominal_prediction.speed_ratio,
        predicted_fallback_speed_ratio=None,
        fallback_predicted_to_exceed_threshold=None,
        veto_applied=False,
    )
