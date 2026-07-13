from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from simulator.phase34_35_transition import (
    CartesianState2D,
    Phase3435TransitionResult,
)


Action2D = tuple[float, float]
TransitionPredictor = Callable[
    [CartesianState2D, Action2D],
    Phase3435TransitionResult,
]
SpeedRatioFunction = Callable[[CartesianState2D], float]


@dataclass(frozen=True, slots=True)
class RolloutCaseContext:
    case_id: str
    controller_id: str
    controller_family: str
    r0_over_target: float
    initial_velocity_angle_deg: float
    thrust_scale: float
    target_radius: float
    target_circular_speed: float
    post_cross_mode: str
    upstream_variant: str | None = None


@dataclass(frozen=True, slots=True)
class PreTransitionActionContext:
    step: int
    phase: str
    active_stage: str
    current_state: CartesianState2D
    nominal_action: Action2D
    predict_transition: TransitionPredictor
    compute_speed_ratio: SpeedRatioFunction
    case: RolloutCaseContext


@dataclass(frozen=True, slots=True)
class ActionInterceptionResult:
    nominal_action: Action2D
    executed_action: Action2D
    intervention_applied: bool
    decision_metadata: object | None = None


@dataclass(frozen=True, slots=True)
class PostTransitionObservation:
    step: int
    phase: str
    active_stage: str
    previous_state: CartesianState2D
    nominal_action: Action2D
    executed_action: Action2D
    realized_next_state: CartesianState2D
    realized_next_speed_ratio: float
    intervention_applied: bool
    decision_metadata: object | None
    case: RolloutCaseContext


PreTransitionActionHook = Callable[
    [PreTransitionActionContext],
    ActionInterceptionResult,
]
PostTransitionObservationHook = Callable[[PostTransitionObservation], None]


__all__ = [
    "Action2D",
    "ActionInterceptionResult",
    "PostTransitionObservation",
    "PostTransitionObservationHook",
    "PreTransitionActionContext",
    "PreTransitionActionHook",
    "RolloutCaseContext",
    "SpeedRatioFunction",
    "TransitionPredictor",
]
