from __future__ import annotations

import math
from dataclasses import dataclass


ACTION_COMPONENT_MIN = -1.0
ACTION_COMPONENT_MAX = 1.0
GRAVITY_DENOMINATOR_EPSILON = 1.0e-12


@dataclass(frozen=True, slots=True)
class CartesianState2D:
    x: float
    y: float
    vx: float
    vy: float


@dataclass(frozen=True, slots=True)
class NormalizedAction2D:
    action_x: float
    action_y: float


@dataclass(frozen=True, slots=True)
class Phase3435DynamicsContext:
    mu: float
    dt: float
    mass: float
    thrust_scale: float


@dataclass(frozen=True, slots=True)
class Phase3435TransitionResult:
    next_state: CartesianState2D
    executed_action: NormalizedAction2D


def _clamp_action_component(value: float) -> float:
    return max(ACTION_COMPONENT_MIN, min(ACTION_COMPONENT_MAX, value))


def step_phase34_35_transition(
    state: CartesianState2D,
    proposed_action: NormalizedAction2D,
    context: Phase3435DynamicsContext,
) -> Phase3435TransitionResult:
    """Reproduce the scalar one-step transition used by Phase34 and Phase35."""
    action_x = _clamp_action_component(proposed_action.action_x)
    action_y = _clamp_action_component(proposed_action.action_y)
    executed_action = NormalizedAction2D(action_x=action_x, action_y=action_y)

    radius = math.sqrt(state.x * state.x + state.y * state.y)
    denominator = radius**3 + GRAVITY_DENOMINATOR_EPSILON
    acceleration_x = (
        -context.mu * state.x / denominator
        + context.thrust_scale * action_x / context.mass
    )
    acceleration_y = (
        -context.mu * state.y / denominator
        + context.thrust_scale * action_y / context.mass
    )
    next_vx = state.vx + acceleration_x * context.dt
    next_vy = state.vy + acceleration_y * context.dt
    next_x = state.x + next_vx * context.dt
    next_y = state.y + next_vy * context.dt

    return Phase3435TransitionResult(
        next_state=CartesianState2D(
            x=next_x,
            y=next_y,
            vx=next_vx,
            vy=next_vy,
        ),
        executed_action=executed_action,
    )
