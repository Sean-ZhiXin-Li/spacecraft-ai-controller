from __future__ import annotations

import math

from scipy.integrate import solve_ivp

from simulator.phase34_35_transition import (
    ACTION_COMPONENT_MAX,
    ACTION_COMPONENT_MIN,
    GRAVITY_DENOMINATOR_EPSILON,
    CartesianState2D,
    NormalizedAction2D,
    Phase3435DynamicsContext,
    Phase3435TransitionResult,
)


def _clamp(value: float) -> float:
    return max(ACTION_COMPONENT_MIN, min(ACTION_COMPONENT_MAX, value))


def step_phase34_35_transition_dop853(
    state: CartesianState2D,
    proposed_action: NormalizedAction2D,
    context: Phase3435DynamicsContext,
) -> Phase3435TransitionResult:
    """Propagate one frozen control interval with SciPy DOP853."""

    action_x = _clamp(proposed_action.action_x)
    action_y = _clamp(proposed_action.action_y)

    executed_action = NormalizedAction2D(
        action_x=action_x,
        action_y=action_y,
    )

    def rhs(
        t: float,
        values,
    ):
        x, y, vx, vy = values

        radius = math.sqrt(x * x + y * y)
        denominator = radius**3 + GRAVITY_DENOMINATOR_EPSILON

        acceleration_x = (
            -context.mu * x / denominator
            + context.thrust_scale * action_x / context.mass
        )

        acceleration_y = (
            -context.mu * y / denominator
            + context.thrust_scale * action_y / context.mass
        )

        return [
            vx,
            vy,
            acceleration_x,
            acceleration_y,
        ]

    solution = solve_ivp(
        rhs,
        t_span=(0.0, context.dt),
        y0=[state.x, state.y, state.vx, state.vy],
        method="DOP853",
        rtol=1.0e-12,
        atol=[
            1.0e-3,
            1.0e-3,
            1.0e-9,
            1.0e-9,
        ],
    )

    if not solution.success:
        raise RuntimeError(
            f"DOP853 propagation failed: {solution.message}"
        )

    x, y, vx, vy = solution.y[:, -1]

    return Phase3435TransitionResult(
        next_state=CartesianState2D(
            x=float(x),
            y=float(y),
            vx=float(vx),
            vy=float(vy),
        ),
        executed_action=executed_action,
    )
