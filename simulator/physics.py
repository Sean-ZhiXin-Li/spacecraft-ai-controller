from __future__ import annotations
import math
from typing import Tuple, Dict

def circular_speed(mu: float, r: float) -> float:
    """Return circular orbital speed at radius r."""
    if r <= 0:
        raise ValueError("Radius must be positive for circular orbit.")
    return math.sqrt(mu / r)

def make_circular_state(mu: float, r0: float, mass: float) -> Dict[str, float]:
    """
    Produce a canonical circular orbit state in 2D plane.
    Position aligned on +x axis; velocity along +y axis.
    """
    v_circ = circular_speed(mu, r0)
    # State dictionary used consistently across the env
    return {
        "x": r0, "y": 0.0,
        "vx": 0.0, "vy": v_circ,
        "mass": float(mass),
    }

def step_dynamics(state: Dict[str, float], thrust_xy: Tuple[float, float], dt: float, mu: float) -> Dict[str, float]:
    """
    Very simple point-mass, 2D gravitational central field, with external thrust.
    - thrust_xy: (Tx, Ty) in Newtons
    - Euler semi-implicit (symplectic) integration to preserve energy better than naive.
    """
    x, y = state["x"], state["y"]
    vx, vy = state["vx"], state["vy"]
    m = state["mass"]
    Tx, Ty = thrust_xy

    # Gravitational acceleration
    r2 = x*x + y*y
    r = math.sqrt(r2)
    if r < 1e-6:
        # Avoid singularity; clamp by returning same state
        return dict(state)

    a_g = -mu / (r2 * r)  # acceleration magnitude along -r_hat
    ax_g = a_g * x
    ay_g = a_g * y

    # Thrust acceleration
    ax_t = (Tx / m) if m > 0 else 0.0
    ay_t = (Ty / m) if m > 0 else 0.0

    # Velocity update
    vx_new = vx + (ax_g + ax_t) * dt
    vy_new = vy + (ay_g + ay_t) * dt

    # Position update (semi-implicit)
    x_new = x + vx_new * dt
    y_new = y + vy_new * dt

    out = dict(state)
    out.update({"x": x_new, "y": y_new, "vx": vx_new, "vy": vy_new})
    return out

def orbital_radius(state: Dict[str, float]) -> float:
    """Return instantaneous radius."""
    return math.hypot(state["x"], state["y"])
