"""Offline prototype: coast, then finite-propellant radial braking.

Uses a 5D state [x, y, vx, vy, mass]. No frozen data is modified.
This is NOT a complete fuel-aware insertion controller.
Run from repository root: python -m scripts.variable_mass_capture_dev
"""
import json
import math
from pathlib import Path

from scipy.integrate import solve_ivp

MU = 1.3275182699999999e20
TARGET_R = 7.5e12
TARGET_V = math.sqrt(MU / TARGET_R)
DT = 100.0
WET_MASS = 722.0
DRY_MASS = 300.0
ISP = 450.0
G0 = 9.80665
THRUST = 8000.0
ACTION_LIMIT = 0.25
FORCE = THRUST * ACTION_LIMIT
MASS_FLOW = FORCE / (ISP * G0)
A_WET = FORCE / WET_MASS

path = Path(
    "analysis/stage2a_post_veto_alternative_audit_v0/"
    "exact_state_comparisons.json"
)
case = json.loads(path.read_text(encoding="utf-8"))["comparisons"][0]
state = [*map(float, case["state_values"]), WET_MASS]


def components(s):
    x, y, vx, vy, _ = s
    r = math.hypot(x, y)
    vr = (x * vx + y * vy) / r
    vt = (x * vy - y * vx) / r
    return r, vr, vt


def propagate(s, duration, fx=0.0, fy=0.0):
    """Hold inertial force components constant during this integration."""
    if duration <= 0:
        return list(s)

    def rhs(t, z):
        x, y, vx, vy, mass = z
        r = math.hypot(x, y)
        gravity = MU / (r**3 + 1e-12)
        return [
            vx,
            vy,
            -gravity * x + fx / mass,
            -gravity * y + fy / mass,
            -math.hypot(fx, fy) / (ISP * G0),
        ]

    solution = solve_ivp(
        rhs, (0.0, duration), s, method="DOP853",
        rtol=1e-11, atol=[1e-3, 1e-3, 1e-9, 1e-9, 1e-9],
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return [float(v) for v in solution.y[:, -1]]


def braking_event(t, s):
    r, vr, _ = components(s)
    return TARGET_R - r - 1.2 * max(vr, 0.0)**2 / (2 * A_WET)


braking_event.terminal = True
braking_event.direction = -1


def gravity_only_rhs(t, z):
    x, y, vx, vy, mass = z
    r = math.hypot(x, y)
    g = MU / (r**3 + 1e-12)
    return [vx, vy, -g * x, -g * y, 0.0]


coast = solve_ivp(
    gravity_only_rhs,
    (0, 365 * 86400),
    state, method="DOP853", events=braking_event,
    rtol=1e-12, atol=[1e-3, 1e-3, 1e-9, 1e-9, 1e-9],
)
if not coast.success or not len(coast.t_events[0]):
    raise RuntimeError("No coast-to-brake boundary found")
state = [float(v) for v in coast.y[:, -1]]

print(f"Coast: {coast.t[-1] / 86400:.5f} days")
print(f"Initial mass: {state[4]:.3f} kg")
print(f"Full-thrust mass flow: {MASS_FLOW:.6f} kg/s")
print(f"Available full-thrust burn: {(WET_MASS-DRY_MASS)/MASS_FLOW:.3f} s")
print("\n=== Variable-mass braking ===")

cumulative_dv = 0.0
for step in range(1, 65):
    r, vr, vt = components(state)
    if state[4] <= DRY_MASS + 1e-8:
        print(f"Fuel depleted before step {step}")
        break

    # Recompute desired inward radial force every 100 s.
    direction = -1.0 if vr > 0 else 1.0
    fx = direction * FORCE * state[0] / r
    fy = direction * FORCE * state[1] / r

    previous_mass = state[4]
    remaining_burn = max(0.0, (previous_mass - DRY_MASS) / MASS_FLOW)
    burn_time = min(DT, remaining_burn)

    state = propagate(state, burn_time, fx, fy)
    state[4] = max(DRY_MASS, state[4])
    actual_dv = ISP * G0 * math.log(previous_mass / state[4])
    cumulative_dv += actual_dv

    # If fuel runs out mid-interval, coast through the remaining time.
    if burn_time < DT:
        state = propagate(state, DT - burn_time)

    r, vr, vt = components(state)
    eccentricity = math.hypot(r * vt * vt / MU - 1.0, r * vr * vt / MU)
    if step in (1, 5, 8, 9, 10) or burn_time < DT:
        print(
            f"step={step:2d} burn={burn_time:7.3f}s "
            f"mass={state[4]:8.3f}kg radial_v={vr:9.3f}m/s "
            f"dv={cumulative_dv:8.3f}m/s e={eccentricity:.5f}"
        )

    if math.hypot(state[2], state[3]) / TARGET_V > 1.90:
        raise RuntimeError("Exploratory trajectory crossed overspeed threshold")

    if state[4] <= DRY_MASS + 1e-8:
        print("Propellant exhausted; further powered braking unavailable")
        break

print("\nFinal radius error ratio:", abs(r - TARGET_R) / TARGET_R)
print("Final radial velocity ratio:", abs(vr) / TARGET_V)
print("Final tangential error ratio:", abs(vt - TARGET_V) / TARGET_V)
print("Final mass (kg):", state[4])
print("Ideal propulsive delta-v (m/s):", cumulative_dv)
print("Near-target near-circular state achieved:", (
    eccentricity < 0.001
    and abs(r - TARGET_R) / TARGET_R <= 0.0025
    and abs(vr) / TARGET_V <= 0.02
    and abs(vt - TARGET_V) / TARGET_V <= 0.25
))
