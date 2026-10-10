
import json
import math
from pathlib import Path

from scipy.integrate import solve_ivp

from simulator.phase34_35_transition import (
    CartesianState2D,
    NormalizedAction2D,
    Phase3435DynamicsContext,
)
from runtime_assurance.dop853_validation import (
    step_phase34_35_transition_dop853,
)

source = Path(
    "analysis/stage2a_post_veto_alternative_audit_v0/"
    "exact_state_comparisons.json"
)
case = json.loads(source.read_text(encoding="utf-8"))["comparisons"][0]

MU = 1.3275182699999999e20
R_TARGET = 7.5e12
V_TARGET = math.sqrt(MU / R_TARGET)
MASS = 722.0
THRUST = 8000.0
DT = 100.0
ACTION_LIMIT = 0.25
A_MAX = THRUST * ACTION_LIMIT / MASS
dv_radial = 0.0
dv_trim = 0.0

def commanded_delta_v(action):
    return (
        THRUST / MASS * DT
        * math.hypot(action.action_x, action.action_y)
    )

initial = case["state_values"]

context = Phase3435DynamicsContext(
    mu=MU, dt=DT, mass=MASS, thrust_scale=THRUST
)


def components(s):
    x, y, vx, vy = s
    r = math.hypot(x, y)
    vr = (x * vx + y * vy) / r
    vt = (x * vy - y * vx) / r
    return r, vr, vt


def rhs(t, s):
    x, y, vx, vy = s
    r = math.hypot(x, y)
    return [
        vx,
        vy,
        -MU * x / (r**3 + 1e-12),
        -MU * y / (r**3 + 1e-12),
    ]


def braking_event(t, s):
    r, vr, _ = components(s)
    stopping_distance = max(vr, 0.0)**2 / (2 * A_MAX)
    return R_TARGET - r - 1.2 * stopping_distance


braking_event.terminal = True
braking_event.direction = -1

# Offline, zero-thrust coast to the braking boundary.
coast = solve_ivp(
    rhs,
    (0, 365 * 86400),
    initial,
    method="DOP853",
    events=braking_event,
    rtol=1e-12,
    atol=[1e-3, 1e-3, 1e-9, 1e-9],
)

if not coast.success or len(coast.t_events[0]) == 0:
    raise RuntimeError("Braking boundary was not reached.")

s = list(coast.y[:, -1])
r, vr, vt = components(s)

print("=== Coast Complete ===")
print("coast_days:", coast.t[-1] / 86400)
print("remaining_distance_m:", R_TARGET - r)
print("radial_velocity_mps:", vr)
print("braking_distance_m:", vr**2 / (2 * A_MAX))

print("\n=== Radial Braking ===")

for step in range(1, 65):
    r, vr, vt = components(s)
    erx, ery = s[0] / r, s[1] / r

    magnitude = min(
        ACTION_LIMIT,
        abs(vr) * MASS / (THRUST * DT),
    )
    direction = -1.0 if vr > 0 else 1.0

    action = NormalizedAction2D(
        direction * magnitude * erx,
        direction * magnitude * ery,
    )

    result = step_phase34_35_transition_dop853(
        CartesianState2D(*s), action, context
    )
    next_state = result.next_state
    s_next = [
        next_state.x, next_state.y,
        next_state.vx, next_state.vy,
    ]

    speed_ratio = math.hypot(s_next[2], s_next[3]) / V_TARGET

    if speed_ratio > 1.90:
        print("Overspeed proposal rejected at step", step)
        break

    dv_radial += commanded_delta_v(action)
    s = s_next
    r, vr, vt = components(s)

    radius_error = abs(r - R_TARGET) / R_TARGET
    radial_error = abs(vr) / V_TARGET
    tangential_error = abs(vt - V_TARGET) / V_TARGET

    if step in (1, 8, 16, 20, 22, 32):
        print(
            f"step={step}",
            f"radius_error={radius_error:.9f}",
            f"radial_error={radial_error:.6f}",
            f"tangential_error={tangential_error:.6f}",
        )

    if (
        radius_error <= 0.0025
        and radial_error <= 0.02
        and tangential_error <= 0.25
    ):
        print("\nCOMPONENT THRESHOLDS REACHED")
        print("step:", step)
        print("remaining_distance_m:", R_TARGET - r)
        print("radial_velocity_mps:", vr)
        print("tangential_velocity_mps:", vt)
        break
else:
    print("Component thresholds not reached within 64 braking steps.")



# Phase C: experimental local circularization.
# This is a development controller, not a frozen result.

print("\n=== Orbital Circularization Trim ===")

r, vr, vt = components(s)

ready = (
    abs(r - R_TARGET) / R_TARGET <= 0.0025
    and abs(vr) / V_TARGET <= 0.02
    and abs(vt - V_TARGET) / V_TARGET <= 0.25
)

if not ready:
    print("Skipped: radial braking milestone not reached.")

else:
    for step in range(1, 9):
        r, vr, vt = components(s)

        erx, ery = s[0] / r, s[1] / r
        etx, ety = -ery, erx

        # Local circular velocity, not target-radius velocity.
        vc = math.sqrt(MU / r)

        delta_vr = -vr
        delta_vt = vc - vt

        dvx = delta_vr * erx + delta_vt * etx
        dvy = delta_vr * ery + delta_vt * ety

        error = math.hypot(dvx, dvy)

        if error < 1e-3:
            break

        magnitude = min(
            ACTION_LIMIT,
            error * MASS / (THRUST * DT),
        )

        action = NormalizedAction2D(
            magnitude * dvx / error,
            magnitude * dvy / error,
        )

        result = step_phase34_35_transition_dop853(
            CartesianState2D(*s), action, context
        )

        next_state = result.next_state

        candidate = [
            next_state.x, next_state.y,
            next_state.vx, next_state.vy,
        ]

        if math.hypot(candidate[2], candidate[3]) / V_TARGET > 1.90:
            print("Safety rejection at step", step)
            break

        dv_trim += commanded_delta_v(action)
        s = candidate
        r, vr, vt = components(s)

        # Two-body osculating eccentricity.
        eccentricity = math.hypot(
            r * vt * vt / MU - 1.0,
            r * vr * vt / MU,
        )

        print(
            f"step={step}",
            f"vr={vr:.4f} m/s",
            f"vt={vt:.4f} m/s",
            f"eccentricity={eccentricity:.8f}",
            f"radius_error={abs(r-R_TARGET)/R_TARGET:.9f}",
        )

        if (
            eccentricity < 0.001
            and abs(r - R_TARGET) / R_TARGET <= 0.0025
        ):
            print("NEAR-CIRCULAR STATE REACHED")
            break

print("\n=== Propulsion Budget ===")

dv_total = dv_radial + dv_trim

print(f"Radial braking Δv: {dv_radial:.3f} m/s")
print(f"Circularization Δv: {dv_trim:.3f} m/s")
print(f"Total commanded Δv: {dv_total:.3f} m/s")

# Illustrative rocket-equation estimates.
# These do not change the simulated spacecraft mass.
G0 = 9.80665

for isp in (300.0, 450.0):
    propellant_fraction = 1.0 - math.exp(
        -dv_total / (G0 * isp)
    )
    print(
        f"Isp {isp:.0f} s: "
        f"ideal propellant mass fraction "
        f"{propellant_fraction:.2%}"
    )
