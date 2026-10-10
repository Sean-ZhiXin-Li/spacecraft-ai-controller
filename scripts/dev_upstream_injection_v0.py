"""Standalone development probe: early impulse and target-radius crossing.

The true Phase35 initial-state formula is reproduced here, but Phase35's
controller is NOT executed. Impulses are idealized instantaneous burns,
not variable-mass finite-duration thrust. This is not a Phase35 regression.
"""
import math

from scipy.integrate import solve_ivp

MU = 1.3275182699999999e20
TARGET_R = 7.5e12
R0_OVER_TARGET = 0.98
INITIAL_VELOCITY_ANGLE_DEG = 150.0
DT = 100.0
MAX_STEPS = 100_000
WET_MASS = 722.0
DRY_MASS = 300.0
ISP = 450.0
G0 = 9.80665
DV_CAP = ISP * G0 * math.log(WET_MASS / DRY_MASS)
VC = math.sqrt(MU / TARGET_R)


def components(state):
    x, y, vx, vy = state
    radius = math.hypot(x, y)
    radial_velocity = (x * vx + y * vy) / radius
    tangential_velocity = (x * vy - y * vx) / radius
    return radius, radial_velocity, tangential_velocity


def rhs(_t, state):
    x, y, vx, vy = state
    radius = math.hypot(x, y)
    return [vx, vy, -MU * x / radius**3, -MU * y / radius**3]


def target_crossing(_t, state):
    return math.hypot(state[0], state[1]) - TARGET_R


target_crossing.terminal = True
target_crossing.direction = 1


def propagate(initial_state, duration_s):
    result = solve_ivp(
        rhs, (0, duration_s), initial_state,
        method="DOP853", rtol=1e-11,
        atol=[1e-2, 1e-2, 1e-7, 1e-7],
        events=target_crossing,
    )
    if not result.success:
        raise RuntimeError(result.message)
    state = result.y[:, -1]
    radius, vr, vt = components(state)
    reached = len(result.t_events[0]) > 0
    ecc = math.hypot(radius * vt * vt / MU - 1.0,
                     radius * vr * vt / MU)
    return reached, result.t[-1] / 86400.0, radius, vr, vt, ecc


def main():
    radius0 = R0_OVER_TARGET * TARGET_R
    speed0 = math.sqrt(MU / radius0)
    angle = math.radians(INITIAL_VELOCITY_ANGLE_DEG)
    initial = (0.0, radius0,
               speed0 * math.cos(angle), speed0 * math.sin(angle))

    # Impulses expressed in the local (radial outward, prograde) frame.
    candidates = [
        ("coast_only", 0.0, 0.0),
        ("radial_2500", 2500.0, 0.0),
        ("radial_3500", 3500.0, 0.0),
        ("mixed_3000_neg1500", 3000.0, -1500.0),
    ]
    horizons = [("frozen_duration", MAX_STEPS * DT),
                ("exploratory_365d", 365 * 86400.0)]

    print(f"True Phase35 initial state: r0/R={R0_OVER_TARGET}, "
          f"angle={INITIAL_VELOCITY_ANGLE_DEG} deg")
    print(f"Ideal instantaneous delta-v cap = {DV_CAP:.2f} m/s")
    print("This is a new impulse planner probe, NOT a rerun of Phase35.")
    for name, dvr, dvt in candidates:
        cost = math.hypot(dvr, dvt)
        if cost > DV_CAP + 1e-9:
            print(f"\n{name}: invalid, exceeds budget")
            continue
        # At x=0 and y>0: e_r=(0, 1), e_t=(-1, 0).
        proposed = (initial[0], initial[1],
                    initial[2] - dvt, initial[3] + dvr)
        remaining_mass = WET_MASS * math.exp(-cost / (ISP * G0))
        initial_speed_ratio = math.hypot(proposed[2], proposed[3]) / VC
        print(f"\n{name} | ideal Δv={cost:.1f} m/s | "
              f"mass_after_impulse={remaining_mass:.1f} kg | "
              f"speed_ratio_after={initial_speed_ratio:.4f}")
        for label, duration in horizons:
            reached, days, radius, vr, vt, eccentricity = propagate(
                proposed, duration
            )
            print(f"  {label:17} crossing={'YES' if reached else 'NO ':3} "
                  f"t={days:8.2f} d "
                  f"gap={(TARGET_R-radius)/1e9:8.2f} Gm "
                  f"vr={vr:8.1f} m/s "
                  f"vt_err={abs(vt-VC)/VC:.4f} "
                  f"e={eccentricity:.4f}")


if __name__ == "__main__":
    main()
