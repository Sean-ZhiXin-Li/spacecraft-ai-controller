"""Development-only long-coast, finite-thrust target-orbit capture probe.

Reconstructs one Phase35 initial state, but DOES NOT run the frozen Phase35
controller or modify any repository evidence. The time horizon is exploratory.
Solar point-mass gravity + variable spacecraft mass, constant Isp / thrust.
"""
import math
from scipy.integrate import solve_ivp

MU = 1.3275182699999999e20
TARGET_R = 7.5e12
R0_RATIO = 0.98
ANGLE_DEG = 150.0
WET_MASS = 722.0
DRY_MASS = 300.0
THRUST = 2000.0  # N: 8000 thrust_scale * 0.25 action norm
ISP = 450.0
G0 = 9.80665
DT_CONTROL = 100.0
V_CIRC = math.sqrt(MU / TARGET_R)


def components(s):
    x, y, vx, vy = s[:4]
    r = math.hypot(x, y)
    erx, ery = x / r, y / r
    return r, vx * erx + vy * ery, -vx * ery + vy * erx


def orbit_eccentricity(s):
    r, vr, vt = components(s)
    return math.hypot(r * vt**2 / MU - 1.0, r * vr * vt / MU)


def dynamics(_t, s, thrust_x=0.0, thrust_y=0.0, thrust_n=0.0):
    x, y, vx, vy, mass = s
    radius = math.hypot(x, y)
    grav_scale = -MU / radius**3
    return (
        vx, vy,
        grav_scale * x + thrust_x / mass,
        grav_scale * y + thrust_y / mass,
        -thrust_n / (ISP * G0),
    )


def target_crossing(_t, state):
    return math.hypot(state[0], state[1]) - TARGET_R


target_crossing.terminal = True
target_crossing.direction = 1


def integrate(state, seconds, ux=0.0, uy=0.0, thrust_n=0.0, **kwargs):
    return solve_ivp(
        lambda t, s: dynamics(t, s, thrust_n * ux, thrust_n * uy, thrust_n),
        (0.0, seconds), state,
        method="DOP853", rtol=1e-11,
        atol=[1e-2, 1e-2, 1e-7, 1e-7, 1e-7],
        **kwargs,
    )


def main():
    r0 = R0_RATIO * TARGET_R
    v0 = math.sqrt(MU / r0)
    theta = math.radians(ANGLE_DEG)
    s0 = [0.0, r0, v0 * math.cos(theta), v0 * math.sin(theta), WET_MASS]
    dv_budget = ISP * G0 * math.log(WET_MASS / DRY_MASS)
    print("=== Development-only long coast + finite-thrust capture ===")
    print(f"Initial r/R={R0_RATIO}, angle={ANGLE_DEG} degrees")
    print(f"Ideal delta-v cap: {dv_budget:.3f} m/s")
    print("No upstream burn; coast horizon is NOT the frozen Phase35 horizon.")

    coast = integrate(s0, 1200.0 * 86400.0, events=target_crossing)
    if not coast.success or not len(coast.t_events[0]):
        raise RuntimeError("No outward crossing in 1200 days: " + coast.message)
    s = coast.y[:, -1].tolist()
    r, vr, vt = components(s)
    needed_impulsive_dv = math.hypot(vr, math.sqrt(MU / r) - vt)
    print("\n=== Target-radius crossing ===")
    print(f"Coast time: {coast.t[-1] / 86400.0:.6f} days")
    print(f"Radial speed: {vr:.6f} m/s")
    print(f"Tangential speed: {vt:.6f} m/s")
    print(f"Circular speed here: {math.sqrt(MU/r):.6f} m/s")
    print(f"Ideal instantaneous insertion requirement: {needed_impulsive_dv:.6f} m/s")
    print(f"Pre-insertion eccentricity: {orbit_eccentricity(s):.8f}")
    if needed_impulsive_dv > dv_budget:
        print("Budget insufficient even for ideal instantaneous insertion.")
        return

    # Receding-horizon direction updates, full thrust except the last partial
    # control interval. Propellant consumption is integrated continuously.
    print("\n=== Variable-mass finite-thrust insertion ===")
    elapsed = 0.0
    for step in range(1, 41):
        r, vr, vt = components(s)
        vc = math.sqrt(MU / r)
        erx, ery = s[0] / r, s[1] / r
        etx, ety = -ery, erx
        desired_vx, desired_vy = vc * etx, vc * ety
        ex, ey = desired_vx - s[2], desired_vy - s[3]
        needed_dv = math.hypot(ex, ey)
        if needed_dv < 0.001:
            print("Insertion velocity-vector tolerance reached.")
            break
        if s[4] <= DRY_MASS + 1e-8:
            print("Propellant exhausted.")
            break

        burn_mass_target = s[4] * math.exp(-needed_dv / (ISP * G0))
        mdot = THRUST / (ISP * G0)
        desired_burn_time = (s[4] - burn_mass_target) / mdot
        remaining_burn_time = (s[4] - DRY_MASS) / mdot
        burn_time = min(DT_CONTROL, desired_burn_time, remaining_burn_time)
        ux, uy = ex / needed_dv, ey / needed_dv
        result = integrate(s, burn_time, ux, uy, THRUST)
        if not result.success:
            raise RuntimeError(result.message)
        s = result.y[:, -1].tolist()
        elapsed += burn_time
        if s[4] < DRY_MASS - 1e-6:
            raise RuntimeError("Integration violated dry-mass floor")
        if step in (1, 3, 5, 6, 7) or burn_time < DT_CONTROL:
            r, vr, vt = components(s)
            print(
                f"step={step:2d} burn={burn_time:7.3f}s "
                f"mass={s[4]:8.3f}kg vr={vr:9.3f}m/s "
                f"vt={vt:9.3f}m/s e={orbit_eccentricity(s):.8f}"
            )
    else:
        print("Stopped at 40 guidance steps.")

    r, vr, vt = components(s)
    used_dv = ISP * G0 * math.log(WET_MASS / s[4])
    radius_error = abs(r - TARGET_R) / TARGET_R
    print("\n=== End of powered capture ===")
    print(f"Powered duration: {elapsed:.3f} s")
    print(f"Remaining mass: {s[4]:.6f} kg")
    print(f"Fuel used: {WET_MASS - s[4]:.6f} kg")
    print(f"Ideal delivered delta-v (mass ratio): {used_dv:.6f} m/s")
    print(f"Remaining delta-v budget: {max(0.0, dv_budget-used_dv):.6f} m/s")
    print(f"Radius error ratio: {radius_error:.10f}")
    print(f"Radial speed: {vr:.6f} m/s")
    print(f"Tangential speed error: {vt - math.sqrt(MU/r):.6f} m/s")
    print(f"Osculating eccentricity: {orbit_eccentricity(s):.10f}")
    near_circular = orbit_eccentricity(s) < 0.001 and radius_error < 0.0025
    print(f"Near-circular state: {near_circular}")
    if near_circular:
        hold = integrate(s, 100 * 86400.0, t_eval=[0, 25*86400, 50*86400, 75*86400, 100*86400])
        if not hold.success:
            raise RuntimeError(hold.message)
        max_radius_error = max(abs(math.hypot(x, y)-TARGET_R)/TARGET_R
                               for x, y in zip(hold.y[0], hold.y[1]))
        print("\n=== 100-day unpowered follow-through (5 samples) ===")
        print(f"Max sampled radius error ratio: {max_radius_error:.10f}")
        print(f"Final eccentricity: {orbit_eccentricity(hold.y[:, -1]):.10f}")
    print("Exploratory 2D solar two-body model, NOT frozen Phase35 success.")


if __name__ == "__main__":
    main()
