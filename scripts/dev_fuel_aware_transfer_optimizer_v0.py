"""Development-only fuel-aware two-burn transfer planner.

From a Phase35-like initial condition (NOT its frozen rollout), optimize one
initial radial/tangential velocity change for the earliest outward target-radius
crossing, assuming an ideal instantaneous departure and capture for SEARCH.
Then REPLAY both burns using finite 2000 N thrust, variable mass, continuous
solar two-body dynamics, and 100-second receding guidance updates.

The objective is restricted to a single departure burn + coast + capture family.
It is NOT global optimum, original Phase35 success, or flight validation.
"""
import math
from dataclasses import dataclass

from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize

MU = 1.3275182699999999e20
TARGET_R = 7.5e12
R0_RATIO = 0.98
ANGLE_DEG = 150.0
WET_MASS = 722.0
DRY_MASS = 300.0
THRUST_N = 2000.0
ISP = 450.0
G0 = 9.80665
GUIDANCE_DT = 100.0
RESERVE_PROPELLANT_KG = 20.0
SEARCH_MARGIN_KG = 0.5  # avoid choosing the exact numerical feasibility boundary
MAX_COAST_DAYS = 1200.0
VC = math.sqrt(MU / TARGET_R)
R0 = R0_RATIO * TARGET_R
V0 = math.sqrt(MU / R0)
THETA = math.radians(ANGLE_DEG)
S0 = [0.0, R0, V0 * math.cos(THETA), V0 * math.sin(THETA), WET_MASS]
VR0, VT0 = V0 * math.sin(THETA), -V0 * math.cos(THETA)
VE = ISP * G0
MDOT = THRUST_N / VE
DV_BUDGET = VE * math.log(WET_MASS / DRY_MASS)
DV_PLANNING = VE * math.log(WET_MASS / (DRY_MASS + RESERVE_PROPELLANT_KG + SEARCH_MARGIN_KG))


@dataclass
class Prediction:
    days: float
    departure_dv: float
    insertion_dv: float
    total_dv: float
    crossing_vr: float
    crossing_vt: float


def impulse_prediction(dvr, dvt):
    """Energy/angular momentum + radial quadrature, outward crossing only."""
    vr0, vt0 = VR0 + dvr, VT0 + dvt
    if vr0 <= 1.0:
        return None
    h = R0 * vt0
    energy = 0.5 * (vr0 * vr0 + vt0 * vt0) - MU / R0

    def vr_sq_at_fraction(u):
        r = TARGET_R * u
        return 2 * (energy + MU / r) - (h / r) ** 2

    if vr_sq_at_fraction(1.0) <= 0.0:
        return None
    try:
        days = quad(
            lambda u: TARGET_R / math.sqrt(vr_sq_at_fraction(u)),
            R0_RATIO, 1.0, epsrel=1e-10,
        )[0] / 86400.0
    except (ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(days) or days > MAX_COAST_DAYS:
        return None
    vr1 = math.sqrt(vr_sq_at_fraction(1.0))
    vt1 = h / TARGET_R
    dv0 = math.hypot(dvr, dvt)
    dv1 = math.hypot(vr1, VC - vt1)
    return Prediction(days, dv0, dv1, dv0 + dv1, vr1, vt1)


def optimize_injection():
    """Bounded multi-start SLSQP; no global optimality claim."""
    def predicted(x):
        return impulse_prediction(x[0], x[1])

    def objective(x):
        p = predicted(x)
        return p.days if p is not None else 1e6

    def available(x):
        p = predicted(x)
        return DV_PLANNING - p.total_dv if p is not None else -1e6

    def overspeed_headroom(x):
        speed = math.hypot(VR0 + x[0], VT0 + x[1])
        return 1.90 * VC - speed

    seeds = [(0, 0), (350, 100), (550, 250), (700, 150), (750, 300)]
    feasible = []
    for seed in seeds:
        result = minimize(
            objective,
            seed,
            method="SLSQP",
            bounds=[(-1500, 2000), (-1600, 2000)],
            constraints=[
                {"type": "ineq", "fun": available},
                {"type": "ineq", "fun": overspeed_headroom},
                {"type": "ineq", "fun": lambda x: VR0 + x[0] - 1.0},
            ],
            options={"maxiter": 200, "ftol": 1e-8},
        )
        p = predicted(result.x)
        if p and p.total_dv <= DV_PLANNING + 1e-5 and overspeed_headroom(result.x) >= -1e-5:
            feasible.append((p.days, result.x, p))
    if not feasible:
        raise RuntimeError("No feasible candidate in this restricted two-burn search")
    return min(feasible, key=lambda item: item[0])


def components(s):
    x, y, vx, vy = s[:4]
    r = math.hypot(x, y)
    erx, ery = x / r, y / r
    return r, vx * erx + vy * ery, -vx * ery + vy * erx


def eccentricity(s):
    r, vr, vt = components(s)
    return math.hypot(r * vt * vt / MU - 1.0, r * vr * vt / MU)


def dynamics(_t, s, ux, uy, thrust):
    x, y, vx, vy, m = s
    r = math.hypot(x, y)
    return [vx, vy, -MU * x / r**3 + thrust * ux / m,
            -MU * y / r**3 + thrust * uy / m, -thrust / VE]


def propagate(s, seconds, ux=0.0, uy=0.0, thrust=0.0, **kwargs):
    result = solve_ivp(
        lambda t, z: dynamics(t, z, ux, uy, thrust),
        (0, seconds), s, method="DOP853", rtol=1e-11,
        atol=[1e-2, 1e-2, 1e-7, 1e-7, 1e-7], **kwargs,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result


def outward_crossing(_t, s):
    return math.hypot(s[0], s[1]) - TARGET_R


outward_crossing.terminal = True
outward_crossing.direction = 1


def finite_burn(s, phase, dvr=0.0, dvt=0.0):
    """Fixed inertial departure velocity target; local circular capture target."""
    t_burn = 0.0
    if phase == "departure":
        r = math.hypot(s[0], s[1])
        erx, ery = s[0] / r, s[1] / r
        etx, ety = -ery, erx
        goal = (s[2] + dvr * erx + dvt * etx,
                s[3] + dvr * ery + dvt * ety)

    for step in range(1, 41):
        r = math.hypot(s[0], s[1])
        erx, ery = s[0] / r, s[1] / r
        etx, ety = -ery, erx
        if phase == "departure":
            dx, dy = goal[0] - s[2], goal[1] - s[3]
        else:
            vc = math.sqrt(MU / r)
            dx, dy = vc * etx - s[2], vc * ety - s[3]
        dv = math.hypot(dx, dy)
        if dv < 0.001:
            return s, t_burn, True
        if s[4] <= DRY_MASS + 1e-7:
            return s, t_burn, False
        # Ideal rocket-equation duration for requested correction.
        target_m = s[4] * math.exp(-dv / VE)
        dt = min(GUIDANCE_DT, (s[4] - target_m) / MDOT,
                 (s[4] - DRY_MASS) / MDOT)
        if dt <= 1e-9:
            return s, t_burn, False
        out = propagate(s, dt, dx / dv, dy / dv, THRUST_N)
        s = out.y[:, -1].tolist()
        t_burn += dt
        if s[4] < DRY_MASS - 1e-5:
            raise RuntimeError("Propellant integration exceeded the dry-mass floor")
        if math.hypot(s[2], s[3]) > 1.90 * VC + 1e-7:
            raise RuntimeError("Exploratory post-burn speed bound exceeded")
    return s, t_burn, False


def main():
    print("=== Fuel-aware upstream transfer optimizer v0 (development only) ===")
    print(f"Initial state: r/R={R0_RATIO}; angle={ANGLE_DEG} deg")
    print(f"Full ideal delta-v budget: {DV_BUDGET:.3f} m/s")
    print(f"Planner budget with {RESERVE_PROPELLANT_KG:.1f} kg propellant reserve: {DV_PLANNING:.3f} m/s")

    base = impulse_prediction(0.0, 0.0)
    _, controls, prediction = optimize_injection()
    dvr, dvt = controls
    print("\n=== Impulsive optimization (restricted two-burn family) ===")
    print(f"Baseline coast to crossing: {base.days:.6f} days")
    print(f"Proposed departure dvr={dvr:.3f}, dvt={dvt:.3f} m/s")
    print(f"Predicted crossing time: {prediction.days:.6f} days")
    print(f"Predicted departure delta-v: {prediction.departure_dv:.3f} m/s")
    print(f"Predicted capture delta-v: {prediction.insertion_dv:.3f} m/s")
    print(f"Predicted total delta-v: {prediction.total_dv:.3f} m/s")
    print(f"Predicted coast-time saving: {base.days - prediction.days:.3f} days")

    # Replace both impulses with finite-thrust, variable-mass guidance.
    s, dep_time, dep_done = finite_burn(S0.copy(), "departure", dvr, dvt)
    if not dep_done:
        raise RuntimeError("Departure guidance failed or ran out of propellant")
    after_dep_mass = s[4]
    coast = propagate(s, MAX_COAST_DAYS * 86400.0, events=outward_crossing)
    if len(coast.t_events[0]) == 0:
        raise RuntimeError("No outward crossing in finite-thrust replay")
    s = coast.y[:, -1].tolist()
    r, vr, vt = components(s)
    arrival_time_days = (dep_time + coast.t[-1]) / 86400.0
    print("\n=== Finite-thrust, variable-mass replay ===")
    print(f"Departure burn: {dep_time:.3f} s; mass after departure: {after_dep_mass:.3f} kg")
    print(f"Coast + departure time to crossing: {arrival_time_days:.6f} days")
    print(f"Crossing radial speed: {vr:.3f} m/s; tangential speed: {vt:.3f} m/s")
    s, cap_time, cap_done = finite_burn(s, "capture")
    r, vr, vt = components(s)
    e = eccentricity(s)
    radius_error = abs(r - TARGET_R) / TARGET_R
    remaining_propellant = s[4] - DRY_MASS
    used_dv = VE * math.log(WET_MASS / s[4])
    print(f"Capture burn: {cap_time:.3f} s")
    print(f"Final mass: {s[4]:.6f} kg")
    print(f"Remaining propellant: {remaining_propellant:.6f} kg")
    print(f"Used ideal delta-v by mass ratio: {used_dv:.3f} m/s")
    print(f"Radius error ratio: {radius_error:.10f}; eccentricity: {e:.10f}")
    print(f"Radial velocity: {vr:.6f} m/s; local tangential error: {vt - math.sqrt(MU/r):.6f} m/s")
    reached = cap_done and e < 0.001 and radius_error < 0.0025
    reserve_preserved = remaining_propellant >= RESERVE_PROPELLANT_KG - 1e-3
    print(f"Finite-thrust near-circular capture: {reached}")
    print(f"Reserve >= {RESERVE_PROPELLANT_KG:.1f} kg: {reserve_preserved}")
    if reached:
        days = [0, 25, 50, 75, 100]
        hold = propagate(s, 100 * 86400.0, t_eval=[d * 86400.0 for d in days])
        errors = [abs(math.hypot(x, y) - TARGET_R) / TARGET_R
                  for x, y in zip(hold.y[0], hold.y[1])]
        print(f"100-day follow-through sampled max radius error ratio: {max(errors):.10f}")
        print(f"Final sampled eccentricity: {eccentricity(hold.y[:, -1]):.10f}")
    print("Experiment changes the time horizon and model, NOT frozen Phase35 metrics.")


if __name__ == "__main__":
    main()
