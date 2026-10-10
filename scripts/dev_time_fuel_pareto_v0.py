"""Explore a time/fuel frontier for the *restricted* Phase35-like two-burn family.

Run from the repository root with:
    python -m scripts.dev_time_fuel_pareto_v0

Requires scripts/dev_fuel_aware_transfer_optimizer_v0.py next to this file.
The optimizer uses impulsive predictions; each candidate is independently
replayed with finite thrust and variable mass. Frozen evidence is untouched.
This is a development-only 2-D solar two-body model, not flight validation.
"""

import math

from scripts import dev_fuel_aware_transfer_optimizer_v0 as core

# Reserve requirements selected in advance, in kilograms.
RESERVES_KG = (0.0, 20.0, 50.0, 80.0, 110.0, 140.0, 150.0)


def evaluate(reserve):
    """Return a single prediction/replay record; no files are written."""
    planning_dry = core.DRY_MASS + reserve + core.SEARCH_MARGIN_KG
    if planning_dry >= core.WET_MASS:
        return None, "reserve exceeds onboard propellant"

    # optimize_injection reads this module-level budget in the existing v0.
    core.DV_PLANNING = core.VE * math.log(core.WET_MASS / planning_dry)
    try:
        _, controls, predicted = core.optimize_injection()
    except RuntimeError as exc:
        return None, str(exc)

    dvr, dvt = map(float, controls)
    try:
        state, depart_seconds, depart_ok = core.finite_burn(
            core.S0.copy(), "departure", dvr, dvt
        )
        if not depart_ok:
            return None, "finite departure burn did not converge"

        coast = core.propagate(
            state, core.MAX_COAST_DAYS * 86400.0,
            events=core.outward_crossing,
        )
        if len(coast.t_events[0]) == 0:
            return None, "no outward crossing in continuous replay"

        state = coast.y[:, -1].tolist()
        arrival_days = (depart_seconds + float(coast.t[-1])) / 86400.0
        state, capture_seconds, capture_ok = core.finite_burn(state, "capture")
    except (RuntimeError, ValueError) as exc:
        return None, f"continuous replay failed: {exc}"

    r, vr, vt = core.components(state)
    e = core.eccentricity(state)
    radius_err = abs(r - core.TARGET_R) / core.TARGET_R
    actual_reserve = state[4] - core.DRY_MASS
    used_dv = core.VE * math.log(core.WET_MASS / state[4])
    captured = capture_ok and e < 0.001 and radius_err <= 0.0025
    reserve_met = actual_reserve >= reserve - 1e-3

    return {
        "required_reserve": reserve,
        "predicted_days": predicted.days,
        "arrival_days": arrival_days,
        "departure_dvr": dvr,
        "departure_dvt": dvt,
        "predicted_dv": predicted.total_dv,
        "actual_dv": used_dv,
        "actual_reserve": actual_reserve,
        "captured": captured,
        "reserve_met": reserve_met,
        "eccentricity": e,
        "radius_error": radius_err,
        "departure_seconds": depart_seconds,
        "capture_seconds": capture_seconds,
    }, None


def main():
    print("=== Time-Fuel Pareto v0: restricted two-burn family ===")
    print(f"Phase35-like initial condition: r/R={core.R0_RATIO}, angle={core.ANGLE_DEG} deg")
    print(f"Initial fuel: {core.WET_MASS - core.DRY_MASS:.1f} kg")
    print(f"Planner reserve margin: {core.SEARCH_MARGIN_KG:.1f} kg")
    print("\nReserve | Pred days | Replay days | Used Δv | Fuel left | Capture | Reserve OK")
    print("-" * 90)

    rows = []
    for reserve in RESERVES_KG:
        result, error = evaluate(reserve)
        if error:
            print(f"{reserve:7.1f} | {'NO FEASIBLE CANDIDATE':<21} | {error}")
            continue
        rows.append(result)
        print(
            f"{reserve:7.1f} | {result['predicted_days']:9.2f} | "
            f"{result['arrival_days']:11.2f} | "
            f"{result['actual_dv']:7.1f} | "
            f"{result['actual_reserve']:9.2f} | "
            f"{str(result['captured']):7} | "
            f"{result['reserve_met']}"
        )

    print("\n=== Candidate departure controls ===")
    for row in rows:
        print(
            f"reserve={row['required_reserve']:.0f} kg: "
            f"dvr={row['departure_dvr']:+.2f} m/s, "
            f"dvt={row['departure_dvt']:+.2f} m/s, "
            f"e={row['eccentricity']:.2e}, "
            f"radius_error={row['radius_error']:.2e}"
        )

    if rows:
        success = [r for r in rows if r['captured'] and r['reserve_met']]
        print(f"\nFinite-replay candidates meeting both conditions: {len(success)}/{len(rows)}")
        if success:
            quickest = min(success, key=lambda r: r['arrival_days'])
            print(
                f"Fastest confirmed among these reserves: "
                f"{quickest['arrival_days']:.2f} days, "
                f"fuel left={quickest['actual_reserve']:.2f} kg"
            )
    print("Not a global Pareto frontier, not an original Phase35 benchmark result.")


if __name__ == "__main__":
    main()
