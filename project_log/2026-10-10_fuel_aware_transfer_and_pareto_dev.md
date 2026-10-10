# Project Log — Fuel-Aware Transfer and Time–Fuel Tradeoff (2026-10-10)

**Date:** 2026-10-10
**Project:** Spacecraft AI Controller — exploratory 2D orbital-transfer development
**Status:** Development-only numerical experiments; **not** a new frozen Phase35/Stage2 result
**Starting point:** Phase35-like initial condition, `r0 / R_target = 0.98`, initial velocity angle `150 deg`, `thrust_scale = 8000` where relevant.

## 1. Today's Objective

Move beyond a controller that merely crosses a target radius. Determine whether a spacecraft with finite propulsion can arrive with a capturable velocity, and whether an optimizer can trade flight time against remaining propellant. Do this without modifying the frozen Phase34/35 controller, Final Veto logic, Stage2 evidence, or established benchmark thresholds.

The engineering question evolved during the day:

> Can the transfer trajectory be shaped early enough to reserve fuel for finite-thrust circularization, instead of relying on an expensive last-minute braking maneuver?

## 2. Sequence of Experiments and Decisions

### A. Braking-distance-aware coast, radial braking, circularization

A frozen Phase35 stress **boundary state** was used as a development initial condition, not as a fresh Phase35 rollout. The approximate maximum action magnitude was `0.25`, corresponding to `2000 N` in the experimental model with `722 kg` fixed mass.

- Ideal radial braking acceleration: `2.770083 m/s^2`.
- Ideal stopping time: `2216.782 s`.
- Ideal stopping distance: `6,806,266 m`.
- Initial target-radius gap / braking distance: `22,036.85`.

Rather than immediately braking, an event-triggered zero-thrust DOP853 coast located a later braking point at **282.135670 days**. Radial braking then reached the broad component thresholds after **22** control intervals (`100 s` each): radial speed `70.030958 m/s`, tangential speed `4954.537344 m/s`, and remaining radius gap `1,372,592.966 m`.

Those component thresholds did **not** mean circular insertion: tangential speed was still substantially above the local circular value. Three additional circularization trim intervals reduced the osculating eccentricity to the printed numerical-zero range (`e` printed `0.00000000`).

**New insight:** Separate a recoverability/component-threshold milestone from a physically near-circular osculating state.

### B. Propulsion cost exposed the failure of late braking

The fixed-mass development controller accumulated:

| Maneuver | Commanded delta-v |
|---|---:|
| Radial braking | `6094.183 m/s` |
| Circularization trim | `750.645 m/s` |
| **Total** | **`6844.828 m/s`** |

An illustrative rocket-equation budget using `m_wet=722 kg`, `m_dry=300 kg`, and `Isp=450 s` gave **`3875.678 m/s`** available. Under the fixed-mass command-budget approximation, the final partial-action version exhausted that budget on radial braking step **14** with `2288.535 m/s` radial speed remaining. Circularization could not start.

The model was then upgraded to integrate spacecraft mass as a fifth state, with `2000 N` maximum thrust and constant `450 s` specific impulse. Propellant exhaustion occurred **31.141 s into step 10**; final mass was **`300.000 kg`**, radial speed **`2288.534 m/s`**, and eccentricity **`0.74833`**. This reproduced the broad budget limitation with physically explicit mass depletion, not simply an imposed command counter.

### C. Fixed thrust steering helped orbit shape, not target capture

With the same fuel supply and variable-mass propagation, a limited thrust-angle experiment gave:

| Steering angle | End eccentricity | End radial speed | Tangential error ratio |
|---|---:|---:|---:|
| `0 deg` | `0.74833` | `2288.534 m/s` | `0.177643` |
| `10 deg` | `0.56894` | `2347.414 m/s` | `0.017677` |
| `15 deg` | `0.55308` | `2420.594 m/s` | `0.060783` |
| `20 deg` | `0.57701` | `2522.266 m/s` | `0.137429` |

Changing thrust direction significantly changed the final osculating orbit but none of the tested angles captured the target. A reduced eccentricity was not interchangeable with reduced radial velocity, and a fixed steering angle did not fix the large terminal-capture demand.

**Decision:** Stop tuning late-only braking and move the control decision upstream.

### D. Reconstruct the actual Phase35-like initial geometry

The earlier stress boundary was replaced by the initial-state formula used in Phase35: `r0/R_target=0.98`, angle `150 deg`. The development experiments reconstructed this geometry but **did not run the frozen Phase35 controller**.

An ideal impulsive upstream probe showed that large initial radial pushes could produce a target-radius crossing in an extended 365-day horizon, yet retained insufficient delta-v for subsequent capture. The unpowered trajectory did not cross in that 365-day window.

Extending the unpowered coast to its first outward crossing gave **`825.584095 days`**, radial speed **`2080.705432 m/s`**, tangential speed **`3606.893268 m/s`**, and ideal instantaneous capture requirement **`2165.562776 m/s`**. A separate finite-thrust, variable-mass local capture replay completed in **`617.828 s`**, used **`280.003993 kg`** propellant, and ended at **`441.996007 kg`** with near-zero numerical eccentricity. Five sampled points spanning the next 100 unpowered days stayed near the target radius.

**New insight:** A long transfer can lower arrival-energy correction enough to permit capture within the same fuel supply. The tradeoff is significantly increased flight time.

### E. Fuel-aware two-burn optimizer

An exploratory SciPy multi-start SLSQP optimizer selected initial radial and tangential velocity changes in an **impulsive search model**. Every selected candidate was then replayed as an **initial finite-thrust burn -> unpowered coast -> final finite-thrust capture**, using continuous DOP853 integration and variable spacecraft mass.

For a `20 kg` requested propellant reserve (plus a `0.5 kg` planner margin):

- Departure proposal: `dvr=+717.281 m/s`, `dvt=+120.668 m/s`.
- Predicted crossing: `613.679163 days`; continuous replay crossing: **`613.679425 days`**.
- Finite departure burn: `242.079 s`; finite capture burn: `643.829 s`.
- Delivered delta-v from mass ratio: **`3583.981 m/s`**.
- Final mass: **`320.499941 kg`**, propellant remaining **`20.499941 kg`**.
- Final radius error ratio: `0.0000001338`; osculating eccentricity: approximately `8e-10`.
- Five sampled points across a 100-day unpowered follow-through remained near the circular target in this numerical model.

Relative to unpowered departure and later capture, target-radius arrival was about **`211.905 days` earlier**. This is a feasible *candidate* within the restricted two-burn family, **not a globally optimal solution**.

### F. Time–fuel reserve sweep

Using the same optimizer and finite-thrust replay, the following six candidate reserve levels were achieved:

| Required reserve (kg) | Crossing time (days) | Delivered delta-v (m/s) | Remaining propellant (kg) | Near-circular capture / reserve met |
|---:|---:|---:|---:|---|
| 0 | 583.95 | 3868.3 | 0.50 | True / True |
| 20 | 613.68 | 3584.0 | 20.50 | True / True |
| 50 | 660.52 | 3189.1 | 50.50 | True / True |
| 80 | 710.50 | 2826.7 | 80.50 | True / True |
| 110 | 764.25 | 2491.8 | 110.50 | True / True |
| 140 | 822.54 | 2180.5 | 140.50 | True / True |

For a `150 kg` required reserve the constrained search reported **`NO FEASIBLE CANDIDATE`**. This means only that its numerical search did not return a feasible candidate in the tested two-burn parameterization; it is **not** a proof of physical infeasibility.

Across the **six returned candidates**, finite-thrust replay satisfied both the code's near-circular capture predicate and the corresponding reserve requirement (`6/6`). It is **not** `6/6` Phase35 benchmark success.

## 3. Main Engineering Findings

1. **Capture condition matters:** Radius crossing, Phase34-style component thresholds, eccentricity reduction, and sustained orbit retention are different measurements.
2. **Propellant is a binding constraint:** The 282-day reactive braking scheme demanded `6844.828 m/s` in its fixed-mass command estimate, well above the ideal `3875.678 m/s` supply for the selected propulsion assumptions.
3. **Upstream planning changes feasibility:** Starting from the Phase35-like initial geometry, long coasting enabled low-cost capture; adding an optimized departure maneuver reduced crossing time while preserving user-selected propellant reserves.
4. **There is a time–fuel tradeoff:** In this restricted family, shorter demonstrated crossing times required greater propellant use.
5. **A single narrow case is not a general controller:** Neither robustness nor full-benchmark generalization has been established.

## 4. Evidence and Scope Boundaries

- Development scripts used: `scripts/dev_coast_brake.py`, `scripts/dev_coast_brake_fuelcap.py`, `scripts/variable_mass_capture_dev.py`, `scripts/dev_thrust_steering.py`, `scripts/dev_upstream_injection_v0.py`, `scripts/dev_long_coast_capture_v0.py`, `scripts/dev_fuel_aware_transfer_optimizer_v0.py`, `scripts/dev_time_fuel_pareto_v0.py`. Their presence in this day's working tree does not imply they are already tracked on `main`.
- Model: planar Cartesian Sun-centered point-mass gravity (`mu=1.32751827e20` SI); target radius `7.5e12 m`; wet/dry mass `722/300 kg`; ideal constant `2000 N` thrust; constant `450 s` specific impulse for variable-mass runs. DOP853 numerical integration was used for the exploratory continuous dynamics.
- The optimizer searches **only one initial burn, coast, and one capture burn**, with impulsive planning and continuous finite-thrust replay. No global-optimality certificate exists.
- Time horizons were extended well beyond the original `100000 x 100 s` (~`115.74 day`) Phase35 rollout window.
- No external perturbations, measurement noise, thrust misalignment, engine duty-cycle limits, communications delay, third-body forces, or uncertainty-aware feedback were demonstrated.
- Five-point/100-day follow-through was shown for the named baseline and 20 kg optimizer case; it was **not separately reported for all six reserve candidates**.
- Frozen Phase34/35 outcomes, Final Veto, Stage2 artifacts, and benchmark counts remain unchanged. This work must not be appended to the `8/24` benchmark as new successes.

## 5. Next Research Decision

**Keep the successful two-burn optimizer as the development baseline; do not integrate it into protected runtime authority yet.** Next, make a small robustness experiment around the `80 kg` required-reserve candidate (`710.50 days`, `80.50 kg` remaining). Introduce bounded initial velocity / navigation-state errors, then compare open-loop replay against one midcourse replan with the same fuel and terminal definitions. Report failure as well as success, and keep the original development result reproducible.

## 6. Bottom Line

Today converted an overspeed/recoverability stress-state diagnosis into a restricted **time–fuel-aware orbital-transfer planning prototype**. The key achievement is **not** improving Phase35's historical crossing count. It is demonstrating, for one idealized initial condition, that a fuel-constrained trajectory can be planned from departure, replayed with variable-mass finite thrust, captured near circular, and parameterized by propellant reserve.
