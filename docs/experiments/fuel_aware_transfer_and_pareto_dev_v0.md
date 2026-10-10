# Fuel-Aware Transfer and Time–Fuel Candidates — Development Experiment v0

**Observed on:** 2026-10-10
**Classification:** Offline exploratory simulation, **not a frozen Phase35 benchmark or Final Veto result**
**Primary log:** [2026-10-10 project log](../../project_log/2026-10-10_fuel_aware_transfer_and_pareto_dev.md)

## 1. Research Question

Within a restricted `departure burn -> coast -> capture burn` family, can a controller starting at the Phase35-like nominal initialization reach a near-circular orbit while honoring a finite mass-based propulsion budget? How does the earliest *found* target-radius crossing time change when minimum remaining propellant is varied?

## 2. Dynamics and Modeling Contracts

| Quantity | Development value |
|---|---:|
| Gravitational parameter `mu` | `1.3275182699999999e20 m^3/s^2` |
| Target radius | `7.5e12 m` |
| Initial radius ratio | `0.98` |
| Initial velocity angle | `150 deg` |
| Starting state | `x=0, y=0.98*R, v=sqrt(mu/r0)*(cos(150 deg), sin(150 deg))` |
| Wet / dry spacecraft mass | `722 / 300 kg` |
| Propellant capacity | `422 kg` |
| Ideal thrust during burn | `2000 N` |
| Constant specific impulse | `450 s` |
| Standard gravity | `9.80665 m/s^2` |
| Direction update interval | `100 s` maximum |
| Gravity / powered propagation | SciPy `solve_ivp(..., method="DOP853")` |
| Search horizon in optimizer | `1200 days` |
| Full ideal delta-v budget | `3875.678290... m/s` |
| Additional optimizer reserve margin | `0.5 kg` |

The 2000 N propulsion assumption is a development interpretation of `thrust_scale=8000` and a `0.25` normalized action magnitude. It must not be represented as a real spacecraft engine specification or as unchanged Phase35 physics.

The finite-thrust model integrates the 5D state `(x, y, vx, vy, m)` with `dm/dt=-T/(Isp*g0)` during burns. At each guidance interval it recomputes a desired velocity-error direction, holds thrust direction for that integration segment, and caps duration against the dry-mass floor. Coast segments have no mass loss. The search model, by contrast, uses ideal instantaneous velocity changes and orbital energy/angular-momentum propagation; it is **not** itself a finite-thrust optimal-control transcription.

## 3. Experiment Phases

### 3.1 Stress-boundary controller diagnostics (a different starting state)

**Read-only source:** `analysis/stage2a_post_veto_alternative_audit_v0/exact_state_comparisons.json`, first frozen comparison. This is a **preterminal diagnostic state**, not the Phase35 nominal initial state.

- Event-triggered coast-to-brake boundary at `282.135670 days`.
- Radial brake: `22 x 100 s`, ends at `vr=70.030958 m/s` and `vt=4954.537344 m/s`.
- Circularization: `3` more intervals to an approximately circular osculating state.
- Fixed-mass accumulated commanded delta-v: `6094.183 m/s` radial + `750.645 m/s` trim = `6844.828 m/s` total.
- Delta-v-capped fixed-mass replay (including reduced final action) consumed `3875.678 m/s`, ended with `vr=2288.535 m/s`, and failed capture.
- Variable-mass radial-only replay hit dry mass during step 10 and ended with `vr=2288.534 m/s`, `e=0.74833`.
- Constant thrust-direction offsets of 10/15/20 degrees reduced terminal eccentricity relative to 0 degrees, but all failed target capture.

These are distinct development models and should not be pooled as one formal comparison.

### 3.2 Nominal-initialization impulse probes

Reconstructed the true Phase35-like initial geometry above, with no frozen Phase35 controller rollout. Initial adjustment candidates tested over ~115.74 and 365 days:

| Initial ideal impulse | Total impulse (m/s) | Crossing by 115.74d? | Crossing by 365d? | If crossed: arrival time |
|---|---:|---|---|---:|
| `coast_only` | `0` | No | No | — |
| `radial_2500` | `2500` | No | No | — |
| `radial_3500` | `3500` | No | Yes | `309.10 days` |
| `mixed_3000_neg1500` | `3354.1` | No | Yes | `340.51 days` |

The crossing-producing impulse probes did not show terminal capture; their high arrival radial velocities and small remaining propellant budgets motivated an upstream optimization that reserves capture fuel explicitly.

### 3.3 Long unpowered coast plus finite-thrust capture

- First outward target-radius crossing: **`825.584095 days`**.
- At crossing: `vr=2080.705432 m/s`, `vt=3606.893268 m/s`.
- Ideal local circular speed: `4207.165744 m/s`.
- Ideal single-impulse velocity correction: `2165.562776 m/s`.
- Finite capture: `617.828 s` total powered time, mass `441.996007 kg`, propellant used `280.003993 kg`.
- Terminal reported radial velocity and local tangential error near zero; eccentricity reported `~1e-10`.
- One 100-day unpowered follow-through sampled at five points remained close to target radius in this deterministic model.

This is the direct evidence that simply extending the time horizon can change fuel feasibility in the development model.

### 3.4 Restricted upstream optimization and replay

**Optimizer:** SciPy `minimize(method="SLSQP")`, bounded `dvr,dvt`, multiple seeds. Search feasibility requires a predicted outward crossing, a predicted total impulse within the chosen mass-derived budget, and an initial speed headroom constraint. Pick the fastest **returned feasible** candidate. This is local, multi-start numerical search; it cannot certify a global optimum or all possible two-burn solutions.

**Execution replay:** Run a finite-thrust departure velocity-targeting burn; propagate under gravity until outward target-radius crossing; run a finite-thrust local circular-velocity-targeting capture burn. Require eccentricity `<0.001`, absolute radius error ratio `<=0.0025`, and successful burn convergence. Also require final propellant to meet the requested minimum reserve.

For the reserve-20 run:

| Field | Prediction / replay observation |
|---|---:|
| Departure proposal `dvr` | `+717.281 m/s` |
| Departure proposal `dvt` | `+120.668 m/s` |
| Predicted first crossing | `613.679163 days` |
| Continuous first crossing | `613.679425 days` |
| Departure burn duration | `242.079 s` |
| Capture burn duration | `643.829 s` |
| Final mass | `320.499941 kg` |
| Final propellant | `20.499941 kg` |
| Integrated mass-ratio delta-v | `3583.981 m/s` |
| Radius error ratio | `0.0000001338` |
| Terminal eccentricity | `~8e-10` |
| Capture / reserve requirement | `True / True` |

The predicted total impulse `3583.980 m/s` and mass-ratio replay `3583.981 m/s` are close, but the search and replay are different dynamical models. Their agreement here is a useful single-case observation, not a validated global modeling-error bound.

## 4. Propellant-Reserve Sweep: Complete Observed Data

**Run:** `python -m scripts.dev_time_fuel_pareto_v0`
**Candidate reserve requirements (kg):** `0, 20, 50, 80, 110, 140, 150`
**Extra numerical planning margin:** `0.5 kg`

| Required reserve (kg) | Predicted crossing (days) | Replay crossing (days) | Used ideal Δv (m/s) | Remaining propellant (kg) | `dvr` proposal (m/s) | `dvt` proposal (m/s) | Eccentricity | Radius error ratio | Capture / reserve met |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 583.94 | **583.95** | 3868.3 | 0.50 | +860.35 | +133.96 | 8.70e-10 | 1.42e-7 | True / True |
| 20 | 613.68 | **613.68** | 3584.0 | 20.50 | +717.28 | +120.67 | 7.83e-10 | 1.34e-7 | True / True |
| 50 | 660.52 | **660.52** | 3189.1 | 50.50 | +518.33 | +98.21 | 6.05e-10 | 1.23e-7 | True / True |
| 80 | 710.50 | **710.50** | 2826.7 | 80.50 | +335.35 | +71.91 | 4.09e-10 | 1.12e-7 | True / True |
| 110 | 764.25 | **764.25** | 2491.8 | 110.50 | +165.80 | +40.51 | 2.32e-10 | 1.02e-7 | True / True |
| 140 | 822.54 | **822.54** | 2180.5 | 140.50 | +7.62 | +2.14 | 1.00e-10 | 9.31e-8 | True / True |
| 150 | — | — | — | — | — | — | — | — | **No feasible candidate returned** |

The solver returned six candidates, and **six of six returned candidates** passed the defined continuous-replay capture and reserve predicates. The denominator here is *returned candidate reserve levels*, not all cases in Phase35 or any out-of-sample test set.

The `150 kg` row is a search failure / no-candidate result within this restricted formulation, **not proof of physical infeasibility**. The `0 kg` requirement succeeded only with the deliberate `0.5 kg` margin and is not a recommended mission operating reserve.

## 5. Interpretation and Evidence Limits

- **Observed tradeoff:** Increased required propellant reserve corresponded to increased earliest *found* crossing time in this restricted family.
- **Do not claim:** A global Pareto frontier, globally minimum-time transfer, robust spacecraft autonomy, real engineering feasibility, or updated Phase35 benchmark success.
- **Original Phase35 context:** Original benchmark has a ~115.74-day horizon and fixed controller/threshold conditions; today used much longer horizons, newly introduced mass depletion, different feedback guidance, and a new two-burn optimizer.
- **No generalized survival evidence:** The baseline long-coast and one `20 kg` candidate had five-point, 100-day follow-through outputs. The reserve sweep itself reported terminal states rather than independent follow-through tests for every row.
- **Idealizations:** exact state knowledge; 2D Sun-only gravity; fixed constant thrust and Isp; no stochastic navigation errors, actuator errors, duty-cycle/thermal constraints, third bodies, or hardware.
- **Numerical qualification:** Printed `~1e-10` eccentricities are finite-precision modeled results, not proofs of exactly circular trajectories.

## 6. Reproduction Entry Points

Run from the repository root in the `spacecraft` Conda environment, with the development scripts present under `scripts/`:

```powershell
conda activate spacecraft
python -m scripts.dev_coast_brake_fuelcap
python -m scripts.variable_mass_capture_dev
python -m scripts.dev_upstream_injection_v0
python -m scripts.dev_long_coast_capture_v0
python -m scripts.dev_fuel_aware_transfer_optimizer_v0
python -m scripts.dev_time_fuel_pareto_v0
```

`dev_time_fuel_pareto_v0.py` imports `dev_fuel_aware_transfer_optimizer_v0.py` and changes its in-memory planning-budget variable for each reserve. It prints to stdout and does not, by design, write a result CSV. For durable comparison, capture console outputs separately if desired, without overwriting the frozen `analysis/` tree.

**Current evidence basis:** User-provided console outputs from 2026-10-10 and the associated local development scripts. No SHA-pinned commit, raw trajectory bundle, or formal benchmark manifest is claimed for this development series.
