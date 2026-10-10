# Fuel-Aware Two-Burn Transfer Architecture — Development v0

**Status:** Experimental controller and optimizer architecture, 2026-10-10
**Authority:** Offline candidate generation and continuous replay **only**; no connection to frozen Phase34/35 runtime control or Final Veto authority.

## 1. Motivation

Earlier development experiments showed that a long coast followed by aggressive radial braking could recover a near-circular state in a constant-mass model but required an estimated `6844.828 m/s` command delta-v. This exceeded the `3875.678 m/s` ideal mass-derived budget with `722 kg` wet mass, `300 kg` dry mass and `450 s` Isp.

The architecture therefore moves the planning decision **before** the transfer arc. Rather than picking only a terminal brake, it chooses a departure velocity adjustment while reserving mass for terminal circularization.

## 2. Control/Data Flow

```text
Phase35-like nominal initial state (reconstructed; not frozen rollout)
    |
    v
Two-variable impulsive search: (departure delta-v_radial, delta-v_tangential)
    |   constraints: predicted crossing, speed headroom,
    |                total predicted delta-v <= mass-based planning budget
    v
Candidate departure maneuver
    |
    v
Finite-thrust variable-mass departure guidance (receding direction updates)
    |
    v
Unpowered DOP853 coast -> first outward target-radius event
    |
    v
Finite-thrust variable-mass local circular-velocity matching
    |
    v
Terminal diagnostics: radius error, radial/tangential velocity, eccentricity,
                      propellant remaining, reserve predicate
```

The optimizer's prediction and the finite-thrust replay are deliberately separate. A feasible impulsive prediction does not automatically count as a successful finite-thrust outcome.

## 3. State and Dynamics

Development state: `s = [x, y, vx, vy, m]`.

For planar gravity, thrust `T`, constant specific impulse `Isp`, and unit thrust direction `(ux, uy)`:

```text
r = sqrt(x^2 + y^2)
xdot = vx
ydot = vy
vxdot = -mu*x/r^3 + T*ux/m
vydot = -mu*y/r^3 + T*uy/m
mdot = -T/(Isp*g0)
```

During coast, `T=0`. Thrust is limited by the development constant `T=2000 N` during commanded burns and propellant use stops at `m_dry=300 kg`. Control direction is recomputed at most every `100 s`; terminal intervals may be shorter. The controller uses ideal instantaneous velocity measurement in its guidance computation, even though acceleration and mass are integrated continuously during each held-direction segment.

## 4. Terminal Objectives Versus Other Repo Milestones

A **geometric crossing** is an outward target-radius event. A **near-circular development capture** means the capture guidance converged and its osculating eccentricity is `<0.001` with target radius error ratio `<=0.0025` (and sufficient residual mass when required). These conditions are distinct from Phase34's historical recovery/component thresholds and Phase35's frozen CAPTURE/LOCK/success labels.

The development architecture cannot inherit frozen Phase35 result counts because it changes the controller, numerical integrator, mass state, objective, and time horizon.

## 5. Search Contract

- Decision variables: just `dvr` and `dvt` at departure in the impulsive surrogate.
- Optimizer: bounded, multi-start `SLSQP`, initialized with a small fixed seed set.
- Objective: minimize predicted first outward crossing time, not actual end-of-powered-capture time; burn durations are added during replay.
- Planning budget: rocket equation using a required propellant reserve and a separate `0.5 kg` search margin.
- Terminal capture: controlled velocity-vector matching to local circular velocity at actual radius, not merely reduction of total speed.
- Feasibility assessment: final finite-thrust replay convergence + terminal near-circular predicate + remaining propellant threshold.

**Limitation:** This is an approximate, constrained two-burn family and finite-thrust replay, not optimized continuous steering, direct collocation, MPC, or a certificate of global optimality. Multi-start SLSQP may fail to identify existing feasible solutions.

## 6. Interfaces and Isolation

Relevant development files:

- `scripts/dev_fuel_aware_transfer_optimizer_v0.py` — optimizer and continuous replay.
- `scripts/dev_time_fuel_pareto_v0.py` — fixed reserve sweep over the optimizer.
- `scripts/dev_long_coast_capture_v0.py` — no-departure-burn finite-thrust reference.
- `scripts/variable_mass_capture_dev.py` — diagnostic boundary-state variable-mass radial braking.

These scripts should remain isolated from:

- `scripts/explicit_controller_phase35_crossing_basin_expansion.py` and Phase34 terminal behavior;
- `runtime_assurance/` Final Veto and recovery authority;
- frozen `analysis/` records and manifests.

The development scripts may be staged as new files in a later explicit commit, but frozen artifacts should not be regenerated or silently replaced.

## 7. Next Engineering Extension

Use the `80 kg` reserve candidate as a nominal planning test case, then add a bounded initial-velocity mismatch and a simulated navigation error. Evaluate **open-loop vs one midcourse replan** under identical propulsion limits and terminal metrics. The research milestone is whether feedback can preserve capture and fuel reserve under uncertainty, not whether a new standalone best-time point can be printed.
