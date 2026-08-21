# Stage 2A Prediction-Boundary Discovery v0

Completed: 2026-08-21

## Status

Frozen bounded discovery infrastructure and source plan implemented. No discovery trajectory is executed by implementation, validation, plan, or checker modes.

## Purpose

This discovery asks whether existing simulator behavior and existing recovery branches naturally reach a state where the realized speed ratio is at most 1.90 while the one-step prediction for the declared normal action is strictly greater than 1.90. Such a state is a prediction-boundary candidate. It is not evidence that a hazard-arrest intervention works.

## Existing Parameter-Space Audit

The frozen Final Veto inventory supplies 13 complete cases. The Phase34 family uses initial radius ratio 1.00, velocity angles 150, 165, 170, and 175 degrees, and thrust scales 8000 and 10000. The Phase35 radial-energy-push family uses initial radius ratio 0.98, thrust 10000 at the same four angles, plus the legacy 150-degree, thrust-8000 case. All cases use seed 0 and the frozen radius-priority post-cross mode.

The physical dimensions affecting approach to the overspeed boundary are the initial radius ratio, initial velocity angle, thrust scale, source controller family, case-specific branch boundary, and recovery branch. The plan does not form a new Cartesian product. It uses only the 13 explicit repository case identities and their provenance-bound case-specific boundaries.

## Prior Coverage Gap

Stage 1B published recovery traces for four registered branch states and three branches. Stage 2A qualification inspected those frozen traces and found zero eligible boundaries. Nine other provenance-complete Final Veto source cases were not represented as Stage 1B branch traces. The bounded extension therefore uses all 13 frozen source cases, extracts each existing case-specific boundary deterministically, and applies the three already implemented recovery branches for at most 32 transitions.

## Frozen Search

The source plan is `configs/stage2a_prediction_boundary_discovery_v0.json`. It freezes 13 cases, three branches, 39 trajectories, deterministic case-lexical and declared-branch ordering, and at most 32 recovery-branch transitions per trajectory. Prefix extraction transitions are separately accounted because the source cases have different validated boundaries and do not represent synchronized physical time.

The allowed branch IDs are:

- `zero_action_reference_v0`
- `tangential_error_correction_v0`
- `velocity_opposed_thrust_v0`

Velocity-opposed thrust is an ordinary existing recovery branch in this dataset. It is not Stage 2A hazard authority.

## Candidate Contract

A candidate requires both:

```text
current realized_speed_ratio <= 1.90
normal predicted_speed_ratio > 1.90
```

The comparator is exact and inherited. Exactly 1.90 is clear. Candidate evidence is captured before execution. The unchanged Final Veto must reject the proposal. The trajectory stops at its first candidate, the vetoed action executes no transition, no fallback executes, and no Stage 2A authority adapter is called.

## Runtime Reuse

The implementation reuses:

- `build_source_case_inventory` and `execute_nominal_prefix` for complete source provenance and deterministic case-boundary extraction;
- existing zero, tangential-correction, and velocity-opposed action generators;
- `evaluate_overspeed_veto` for the sole Final Veto decision;
- `step_phase34_35_transition` for prediction and allowed physical transitions;
- the existing canonical state and artifact hashing conventions;
- the existing bounded atomic directory publisher.

No orbital dynamics, action law, or veto implementation is duplicated.

## Ordering and Termination

At each opportunity, the discovery computes the current ratio, existing branch action, one-step prediction, predicted ratio, and Final Veto result. A candidate or any veto stops the trajectory before physical execution. Current realized overspeed remains an adverse terminal condition. Allowed actions execute through the existing transition function, and predicted versus realized transition equality is required.

## Diagnostics

The result profiles predictions below 1.80, from 1.80 to below 1.85, from 1.85 through 1.90, and above 1.90. The 1.80 and 1.85 values are diagnostic bins only. They are not active triggers, release thresholds, calibrated guard values, or scientific claims.

## Authority Boundary

```text
active_authority_granted = false
hazard_arrest_interventions = 0
fallback_execution_count_at_veto = 0
```

The authority adapter is not imported. No authority token exists in the discovery path. Existing normal branches alone supply physical actions.

## Determinism and Provenance

Every trajectory records source-case and configuration hashes, simulator and constants hashes, transition and controller hashes, case-specific prefix counts, prefix action and state trace hashes, boundary-state hash, executed action and realized state trace hashes, and an aggregate source trajectory hash. Candidate records additionally bind the implementation commit and frozen plan hash.

## Publication

The formal command publishes one complete directory atomically. Existing output is rejected. Infrastructure failure publishes nothing. Baselines, active-intervention traces, fabricated states, and partial results are not published.

## Claim Restrictions

This bounded discovery can establish only whether the frozen search did or did not naturally produce the defined prediction boundary under existing simulator, branch-action, prediction, and Final Veto semantics. It does not demonstrate hazard-arrest effectiveness, recovery improvement, active-controller safety, stability, optimality, new threshold validity, handoff readiness, multi-step recovery, hardware validity, or deployment readiness.
