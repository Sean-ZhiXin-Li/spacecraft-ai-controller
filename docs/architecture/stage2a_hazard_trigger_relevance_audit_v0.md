# Stage 2A Hazard Trigger Relevance Audit v0

Completed: 2026-09-04

## Status

Frozen offline evidence audit implemented. No simulator, controller, trajectory,
Final Veto, or Stage 2A authority change is part of this audit.

## Purpose

This audit distinguishes two predicates that must not be merged:

1. **Trigger A** applies the strict overspeed predicate to a proposed recovery
   action while the current realized state is clear.
2. **Trigger B** applies the predicate to the nominal controller proposal and
   requires an observed Final Veto rejection.

The analysis identifies which mechanism exists in frozen evidence. It does not
rank controller quality or authorize an intervention.

## Frozen Sources

The audit reads only:

- `analysis/staged_recovery_shadow_calibration_v0/`;
- `analysis/stage2a_prediction_boundary_discovery_d2_v0/`;
- `analysis/recovery_branch_state_registry_v0/`;
- `analysis/final_veto_ablation_v0/`.

The D2 manifest identity is
`d5e77b0e4d3abe6b0bc67b2efa94ed3865517543fa4fdad6ed6705d5d97ebe9a`.
All source hashes are checked before publication and again by the published
audit validator.

## Evidence Semantics

The frozen overspeed contract remains:

```text
overspeed = speed_ratio > 1.90
clear = speed_ratio <= 1.90
```

Realized ratios are derived from measured Cartesian state. Predicted ratios are
one-step predictions under a named proposed action. A predicted value is not a
realized measurement. Missing per-step Cartesian detail in compact evidence is
`not_evaluated`; it is not reconstructed.

## Trigger A

```text
realized_speed_ratio <= 1.90
and recovery_action_predicted_speed_ratio > 1.90
```

The recovery-action evidence universe includes every Stage 1B calibration
transition and every D2 zero-action recovery record. Exact replicated
action/state observations are retained in raw source totals and deduplicated in
a second total. The provisional Stage 2A action identity,
`velocity_opposed_thrust_v0`, is reported separately.

Trigger A is not the trigger frozen in the current Stage 2A runner. That runner
uses a vetoed **normal-action** prediction to request authority and then checks
the recovery proposal separately. Calling Trigger A the current runner trigger
would silently change the implemented contract.

## Trigger B

```text
nominal controller action
and nominal_action_predicted_speed_ratio > 1.90
and Final Veto decision = veto
```

Final Veto compact segments are counted using their validated `step_count`.
Those counts are logical decision observations; compact segments do not expose
every per-step Cartesian state. D2 first-veto boundaries are separately reported
and exact cross-artifact reproductions are identified to prevent double counting.

## Same-Boundary Comparison

Comparisons require exact Cartesian boundary identity. A nominal prediction and
a recovery prediction are never treated as interchangeable because their action
inputs differ. Source-native state hashes are preserved even where historical
artifacts use different canonical state-hash schemas.

## Final Veto Role

An observed veto establishes that the specified nominal proposal was not
executed. It does not establish that every future unsafe proposal will be
detected, that fallback behavior is always safe, or that the closed-loop system
is formally safe.

## Scientific Boundary

The strongest permitted conclusion is which one-step action-conditional hazard
mechanism is present in the frozen records. Zero Trigger A observations do not
prove that recovery-action predicted overspeed is physically impossible. Same-
state action comparisons do not prove controller superiority or recovery
performance.

## Authority Boundary

`Stage_2A_authority_granted = false`. No active intervention, phase transition,
action replacement, threshold tuning, or Final Veto modification is authorized
or performed by this audit.
