# Staged Recovery Guard Evidence v0

## Status

Offline guard-observability and threshold-free progress evidence implemented; no phase guard, no-progress threshold, hysteresis parameter, phase action, or staged execution authorized.

Completed: 2026-07-31

This document defines offline guard evidence, signal profiles, threshold-free directional atoms, phase observability, and no-progress window structure from existing validated artifacts. It does not execute a simulator transition, generate an action, select a phase, authorize a guard, freeze a no-progress threshold, define hysteresis, demonstrate recovery performance, or support formal safety claims.

## Purpose

Stage 1A converts the checked-in Stage 0C instrumentation-validation trace into deterministic offline evidence about signal availability and guard-atom evaluability. It separates four distinct statements:

1. a signal is observable;
2. a mathematical guard atom can be evaluated;
3. a candidate phase guard can be assembled structurally;
4. a phase-transition policy is frozen and authorized.

Only the first two can be supported by Stage 1A. Structural availability does not authorize execution.

## Evidence Basis

The measured source is `analysis/staged_recovery_instrumentation_validation_v0/`, committed by `7844cc5824cf83dc84d8732e96d361d9f4b06aeb`. Its trace-manifest canonical hash is `23dd44711641eb2fcae9f1be81f405ee0660146862e8193bb8a0ebd871140680`; its ordered trace aggregate is `4f3d700422c47abf4ece93c0dd54770be5f3109a49ef49708485c41ca67e962e`.

The source contains one initial boundary, eight measured transitions, and one zero-transition terminal record from `velocity_opposed_thrust_v0`. Historical recovery and mechanism-diagnosis summaries provide qualitative motivation only. Missing historical state is not reconstructed.

## Stage 0C Source Trace

The trace is an engineering instrumentation-validation trace, not a recovery result. Stage 1A validates every source event hash, event ordering, aggregate hash, equivalence check, and field-completeness result before deriving evidence.

The terminal record duplicates the final measured state without adding a transition. Window analysis therefore uses nine unique state samples: the initial boundary plus eight realized post-transition states.

## Analysis Boundary

The analyzer reads checked-in JSON and JSONL only. It imports no runner, simulator, controller, action generator, phase selector, or stop selector. It creates an offline derived trace; it does not create a measured runtime trace.

One eight-transition path can establish deterministic availability, local direction, and structural window coverage. It cannot estimate cross-case variability, noise, false-trigger rates, safe dwell, hysteresis, or general policy performance.

## Evidence-Status Semantics

Guard evidence statuses are `true`, `false`, `not_evaluated`, `invalid`, `unsupported`, and `policy_unresolved`. Evidence level is separate: `measured`, `derived`, `externally_supplied`, `diagnostic_proxy`, `not_evaluated`, or `invalid`.

Unknown, unsupported, invalid, and policy-unresolved evidence remains null. None is converted to false. A true atom remains observational evidence and cannot authorize a phase transition.

## Measured Signal Profiles

Profiles retain explicit event paths so pre-state, post-state, predicted, progress, action, evaluator, and phase evidence are not silently merged. Numeric profiles preserve extrema, first-to-last change, adjacent delta counts, sign changes, monotonicity, and constant status. Boolean and categorical profiles preserve counts and value changes without meaningless numeric means.

`observed_min_nonzero_adjacent_delta` is a deterministic trace-resolution statistic only. It is not a sensor-noise floor, process-noise estimate, numerical-uncertainty estimate, or recommended threshold.

## Exact Inherited Guard Atoms

The following semantics are inherited unchanged:

- realized and predicted overspeed use strict `speed_ratio > 1.90`;
- corresponding clear atoms use `speed_ratio <= 1.90`;
- Phase34-compatible radius error uses `abs(radius_error_ratio) <= 0.0025`;
- radial velocity uses `abs(radial_velocity_ratio) <= 0.02`;
- tangential error uses `abs(tangential_velocity_error_ratio) <= 0.25`;
- combined recoverability requires all three components;
- discrete crossing follows `derive_crossing_event` and requires measured previous/current states.

Predicted clear is not realized clear. Hazard clear is not task recovery.

## Threshold-Free Directional Atoms

Consecutive measured states support exact directional comparisons without a selected tolerance:

- absolute radius gap improves, is unchanged, or worsens;
- signed radius error times radial velocity identifies toward-target, away-from-target, or zero directional commitment;
- absolute tangential error improves, is unchanged, or worsens;
- overspeed headroom improves, is unchanged, or worsens;
- each recoverability component changes independently toward or away from zero;
- absolute diagnostic energy-proxy error changes independently.

No component is silently combined into a score. Direction does not establish adequate magnitude or eventual recovery.

## Predicted Versus Realized Evidence

Predicted speed ratio and headroom retain one-step predicted provenance. Realized values come only from the measured post-state. Stage 1A does not use predicted state as a measured state and does not convert prediction error into a controller-failure claim.

## Recoverability Components

Radius, radial-velocity, and tangential-error component values and pass atoms remain separate. A passing tangential component does not override failing radius or radial components. The combined predicate remains separate from crossing, Recovery Success v0, simulator success, and handoff readiness.

## Crossing Evidence

Initial and terminal events do not fabricate crossing. Transition events use existing signed-radius-error crossing semantics, without interpolation. Eligible, pre-branch-only, and no-eligible-crossing atoms remain distinct.

## Windowed Progress Evidence

Every integer window length from one through eight transitions is enumerated. Each window reports raw component changes, target-directed radial commitment count, crossing count, and evidence availability. Ambiguous raw signed changes remain descriptive rather than being assigned an unsupported target direction.

No preferred window, combined score, or `stalled`/`progressing`/`regressing` policy result is defined.

## Unresolved No-Progress Parameters

The unresolved parameters are:

- `NO_PROGRESS_WINDOW_LENGTH`
- `NO_PROGRESS_MIN_RADIUS_GAP_IMPROVEMENT`
- `NO_PROGRESS_MIN_RADIAL_COMPONENT_IMPROVEMENT`
- `NO_PROGRESS_MIN_TANGENTIAL_COMPONENT_IMPROVEMENT`
- `NO_PROGRESS_MIN_HEADROOM_IMPROVEMENT`
- `NO_PROGRESS_REQUIRED_COMPONENT_COUNT`
- `NO_PROGRESS_CONSECUTIVE_WINDOWS`
- `NO_PROGRESS_MIN_PHASE_DWELL`
- `NO_PROGRESS_COOLDOWN`

No numerical value is selected. Observed ranges are not recommendations.

## Phase Observability

All nine architecture phases receive separate entry, stay, and exit evidence inventories. Each row reports Stage 0A schema support, Stage 0C measured availability, previous/predicted-state dependencies, phase-runtime dependencies, future-evaluator dependencies, unsupported dependencies, unresolved parameters, and action-law status.

No existing branch action is relabeled as a staged phase action. Every candidate guard remains `not_authorized`.

## Hazard-Arrest Evidence

Realized and predicted speed ratios, signed headroom, Final Veto decision, action disposition, and measured state validity are available. Consecutive-clear count, exit dwell, hysteresis, and a hazard-arrest action law remain unresolved.

## Stabilization Evidence

Hazard status, radial velocity, tangential error, and short-window trends are observable. Absence of overspeed is not stabilization. Instability, unsafe-state, minimum dwell, and oscillation criteria require future evidence or evaluators.

## Radial-Recommitment Evidence

Radius gap, signed error, radial velocity, radial ratio, target direction, and local trend are observable. Required radial commitment magnitude, correction authority, phase dwell, exit guard, and action law remain unresolved.

## Tangential-Alignment Evidence

Tangential velocity error, error ratio, pass status, and local trend are observable. Entry/exit hysteresis, allowed radial degradation, dwell, saturation policy, and phase action remain unresolved.

## Crossing-Preparation Evidence

Measured radius gap, radial direction, recoverability components, and discrete crossing evidence are available. Reliable future-crossing prediction, crossing proximity, and correction authority are unavailable. No time-to-crossing estimate is invented.

## Recoverability-Verification Evidence

Existing component values, inclusive pass atoms, discrete crossing eligibility, and combined Phase34-compatible evidence are observable. Consecutive verification, nominal-controller compatibility, and handoff readiness remain unresolved.

## Nominal-Handoff Limitations

Phase34-compatible recoverability is not handoff readiness. The repository has no handoff-readiness evaluator or nominal-controller acceptance evidence for this trace. Nominal handoff remains unauthorized.

## Retreat and Abort Limitations

Retreat authority, target, action, and success evaluator are unavailable. Explicit abort can be observed only when externally supplied; Stage 1A creates no autonomous abort policy. Neither retreat nor abort is Recovery Success v0.

## Current Unsupported Evidence

Available correction authority remains explicitly unsupported. Runtime phase identity, phase dwell, phase history, transition reason, no-progress policy output, handoff readiness, instability, and unsafe-state evidence remain unavailable in this trace as declared by Stage 0C.

## Next Evidence Requirement

The smallest next milestone is a predeclared Stage 1B hazard-arrest/stabilization observational trace set across repeated boundary conditions. It should measure variability and evaluator availability without implementing phase actions or authorizing transitions.

## Claim Restrictions

Stage 1A does not establish recovery performance, phase-policy validity, guard false-positive or false-negative rates, general signal noise, safe hysteresis, controller superiority, formal safety, hardware validity, or deployment readiness. Staged recovery execution remains `not_authorized`.
