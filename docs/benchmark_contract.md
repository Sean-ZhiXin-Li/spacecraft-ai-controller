# Phase36B Benchmark Contract

## Purpose

This document defines the stable benchmark contract for Phase36B. It exists to prevent future phases from drifting into incomparable scorecards or uncontrolled local tuning.

Phase36B is a 2D sandbox benchmark. It is not real spacecraft validation.

## Benchmark Scope

Phase36B must use the same 24-case reduced benchmark used for Phase34 and Phase35:

- `r0_over_target`: `0.98`, `1.00`, `1.02`
- `initial_velocity_angle_deg`: `150`, `165`, `170`, `175`
- `thrust_scale`: `8000`, `10000`

The benchmark unit is one transfer family evaluated on all 24 cases.

Phase36A representative-case results may be used for interpretation only. They must not be reported as full benchmark evidence.

## Fixed Terminal Controller

The terminal/post-cross controller is fixed:

- Phase34 `radius_priority`
- post-cross synchronization after first target-radius crossing
- unchanged physics
- unchanged CAPTURE/LOCK thresholds
- unchanged recoverability thresholds
- unchanged overspeed and instability checks

Phase36B may change only the upstream transfer-family behavior before first crossing.

## Required Families

Phase36B must include:

- `baseline_phase34`
- `spiral_approach`
- `grazing_corridor`
- `redesigned_delayed_crossing`

The baseline is required for comparison. The other three are the only Phase36A-derived candidates that currently justify full-benchmark testing.

## Families Excluded Unless Redesigned

Do not include these families without redesign:

- `energy_bleed_then_cross`
- `overshoot_return`
- `two_stage_transfer`

Reason: in Phase36A, each produced `0 / 3` crossings and `3 / 3` overspeed on the representative subset. The labels may remain conceptually interesting, but the current implementations are not ready for full benchmark comparison.

## Primary Metrics

Primary benchmark metrics:

- `crossing_count`
- `phase34_compatible_crossing_count`
- `recoverable_crossing_count`
- `overspeed_count`
- `instability_count`

These metrics should be reported per family across all 24 cases.

## Crossing-State Quality Metrics

For every crossing case, record:

- `crossing_vr_ratio`
- `crossing_vt_error_ratio`
- `crossing_sync_error`
- `best_post_cross_distance`
- `min_abs_radius_error`

These metrics determine whether a family creates a useful handoff state, not just a geometric crossing.

## Diagnostic Fields

Each result row should include:

- `transfer_family`
- `family_qualitative_label`
- `r0_over_target`
- `initial_velocity_angle_deg`
- `thrust_scale`
- `crossing_occurs`
- `first_crossing_step`
- `phase34_compatible_crossing`
- `recoverable_crossing`
- `dominant_failure_label`
- `overspeed`
- `instability`
- `termination_reason`
- `representative_subset_note`

`representative_subset_note` is optional. It is needed only when a row comes from a representative subset or visualization pass. Full 24-case Phase36B rows may leave it absent or empty.

## Terminology Rules

Use these definitions consistently:

- `crossing`: a target-radius crossing, a geometric event.
- `recoverable crossing`: a crossing followed by entry into the recoverability basin during post-cross evaluation.
- `Phase34-compatible crossing`: in the current Phase36B implementation, this field means that a geometric crossing occurred without overspeed or instability under unchanged Phase34 handoff assumptions. It should not be overread as formal proof of robust terminal-basin membership.
- `CAPTURE` and `LOCK`: simulator state-machine labels, not real flight-validation states.
- `success`: use only as `simulator-defined success label`, never as mission success.

In the current Phase36B data, all `8 / 24` Phase34-compatible crossings per family were also recoverable crossings. The aggregate claims are therefore not inflated by the implementation-level compatibility field. Future versions may tighten this field by requiring an explicit Phase34 recoverability-distance or handoff-quality threshold.

## Reporting Rules

Phase36B reports must state:

- whether crossing count improved over `baseline_phase34`
- whether recoverable crossing count improved over `baseline_phase34`
- whether any new crossings were Phase34-compatible
- whether any family increased overspeed or instability
- whether the result supports or weakens the transfer-family geometry hypothesis

If no family improves crossing count, the correct conclusion is:

Phase36B narrowed the transfer-family hypothesis space but did not expand the crossing basin under the tested family set.
