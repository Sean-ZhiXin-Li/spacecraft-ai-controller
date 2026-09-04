# Stage 2A Post-Veto Alternative Proposal Audit v0

Completed: 2026-09-04

## Status

Frozen offline post-veto evidence audit implemented. No simulator, controller,
trajectory, Final Veto, threshold, or Stage 2A authority change is included.

## Purpose

This audit asks whether nominal proposals rejected by Final Veto have existing
frozen evidence for an alternative that remains at or below the strict `1.90`
one-step predicted-speed threshold. It does not choose or authorize a replacement
policy.

## Evidence Boundary

The analysis reads only frozen Final Veto compact logs, Stage 1B calibration
traces, D2 source boundaries, the branch-state registry, and their manifests.
Stage 2A-T is used as an integrity-checked source audit. D1 and D2 are not rerun.

The overspeed contract remains:

```text
overspeed = predicted_speed_ratio > 1.90
clear = predicted_speed_ratio <= 1.90
```

No new tolerance or threshold is introduced.

## Veto Event Identity

The Final Veto decision log is compact. Its five veto segments losslessly encode
logical event identities as `(case_id, step)` ranges and publish decision counts,
nominal and fallback prediction extrema, and fallback status. They do not publish
every Cartesian state or every exact action value. Missing per-event state and
action details remain `not_evaluated`.

D2 and the branch-state registry provide four exact Cartesian first-veto boundary
states. Those states support richer action-by-action comparisons without
reconstructing unlogged physics.

## Alternative Semantics

### Zero Action

`zero_action_reference_v0` is the declared Final Veto fallback. A compact veto
segment counts as safe zero-action evidence only when fallback was executed,
fallback failure is false, and the segment maximum fallback prediction is at or
below `1.90`.

### Velocity Opposed And Tangential Correction

`velocity_opposed_thrust_v0` and `tangential_error_correction_v0` are counted only
at exact veto boundary states that also occur in frozen Stage 1B traces. Evidence
at one state is not propagated to another state.

### Explicit Abort

`explicit_abort_v0` is terminal semantics, not a physical proposal. It has no
fabricated action and no predicted speed ratio. Its allow/reject status remains
`not_evaluated`. The one frozen observation has zero physical transitions.

## Safe Alternative Definition

A physical alternative is safe under this audit only when frozen evidence records:

```text
predicted_speed_ratio <= 1.90
and Final Veto status = allow
```

For the compact zero-action fallback, `executed_as_declared_fallback` plus a
segment maximum at or below `1.90` is the equivalent frozen evidence. This is a
one-step proposal classification, not a recovery or formal-safety result.

## Interpretation

Of the supplied choices, the observed behavior is better described as an
**action replacement opportunity**. More precisely, Final Veto is a
proposal-level safety barrier with an observed zero-action replacement. It is not
a terminal barrier in these runs because execution continued under fallback.

This distinction does not grant Stage 2A authority. Any future replacement must
remain subject to Final Veto and a separately reviewed authority contract.

## Claim Restrictions

The audit does not establish controller superiority, recovery improvement,
general alternative safety, formal safety, handoff readiness, or active action-
replacement authority. Sparse exact-state coverage for velocity-opposed and
tangential alternatives is reported rather than generalized.
