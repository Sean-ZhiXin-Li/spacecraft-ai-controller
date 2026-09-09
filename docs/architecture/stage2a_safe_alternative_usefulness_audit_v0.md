# Stage 2A Safe Alternative Usefulness Audit v0

Completed: 2026-09-09

## Status

Frozen offline evidence audit implemented. No simulator or controller execution occurred,
and Stage 2A authority remains unauthorized.

## Purpose

This audit asks whether safe physical alternatives observed after nominal Final Veto
rejection also provide one-step orbital evidence relevant to mission progress. It keeps
four meanings separate:

- **Safe:** the action prediction is at or below the frozen `1.90` threshold and Final
  Veto allows the proposal.
- **Progress:** a named one-step component changes in its target direction.
- **Potentially recoverable:** the existing Phase34-compatible component predicate is
  observed at the resulting state. This is not a forecast of future recovery.
- **Unknown:** the exact-state field is absent or the evidence cannot support the claim.

## Frozen Evidence

The audit reads only the published Stage 2A-T trigger audit, Stage 2A-H post-veto
alternative audit, Stage 1B measured calibration traces, D2 discovery artifacts,
branch-state registry, and Final Veto evidence. It does not run their generators.

Stage 2A-H exposes four exact nominal-veto states. Three match Stage 1B one-step records
for all three physical alternatives. The fourth, angle 155, contains only zero-action
safety evidence; usefulness fields remain `not_evaluated`.

## Alternative Evidence

The alternatives are:

- `zero_action_reference_v0`
- `velocity_opposed_thrust_v0`
- `tangential_error_correction_v0`

For each exact match the audit records predicted speed ratio, Final Veto status, absolute
radius-gap change, radial target direction and component change, absolute tangential
error change, recoverability components, diagnostic specific-energy-proxy error, and
crossing evidence.

The energy quantity remains a diagnostic proxy. It is not treated as an exact conserved
invariant or an active decision criterion.

## Progress Semantics

Progress is component-wise and threshold-free:

- radius gap improves when absolute target-radius error decreases;
- the radial component improves when absolute radial-velocity ratio decreases;
- tangential error improves when its absolute value decreases;
- overspeed headroom improves when it increases;
- diagnostic energy-proxy error improves when its absolute value decreases.

No scalar progress score, weighting, preferred action, or utility threshold is created.
One-step ballistic improvement under zero action is observational evidence, not proof that
zero action causally improves recovery.

## Safety And Usefulness

Safety is a necessary proposal-level gate. It is not a sufficient action-selection
objective. The frozen evidence shows component tradeoffs: lower predicted speed ratio
can accompany stronger radial progress, while a higher but still safe ratio can accompany
stronger radius-gap or tangential-error progress.

A future separately reviewed replacement selector should therefore apply Final Veto
safety first and then evaluate explicitly declared usefulness components. This document
does not define that selector or grant it authority.

## Correlation Boundary

Descriptive correlations use nine action observations from only three exact initial
states. The observations are not nine independent physical cases. The reported
coefficients are audit summaries, not estimates of general correlation, controller
quality, or recovery probability.

## Authority Boundary

The audit performs zero physical executions and zero controller executions. It does not
modify Final Veto, threshold `1.90`, actions, Stage 2A authority, or any source artifact.
Unknown values remain `not_evaluated`.

## Claim Restrictions

The audit does not demonstrate recovery success, identify an optimal controller, select
a best action, establish a universal replacement policy, validate active thresholds,
prove formal safety, or support deployment readiness.
