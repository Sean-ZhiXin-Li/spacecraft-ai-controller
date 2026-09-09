# Stage 2A Recovery Objective Definition Audit v0

Completed: 2026-09-09

## Status

Frozen offline objective-semantics audit completed. Physical executions: 0. Controller
executions: 0. Stage 2A authority remains unauthorized.

## Safety Constraints

Evidence validity is a prerequisite. Proposed-action safety is defined by the existing
Final Veto semantics: predicted speed ratio above `1.90` is overspeed, while a proposal at
or below `1.90` is clear under this one hazard predicate. Realized overspeed remains a
separate measured-state hazard. Signed headroom is a safety margin, not a mission score.

## Recovery Progress

Recovery progress is component-wise: decreasing absolute radius gap, target-directed
radial motion together with decreasing absolute radial-velocity ratio, decreasing
absolute tangential error, and increasing overspeed headroom. The exact Phase34-compatible
component predicate and eligible target-radius crossing are milestones, not continuous
scores. The specific-energy quantity remains a diagnostic proxy only.

## Conflicts

The evidence contains real objective conflicts. Closing radius gap can coexist with a
worsening radial arrival component. Lower predicted speed ratio can trade against radius
or tangential progress. Directional progress can occur without the combined recoverability
predicate or eligible crossing. These conflicts are preserved rather than weighted away.

## Future Evaluators

Future recovery success, handoff readiness, correction authority, crossing prediction,
active-intervention release readiness, action cost, progress persistence, causal action
effect, and general cross-case ordering remain `not_evaluated` or unsupported.

## Recommended Semantics

For a future separately authorized design, the most defensible structure is
**lexicographic with safety first**: validate evidence, enforce Final Veto feasibility,
preserve recoverability and crossing milestones, then compare component-wise progress.
This is a conceptual semantics recommendation, not a policy. It defines no combined
score, weights, optimizer, tie-break, action selector, or active authority.

## Claim Restrictions

This audit does not demonstrate recovery success, select an action, establish an optimal
metric ordering, validate active thresholds, prove formal safety, or authorize Stage 2A.
