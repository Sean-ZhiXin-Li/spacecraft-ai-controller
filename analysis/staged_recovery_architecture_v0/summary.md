# Staged Recovery Architecture v0 Summary

## Status

Published recovery evidence translated into a staged architecture contract; no new trajectory executed.

Completed: 2026-07-26

## Evidence Translation

The frozen one-case recovery experiment established that immediate hazard avoidance and task recovery are different outcomes. All three physical branches avoided realized overspeed for 10,000 transitions, while none crossed the target radius, reached Phase34-compatible recoverability, achieved Recovery Success v0, or reached simulator success. Final Veto allowed all 30,000 physical proposals, so the common failure was not repeated post-branch veto intervention.

The mechanism diagnosis motivates each staged responsibility:

- `hazard_arrest`: preserve the demonstrated ability to suppress immediate hazard exposure without treating it as recovery;
- `stabilization_assessment`: determine whether a hazard-arrested state is valid and usable rather than assuming no overspeed means stable;
- `radial_recommitment`: monitor and restore target-directed radial progress missing from all tested outcomes;
- `tangential_alignment`: improve tangential geometry without erasing radial progress or treating one improved component as recovery;
- `crossing_preparation`: coordinate radial and tangential conditions before a crossing attempt;
- `recoverability_verification`: apply the existing crossing, Phase34, and Recovery Success v0 predicates separately;
- `nominal_handoff`: require complete valid recovery evidence before returning authority;
- `retreat`: represent a future lower-risk objective distinct from task recovery;
- `explicit_abort`: retain zero-transition termination as distinct from retreat, unsafe state, or recovery.

## Architecture Graph

The graph begins at `hazard_arrest`, advances through assessment and component-specific recovery phases, and permits controlled reassessment, retreat, or abort. Only `recoverability_verification` may reach `nominal_handoff`. `nominal_handoff` and `explicit_abort` are terminal. The validator forbids direct handoff from arrest, assessment, radial recommitment, or tangential alignment.

All 26 graph edges are architecture-only. No edge is executable in v0.

## Existing Semantics Preserved

- Phase34-compatible recovery uses inclusive absolute bounds: radius error ratio `<= 0.0025`, radial velocity ratio `<= 0.02`, and tangential velocity error ratio `<= 0.25`.
- Adverse stops override phase progression in the frozen order: invalid simulation, invalid recovery evaluation, overspeed, instability, unsafe state, action rejection, explicit abort, recovery success, recovery-horizon exhaustion, total-horizon exhaustion.
- Unknown numeric and boolean evidence remains `not_evaluated` or `invalid`; it is never coerced to zero or false.
- Hazard avoidance, crossing, recoverability, Recovery Success v0, and simulator success remain separate.

## Unresolved Boundaries

Execution remains unauthorized. The manifest lists unresolved numerical definitions for arrest/stabilization guards, radial and tangential phase boundaries, crossing proximity, retreat semantics, no-progress windows and meaningful-improvement thresholds, dwell limits, cooldown, evidence counts, finite switching budget, and repeated-cycle detection.

No phase action law is frozen. In particular, the previous zero-action, velocity-opposed, and magnitude-0.25 tangential branch laws are evidence about failure modes, not staged phase controllers.

## Missing Instrumentation

The published decision log was sufficient for outcome validation but did not preserve per-step Cartesian state, radius, target-radius error, radial velocity, tangential velocity, orbital energy, or recoverability-component margins. Staged execution therefore requires validated per-step physical state, progress, phase, action, Final Veto, evidence-level, dwell, and transition-reason instrumentation before any rollout.

## Progress And Chatter Controls

The no-progress contract requires one named signal, desired direction, observation window, minimum meaningful improvement, phase-specific dwell limit, and explicit missing/invalid handling. Timeout or one flat sample cannot establish a stall, and no-progress cannot establish recovery.

The hysteresis contract requires minimum dwell, distinct justified entry/exit thresholds, cooldown, consecutive evidence, a finite phase-transition budget, repeated-cycle detection, new evidence for repeated transitions, and raw reason preservation. Numerical values remain unresolved.

## Handoff, Retreat, And Abort

Nominal handoff requires valid simulation and evaluation, no active adverse stop, target-radius crossing, Phase34-compatible recoverability, Recovery Success v0, explicit handoff readiness, and provenance. Crossing, no overspeed, simulator success, or tangential improvement alone cannot authorize it.

Retreat is a future physical objective whose target and success predicate are not specified. Explicit abort is a terminal zero-transition decision. Neither is Recovery Success v0.

## Next Smallest Milestone

Implement and validate instrumentation completeness only: status-bearing per-step Cartesian and orbital-component signals, phase identity and transition reason, dwell and progress evidence, proposed/executed action, and Final Veto decision. This must be proven behavior-preserving without implementing a staged controller or running a recovery comparison.

## Scoped Conclusion

The published one-case evidence supports separating hazard arrest from task recovery and motivates state-dependent recovery phases with explicit radial, tangential, energy, progress, and recoverability monitoring. The evidence does not determine the optimal phase actions, numerical switching thresholds, or whether the proposed staged architecture will recover the frozen state.

This contract establishes no controller implementation, execution readiness, task recovery, phase optimality, formal safety, hardware validity, benchmark-wide effectiveness, cross-domain validation, or deployment claim.
