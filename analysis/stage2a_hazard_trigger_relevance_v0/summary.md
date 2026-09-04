# Stage 2A Hazard Trigger Relevance Audit v0

Completed: 2026-09-04

## Status

Frozen offline evidence audit completed. Stage 2A active authority remains unauthorized.

## Trigger A

The recovery-proposal predicate `realized_speed_ratio <= 1.90` and recovery-action
`predicted_speed_ratio > 1.90` occurred **0 times** across 400 frozen proposal records
(392 duplicate-aware action/state observations). The maximum recovery-action prediction
was `1.8906024003603095`. Trigger A is not
empirically supported by the audited frozen evidence; absence here is not proof of
physical impossibility outside the audited cases and actions.

## Trigger B

The Final Veto compact log contains 5 veto segments representing
**499877 logical nominal-action decisions**
above `1.90`, across five stress cases. D2 separately reproduces two first-veto
boundaries, one of which is the same first event already represented by the compact log.
The duplicate-aware combined count is **499878**
across six cases. The maximum nominal prediction was
`1.9183887199363643`.

Final Veto prevented those observed nominal proposals from being executed. This is an
operational statement about the frozen records, not a general safety proof.

## Same Boundary Evidence

Two D2 boundary states have exact Cartesian identity for nominal-versus-recovery
comparison. In both, the nominal proposal predicted above `1.90` and was vetoed, while
the published zero-action recovery prediction remained clear. At the canonical boundary,
the Stage 1B velocity-opposed and tangential-correction predictions were also clear.
These comparisons do not establish that one controller is better.

## Strongest Supported Conclusion

The hazard mechanism actually observed in frozen evidence is nominal-action one-step
predicted overspeed followed by Final Veto rejection. Recovery-action predicted
overspeed under Trigger A was not observed. The two trigger classes are not
interchangeable.

## Restrictions

No simulation, controller, trajectory, threshold tuning, Final Veto modification, or
Stage 2A authority change was performed. Unknown evidence remains `not_evaluated`.
