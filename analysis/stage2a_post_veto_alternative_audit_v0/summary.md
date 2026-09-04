# Stage 2A Post-Veto Alternative Proposal Audit v0

Completed: 2026-09-04

## Status

Frozen offline evidence audit completed. Physical executions: 0. Stage 2A authority remains unauthorized.

## Veto Universe

The frozen Final Veto compact log represents 499,877 logical nominal-proposal
rejections. D2 records two exact first-veto boundaries, one reproducing a compact-log
event and one additional angle-155 event. The duplicate-aware universe therefore
contains **499878 veto events across six cases**.

Compact event identity is recoverable as `(case_id, step)`, but per-event Cartesian state
identity is not published and remains `not_evaluated`. Four exact boundary states are
available from D2 and the branch-state registry.

## Safe Alternatives

All **499878** duplicate-aware veto
events have at least one safe alternative in frozen evidence: `zero_action_reference_v0`.
The five Final Veto segments executed that fallback with zero recorded fallback failures,
and their maximum predicted fallback ratio was
`1.8906024003603095`.

`velocity_opposed_thrust_v0` and `tangential_error_correction_v0` were each evaluated and
allowed at three exact veto boundary states. Every available prediction for those actions
was at or below `1.90`, but the other 499875
veto events remain `not_evaluated` for each action. This sparse coverage does not support
a general safety claim.

`explicit_abort_v0` is not a physical alternative proposal. Frozen evidence contains one
terminal-only, zero-transition observation at the canonical boundary; predicted speed and
allow/reject status remain `not_evaluated`.

## Interpretation

Of the two supplied choices, the frozen behavior is better interpreted as an **action
replacement opportunity**, with an important qualification: Final Veto is the
proposal-level safety barrier and the observed replacement is specifically zero action.
It is not a terminal barrier because the vetoed runs continued under fallback.

No evidence here authorizes active Stage 2A replacement, establishes controller
superiority, proves recovery improvement, or proves formal safety.
