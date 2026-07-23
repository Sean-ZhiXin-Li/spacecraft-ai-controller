# Recovery Evaluators v0

## Status

Evaluation infrastructure implemented; recovery experiment not run.

Completed: 2026-07-22

This document describes pure evaluation infrastructure only. No recovery branch comparison or recovery-performance experiment has been performed.

## Purpose

Recovery Evaluators v0 supplies deterministic, import-pure evaluators for:

- the frozen Recovery Success v0 predicate;
- the Phase34-compatible recoverable-state predicate used by recovery success;
- repository-supported instability evidence;
- repository-supported unsafe-state evidence.

The evaluators consume declared event and trajectory-summary evidence. They do
not integrate spacecraft dynamics, execute a rollout, select an action, change
a controller, or write artifacts. They do not add physical thresholds beyond
those already present in repository semantics.

Implementation:

`runtime_assurance/recovery_evaluators.py`

## Structured Result

Every evaluator returns an immutable `RecoveryEvaluationResult` containing:

| Field | Meaning |
| --- | --- |
| `evaluator_id` | Stable identity of the predicate or evidence adapter |
| `status` | One of `triggered`, `clear`, `not_evaluated`, or `invalid` |
| `triggered` | `true`, `false`, or `null` in exact agreement with `status` |
| `reason` | Deterministic reason code |
| `evidence_level` | Declared evidence class supplied to the evaluator |
| `evaluated_step` | Nonnegative step or `null` |
| `required_inputs_present` | Whether all mandatory evidence was supplied |
| `details` | Immutable, key-sorted scalar evidence suitable for canonical serialization |

Status semantics are:

| Status | `triggered` | Interpretation |
| --- | ---: | --- |
| `triggered` | `true` | Complete valid evidence satisfies the evaluator predicate |
| `clear` | `false` | Complete valid evidence does not satisfy the predicate |
| `not_evaluated` | `null` | Required instrumentation or evidence is absent |
| `invalid` | `null` | Supplied mandatory evidence is malformed, nonfinite, contradictory, or violates the frozen contract |

Unknown evidence is never converted to `false`. Both `triggered` and `clear`
require complete mandatory inputs.

## Recovery Success v0

The evaluator preserves the frozen predicate exactly:

```text
declared_hazard_avoided
AND not invalid_simulation
AND not invalid_recovery_evaluation
AND target_radius_crossing
AND phase34_compatible_recoverable_crossing
AND recovery_target_reached_within_10000_transitions
```

For the current experiment, `declared_hazard_avoided` means the declared
overspeed hazard did not occur. This is not a claim that every safety condition
was evaluated.

The evaluator also enforces:

- the recovery horizon must be exactly 10,000 realized transitions;
- a crossing is recovery evidence only at or after the branch point;
- crossing at the branch-step boundary is allowed;
- transition 10,000 is within the horizon;
- a target reached after transition 10,000 is not recovery success;
- explicit abort and recovery-action rejection are not recovery success;
- invalid simulation or invalid recovery evaluation prevents recovery success;
- missing recoverable-crossing evidence returns `not_evaluated`;
- horizon exhaustion without recovery is clear negative evidence, not malformed evidence.

Simulator-defined success is intentionally not an input to this predicate and
must remain a separate result field. No combined recovery score is created.

## Phase34-Compatible Recoverability

### Exact Source Of Truth

The source behavior is `recoverable_state` in
`scripts/explicit_controller_phase34_post_cross_sync.py`. That function uses
the constants imported from
`scripts/explicit_controller_phase21_orbital_transfer_planner.py`:

| Component | Exact condition |
| --- | --- |
| Radius error ratio | `abs(r_error_ratio) <= 2.5e-3` |
| Radial velocity ratio | `abs(vr_ratio) <= 2.0e-2` |
| Tangential velocity error ratio | `abs(vt_error_ratio) <= 2.5e-1` |

All three comparisons are inclusive and must hold simultaneously.

The pure evaluator is a narrowly scoped compatibility adapter because the
Phase34 and Phase21 scripts have import-time plotting/output configuration and
are not suitable dependencies for an import-pure runtime evaluator. Tests parse
the checked-in source predicate and constants without importing or executing
the phase scripts, then compare representative and exact-boundary behavior.
The historical scripts and artifacts are unchanged.

This predicate says only that the three declared Phase34 state components are
inside their existing bounds. It is not a control invariant, reachability
proof, or formal-safety certificate.

## Instability Evidence

Recovery Evaluators v0 supports only existing repository evidence:

- an explicit `instability` boolean field;
- controlled terminal label `instability`;
- the Final Veto normalization of legacy termination reasons `out_range`,
  `too_close`, and `radial_stall` as instability.

An explicit `false` instability field is required for a clear result. If no
explicit field or supported positive indicator exists, the evaluator returns
`not_evaluated`. A false explicit flag combined with a supported positive
indicator is contradictory and returns `invalid`.

Overspeed does not imply instability. No velocity, oscillation, mode-count, or
state-magnitude heuristic has been introduced. Future instability criteria must
be separately declared before use.

## Unsafe-State Evidence

Recovery Evaluators v0 supports only:

- an explicit `unsafe_state` boolean field; or
- controlled terminal label `unsafe_state`.

There is no independent unsafe-state threshold in the current recovery
infrastructure. Without explicit instrumentation, the result is
`not_evaluated`.

The Final Veto v0 result normalizer populated its `unsafe_state` artifact field
from the same legacy instability evidence. That is an existing artifact mapping,
not an independent physical unsafe-state criterion. This evaluator consumes an
explicit field when supplied but does not recreate or broaden that mapping.

The evaluator does not infer unsafe state from:

- overspeed;
- instability;
- invalid simulation;
- simulator failure;
- action rejection;
- explicit abort;
- failure to cross or recover.

Those outcomes remain separately reportable. A clear unsafe-state evaluator
means only that the declared simulator field is clear; it does not establish
broad safety.

## Stop-Condition Integration

`runtime_assurance/recovery_stop_conditions.py` accepts optional evaluator
results and preserves the frozen priority:

1. `invalid_simulation`
2. `invalid_recovery_evaluation`
3. `overspeed`
4. `instability`
5. `unsafe_state`
6. `action_rejected`
7. `explicit_abort`
8. `recovery_success`
9. `recovery_horizon_exhausted`
10. `total_horizon_exhausted`

An invalid evaluator result triggers `invalid_recovery_evaluation`. A
`not_evaluated` result never triggers a positive stop. If recovery-success
evidence conflicts with a higher-priority adverse condition, the adverse stop
wins and recovery success is cleared for that stop report.

The adapter remains backward-compatible: callers that supply no evaluator
results retain `not_evaluated` for recovery success, instability, and unsafe
state.

## Bounded Runner Relationship

Recovery Branch Runner v0 remains capped at 32 transitions. It does not invoke
the scientific Recovery Success v0 evaluator because it does not collect the
complete crossing/recoverability trajectory summary or run the frozen
10,000-transition horizon. It also lacks independent instability and
unsafe-state instrumentation for bounded continuation states.

The runner may continue to expose `not_evaluated` statuses in internal records.
It does not emit Result Schema v1 rows, create frozen recovery artifacts,
compare branches, or claim recovery performance.

## Missing-Evidence Rules

- Missing mandatory evidence produces `not_evaluated`.
- Malformed, nonfinite, or contradictory supplied evidence produces `invalid`.
- Conditional `crossing_step` may be null only when crossing is false.
- A recoverable crossing without a target-radius crossing is contradictory.
- Unknown multi-step evidence must remain null; it must not be guessed.
- Terminal labels and explicit fields must not silently override each other
  when they conflict.

## Current Limitations

- No recovery branch comparison has run.
- The 10,000-transition recovery execution path remains disabled.
- The bounded runner does not yet collect complete evaluator inputs.
- Instability support is limited to explicit repository fields and three
  legacy termination reasons already normalized by Final Veto.
- Unsafe-state support is limited to explicit field or controlled-label
  evidence; no physical unsafe-state envelope is defined.
- The Phase34 recoverability adapter preserves existing scalar thresholds but
  does not prove future reachability or invariance.
- No recovery Result Schema writer, decision-log writer, summary, or plot is
  implemented by this milestone.
- No formal safety, hardware, deployment, or cross-domain claim is supported.

A separate no-execution preflight and artifact layer now validates evaluator
availability, full-horizon contracts, and synthetic four-branch bundles. It
does not collect real evaluator evidence or authorize execution. A future
execution entry point still requires explicit authorization before enabling
the frozen horizon or comparing all four branches.
