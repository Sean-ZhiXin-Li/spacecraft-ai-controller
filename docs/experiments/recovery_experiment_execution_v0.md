# Recovery Experiment Execution v0

## Status

Frozen one-case four-branch nonformal experiment executed and published.

Implementation completed: 2026-07-24

Executed: 2026-07-24

This experiment is a one-case nonformal recovery diagnostic. Its results do not establish universal recovery, optimal control, formal safety, hardware validity, or cross-domain effectiveness.

## Purpose

Recovery Experiment Execution v0 is the authorized execution boundary for the
frozen `recovery_action_branching_nonformal_v0` design. It continues the same
canonical branch state through all four predeclared recovery branches, records
each branch independently, validates four-branch comparability in memory, and
publishes a complete artifact bundle transactionally.

The implementation does not add an adaptive recovery policy, tune an action,
rank branches, or change the existing simulator transition. Scientific values
come from the frozen manifest and canonical branch-state artifact rather than
CLI overrides.

## Frozen Inputs

| Input | Frozen source |
| --- | --- |
| Experiment manifest | `analysis/recovery_action_branching_nonformal_v0/manifest.json` |
| Canonical branch state | `analysis/recovery_action_branching_nonformal_v0/branch_state.json` |
| Source case | `phase35_radial_energy_push_overspeed_stress_v0__r0_0p98__angle_150__thrust_8000` |
| Seed | `0` |
| Branch point | Step 28, before nominal action or Final Veto fallback execution |
| Nominal prefix | 27 realized transitions |
| Hazard | Realized or predicted `speed_ratio > 1.90` |
| Transition source | `simulator.phase34_35_transition.step_phase34_35_transition` through Recovery Branch Executor v0 |
| Recovery evaluator | `recovery_success_v0` |

The frozen manifest and branch state are immutable experiment inputs. Their
design-era implementation flags are historical freeze metadata and are not
rewritten when execution infrastructure is added.

## Branch Order

The runner executes exactly this manifest order:

1. `zero_action_reference_v0`
2. `velocity_opposed_thrust_v0`
3. `tangential_error_correction_v0`
4. `explicit_abort_v0`

The three physical branches use repeated one-step branch actions. The action is
recomputed from the current state where its declared formula requires it, then
evaluated by the unchanged Final Veto before every possible transition.
Rejected recovery actions terminate their branch without executing the rejected
action, substituting zero action, or selecting another branch. Explicit abort
produces one terminal decision event and zero physical transitions.

## Branch Independence

Before each branch, the runner reloads `branch_state.json`, recomputes and checks
its canonical hash, creates a fresh continuation state, and discards that state
when the branch ends. No mutable simulator state is shared between branches.

Every result row must retain the same:

- branch-state, manifest, case, simulator-configuration, and constants hashes;
- source case and seed;
- branch step and nominal-prefix count;
- implementation commit;
- hazard threshold and comparator;
- horizons and stop-priority version.

Only the selected predeclared branch behavior differs.

## Full Horizons

The scientific recovery horizon is exactly 10,000 realized post-branch
transitions. The total episode horizon is exactly 100,000 realized transitions,
including the 27-transition nominal prefix. A rejected action is not a realized
transition. Recovery transition 10,000 is evaluated and remains within the
recovery horizon; no transition 10,001 is permitted.

This full-horizon runner is separate from
`scripts/run_recovery_branch_runner.py`, which remains infrastructure-only and
capped at 32 steps.

## Evaluation Semantics

Recovery Success v0 requires all of:

```text
declared overspeed hazard avoided
AND valid simulation
AND valid recovery evaluation
AND target-radius crossing at or after the branch point
AND Phase34-compatible recoverable crossing
AND recovery target reached within 10000 realized recovery transitions
```

The Phase34-compatible component limits remain inclusive:

- absolute radius-error ratio `<= 0.0025`;
- absolute radial-velocity ratio `<= 0.02`;
- absolute tangential-velocity-error ratio `<= 0.25`.

Crossing, recoverable crossing, Recovery Success v0, strict simulator-defined
success, and overspeed are reported separately. No combined recovery score is
created. Strict simulator success retains the Phase21/34/35 persistence rule
but is not itself a recovery stop condition.

Existing repository instability evidence remains limited to `out_range`,
`too_close`, and persistent `radial_stall`. The scoped `unsafe_state` field
preserves the existing Final Veto artifact mapping from that legacy instability
evidence; it is not a new physical safety envelope.

## Stop Priority

At every realized transition or terminal decision boundary, the runner uses:

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

Higher-priority adverse evidence cannot be replaced by simultaneous recovery
evidence. A `not_evaluated` status never becomes positive evidence.

## Execution Procedure

The only real mode is:

```powershell
python scripts/run_recovery_experiment_v0.py --execute-frozen-experiment
```

The CLI exposes no branch, seed, threshold, action-magnitude, horizon, retry,
resume, or branch-selection override. Before execution it requires a committed
HEAD, a clean tracked tree and staging area, matching frozen hashes, no reserved
output collision, complete evaluator availability, and `READY = true` from the
experiment preflight.

Default invocation, `--plan`, `--validate-only`, and
`--experiment-preflight` execute no recovery transition.

## Artifact Contract

The immutable inputs remain in place. A successful run publishes exactly:

- `analysis/recovery_action_branching_nonformal_v0/results.csv`
- `analysis/recovery_action_branching_nonformal_v0/decision_log.jsonl`
- `analysis/recovery_action_branching_nonformal_v0/summary.md`
- `analysis/recovery_action_branching_nonformal_v0/comparison.png`

The CSV has one row per branch in frozen order. JSONL events are ordered by
branch, recovery boundary, and event index. The generated summary reports
structural validity before outcomes and retains failures and missing values.
The deterministic plot separates hazard outcome, crossing/recoverability/task
recovery, and intervention burden. No artifact declares an unsupported winner.

Unsupported quantities remain `null`; unevaluated and invalid evidence remain
distinct from evaluated `false` outcomes.

## Atomic Publication

All branches complete in memory before publication. The artifact layer then:

1. validates the four-branch bundle;
2. writes every reserved output to a task-owned sibling staging directory;
3. validates staged CSV, JSONL, Markdown, and PNG files;
4. computes artifact hashes;
5. moves the complete set into the frozen output directory;
6. verifies final hashes against staged hashes.

Existing artifacts are never overwritten. Any execution, validation, writer,
or publication failure publishes no complete bundle and removes only the
task-owned staging directory.

## Failure And No-Retry Policy

Scientific failures such as overspeed, instability, action rejection, explicit
abort, or horizon exhaustion remain valid measured outcomes and are not rerun.

Contract drift, dirty tracked state, hash mismatch, implementation exception,
incomplete records, writer failure, or publication failure aborts publication.
The runner has no automatic retry or resume path. Another real attempt after an
infrastructure failure requires a separately reviewed implementation commit and
explicit authorization.

## Claim Limits

Any published evidence applies only to the one frozen source case and the four
predeclared branches. It may describe branch-specific hazard, recovery, cost,
and intervention outcomes for that case. It may not establish formal safety,
universal recovery, controller superiority, benchmark-wide effectiveness,
cross-case generalization, action optimality, hardware validity, deployment
readiness, or cross-embodiment validation.

## Measured Artifacts

Measured artifacts are pending execution. After successful atomic publication,
the status section will record the execution date and the paths above will be
the generated sources of measured outcomes.
