# Recovery Experiment Runner v0

## Status

Preflight and artifact infrastructure implemented; recovery experiment not run.

Completed: 2026-07-23

This document describes experiment preflight and artifact-publication infrastructure only. No real recovery branch trajectory, four-branch comparison, or recovery-performance experiment has been executed.

## Purpose

Recovery Experiment Runner v0 provides the last no-execution boundary before a
future one-case recovery experiment can be authorized. It checks whether the
frozen design, branch state, implementations, evaluators, stop priority, Git
state, output paths, and artifact contract are mutually consistent. It also
defines and tests a deterministic, transactional publication layer with
synthetic records.

This milestone does not expose an execution command. It cannot answer which
branch performs best, whether any branch recovers, whether the combined
monitor and branch policy is effective, or whether the system is safe.

Implementation:

- `runtime_assurance/recovery_experiment_preflight.py`
- `runtime_assurance/recovery_experiment_artifacts.py`
- `scripts/check_recovery_experiment_preflight.py`

## Frozen Sources Of Truth

The preflight derives its contract from, and does not modify:

| Contract element | Repository source |
| --- | --- |
| Experiment identity, source case, seed, branch point, branch IDs, action definitions, horizons, hazard, stop conditions, artifact names, and claim limits | `analysis/recovery_action_branching_nonformal_v0/manifest.json` |
| Canonical common state, branch step, realized nominal-prefix count, simulator/configuration hashes, and state hash | `analysis/recovery_action_branching_nonformal_v0/branch_state.json` |
| Branch action implementations and rejection behavior | `runtime_assurance/recovery_branch_executor.py` |
| Recovery Success v0 and repository-supported instability/unsafe-state evidence | `runtime_assurance/recovery_evaluators.py` |
| Runtime stop labels and priority | `runtime_assurance/recovery_stop_conditions.py` |
| Manifest and branch-state validation rules | `scripts/check_recovery_action_branching_manifest.py` and `scripts/check_recovery_branch_state.py` |

The exact canonical input hashes are pinned by preflight so a changed document
cannot pass merely by recomputing its own internal hash.

## Relationship To Existing Infrastructure

### Branch Executor v0

The executor remains the one-step branch-action and transition boundary. The
preflight imports only pure action generators to verify the frozen zero action,
`0.25` velocity-opposed magnitude, `0.25` tangential-correction magnitude, and
zero-transition explicit abort. It does not call
`execute_recovery_branch`.

### Bounded Branch Runner v0

The existing runner remains a separate infrastructure tool capped at 32
transitions. Its explicit nonformal execution mode is not reused or expanded by
this milestone. The scientific recovery experiment remains frozen at 10,000
realized post-branch transitions with a 100,000-transition total-episode limit.
The preflight validates those full-horizon values but cannot execute them.

### Recovery Evaluators v0

The preflight verifies importability and exact identity of:

- Recovery Success v0;
- the Phase34-compatible recoverability predicate;
- repository-supported instability evidence;
- repository-supported unsafe-state evidence.

It also checks that missing evidence remains `not_evaluated`, malformed evidence
remains `invalid`, and the evaluator recovery horizon remains exactly 10,000.

## Preflight Modes

The CLI supports only no-execution modes:

| Mode | Behavior |
| --- | --- |
| No mode | Prints help, returns a documented no-op status, writes nothing |
| `--plan` | Prints experiment ID, source case, branch-state hash, four branches, horizons, stop priority, and reserved outputs |
| `--validate-only` | Validates frozen documents, implementations, evaluator imports, hashes, output paths, and artifact APIs without requiring a clean tracked tree |
| `--experiment-preflight` | Adds committed-HEAD, clean tracked-tree, clean staging-area, untracked-output-collision, and publication-readiness checks |

No `--execute`, `--run`, `--run-all`, or real-publication mode exists.

Unrelated untracked workspace files do not fail preflight. An untracked file
that collides with a reserved experiment output does fail. A future real run
must begin from a clean tracked and staged implementation tree.

## Four-Branch Completeness

A structurally comparable bundle must contain exactly one record, in frozen
manifest order, for each branch:

1. `zero_action_reference_v0`
2. `velocity_opposed_thrust_v0`
3. `tangential_error_correction_v0`
4. `explicit_abort_v0`

All records must share:

- experiment and source-case identity;
- seed;
- canonical branch-state hash;
- manifest hash;
- implementation commit;
- case, simulator, and constants hashes;
- branch step and nominal-prefix transition count;
- 10,000-transition recovery horizon;
- 100,000-transition total horizon;
- strict `speed_ratio > 1.90` hazard contract;
- evaluator versions, artifact schema, and frozen stop priority.

The validator rejects missing, duplicate, reordered, or undeclared branches;
hash or horizon drift; unavailable required evaluator evidence; invalid branch
records; explicit abort with a physical transition; execution of a rejected
action; excessive transition counts; and payload hashes that do not recompute.

A complete bundle means only that the four outcomes are structurally
comparable. It does not imply hazard avoidance or recovery success.

## Stop Priority

The runtime priority remains:

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

The frozen manifest uses the design labels `realized_overspeed`,
`recovery_action_rejection`, `recovery_horizon_exhaustion`, and
`total_horizon_exhaustion`. Preflight validates those exact manifest values and
their explicit one-to-one mapping to the existing runtime labels above.

## Artifact Schemas

### Branch record

`RecoveryBranchExperimentRecord` keeps identity, provenance, execution counts,
hazard and validity evaluator statuses, crossing and recoverability events,
Recovery Success v0, simulator success, supported component costs,
intervention burden, action-rejection evidence, evaluator versions, and a
canonical result-payload hash.

The record is an experiment-specific internal contract. It does not modify or
silently promote Result Schema v1.

### Decision event

`RecoveryDecisionEvent` uses a compact deterministic JSONL contract. Each event
identifies the experiment and branch, post-branch step, proposed action, Final
Veto decision, executed action, transition occurrence, state hashes, predicted
and realized speed ratios, stop condition, evaluator statuses, terminal reason,
and evidence level.

Objects use canonical UTF-8 JSON with sorted keys and stable separators. Python
representations, memory addresses, and arbitrary undeclared fields are not
accepted.

### Bundle and publication result

`RecoveryExperimentBundle` contains the frozen-order records and deterministic
events. `RecoveryArtifactWriteResult` reports only publication status, paths,
hashes, and synthetic status. Neither type ranks branches or defines a winner.

## Reserved Artifact Bundle

The manifest reserves these future result artifacts under
`analysis/recovery_action_branching_nonformal_v0/`:

- `results.csv`
- `decision_log.jsonl`
- `summary.md`
- `comparison.png`

`manifest.json` and `branch_state.json` are immutable inputs, not newly
published result files.

The results CSV uses fixed columns and one row per branch in frozen order. The
decision log uses deterministic branch/event ordering. The summary presents
structural validity first, then hazard outcomes, state/task recovery, cost and
burden, failure mechanisms, and non-claims. The comparison plot uses the
noninteractive Agg renderer and separates hazard, recovery, and intervention
components without a combined score.

## Missing Values

- `false`: the condition was evaluated and did not occur;
- `true`: the condition was evaluated and occurred;
- `null`: a numeric or structured value is unavailable or inapplicable;
- `not_evaluated`: the evaluator did not run or required evidence was absent;
- `invalid`: evidence was malformed, contradictory, nonfinite, or drifted from
  the contract.

Unknown values are never converted to zero or `false`. Unsupported fields are
never used to rank branches.

## Deterministic Encoding

- CSV has fixed field and branch ordering, UTF-8 encoding, and `\n` newlines.
- List and tuple values use compact valid JSON.
- JSONL uses one sorted-key canonical object per line.
- Summary generation has no timestamps or environment-specific data.
- Plot dimensions, DPI, labels, branch order, colors, and metadata are fixed.
- Scientific result hashes exclude volatile timestamps.

Repeated synthetic writes are tested for byte equality in the current declared
environment.

## Atomic Publication

Publication validates a bundle in memory before writing. It then:

1. verifies the target and absence of every reserved output;
2. writes all four artifacts into a temporary sibling staging directory;
3. validates staged CSV, JSONL, Markdown, PNG, branch order, and nonempty files;
4. computes staged hashes;
5. publishes each new file with same-filesystem atomic replacement;
6. verifies final hashes;
7. rolls back every file created by that attempt if any later step fails;
8. removes the staging directory on success or failure.

The implementation provides an all-or-none final directory state. It never
overwrites or deletes a pre-existing user artifact. Synthetic bundles are
explicitly prohibited from using the frozen experiment directory.

## Protected Paths

Publication rejects all manifest-protected Phase34-37 and Final Veto locations.
Preflight also refuses a collision with any reserved recovery result path. No
writer in this milestone targets the real frozen output directory.

## Conditions Before Execution Authorization

Synthetic tests alone do not authorize execution. Before a separate execution
entry point may be added or invoked, all of these must hold from a clean,
committed implementation tree:

1. `--experiment-preflight` reports `ready=true`.
2. Repository-native Phase, Final Veto, recovery-manifest, and branch-state
   validators pass.
3. The complete bounded no-rollout test suite passes.
4. Protected evidence and frozen-input hashes match their pre-implementation
   values.
5. All four reserved result paths remain absent.
6. The execution entry point receives separate review and authorization.
7. It preserves the exact branch state, actions, evaluators, stop priority,
   horizons, and transactional publication contract.
8. The run remains a one-case nonformal diagnostic.

This document does not state that the experiment is ready merely because the
synthetic artifact layer works.

## Current Limitations

- No real branch trajectory or four-branch comparison has run.
- There is no authorized full-horizon execution entry point.
- The existing runner remains capped at 32 steps.
- Synthetic fixtures do not establish physical correctness or performance.
- Recovery margin components remain separate; no weighted score exists.
- Unsupported cost and burden fields remain null.
- Atomic all-or-none visibility across several files is approximated through
  same-filesystem atomic file replacement plus verified rollback; the final
  directory state is complete or unchanged.
- No benchmark-wide, hardware, deployment, cross-domain, or formal-safety claim
  is supported.

## Claim Restrictions

A future valid run may report branch-specific diagnostic outcomes for the one
frozen case, declared overspeed occurrence or avoidance, Recovery Success v0,
and component cost/intervention burden. It may not claim formal safety,
universal recovery, controller superiority, benchmark-wide effectiveness,
cross-case generalization, hardware validity, deployment readiness,
cross-embodiment validation, optimality of action magnitude `0.25`, or
sufficiency of the 10,000-transition horizon.
