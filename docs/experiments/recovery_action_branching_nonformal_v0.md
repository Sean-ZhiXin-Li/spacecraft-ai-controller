# Recovery Action Branching Nonformal v0

## Status

Experiment design frozen; not run.

Completed: 2026-07-18

Experiment ID: `recovery_action_branching_nonformal_v0`

Formal status: `is_formal_experiment=false`

This document specifies one common-state, four-branch recovery diagnostic. It does not implement a runner or branch policy, execute a simulation, select a winning branch, or add recovery evidence.

## Purpose

Final Veto Overspeed Ablation v0 showed that a one-step veto can avoid the declared overspeed hazard while preserving known recoverable cases. Its diagnostic stress trajectories did not recover the task: the zero-action fallback was repeatedly selected and all five monitor-on stress runs reached `max_steps` without crossing.

Recovery Metrics v0 therefore asks a narrower next question:

> From one identical pre-execution veto state, do four explicitly different responses produce distinguishable hazard, recovery, burden, and cost outcomes?

The experiment compares:

1. repeated zero action;
2. bounded velocity-opposed thrust;
3. bounded tangential-error correction;
4. explicit abort.

The comparison is diagnostic. No branch is expected to win in advance.

## Evidence Boundary

Read-only design sources are:

- `docs/theory/recovery_metrics_v0.md`;
- `analysis/final_veto_ablation_v0/manifest.json`;
- `analysis/final_veto_ablation_v0/results.csv`;
- `analysis/final_veto_ablation_v0/paired_results.csv`;
- `analysis/final_veto_ablation_v0/decision_log.jsonl`;
- `analysis/final_veto_ablation_v0/summary.md`;
- `analysis/final_veto_ablation_v0/comparison.png`.

The Final Veto artifact directory is frozen and protected. It supplies case-selection and design context only. This manifest contains no measured outcomes, result rows, branch ranking, or positive recovery claim.

## Frozen Source Case

Use exactly one case:

| Field | Frozen value |
| --- | --- |
| `case_id` | `phase35_radial_energy_push_overspeed_stress_v0__r0_0p98__angle_150__thrust_8000` |
| `subset_id` | `phase35_radial_energy_push_overspeed_stress_v0` |
| `r0_over_target` | `0.98` |
| `initial_velocity_angle_deg` | `150` |
| `thrust_scale` | `8000` |
| `seed` | `0` |
| Nominal controller | Phase35 `radial_energy_push` |
| Post-cross context | Phase34 `radius_priority` |

### Why This Case Was Selected

This is the first case in the frozen diagnostic stress manifest's stable order. Selection therefore does not compare the five formal stress outcomes and choose the most favorable case for a recovery action. The case is known to exercise the original Final Veto hazard context, but the frozen outcome must not be treated as a result for any new recovery branch.

## Branch Point Contract

The branch point is:

> The first valid monitor evaluation where the nominal proposed action has predicted one-step `speed_ratio > 1.90`, before either the nominal action or the Final Veto zero-action fallback is executed.

The strict comparator remains `>`. A prediction exactly equal to `1.90` is not this branch point.

The canonical branch-state object must record:

- `step`;
- state vector ordered as `(x, y, vx, vy)`;
- position `(x, y)`;
- velocity `(vx, vy)`;
- phase;
- active stage;
- nominal proposed action `(action_x, action_y)`;
- predicted nominal next state;
- predicted nominal speed ratio;
- hazard threshold and comparator;
- monitor decision;
- implementation commit;
- simulator constants hash;
- case configuration hash;
- canonical branch-state hash.

All numerical state, action, and prediction values must be finite. The monitor evaluation must be valid and its decision must be `veto`.

### Canonical Hashing

The hash input is the complete branch-state object excluding its own `canonical_branch_state_hash` field. Serialize it as canonical UTF-8 JSON with sorted keys, separators `,` and `:`, and `allow_nan=false`. Hash those exact bytes with SHA-256.

Every branch must receive a byte-equivalent canonical input object and the identical canonical branch-state hash. A runner must reject the comparison if any branch-state byte or shared configuration hash differs.

## Action Coordinate Convention

The action frame is the same 2D inertial Cartesian frame used by the active Phase35 rollout context.

```text
r = (x, y)
v = (vx, vy)
r_norm = sqrt(x*x + y*y)
v_norm = sqrt(vx*vx + vy*vy)
e_r = r / r_norm
e_t = (-e_r_y, e_r_x)
v_t = dot(v, e_t)
tangential_error = v_t - target_circular_speed
```

`e_t` is the positive counterclockwise 90-degree rotation of `e_r`. The target circular speed must come from the same active rollout context used by Final Veto prediction; it must not be recomputed with a second physics definition.

The signed tangential-error convention is:

```text
sign = +1 when tangential_error > 1e-12 m/s
sign = -1 when tangential_error < -1e-12 m/s
sign = 0 otherwise
```

The velocity-zero tolerance is also `1e-12 m/s`.

Treat the evaluation as invalid when:

- position norm is zero;
- position or velocity contains a nonfinite value;
- a derived unit vector is nonfinite;
- target circular speed is nonfinite;
- a proposed action contains a nonfinite value.

Invalid geometry must not be converted into zero action, allow, veto success, or hazard avoidance.

## Frozen Branches

### 1. `zero_action_reference_v0`

Propose:

```text
u = (0.0, 0.0)
```

Apply the existing component clipping, record pre-clip and post-clip actions, and evaluate the resulting candidate through the unchanged Final Veto monitor. If allowed, execute exactly one transition and re-evaluate the same branch policy. If rejected, apply the recovery-action rejection contract.

This is a reference branch, not a safe-action claim.

### 2. `velocity_opposed_thrust_v0`

Using inertial Cartesian velocity, propose:

```text
u = -0.25 * v / v_norm
```

If `v_norm <= 1e-12 m/s`, propose `(0.0, 0.0)`.

Rules:

- pre-clip action magnitude is exactly `0.25` when velocity is nonzero;
- record pre-clip and post-clip actions;
- use only the existing component clipping;
- pass the post-clip action through unchanged Final Veto evaluation;
- execute it only on `allow`;
- execute one realized transition, then re-evaluate the branch policy.

This is heuristic velocity-opposed thrust. It is not called proven braking and has no assumed recovery property.

### 3. `tangential_error_correction_v0`

Define:

```text
tangential_error = v_t - target_circular_speed
u = -0.25 * sign(tangential_error) * e_t
```

When `abs(tangential_error) <= 1e-12 m/s`, propose `(0.0, 0.0)`.

Map explicitly to Cartesian components:

```text
u_x = -0.25 * sign(tangential_error) * e_t_x
u_y = -0.25 * sign(tangential_error) * e_t_y
```

Rules:

- nonzero pre-clip action magnitude is exactly `0.25`;
- record radial and tangential action decomposition;
- add no radial correction in v0;
- record pre-clip and post-clip actions;
- pass the post-clip action through unchanged Final Veto evaluation;
- execute it only on `allow`;
- execute one realized transition, then re-evaluate the branch policy.

### 4. `explicit_abort_v0`

At the branch point:

- execute no further transition;
- emit a terminal decision event;
- set experiment-specific branch terminal label `explicit_recovery_abort`;
- execute no fallback action;
- set task recovery to false;
- preserve simulator success as false unless the existing pre-branch simulator state independently already defines success;
- assign `hazard_avoided_through_termination` only when no declared hazard occurred before termination.

Explicit abort is included to expose the difference between refusing further exposure and recovering the task. It also provides a zero-post-branch-transition operational baseline.

## Recovery Action Rejection

If unchanged Final Veto evaluation rejects a proposed recovery action:

1. Record `recovery_action_rejected`.
2. Preserve the pre-clip and post-clip proposed actions.
3. Preserve the predicted recovery next state and speed ratio.
4. Preserve threshold, comparator, monitor decision, and reason.
5. Do not execute the rejected action.
6. Do not substitute zero action.
7. Do not recursively select another branch.
8. Terminate only the affected branch.

This rule prevents hidden fallback behavior from contaminating the four-branch comparison.

## Horizon And Step Counting

Freeze:

- total episode horizon: 100000 realized transitions;
- recovery horizon: 10000 realized transitions after the branch point;
- explicit abort recovery transitions: zero;
- rejected-action recovery transitions: zero.

The total transition counter begins at the initial simulator state and includes nominal-prefix and post-branch realized transitions.

At branch selection, the recovery counter is zero. The recovery horizon becomes active when the selected physical branch first attempts a transition. The counter increments only after a branch-selected transition is actually realized. The first realized branch transition is recovery transition 1. A rejected action and explicit abort realize no transition and therefore retain a recovery count of zero.

Recovery-horizon exhaustion occurs after 10000 recovery transitions have been realized without an earlier stop. Total-horizon exhaustion occurs after 100000 total transitions have been realized without an earlier stop.

## Recovery Success v0

Recovery success requires all of:

```text
declared_hazard_avoided
and not invalid_simulation
and not invalid_recovery_evaluation
and target_radius_crossing
and phase34_compatible_recoverable_crossing
and recovery_target_reached_within_10000_transitions
```

The crossing must occur at or after the branch point. The Phase34-compatible recoverable condition uses the existing benchmark definition; this experiment does not create a new recoverability threshold.

Report these independently:

- simulator-defined success;
- target-radius crossing;
- Phase34-compatible recoverable crossing;
- Recovery Success v0;
- retreat or termination;
- hazard outcome.

`No overspeed` alone is not recovery success.

## Stop Conditions

Terminate a branch on the first chronological occurrence of any listed condition. When multiple conditions are detected at the same boundary, invalid and hazard conditions take priority over recovery success.

| Priority | Condition | Branch terminal label | Controlled taxonomy mapping |
| ---: | --- | --- | --- |
| 1 | Invalid simulation | `invalid_simulation` | `invalid_simulation` |
| 2 | Invalid recovery evaluation | `invalid_recovery_evaluation` | `unknown` with manual audit |
| 3 | Realized overspeed | `overspeed` | `overspeed` |
| 4 | Instability | `instability` | `instability` |
| 5 | Unsafe state | `unsafe_state` | `unsafe_state` |
| 6 | Recovery action rejection | `recovery_action_rejected` | `unknown` with manual audit |
| 7 | Explicit abort | `explicit_recovery_abort` | `unknown` with manual audit |
| 8 | Recovery success | `recovery_success` | `success` |
| 9 | Recovery-horizon exhaustion | `recovery_horizon_exhausted` | `timeout` |
| 10 | Total-horizon exhaustion | `total_horizon_exhausted` | `timeout` |

The branch labels are experiment-specific diagnostic extensions. A future Result Schema writer must also populate the controlled Failure Label Taxonomy field using the explicit mapping above; it must not silently add these branch labels to the controlled taxonomy.

## Required Metrics

### Margin

- `overspeed_headroom`;
- `action_saturation_margin`;
- `available_correction_authority`, only where directly supported;
- `required_to_available_correction_ratio`, only when both inputs are valid and unit-consistent;
- unsupported multi-step quantities as `null`, never guessed.

### Cost

- recovery steps;
- total steps;
- normalized action effort;
- delta-v proxy;
- crossing delay;
- final radius error;
- final radial-velocity error;
- final tangential-velocity error;
- task-abandonment status.

### Intervention

- evaluation count;
- allow count;
- veto count;
- recovery-action rejection count;
- first and last intervention steps;
- longest veto streak;
- veto segment count;
- action-suppression duration.

### Outcomes

- hazard avoided;
- recovery success;
- simulator success;
- controlled terminal label;
- experiment-specific branch terminal label;
- recovery outcome taxonomy;
- task recovery;
- explicit abort;
- invalid evaluation;
- new failure caused by a recovery action.

Unsupported or inapplicable fields must be `null` with a reason. Zero must be reserved for measured zero.

## Common-State Comparison Contract

This is not a monitor-off versus monitor-on paired experiment. It is a four-branch comparison from one common state.

Every branch must share:

- canonical branch-state hash;
- case configuration hash;
- simulator constants hash;
- nominal prefix;
- seed;
- total and recovery horizons;
- metric and logging contracts.

Only the branch decision may differ. Explicit abort must have zero post-branch transitions. An incomplete, hash-mismatched, or nonidentical common-state set is not a valid comparison.

## Artifact Contract

All artifacts are reserved under:

```text
analysis/recovery_action_branching_nonformal_v0/
```

| Artifact | Design-freeze status |
| --- | --- |
| `manifest.json` | Exists now; frozen design contract |
| `branch_state.json` | Future; must not exist today |
| `results.csv` | Future; must not exist today |
| `decision_log.jsonl` | Future; must not exist today |
| `summary.md` | Future; must not exist today |
| `comparison.png` | Future; must not exist today |

No future writer may target the frozen Final Veto directory or protected Phase34-37 directories.

## Why The Experiment Is Nonformal

- It uses one diagnostic case.
- It does not include the eight-case preservation set.
- The magnitude `0.25` is an unvalidated fixed design choice.
- The 10000-transition recovery horizon is an unvalidated fixed design choice.
- The physical-action branches are heuristic and unimplemented.
- It cannot establish cross-case or benchmark-wide behavior.

The experiment can produce a structured diagnostic result, not accepted benchmark progress by itself.

## Why The Branches Are Diagnostic

The branches test distinct response mechanisms:

| Branch | Diagnostic question |
| --- | --- |
| Zero action | Does repeated coast preserve the v0 hazard result while remaining stalled? |
| Velocity-opposed thrust | Does bounded inertial velocity opposition create usable speed headroom? |
| Tangential correction | Does correcting tangential error preserve more orbital task structure than general velocity opposition? |
| Explicit abort | What hazard exposure and operational cost remain when task pursuit ends immediately? |

No mechanism is assumed effective. A branch that avoids overspeed but fails crossing remains `hazard_avoided_task_stalled` or the appropriate termination outcome.

## Allowed Claims After A Future Valid Run

- Branch-specific diagnostic outcomes for this one case.
- Whether a branch avoided the declared overspeed hazard in this case.
- Whether a branch reached the predeclared recovery target in this case.
- Relative intervention burden and cost among the four valid common-state branches in this case.

## Prohibited Claims

- Formal safety.
- Universal recovery.
- Controller superiority.
- Benchmark-wide effectiveness.
- Cross-case generalization.
- Hardware validity.
- Deployment readiness.
- Cross-embodiment validation.
- Proof that magnitude `0.25` is optimal.
- Proof that 10000 recovery transitions are sufficient.

## Expansion Gate

Expansion to more cases would require all of the following evidence from the one-case diagnostic:

1. The branch point can be reproduced with identical canonical bytes and hashes.
2. All four branches produce structurally valid records with no hidden fallback.
3. Prediction and realized transitions are logged distinctly.
4. Margin, cost, burden, and outcome fields can be populated without guessing.
5. Any apparent recovery is based on the frozen Recovery Success v0 predicate, not only hazard absence or closest approach.
6. Action parameters remain fixed during the one-case comparison.
7. New failure mechanisms and rejected actions remain visible.
8. A new manifest predeclares any additional cases and restores known-success preservation testing before broader claims.

One favorable branch outcome would justify only a predeclared multi-case diagnostic. It would not justify declaring a recovery controller validated.

## Implementation Boundary

No runner, recovery controller, action library, output writer, or result artifact is created by this milestone. Implementation requires a separate task after this manifest and validator remain stable under review.

## Non-Claims At Freeze Time

At design freeze:

- no simulation has run;
- no branch has a measured outcome;
- no result row exists;
- no branch winner exists;
- no positive recovery claim exists;
- no branch policy or runner is implemented;
- Final Veto v0 evidence remains unchanged.
