# Recovery Branch Executor v0

## Status

Infrastructure implemented; recovery experiment not run.

Completed: 2026-07-19

## Purpose

Recovery Branch Executor v0 provides the smallest execution boundary needed to test one frozen recovery action from the common branch state. It validates the branch-state hash, generates one predeclared action, applies the unchanged Final Veto decision rule, and delegates an allowed transition to the existing Phase34/35 transition function.

The executor performs no optimization, learning, tuning, branch selection, comparison, result publication, or recovery-success evaluation.

This document describes execution infrastructure only. No recovery comparison experiment has been performed.

## Relationship To Recovery Action Branching v0

The governing experiment contract remains `analysis/recovery_action_branching_nonformal_v0/manifest.json`, with its interpretation in `docs/experiments/recovery_action_branching_nonformal_v0.md`. The executor implements only the common mechanics needed before a future runner can evaluate that design:

1. validate the frozen branch boundary;
2. generate one named branch action;
3. evaluate that action through unchanged Final Veto logic;
4. execute at most one existing simulator transition;
5. return an internal execution record.

It does not implement the frozen 10,000-transition recovery horizon, stop-condition taxonomy, metrics, logging, four-branch comparison, or result artifacts.

## Frozen Branch-State Dependency

The only supported source boundary is:

`analysis/recovery_action_branching_nonformal_v0/branch_state.json`

Before action generation, the executor verifies:

- the complete canonical branch-state SHA-256 hash;
- schema version and frozen source case;
- strict overspeed threshold `1.90` and comparator `>`;
- Final Veto branch decision identity;
- capture before nominal and fallback execution;
- case configuration hash;
- simulator configuration hash;
- simulator constants hash;
- finite state and dynamics inputs.

The executor rejects modified or inconsistent branch-state data. It does not repair, normalize, or guess missing values.

## Supported Branches

| Branch ID | Generated response | Transition behavior |
| --- | --- | --- |
| `zero_action_reference_v0` | `(0.0, 0.0)` | Final Veto evaluation, then at most one transition |
| `velocity_opposed_thrust_v0` | `-0.25 * v / ||v||` | Final Veto evaluation, then at most one transition |
| `tangential_error_correction_v0` | `-0.25 * sign(v_t - target_circular_speed) * e_t` | Final Veto evaluation, then at most one transition |
| `explicit_abort_v0` | no action | no transition; terminal reason `explicit_recovery_abort` |

No branch is preferred, ranked, or identified as a recovery winner.

## Action Definitions

### Zero Action

```text
u = (0.0, 0.0)
```

This is a reference action, not a proven-safe action.

### Velocity-Opposed Thrust

```text
u = -0.25 * v / ||v||
```

The velocity is the inertial Cartesian velocity `(vx, vy)`. When `||v|| <= 1e-12`, the action is `(0.0, 0.0)`. A nonzero action has magnitude `0.25` before the existing component clamp. This remains a heuristic response and is not proven braking.

### Tangential-Error Correction

```text
e_r = r / ||r||
e_t = (-e_r_y, e_r_x)
v_t = dot(v, e_t)
tangential_error = v_t - target_circular_speed
u = -0.25 * sign(tangential_error) * e_t
```

The target circular speed comes from the frozen simulator configuration. The sign is zero when the absolute error is at most `1e-12 m/s`; otherwise it follows the frozen positive and negative convention. Zero position norm or nonfinite geometry is an invalid evaluation. No radial correction is added.

### Explicit Abort

Explicit abort returns no action, executes no transition, and records `explicit_recovery_abort`. It is termination infrastructure, not task recovery.

## Final Veto Handling

Every non-abort proposed action is evaluated through `one_step_overspeed_veto_v0` with the frozen strict `speed_ratio > 1.90` rule. The predictor delegates to `simulator.phase34_35_transition.step_phase34_35_transition` using dynamics values stored in the frozen branch state.

When Final Veto allows the action, the executor calls the same transition function for the realized transition and requires exact equality with the prediction. When Final Veto rejects the recovery action, the executor:

- returns `recovery_action_rejected`;
- executes zero transitions;
- does not execute the proposed action;
- does not substitute the Final Veto zero-action fallback;
- does not choose another branch.

## Internal Execution Record

`RecoveryBranchExecutionResult` is an immutable internal infrastructure record with these required fields:

| Field | Meaning |
| --- | --- |
| `branch_id` | selected frozen branch identifier |
| `executed` | whether one physical transition was executed |
| `action` | generated action, or null for explicit abort |
| `previous_state_hash` | canonical SHA-256 of the pre-transition state |
| `next_state_hash` | canonical SHA-256 of the realized state, or null |
| `terminal_reason` | one-step completion, action rejection, or explicit abort |
| `transition_count` | zero or one |
| `valid` | whether the returned execution record is valid |

The record also carries typed previous and next states and the monitor decision for bounded internal verification. It is separate from Result Schema v1 and creates no experiment result row.

## Reproducibility Boundary

The executor owns no spacecraft physics. It reconstructs the dynamics context from the hash-validated branch-state configuration and calls the existing pure transition function. Prediction and realization must be exactly equal for the same state, action, and context.

Recovery Branch Executor v0 permits exactly `horizon_steps=1`. Values of zero, more than one, noninteger values, and the frozen 10,000-step recovery horizon are rejected. This restriction prevents infrastructure validation from becoming an undeclared recovery experiment.

## Current Limitations

- No recovery comparison has run.
- No branch has been evaluated as a recovery policy.
- No multi-step branch policy is implemented.
- No 10,000-step recovery horizon is available.
- No recovery metrics, stop conditions, or task-recovery criteria are evaluated.
- No result CSV, decision log, summary, or comparison plot is written.
- No controller switching, retreat manager, safe-mode manager, or Decision Manager is implemented.
- No formal safety, hardware, deployment, or cross-domain claim is supported.

The next implementation boundary is a separately reviewed nonformal runner. It must preserve the common branch-state hash, add declared logging and stop conditions, and pass its own tests before any branch comparison is authorized.
