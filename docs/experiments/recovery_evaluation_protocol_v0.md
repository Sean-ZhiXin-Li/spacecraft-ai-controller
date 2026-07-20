# Recovery Evaluation Protocol v0

## 1. Status

Infrastructure specification only; recovery experiment not run.

Completed: 2026-07-20

Protocol ID: `recovery_evaluation_protocol_v0`

This protocol defines future evaluation rules. It contains no measured branch
outcomes, branch ranking, or recovery-performance claim.

## 2. Purpose

Recovery Evaluation Protocol v0 defines how future recovery branches are to be
evaluated after execution. It creates a common interpretation boundary for:

- declared-hazard outcomes;
- state recovery;
- task recovery;
- recovery margin components;
- physical, performance, and operational cost;
- intervention burden;
- common-state branch comparisons;
- failure interpretation and allowed claims.

The protocol does not define a controller, select an optimal action, tune a
branch, or claim that recovery capability exists. It also does not turn the
bounded Recovery Branch Runner v0 into an experiment runner.

The governing experiment remains the one-case, four-branch, nonformal design in
`analysis/recovery_action_branching_nonformal_v0/manifest.json`. Its frozen
10,000-transition recovery horizon is distinct from the current runner's
infrastructure-only cap of 32 transitions. Until the missing evaluators and
experiment artifact layer are implemented and separately authorized, bounded
runner output is diagnostic infrastructure evidence only and
`recovery_success` remains `not_evaluated`.

### Evaluation Unit

The unit of evaluation is one complete branch trajectory beginning at the
frozen canonical branch state and ending at the first frozen stop condition.
The unit of comparison is the complete four-branch set that shares that state.

The post-branch evaluation interval begins at the branch point before any
branch action is executed. For a physical branch, recovery transition 1 is the
first realized branch-selected transition. Explicit abort realizes zero
post-branch transitions. The frozen recovery horizon is 10,000 realized branch
transitions, and the frozen total episode horizon is 100,000 realized
transitions including the nominal prefix.

Unknown, unsupported, or unevaluated quantities must be `null` or
`not_evaluated`, with a reason. They must not be converted to zero, `false`, or
a favorable outcome.

## 3. Evaluation Hierarchy

The three levels below answer different questions and must be stored and
reported separately.

| Level | Question | Required evidence | What it does not establish |
| --- | --- | --- | --- |
| 1. Hazard outcome | Did a declared realized hazard occur during the evaluation interval? | Valid realized states, the predeclared hazard signal, threshold, comparator, interval, and terminal reason | State recovery, task recovery, broad safety, or causal benefit |
| 2. State recovery | Did the trajectory enter a predeclared recoverable region? | Declared state-region components and the first step at which all required components hold | Simulator success, task completion, or acceptable cost |
| 3. Task recovery | Did the branch avoid the declared hazard and regain the declared task condition within the recovery horizon? | Valid Level 1 and Level 2 evidence, no disqualifying stop condition, and horizon compliance | General recovery, optimality, formal safety, or cross-case effectiveness |

### Level 1: Hazard Outcome

Every hazard must be named and evaluated independently. For the frozen
branching design, the declared hazard is realized overspeed under the strict
rule:

```text
realized_speed_ratio > 1.90
```

A value exactly equal to `1.90` is not overspeed. `overspeed_avoided=true`
requires valid realized evidence throughout the applicable post-branch
interval and no realized overspeed before the branch terminates. An invalid
simulation or invalid recovery evaluation makes the hazard outcome unknown; it
must not be reported as avoided.

Instability and unsafe state remain separate hazards. Simulation validity is a
separate prerequisite, although a report may state that invalid simulation was
not observed. Avoiding one hazard does not imply the others were evaluated or
avoided.

Explicit abort may be classified as
`hazard_avoided_through_termination` only when no declared hazard occurred
before termination. That classification is not task recovery. A causal claim
that a branch avoided a failure additionally requires a valid common-state
reference branch in which the declared hazard actually occurred.

### Level 2: State Recovery

State recovery asks whether the spacecraft returns to a declared recoverable
region. For Recovery Success v0, the required state events are:

1. a target-radius crossing at or after the branch point; and
2. a Phase34-compatible recoverable crossing.

The following may also be reported as separate state components when their
definitions are frozen before execution:

- bounded radius error;
- bounded radial-velocity error;
- bounded tangential-velocity error;
- orbital energy or state margin;
- distance from a declared irreversible region.

These components must not be collapsed into an unsupported recovery score.
Closest approach, reduced orbital error, or positive overspeed headroom without
the required crossing and recoverability event remains diagnostic evidence.

### Level 3: Task Recovery

For the frozen orbital branching experiment, Recovery Success v0 is:

```text
recovery_success =
    declared_hazard_avoided
    and not invalid_simulation
    and not invalid_recovery_evaluation
    and target_radius_crossing
    and phase34_compatible_recoverable_crossing
    and recovery_target_reached_within_10000_transitions
```

The crossing must occur at or after the branch point. The frozen stop-condition
priority means overspeed, instability, unsafe state, invalid simulation,
invalid recovery evaluation, action rejection, or explicit abort terminates the
branch before recovery success can be assigned.

Report `final_simulator_success`, target-radius crossing, recoverable crossing,
and `recovery_success` independently. No overspeed alone is insufficient for
task recovery because a branch may avoid overspeed while stalling, abandoning
the task, entering another failure mode, or exhausting its horizon.

## 4. Metrics

All metrics use a declared interval and evidence level. Realized and predicted
values must remain distinct. The evidence level should be one of `measured`,
`one_step_predicted`, `multi_step_predicted`, `heuristic`, or
`unvalidated_assumption`, consistent with Recovery Metrics v0.

### Recovery Success

`recovery_success` is a boolean outcome only after every required predicate has
been evaluated. It must be `null` or `not_evaluated` when the recovery criterion,
validity checks, or full horizon evidence is unavailable.

It remains separate from:

- `hazard_avoided`;
- `final_simulator_success`;
- `crossed_target_radius`;
- `recoverable_crossing`;
- retreat, safe-mode, or termination outcomes.

The current bounded runner does not evaluate recovery success and cannot emit a
scientific Recovery Success v0 result.

### Recovery Margin

Recovery margin is a component vector, not a combined scalar.

| Component | Definition | Units | Desired direction | Evidence and limitation |
| --- | --- | --- | --- | --- |
| `overspeed_headroom` | `1.90 - speed_ratio`, reported separately for realized and predicted states | Dimensionless | Higher; positive is below this one threshold | Measures only overspeed margin, not task recoverability |
| `available_correction_authority` | Available acceleration or normalized action authority in a declared correction direction after limits | `m/s^2` or declared normalized-action units | Higher for the same direction and state | Derived from the declared actuator/dynamics model; direction-dependent |
| `action_saturation_margin` | `1 - max(abs(u_x), abs(u_y))` under the frozen component limit | Dimensionless | Higher | Instantaneous; does not show whether remaining authority is useful |
| `recovery_horizon_remaining` | Frozen recovery horizon minus realized recovery transitions | Transitions | Higher | Administrative budget only; not proof that recovery is reachable |
| `minimum_irreversible_region_distance` | Minimum declared distance to an irreversible or disqualifying region | Declared physical or normalized state units | Higher | `null` until the region and state scaling are defined |
| `resource_margin` | Resource budget predicted or measured to remain after recovery | Declared effort, impulse, energy, or resource units | Higher | `null` when no validated resource model exists |
| `required_to_available_correction_ratio` | Required correction divided by available authority for the same direction, units, and horizon | Dimensionless | Lower; values below one may suggest adequate modeled authority | `null` for zero authority or unsupported required correction; not a feasibility proof |

Signed values must be preserved. Unknown values must be `null`, not zero. Any
future aggregate margin requires a separate predeclared normalization,
weighting, monotonicity argument, and validation against realized recovery.

### Recovery Cost

Cost is a vector. A branch may recover at unacceptable cost, and a low-cost
branch may fail recovery.

| Cost class | Required components | Comparison rule |
| --- | --- | --- |
| Physical | `recovery_control_effort`, `additional_control_effort`, `recovery_delta_v_proxy` | Use identical action norm, time step, and interval; normalized effort is not fuel |
| Performance | `recovery_steps`, `extra_steps`, `delay_to_crossing`, `crossing_delay`, `progress_loss`, final radius/radial/tangential error | Compare like events; crossing delay is `null` if a required crossing is absent |
| Operational | `intervention_count`, `controller_switch_count`, veto or suppression duration, task-abandonment status, termination reason | Report components; do not hide abort or termination inside a favorable aggregate |

`recovery_control_effort` is the sum or integral of executed normalized-action
magnitude over the declared recovery interval. `recovery_delta_v_proxy` is the
integrated thrust-acceleration magnitude and must remain labeled a proxy.
Physical units may be used only when the mapping is declared and supported.

### Intervention Burden

Let `N_eval` be valid monitor evaluations, `N_allow` allow decisions, and
`N_intervention` proposals not executed unchanged during the declared interval.

| Metric | Definition | Missing or edge rule |
| --- | --- | --- |
| `intervention_rate` | `N_intervention / N_eval` | `null` when `N_eval = 0` |
| `allow_rate` | `N_allow / N_eval` | `null` when `N_eval = 0`; do not assume complementarity for nonbinary decision layers |
| `first_intervention_step` | First step where proposed and executed authority differ | `null` when no intervention occurs |
| `longest_intervention_streak` | Maximum adjacent intervention decisions without nominal execution between them | Zero only when a valid evaluation stream exists and contains no intervention |
| `total_intervention_duration` | Steps or elapsed time during which nominal action is not executed unchanged | Requires proposed and executed actions at every counted boundary |
| `progress_per_intervention` | Change in a predeclared task-progress measure divided by intervention count | Exploratory; `null` without a valid progress measure or when count is zero |

Also report `last_intervention_step`, veto segment count, action-suppression
duration, recovery-action rejection count, and controller switches when
available. `progress_per_intervention` is not permitted until its numerator,
denominator, interval, and sign convention are frozen. Closest approach alone
is not an acceptable task-progress numerator.

## 5. Branch Comparison Rules

The future experiment is a four-branch common-state diagnostic, not a
monitor-off versus monitor-on ablation. A comparison is eligible only when all
four records are structurally valid and the comparison contract below passes.

| Must be identical | Allowed difference |
| --- | --- |
| Canonical `branch_state` bytes and hash | Selected frozen recovery branch |
| Case configuration and case hash | Generated branch action or explicit abort decision |
| Simulator configuration and constants hash | Consequent realized trajectory and outcome |
| Nominal prefix and branch step | Branch-specific intervention and cost |
| Seed | Branch-specific terminal condition |
| Recovery and total horizons | None other than branch decision consequences |
| Stop-condition definitions and priority | None |
| Hazard threshold and strict comparator | None |
| Metric definitions, units, and missing-value rules | None |

Additional comparison rules:

1. Every branch must start from the frozen canonical branch-state hash.
2. Branch actions, magnitude, zero tolerance, and Final Veto handling must match
   the frozen manifest; no action may be tuned after observing another branch.
3. Every physical branch must use the same transition semantics and Final Veto
   threshold. Explicit abort must execute zero post-branch transitions.
4. The first chronological stop condition must be selected with the frozen
   priority order.
5. An incomplete branch set, hash mismatch, invalid record, changed horizon, or
   changed stop definition makes the comparison incomplete and claim-ineligible.
6. Raw outcomes and component metrics must be reported before differences or
   rankings. No weighted winner score is defined in v0.
7. Hazard comparison must name the hazard. A branch-level veto or action
   rejection is not evidence that a realized failure was avoided.
8. State and task outcomes must be compared separately from cost and burden.
9. Missing values must remain missing; branch ordering must not be inferred from
   unsupported values.
10. The one-case comparison remains diagnostic even if one branch satisfies
    Recovery Success v0. Expansion requires a new predeclared manifest and
    restoration of benchmark preservation checks.

### Comparison Reporting Order

Future reporting should present results in this order:

1. structural validity and common-state equality;
2. hazard and validity outcomes by branch;
3. state-recovery components by branch;
4. task-recovery outcome by branch;
5. physical, performance, and operational cost;
6. intervention burden;
7. failure labels and new failure mechanisms;
8. a scoped one-case interpretation and non-claims.

This order prevents a favorable proxy or low cost from overriding a failed
task outcome.

## 6. Failure Interpretation

| Observed pattern | Required interpretation | Not permitted |
| --- | --- | --- |
| Declared hazard avoided, task criterion not met | `safety improvement without recovery`, scoped only to the declared simulated hazard; use the more specific recovery outcome such as `hazard_avoided_task_stalled` | Successful recovery or broad safety claim |
| Declared hazard avoided, valid recovery target reached within horizon | `successful recovery` for this frozen case and branch | Universal recovery, optimality, or benchmark-wide effectiveness |
| Task recovered with materially higher cost or burden | `recovery with increased burden`, with each cost component reported | Hiding burden inside the success boolean |
| Hazard avoided through explicit abort | `hazard_avoided_through_termination`; task recovery is false under the current criterion | Recovery success or simulator success inferred from refusal |
| Hazard avoided and declared retreat target reached, original task not recovered | `hazard_avoided_through_retreat` | Original task completion |
| Recovery action rejected before execution | `recovery_action_rejected`; preserve prediction and rejection evidence | Treating a non-executed action as a failed or successful realized recovery |
| Declared hazard occurs after an allowed recovery action | `hazard_not_avoided`; audit prediction and realization | Hazard avoidance |
| Recovery action causes a different failure | `recovery_action_caused_new_failure`; report both original hazard context and terminal mechanism | Counting absence of the original hazard as success |
| Invalid simulation or recovery evaluation | `invalid_evaluation` or the applicable controlled label with manual audit | Optimistic hazard avoidance or recovery assignment |
| Horizon expires without recovery | `hazard_avoided_task_stalled` only if the declared hazard was validly absent; terminal label maps to `timeout` | Recovery success |
| Recovery criterion is unavailable in bounded infrastructure output | `not_evaluated` | Converting missing evaluation to failure or success |

The phrase `safety improvement without recovery` is shorthand for a reduction
in one predeclared simulated hazard. It is not a formal-safety or all-hazards
claim.

Controlled terminal labels must continue to follow Failure Label Taxonomy v0.
Experiment-specific branch labels such as `recovery_action_rejected` or
`explicit_recovery_abort` are diagnostic extensions and require the frozen
controlled-label mapping and manual-audit treatment.

## 7. Claim Restrictions

After a complete and valid future run, the strongest allowed statements are:

- branch-specific diagnostic outcomes for the one frozen source case;
- whether a branch encountered or avoided the declared overspeed hazard;
- whether a branch reached Recovery Success v0 within the frozen horizon;
- component-wise cost and intervention-burden differences for that case;
- observed failure-mode substitutions and action rejections.

The protocol explicitly prohibits:

- universal recovery claims;
- optimal controller or branch claims;
- benchmark-wide effectiveness from the one-case diagnostic;
- cross-case generalization;
- cross-domain or cross-embodiment validation;
- formal safety, verified Runtime Assurance, or guaranteed hazard avoidance;
- hardware, flight, deployment, or sim-to-real claims;
- claims that action magnitude `0.25` is optimal;
- claims that 10,000 recovery transitions are sufficient in general;
- treating a bounded runner infrastructure trace as a recovery experiment;
- treating hazard absence, crossing, simulator success, or low intervention
  burden alone as Recovery Success v0.

No comparison claim is allowed when common-state equality, full branch
completeness, validity, or predeclared metric availability fails. Negative and
inconclusive outcomes must be retained rather than removed from the comparison.

## 8. Relationship To Existing Documents

| Document | Relationship to this protocol |
| --- | --- |
| `docs/benchmarks/recoverability_benchmark_v1.md` | Supplies the separation between target-radius crossing, recoverable crossing, final simulator success, diagnostics, and accepted progress. |
| `docs/benchmarks/failure_label_taxonomy_v0.md` | Controls terminal failure labels and priority; branch-specific outcomes remain diagnostic extensions. |
| `docs/benchmarks/result_schema_v1.md` | Supplies future field meanings and missing-value rules; this protocol does not modify or implement that schema. |
| `docs/benchmarks/recoverability_regression_policy_v0.md` | Prevents one-case or proxy improvements from becoming broader progress claims and requires known-success preservation before expansion. |
| `docs/theory/recovery_metrics_v0.md` | Defines the preliminary margin, cost, intervention, outcome, decision, and evidence-level concepts refined here into evaluation rules. |
| `docs/experiments/recovery_action_branching_nonformal_v0.md` | Freezes the source case, common branch point, four actions, horizons, stop conditions, Recovery Success v0 predicate, and claim scope. |
| `docs/experiments/recovery_branch_executor_v0.md` | Documents the one-step execution boundary and unchanged physics/Final Veto dependencies; it produces no experiment result. |
| `docs/experiments/recovery_branch_runner_v0.md` | Documents bounded orchestration and partial stop tracking; its 32-step cap and `not_evaluated` fields are infrastructure constraints, not this protocol's scientific horizon. |

### Implementation Boundary

This document does not authorize branch execution. Before a recovery comparison
can run, a separate implementation milestone must add and test the missing
recovery-success, instability, and unsafe-state evaluators; implement the full
frozen horizon and artifact contract; validate common-state branch completeness;
and preserve the protected Phase34-37 and Final Veto evidence.

No measured result is created by defining this protocol.
