# Final Veto Ablation Plan v0

## Status

Week 7 experiment-design document.

Completed: 2026-07-10

Scope: design for one minimal rule-based Final Veto ablation in the simplified 2D spacecraft simulator.

This document does not implement a monitor, runner, result checker, Decision Manager, or Runtime Assurance system. It does not modify controllers, historical artifacts, simulator thresholds, or protected Phase34/36/37 evidence.

## Purpose

The purpose is to specify the smallest paired experiment that can test whether a pre-action rule-based monitor reduces one declared simulated hazard without destroying known recoverability behavior.

The plan separates:

- a protected preservation set for known Phase34 recoverable cases,
- a diagnostic stress set selected because archived monitor-off behavior contains the target hazard,
- monitor-off and monitor-on counterfactual arms,
- hazard reduction from recoverability preservation,
- avoided failures from blocked successes and unnecessary vetoes.

Any future positive result would be evidence about one rule-based simulator monitor under one declared experiment. It would not establish formal safety or universal Runtime Assurance.

## Existing Evidence Boundary

The protected guard reports:

- Phase34 `radius_priority`: `8 / 24` crossings and `8 / 24` recoverable crossings.
- Phase36B families: `0` overspeed and `0` instability on each protected 24-case family.
- Phase37A: `0` overspeed and `0` instability across 144 protected rows.
- Phase37B: `0` overspeed and `0` instability across 24 protected rows.

Those facts define preservation evidence, not a monitor test. A stress set is necessary because a monitor cannot demonstrate hazard reduction when the monitor-off arm contains no hazard.

Archived, non-protected diagnostic artifacts contain overspeed-positive rows. They may guide predeclared stress-case selection, but they must not be overwritten or relabeled as new monitor evidence. The future ablation must generate fresh paired outputs in a new artifact directory.

## Candidate Hazard Audit

| Candidate hazard | Existing repository signal | Advantages for a first ablation | Limitations | Decision |
| --- | --- | --- | --- | --- |
| Overspeed | `overspeed`, `max_speed_ratio`, terminal check `speed_ratio > 1.90` | Explicit scalar signal, controlled taxonomy label, existing protected reporting, archived hazard-positive cases, one-step prediction available | Protected benchmark has zero overspeed, so a separate stress set is required | Selected |
| Instability | `instability`, plus `out_range`, `too_close`, and `radial_stall` mechanisms in phase scripts | Safety-relevant and already reported | Composite, phase-specific mechanism; a single first monitor would obscure which instability was targeted | Not selected |
| Low recoverability margin | Recoverability distance exists in some scripts; `minimum_recovery_margin` is proposed in Result Schema v1 | Closely tied to the project thesis | No normalized margin field or accepted threshold across current scripts | Defer |
| Controller saturation | Action norms and some saturation diagnostics exist in parts of the repository | Pre-action and controller-local | Not consistently logged in protected Phase34/36/37 result rows; no single controlled hazard definition | Defer |

## Selected Hazard Rationale

The selected hazard is simulator-defined overspeed:

```text
overspeed := executed-state speed_ratio > 1.90
```

The value `1.90` is already used by the relevant explicit-controller scripts. It is not newly inferred from the planned experiment.

| Selection criterion | Overspeed rationale |
| --- | --- |
| Existing measurement | Current rows already report `overspeed` and `max_speed_ratio`. |
| Existing label | Failure Label Taxonomy v0 gives `overspeed` high terminal priority. |
| Pre-action testability | Existing deterministic one-step dynamics can evaluate the nominal proposed action before execution. |
| Minimal fallback | Existing controller code already uses zero-action safe-coast behavior in declared conditions. |
| Preservation evidence | Known Phase34 recoverable rows have no overspeed and maximum speed ratio near `1.0`. |
| Stress evidence | Five archived Phase35 `radial_energy_push` rows terminate in overspeed with maximum speed ratios above `1.90`. |
| Scope discipline | The outcome can be stated as a simulator overspeed ablation without broader safety claims. |

The repository also contains `SAFE_SPEED_RATIO = 1.65` in existing transfer logic. That controller-internal current-state guard remains unchanged. This ablation instead asks whether a separate pre-action monitor can veto a nominal action whose predicted next state crosses the existing `1.90` terminal overspeed boundary.

## Minimal Monitor Design

The future monitor should remain deterministic and single-step.

At every action step:

1. Receive current simulator state and nominal controller action.
2. Use the same one-step dynamics and constants as the rollout runner to predict the nominal next state.
3. Compute predicted next-state speed ratio using the declared target circular speed.
4. If predicted speed ratio is greater than `1.90`, veto the nominal action.
5. Execute the single fallback action `(0.0, 0.0)` for that step.
6. Re-evaluate the nominal controller and monitor on the next step.
7. Log every evaluation, allow, veto, fallback, and resulting state.

Design constants:

| Parameter | Week 7 design value | Status |
| --- | ---: | --- |
| Hazard target | `overspeed` | Existing simulator hazard |
| Realized hazard threshold | `speed_ratio > 1.90` | Existing script threshold |
| Veto trigger | predicted nominal next-state `speed_ratio > 1.90` | Experiment-design rule, not a proven safety boundary |
| Prediction horizon | `1` simulator step | Experiment-design threshold |
| Fallback | zero action `(0.0, 0.0)` for one step | Experiment-design choice |
| Monitor evaluation frequency | every proposed action | Experiment-design choice |

These values must be frozen before future evaluation. They are not proven constants, invariant sets, or formally verified thresholds.

If the fallback itself cannot prevent overspeed, the rollout must record the unresolved hazard or fallback failure. The result must not be converted into a successful veto merely because the nominal action was blocked.

## Preservation Set Versus Diagnostic Stress Set

| Set | Cases | Controller context | Purpose | Allowed scope |
| --- | ---: | --- | --- | --- |
| Protected preservation set | 8 | Phase34 `radius_priority` known recoverable cases | Test whether monitor-on preserves every known crossing and recoverable crossing | Preservation claim only |
| Diagnostic overspeed stress set | 5 | Phase35 `radial_energy_push` with Phase34 `radius_priority` post-cross mode | Exercise paired monitor-off/on behavior on archived overspeed-positive case/controller combinations | Diagnostic hazard-reduction claim only |

Neither set is the full 24-case benchmark. Both should use `is_full_benchmark=false` with explicit `subset_id`. If a future 24-case monitor run is added, it must receive a separate experiment ID and be reported separately from the stress result.

### Protected Preservation Cases

The preservation set is the exact eight Phase34 `radius_priority` cases currently known to produce recoverable crossings:

| `r0_over_target` | `initial_velocity_angle_deg` | `thrust_scale` | Protected expectation |
| ---: | ---: | ---: | --- |
| `1.00` | `150` | `8000` | crossing and recoverable crossing |
| `1.00` | `165` | `8000` | crossing and recoverable crossing |
| `1.00` | `170` | `8000` | crossing and recoverable crossing |
| `1.00` | `175` | `8000` | crossing and recoverable crossing |
| `1.00` | `150` | `10000` | crossing and recoverable crossing |
| `1.00` | `165` | `10000` | crossing and recoverable crossing |
| `1.00` | `170` | `10000` | crossing and recoverable crossing |
| `1.00` | `175` | `10000` | crossing and recoverable crossing |

Suggested metadata:

- `subset_id=phase34_known_recoverable_preservation_v1`
- `regression_set_membership=known_phase34_recoverable`
- `known_phase34_recoverable_case=true`

### Diagnostic Overspeed Stress Cases

The stress set should rerun these five archived Phase35 `radial_energy_push` case/controller combinations as new paired experiments:

| `r0_over_target` | `initial_velocity_angle_deg` | `thrust_scale` | Archived selection evidence |
| ---: | ---: | ---: | --- |
| `0.98` | `150` | `8000` | overspeed |
| `0.98` | `150` | `10000` | overspeed |
| `0.98` | `165` | `10000` | overspeed |
| `0.98` | `170` | `10000` | overspeed |
| `0.98` | `175` | `10000` | overspeed |

Suggested metadata:

- `subset_id=phase35_radial_energy_push_overspeed_stress_v0`
- `regression_set_membership=diagnostic_overspeed_stress`
- `known_phase34_recoverable_case=false`

The archived Phase35 values select the cases only. Future monitor-off behavior must be rerun and must contain at least one overspeed event before hazard-reduction language is allowed.

## Monitor-Off Versus Monitor-On Arms

| Property | Monitor-off arm | Monitor-on arm |
| --- | --- | --- |
| Initial state and case parameters | Fixed paired case | Identical paired case |
| Nominal controller | Declared preservation or stress controller | Same nominal controller |
| Nominal proposed action | Executed unchanged | Evaluated before execution |
| Monitor | Disabled | One-step overspeed monitor enabled |
| Trigger | Not applicable | Predicted nominal next-state speed ratio `> 1.90` |
| Fallback | None | One-step zero action |
| Simulator physics and thresholds | Unchanged | Unchanged |
| Result logging | Result Schema v1 plus monitor extension candidates | Same fields plus evaluations, vetoes, and fallbacks |
| Decision logging | Optional no-monitor reference event | Allow and veto evaluations logged |
| Pairing | `paired_run_id` and `counterfactual_arm=monitor_off` | Same `paired_run_id`; `counterfactual_arm=monitor_on` |

The only intended arm difference is monitor intervention. Case order, simulator constants, nominal controller settings, horizon, and success/failure thresholds must remain identical within each pair.

## Hypotheses

### H1: Hazard Reduction

The monitor-on arm reduces the selected simulated hazard relative to monitor-off on the diagnostic stress set.

### H2: Recoverability Preservation

The monitor-on arm preserves all known Phase34 recoverable cases on the protected preservation set.

### H3: Nontriviality

The monitor does not achieve hazard reduction by vetoing every useful action.

### H4: Measurable Tradeoff

Blocked successes, unnecessary vetoes, false negatives, and performance cost are reported rather than hidden.

### H5: Scope Discipline

Any positive result supports only a rule-based simulator monitor ablation, not formal safety or universal Runtime Assurance.

## Metric Definitions

| Metric | Exact design definition | Interpretation |
| --- | --- | --- |
| Monitor evaluations | Number of nominal proposed actions evaluated by the monitor | Denominator for intervention rate |
| Veto count | Number of nominal proposed actions replaced by fallback | Intervention volume |
| Veto rate | `veto_count / monitor_evaluation_count` | Nontriviality and performance context |
| Fallback count | Number of fallback actions executed | Must equal veto count for this single-fallback design |
| Overspeed count | Rollouts with executed-state `speed_ratio > 1.90` | Selected realized hazard count |
| Overspeed rate | `overspeed_count / rollouts_in_arm` | Arm-level hazard rate |
| Avoided failure | Paired case where monitor-off has overspeed and monitor-on does not | Positive paired hazard evidence |
| Blocked success | Paired case where monitor-off has a recoverable crossing or final simulator success and monitor-on loses that outcome after at least one veto | Safety-performance regression |
| Unnecessary veto | Veto in a hazard-negative monitor-off pair that does not avoid the selected hazard; report event count and affected rollouts | Over-conservatism signal |
| False negative | Monitor-on rollout reaches overspeed without a preceding veto for the action that crossed the threshold | Monitor miss |
| Fallback failure | Monitor vetoes, but monitor-on still reaches overspeed | Veto occurred but fallback was insufficient |
| Preservation count | Monitor-on preservation cases retaining both crossing and recoverable crossing | H2 primary metric |
| Invalid simulation count | Rows with `invalid_simulation=true` | Monitor integrity check |
| Unchanged nominal action count | Monitor evaluations that execute the nominal action unchanged | H3 nontriviality evidence |
| Performance cost summary | Paired deltas in steps, event/recovery time when available, control effort when available, and final outcome | Explicit tradeoff, not hidden optimization |

If a current script cannot populate a cost component, that component must be null and named as unavailable. It must not be guessed.

## Acceptance And Rejection Criteria

These are predeclared experiment-design thresholds for a future v0 ablation. They are not proven constants or safety guarantees.

| Criterion | Exact v0 threshold | If not met |
| --- | --- | --- |
| Historical protection | `python scripts/check_phase_results.py` exits `0` | Stop; no monitor claim |
| Artifact protection | No protected historical artifact is modified | Stop; repair experiment isolation |
| Pair completeness | Every declared case has exactly one monitor-off and one monitor-on row with matching `paired_run_id` | Incomplete evidence |
| Preservation crossing | Monitor-on crosses in `8 / 8` protected preservation cases | Regression; diagnostic only |
| Preservation recoverability | Monitor-on has recoverable crossing in `8 / 8` protected preservation cases | Regression; diagnostic only |
| Protected blocked successes | `0` blocked successes on the preservation set | Regression; diagnostic only |
| Invalid simulation | `invalid_simulation_on <= invalid_simulation_off`, with both counts reported | Integrity regression or incomplete evidence |
| Stress monitor exercise | Monitor-off overspeed count is at least `1` | Label `monitor_not_exercised`; no hazard-reduction claim |
| Hazard reduction | Monitor-on stress overspeed count is at least `1` lower than monitor-off | H1 not supported |
| Paired avoided failure | At least `1` avoided failure | No positive veto evidence |
| Nontrivial action execution | Aggregate `veto_count < monitor_evaluation_count` and each monitor-on rollout executes at least one unchanged nominal action when any action is proposed | Trivial refusal; reject useful-assurance claim |
| Tradeoff reporting | Blocked successes, unnecessary vetoes, false negatives, fallback failures, and performance cost are all reported, including zero values | Incomplete evidence |
| Scope separation | Preservation and stress rows use different `subset_id`; neither is labeled full benchmark | Scope violation |
| Threshold discipline | Hazard, trigger, fallback, sets, and acceptance thresholds are recorded before evaluation | Post-hoc design; diagnostic only |

If no monitor-off hazard occurs, the result is `monitor_not_exercised`, not `monitor_proved_safe`.

Passing these criteria permits a scoped positive ablation interpretation. It does not prove safety.

## Failure And Diagnostic Interpretation

| Outcome | Required interpretation |
| --- | --- |
| Hazard avoided with recoverability preserved | Possible positive rule-based ablation evidence |
| Hazard avoided but known success blocked | Safety-performance tradeoff; diagnostic only |
| No hazard in either arm | Monitor not exercised |
| Veto in safe baseline case | Unnecessary veto unless paired evidence shows the selected hazard was avoided |
| Monitor allows hazard | False negative if no pre-hazard veto occurred |
| Veto occurs but fallback still reaches hazard | Fallback failure, not an avoided failure |
| Monitor blocks every action | Trivial refusal, not useful assurance |
| Stress-set improvement without protected-set preservation | Diagnostic only |
| Result missing counterfactual pair | Incomplete evidence |
| Post-hoc detection without pre-action intervention | Detector evaluation, not Final Veto |

Terminal labels must still describe the realized rollout mechanism. For example, a monitor-on rollout that still exceeds the threshold remains `terminal_label=overspeed`. `avoided_failure` is a paired diagnostic field, not a replacement terminal label.

## Result Schema v1 Mapping

Future result artifacts should preserve Result Schema v1 fields and use a declared extension for monitor-specific data.

| Result Schema v1 field | Monitor-ablation use |
| --- | --- |
| `schema_version` | Use `result_schema_v1` until a declared extension version exists |
| `benchmark_id`, `benchmark_version` | Identify Recoverability Benchmark v1 |
| `experiment_id` | Separate preservation, stress, and any future full-benchmark run |
| `controller_id`, `controller_family` | Identify the nominal controller independently of the monitor |
| `case_id` | Stable identity shared by paired arms |
| `crossed_target_radius` | Report event outcome independently of hazard outcome |
| `recoverable_crossing` | Primary preservation outcome |
| `final_simulator_success` | Simulator-defined final outcome only |
| `overspeed` | Selected realized hazard boolean |
| `instability`, `unsafe_state`, `invalid_simulation` | Report other safety and validity outcomes; do not hide tradeoffs |
| `terminal_label` | Controlled realized terminal label from Failure Label Taxonomy v0 |
| `precursor_labels` | May record a declared monitor evaluation or veto precursor under an extension |
| `diagnostic_labels` | May record `monitor_not_exercised`, `avoided_failure`, `blocked_success`, or other declared diagnostics |
| `is_full_benchmark` | `false` for both v0 preservation and stress sets |
| `subset_id` | Distinguish preservation from diagnostic stress |
| `regression_set_membership` | Identify known Phase34 preservation or diagnostic overspeed stress |
| `known_phase34_recoverable_case` | `true` only for the protected eight-case set |
| `accepted_as_progress` | Keep subset rows `false`; aggregate summary may support scoped ablation evidence only if all criteria pass |
| `acceptance_reason` | Explain preservation, stress-only scope, rejection, or diagnostic status |

### Monitor-Specific Extension Candidates

| Candidate field | Purpose |
| --- | --- |
| `monitor_enabled` | Distinguish monitor-off and monitor-on arms |
| `monitor_id` | Identify exact rule and version |
| `monitor_evaluation_count` | Count pre-action evaluations |
| `veto_count` | Count replaced nominal actions |
| `fallback_count` | Count executed fallback actions |
| `hazard_target` | Use `overspeed` for v0 |
| `hazard_threshold` | Record `1.90` and units/normalization |
| `paired_run_id` | Link counterfactual arms |
| `counterfactual_arm` | `monitor_off` or `monitor_on` |
| `avoided_failure` | Paired hazard outcome |
| `blocked_success` | Paired loss of recoverable or final success |
| `unnecessary_veto` | Report over-conservative intervention |
| `false_negative` | Report missed pre-action hazard |
| `performance_cost_summary` | Store declared paired cost summary or reference |

These are design candidates only. They require a declared Result Schema extension before implementation. This document does not change Result Schema v1.

## Decision Log Schema v0 Mapping

Every monitor evaluation that allows, modifies, or vetoes an action should be reconstructable. At minimum, every veto must produce a Decision Log Schema v0 event.

| Decision Log field | Monitor-ablation use |
| --- | --- |
| `decision_type` | `veto_action` for blocked nominal action; `continue` for a logged allow event |
| `decision_reason` | `overspeed_risk` for v0 veto; a declared non-veto reason for allow |
| `decision_scope` | `veto` |
| `decision_authority` | `runtime_assurance` |
| `state_summary` | Current speed ratio, nominal predicted next ratio, fallback predicted ratio, proposed action |
| `safety_level` | `nominal`, `warning`, or `critical` under predeclared monitor rules |
| `recoverability_level` | Current general recoverability assessment or `unknown` if not computed |
| `trust_flags` | `none` or `unknown`; this v0 monitor does not implement trust estimation |
| `fallback_available` | Whether zero-action fallback can be issued |
| `fallback_action` | `reduce_action`, realized as zero action / safe coast for v0 |
| `veto_status` | `allow` for unchanged action or `modify_action` / `blocked` for vetoed nominal action |
| `veto_reason` | Numerical evidence, threshold, and rule identifier |
| `manual_audit_note` | Explain missing fields, fallback failure, or design caveats |

### Conceptual Allow Example

This is a design example, not an implemented result.

```text
decision_type=continue
decision_reason=known_success_preserved
decision_scope=veto
decision_authority=runtime_assurance
state_summary=current_speed_ratio=1.02; nominal_predicted_next_speed_ratio=1.03; threshold=1.90
safety_level=nominal
recoverability_level=recoverable
trust_flags=none
fallback_available=true
fallback_action=reduce_action
veto_status=allow
veto_reason=predicted risk below threshold; nominal action executed unchanged
manual_audit_note=conceptual allow record only
```

### Conceptual Veto Example

This is a design example, not an implemented result.

```text
decision_type=veto_action
decision_reason=overspeed_risk
decision_scope=veto
decision_authority=runtime_assurance
state_summary=current_speed_ratio=1.88; nominal_predicted_next_speed_ratio=1.91; threshold=1.90
safety_level=critical
recoverability_level=marginal
trust_flags=none
fallback_available=true
fallback_action=reduce_action
veto_status=modify_action
veto_reason=nominal one-step prediction exceeded declared overspeed threshold
manual_audit_note=nominal action replaced by zero action; outcome must still be evaluated
```

## Allowed Claims Versus Prohibited Claims

| Allowed if future evidence supports it | Prohibited |
| --- | --- |
| A rule-based monitor reduced a declared simulated hazard on a predeclared diagnostic stress set. | Formal safety |
| The monitor preserved the known Phase34 recoverable cases under the tested preservation set. | Verified Runtime Assurance |
| The ablation measured avoided failures, blocked successes, unnecessary vetoes, false negatives, fallback failures, and performance cost. | Guaranteed hazard avoidance |
| The experiment provides simulator-level evidence for a minimal veto mechanism. | Flight readiness or hardware readiness |
| The result is scoped to the declared rule, hazard, cases, threshold, and fallback. | Real spacecraft validation or sim-to-real transfer |
| A negative result identifies detector, fallback, preservation, or tradeoff limitations. | Drone, robotic manipulation, legged-robot, or marine-autonomy validation |
| Cross-embodiment architecture may motivate future domain-specific experiments. | Universal cross-embodiment safety |
| The experiment can inform a later Decision Manager design. | A claim that one ablation completes the Decision Manager architecture |

## Suggested Future Artifact Layout

Design only; do not create in Week 7:

```text
analysis/final_veto_ablation_v0/results.csv
analysis/final_veto_ablation_v0/paired_results.csv
analysis/final_veto_ablation_v0/decision_log.jsonl
analysis/final_veto_ablation_v0/summary.md
analysis/final_veto_ablation_v0/comparison.png
analysis/final_veto_ablation_v0/manifest.json
```

Protected Phase34/36/37 directories must not be overwritten.

## Suggested Future Implementation Components

The repository already uses top-level domain modules and `scripts/` entry points. A future implementation could use:

```text
runtime_assurance/__init__.py
runtime_assurance/final_veto_monitor.py
scripts/run_final_veto_ablation.py
scripts/check_final_veto_results.py
```

`runtime_assurance/` does not currently exist. Its creation should be an explicit implementation decision, not an implicit consequence of this document.

The future implementation should remain small:

- one monitor,
- one selected hazard,
- one fallback,
- paired monitor-off/on runs,
- one protected preservation set,
- one diagnostic stress set,
- explicit result and decision logging.

## Future Implementation Steps

| Step | Future action | Required output or check |
| ---: | --- | --- |
| 1 | Freeze monitor rule, threshold, fallback, cases, and acceptance criteria | Versioned experiment manifest |
| 2 | Add the minimal monitor module without changing nominal controllers | Unit checks for allow and veto decisions |
| 3 | Add paired runner using fresh output directory | Complete off/on pairs with stable IDs |
| 4 | Emit Result Schema v1 rows plus declared extension fields | `results.csv` and `paired_results.csv` |
| 5 | Emit Decision Log Schema v0 events | `decision_log.jsonl` with allow/veto evidence |
| 6 | Validate logical consistency and paired metrics | `check_final_veto_results.py` output |
| 7 | Run protected historical guard | `python scripts/check_phase_results.py` passes |
| 8 | Write scoped summary including negative tradeoffs | No formal-safety or cross-domain claims |

None of these steps is implemented by this Week 7 plan.

## Week 8 Handoff Questions

Week 8 is the August platform-transition snapshot, not an automatic full implementation phase.

- Which Week 1-7 concepts are now measurable?
- Which remain documentation-only?
- Is the first Final Veto ablation ready for implementation?
- What exact repository changes would be required?
- What scientific risks remain?
- What should be postponed?
- Has the project remained focused on recoverability-aware autonomy?
- Does the architecture still distinguish cross-embodiment relevance from domain validation?
- What should the September implementation priority be?

## Week 7 Completion Boundary

Week 7 is complete when this plan exists, the protected regression guard still passes, no historical evidence or controller code has been modified, and no monitor implementation or new analysis artifact has been created.
