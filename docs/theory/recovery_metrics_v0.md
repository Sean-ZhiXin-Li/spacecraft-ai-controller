# Recovery Metrics v0

## Status

Preliminary measurement specification.

Completed: 2026-07-17

Scope: recovery-aware intervention research in the simplified 2D spacecraft testbed, with carefully bounded cross-embodiment interpretation.

This document defines candidate metrics and decision evidence. It does not implement a recovery controller, change Final Veto v0, execute a rollout, or add experimental evidence. Candidate thresholds and action parameters below are experiment-design values, not proven constants.

Recovery Metrics v0 is consistent with:

- `docs/theory/recoverability_formalism_v0.md`;
- `docs/roadmaps/post_final_veto_recovery_architecture_plan.md`;
- `docs/reports/final_veto_v0_interpretation.md`;
- `docs/benchmarks/result_schema_v1.md`;
- `docs/architecture/decision_log_schema_v0.md`.

## 1. Motivation

Final Veto Overspeed Ablation v0 exposed a specific failure pattern:

1. The one-step monitor detects that the nominal proposed action is predicted to produce `speed_ratio > 1.90`.
2. The nominal action is vetoed.
3. The declared one-step zero-action fallback executes.
4. The next nominal action remains predicted to exceed the same threshold.
5. The monitor vetoes again.
6. Repeated veto and zero-action execution suppress the declared overspeed hazard but do not restore task progress.
7. The episode reaches `max_steps` without target-radius crossing, recoverable crossing, or simulator-defined success.

In the five frozen diagnostic stress pairs, monitor-off produced overspeed in `5 / 5` cases and monitor-on produced overspeed in `0 / 5`. Task recovery remained `0 / 5`. The monitor evaluated 511327 nominal proposals and vetoed 499877 of them. The overall intervention rate was approximately `0.977607`, and each stress-case rate was approximately `0.99973` to `0.99979`.

The evidence therefore separates two questions:

```text
Did the declared hazard occur?

Did useful task progress resume after intervention?
```

Final Veto v0 answered the first question positively for the tested pairs and the second negatively. Recovery Metrics v0 defines how future work should measure the gap without calling inactivity, horizon extension, or termination task recovery.

## 2. Terminology

| Term | Recovery Metrics v0 definition | Required qualification |
| --- | --- | --- |
| Hazard avoidance | The declared hazard does not occur during the specified evaluation interval. A causal avoided-failure claim additionally requires a matched hazard-positive counterfactual. | Name the hazard, signal, comparator, threshold, interval, and counterfactual rule. `No overspeed` means only that the overspeed condition was not observed. |
| Task recovery | After an intervention or detected failure, the system reaches a predeclared task-recovery target within the recovery horizon without a disqualifying invalid result. | Name the target set or task predicate, controller or policy, horizon, and constraints. |
| Recoverable state | A state or belief from which an available controller or policy can reach an acceptable continuation condition within a declared horizon, resource budget, model, and constraints. | Recoverability is policy-relative, horizon-relative, resource-relative, and evidence-relative. |
| Recovery action | A bounded action or maneuver proposed specifically to restore access to an acceptable continuation condition. | Declare applicability, duration, authority, expected mechanism, stop condition, and failure condition. |
| Recovery policy | A rule that selects and sequences recovery actions from observations or beliefs until a return condition, retreat condition, or termination condition is reached. | A fixed fallback is not automatically a recovery policy. |
| Intervention | Any recorded runtime decision that prevents the nominal proposed action from executing unchanged, including modification, controller switch, retreat, safe mode, or termination. | Record both nominal and executed actions and the intervention authority. |
| Fallback | A predeclared substitute selected when the nominal proposal is rejected or unavailable. | A fallback has no implied recovery or safety property. Final Veto v0's zero-action fallback is not proven safe. |
| Retreat | An intentional maneuver toward a declared retry-ready, lower-commitment, degraded-mission, or abort state. | Name the retreat target and completion criterion. Moving away geometrically is not sufficient by itself. |
| Termination | An explicit end to an attempt because continuation is prohibited, unavailable, or no longer justified. | Termination may avoid further exposure but is not task recovery unless the benchmark explicitly defines an acceptable abort objective. |
| Recovery horizon | The maximum steps or time allowed from recovery initiation to the declared recovery target or stop condition. | Report both units and the event that starts the horizon. |
| Recovery authority | The component allowed to select, modify, reject, or terminate recovery behavior. | Distinguish proposal authority, decision authority, and final veto authority in the decision log. |

The word `safe` must not be used as an undefined synonym for `did not overspeed`. A state, fallback, mode, or termination may be called safe only relative to declared constraints and evidence. The enum name `safe_mode` remains available for schema compatibility, but entering that mode does not prove safety.

## 3. Recovery Margin Candidates

No single recovery-margin scalar is accepted in v0. The candidates below form a margin vector whose components have different units, prediction requirements, and failure modes. A future scalarization would require justified normalization, weights, monotonicity, and validation against recovery outcomes.

Let:

- `x` be the current state;
- `u` be an action;
- `H_rec` be the recovery horizon;
- `R_rec` be the declared recovery target;
- `I` be the declared irreversible or disqualifying region;
- `speed_ratio = speed / target_circular_speed`;
- `r_limit = 1.90` only when evaluating the frozen overspeed condition.

| Symbol / field name | Qualitative meaning | Expected units | Desired direction | Evidence class | Known limitations |
| --- | --- | --- | --- | --- | --- |
| `m_speed`, `overspeed_headroom` | Distance from the current or predicted speed ratio to the declared overspeed threshold: `r_limit - speed_ratio`. | Dimensionless speed ratio | Higher is better; positive is below the declared threshold. | Derived from measured state, or one-step/multi-step predicted when evaluated on predicted state. | Measures one hazard only. Positive headroom does not imply recoverability, stability, or task progress. |
| `a_corr_available`, `available_correction_authority` | Maximum acceleration available in a declared correction direction after action limits, thrust scale, and mass are applied. | `m/s^2`, or normalized-action units if physical mapping is unavailable | Higher is better for the declared direction. | Derived from the dynamics and actuator model. | Direction-dependent; total thrust authority may not be usable for braking or geometry correction. Does not include time or resource limits by itself. |
| `m_brake_H`, `braking_authority_margin` | Available braking delta-v over `H_rec` minus the predicted delta-v required to remain below the declared speed limit and reach the next recovery condition. | `m/s` | Positive and higher are better. | Multi-step predicted or derived under a declared braking policy. | Depends on policy, horizon, gravity, changing geometry, and the definition of required braking. A one-step approximation may be misleading. |
| `m_radial_H`, `radial_reversibility_margin` | Available radial correction over `H_rec` minus the correction required to enter a declared radial-velocity envelope. | `m/s` | Positive and higher are better. | Multi-step predicted or derived. | Zero radial velocity is not always the correct immediate objective. Coupling to tangential motion and orbital phase must remain explicit. |
| `d_I_min`, `minimum_irreversible_region_distance` | Minimum predicted distance to a declared failure region during a candidate recovery trajectory. | Physical units when the boundary is physical; otherwise declared normalized state-space units | Higher is better. | Multi-step predicted. | A state-space distance is meaningless without variable scaling and a declared region. Distance does not prove a viable control path. |
| `m_time`, `irreversibility_time_slack` | Predicted steps or seconds until entry into `I` minus predicted steps or seconds required to reach `R_rec`. | Steps or seconds | Positive and higher are better. | Multi-step predicted. | Both times are model- and policy-dependent. Undefined when either event is not predicted within the horizon. |
| `reach_R_H`, `recovery_target_reachable_within_horizon` | Whether an available policy is predicted to reach `R_rec` before `I` within `H_rec`; a probabilistic version may report a calibrated probability. | Boolean, or probability with an explicit uncertainty model | `true` or higher calibrated probability is better. | Multi-step predicted. | This is a reachability predicate, not a geometric margin. It is only as credible as the model, policy set, and uncertainty assumptions. |
| `m_horizon`, `recovery_horizon_slack` | Remaining recovery horizon minus the minimum predicted steps needed to reach `R_rec`. | Steps or seconds | Positive and higher are better. | Multi-step predicted. | Minimum recovery time is usually unknown and may change after every action. A heuristic estimate must be labeled heuristic. |
| `m_action`, `action_saturation_margin` | Remaining component authority for a normalized action, for example `1 - max(abs(u_x), abs(u_y))` under a component limit of one. | Dimensionless normalized action | Higher is better. | Directly derived from a proposed or executed action. | Instantaneous and component-based. It does not measure whether the remaining direction is useful or whether thrust and mass mapping are constant. |
| `rho_corr`, `required_to_available_correction_ratio` | Required correction magnitude divided by available correction authority over the same declared direction and horizon. | Dimensionless | Lower is better; values below one suggest nominal authority sufficiency under the model. | Derived or multi-step predicted. | Undefined when available authority is zero. Numerator and denominator must share direction, horizon, and units. It is not a proof of feasibility. |
| `m_resource`, `resource_after_predicted_recovery` | Predicted resource budget remaining after reaching `R_rec`. | Fuel proxy, impulse, energy, action-effort units, or another declared resource | Higher is better. | Multi-step predicted or derived. | The current simulator does not provide a validated universal fuel metric. A proxy must be named and must not be reported as physical fuel. |

### Margin Reporting Rules

- Report the component vector before proposing any aggregate score.
- Record the state, action or policy, horizon, target, constraints, and evidence class used for every predicted margin.
- Use `null` when a component cannot be supported. Do not substitute zero for unknown.
- Preserve signed values. Clipping a negative margin to zero hides severity.
- Report minimum predicted and minimum realized values separately when both exist.
- Validate predicted margins against later realized outcomes before using them for decision authority.
- A positive heuristic margin must not be converted into a formal safety claim.

## 4. Recovery Cost Candidates

Recovery cost is a vector. Physical, performance, and operational costs must remain separately reportable before any combined objective is considered.

### Physical Cost

| Field | Definition | Units | Comparison rule | Limitation |
| --- | --- | --- | --- | --- |
| `recovery_control_effort` | Sum or integral of executed action magnitude during the recovery interval. | Normalized-action steps, or declared physical impulse units | Compare under the same action norm, time step, and interval. | Normalized effort is not fuel unless a validated mapping exists. |
| `additional_control_effort` | Recovery-arm effort minus a matched declared baseline over equivalent intervals. | Same as `recovery_control_effort` | Paired difference; report both raw values. | A baseline that terminates early has a shorter exposure interval, so interval matching must be explicit. |
| `recovery_delta_v_proxy` | Integrated thrust-acceleration magnitude over the recovery interval. | `m/s` proxy | Lower is cheaper for otherwise equivalent outcomes. | Still a proxy when mass depletion and propulsion efficiency are absent. |
| `retreat_path_length` | State-space or physical path length from retreat initiation to the declared retreat target. | Meters for positional path, or declared normalized units | Lower is cheaper only when retreat targets and outcomes match. | Geometric distance alone omits effort, time, and orbital energy. |
| `retreat_energy_proxy` | Change or expenditure in a declared energy-like quantity during retreat. | Joules only with a physical model; otherwise declared normalized energy units | Report signed and absolute changes separately. | Do not call a normalized orbital quantity physical energy without dimensional validation. |

### Performance Cost

| Field | Definition | Units | Comparison rule | Limitation |
| --- | --- | --- | --- | --- |
| `extra_steps` | Recovery-run steps minus matched baseline steps to the same comparison event or terminal outcome. | Steps | Positive means longer execution. | Longer duration may represent useful recovery time or unproductive stalling; interpret with outcome. |
| `delay_to_crossing` | Recovery-run first crossing step minus matched baseline first crossing step. | Steps or seconds | Lower is better when both arms cross. | `null` if either required crossing is absent; do not encode no crossing as an arbitrary large delay. |
| `recovery_time` | Steps or time from recovery initiation to the declared recovery target. | Steps or seconds | Lower is better among runs reaching the same target. | `null` when recovery does not occur. |
| `final_orbital_error_delta` | Paired changes in final radius error, radial-velocity error, and tangential-velocity error. | Meters and `m/s`, reported as separate components | Lower absolute errors are generally better under a declared task phase. | Do not collapse components without justified normalization. A lower error without crossing may remain diagnostic. |
| `lost_task_progress` | Decrease in a predeclared task-progress measure caused during recovery. | Units of the declared progress measure | Lower is better. | Requires a monotone, task-relevant progress definition; closest approach alone is insufficient. |
| `task_abandoned` | Whether the decision intentionally stops pursuit of the current task objective. | Boolean plus reason | Report, do not rank as automatic failure or success. | Abort may be operationally appropriate while still not being task recovery. |

### Operational Cost

| Field | Definition | Units | Comparison rule | Limitation |
| --- | --- | --- | --- | --- |
| `intervention_count` | Number of nominal proposals not executed unchanged. | Count | Lower is less intrusive for equivalent hazard and task outcomes. | Modifications, switches, and terminations may have different costs despite equal counts. |
| `consecutive_intervention_duration` | Duration of each uninterrupted intervention sequence; report maximum and distribution. | Steps or seconds | Shorter generally indicates faster return to nominal authority. | A short sequence can still cause failure; duration is burden, not quality. |
| `controller_switch_count` | Number of transitions between nominal, recovery, retreat, or degraded controllers. | Count | Lower is simpler for equivalent outcomes. | A necessary switch may be beneficial; chattering requires separate detection. |
| `re_observation_count` | Number of explicit observation refresh decisions during recovery. | Count | Interpret with trust and outcome. | More observations may improve decisions while increasing latency. |
| `termination_outcome` | Structured record of termination reason, authority, step, remaining resources, and whether an acceptable abort target was reached. | Categorical fields plus associated units | Report components; do not assign a universal scalar penalty. | A single `termination_cost` number would hide mission-specific consequences. |
| `termination_cost_components` | Vector containing elapsed time, resources consumed, task-abandonment status, lost progress, and acceptable-abort status at termination. | Mixed units, reported component by component | Compare only like-for-like termination objectives and baselines. | No universal scalar ordering exists; mission-specific weighting would require a separate declared contract. |

Task abandonment, retreat, and termination must be visible outcomes. They must not disappear inside a favorable hazard count or an arbitrary combined cost.

## 5. Intervention-Burden Metrics

Let `N_eval` be valid monitor evaluations, `N_veto` be veto decisions, and `N_allow` be allow decisions during a declared interval.

| Metric | Definition | Missing / edge rule | Interpretation |
| --- | --- | --- | --- |
| `intervention_rate` | `N_veto / N_eval` for the Final Veto allow/veto profile; future modified-action policies should use interventions divided by valid proposals. | `null` when `N_eval = 0`. | Fraction of evaluated nominal proposals not executed unchanged. |
| `allow_rate` | `N_allow / N_eval`. For a valid binary Final Veto stream, this equals `1 - intervention_rate`. | `null` when `N_eval = 0`. | Fraction of valid proposals executed unchanged. |
| `first_intervention_step` | First step at which nominal and executed authority diverge. | `null` when no intervention occurs. | How early nominal authority is lost. |
| `last_intervention_step` | Last step with an intervention in the declared interval. | `null` when no intervention occurs. | How late intervention remains active. |
| `longest_consecutive_veto_streak` | Maximum number of adjacent valid veto decisions without an allow between them. | Zero when valid evaluations occur but no veto occurs; `null` if no evaluation stream exists. | Persistence of veto-only behavior. |
| `veto_segment_count` | Number of contiguous veto segments. | Zero when no veto occurs. | Distinguishes one sustained refusal from repeated allow/veto chattering. |
| `monitor_induced_horizon_extension` | Matched monitor-on terminal step minus monitor-off terminal step. | Requires a complete matched pair; otherwise `null`. | Positive values may be recovery opportunity or unproductive stalling. Report terminal outcomes with it. |
| `action_suppression_duration` | Count or elapsed time for steps where executed action differs from the nominal proposed action. | Requires both actions. Do not infer solely from terminal outcomes. | Duration for which nominal control authority is suppressed. |
| `useful_progress_per_intervention` | Change in a predeclared useful-progress measure divided by intervention count. | Exploratory; `null` when the progress measure is undefined or intervention count is zero. | May reveal whether interventions restore progress, but only after a valid progress denominator is declared. |
| `hazard_reduction_per_intervention` | Paired reduction in declared hazard outcomes divided by intervention count over a predeclared set. | Exploratory; requires complete pairs and a hazard-positive baseline. | Measures intervention burden relative to aggregate hazard reduction, not causal value of each individual veto. |

The last two metrics are exploratory diagnostics. They must not be reported until their numerator, denominator, interval, and aggregation level are fixed. In particular, five avoided pair-level hazards cannot be causally assigned to 499877 individual vetoes without a more detailed counterfactual model.

## 6. Recovery Outcome Taxonomy

Recovery outcomes remain separate from the controlled terminal failure label and from simulator-defined success.

| Proposed outcome code | Required condition | Interpretation |
| --- | --- | --- |
| `hazard_avoided_task_recovered` | The declared hazard is absent over the evaluation interval, the run is valid, and the predeclared task-recovery criterion is met within `H_rec`. | Positive recovery evidence for the declared case and policy. A paired avoided-failure claim still requires a hazard-positive counterfactual. |
| `hazard_avoided_task_stalled` | The declared hazard is absent, but no task-recovery target is reached before the recovery horizon or episode horizon. | Final Veto v0 stress behavior. Hazard avoidance without task recovery. |
| `hazard_avoided_through_retreat` | The hazard is absent and a declared retreat target is reached, but the original task-recovery target is not yet reached. | Recovery of optionality or retry readiness, not original task completion. |
| `hazard_avoided_through_termination` | The hazard is absent because execution is terminated before exposure, and no declared task-recovery target is reached. | Explicit refusal or abort. Do not count as task recovery unless acceptable abort is itself the predeclared objective. |
| `hazard_not_avoided` | The declared realized hazard occurs despite the recovery decision. | Recovery or assurance failure for the declared hazard. |
| `invalid_evaluation` | State, action, prediction, monitor result, or simulation validity is insufficient to assign a valid allow/recovery outcome. | Excluded from valid allow/veto success counts and routed to audit. |
| `recovery_action_caused_new_failure` | A recovery action avoids or does not encounter the original hazard but causes a different declared terminal or safety failure. | Failure-mode substitution with a new adverse mechanism; report both precursor and terminal labels. |
| `preservation_success` | A known-success case retains its declared crossing, recoverability, and simulator outcome under the recovery architecture. | Regression preservation evidence, not new task-generation evidence. |
| `blocked_nominal_success` | A matched nominal arm succeeds or reaches recoverability and the intervention arm loses that outcome. | Safety-performance tradeoff or regression. |
| `monitor_not_exercised` | The declared hazard does not occur in the matched no-monitor arm. | No hazard-reduction claim is permitted for that pair. |

One run may require a terminal label, recovery outcome, and simulator-success field simultaneously. For example, a run may have:

```text
recovery_outcome = hazard_avoided_task_stalled
terminal_label = no_crossing
final_simulator_success = false
```

## 7. Proposed Recovery Success Definition

For a declared recovery attempt beginning at step `t_rec`, define preliminary recovery success as:

```text
recovery_success_v0 =
    declared_hazard_avoided
    and not invalid_simulation
    and not invalid_recovery_evaluation
    and task_recovery_criterion_met_within_H_rec
```

For the current orbital testbed, the default task-recovery criterion should be:

```text
target-radius crossing occurs
and the crossing reaches the declared Phase34-compatible recoverable condition
```

A future experiment may substitute another criterion only if it is frozen before evaluation, such as reaching a declared retry-ready state or acceptable abort state. The result must then be named `retreat recovery`, `retry readiness`, or `acceptable abort`, not orbital-insertion recovery.

`final_simulator_success` remains a separate simulator-defined field. It may be reported alongside `recovery_success_v0`, but neither field should be inferred from the other without an explicit benchmark rule.

`No overspeed` alone is insufficient because it says nothing about crossing, recoverability, useful progress, resource exhaustion, other hazards, or indefinite inactivity. A controller that executes no useful action for the entire horizon may avoid overspeed and still fail recovery.

For claims that a policy caused recovery, use a matched comparison with identical case, initial state, nominal controller, constants, seed, and horizon. A standalone successful trajectory establishes an outcome, not the counterfactual effect of the recovery policy.

## 8. Recovery Decision Interface

The future decision layer may emit the following outputs. This is an interface specification, not an implementation.

| Decision output | Minimum evidence before selection | Required record |
| --- | --- | --- |
| `continue_nominal_control` | Valid state; acceptable trust; declared hazards below thresholds over the available prediction horizon; no evidence that nominal continuation has exhausted recoverability or resources. | Nominal action, predictions, margin vector, thresholds, authority, and why continuation remains justified. |
| `adjusted_nominal_action` | A bounded modification is predicted to improve a named margin or avoid a named hazard while preserving nominal task intent; the modified action passes final veto evaluation. | Original action, adjusted action, adjustment rule, predicted effects, realized effects, and modification authority. |
| `bounded_corrective_action` | A recovery action contract applies; correction authority is adequate under the declared model; a recovery target, duration, stop condition, and failure condition exist; final veto allows execution. | Recovery-action ID, applicability evidence, target, horizon, cost budget, predicted margin trajectory, and return condition. |
| `retreat` | A declared retry-ready or lower-commitment target is predicted reachable; local correction is inadequate or too costly; retreat resources remain; retreat does not violate a declared hard constraint. | Retreat target, expected path, resource estimate, retry condition, abort condition, and decision authority. |
| `safe_mode` | A specifically defined bounded hold or degraded mode exists; its constraints and exit conditions are declared; active continuation has insufficient evidence. | Mode invariant or operating envelope, entry reason, maximum duration, observation plan, exit conditions, and fallback if the mode degrades. |
| `terminate` | No acceptable continuation, correction, retreat, or bounded mode is available within declared constraints, or a hard stop condition is reached. | Stop condition, evidence exhausted, authority, remaining resources, terminal label, and whether an acceptable abort objective was reached. |

The Decision Manager may select among admissible options, while Runtime Assurance may reject a proposed option. Recovery authority must never be inferred from a controller merely producing an action.

## 9. Evidence Levels

Every recovery metric and decision input should carry an evidence-level field.

| Evidence level | Definition | Allowed use | Prohibited interpretation |
| --- | --- | --- | --- |
| `measured` | Computed from a realized state, action, event, or artifact under a declared measurement rule. | Outcome reporting and calibration targets. | Measurement in a simulator is not hardware validation or proof of future behavior. |
| `one_step_predicted` | Computed from one transition using the declared current state, action, and exact or identified one-step model. | Immediate action screening and prediction-realization comparison. | Does not establish multi-step recoverability or invariant safety. |
| `multi_step_predicted` | Computed over more than one future transition under a declared policy, model, horizon, and assumptions. | Candidate recovery ranking and horizon-aware margin estimation. | Not a guarantee; model drift and policy mismatch must be reported. |
| `heuristic` | A hand-designed proxy or rule with a stated rationale but no outcome calibration. | Diagnostic ranking and hypothesis generation. | Must not authorize a formal safety claim or be presented as a validated recovery probability. |
| `unvalidated_assumption` | A quantity, threshold, model relation, or action effect assumed for design but not yet checked against evidence. | Explicit experiment planning only. | Must not be treated as measured, predictive, or guaranteed. |

Required evidence metadata should include:

- `evidence_level`;
- source state or artifact reference;
- model or computation ID;
- prediction horizon;
- controller or recovery-policy ID;
- target and constraint definitions;
- units and normalization;
- uncertainty assumptions;
- calibration status;
- missing-value reason.

Heuristic and unvalidated fields may inform what to test. They must not silently acquire runtime authority or formal-safety meaning.

## 10. First Nonformal Experiment Proposal

### Purpose And Status

Design one bounded, nonformal branching comparison to determine whether simple recovery-action categories produce measurably different post-veto behavior. Do not use it for a benchmark, preservation, or positive architecture claim.

Proposed experiment ID:

```text
recovery_metrics_single_case_nonformal_v0
```

### Frozen Case Selection

Use exactly one case from the frozen diagnostic stress manifest:

```text
case_id: phase35_radial_energy_push_overspeed_stress_v0__r0_0p98__angle_150__thrust_8000
r0_over_target: 0.98
initial_velocity_angle_deg: 150
thrust_scale: 8000
seed: 0
nominal upstream variant: radial_energy_push
post-cross context: phase34 radius_priority
```

This case is selected by stable manifest order, not by post-hoc comparison among the five stress cases. Its frozen result is selection context only. Any future recovery run must use a new nonformal artifact directory and must not modify `analysis/final_veto_ablation_v0/`.

### Common Branch Point

All candidate branches should share the same initial state, nominal controller, simulator constants, seed, and nominal prefix. Branch at the first valid Final Veto decision that rejects a nominal proposal under the unchanged strict `speed_ratio > 1.90` condition.

Record the branch state once and verify that every branch starts from the identical canonical state and configuration hash. This comparison must not tune the branch point separately for each action.

### Candidate Responses

The values below are fixed preliminary experiment-design choices, not validated control constants.

| Branch | Preliminary action contract | Intended diagnostic mechanism | Outcome interpretation |
| --- | --- | --- | --- |
| `zero_action_reference` | Execute normalized action `(0.0, 0.0)` for one step, then re-evaluate. | Reproduce the v0 fallback mechanism in a new explicitly nonformal comparison. | Expected reference for hazard avoidance with possible stalling; not assumed in advance. |
| `bounded_velocity_braking` | Propose a normalized action of magnitude `0.25` opposite the instantaneous velocity direction for one step, then re-evaluate. If speed is zero, the action is `(0.0, 0.0)`. | Test whether bounded kinetic-energy reduction restores overspeed headroom more efficiently than coast. | Heuristic action; no recovery or safety property is assumed. |
| `bounded_tangential_correction` | Propose a normalized tangential action of magnitude `0.25` opposite the signed tangential-speed error relative to target circular speed for one step, then re-evaluate. If the error is zero, use `(0.0, 0.0)`. | Test whether correcting tangential-speed error preserves more orbital task structure than velocity-opposed braking. | Heuristic action; requires declared radial/tangential frame and sign convention before implementation. |
| `explicit_abort` | On the first veto, execute no further transition and terminate the attempt with an explicit abort decision and reason. | Provide a bounded refusal baseline with zero post-decision action exposure. | Hazard avoidance through termination, never task recovery under the orbital criterion. |

Every proposed physical action must be evaluated by the unchanged final veto before execution. If a candidate action is rejected, mark `recovery_action_rejected` and end that branch for this first diagnostic. Do not recursively substitute another unlogged action.

### Horizons

- Maximum total episode horizon: 100000 steps, matching the existing diagnostic context.
- Maximum recovery horizon after the first veto: 10000 steps.
- Stop at the first recovery success, declared failure, explicit abort, invalid evaluation, recovery-horizon exhaustion, or total-horizon exhaustion.

The 10000-step recovery horizon is a preliminary design threshold selected to bound this first diagnostic. It is not evidence that all feasible recoveries must occur within that duration.

### Required Metrics

Record at minimum:

- realized overspeed, instability, unsafe-state, and invalid-simulation outcomes;
- crossing, recoverable crossing, and simulator-defined success separately;
- recovery outcome taxonomy code;
- all recovery-margin candidate components that can be computed without guessing;
- first and minimum overspeed headroom;
- predicted and realized margin trajectories kept separate;
- recovery time and horizon exhaustion;
- control effort and delta-v proxy;
- final radius, radial-velocity, and tangential-velocity errors separately;
- intervention rate, allow rate, first and last intervention steps;
- longest veto streak, veto segments, and action-suppression duration;
- branch terminal label and manual-audit note;
- task abandonment and termination reason;
- action parameters and evidence levels.

### Preliminary Success Conditions

A branch is a task-recovery success only if:

1. realized overspeed does not occur after the branch point;
2. no invalid simulation or invalid recovery evaluation occurs;
3. target-radius crossing occurs;
4. the trajectory reaches the declared Phase34-compatible recoverable condition within 10000 recovery steps.

Report simulator-defined success separately.

### Preliminary Failure And Diagnostic Conditions

- Realized overspeed: `hazard_not_avoided`.
- Instability, unsafe state, or another terminal mechanism caused after recovery action: `recovery_action_caused_new_failure` with the controlled terminal label.
- No recovery target by 10000 recovery steps: `hazard_avoided_task_stalled` if the hazard was absent; otherwise the applicable failure outcome.
- Explicit abort: `hazard_avoided_through_termination` only if no hazard occurred before termination.
- Invalid state, prediction, action, or simulation: `invalid_evaluation` or `invalid_simulation`, not failure avoidance.
- Candidate rejected by final veto: diagnostic `recovery_action_rejected`; do not count as executed recovery.

### Logging Requirements

- New nonformal manifest with implementation commit and `is_formal_experiment=false`.
- One result row per branch using Result Schema v1 names where applicable.
- Decision Log-compatible event for the initial veto, every selected recovery action, every rejected recovery action, every mode change, and termination.
- Nominal proposed action, recovery proposed action, final executed action, predictions, and realized next state kept distinct.
- Evidence level, units, horizon, target, and model ID for every margin.
- Compact intervention segments plus exception and terminal events.
- Canonical case/configuration hashes proving the common branch state.
- No output under the frozen Final Veto directory or protected Phase34-37 directories.

### Why The Experiment Remains Nonformal

- It uses one selected stress case and provides no distributional evidence.
- It does not rerun the eight-case preservation set.
- The action magnitude `0.25` and recovery horizon 10000 are unvalidated design choices.
- Braking and tangential actions are heuristic, not calibrated recovery policies.
- No uncertainty, trust, or multi-horizon model is validated.
- Comparing four branches on one case can rank hypotheses only for that case.

Do not tune the actions on the other four frozen stress cases before reporting this single-case diagnostic. Any later expansion requires a new predeclared manifest and preservation gate.

## 11. Cross-Embodiment Mapping

The shared object is a measurement and decision structure, not a universal metric value or low-level controller.

| Embodiment | Intermediate event | Hazard / margin examples | Recovery target | Candidate decisions | Domain-specific boundary |
| --- | --- | --- | --- | --- | --- |
| Orbital control | Target-radius approach or crossing | Overspeed headroom, radial reversibility, tangential correction authority, horizon slack | Phase34-compatible recoverable crossing, retry-ready transfer state, or declared abort state | Continue, adjust thrust, select recovery action, retreat in transfer geometry, terminate | Orbital dynamics, target circular speed, thrust mapping, and crossing criteria are spacecraft-specific. |
| Plug insertion | First or ambiguous contact | Force headroom, alignment error, withdrawal authority, jam likelihood, retry budget | Stable insertion, locally correctable contact, or retry-ready withdrawn pose | Continue insertion, local correction, re-observe, retreat, abort | Contact force, friction, compliance, pose uncertainty, and insertion depth have no direct orbital equivalent. |
| Mobile robot or drone | Waypoint, landing-zone detection, slip, or obstacle event | Braking distance, attitude margin, obstacle clearance, localization trust, return-energy margin | Stable hover/stop, collision-free route, recoverable attitude, return-ready state | Continue, slow, replan, hover/stop, retreat/return, land or terminate mission | Vehicle dynamics, terrain or airflow, sensor latency, and energy limits require separate models and validation. |

Transferable principles are:

- hazard avoidance and task recovery are distinct;
- margins are relative to available control authority and horizon;
- intervention burden must be measured;
- retreat or termination may preserve value without completing the original task;
- evidence level must constrain decision authority;
- outcome and decision logs must remain separate.

Metric thresholds, units, action definitions, and validation evidence do not transfer automatically between embodiments.

## 12. Non-Claims

Recovery Metrics v0 establishes no:

- formal safety guarantee;
- validated recovery controller or recovery policy;
- proof that any proposed recovery margin is a control invariant;
- proof that the candidate margin vector predicts recovery;
- proof that the proposed one-case actions are effective;
- hardware or real-spacecraft validation;
- sim-to-real transfer;
- universal cross-domain recovery metric;
- drone, mobile-robot, or manipulation validation;
- deployment-readiness claim;
- change to the frozen Final Veto threshold, fallback, cases, or evidence.

## 13. Open Questions

1. What exact state predicate should define a Phase34-compatible recoverable target when recovery begins before target-radius crossing?
2. Which margin components can be measured directly, and which require a new multi-step predictor?
3. How should required correction be estimated without embedding an unvalidated recovery controller inside the metric?
4. What resource proxy is scientifically defensible before mass depletion and propulsion efficiency are modeled?
5. Is a 10000-step recovery horizon long enough to distinguish bounded recovery from stalling?
6. Does velocity-opposed braking destroy orbital geometry that tangential correction could preserve?
7. What retreat target represents genuine retry readiness in the current orbital task?
8. How should prediction-realization error reduce recovery authority?
9. Which margin components predict recovery outcomes across cases rather than only within one trajectory?
10. What preservation test is required before a recovery policy advances beyond a one-case nonformal diagnostic?

## Preliminary Use Rule

Recovery Metrics v0 may be used to design logging, offline calculations, and a one-case nonformal comparison. It must not be used to claim recovery until a predeclared task-recovery target is reached under valid evidence. Hazard avoidance, task recovery, preservation, simulator success, intervention burden, and cost remain separate reported dimensions.
