# ROAHM Contact Recoverability Notes

## Status

Week 6 research note.

Completed: 2026-07-10

Scope: cross-embodiment recoverability and decision-making principles, grounded primarily in the existing spacecraft testbed and informed by contact-rich plug insertion as a second research direction.

This document is an abstraction note. It does not implement robotics, contact simulation, Runtime Assurance, Final Veto, sensors, ROS, Isaac Sim, Franka control, or hardware interfaces. It uses only intentionally stored repository material and general conceptual reasoning. It includes no private lab code, unpublished lab data, or private mentor discussions.

`ROAHM` in the title identifies the requested contact-recoverability research framing. It is not attribution of any specific private lab method, result, discussion, or dataset.

The central bridge is:

```text
Spacecraft: target-radius crossing is not recoverable orbital insertion.
Plug insertion: first contact is not successful insertion.
Perception: detection or pose estimation is not task completion.
Autonomy: an intermediate event is not sufficient evidence to continue blindly.
```

## Purpose And Scope

The purpose of this note is to answer:

```text
What recoverability, failure-recognition, trust, retry, retreat, re-observation,
and veto concepts can transfer from contact-rich plug insertion into a general
autonomous-control architecture?
```

The useful transfer is structural. Spacecraft and contact-rich manipulation are the first two concrete viewpoints in this note, but the intended architecture is broader. Physically embodied autonomous systems contain intermediate events that change the control problem without establishing task completion. They therefore benefit from event review, mechanism-aware failure labels, recoverability assessment, trust-aware authority, explicit retry and retreat criteria, and decision evidence.

The transfer is not physical equivalence. Every embodiment has its own state space, observations, failure mechanisms, time horizon, safety constraints, and evidence requirements.

## Repository Identity Guardrail

The long-term project identity is:

```text
A cross-embodiment, recoverability-aware autonomous-control framework for
physical systems operating under uncertainty, limited resources, and safety
constraints.
```

The simplified 2D spacecraft simulator is currently the primary implemented and evidence-supported research testbed. Contact-rich robotic manipulation provides a second conceptual and practical research direction.

Future embodiments may include:

- spacecraft,
- drones and aerial robots,
- robotic manipulators,
- legged and animal-like robots,
- ground mobile robots,
- autonomous marine vehicles such as AUVs and ROVs,
- planetary rovers,
- other physically embodied autonomous systems.

These are future applications of the same general architecture, not domains already validated by current repository evidence. The implementation can remain spacecraft-centered while the architecture and evaluation questions are domain-general.

The contact notes may influence:

- evaluation concepts,
- failure-taxonomy concepts,
- decision-evidence concepts,
- future sensor-abstraction concepts,
- runtime-assurance hypotheses.

They do not erase the spacecraft research problem, turn the repository into a collection of unrelated robotics demos, or weaken the requirement for domain-specific evidence.

The repository evidence inspected for this note includes the benchmark, taxonomy, result schema, regression policy, decision architecture, decision-log schema, public hardware/vision roadmap, recoverability formalism, and public concept-to-metric notes. No external private material was inspected or reproduced.

## Cross-Embodiment Control Framework

The shared high-level loop is:

```text
observation or belief estimation
-> event detection
-> trust assessment
-> recoverability assessment
-> decision manager
-> controller or planner selection
-> runtime assurance or veto
-> physical action
-> result, failure, and decision logging
```

The framework does not attempt to use one controller across every robot. It attempts to use one decision structure for determining when a controller may continue, when recovery remains possible, when another controller is needed, and when the system should refuse unsafe continuation.

The shared object is not one universal low-level controller. The shared object is a general control and decision architecture.

Each embodiment still requires different:

- dynamics,
- sensors and belief estimators,
- actuators,
- event definitions,
- safety thresholds,
- recovery actions,
- resource models,
- success criteria.

The architecture is shared, but the physics, thresholds, benchmarks, and validation evidence remain domain-specific.

## Cross-Domain Recoverability Table

All non-spacecraft rows below are conceptual future embodiments, not systems currently implemented or validated in this repository.

| Embodiment | Intermediate event | Why insufficient | Recoverability question | Possible decisions |
| --- | --- | --- | --- | --- |
| Spacecraft | Target-radius crossing | The crossing state may be dynamically unusable. | Can downstream control stabilize insertion under the declared horizon and constraints? | Continue, switch controller, stabilize, abort |
| Drone or aerial robot | Waypoint arrival or landing-zone detection | Wind, attitude, battery, localization, or obstacle state may make continuation unsafe. | Can the mission continue, replan, return, or land safely? | Replan, hover, return, land, abort |
| Robotic manipulator | First contact | Contact may be misaligned, jammed, or misleading. | Can local correction complete insertion safely? | Continue, retry, re-observe, retreat, veto |
| Legged or animal-like robot | Foot contact, slip, or gait transition | Support may be unstable and the next step may lose balance. | Can balance and locomotion recover within support and joint limits? | Change gait, slow, reposition, recover, stop |
| Ground mobile robot | Waypoint, obstacle detection, or traction-loss event | Terrain, localization, or wheel traction may invalidate the route. | Can the robot replan or recover mobility without exhausting energy or entering hazard? | Stop, re-observe, reroute, reverse, safe mode |
| Marine exploration vehicle | Target depth, sonar event, or communication loss | Current, localization drift, pressure, energy, or communication state may invalidate the plan. | Can the vehicle continue and still surface or return safely? | Station keep, replan, reverse course, surface, abort |
| Planetary rover | Terrain transition or wheel slip | Mobility may be degrading while communication and energy are constrained. | Can the rover escape or reroute without unacceptable resource loss? | Backtrack, reroute, reduce motion, safe mode, wait |

The table transfers decision structure only. It does not transfer spacecraft metrics, contact thresholds, or experimental claims into the other embodiments.

## Why Crossing Is Not Insertion

A target-radius crossing is a geometric event. The state at or after crossing may still have radial velocity, tangential velocity, synchronization, safety, or controller-handoff errors that make continued control unsuccessful or unrecoverable.

The protected spacecraft evidence demonstrates this distinction:

- The reduced Phase31-style reference produced `8 / 24` crossings and `0 / 24` recoverable crossings.
- Phase34 `radius_priority` produced the same `8 / 24` crossings and converted all eight crossing-producing cases into recoverable crossings.

The scientific result is post-cross recoverability improvement, not expanded crossing generation.

## Why Contact Is Not Insertion

First contact only establishes that a physical interaction occurred. It does not establish that the plug is aligned, progressing, safely loaded, locally correctable, seated, or locked.

Conceptually, first contact may be:

- aligned and useful,
- an edge or lateral contact,
- sliding,
- prematurely triggered by pose error,
- jammed,
- force-limited,
- ambiguous because observation trust is low.

The post-contact question is therefore not only whether contact happened. It is whether the observed contact state remains recoverable under the available observations, controller, correction budget, safety limits, retry options, and retreat options.

## Shared Structural Pattern

| Domain | Intermediate event | Why it is insufficient | Recoverability question |
| --- | --- | --- | --- |
| Spacecraft | Target-radius crossing | The crossing state may be dynamically unrecoverable. | Can downstream control stabilize and complete the declared task? |
| Plug insertion | First contact | Contact may be misaligned, jammed, unstable, or misleading. | Can local correction still achieve insertion safely? |
| Perception | Detection or pose estimate | The estimate may be uncertain, biased, stale, or inconsistent. | Is the estimate trustworthy enough to authorize the next action? |
| Learning | Low training loss or high reward | The offline metric may not improve rollout behavior. | Does the policy improve recovery under the same benchmark without regressions? |

The shared sequence is:

```text
approach -> intermediate event -> event review -> recovery or continuation -> stable outcome
```

For spacecraft this can be read as:

```text
transfer -> target-radius crossing -> post-cross assessment -> synchronization -> recoverable state
```

For plug insertion this can be read conceptually as:

```text
approach -> first contact -> contact classification -> local correction -> stable insertion
```

The common research principle is that the event changes what must be measured and decided. It does not grant automatic permission to continue.

## Important Domain Differences

- Orbital motion is continuous, long-horizon dynamics; plug insertion is contact-rich and may be discontinuous or hybrid.
- Spacecraft state can be fully available in the current simulator; robotic state is often only partially observed.
- Contact forces have no direct equivalent in the current spacecraft simulator.
- Target-radius crossing is a geometric state event; contact is a physical interaction event.
- Recoverability horizons and resource constraints differ substantially.
- Contact-rich robotics may involve occlusion, calibration error, latency, compliance, backlash, and unmodeled friction.
- A safe robot retreat may be a short geometric withdrawal; a spacecraft retreat may require a dynamically feasible trajectory and resource budget.
- Contact classification may depend on force, tactile, vision, and motion evidence; current spacecraft evidence uses simulator state and derived orbital metrics.
- Spacecraft claims must remain based on spacecraft evidence. Robotics claims must remain based on robotics evidence.

These differences prevent direct transfer of thresholds, terminal criteria, safety metrics, or controller conclusions.

## Conceptual Embodiment Examples

The examples in this section show how the same decision architecture would require different evidence and recovery actions. They are conceptual only.

### Drone Or Aerial Robot

- Event: waypoint arrival or landing-zone detection.
- Uncertainty: wind, visual localization, GPS degradation, obstacle motion, and battery state.
- Recovery: hover, re-observe, replan, return-to-home, or emergency landing.
- Veto hypothesis: reject an action predicted to exceed declared attitude, altitude, obstacle-clearance, or energy limits.

### Legged Or Animal-Like Robot

- Event: foot contact, slip, or gait transition.
- Uncertainty: terrain shape, friction, support state, actuator response, and body momentum.
- Recovery: balance correction, gait switch, body repositioning, slower locomotion, or stop.
- Veto hypothesis: block a step predicted to lose support or exceed declared joint or stability limits.

### Marine Exploration Vehicle

- Event: target depth, sonar detection, or communication loss.
- Uncertainty: current, localization drift, pressure, energy reserve, and communication availability.
- Recovery: surface, station keep, re-localize, reverse course, or abort the mission segment.
- Veto hypothesis: reject continuation that leaves insufficient energy or recoverability margin for a safe return or surfacing path.

Ground vehicles and planetary rovers would use the same decision structure with terrain, traction, route, communication, and energy evidence. Their recovery actions and safety criteria would still require separate domain models and validation.

## Contact Event Types

The following are conceptual event-taxonomy candidates, not events already measured by current hardware or simulation:

- `no_contact`
- `first_contact`
- `aligned_contact`
- `edge_contact`
- `lateral_contact`
- `jammed_contact`
- `sliding_contact`
- `insertion_progress`
- `stable_insertion`
- `contact_lost`
- `force_limit_risk`
- `pose_confidence_low`
- `unknown_contact`

These belong, if ever implemented, in a future contact-domain event or diagnostic schema. They should not be added directly as terminal labels in the spacecraft Failure Label Taxonomy v0.

## Contact-Phase Abstraction

The following sequence is conceptual. It does not claim that the spacecraft simulator uses these physical phases.

| Phase | Observable evidence | Possible uncertainty | Possible failure mechanisms | Possible recoverability status | Appropriate decisions |
| --- | --- | --- | --- | --- | --- |
| `approach` | Relative pose trend, commanded motion, distance estimate | Pose bias, latency, target motion | Missed approach, unsafe speed, stale estimate | `not_yet_committed`, `recoverable_with_retry` | Continue cautiously, re-observe, retry, abort |
| `pre_contact_alignment` | Lateral and angular error estimates, approach direction | Calibration error, occlusion, estimator inconsistency | Off-axis approach, bad orientation | `recoverable`, `marginal`, `unknown_due_to_low_trust` | Continue, reduce action, re-observe, retreat |
| `first_contact` | Force or motion discontinuity, contact cue, velocity change | False contact, delayed contact detection, ambiguous surface | Early contact, edge contact, lateral contact | `locally_recoverable`, `marginal`, `unsafe_to_continue` | Hold, classify, re-observe, retreat, veto |
| `contact_classification` | Force direction, motion response, pose/contact consistency | Sparse sensing, friction ambiguity, model mismatch | Wrong contact class, unknown contact | `unknown_due_to_low_trust`, `recoverable_with_retreat` | Re-observe, conservative continuation, retreat |
| `local_correction` | Error reduction, force change, measurable motion | Compliance and friction uncertainty | Correction increases force, oscillation, jam | `locally_recoverable`, `marginal` | Continue correction, modify action, switch controller, retreat |
| `insertion_progress` | Increasing depth, decreasing alignment error, bounded force | Depth bias, slip, false progress | Partial insertion, binding, contact loss | `recoverable`, `robustly_recoverable`, `marginal` | Continue, stabilize, re-observe, veto aggressive action |
| `stable_insertion` | Stable depth or seating evidence, bounded load, no reversal | Sensor bias, incomplete seating evidence | False seat, residual load, latent jam | `robustly_recoverable` only under declared criteria | Confirm, hold, complete, manual audit |
| `success_or_recovery` | Declared completion or recovery-target evidence | Criterion mismatch, stale status | Late failure, recovery not maintained | `recoverable`, `recoverable_with_retry`, or failed | Complete, retry, degrade goal, safe mode |
| `retreat_or_abort` | Safe withdrawal progress, reduced load, restored observation | Retreat path uncertainty, actuator limits | Withdrawal jam, lost object, resource exhaustion | `recoverable_with_retreat`, `irrecoverable_under_current_policy` | Retreat, safe mode, abort |

The key design point is that each phase has different evidence needs and different acceptable authority. Contact alone should transition into classification or event review, not directly into success.

## Recoverability-State Abstraction

Conceptual recoverability states:

| State | Meaning |
| --- | --- |
| `not_yet_committed` | The critical interaction has not occurred and multiple options remain. |
| `recoverable` | An acceptable continuation or recovery target is reachable under the declared policy. |
| `marginal` | Recovery appears possible but with little margin or high sensitivity. |
| `locally_recoverable` | A bounded local correction can restore progress without a full reset. |
| `recoverable_with_retry` | Progress requires returning to a retry-ready state and attempting again. |
| `recoverable_with_retreat` | Safe withdrawal remains available, but local continuation is not justified. |
| `irrecoverable_under_current_policy` | The available policy cannot reach an acceptable recovery target within the declared horizon and constraints. |
| `unsafe_to_continue` | Continuing would violate or predictably approach a declared safety boundary. |
| `unknown_due_to_low_trust` | Available observations or models cannot support a precise recoverability judgment. |

Recoverability is always relative to:

- available observations,
- available controller or policy,
- remaining action and resource budget,
- safety constraints,
- retry options,
- retreat options,
- time horizon.

It is not an absolute property of contact geometry or orbital state.

## Failure-Mode Mapping

| Contact-rich manipulation observation | General autonomy interpretation | Possible decision | Careful spacecraft analogy |
| --- | --- | --- | --- |
| No contact after expected approach | Expected event missing | Re-observe or retry | No target-radius crossing after planned transfer, without equating `no_contact` to `no_crossing`. |
| Contact occurs earlier than expected | Model, pose, or timing mismatch | Stop, re-observe, or retreat | An event occurs outside the expected state corridor or timing window. |
| High force with no insertion progress | Jam or unrecoverable local geometry | Retreat or abort | A control action consumes safety or resource margin without improving recoverability. |
| Small correction restores progress | Locally recoverable failure | Continue with corrected action | Post-cross synchronization restores a usable state, without physical equivalence. |
| Pose confidence becomes low | Trust degradation | Re-observe | State-estimate trust becomes insufficient for aggressive continuation. |
| Repeated retries make no progress | Retry exhaustion | Retreat, safe mode, or abort | Repeated transfer or recovery attempts fail within a declared budget. |
| Contact state is ambiguous | Insufficient evidence | Hold or re-observe | Event state cannot support a precise failure or recoverability label. |
| Force or action threshold is exceeded | Safety-boundary risk | Veto or safe mode | Overspeed, instability, saturation, or another declared simulated hazard triggers intervention. |

The analogy is about decision structure: expected event, evidence quality, recovery options, and safety authority. It does not assert that contact force and orbital speed are the same physical quantity.

## Runtime Decision Mapping

These are design hypotheses only.

### `continue`

Appropriate when:

- progress is measurable,
- trust is acceptable,
- safety margin remains,
- recoverability remains acceptable.

### `retry`

Appropriate when:

- the failure is local,
- the system can return to a retry-ready state,
- retry budget remains,
- evidence suggests another attempt may differ meaningfully.

### `retreat`

Appropriate when:

- local correction is unlikely to succeed,
- contact or force risk is rising,
- a safe withdrawal state remains available,
- retry requires resetting geometry.

### `re_observe`

Appropriate when:

- pose confidence is low,
- contact classification is ambiguous,
- expected and observed events disagree,
- sensor evidence is stale or conflicting.

### `safe_mode`

Appropriate when:

- active progress is suspended,
- uncertainty or risk is too high,
- a stable hold or low-action state exists.

### `abort`

Appropriate when:

- recoverability is exhausted,
- retry and retreat are unavailable or unsafe,
- resource or risk budget is exhausted,
- continuation would violate a hard safety condition.

### `veto_action`

Appropriate when a proposed action is predicted to worsen force, speed, instability, or recoverability beyond a declared threshold.

## Retry Versus Retreat

| Question | Retry is favored when | Retreat is favored when |
| --- | --- | --- |
| Can a retry-ready state be reached? | Yes, through a bounded reset. | No local reset exists, but safe withdrawal remains. |
| Is the failure local? | Evidence supports a correctable local error. | Evidence suggests a jam, wrong geometry, or rising hazard. |
| Will another attempt differ? | Observation, alignment, controller, or plan can change meaningfully. | Repeating would reproduce the same failure mechanism. |
| Is margin adequate? | Safety, resource, and retry budgets remain. | Margin is shrinking or a hard threshold is near. |
| Is trust adequate? | Trust supports the reset and next attempt. | Trust is too low for another commitment; withdrawal is safer. |
| What evidence is required? | Retry reason, changed condition, budget, and retry-ready state. | Retreat path, withdrawal feasibility, and fallback evidence. |

Retry must not mean repeating the same action without changed evidence. Retreat can be progress-preserving because it protects a future attempt or safe termination.

## Re-Observe Rules

Re-observation is justified when decision quality, rather than control authority alone, is the immediate bottleneck.

Triggers may include:

- low pose-estimation trust,
- ambiguous contact classification,
- disagreement between predicted and observed events,
- stale observations,
- conflicting sensing channels,
- an unexplained change in progress or force proxy,
- recoverability estimate trust below a declared level.

Re-observation should state what new evidence is expected and what decision it will unlock. It should not become an unlimited delay strategy that hides timeout or resource depletion.

## Veto And Safe-Action Hypotheses

A future veto hypothesis should evaluate a proposed action before execution and record both the blocked action and fallback.

Conceptual triggers include:

- predicted force-limit risk,
- predicted overspeed,
- instability risk,
- controller saturation persistence,
- recovery-margin loss,
- low trust for the proposed authority level,
- no safe retreat or retry branch after the action.

Possible responses include action reduction, controller switch, retreat, re-observation, safe mode, or abort.

A veto experiment must report blocked successes as well as avoided failures. A monitor that avoids all hazards by blocking all useful action has not demonstrated useful runtime assurance. No formal-safety claim follows from these hypotheses.

## Trust And Observation Mapping

| Trust category | Example evidence | Effect on decision authority |
| --- | --- | --- |
| Perception trust | Detection consistency, occlusion, stale image evidence | Low trust favors re-observation or reduced approach authority. |
| Pose-estimation trust | Innovation, covariance, cross-frame consistency | Low trust blocks precise or aggressive continuation. |
| Contact-classification trust | Agreement among force, motion, and pose cues | Low trust favors hold, re-observe, or retreat. |
| Dynamics-model trust | Predicted versus observed motion or contact response | Low trust reduces planner confidence and action magnitude. |
| Planner trust | Option feasibility and prediction error history | Low trust favors simpler fallback or manual audit. |
| Controller trust | Saturation, tracking error, regime validity | Low trust favors controller switch, retreat, or veto evaluation. |
| Recoverability-estimate trust | Calibration and prediction consistency | Low trust prevents aggressive continuation based on the estimate alone. |
| Hardware or actuator trust | Command-response mismatch, latency, fault flags | Low trust favors degraded goal, safe mode, or abort. |

Low trust should not automatically mean failure. It should reduce confidence in aggressive continuation and may lead to re-observation, action reduction, controller switch, retreat, conservative continuation, manual audit, or veto evaluation.

## Mapping Into Decision Log Schema v0

The existing general decision fields can represent contact-inspired decisions without adding robotics-specific runtime code:

| Decision Log field | Contact-inspired use |
| --- | --- |
| `event_detected` | Whether a conceptual contact or confidence event was observed. |
| `event_type` | Candidate event such as `first_contact`, `unknown_contact`, or `force_limit_risk`. |
| `state_summary` | Compact pose, contact, progress, and safety evidence. |
| `safety_level` | General safety assessment, not a formal proof. |
| `recoverability_level` | General recovery assessment relative to policy and budget. |
| `trust_flags` | Low estimator, contact-classification, controller, or hardware trust. |
| `decision_type` | Continue, retry, retreat, re-observe, safe mode, abort, or veto. |
| `decision_reason` | Existing general reason where possible, with detail in `manual_audit_note`. |
| `fallback_available` / `fallback_action` | Whether retreat, re-observation, safe mode, or another branch remains. |
| `veto_status` / `veto_reason` | Whether a proposed action was allowed, modified, or blocked. |

The more detailed conceptual states in this note do not silently extend Decision Log Schema v0. For example, a conceptual `recoverable_with_retreat` condition should use the existing general `recoverability_level=marginal`, with retreat feasibility recorded in `state_summary`, `fallback_action`, or `manual_audit_note`, until a declared schema revision exists.

All examples below are conceptual and non-hardware-validated.

### Example 1: First Contact, Continue

```text
event_detected=true
event_type=first_contact
state_summary=acceptable pose confidence; low contact force; measurable insertion progress
safety_level=nominal
recoverability_level=recoverable
decision_type=continue
decision_reason=unknown
veto_status=allow
manual_audit_note=current v0 reason enum has no contact-progress reason
```

### Example 2: Ambiguous Contact, Re-Observe

```text
event_detected=true
event_type=unknown_contact
state_summary=conflicting pose and contact evidence
trust_flags=low_estimator_trust
recoverability_level=unknown
decision_type=re_observe
decision_reason=low_trust
```

### Example 3: Jam Risk, Retreat

```text
event_detected=true
event_type=jammed_contact
state_summary=high force; no insertion progress; withdrawal remains feasible
safety_level=warning
recoverability_level=marginal
decision_type=retreat
decision_reason=safety_violation
fallback_available=true
fallback_action=retreat
```

### Example 4: Unsafe Proposed Action, Veto

```text
event_detected=true
event_type=force_limit_risk
state_summary=predicted force or geometry risk exceeds declared threshold
safety_level=critical
decision_type=veto_action
decision_reason=safety_violation
veto_status=retreat
veto_reason=proposed action exceeds declared contact-risk threshold
fallback_action=retreat
```

These examples illustrate decision evidence only. They do not establish measured contact capability, final-veto performance, hardware validation, or formal safety.

## Mapping Into Failure Label Taxonomy v0

The current controlled terminal taxonomy remains spacecraft-oriented and general. This note does not add robotics-specific terminal labels.

| Contact concept | Appropriate relationship | Constraint |
| --- | --- | --- |
| `no_contact` | Domain-specific event or diagnostic label for an expected event not occurring | Do not automatically map it to `no_crossing`. |
| `jammed_contact` | May support `unsafe_state`, `timeout`, or a future domain-specific diagnostic label | Terminal mapping requires declared criteria and evidence. |
| `force_limit_risk` | Safety precursor, decision reason, or veto reason | Risk prediction is not necessarily a realized terminal failure. |
| `unknown_contact` | Supports `unknown` or manual audit when evidence is insufficient | Do not force a precise contact mechanism. |
| `insertion_progress` | Precursor event | It is not terminal success. |
| `stable_insertion` | May correspond structurally to a recoverable or stabilized state | Only under a declared contact benchmark definition. |

The label layers should remain distinct:

- Transferable structural terminal labels: `success`, `timeout`, `unsafe_state`, `invalid_simulation`, `unknown`, when a domain contract defines them.
- Domain-specific precursor labels: `first_contact`, `insertion_progress`, `contact_lost`.
- Domain-specific diagnostic labels: `edge_contact`, `lateral_contact`, `jammed_contact`, `unknown_contact`.

Contact labels must not be inserted casually into Failure Label Taxonomy v0. A future domain extension would need its own definitions, priority rules, and evidence requirements.

## Mapping Into Result Schema v1

Transferable Result Schema v1 concepts include:

- event detection, represented by domain-specific event booleans such as current `crossed_target_radius`,
- event timing, represented by domain-specific event-step or event-time fields,
- state summary, represented by an event-state summary such as current `state_at_crossing_summary`,
- recoverability status, represented by current `recoverable_crossing` or a future domain equivalent,
- safety status,
- `terminal_label`,
- `precursor_labels`,
- `diagnostic_labels`,
- subset status,
- `accepted_as_progress` and `acceptance_reason`.

The current spacecraft field names, such as `crossed_target_radius` and `first_crossing_step`, remain tied to the spacecraft benchmark. A contact domain should use a separate schema or declared extension rather than silently reusing those names.

Potential future contact fields, listed only as conceptual examples, include:

- `contact_detected`
- `first_contact_step`
- `contact_mode`
- `contact_force_summary`
- `pose_confidence`
- `insertion_depth`
- `lateral_error`
- `retry_count`

None of these fields is implemented here.

## Concepts That Should Transfer

- Event success is not task success.
- Failure mechanism should be labeled.
- Recoverability should be evaluated after the event.
- Recoverability is policy-relative and resource-relative.
- Trust affects action authority.
- Retry requires a changed condition or a retry-ready state.
- Retreat may preserve future mission value.
- Controller selection should depend on regime and evidence.
- Runtime assurance may override a nominal controller under declared rules.
- Failures and decisions must be logged separately.
- Safe refusal is a legitimate autonomous behavior.
- Vetoes must report blocked success as well as avoided failure.
- Diagnostic improvements must not become benchmark claims.
- False progress must be rejected across domains.

## Transferable Versus Non-Transferable Concepts

| Transferable structural concept | Non-transferable domain detail |
| --- | --- |
| Intermediate-event review | Spacecraft thresholds cannot become robot contact thresholds. |
| Event-to-recoverability separation | Contact force cannot become an orbital safety metric. |
| Mechanism-aware labels | Contact labels cannot be inserted directly into the spacecraft terminal taxonomy. |
| Trust-aware action authority | Robot pose confidence cannot be assumed equivalent to spacecraft state confidence. |
| Retry-ready state requirement | A robot withdrawal motion is not an orbital retreat trajectory. |
| Retreat as a valid outcome | Insertion-depth criteria cannot become orbital completion criteria. |
| Logged veto and fallback evidence | Plug-insertion demonstrations cannot validate spacecraft control. |
| Regression protection | Phase34 controller conclusions cannot validate plug insertion. |
| Diagnostic-only treatment of proxies | Hardware behavior cannot establish simulator safety claims. |

## Concepts That Should Not Transfer Directly

Do not directly transfer:

- spacecraft thresholds into robot contact thresholds,
- robot contact forces into orbital safety metrics,
- insertion-depth criteria into orbital completion criteria,
- Phase34 controller conclusions into plug insertion,
- plug-insertion demonstrations into spacecraft validation,
- hardware behavior into simulator safety claims,
- domain-specific labels without a declared schema extension.

## Small Future Experiments Within Repository Scope

These are future designs only. They are not implemented in Week 6.

| Small experiment | Spacecraft-repo purpose | Required restraint |
| --- | --- | --- |
| Add synthetic observation noise to selected state variables | Test whether event and recovery decisions remain reliable under imperfect state | Do not claim sensor or hardware validation. |
| Add a delayed-observation wrapper | Test stale-state effects on crossing and recoverability | Keep controller and benchmark comparisons explicit. |
| Create hypothetical trust flags from estimation mismatch | Exercise Decision Log Schema v0 | Treat flags as synthetic and uncalibrated. |
| Record `re_observe` or `safe_mode` in example decision logs | Test decision-evidence completeness | Do not present logs as an implemented Decision Manager. |
| Design a small monitor that blocks one unsafe simulated action | Prepare a final-veto ablation | Report both avoided failures and blocked successes. |
| Compare monitor versus no-monitor on the existing benchmark | Measure safety-performance tradeoff | Preserve known Phase34 recoverable cases. |
| Measure avoided failures and blocked successes | Prevent trivial always-veto conclusions | Define counterfactual and thresholds before evaluation. |
| Define a general event-review state | Unify crossing and contact analogies at architecture level | Keep spacecraft metrics and contact metrics domain-specific. |

## Non-Claims And Evidence Limits

### Architectural Relevance Is Not Domain Validation

The document may claim that the same abstract architecture is relevant to multiple physically embodied autonomous systems. That architectural relevance does not equal domain validation.

This document does not establish:

- spacecraft-to-robotics transfer,
- robotics-to-spacecraft validation,
- tested drone or aerial-robot control,
- tested legged or animal-like locomotion,
- tested ground mobile-robot autonomy,
- tested marine autonomy,
- tested rover autonomy,
- universal controller performance,
- cross-domain empirical validation,
- real-robot performance,
- hardware readiness,
- sim-to-real transfer,
- formal runtime assurance,
- formal safety,
- contact-force measurement capability in the spacecraft simulator,
- a completed Decision Manager,
- a completed Final Veto system.

It also does not claim that current spacecraft controllers can solve contact-rich manipulation or that conceptual contact policies can solve orbital insertion.

The contact mappings are hypotheses for evaluation, logging, and architecture design. They are not experimental results. No private code, private lab data, unpublished evidence, or private mentor discussion is used as support.

## Conclusion

The long-term project is not about forcing every physical system to use the same low-level controller. It is about developing a shared autonomy framework for recognizing events, estimating trust and recoverability, selecting or switching controllers, refusing unsafe continuation, and preserving evidence about why each decision was made.

The spacecraft simulator is currently the strongest implemented and evidence-supported testbed. Contact-rich manipulation is a second major conceptual and practical research direction. Drones, aerial robots, ground mobile robots, legged or animal-like robots, autonomous marine vehicles, planetary rovers, and other embodied systems remain future applications that require their own models, sensors, thresholds, benchmarks, and validation.

## Week 7 Handoff Questions For Final-Veto Ablation Design

Week 7 should answer:

- What minimal simulated hazard should the first final-veto ablation monitor?
- What is the no-monitor baseline?
- What constitutes an avoided failure?
- What constitutes a blocked success?
- What is an unnecessary veto?
- What performance cost should be measured?
- Should the first monitor target overspeed, instability, low recoverability margin, or controller saturation?
- Which risk signal already exists in current artifacts and therefore requires the least refactoring?
- How will veto events be represented in Decision Log Schema v0?
- How will monitor results be represented in Result Schema v1?
- What claims are allowed from a rule-based monitor ablation?
- What claims remain prohibited without formal verification?

## Week 6 Completion Rule

Week 6 is complete when this document exists, the protected regression guard still passes, no historical evidence has been modified, and no private lab material has been added.
