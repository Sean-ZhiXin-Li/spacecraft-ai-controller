# Decision and Runtime Assurance Architecture

Status: architecture design document.

This document defines the missing decision-making layer for the long-term autonomy platform. The repository already points toward a stack containing sensors, perception, estimator, planner, controller, safety monitor, and hardware. Those components are necessary but incomplete. A runtime autonomy system also needs an explicit answer to:

- Who decides what the system is trying to do now?
- Who decides whether to continue, retry, retreat, re-observe, enter safe mode, or abort?
- Who can veto a controller?
- Who estimates uncertainty?
- Who changes mission goals?
- Who owns failure escalation?
- Who records why a decision was made?

The proposed answer is a Decision Manager coordinated with a Runtime Assurance layer. The Decision Manager owns task-mode selection and mission-level choices. Runtime Assurance owns safety constraints, final veto, and irreversible-failure prevention. The estimator owns uncertainty estimates. The planner proposes options. The controller executes local actions. The safety monitor and fault manager can restrict, override, or stop execution.

This is not flight software and not a claim of hardware readiness. It is an architecture for research experiments in recoverability-aware physical autonomy.

## 1. Core Principle

The current project's lesson is:

```text
crossing is not insertion
```

The runtime architecture generalizes this:

```text
event success is not authority to continue blindly
```

A system may detect a target, estimate a pose, cross a radius, make contact, or reach an approach corridor. Those events should trigger decision review:

- Is the state recoverable?
- Is uncertainty acceptable?
- Is the controller still trusted?
- Is the risk budget still available?
- Is the current goal still valid?
- Should the system continue, retry, retreat, re-observe, safe, or abort?

The autonomy stack should not treat every intermediate event as automatic permission to proceed.

## 2. Runtime Stack

The proposed runtime stack is:

```text
Mission Specification
        |
        v
Mission / Task Manager
        |
        v
Decision Manager <-----------------------------+
        |                                      |
        v                                      |
Planner / Policy Selector                      |
        |                                      |
        v                                      |
Controller --------------------------------+   |
        |                                  |   |
        v                                  v   |
Runtime Assurance / Final Veto -----> Fallback |
        |                                      |
        v                                      |
Hardware Interface                             |
        |                                      |
        v                                      |
Physical / Simulated Environment               |
        |                                      |
        v                                      |
Sensors -> Perception -> Estimator -> Monitor -+
```

The key distinction is that the planner proposes, the controller executes, the estimator assesses belief, the monitors assess risk and trust, and the Decision Manager chooses the current mode and allowed intent. Runtime Assurance can veto any proposed action or mode transition that violates safety or recoverability constraints.

## 3. Responsibility Boundaries

| Component | Owns | Does not own |
| --- | --- | --- |
| Mission / Task Manager | Mission goals, task phase graph, allowed degraded goals, completion criteria. | Low-level control actions, sensor fusion, final veto. |
| Decision Manager | Runtime mode choice: continue, retry, retreat, re-observe, safe mode, abort, controller switch. | Raw state estimation, low-level dynamics, direct actuator commands. |
| Planner | Candidate trajectories, subgoals, controller schedules, recovery options. | Final authority to execute unsafe plans. |
| Controller | Local action generation for the selected mode. | Mission goal changes, safety override authority. |
| Estimator | Belief state, uncertainty, confidence, estimator consistency. | Mission decisions, controller objectives. |
| Perception | Observations, detections, pose estimates, feature tracks. | Trust authority over task completion. |
| Safety Monitor | Constraint checks, hazard detection, safety margins. | Nominal mission optimization. |
| Recoverability Monitor | Recovery feasibility, recovery margin, event-to-continuation analysis. | Direct mission goal selection. |
| Trust Manager | Module trust scores, trust decay, confidence calibration. | Raw control commands or final mission authority. |
| Fault Manager | Fault labels, escalation, isolation, degraded capability model. | Nominal path planning. |
| Runtime Assurance / Final Veto | Action veto, mode veto, safe fallback command, abort trigger. | Optimizing nominal performance. |
| Hardware Interface | Command translation, rate limits, actuator feedback, hardware status. | Mission semantics. |
| Logger | Decision trace, evidence, metrics, artifacts. | Runtime authority. |

## 4. Who Decides?

The Decision Manager decides runtime intent. It chooses among:

- `continue`
- `retry`
- `retreat`
- `re_observe`
- `safe_mode`
- `abort`
- `degrade_goal`
- `switch_controller`

It makes those decisions using inputs from:

- Mission / Task Manager: current goal, phase, allowed transitions.
- Estimator: belief state and uncertainty.
- Planner: feasible options and predicted outcomes.
- Controller: current mode and action feasibility.
- Safety Monitor: safety margin and constraint violations.
- Recoverability Monitor: recovery margin and recovery target status.
- Trust Manager: trust scores and trust-decay events.
- Fault Manager: active faults and degraded capabilities.
- Hardware Interface: actuator, sensor, compute, power, latency, and health status.

The Decision Manager should not invent new safety permissions. It can choose among allowed actions and modes, but Runtime Assurance can reject the choice.

## 5. Who Can Veto?

Veto authority belongs to Runtime Assurance.

Runtime Assurance can veto:

- A proposed control action.
- A controller switch.
- A planner trajectory.
- A decision to continue.
- A decision to retry.
- A decision to approach closer.
- A decision to remain in a risky mode.

Runtime Assurance should veto when:

- The proposed action is predicted to enter an irreversible failure set.
- Safety margin falls below a hard threshold.
- Recovery margin falls below a hard threshold.
- Trust in required modules has decayed below the required level.
- Uncertainty exceeds the allowed bound for the current mode.
- Hardware is degraded below the capability required by the selected action.
- Risk budget is exhausted.

Veto does not mean "stop forever." A veto must produce one of:

- Modified safe action.
- Controller switch.
- Retreat.
- Re-observe.
- Safe hold.
- Abort.
- Degraded mission transition.

Every veto must be logged with a reason and the evidence that triggered it.

## 6. Who Estimates Uncertainty?

The Estimator owns uncertainty over state. Perception owns uncertainty over perception outputs. The Trust Manager owns trust in modules. These should not be conflated.

| Uncertainty type | Owner | Example |
| --- | --- | --- |
| State uncertainty | Estimator | Covariance over relative pose or velocity. |
| Observation uncertainty | Perception / sensor model | Image-based pose confidence, depth noise, star tracker confidence. |
| Model uncertainty | Planner / estimator | Dynamics mismatch, contact uncertainty, gravity approximation. |
| Controller uncertainty | Trust Manager / evaluator | Controller has poor history in current regime. |
| Hardware uncertainty | Hardware Interface / Fault Manager | Actuator response degraded, sensor dropout, latency spike. |
| Mission uncertainty | Mission / Task Manager | Goal validity, task phase ambiguity, target state unknown. |

The Decision Manager consumes uncertainty. It does not own the estimation algorithms.

## 7. Who Changes Mission Goals?

Mission goals are changed by the Mission / Task Manager, but only through allowed transition rules.

Examples:

- Primary insertion goal -> retry goal.
- Docking goal -> retreat and re-approach.
- Capture goal -> safe abort corridor.
- Assembly goal -> withdraw and inspect.
- Science objective -> degraded safe observation.

The Decision Manager may request a goal change when evidence indicates the current goal is unsafe, unrecoverable, or no longer feasible. Runtime Assurance may force a goal change if continuing the current goal violates hard constraints.

Goal changes must be logged as mission-level decisions, not hidden inside controller parameters.

## 8. Decision Hierarchy

The hierarchy should be:

```text
1. Irreversible failure avoidance
2. Hard safety constraints
3. Recoverability preservation
4. Mission survival / safe continuation
5. Goal completion
6. Efficiency / fuel / time / smoothness
7. Secondary optimization
```

This hierarchy means:

- A controller may not trade irreversible failure risk for small efficiency gains.
- A planner may not pursue a goal if doing so destroys recoverability without explicit mission authorization.
- A retry may be preferred over a fast completion attempt if the fast attempt has high unrecoverable-failure risk.
- Optimization is valuable only inside the envelope defined by safety and recoverability.

## 9. Information Flow

The runtime loop should follow this information pattern:

```text
1. Sensors produce observations.
2. Perception extracts task-relevant measurements.
3. Estimator updates belief and uncertainty.
4. Monitors compute safety, trust, fault, and recoverability signals.
5. Planner proposes feasible options under the current mission phase.
6. Decision Manager selects runtime intent and controller mode.
7. Controller proposes action.
8. Runtime Assurance evaluates action and mode.
9. If accepted, Hardware Interface executes command.
10. If vetoed, fallback behavior is selected and logged.
11. Logger records state, belief, decision, action, veto status, and outcome.
```

The Decision Manager should always receive monitor outputs before selecting a mode. Runtime Assurance should always receive the proposed action before hardware execution.

## 10. Authority Hierarchy

Authority should be explicit:

| Authority level | Component | Authority |
| --- | --- | --- |
| Hard stop | Runtime Assurance / Safety Monitor | Prevent unsafe or unrecoverable execution. |
| Mission transition | Mission / Task Manager | Change mission goal within allowed specification. |
| Runtime mode | Decision Manager | Choose continue, retry, retreat, re-observe, safe, abort, or switch. |
| Option proposal | Planner | Propose paths, subgoals, recovery options. |
| Local execution | Controller | Generate actions for selected mode. |
| Command translation | Hardware Interface | Enforce hardware command limits and report execution status. |

No lower layer should silently override a higher layer's intent. No higher layer should bypass lower-level safety checks.

## 11. Safety Hierarchy

Safety has multiple levels:

| Level | Meaning | Example response |
| --- | --- | --- |
| Nominal | All constraints satisfied with margin. | Continue. |
| Caution | Margins decreasing but still acceptable. | Slow down, replan, increase observation. |
| Warning | Safety or recovery margin near threshold. | Switch controller, retreat, re-observe. |
| Critical | Predicted irreversible failure or hard constraint violation. | Veto, safe mode, abort. |
| Failed | Irreversible failure occurred or benchmark terminal failure reached. | Terminate and label. |

The Safety Monitor should classify the current safety level. The Decision Manager chooses a response. Runtime Assurance can force the response at critical level.

## 12. Trust Hierarchy

Trust should be module-specific, not a single global confidence value.

Example trust channels:

- Perception trust.
- Estimator trust.
- Planner trust.
- Controller trust.
- Hardware trust.
- Recoverability-estimator trust.

Trust can decay due to:

- Prediction error.
- Estimator innovation spikes.
- Repeated failed retries.
- Controller saturation.
- Disagreement between models.
- Sensor dropout.
- Latency spikes.
- Recovery margin loss.
- Unexpected contact.

Trust affects decisions:

| Trust condition | Decision implication |
| --- | --- |
| High trust, high margin | Continue. |
| Low perception trust, high safety margin | Re-observe or slow down. |
| Low estimator trust | Re-observe, switch estimator, retreat, or safe mode. |
| Low controller trust | Switch controller or reduce authority. |
| Low hardware trust | Degrade goal, safe mode, or abort. |
| Low recoverability trust | Avoid aggressive continuation; request fallback. |

Trust decay should never by itself prove failure. It is evidence that changes authority and risk tolerance.

## 13. Runtime State Machine

The runtime system should use an explicit state machine. A generic version is:

```text
INIT
  -> OBSERVE
  -> PLAN
  -> APPROACH
  -> EVENT_REVIEW
  -> EXECUTE
  -> STABILIZE
  -> COMPLETE

Any active state may transition to:
  -> RE_OBSERVE
  -> RETRY
  -> RETREAT
  -> SAFE_MODE
  -> ABORT
  -> FAILED
```

### State Definitions

| State | Meaning |
| --- | --- |
| `INIT` | Load mission, benchmark, hardware/sim status, controller set, and thresholds. |
| `OBSERVE` | Gather enough information to estimate state and task phase. |
| `PLAN` | Generate candidate options and recovery branches. |
| `APPROACH` | Move toward intermediate event region. |
| `EVENT_REVIEW` | Reassess after crossing, contact, detection, alignment, or other event. |
| `EXECUTE` | Perform local task action under selected controller. |
| `STABILIZE` | Convert event state into recoverable or completed state. |
| `COMPLETE` | Task success under defined criteria. |
| `RE_OBSERVE` | Pause or adjust to improve belief before deciding. |
| `RETRY` | Return to a retry-ready state and attempt again. |
| `RETREAT` | Move away from risk or contact while preserving recovery. |
| `SAFE_MODE` | Maintain safety with minimal task ambition. |
| `ABORT` | Terminate mission attempt through an acceptable abort path. |
| `FAILED` | Terminal failure or irreversible failure. |

### Required Event Review

The architecture should require `EVENT_REVIEW` after:

- Target-radius crossing.
- First contact.
- Docking corridor entry.
- Pose lock.
- Latch attempt.
- Plug tip entry.
- Major controller switch.
- Trust threshold crossing.
- Safety warning.

This is how the architecture enforces the principle that intermediate events do not automatically imply task success.

## 14. Mission Modes

Mission modes are higher-level than controller modes.

Recommended generic modes:

- `nominal`
- `cautious`
- `recovery`
- `retry`
- `retreat`
- `reobserve`
- `degraded`
- `safe`
- `abort`
- `failed`
- `complete`

Mode transitions should require explicit reasons:

```text
nominal -> cautious: margin decreasing
cautious -> recovery: event occurred but recovery margin low
recovery -> retry: recovery failed but retry set reachable
retry -> nominal: retry setup restored
any -> safe: hard safety warning
any -> abort: no acceptable continuation except abort
any -> failed: irreversible failure
```

## 15. Fallback Behaviors

Fallback behavior must be defined before it is needed. A veto without fallback is only a stop signal.

Fallback options:

- Hold current safe state.
- Reduce action magnitude.
- Switch to recovery controller.
- Retreat from target.
- Re-observe target.
- Enter safe orbit or safe pose.
- Withdraw from contact.
- Reinitialize estimator.
- Switch sensor modality.
- Abort attempt.
- Degrade mission objective.

Fallback behavior should be evaluated by the same metrics as nominal behavior:

- Did it avoid irreversible failure?
- Did it preserve recoverability?
- Did it allow retry?
- Did it waste resources?
- Did it cause oscillation?
- Did it block valid success?

## 16. Failure Escalation

Failure escalation should be staged:

```text
anomaly -> caution -> warning -> recovery -> retreat/retry -> safe/abort -> failed
```

Example escalation triggers:

| Trigger | Escalation |
| --- | --- |
| Small estimator inconsistency | Caution, re-observe soon. |
| Repeated estimator inconsistency | Warning, re-observe now. |
| Low recovery margin | Recovery mode. |
| Recovery mode fails | Retry or retreat. |
| Retry budget exhausted | Degraded goal, safe mode, or abort. |
| Predicted irreversible failure | Final veto and abort/safe mode. |
| Irreversible failure reached | Failed. |

Escalation should include hysteresis or dwell-time rules to prevent rapid oscillation between modes.

## 17. Controller Switching

Controller switching should be explicit and logged.

Possible controller roles:

- Approach controller.
- Alignment controller.
- Post-event stabilization controller.
- Recovery controller.
- Retreat controller.
- Safe-mode controller.
- Abort controller.
- Learned advisory controller.
- Explicit fallback controller.

Switching conditions:

- Task phase changed.
- Event occurred.
- Recoverability margin changed.
- Trust decayed.
- Safety warning triggered.
- Planner selected new option.
- Controller saturation persisted.
- Failure label changed.

Switching risks:

- Mode oscillation.
- Discontinuous action commands.
- Hidden changes in objective.
- Loss of stability assumptions.
- Controller chosen outside validated domain.

Every controller should publish its intended operating envelope. The Decision Manager should not select a controller outside that envelope unless Runtime Assurance approves a degraded emergency use case.

## 18. Recovery Policies

Recovery policies are not just emergency stops. They are structured behaviors for returning to acceptable continuation.

Types:

- Stabilize after event.
- Retreat to safe distance.
- Return to retry-ready state.
- Reacquire target.
- Reduce velocity or force.
- Clear contact.
- Replan with degraded goal.
- Switch sensing mode.
- Abort through safe corridor.

Recovery policy selection depends on:

- Failure label.
- Current belief.
- Recovery margin.
- Available resources.
- Trust scores.
- Contact or crossing state.
- Mission priority.

Recovery policies should have their own success criteria and resource budgets.

## 19. Risk Budget

The architecture should maintain explicit risk budgets.

Risk budget can include:

- Probability of irreversible failure.
- Safety-margin consumption.
- Recovery-margin consumption.
- Fuel or energy budget.
- Time budget.
- Retry budget.
- Contact-force budget.
- Compute-latency budget.

The Decision Manager spends risk budget when selecting actions or modes. Runtime Assurance prevents spending beyond hard limits.

Example:

```text
If retry_budget > 0 and recovery_margin >= threshold:
    retry may be allowed
else if safe_abort_reachable:
    abort
else:
    safe_mode or failed depending on state
```

Risk budgets should be logged so experiments can distinguish brave success from reckless success.

## 20. Confidence Thresholds

Confidence thresholds determine when the system may act.

Examples:

- Minimum pose confidence for approach.
- Minimum estimator consistency for contact.
- Minimum recovery probability for continue.
- Minimum trust score for learned controller authority.
- Maximum latency for closed-loop control.
- Maximum force uncertainty for insertion.
- Minimum fuel margin for retry.

Thresholds should be mode-specific. A threshold acceptable for re-observation may be unacceptable for contact or docking.

Thresholds should be stress-tested:

- Sweep threshold values.
- Report sensitivity.
- Check known successes for regression.
- Check known failures for false acceptance.
- Evaluate held-out initial conditions.

## 21. Decision Logging

Every runtime decision should be logged as a first-class artifact.

Recommended decision log fields:

```text
timestamp
rollout_id
benchmark_id
mission_mode
task_phase
state_summary
belief_summary
uncertainty_summary
safety_level
trust_scores
recoverability_estimate
recovery_margin
risk_budget_remaining
planner_options
selected_option
selected_controller
proposed_action
veto_status
veto_reason
fallback_action
failure_label
decision_reason
thresholds_used
resource_usage
```

The log should make it possible to reconstruct:

- Why the system continued.
- Why it retried.
- Why it retreated.
- Why it re-observed.
- Why it entered safe mode.
- Why it aborted.
- Why an action was vetoed.

Without decision logging, the architecture cannot be evaluated scientifically.

## 22. Interaction with Final Veto

Final Veto is the hard authority that protects against unsafe or unrecoverable execution.

Relationship:

- Decision Manager chooses intent.
- Controller proposes action.
- Runtime Assurance evaluates action and intent.
- Final Veto blocks or modifies execution if hard criteria are violated.

Final Veto should use:

- Safety margins.
- Irreversible failure predictions.
- Recovery margins.
- Trust thresholds.
- Hardware constraints.
- Risk budget.
- Mission rules.

The veto output should be one of:

- `allow`
- `modify_action`
- `switch_controller`
- `retreat`
- `re_observe`
- `safe_mode`
- `abort`

Final Veto should not be an unlogged if-statement hidden inside a controller. It is an architecture-level authority.

## 23. Interaction with Trust Decay

Trust Decay changes how much authority a module has.

Examples:

- Low perception trust prevents close approach.
- Low estimator trust forces re-observation.
- Low controller trust triggers controller switch.
- Low planner trust requires simpler fallback.
- Low hardware trust forces degraded or safe mode.

Trust should decay from evidence:

- Prediction mismatch.
- Repeated failed attempts.
- Sensor inconsistency.
- Controller saturation.
- Unexpected contact.
- Recovery-margin loss.
- Failure-label recurrence.

Trust can recover, but recovery should require evidence, not time alone. For example, perception trust can recover after a successful re-observation with consistent multi-frame pose estimates.

## 24. Interaction with Recoverability

Recoverability is the central decision variable for post-event autonomy.

The Decision Manager should ask:

- Is the current state recoverable?
- Is the current belief recoverable?
- Is the post-event state recoverable?
- Is the recovery margin increasing or decreasing?
- Which recovery target is still reachable?
- Is final success reachable, or only retry/abort?

Recoverability should affect decisions:

| Recoverability condition | Decision |
| --- | --- |
| High recoverability, high trust | Continue. |
| Recoverable but low margin | Cautious mode or stabilization. |
| Marginal recoverability | Recovery controller, re-observe, or retreat. |
| Recoverable only to retry | Retry. |
| Recoverable only to abort | Abort. |
| Irrecoverable | Failed, unless definition permits external intervention. |

The architecture should avoid false progress by refusing to treat event occurrence as success unless the event state is recoverable under the benchmark definition.

## 25. Application: Spacecraft

For spacecraft control, runtime modes may be:

- Observe orbit state.
- Plan transfer.
- Approach target radius or docking corridor.
- Cross target radius or enter corridor.
- Event review.
- Post-cross synchronization or docking alignment.
- Stabilize.
- Complete insertion, dock, retreat, or abort.

Decision examples:

- Continue if target-radius crossing is predicted to enter a recoverable post-cross state.
- Replan if closest approach improves but crossing remains unlikely.
- Stabilize if crossing occurs with acceptable velocity but low margin.
- Retreat or abort if overspeed risk exceeds threshold.
- Safe mode if estimator uncertainty or actuator fault makes continued control unsafe.

The architecture preserves the repository's current scientific claim: crossing is an event, not success. Decision authority should shift to event review after crossing.

## 26. Application: Robotic Plug Insertion

For plug insertion, runtime modes may be:

- Observe socket.
- Estimate pose.
- Approach.
- Align.
- Contact.
- Event review.
- Insert.
- Stabilize or seat.
- Retry, withdraw, re-observe, safe, or abort.

Decision examples:

- Continue if pose uncertainty and force readings indicate recoverable contact.
- Re-observe if vision confidence decays before contact.
- Retreat if force direction suggests misalignment.
- Retry if contact is bad but withdrawal is safe.
- Abort if excessive force or jam risk enters irreversible-failure territory.
- Switch from vision-guided control to force-guided control after reliable contact.

The key lesson is that contact is not insertion. First contact should trigger event review and recoverability assessment.

## 27. Application: Docking

For docking, runtime modes may be:

- Target search.
- Relative pose estimation.
- Approach corridor.
- Alignment.
- Soft contact.
- Capture review.
- Latch.
- Retreat.
- Safe hold.
- Abort.

Decision examples:

- Re-observe if pose confidence is insufficient for corridor entry.
- Continue if relative velocity and alignment are inside recoverable capture bounds.
- Retreat if target motion or uncertainty grows.
- Abort if keep-out-zone risk or fuel margin violates threshold.
- Switch controller from approach to capture stabilization after soft contact.

Docking combines spacecraft dynamics, perception uncertainty, contact/capture behavior, and mission-level abort rules. It is a natural long-term domain for this architecture, but any docking claim must remain simulation- or experiment-specific until validated.

## 28. What This Architecture Should Prevent

The architecture is designed to prevent:

- Treating intermediate events as completion.
- Letting controllers silently change mission goals.
- Letting planners bypass safety constraints.
- Letting perception confidence imply task success.
- Letting learned policies act outside their validated domain.
- Continuing after trust has decayed below required thresholds.
- Burning all recovery margin for local reward.
- Reporting success without decision traces.
- Hiding vetoes, retries, or aborts inside aggregate metrics.

## 29. Minimal Research Implementation Path

Although this document does not implement code, the first engineering steps should be lightweight:

1. Define mission modes and task phases in benchmark metadata.
2. Add decision log fields to evaluator outputs.
3. Add structured failure labels.
4. Add recoverability margin and event-review fields.
5. Implement a simple rule-based Decision Manager for experiments.
6. Implement a simple Final Veto monitor.
7. Add trust-score placeholders before adding complex trust models.
8. Evaluate continue/retry/retreat/re-observe decisions on known benchmark failures.

The first version should be explicit and simple. A transparent rule-based system is better than an opaque learned decision layer before the repository has enough logged evidence.

## 30. Closing Architecture Statement

The runtime autonomy stack should be organized around explicit decision authority:

```text
Mission Manager defines allowed goals.
Estimator owns uncertainty.
Planner proposes options.
Decision Manager chooses runtime intent.
Controller proposes actions.
Runtime Assurance can veto.
Hardware executes only approved commands.
Logger records the full decision trace.
```

The central decision question is not "can the system optimize the next action?" but:

```text
What action preserves safe, recoverable, mission-meaningful autonomy from here?
```

That question connects final veto, trust decay, recoverability, failure labeling, and refusal of false progress into one runtime architecture.
