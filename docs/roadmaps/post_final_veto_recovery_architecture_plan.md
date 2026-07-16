# Post-Final-Veto Recovery Architecture Plan

## Status

Research roadmap and experiment-design document only.

Completed: 2026-07-16

This document defines the next research question after Final Veto Overspeed Ablation v0. It does not implement a recovery controller, Decision Manager, trust estimator, or new experiment.

## Problem

Final Veto v0 can reject a nominal action whose one-step predicted `speed_ratio` is strictly greater than `1.90`. In the frozen diagnostic stress set, this changed five monitor-off overspeed outcomes into five monitor-on `max_steps` outcomes.

The monitor stopped the declared bad transitions, but its only replacement was one step of zero action. Repeated reevaluation then produced near-continuous veto activity. The stress trajectories avoided overspeed but did not cross the target radius, enter recoverability, or achieve simulator-defined success.

The unresolved problem is:

> Stopping an unsafe proposed action does not determine how to recover useful task progress.

## Next Research Question

How can an autonomous controller decide, using explicit evidence, between:

- continue;
- adjust;
- recover;
- retreat;
- enter safe mode;
- terminate safely?

The decision must account for predicted risk, current recoverability, observation trust, available controllers, remaining resources, and the cost of intervention. It must also preserve the known Phase34 recoverable cases.

## Scope

The next phase should remain centered on the simplified 2D spacecraft testbed. It should develop general decision structure around domain-specific controllers, not one universal low-level controller.

Contact-rich manipulation and other physical embodiments remain future validation directions. They may share decision concepts, but each requires its own dynamics, sensors, safety thresholds, recovery actions, benchmarks, and evidence.

## Proposed Architecture

The next architecture should separate risk detection from recovery selection:

```text
observation or belief
        |
        v
prediction and trust assessment
        |
        v
risk classification
        |
        v
recoverability assessment
        |
        v
Decision Manager
        |
        +--> continue nominal control
        +--> adjust the proposed action
        +--> select a recovery controller
        +--> retreat to a retry-ready state
        +--> enter safe mode
        `--> terminate
        |
        v
Runtime Assurance / Final Veto
        |
        v
executed physical action
        |
        v
result, failure, and decision evidence
```

The existing one-step overspeed monitor occupies one narrow part of the Runtime Assurance layer. It should remain an independent final check on a proposed action. It should not become responsible for estimating trust, selecting a controller, or planning recovery.

## Decision Semantics

| Decision | Required evidence | Intended effect | Failure to avoid |
| --- | --- | --- | --- |
| `continue` | Risk below declared limits, acceptable trust, and adequate recovery margin | Execute the nominal controller action | Blind continuation after evidence has invalidated the nominal regime |
| `adjust` | A bounded action change can restore margin without changing the task regime | Modify magnitude, direction, or timing while preserving nominal intent | Treating every threshold approach as a full mode switch |
| `recover` | A declared recovery controller is applicable and has a plausible return condition | Restore a state from which nominal task progress can resume | Repeating zero action without a path back to the task |
| `retreat` | Local continuation is poor but a reset or retry-ready state remains reachable | Preserve future mission value by reversing commitment | Consuming risk and resources in an unrecoverable local regime |
| `safe_mode` | Active progress should stop and a bounded hold or degraded state exists | Stabilize while waiting for evidence or authority | Calling indefinite inactivity successful recovery |
| `terminate` | Recovery, retreat, and safe continuation are unavailable or violate hard constraints | End the attempt with an explicit failure record | Hiding exhaustion behind repeated retries or timeouts |

These are decision contracts, not implemented behaviors.

## Required Evidence Inputs

A recovery-aware decision should eventually consume:

- current state or belief summary;
- prediction horizon and predicted state sequence;
- declared hazard signals and thresholds;
- recoverability level and margin;
- trust in observation, prediction, planner, and controller;
- available nominal, recovery, retreat, and safe-mode options;
- remaining time, control effort, and resource budget;
- retry count and whether the state is retry-ready;
- prior vetoes, failed retries, and mode transitions;
- expected recovery cost and probability of restored task progress.

Missing evidence must produce `unknown`, `manual_audit_required`, or conservative authority. It must not produce optimistic recovery claims.

## Future Experiment 1: Recovery Action Library

### Question

What small set of mutually distinct actions is sufficient to test recovery selection after an overspeed veto?

### Design Work

Define, without implementing yet:

- action or controller identity;
- applicability preconditions;
- predicted hazard effect;
- expected recovery effect;
- return-to-nominal condition;
- maximum duration;
- resource budget;
- failure and abort condition.

Candidate categories may include reduced-action continuation, bounded coast, energy-reducing correction, geometry-reset retreat, and safe termination. Names must not imply safety before evidence exists.

### Required Metrics

- hazard recurrence;
- restored crossing and recoverable crossing;
- recovery completion rate;
- time to return to nominal control;
- control effort and resource proxy;
- blocked success and unnecessary recovery;
- terminal failure mechanism.

### Stop Condition

Do not implement a large action library until each candidate has a distinct mechanism and a measurable return condition. Multiple labels for equivalent zero-action behavior would not constitute architectural diversity.

## Future Experiment 2: Recovery Margin Metric

### Question

Can a scalar or structured margin predict whether available recovery actions can restore a recoverable task state within the remaining horizon and constraints?

### Design Work

Specify whether margin is based on:

- distance to a recoverability set;
- predicted controllability under an available recovery policy;
- time-to-hazard versus time-to-recovery;
- resource-constrained reachable states;
- a vector of radius, radial velocity, tangential velocity, speed, and resource margins.

Recoverability must remain relative to the available policy, observations, horizon, resources, and safety constraints. A geometric distance alone must not be labeled recovery margin without validation.

### Required Metrics

- calibration against future recovery outcomes;
- false-recoverable and false-irrecoverable classifications;
- margin at veto time;
- minimum margin during recovery;
- relationship to eventual crossing, recoverability, and simulator outcome.

## Future Experiment 3: Recovery Cost Metric

### Question

How should the platform compare a successful but expensive recovery with retreat, safe mode, or termination?

### Design Work

Define a cost vector before combining it into any score:

- additional steps;
- control effort;
- fuel or action proxy;
- peak action and saturation;
- lost task progress;
- number and duration of vetoes;
- controller switches;
- remaining recovery options after the maneuver.

Do not collapse safety, task completion, and cost into one reward before reporting each component separately.

## Future Experiment 4: Trust Estimation

### Question

When should prediction, state estimation, controller selection, or recoverability assessment lose authority?

### Design Work

Define observable trust evidence such as:

- prediction-realization error;
- stale or delayed observations;
- inconsistent state estimates;
- disagreement among controller predictions;
- repeated recovery failure;
- action saturation;
- mismatch between predicted and realized margin.

Trust should be calibrated against future error or failure, not assigned only through descriptive flags. Low trust should reduce aggressive authority, but it should not automatically be counted as task failure.

### Required Metrics

- trust calibration error;
- false trust alarms;
- missed degradation;
- decision changes caused by trust;
- outcomes after re-observe, controller switch, or retreat;
- overhead and latency.

## Future Experiment 5: Multi-Horizon Prediction

### Question

Does prediction beyond one step identify hazards and recovery opportunities early enough to reduce repeated vetoes and improve task recovery?

### Design Work

Predeclare a small horizon comparison rather than an open-ended planner search. Compare one-step prediction with a small number of fixed horizons under identical dynamics, controller proposals, cases, and thresholds.

Multi-horizon evaluation must report:

- earliest correct hazard warning;
- false-positive intervention rate;
- missed hazards;
- compute cost;
- prediction-realization drift;
- selected recovery action;
- recoverability and task outcome.

Longer prediction is not automatically better. Model error, compute cost, and excessive conservatism may grow with horizon.

## Proposed Research Sequence

| Order | Deliverable | Exit criterion before proceeding |
| ---: | --- | --- |
| 1 | Recovery action contract | Each candidate has preconditions, duration, return condition, failure condition, and cost fields. |
| 2 | Recovery margin definition | The metric has explicit policy, horizon, resource, and constraint dependence plus a validation protocol. |
| 3 | Recovery cost definition | Task outcome, hazard outcome, and cost remain separately reportable. |
| 4 | Offline decision table and log examples | Every continue/adjust/recover/retreat/safe-mode/terminate choice cites evidence and authority. |
| 5 | Small paired recovery experiment design | Preservation and diagnostic sets, baselines, hypotheses, and acceptance rules are frozen before code. |
| 6 | Trust and multi-horizon extensions | Added only after one recovery action can be evaluated without ambiguity. |

This sequence is a research-design order, not authorization to implement or execute the experiments.

## Evaluation Rules For Future Recovery Claims

A future recovery claim must report separately:

- whether the original hazard was avoided;
- whether task progress resumed;
- whether target-radius crossing occurred;
- whether the crossing became recoverable;
- whether simulator-defined success occurred;
- whether known Phase34 recoverable cases were preserved;
- intervention, recovery, and resource cost;
- blocked successes and unnecessary recovery actions;
- invalid simulations and unknown labels;
- full-benchmark, preservation-set, and diagnostic-subset status.

A stress case that avoids overspeed but ends at `max_steps` remains hazard avoidance without task recovery.

## Regression And Artifact Guardrails

- Keep the frozen Final Veto v0 package byte-identical.
- Create new manifests and artifact directories for every recovery experiment.
- Keep the strict `1.90` v0 threshold and zero-action v0 fallback unchanged in historical evidence.
- Run `scripts/check_phase_results.py` before accepting any future result.
- Preserve all eight known Phase34 recoverable cases before claiming broader progress.
- Keep diagnostic stress claims separate from full-benchmark claims.
- Record every controller switch, retreat, safe-mode entry, and termination decision.
- Report negative and incomplete recovery outcomes rather than hiding them in aggregate success.

## Work To Postpone

- A full mission-level Decision Manager implementation.
- A multi-hazard assurance stack.
- A universal trust manager.
- Formal safety proof without a formal model and invariant.
- Large controller tuning or learning expansion.
- 3D dynamics and high-fidelity simulator conversion.
- Hardware, ROS, or sim-to-real integration.
- Drone, manipulation, legged, ground, marine, and rover implementations.

Postponement keeps the next question measurable: can one declared recovery choice restore task progress after a veto while preserving known recoverability?

## Non-Claims

This roadmap does not establish a working recovery controller, Decision Manager, trust estimator, multi-horizon planner, formal Runtime Assurance system, formal safety result, or validation in any physical embodiment beyond the existing simulator evidence.

## Transition Principle

> The next milestone should convert a veto from repeated refusal into an evidence-based choice among bounded recovery outcomes, without allowing task-recovery language to outrun measured task recovery.
