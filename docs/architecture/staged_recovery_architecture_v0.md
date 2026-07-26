# Staged Recovery Architecture v0

## Status

Architecture and evidence contract frozen; staged recovery execution not implemented.

Completed: 2026-07-26

This document defines a staged recovery architecture and evidence contract only. It does not implement a recovery controller, authorize a rollout, demonstrate task recovery, establish optimal phase logic, or support formal safety claims.

The machine-readable sources are:

- `runtime_assurance/staged_recovery_contract.py`;
- `analysis/staged_recovery_architecture_v0/architecture_manifest.json`;
- `analysis/staged_recovery_architecture_v0/evidence_traceability.json`.

Execution status is `not_authorized` because phase actions, numerical transition guards, no-progress thresholds, hysteresis parameters, and required trajectory instrumentation are not yet frozen and validated.

## Purpose

The architecture separates immediate hazard response from the state-dependent work required to regain task recoverability:

```text
hazard arrest
  -> stabilization assessment
  -> radial recommitment
  -> tangential alignment
  -> crossing preparation
  -> recoverability verification
  -> nominal handoff / retreat / explicit abort
```

It defines phase responsibilities, an allowed transition graph, evidence-bearing observations, adverse-stop precedence, no-progress and phase-chatter requirements, and claim limits. It deliberately leaves action laws and unsupported numerical guards unresolved.

## Evidence Basis

The source evidence is the frozen one-case four-branch recovery experiment and its read-only mechanism diagnosis:

- recovery result commit `5f31c3fd74dbf8e8ea5a60d70d7b88f5a9def7c8`;
- mechanism-diagnosis commit `03c8a355586cc5adeec6e8e3d7e192bb84d5c1d0`;
- `analysis/recovery_action_branching_nonformal_v0/`;
- `analysis/recovery_branch_mechanism_diagnosis_v0/`;
- Recovery Success v0 and Phase34-compatible predicates in `runtime_assurance/recovery_evaluators.py`;
- frozen stop priority in `runtime_assurance/recovery_stop_conditions.py`.

Evidence labels have exact meanings:

| Label | Meaning |
| --- | --- |
| `directly_supported` | Published measured artifacts or an already-frozen runtime contract directly demonstrate the need for the rule. |
| `partially_supported` | Evidence supports the concern or one component, but not a complete transition rule or threshold. |
| `consistent_with_evidence` | The design is compatible with the evidence but is not uniquely implied by it. |
| `design_hypothesis` | The rule is proposed for future validation and is not a measured finding. |
| `not_yet_specified` | The architecture requires the item, but no valid definition or threshold is frozen. |

Every phase, signal, transition, guard, no-progress rule, hysteresis rule, and claim boundary is linked to evidence in `evidence_traceability.json`.

## Why Single-Mode Recovery Was Insufficient

For the frozen branch state, the three physical responses avoided realized overspeed and exhausted 10,000 transitions without crossing or Recovery Success v0. Final Veto allowed all 30,000 physical proposals, so repeated post-branch veto was not the stall mechanism.

- Zero action retained approximately ballistic behavior and closed only a small part of the radius gap.
- Velocity-opposed thrust suppressed useful motion and removed excessive orbital energy for this case.
- Tangential correction improved the tangential component but did not resolve radius and radial-velocity conditions.
- The active actions were recomputed from state, but each branch remained one response mode for its entire horizon.

These findings directly support separating hazard arrest from task recovery. They motivate, but do not validate, staged phase switching.

## Phase Definitions

### `hazard_arrest`

Suppress the immediate declared hazard-producing behavior while preserving simulation validity and proposed-versus-executed action evidence. Required outputs include hazard status, Final Veto status, realized overspeed headroom, and an arrest-completion status. This phase never establishes task recovery. No new physical arrest action is frozen.

### `stabilization_assessment`

Determine whether the post-arrest state is valid, observable, and usable for recovery planning. Inputs include simulation validity, instability, unsafe-state and speed-ratio evidence, recoverability-component availability, and progress-signal availability. Absence of overspeed alone is not stabilization. Missing or invalid evidence can retain assessment, route to controlled failure, retreat, or abort; exact guards remain unresolved.

### `radial_recommitment`

Restore explicitly monitored task-directed radial progress. Required signals include radius, target-radius error, radial velocity and error, progress direction/rate, overspeed headroom, correction authority, and energy status when valid. The need to observe radial progress is partially supported. The radial action, target radial velocity, threshold, and dwell are not specified.

### `tangential_alignment`

Reduce tangential velocity error without erasing required radial progress. Required signals include tangential velocity, target circular speed, tangential error, radial progress, headroom, energy change, and action saturation. Tangential improvement alone cannot complete this phase or establish recovery. The previous magnitude-0.25 branch is evidence, not the staged policy.

### `crossing_preparation`

Coordinate radial and tangential geometry before target-radius crossing. Inputs include target-radius error, radial-velocity ratio, tangential-error ratio, crossing proximity or progress, headroom, and the recoverability component vector. No validated crossing predictor is claimed; unsupported predictions remain `not_evaluated`.

### `recoverability_verification`

Keep crossing, Phase34-compatible recoverability, Recovery Success v0, and simulator success separate. The existing inclusive component bounds remain unchanged:

- absolute radius error ratio `<= 0.0025`;
- absolute radial velocity ratio `<= 0.02`;
- absolute tangential velocity error ratio `<= 0.25`.

Crossing without recoverability may continue controlled recovery, retreat, or abort. It cannot authorize nominal handoff by itself.

### `nominal_handoff`

Represent a terminal architecture decision to return authority to the nominal controller. It requires valid simulation and recovery evaluation, no active higher-priority adverse stop, verified crossing, Phase34-compatible recoverability, Recovery Success v0, explicit handoff readiness, and provenance. Controller switching is not implemented.

### `retreat`

Represent movement toward a separately declared lower-risk state when original task recovery is unavailable or unjustified. Retreat remains distinct from Recovery Success v0 and simulator success. Its target, action, success predicate, and thresholds are `not_yet_specified`; no retreat capability is claimed.

### `explicit_abort`

Terminate without a further physical recovery transition. Explicit abort is never Recovery Success v0 or nominal handoff, and it does not imply unsafe state or simulator success.

## Allowed Transition Graph

The v0 directed graph is frozen as follows. Every edge is architecture-only and non-executable.

```text
hazard_arrest
  -> stabilization_assessment | explicit_abort

stabilization_assessment
  -> radial_recommitment | retreat | explicit_abort

radial_recommitment
  -> tangential_alignment | stabilization_assessment | retreat | explicit_abort

tangential_alignment
  -> crossing_preparation | radial_recommitment | stabilization_assessment
  -> retreat | explicit_abort

crossing_preparation
  -> recoverability_verification | radial_recommitment | tangential_alignment
  -> stabilization_assessment | retreat | explicit_abort

recoverability_verification
  -> nominal_handoff | radial_recommitment | tangential_alignment
  -> retreat | explicit_abort

retreat -> explicit_abort
nominal_handoff -> terminal
explicit_abort -> terminal
```

Each edge records required evidence, prohibited adverse evidence, a transition reason, evidence support, threshold status, new-evidence and hysteresis requirements, and adverse-stop precedence.

## Forbidden Transitions

The contract rejects:

- `hazard_arrest -> nominal_handoff`;
- `stabilization_assessment -> nominal_handoff`;
- `radial_recommitment -> nominal_handoff`;
- `tangential_alignment -> nominal_handoff`;
- `explicit_abort ->` any active phase;
- `nominal_handoff ->` any recovery phase in v0;
- recovery success or handoff while a higher-priority adverse stop is active.

Terminal phases have no active outgoing edges. Every nonterminal phase has a graph route to retreat or explicit abort.

## Observation Contract

`StagedRecoveryObservation` records identity and provenance separately from status-bearing signals. Identity fields are case ID, seed, branch-state hash, simulator-configuration hash, constants hash, implementation commit, recovery step, total transition count, current phase, and previous phase.

Every signal carries one status:

- `measured`;
- `derived`;
- `one_step_predicted`;
- `multi_step_predicted`;
- `heuristic`;
- `not_evaluated`;
- `invalid`.

An evaluated signal must carry an explicit finite JSON-compatible value. `not_evaluated` and `invalid` carry no favorable value. Unknown numeric values never become zero, and unknown booleans never become false.

The signal set covers validity, hazards, Final Veto and action rejection, physical state components, target geometry, speed and energy, recoverability, horizon state, progress, phase dwell/history, handoff readiness, and simulator success. Evidence levels remain attached to each value; predicted and measured values are never merged.

## Recovery Progress Signals

The architecture requires separate progress observations rather than a combined score:

- radial progress and target-radius error;
- radial velocity and radial-velocity ratio;
- tangential progress and tangential-error ratio;
- crossing progress;
- overspeed headroom;
- energy-change direction or a declared energy proxy;
- correction authority and saturation margin;
- the Phase34 recoverability component vector.

The published log did not retain per-step Cartesian state, radius, radial/tangential velocity, target-radius error, energy, or component margins. Those signals therefore require new validated instrumentation before staged execution.

## No-Progress Detection

The pure no-progress contract accepts exactly one named signal and requires explicit configuration for desired direction, observation window, minimum meaningful improvement, phase dwell limit, missing-evidence handling, and invalid-evidence handling. It returns one of `progressing`, `stalled`, `regressing`, `not_evaluated`, or `invalid`.

Timeout alone does not prove a stall. One flat sample does not prove a stall. Missing samples yield `not_evaluated`; malformed/nonfinite samples yield `invalid`. Different signals are not silently combined. A no-progress outcome can motivate reassessment or retreat, never Recovery Success v0. No windows or improvement thresholds are frozen in v0.

## Hysteresis And Anti-Chatter Requirements

Before any cyclic transition becomes executable, its phase contract must freeze:

- minimum phase dwell;
- distinct, justified entry and exit thresholds;
- transition cooldown;
- consecutive-evidence count;
- finite phase-transition budget;
- repeated-cycle detection window;
- a requirement for new evidence before repeated transitions;
- preservation of the raw transition reason.

The validator rejects an executable transition with incomplete hysteresis, unlimited switching, reused evidence, or lost transition reasons. All numerical hysteresis settings remain unresolved.

## Adverse Stop Priority

The existing priority is unchanged and overrides phase progression:

```text
invalid_simulation
> invalid_recovery_evaluation
> overspeed
> instability
> unsafe_state
> action_rejected
> explicit_abort
> recovery_success
> recovery_horizon_exhausted
> total_horizon_exhausted
```

Thus simultaneous overspeed and crossing terminates as overspeed; invalid simulation cannot hand off; action rejection cannot become reassessment; explicit abort cannot be overridden by success; and `not_evaluated` cannot advance a phase or trigger a favorable stop.

## Action-Law Boundary

For each active phase, the contract defines an objective, required inputs, prohibited outcomes, output shape, saturation-awareness requirement, Final Veto requirement, and progress-monitoring requirement. It defines no thrust magnitude, gain, action vector, target velocity, phase duration, switch threshold, or optimization weight.

The prior branch laws cannot be reused unquestioned:

- zero action demonstrates hazard arrest without task recovery;
- velocity-opposed response demonstrates the risk of suppressing useful motion and energy;
- tangential correction demonstrates component improvement without full geometry recovery.

Any future action law needs a separate machine-readable freeze and bounded tests before it can enter a staged rollout.

## Nominal Handoff Contract

Handoff requires complete valid evidence for simulation validity, recovery-evaluation validity, no overspeed, no instability, no unsafe-state event, no action rejection, no explicit abort, target-radius crossing, Phase34-compatible recoverability, Recovery Success v0, and handoff readiness. Crossing, no overspeed, simulator success, or one improved component alone is insufficient. The contract records why and from which evidence handoff is permitted.

## Retreat And Abort Distinction

Retreat is a future physical mode aimed at a separately frozen lower-risk or retry-ready region. Explicit abort is an immediate terminal decision with zero subsequent physical recovery transitions. Neither is task recovery by default. Abort can avoid further exposure through termination; retreat capability cannot be claimed until its target and predicate exist.

## Evidence-Supported Rules

Directly supported rules include:

- hazard arrest and task recovery must remain separate;
- one improved velocity component does not establish recoverability;
- crossing, recoverability, simulator success, and Recovery Success v0 remain distinct;
- velocity-opposed recovery must monitor useful-motion and energy loss;
- explicit abort executes no post-decision transition and is not recovery;
- the existing Phase34 predicate and stop priority must remain unchanged;
- the observation schema must make unavailable physical mechanism signals explicit.

Partially supported rules include explicit radial-progress monitoring and an energy-aware boundary. The experiment supports their necessity as diagnostics but not numerical targets, guards, or dwell periods.

## Design Hypotheses

The ordering of radial recommitment, tangential alignment, crossing preparation, and verification is a design hypothesis. Reassessment loops, retreat decisions, nominal-handoff readiness, crossing prediction, no-progress routing, and anti-chatter behavior are also hypotheses until bounded evidence exists. The one-case result does not prove that this graph will recover the frozen state.

## Unresolved Numerical Definitions

The manifest lists 22 unresolved IDs covering arrest/stabilization guards, radial and tangential thresholds, crossing proximity, retreat semantics, no-progress windows and improvement thresholds, dwell limits, cooldown, consecutive evidence, transition budget, and cycle detection. All active phase action laws are also unresolved. Existing Phase34 verification thresholds are preserved and are not tuning variables in this architecture.

## Required Instrumentation

Before any staged rollout, record per transition:

- Cartesian position and velocity;
- radius and target-radius error;
- radial and tangential velocity;
- radial-velocity ratio and tangential-error ratio;
- realized and predicted speed ratio;
- orbital energy or a declared energy proxy;
- recoverability components;
- current phase, previous phase, phase dwell, transition count, and raw transition reason;
- no-progress status and source samples;
- proposed and executed actions;
- Final Veto decision;
- evidence status and provenance for every guard input.

Instrumentation must be validated without changing physics.

## Validation Boundary

The pure validator checks phase and edge uniqueness, reachability, terminal behavior, escape paths, forbidden handoffs, adverse priority, evidence classifications, unknown-value behavior, unresolved-threshold disclosure, handoff requirements, abort and retreat semantics, no-progress separation, cycle handling, execution authorization, and canonical manifest hashing.

Structural validity means the contract is internally coherent. It does not mean the architecture is executable or effective.

## Minimal Future Experiment Sequence

The evidence sequence is defined in `docs/experiments/staged_recovery_minimal_experiment_plan_v0.md`:

1. instrumentation completeness;
2. offline transition-guard validation;
3. bounded single-transition phase tests;
4. one short staged trace;
5. a separately frozen predeclared experiment.

No stage is authorized by this document.

## Claim Restrictions

Staged Recovery Architecture v0 establishes no recovery controller, execution readiness, task-recovery outcome, optimal phase sequence, state recoverability, benchmark-wide effectiveness, formal safety, hardware validity, cross-domain validation, or deployment readiness. It does not change the frozen branch state, measured evidence, physics, controllers, Final Veto threshold, prior action magnitude, or historical artifact semantics.
