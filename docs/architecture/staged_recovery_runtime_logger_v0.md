# Staged Recovery Runtime Logger v0

## Status

Runtime logging boundary and synthetic trace validation implemented; real runner integration and measured-trace validation not performed.

Completed: 2026-07-29

> This document defines an observational staged-recovery runtime logging boundary only. The logger records explicitly supplied runtime evidence and applies pure Stage 0A derivations; it does not execute a simulator transition, generate or modify an action, select a phase, select a stop condition, integrate a real runner, authorize a staged rollout, demonstrate task recovery, or support formal safety claims.

## Purpose

Stage 0B defines how a future authorized caller can submit runtime evidence without transferring execution authority to the logger. The logger validates order and internal consistency, builds deterministic records, and can publish a bounded synthetic trace bundle. It does not infer missing events or make runtime decisions.

## Relationship to Stage 0A Instrumentation

The implementation imports the pure definitions in `runtime_assurance/staged_recovery_instrumentation.py`. Explicit Cartesian states and orbital configuration are passed to Stage 0A for orbital, hazard, recoverability, crossing, progress, and action-geometry derivations. Predicted states remain predicted evidence; measured post-states remain realized evidence.

Stage 0B does not change the Stage 0A schema or artifacts. A field being accepted or derivable does not mean it has appeared in a measured trace.

## Relationship to Staged Architecture

The logger preserves the architecture version and phase/provenance fields but does not call phase guards or select a transition. Missing phase runtime fields remain `not_evaluated`. The architecture remains `not_authorized` for execution.

## Observational-Only Boundary

The caller owns current and realized states, proposed and executed actions, Final Veto decisions, predictions, transition flags, evaluator outputs, phase metadata, terminal reasons, and counters. The logger calls no simulator runner, transition function, controller, branch action generator, phase selector, stop selector, or Recovery Success decision-maker.

## Session Header

An immutable header records schema versions, case and seed, implementation commit, source-state and configuration hashes, an explicit positive `max_events`, output purpose, claim restrictions, and authorization status. Stage 0B accepts only synthetic dependency-injected fixtures and requires `scientific_result=false`.

`max_events` is a memory and logging bound. It is not the 32-step infrastructure cap, the 10,000-transition recovery horizon, the 100,000-transition total horizon, or evidence that any horizon is sufficient.

## Event Identities

The frozen event identities are:

- `initial_snapshot`
- `transition`
- `terminal`

No separately executable phase event exists. Phase-transition metadata can be preserved inside supplied runtime evidence.

## Event Ordering

The logger session states are `created`, `started`, `terminal`, and `finalized`. The normal sequence is create, initial snapshot, zero or more transitions, terminal, finalize. A zero-transition terminal session is permitted only when its terminal input preserves the initial measured state.

The logger rejects duplicate or skipped indices, transitions before initialization, decreasing or inconsistent counters, events after terminal or finalize, duplicate finalization, and missing/extra realized-state evidence. It does not fabricate events to repair a sequence.

## Initial Snapshot

The initial event is index zero, contains one measured state, executes no transition, and contains no proposed or executed action. Crossing and progress remain unavailable without a previous measured state. A missing phase remains `not_evaluated`; no default phase or zero dwell is invented.

## Transition Event

The schema preserves this logical order:

1. measured pre-state;
2. proposed action or explicit no-action evidence;
3. supplied monitor decision;
4. supplied predicted state when available;
5. supplied action disposition and executed action;
6. supplied transition-executed flag;
7. supplied measured realized state when execution occurred;
8. pure Stage 0A post-state derivation;
9. supplied evaluator and phase provenance.

This validates the caller boundary, not external wall-clock ordering. Progress compares measured pre/post states only. A predicted state is never substituted for a missing realized state.

## Terminal Event

A terminal event records a caller-selected reason and executes no transition. It retains prior counters. Explicit abort has no physical action; rejection may preserve a proposed action but has no executed rejected transition. A terminal event cannot be followed by another event.

Recovery Success is only preserved when supplied externally. Contradictory success and adverse/invalid evidence is retained and marks the event invalid; the logger does not choose a stop condition or repair the contradiction.

## Counter Semantics

`event_index` increments for every record. `recovery_step` and `total_transition_count` increment exactly once only for a realized physical transition. Terminal and rejected-action events retain both transition counters. Phase dwell and phase-transition counts are externally supplied or unavailable.

## Action Disposition

The vocabulary is `executed_unchanged`, `executed_modified`, `suppressed`, `rejected`, `zero_action_executed`, `no_action`, `not_evaluated`, and `invalid`.

A physical zero action has explicit `(0.0, 0.0)` proposed and executed vectors and a realized transition. Explicit abort has null actions and no transition. Rejection preserves the proposal and has no executed action. Suppression is accepted only as explicit evidence with differing proposed and executed actions; missing execution evidence is not silently called suppression. No fallback action is generated.

## Predicted Versus Realized Evidence

Predicted state, speed ratio, and overspeed headroom remain separate from realized state, ratio, and headroom. Prediction-error diagnostics are derived only when both inputs are valid. State-hash equality denotes exact canonical identity only; unequal hashes do not define physical distance.

## Phase and Provenance Logging

Current/previous phase, dwell, transition count and reason, recent history, no-progress status, handoff readiness, retreat, and abort status are caller-owned evidence. The logger neither supplies defaults nor evaluates phase readiness.

## Evaluator and Terminal Evidence

Simulation validity, recovery-evaluation validity, overspeed, instability, unsafe state, action rejection, explicit abort, Recovery Success, horizon exhaustion, simulator success, and terminal reason may be supplied. Missing evidence remains `not_evaluated`; malformed evidence remains `invalid`. These fields are observations, not a logger-selected stop decision.

## Deterministic Event Hashing

Events use UTF-8, sorted keys, stable separators, finite JSON values, deterministic list order, and one JSON object per line. Each event stores a SHA-256 over its scientific payload excluding its self-hash and optional volatile timestamp. The trace hash covers the ordered event hashes; event reordering changes it.

## Trace Bundle

A complete bundle contains exactly `trace_manifest.json` and `staged_recovery_trace.jsonl`. Stage 0B bundles are labeled `synthetic`, `dependency_injected_fixture`, and `scientific_result=false`. No trace bundle is checked into the repository for this milestone.

The manifest records a full-file hash for JSONL and a scoped canonical payload hash for itself, because a file cannot contain its own complete-byte hash. The publication result separately reports the complete-byte hashes of both files.

## Bounded Logging

Every session requires an explicit finite positive capacity. An over-capacity append fails before mutation and preserves previously accepted in-memory records. Capacity exhaustion is not Recovery Success, recovery-horizon exhaustion, or evidence of scientific sufficiency. Incomplete capacity-stopped bundles cannot be published.

## Atomic Publication

Publication validates a finalized complete bundle, requires an absent target with an existing parent, stages both files in a temporary sibling directory, validates staged bytes/hashes, then uses a same-filesystem atomic directory rename. Overwrite is refused. Failure removes only the task-owned staging directory and never deletes an existing user target.

## Protected Paths

Resolved and case-normalized containment checks reject repository root, Phase34-37 evidence, Final Veto evidence, frozen and published recovery evidence, mechanism diagnosis, staged architecture, Stage 0A, Stage 0B metadata, and controller/simulator/runtime code locations. `..` traversal and symlinks resolving into protected locations are rejected.

## Field Coverage

All 105 Stage 0A catalog fields have one Stage 0B classification:

- runtime direct input: 28;
- Stage 0A derivation during logging: 41;
- previous runtime state required: 20;
- predicted runtime state required: 3;
- phase runtime required: 10;
- future evaluator required: 1;
- unsupported: 2.

The architecture's 52 signals retain corresponding classifications. For every field, `real_trace_has_validated=false`.

## Missing and Invalid Evidence

Unknown numeric values remain null, never zero. Unknown booleans remain null, never false. `not_evaluated` denotes unavailable evidence; `invalid` preserves malformed or contradictory evidence. Neither status authorizes a phase, stop, handoff, or recovery claim.

## Synthetic Validation

Tests use dependency-injected Cartesian states, decisions, evaluator fields, and temporary directories. They exercise sequencing, instrumentation, hashing, capacity, publication failure, and path protection without calling a transition or real runner.

## Stage 0C Real-Trace Integration Boundary

Future work must separately review and freeze real hook points before a measured trace can be recorded: before action decision, after prediction, after action disposition, after realized transition, and at terminal decision. A future integration must demonstrate that all required caller evidence is available without changing physics or decisions.

## Current Limitations

No real runner is connected. No measured trace has been validated. Phase actions, numerical guards, no-progress thresholds, hysteresis parameters, handoff readiness, and available correction authority remain unresolved. This implementation does not establish runtime completeness.

## Claim Restrictions

Stage 0B does not establish task recovery, runtime correctness, controller quality, optimal phase logic, formal safety, hardware validity, cross-domain validity, or deployment readiness. Synthetic trace validity is structural evidence only.
