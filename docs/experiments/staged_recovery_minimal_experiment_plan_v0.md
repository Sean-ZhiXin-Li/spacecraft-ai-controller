# Staged Recovery Minimal Experiment Plan v0

## Status

Experiment design proposal only; execution not authorized.

Completed: 2026-07-26

## Purpose

This plan defines the smallest evidence-building sequence before any staged recovery experiment can be considered. It does not implement phase actions, select thresholds, authorize a rollout, or claim that the frozen branch state is recoverable.

The governing architecture is `docs/architecture/staged_recovery_architecture_v0.md`. Its manifest remains `not_authorized` because phase actions, numerical guards, no-progress thresholds, hysteresis parameters, and required trajectory instrumentation are unresolved.

## Common Rules

Every stage must preserve:

- the existing transition physics;
- Final Veto threshold `1.90` and strict `>` comparator;
- the existing Phase34-compatible inclusive recoverability thresholds;
- the frozen adverse-stop priority;
- explicit `not_evaluated` and `invalid` states;
- separation of hazard avoidance, crossing, recoverability, Recovery Success v0, and simulator success;
- protected Phase34-37, Final Veto, and published recovery evidence.

Passing one stage does not automatically authorize the next. Each stage needs a bounded implementation review, tests, and an explicit authorization decision.

## Stage 0 — Instrumentation Completeness

Before any staged trajectory, add and validate status-bearing logging for:

- per-step Cartesian position and velocity;
- radius and target-radius error;
- radial velocity and tangential velocity;
- radial-velocity ratio and tangential-velocity error ratio;
- realized and predicted speed ratio;
- orbital energy or an explicitly declared energy proxy;
- Phase34 recoverability components;
- current and previous phase;
- raw phase-transition reason;
- phase dwell and cumulative transition count;
- no-progress status and its source window;
- proposed and executed action;
- Final Veto decision.

The instrumentation test must prove that observation and logging do not alter action selection, transition physics, stop priority, or existing bounded behavior. It must preserve measured, derived, predicted, heuristic, unavailable, and invalid evidence levels. No staged action or full trajectory is authorized in Stage 0.

## Stage 1 — Offline Transition-Guard Validation

Using synthetic fixtures and existing read-only evidence, validate:

- the phase graph and forbidden transitions;
- adverse-stop precedence;
- nominal-handoff requirements;
- missing and invalid evidence behavior;
- no-progress detector structure for one named signal at a time;
- hysteresis structure, finite switching budget, and repeated-cycle handling;
- deterministic transition reasons and provenance.

Numerical phase guards, no-progress windows, improvement thresholds, dwell settings, and hysteresis settings must be separately justified and frozen before they become executable. Stage 1 executes no simulator transition.

## Stage 2 — Single-Transition Phase Tests

Only after action laws and guards are separately frozen, test one phase decision at a time with bounded one-step semantics. Each test must:

- start from a declared synthetic or frozen state fixture;
- evaluate one named phase and at most one proposed transition;
- record all guard inputs and evidence levels;
- preserve Final Veto evaluation for physical proposals;
- prove exact predictor/realized-transition consistency where a transition is executed;
- stop immediately after the bounded decision or transition;
- make no full-recovery or comparative claim.

No multi-phase trajectory is permitted in Stage 2.

## Stage 3 — Short Bounded Staged Trace

Only after Stage 2 passes, predeclare one initial state, one staged policy, one bounded diagnostic horizon, phase actions, numerical guards, no-progress configuration, hysteresis, instrumentation, and stop conditions. Execute no branch comparison and make no formal recovery claim.

The trace must test whether phase records, progress evidence, stop priority, and anti-chatter behavior remain coherent across more than one phase. It is an infrastructure and mechanism diagnostic, not a recovery-performance experiment.

## Stage 4 — Predeclared Staged Recovery Experiment

Only a new machine-readable manifest may authorize Stage 4. It must freeze:

- source cases and seeds;
- phase IDs and graph;
- every phase action law and saturation rule;
- every numerical entry, exit, and completion guard;
- no-progress signals, windows, minimum improvements, dwell limits, and missing/invalid handling;
- hysteresis, cooldown, evidence counts, transition budget, and cycle detection;
- prediction semantics and Final Veto boundary;
- recovery and total horizons;
- adverse-stop priority;
- outcome, cost, burden, and preservation metrics;
- artifact paths and overwrite protection;
- scoped allowed and prohibited claims;
- protected benchmark regression checks.

The new manifest must remain nonformal unless a later scientific protocol explicitly promotes it. No Stage 4 experiment is authorized today.

## Advancement Evidence

Expansion beyond one state or one bounded policy requires:

- complete per-step physical and phase instrumentation;
- deterministic guard and no-progress evaluation;
- no phase chatter under boundary fixtures;
- no violation of adverse-stop precedence;
- no premature nominal handoff;
- bounded physical action tests that preserve exact transition semantics;
- a predeclared preservation check;
- a clean implementation handoff and explicit execution authorization.

## Non-Claims

This plan does not establish a valid recovery controller, optimal phase ordering, optimal threshold, recoverability of the canonical branch state, adequacy of a longer horizon, formal safety, hardware validity, benchmark-wide effectiveness, cross-domain transfer, or deployment readiness.
