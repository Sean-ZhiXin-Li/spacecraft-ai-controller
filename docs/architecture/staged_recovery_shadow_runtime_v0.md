# Staged Recovery Shadow Runtime v0

Status: Observational shadow guard FSM implemented; active staged recovery execution remains unauthorized.

Completed: 2026-08-09

## Purpose

This module executes `engineering_shadow_policy_v0` against immutable Stage 0B events and Stage 1A guard-atom evaluations. It maintains an independent phase recommendation state without providing any physical control output.

## Authority Boundary

`shadow_output_consumed_by_physical_runtime = false`

The observer callback returns `None`. A non-`None` return is an infrastructure error. Shadow state cannot replace a proposed or executed action, monitor decision, spacecraft state, evaluator result, terminal reason, or runtime phase. No shadow record contains a physical action command.

## Evidence Inputs

The runtime reuses Stage 0A observation derivations through the Stage 0B logger and evaluates the frozen Stage 1A guard inventory. Missing evidence remains unavailable and invalid evidence remains invalid. Predicted and realized overspeed evidence remains separate.

## Phase Resolution

The engineering priority is explicit abort, invalid or externally unsafe evidence, realized or predicted overspeed, Phase34-compatible recoverability, radial component failure, tangential component failure, crossing preparation, then stabilization assessment. This ordering is an engineering shadow policy, not an optimal or scientifically validated controller policy.

The internal `unassigned` state exists only before the first valid observation and is never an architecture phase. All subsequent phase identities and graph edges come from `staged_recovery_contract.py`.

## Nominal Handoff

Nominal handoff remains blocked. Stage 1B-A supplies no externally validated handoff-readiness signal and grants no handoff authorization. Phase34-compatible recoverability is recorded as recoverability-verification evidence, not handoff readiness.

## Anti-Chatter

The frozen engineering limits are:

- minimum phase dwell: 2 events;
- transition cooldown: 1 event;
- maximum shadow transitions: 8 per trace.

The runtime also records architecture graph blocks, `A -> B -> A` two-cycles, repeated transition reasons, and transition-budget exhaustion. These counters affect only shadow state.

## Smoke Boundary

The smoke plan uses all four validated registry members and the existing `velocity_opposed_thrust_v0` branch for at most 32 physical transitions. Each member is executed once without the observer and once from a fresh reload with the observer. Publication requires exact physical equivalence and includes only observed shadow traces.

## Claim Restrictions

The strongest supported conclusion is that the guard logic can execute deterministically as an observational shadow FSM without changing physical runtime. This does not establish recovery improvement, controller safety, validated thresholds, optimal phase ordering, autonomous handoff readiness, formal assurance, or deployment readiness.
