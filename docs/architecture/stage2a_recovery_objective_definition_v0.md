# Stage 2A Recovery Objective Definition Audit v0

Completed: 2026-09-09

## Status

Offline recovery-objective semantics defined from frozen evidence. No combined score,
weighting, optimizer, policy, action selector, or Stage 2A authority is implemented.

## Purpose

This document distinguishes proposal safety, component-wise orbital progress, exact
recoverability milestones, diagnostic evidence, and future evaluators. The definitions
are intended to prevent a future action selector from treating "safe" as synonymous with
"useful" or treating one-step progress as recovery success.

## Evidence Basis

The direct evidence is Stage 2A-H and Stage 2A-I. Existing Stage 0A instrumentation and
Stage 1A guard atoms provide the frozen field semantics and inherited thresholds. No
simulator, controller, discovery, or calibration command is part of this audit.

## Safety Constraints

Required safety semantics are:

1. Required evidence must be valid. Missing, invalid, and unsupported values do not
   become pass or fail.
2. Proposed-action predicted speed ratio above `1.90` is the existing strict overspeed
   condition. At or below `1.90` is clear under that condition.
3. Final Veto remains the proposal-level decision boundary.
4. Realized overspeed is measured-state hazard evidence and remains separate from an
   action-conditioned prediction.
5. Overspeed headroom is a signed safety margin, not a complete mission objective.

## Recovery Progress

Progress remains component-wise:

- decrease absolute radius gap;
- maintain radial direction toward the target while reducing absolute radial-velocity
  ratio as arrival approaches;
- decrease absolute tangential-velocity error while preserving the signed error crossing;
- increase overspeed headroom without claiming that headroom is mission recovery.

Exact recoverability uses the inherited inclusive component thresholds: radius-error
ratio `<= 0.0025`, absolute radial-velocity ratio `<= 0.02`, and absolute tangential-error
ratio `<= 0.25`.

## Recovery Milestones

The combined Phase34-compatible predicate requires all three components. Eligible target-
radius crossing is separate event evidence. Neither is automatically recovery success,
nominal-handoff readiness, or future trajectory success.

## Diagnostic Evidence

Specific-energy error remains a diagnostic proxy. Its direction can contextualize a
transition but cannot serve as a safety gate, authoritative recovery objective, or tie
breaker without a later contract.

## Metric Conflicts

The frozen evidence demonstrates that radius closure can coexist with a worsening radial
arrival component. More headroom can accompany less radius progress. Tangential progress
can improve at a higher but still safe predicted ratio than another action. Component
progress can occur without combined recoverability or crossing.

These conflicts are retained explicitly. No exchange rate, weight, score, or optimizer is
defined.

## Future Evaluators

The following remain unavailable or unsupported: alternative-specific multistep recovery,
handoff readiness, correction authority, future crossing prediction, intervention-release
readiness, action cost, progress persistence, causal action effect, and general cross-case
action ordering.

## Future Selection Semantics

Among the requested choices, the recommended conceptual structure is **lexicographic with
safety first**:

1. validate required evidence;
2. enforce Final Veto feasibility and the frozen hazard boundary;
3. preserve recoverability components and crossing as explicit milestones;
4. compare radius, radial, tangential, and headroom evidence component by component;
5. leave unresolved tradeoffs and ties unresolved until a later reviewed contract.

Safety-first alone is insufficient because safe alternatives exhibit different progress.
A weighted multi-objective design is unsupported because no scientific weights or exchange
rates exist. This recommendation is not an executable policy and selects no action.

## Authority Boundary

Physical executions and controller executions are zero. Final Veto, threshold `1.90`,
Phase34 thresholds, Stage 2A authority, controllers, and source evidence are unchanged.

## Claim Restrictions

This audit does not demonstrate recovery success, controller superiority, optimality,
universal metric ordering, active-threshold validity, formal safety, or deployment
readiness.
