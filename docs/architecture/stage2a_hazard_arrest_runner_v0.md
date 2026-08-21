# Stage 2A One-Intervention Hazard-Arrest Runner v0

## Status

Runner, offline qualification, and measured-result contracts implemented; no
measured Stage 2A experiment has run at implementation freeze.

Completed: 2026-08-21

## Purpose

This runner implements the narrow active boundary defined by the frozen Stage
2A preflight and authority adapter. It separates normal recovery authority,
hazard observation, explicit Stage 2A authority, proposal generation, Final
Veto submission, physical execution, proposal consumption, release evaluation,
and termination.

## Authority Boundary

Only `hazard_arrest` can receive the one provisional request. The proposal
reuses `generate_velocity_opposed_action()` under the existing
`velocity_opposed_thrust_v0` identity. No action law is copied into the runner.

The trigger is the frozen prediction boundary:

```text
current realized_speed_ratio <= 1.90
and normal predicted_speed_ratio > 1.90
```

Exactly `1.90` remains clear. Missing, invalid, unsupported, shadow-derived,
handoff, and correction-authority evidence cannot create authority. Current
realized overspeed remains an adverse stop rather than an intervention trigger.

## Final Veto

Every normal or hazard proposal is evaluated by
`one_step_overspeed_veto_v0`. A veto executes no transition, does not execute
the monitor's zero-action fallback, and does not choose another branch. The
runner records prediction-only qualification separately from physical
execution.

## Proposal Lifecycle

The active lifecycle is:

```text
not_generated
-> generated
-> submitted_to_final_veto
-> consumed
```

Submission consumes the one-proposal budget whether Final Veto allows or
rejects the proposal. No second intervention is available.

## Offline Qualification

Qualification reads the 13 frozen Stage 1B trace files and executes no physical
transition. It inspects every valid transition pre-state that leaves capacity
inside the 32-transition infrastructure bound. It evaluates the existing
`zero_action_reference_v0` and `tangential_error_correction_v0` normal actions,
then evaluates the existing velocity-opposed proposal only at a valid predicted
overspeed boundary.

Eligibility requires realized clear evidence, a vetoed normal prediction above
`1.90`, and a Final-Veto-allowed velocity-opposed proposal. Candidates are
ordered lexically by registry member, source trace, prefix transition count,
and normal branch. Intervention quality is not a selection objective.

## Prefix Reproduction

Both measured runs start from the selected registered branch state and replay
the source trace branch for the frozen prefix count. Every prefix pre-state,
action, Final Veto decision, predicted state hash, realized state hash, and
transition status must match the frozen Stage 1B trace. A mismatch is an
infrastructure failure.

## Baseline And Active Runs

The baseline submits the selected normal action at the boundary. Its predicted
overspeed is expected to be vetoed, so the boundary adds no physical
transition. The vetoed next state is a counterfactual one-step prediction, not
a measured baseline state.

The active run reproduces the same prefix, reconstructs the same normal
prediction, requests one proposal through the authority adapter, submits it to
unchanged Final Veto, consumes it, and executes at most one transition. It then
evaluates adverse stops and release and terminates. No resumed normal action is
executed.

## Release

Release can return only to the exact selected normal branch. It requires fresh
finite measured state, realized clear evidence, a fresh clear normal-branch
prediction, valid evaluator evidence, and no adverse stop. The current runtime
does not provide a compatible fresh release evaluator, so this evidence remains
unavailable and release remains `not_authorized`; this is a valid measured
outcome. No new evaluator or threshold is invented.

## Publication

Qualification and experiment directories are new-only atomic publications.
The measured command requires a committed clean selection, an absent result
directory, and exactly one top-level invocation. It performs one baseline and
one fresh active run. It never retries automatically.

## Authority Remaining Prohibited

`stabilization_assessment`, `radial_recommitment`,
`tangential_alignment`, `crossing_preparation`,
`recoverability_verification`, `nominal_handoff`, and `retreat` remain without
active authority. Explicit abort remains terminal evidence, not a recovery
action. Shadow output is never a physical control input.

## Claim Restrictions

A valid result establishes one provenance-bound one-intervention execution
path only. It cannot establish general recovery improvement, controller
superiority, stability, optimality, formal safety, new threshold validity,
handoff readiness, retreat capability, multi-step active recovery, hardware
validity, or deployment readiness.
