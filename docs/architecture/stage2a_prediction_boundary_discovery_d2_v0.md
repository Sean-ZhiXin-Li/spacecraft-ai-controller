# Stage 2A-D2 Boundary-Targeted Source-State Discovery v0

Completed: 2026-08-27

## Status

Implementation and scientific plan frozen; formal discovery result not yet executed.

Active authority is not granted. Hazard-arrest interventions remain zero.

## Purpose

Stage 2A-D2 tests whether a predeclared family of naturally generated Phase35 source states reaches the existing strict Stage 2A recovery prediction boundary under `zero_action_reference_v0`. It is a new bounded source-geometry discovery and is not a rerun or modification of Stage 2A-D1.

## D1 Evidence Dependency

The protected Stage 2A-D1 result anchors D2. Its canonical manifest hash is `a288271e615b465e9dbda5c1234df7b6963d8badd83148bb68c9b32367998860`, and its frozen scientific-plan hash is `fd6634648f6d3f690c15466bd95cbbe6dde35e162f247ea39fff49727e0080eb`. D1 evaluated 39 trajectories without finding a strict boundary; its closest zero-action prediction was `1.8906024003603095` at event zero of the legacy angle-150, thrust-8000 source state.

## Frozen Source Grid

The source family is `phase35_radial_energy_push_overspeed_stress_v0`, using the existing `radial_energy_push` upstream variant, radius ratio `0.98`, thrust scale `8000.0`, seed `0`, and unchanged Phase35 controller and mode semantics. The only angles are `150.0`, `155.0`, `160.0`, `162.5`, `165.0`, `167.5`, `170.0`, `172.5`, and `175.0` in ascending order.

No Cartesian state is manually authored, scaled, or perturbed.

## Natural Source Boundary

For each source case, the existing Phase35 rollout is observed at its pre-transition action boundary. The first state satisfying all of the following is captured:

- current realized speed ratio is at most `1.90`;
- the original nominal-controller action predicts a speed ratio strictly greater than `1.90`;
- the unchanged Final Veto decision is `veto`.

Capture occurs before the rejected nominal action or any fallback can execute. The capture exception exits the upstream rollout from inside the pre-transition hook, so the rejected proposal contributes zero physical transitions and no fallback transition. A source trajectory with no such boundary is recorded as `unavailable`; no substitute state is fabricated.

## Anchor Reproduction

The angle-150 source uses the existing version-aware legacy reproduction path. It must reproduce the registered legacy case, boundary type, branch step, prefix count, Cartesian state, nominal and predicted evidence, simulator configuration, constants, threshold and comparator, and prefix action/state hashes. Evaluating `zero_action_reference_v0` at that reproduced state must exactly match the D1 event-zero state identity, predicted-state identity, and predicted ratio `1.8906024003603095`.

Failure of any anchor check prevents publication and does not rewrite the legacy artifact.

## Recovery Discovery

Each available source boundary starts one bounded trajectory under the sole formal branch `zero_action_reference_v0`. At each event, current realized speed ratio, zero action, predicted state and ratio, and unchanged Final Veto decision are evaluated before physical execution.

The exact candidate predicate is:

```text
current realized_speed_ratio <= 1.90
and
zero_action_reference_v0 predicted_speed_ratio > 1.90
```

At a candidate, Final Veto must reject the zero action, no transition or fallback executes, the candidate is recorded, and that recovery trajectory terminates. Realized overspeed is an adverse terminal condition rather than a candidate. An allowed zero action executes only through the existing transition implementation, with exact prediction/realization equality required. The maximum is eight recovery physical transitions per available source.

## Evidence Separation

The nominal-controller predicted ratio used to locate a natural Phase35 source boundary is source-generation evidence. The zero-action predicted ratio evaluated after source capture is recovery candidate evidence. They are stored separately and are not interchangeable.

Upstream prefix physical transitions and downstream zero-action recovery physical transitions are accounted separately. Source Final Veto rejections, candidate Final Veto rejections, fallback executions, and Stage 2A interventions are also separate counters.

## Determinism

Source cases are ordered by ascending angle. Recovery records are ordered by event index. Candidate boundaries are ordered by `(initial_velocity_angle_deg, recovery_event_index, candidate_id)`. Canonical hashes cover the frozen plan, source executions, observations, trajectories, candidates, artifacts, and result manifest without volatile timestamps.

## Publication Boundary

The runner requires a committed, clean implementation HEAD equal to `origin/main`, validates D1 and all protected hashes, performs one formal invocation, validates the complete in-memory artifact set, and atomically creates `analysis/stage2a_prediction_boundary_discovery_d2_v0/`. Existing output is never overwritten, and infrastructure failure publishes nothing.

## Authority Boundary

`active_authority_granted` is always false and `hazard_arrest_interventions` is always zero. D2 does not import the Stage 2A authority adapter, produce a hazard-arrest proposal, execute `velocity_opposed_thrust_v0` as authority, change Final Veto, or invoke Stage 2A-Q1 or Stage 2A-E.

## Claim Restrictions

Even if a boundary is found, D2 establishes only that a bounded, predeclared family of natural Phase35 source states can reach the existing prediction-boundary condition under frozen zero-action semantics. It does not establish hazard-arrest effectiveness, recovery improvement, safety, stability, optimality, new threshold validity, handoff readiness, multi-step active recovery, hardware validity, or deployment readiness.
