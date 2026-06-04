# Planner Search Benchmark Manifest

## Purpose

This manifest defines the 24-case reduced benchmark for the next small parameterized planner-level transfer search. It is documentation-first and does not introduce new simulations, controller logic, physics, thresholds, or result claims.

The fixed terminal controller is Phase34 `radius_priority` post-cross synchronization. The next planner search may change only the upstream pre-cross transfer parameters before first target-radius crossing.

## Fixed Assumptions

- Environment: simplified 2D orbital-control sandbox.
- Benchmark scope: same 24-case reduced benchmark used by Phase34, Phase35, and Phase36B.
- Fixed terminal controller: Phase34 `radius_priority`.
- Fixed terminal role: post-cross synchronization after first target-radius crossing.
- No physics changes.
- No CAPTURE/LOCK threshold changes.
- No recoverability threshold changes.
- No historical CSV overwrite.

## Case Manifest

| case_id | initial_conditions | fixed_controller | status | comments |
|---|---|---|---|---|
| PS-001 | `r0_over_target=0.98; initial_velocity_angle_deg=150; thrust_scale=8000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `near_crossing`; candidate for planner timing/commitment search. |
| PS-002 | `r0_over_target=1.00; initial_velocity_angle_deg=150; thrust_scale=8000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline crossing-producing recoverable | Phase36B baseline crossing case; use as regression guard for Phase34 handoff. |
| PS-003 | `r0_over_target=1.02; initial_velocity_angle_deg=150; thrust_scale=8000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `over_conservative_transfer`; candidate for planner commitment search. |
| PS-004 | `r0_over_target=0.98; initial_velocity_angle_deg=165; thrust_scale=8000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `near_crossing`; candidate for planner timing/commitment search. |
| PS-005 | `r0_over_target=1.00; initial_velocity_angle_deg=165; thrust_scale=8000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline crossing-producing recoverable | Phase36B baseline crossing case; use as regression guard for Phase34 handoff. |
| PS-006 | `r0_over_target=1.02; initial_velocity_angle_deg=165; thrust_scale=8000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `over_conservative_transfer`; candidate for planner commitment search. |
| PS-007 | `r0_over_target=0.98; initial_velocity_angle_deg=170; thrust_scale=8000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `near_crossing`; candidate for planner timing/commitment search. |
| PS-008 | `r0_over_target=1.00; initial_velocity_angle_deg=170; thrust_scale=8000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline crossing-producing recoverable | Phase36B baseline crossing case; use as regression guard for Phase34 handoff. |
| PS-009 | `r0_over_target=1.02; initial_velocity_angle_deg=170; thrust_scale=8000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `over_conservative_transfer`; candidate for planner commitment search. |
| PS-010 | `r0_over_target=0.98; initial_velocity_angle_deg=175; thrust_scale=8000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `near_crossing`; candidate for planner timing/commitment search. |
| PS-011 | `r0_over_target=1.00; initial_velocity_angle_deg=175; thrust_scale=8000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline crossing-producing recoverable | Phase36B baseline crossing case; use as regression guard for Phase34 handoff. |
| PS-012 | `r0_over_target=1.02; initial_velocity_angle_deg=175; thrust_scale=8000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `over_conservative_transfer`; candidate for planner commitment search. |
| PS-013 | `r0_over_target=0.98; initial_velocity_angle_deg=150; thrust_scale=10000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `near_crossing`; candidate for planner timing/commitment search. |
| PS-014 | `r0_over_target=1.00; initial_velocity_angle_deg=150; thrust_scale=10000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline crossing-producing recoverable | Phase36B baseline crossing case; use as regression guard for Phase34 handoff. |
| PS-015 | `r0_over_target=1.02; initial_velocity_angle_deg=150; thrust_scale=10000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `over_conservative_transfer`; candidate for planner commitment search. |
| PS-016 | `r0_over_target=0.98; initial_velocity_angle_deg=165; thrust_scale=10000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `near_crossing`; candidate for planner timing/commitment search. |
| PS-017 | `r0_over_target=1.00; initial_velocity_angle_deg=165; thrust_scale=10000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline crossing-producing recoverable | Phase36B baseline crossing case; use as regression guard for Phase34 handoff. |
| PS-018 | `r0_over_target=1.02; initial_velocity_angle_deg=165; thrust_scale=10000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `over_conservative_transfer`; candidate for planner commitment search. |
| PS-019 | `r0_over_target=0.98; initial_velocity_angle_deg=170; thrust_scale=10000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `near_crossing`; candidate for planner timing/commitment search. |
| PS-020 | `r0_over_target=1.00; initial_velocity_angle_deg=170; thrust_scale=10000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline crossing-producing recoverable | Phase36B baseline crossing case; use as regression guard for Phase34 handoff. |
| PS-021 | `r0_over_target=1.02; initial_velocity_angle_deg=170; thrust_scale=10000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `over_conservative_transfer`; candidate for planner commitment search. |
| PS-022 | `r0_over_target=0.98; initial_velocity_angle_deg=175; thrust_scale=10000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `near_crossing`; candidate for planner timing/commitment search. |
| PS-023 | `r0_over_target=1.00; initial_velocity_angle_deg=175; thrust_scale=10000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline crossing-producing recoverable | Phase36B baseline crossing case; use as regression guard for Phase34 handoff. |
| PS-024 | `r0_over_target=1.02; initial_velocity_angle_deg=175; thrust_scale=10000; target_radius_scale=1.0` | Phase34 `radius_priority` terminal/post-cross controller | baseline non-crossing | Phase36C label: `over_conservative_transfer`; candidate for planner commitment search. |

## Planner Search Discipline

The first planner search should be small and parameterized:

- `coast_duration`
- `radial_push_timing`
- `radial_push_magnitude`
- `tangential_shaping_magnitude`

Primary evaluation should separate:

- geometric crossing
- Phase34-compatible crossing
- recoverable crossing
- simulator success label
- overspeed and instability

The first search should not use MPC-lite, PPO, 3D dynamics, SPICE, C++, or direct trajectory optimization. The immediate research question is whether coarse upstream timing and shaping parameters can create new target-radius crossings that hand off into the fixed Phase34 terminal controller.
