# Phase 16 Explicit Trajectory Construction + Tracking

## Scope

- 2D Python-only explicit radial trajectory construction and tracking test.
- CAPTURE/LOCK logic, physics, and success definition are unchanged.
- DESCENT defines `r_desired(t) = target_radius + A(t) * sin(omega * t)` with slowly decaying amplitude.
- The controller tracks desired radius and radial velocity with bounded PD control.
- A conservative tangential term maintains circular-speed consistency.
- Near the target radius, DESCENT blends back to the Phase 7.6 `soft_linear_3e4` behavior.

## Reduced-Grid Result

| Controller | CAPTURE | Success | Near-miss | Total |
|---|---:|---:|---:|---:|
| `baseline_soft_linear_3e4` | 12 | 12 | 0 | 48 |
| `phase15_oscillation` | 12 | 12 | 0 | 48 |
| `phase16_trajectory_tracking` | 12 | 12 | 0 | 48 |

- Capture improvement: `0`.
- Success improvement: `0`.
- Crossing-case improvement: `0`.
- Total radius crossings: baseline `15`, Phase 16 `15`.
- Turning points detected in Phase 16: `0`.
- No-CAPTURE cases: baseline `36`, planner `36`.
- Overspeed terminations: Phase 15 `36`, Phase 16 `0`.
- Overspeed reduction vs Phase 15: `36`.
- Event counters inherited from scaffold: events `0`, burns `0`.
- Mean max speed ratio: `1.0538`.
- Mean absolute tracking error ratio: `0.0225`.
- Mean energy error ratio: initial `0.0219`, final `0.0475`.
- Mean angular momentum ratio: initial `0.9476`, final `1.0000`.
- Planner mode usage: `{"soft_window": 2188, "trajectory_tracking": 3600000}`.

## Answers

1. Does explicit trajectory construction create crossings? `no`. Total crossings change from `15` to `15`.
2. Does capture increase? It `does not improve` CAPTURE: CAPTURE changes by `0` versus baseline.
3. Does turning point count increase? `no`. Turning points observed: `0`.
4. Does tracking stay bounded without overspeed? `yes`. Phase 16 overspeed count is `0`.
5. Is trajectory design more effective than direct control? `partially on boundedness, not on reachability`. The decisive CAPTURE change is `0` versus baseline and `0` versus Phase 15.

## Main Limitation

The sinusoidal radial reference creates an explicit path, but a bounded trackable timescale is too slow to generate new target-radius crossings on this horizon. Faster references become energetically unsafe.

## Artifacts

- `comparison.csv`
- `radius_tracking_examples.png`
- `radial_velocity_tracking_examples.png`
- `tracking_error_examples.png`
- `crossing_count_comparison.png`
- `trajectories/*.png`
