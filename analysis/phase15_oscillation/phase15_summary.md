# Phase 15 Oscillation-Inducing Reachability Controller

## Scope

- 2D Python-only explicit oscillation-inducing reachability test.
- CAPTURE/LOCK logic, physics, and success definition are unchanged.
- DESCENT uses bounded radial sign/damping control outside the pre-window.
- When already moving toward the target radius with acceptable angular momentum, radial gains are reduced instead of using full correction.
- A small tangential correction preserves angular momentum without continuous full thrust.
- Near the target radius, DESCENT blends back to the Phase 7.6 `soft_linear_3e4` behavior.

## Reduced-Grid Result

| Controller | CAPTURE | Success | Near-miss | Total |
|---|---:|---:|---:|---:|
| `baseline_soft_linear_3e4` | 12 | 12 | 0 | 48 |
| `phase14_event_guidance` | 12 | 12 | 0 | 48 |
| `phase15_oscillation` | 12 | 12 | 0 | 48 |

- Capture improvement: `0`.
- Success improvement: `0`.
- Crossing-case improvement: `0`.
- Total radius crossings: baseline `15`, Phase 15 `15`.
- Turning points detected in Phase 15: `0`.
- No-CAPTURE cases: baseline `36`, planner `36`.
- Overspeed terminations: Phase 14 `0`, Phase 15 `36`.
- Overspeed reduction vs Phase 14: `-36`.
- Event counters inherited from scaffold: events `0`, burns `0`.
- Mean max speed ratio: `1.6803`.
- Mean energy error ratio: initial `0.0219`, final `1.9548`.
- Mean angular momentum ratio: initial `0.9476`, final `0.9818`.
- Planner mode usage: `{"oscillation": 3071, "soft_window": 2188}`.

## Answers

1. Does oscillation control create new crossings? `no`. Total crossings change from `15` to `15`.
2. Does capture increase? It `does not improve` CAPTURE: CAPTURE changes by `0` versus baseline.
3. Does overspeed remain controlled? `no`. Phase 15 overspeed count is `36`.
4. Does this produce usable events for future event guidance? `no`. Turning points observed: `0`.
5. Main limitation: bounded radial oscillation can create or damp radial motion, but without a timed transfer target it can still miss CAPTURE or terminate before producing useful crossings.

## Artifacts

- `comparison.csv`
- `radius_oscillation_examples.png`
- `radial_velocity_examples.png`
- `crossing_count_comparison.png`
- `trajectories/*.png`
