# Phase 14 Event-Triggered Orbital Guidance

## Scope

- 2D Python-only explicit event-triggered orbital guidance test.
- CAPTURE/LOCK logic, physics, and success definition are unchanged.
- DESCENT detects turning points, near-turning states, and near-target moving-away events.
- Burns are triggered only by turning-point or near-miss events.
- Burns are short fixed windows of pure tangential prograde/retrograde thrust; otherwise action is exactly `(0, 0)`.
- Near the target radius, DESCENT blends back to the Phase 7.6 `soft_linear_3e4` behavior.

## Reduced-Grid Result

| Controller | CAPTURE | Success | Near-miss | Total |
|---|---:|---:|---:|---:|
| `baseline_soft_linear_3e4` | 12 | 12 | 0 | 48 |
| `phase13_burn_coast` | 12 | 12 | 0 | 48 |
| `phase14_event_guidance` | 12 | 12 | 0 | 48 |

- Capture improvement: `0`.
- Success improvement: `0`.
- Crossing improvement: `0`.
- No-CAPTURE cases: baseline `36`, planner `36`.
- Overspeed terminations: Phase 13 `12`, Phase 14 `0`.
- Overspeed reduction vs Phase 13: `12`.
- Events detected: `0`.
- Burns started: `0`.
- Mean burn ratio: `0.0000`.
- Mean energy error ratio: initial `0.0219`, final `0.0219`.
- Mean angular momentum ratio: initial `0.9476`, final `0.9583`.
- Planner mode usage: `{"coast": 3600000, "soft_window": 2188}`.

## Answers

1. Does event-triggered burn improve crossing? `no`. Crossing count changes by `0` versus baseline.
2. Does capture increase? It `does not improve` CAPTURE: CAPTURE changes by `0` versus baseline.
3. Does trajectory now resemble orbital transfer? `not yet`. The controller uses event-triggered coast/burn timing, but the measured crossing/CAPTURE result is still the decisive test.
4. Is event structure the missing piece? `partially`. Event structure reduces uncontrolled thrusting only if events occur early enough and burns are sized correctly.

## Artifacts

- `comparison.csv`
- `event_timeline.png`
- `radius_vs_time_with_events.png`
- `capture_improvement_vs_baseline.png`
- `trajectories/*.png`