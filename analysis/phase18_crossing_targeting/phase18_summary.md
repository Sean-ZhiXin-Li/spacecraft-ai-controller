# Phase 18 Crossing-State Targeting

## Scope

- 2D Python-only explicit crossing-state targeting test.
- CAPTURE/LOCK logic, physics, and success definition are unchanged.
- Far from the target, DESCENT reuses Phase 17 elliptical guidance.
- When approaching target-radius crossing, DESCENT switches to targeting mode.
- Targeting prioritizes radial-velocity damping, then softly corrects tangential velocity.
- Near the target radius, DESCENT blends back to the Phase 7.6 `soft_linear_3e4` behavior.

## Reduced-Grid Result

| Controller | CAPTURE | Success | Near-miss | Total |
|---|---:|---:|---:|---:|
| `baseline_soft_linear_3e4` | 12 | 12 | 0 | 48 |
| `phase17_elliptical_orbit` | 12 | 12 | 0 | 48 |
| `phase18_crossing_targeting` | 12 | 12 | 0 | 48 |

- Capture improvement: `0`.
- Success improvement: `0`.
- Crossing-case improvement: `0`.
- Total radius crossings: baseline `15`, Phase 18 `15`.
- Turning points detected in Phase 18: `0`.
- No-CAPTURE cases: baseline `36`, planner `36`.
- Overspeed terminations: Phase 17 `9`, Phase 18 `0`.
- Overspeed reduction vs Phase 17: `9`.
- Event counters inherited from scaffold: events `0`, burns `0`.
- Mean max speed ratio: `0.9930`.
- Mean |crossing v_r| ratio: Phase 17 `0.0120`, Phase 18 `0.0120`.
- Mean |crossing v_t error| ratio: Phase 17 `0.8721`, Phase 18 `0.8721`.
- Planner mode usage: `{"coast": 3595486, "elliptical_guidance": 2365, "soft_window": 2188, "targeting": 2149}`.

## Answers

1. Does targeting reduce v_r at crossing? `no`. Mean |v_r| ratio changes from `0.0120` to `0.0120`.
2. Does capture increase? It `does not improve` CAPTURE: CAPTURE changes by `0` versus baseline.
3. Are crossings now usable for CAPTURE? `no`. Crossing quality changes did not translate into extra CAPTURE entries unless CAPTURE count increases.
4. Is state-targeting the missing piece? `partially`. The crossing state matters, but this explicit targeting layer still needs enough authority and correct phasing before the crossing.

## Main Limitation

The targeting mode only activates when a crossing is already approaching. It cannot create new crossings from regimes that never approach the target radius, and aggressive damping can still trade crossing quality against reachability.

## Artifacts

- `comparison.csv`
- `crossing_state_histogram.png`
- `vr_vs_time_near_crossing.png`
- `vt_error_vs_time.png`
- `trajectories/*.png`