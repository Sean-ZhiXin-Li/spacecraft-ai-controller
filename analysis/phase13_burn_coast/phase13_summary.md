# Phase 13 Burn-Coast Guidance

## Scope

- 2D Python-only explicit burn-coast guidance test.
- CAPTURE/LOCK logic, physics, and success definition are unchanged.
- DESCENT computes radius, radial velocity, tangential velocity, specific energy, and angular momentum.
- The controller burns only when energy error is large, radial sign is wrong, or a turning point is near.
- Burn windows are capped and followed by cooldown coast windows; coast action is exactly `(0, 0)`.
- Near the target radius, DESCENT blends back to the Phase 7.6 `soft_linear_3e4` behavior.

## Reduced-Grid Result

| Controller | CAPTURE | Success | Near-miss | Total |
|---|---:|---:|---:|---:|
| `baseline_soft_linear_3e4` | 12 | 12 | 0 | 48 |
| `phase12_orbit_planner` | 12 | 12 | 0 | 48 |
| `phase13_burn_coast` | 12 | 12 | 0 | 48 |

- Capture improvement: `0`.
- Success improvement: `0`.
- Crossing improvement: `0`.
- No-CAPTURE cases: baseline `36`, planner `36`.
- Overspeed terminations: Phase 12 `36`, Phase 13 `12`.
- Overspeed reduction vs Phase 12: `24`.
- Mean burn ratio: `0.2275`.
- Mean energy error ratio: initial `0.0219`, final `0.6380`.
- Mean angular momentum ratio: initial `0.9476`, final `0.9752`.
- Planner mode usage: `{"burn": 443, "coast": 2399914, "cooldown": 60, "soft_window": 2188}`.

## Answers

1. Does burn-coast reduce overspeed? `yes`. Overspeed changes from `36` to `12`.
2. Does capture increase? It `does not improve` CAPTURE: CAPTURE changes by `0` versus baseline.
3. Does trajectory look more structured? `yes`. The burn/coast ratio is `0.2275`, and the schedule plots show explicit coast intervals.
4. Is timing now the key missing factor? `partially`. Timing helps only if it also produces target-radius crossing and CAPTURE.

## Artifacts

- `comparison.csv`
- `burn_schedule.png`
- `speed_vs_time.png`
- `energy_vs_time.png`
- `capture_improvement_vs_baseline.png`
- `trajectories/*.png`