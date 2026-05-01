# Phase 12 Angular Momentum + Orbit Intersection Planning

## Scope

- 2D Python-only explicit orbit-intersection planning test.
- CAPTURE/LOCK logic, physics, and success definition are unchanged.
- DESCENT computes specific energy, angular momentum `L = x * vy - y * vx`, radial velocity, and tangential velocity.
- The planner thrusts toward target-radius crossing while trying to keep angular momentum near `r * v_circular`.
- Near the target radius, DESCENT blends back to the Phase 7.6 `soft_linear_3e4` behavior.

## Reduced-Grid Result

| Controller | CAPTURE | Success | Near-miss | Total |
|---|---:|---:|---:|---:|
| `baseline_soft_linear_3e4` | 12 | 12 | 0 | 48 |
| `phase11_energy_planner` | 12 | 12 | 0 | 48 |
| `phase12_orbit_planner` | 12 | 12 | 0 | 48 |

- Capture improvement: `0`.
- Success improvement: `0`.
- Crossing improvement: `0`.
- No-CAPTURE cases: baseline `36`, planner `36`.
- Planner overspeed terminations: `36`.
- Mean energy error ratio: initial `0.0219`, final `1.8995`.
- Mean angular momentum ratio: initial `0.9476`, final `0.9759`.
- Planner mode usage: `{"orbit_intersection": 643, "soft_window": 2188}`.

## Answers

1. Does angular-momentum control improve crossing? `no`. Crossing count changes by `0` on this reduced grid.
2. Does capture rate increase? It `does not improve` CAPTURE: CAPTURE changes by `0`.
3. Does it reduce `no_capture_access`? `no`. No-CAPTURE cases change from `36` to `36`.
4. Is orbit geometry now correct? `not yet`. The planner explicitly controls radial crossing direction and angular momentum, but the measured CAPTURE result is the deciding test.

## Artifacts

- `comparison.csv`
- `angular_momentum_vs_time.png`
- `crossing_success_examples.png`
- `capture_improvement_vs_baseline.png`
- `trajectories/*.png`