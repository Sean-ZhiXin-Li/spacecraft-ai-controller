# Phase 17 Physics-Consistent Orbit Construction

## Scope

- 2D Python-only explicit elliptical-transfer guidance test.
- CAPTURE/LOCK logic, physics, and success definition are unchanged.
- DESCENT constructs a transfer ellipse from the initial radius to the target radius.
- The controller guides toward transfer-orbit energy and angular momentum.
- Tangential thrust handles energy/angular-momentum correction; a small radial term biases toward the target radius.
- Near the target radius, DESCENT blends back to the Phase 7.6 `soft_linear_3e4` behavior.

## Reduced-Grid Result

| Controller | CAPTURE | Success | Near-miss | Total |
|---|---:|---:|---:|---:|
| `baseline_soft_linear_3e4` | 12 | 12 | 0 | 48 |
| `phase16_trajectory_tracking` | 12 | 12 | 0 | 48 |
| `phase17_elliptical_orbit` | 12 | 12 | 0 | 48 |

- Capture improvement: `0`.
- Success improvement: `0`.
- Crossing-case improvement: `0`.
- Total radius crossings: baseline `15`, Phase 17 `15`.
- Turning points detected in Phase 17: `0`.
- No-CAPTURE cases: baseline `36`, planner `36`.
- Overspeed terminations: Phase 16 `0`, Phase 17 `9`.
- Overspeed reduction vs Phase 16: `-9`.
- Event counters inherited from scaffold: events `0`, burns `0`.
- Mean max speed ratio: `1.1609`.
- Mean energy error ratio: initial `0.0219`, final `0.4911`.
- Mean angular momentum ratio: initial `0.9476`, final `0.7984`.
- Mean transfer energy error ratio: initial `0.0110`, final `0.4839`.
- Mean transfer angular-momentum error ratio: initial `0.0450`, final `0.1910`.
- Planner mode usage: `{"coast": 2697394, "elliptical_guidance": 13752, "soft_window": 2188}`.

## Answers

1. Does physics-consistent orbit improve capture? It `does not improve` CAPTURE: CAPTURE changes by `0` versus baseline.
2. Does turning point count increase? `no`. Turning points observed: `0`.
3. Does system now form real orbital shape? `not reliably`. Transfer energy/angular-momentum errors changed from `0.0110/0.0450` to `0.4839/0.1910`.
4. Is feasibility the missing piece? `partially`. The target is physically consistent, but the controller must still phase the transfer so target-radius crossing occurs inside the CAPTURE window.

## Main Limitation

Matching transfer-orbit invariants is not enough by itself. The spacecraft can approach the intended energy/angular-momentum surface without placing the radius crossing at the right time and radial velocity for CAPTURE.

## Artifacts

- `comparison.csv`
- `energy_convergence.png`
- `angular_momentum_convergence.png`
- `orbit_shape_examples.png`
- `crossing_count_comparison.png`
- `trajectories/*.png`
