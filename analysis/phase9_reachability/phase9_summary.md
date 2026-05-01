# Phase 9 Pre-CAPTURE Reachability Analysis

## Scope

- 2D Python-only analysis of Phase 8 no-CAPTURE failures.
- Phase 7.6 `soft_linear_3e4` logic is used as the baseline and is not modified in place.
- The only tested controller-side change is a DESCENT-only energy-directional reachability term outside the Phase 7 pre-window band.
- CAPTURE/LOCK logic, physics, and success definition are unchanged.

## Phase 8 Failure Structure

- Phase 8 rows analyzed: `1296`.
- CAPTURE entries in Phase 8: `265`.
- No-CAPTURE access cases (`success == False` and `capture_entered == False`): `1031`.
- Dominant factor by success-rate spread: `r0_over_target`.
- Success-rate spread by r0: `0.840`.
- Success-rate spread by angle: `0.031`.
- Success-rate spread by thrust: `0.208`.

## Small-Grid Reachability Comparison

| Controller | Capture | Success | Near-miss | Total |
|---|---:|---:|---:|---:|
| `baseline_soft_linear_3e4` | 12 | 12 | 0 | 48 |
| `reach_energy_directional_light` | 12 | 12 | 0 | 48 |
| `reach_energy_directional_medium` | 12 | 12 | 0 | 48 |

- Best by capture/success/near-miss ranking: `baseline_soft_linear_3e4`.
- Capture improvement over baseline on the reduced grid: `0` regimes.
- Reachability improved: `no`.
- Representative no-CAPTURE trajectory plots saved: `7`.

## Answers

1. Why do most trajectories fail before CAPTURE? Most expanded-grid failures never cross the target radius, so the phase machine remains in DESCENT until `max_steps`. The trajectory diagnostics show radius staying on the same side of the target instead of entering the CAPTURE trigger.
2. Which factor matters most? `r0_over_target` matters most in this Phase 8 map. The r0 grouping is the largest contrast: local starts near `1.0` are reachable, while most broader radius offsets are not.
3. What pattern defines `no_capture_access`? It is a target-radius crossing failure: `capture_entered == False`, usually `crossing_occurs == False`, with long integrations ending at `max_steps` rather than CAPTURE/LOCK instability.
4. Does the new variant improve capture rate? `no`. The best reduced-grid controller is `baseline_soft_linear_3e4` with `12` captures versus baseline `12`.
5. Is improvement due to better trajectory shaping or just parameter scaling? The tested energy-directional extension did not improve the best capture count, so there is no positive mechanism claim beyond the Phase 8 reachability diagnosis.
6. The next step toward global reachability should stay in 2D and map crossing geometry directly: classify whether each no-CAPTURE case needs radius raising, radius lowering, or angular-momentum correction before adding more controller structure.

## Artifacts

- `no_capture_dataset.csv`
- `capture_map_r0_vs_angle_thrust10000.png`
- `capture_map_r0_vs_angle_thrust15000.png`
- `no_capture_rate_by_r0.png`
- `no_capture_rate_by_angle.png`
- `no_capture_rate_by_thrust.png`
- `trajectories/*.png`
- `reachability_comparison.csv`
