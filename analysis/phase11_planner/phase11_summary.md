# Phase 11 Energy-Guided Trajectory Planning + Tracking

## Scope

- 2D Python-only explicit trajectory-planning test.
- CAPTURE/LOCK logic, physics, and success definition are unchanged.
- DESCENT computes specific orbital energy, compares it with the target circular-orbit energy, and thrusts prograde or retrograde accordingly.
- Near the target radius or near the target energy, DESCENT blends back to the Phase 7.6 `soft_linear_3e4` behavior.

## Reduced-Grid Result

| Controller | CAPTURE | Success | Near-miss | Total |
|---|---:|---:|---:|---:|
| `baseline_soft_linear_3e4` | 12 | 12 | 0 | 48 |
| `phase11_energy_planner` | 12 | 12 | 0 | 48 |

- Capture improvement: `0`.
- Success improvement: `0`.
- No-CAPTURE cases: baseline `36`, planner `36`.
- Planner overspeed terminations: `0`.
- Mean energy error ratio: initial `0.0219`, final `0.2356`.
- Planner mode usage: `{"energy_correction": 2045228, "soft_window": 1556960}`.

## Answers

1. Does energy-based planning improve capture? It `does not improve` CAPTURE on this reduced grid: CAPTURE changes by `0`.
2. Does it reduce `no_capture_access`? `no`. No-CAPTURE cases change from `36` to `36`.
3. Does it avoid overspeed? `yes`. The planner produced `0` overspeed terminations.
4. Is trajectory shaping more effective than local control? On this small grid, `not yet`. Energy-guided planning is more interpretable than local classification, but this lightweight version still needs better angular-momentum and radial-crossing planning before it can beat the Phase 7.6 single-controller baseline.

## Artifacts

- `comparison.csv`
- `energy_error_summary.png`
- `capture_improvement_vs_baseline.png`
- `trajectories/*.png`