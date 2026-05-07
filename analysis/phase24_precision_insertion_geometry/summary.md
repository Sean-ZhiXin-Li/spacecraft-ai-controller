# Phase 24 Precision Orbital Insertion Geometry Solver

## Scope

- Python-only 2D staged transfer controller.
- Burn A and coast arc are kept from Phase 22.
- Burn B is replaced with a precision insertion-geometry solver.
- Solver parameters include burn timing, insertion mode, burn angle, magnitude, and duration.
- Physics, reward, PPO, and CAPTURE/LOCK logic are unchanged.

## Reduced-Grid Result

| Controller | Recoverable crossings | CAPTURE | Success | Insertion windows | Conversion rate | Overspeed |
|---|---:|---:|---:|---:|---:|---:|
| Phase 7.6 `baseline_soft_linear_3e4` | 0 | 12 | 12 | 0 | 0.000 | 0 |
| Phase 20 `phase20_predictive_planner` | 0 | 12 | 12 | 0 | 0.000 | 0 |
| Phase 21 `phase21_orbital_transfer_planner` | 0 | 12 | 12 | 0 | 0.000 | 0 |
| Phase 22 `phase22_two_burn_transfer` | 0 | 12 | 12 | 12 | 0.000 | 0 |
| Phase 23 `phase23_windowed_insertion_solver` | 0 | 12 | 12 | 12 | 0.000 | 0 |
| Phase 24 `phase24_precision_insertion_geometry` | 0 | 12 | 12 | 12 | 0.000 | 0 |

## Phase 24 Precision Insertion Diagnostics

- Burn A steps: `720`.
- Burn B steps: `108`.
- Precision insertion evaluations: `2700`.
- Insertion attempts: `12`.
- Insertion windows: `12`.
- Successful insertion windows: `12`.
- Handoff-entered cases: `24`.
- Insertion window conversion rate: `0.000`.
- Mean chosen burn timing: `75.00` steps.
- Mean basin score: `-0.1103`.
- Insertion mode win counts: `{"balanced_circularization": 6, "tangential_priority": 1, "timing_first": 5}`.
- Stage usage: `{"burn_a": 720, "burn_b": 108, "burn_b_timing_wait": 900, "coast_arc": 2399508, "phase76_handoff": 1207785}`.
- Mean final energy error: `0.5986`.
- Mean final angular momentum error: `0.5762`.

## Research Answers

1. Is Burn B timing the dominant missing variable? `no` by conversion-rate criterion.
2. Can insertion geometry outperform brute-force local Burn B? `no`.
3. Which insertion mode is strongest? `balanced_circularization`.
4. Can Phase 24 produce recoverable crossings? `no`.
5. If not, current bottleneck: `Phase 7.6 basin too narrow or true physical limitation`.

## Success Criteria

- Minimum, insertion conversion > Phase 23: `not met`.
- Moderate, recoverable crossings > 0: `not met`.
- Strong, CAPTURE > Phase 7.6: `not met`.
- Major, success > Phase 7.6: `not met`.
- Research success, partial recoverable insertion: `not met`.

## Honesty Note

- This script does not force a positive insertion result.
- Burn A and coast arc are inherited from Phase 22; only Burn B is changed.
- Timing, mode, post-burn geometry, and basin-score diagnostics are reported even for failures.
- Overspeed, failed windows, and no-gain outcomes are preserved in the CSV and summary.

## Artifacts

- `phase24_results.csv`
- `insertion_timing_distribution.png`
- `insertion_mode_comparison.png`
- `recoverable_basin_scores.png`
- `post_burn_geometry_analysis.png`
- `trajectories/phase24_representative_*.png`