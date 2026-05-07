# Phase 23 Windowed Orbital Insertion Solver

## Scope

- Python-only 2D staged transfer controller.
- Burn A and coast arc are kept from Phase 22.
- Burn B is upgraded to a deterministic windowed insertion solver.
- Physics, reward, PPO, and CAPTURE/LOCK logic are unchanged.

## Reduced-Grid Result

| Controller | Recoverable crossings | CAPTURE | Success | Insertion windows | Conversion rate | Overspeed |
|---|---:|---:|---:|---:|---:|---:|
| Phase 7.6 `baseline_soft_linear_3e4` | 0 | 12 | 12 | 0 | 0.000 | 0 |
| Phase 20 `phase20_predictive_planner` | 0 | 12 | 12 | 0 | 0.000 | 0 |
| Phase 21 `phase21_orbital_transfer_planner` | 0 | 12 | 12 | 0 | 0.000 | 0 |
| Phase 22 `phase22_two_burn_transfer` | 0 | 12 | 12 | 12 | 0.000 | 0 |
| Phase 23 `phase23_windowed_insertion_solver` | 0 | 12 | 12 | 12 | 0.000 | 0 |

## Phase 23 Burn B Solver Diagnostics

- Burn A steps: `720`.
- Burn B steps: `552`.
- Burn B solver evaluations: `3864`.
- Insertion attempts: `12`.
- Insertion windows: `12`.
- Successful insertion windows: `12`.
- Handoff-entered cases: `24`.
- Insertion window conversion rate: `0.000`.
- Stage usage: `{"burn_a": 720, "burn_b": 552, "coast_arc": 2399508, "phase76_handoff": 1208241}`.
- Mean final energy error: `0.5983`.
- Mean final angular momentum error: `0.5793`.

## Research Answers

1. Can the windowed solver convert Phase 22 insertion windows into recoverable crossings? `no`.
2. Does Phase 23 improve CAPTURE? `no`. CAPTURE changed from `12` to `12`.
3. Does Phase 23 improve success? `no`. Success changed from `12` to `12`.
4. Current bottleneck: `windowed Burn B insertion geometry`.

## Success Criteria

- Minimum, crossings > Phase 7.6: `not met`.
- Moderate, recoverable crossings > 0: `not met`.
- Strong, CAPTURE > Phase 7.6: `not met`.
- Major, success > Phase 7.6: `not met`.
- Research success, insertion window conversion rate > Phase 22: `not met`.

## Honesty Note

- This script does not force a positive insertion result.
- Burn A and coast arc are inherited from Phase 22; only Burn B is changed.
- Overspeed, failed windows, and no-gain outcomes are preserved in the CSV and summary.

## Artifacts

- `phase23_results.csv`
- `transfer_stage_usage.png`
- `insertion_window_success.png`
- `energy_error_vs_time.png`
- `angular_momentum_error_vs_time.png`
- `trajectories/phase23_representative_*.png`