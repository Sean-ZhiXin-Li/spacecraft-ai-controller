# Phase 22 Two-Burn Transfer Architecture

## Scope

- Python-only 2D staged transfer controller.
- Physics, reward, PPO, and CAPTURE/LOCK logic are unchanged.
- DESCENT uses a mission-style state machine: Burn A, coast arc, Burn B, Phase 7.6 handoff.

## Reduced-Grid Result

| Controller | Crossing cases | Recoverable crossings | CAPTURE | Success | Overspeed |
|---|---:|---:|---:|---:|---:|
| Phase 7.6 `baseline_soft_linear_3e4` | 12 | 0 | 12 | 12 | 0 |
| Phase 20 `phase20_predictive_planner` | 12 | 0 | 12 | 12 | 0 |
| Phase 21 `phase21_orbital_transfer_planner` | 12 | 0 | 12 | 12 | 0 |
| Phase 22 `phase22_two_burn_transfer` | 12 | 0 | 12 | 12 | 0 |

## Phase 22 Transfer Diagnostics

- Burn A steps: `720`.
- Burn B steps: `552`.
- Insertion attempts: `12`.
- Insertion windows: `12`.
- Successful insertion windows: `12`.
- Handoff-entered cases: `24`.
- Stage usage: `{"burn_a": 720, "burn_b": 552, "coast_arc": 2399508, "phase76_handoff": 1208241}`.
- Mean final energy error: `0.5983`.
- Mean final angular momentum error: `0.5793`.

## Research Answers

1. Can Burn A create new orbital corridors? `yes`. Insertion windows: `12`.
2. Does coast arc improve geometry more than local planning? `no` by crossing-count criterion.
3. Can Burn B create recoverable crossings? `no`. Recoverable crossings: `0`.
4. Does Phase 22 finally exceed Phase 7.6? `no`.
5. If not, current bottleneck: `insertion geometry`.

## Success Criteria

- Minimum, crossings > Phase 7.6: `not met`.
- Moderate, recoverable crossings > 0: `not met`.
- Strong, CAPTURE > Phase 7.6: `not met`.
- Major, success > Phase 7.6: `not met`.
- Research success, new insertion windows or recoverable crossings: `met`.

## Honesty Note

- This script does not force a positive insertion result.
- Burn A, coast arc, Burn B, and handoff metrics are reported separately so partial progress is visible.
- Overspeed, failed windows, and no-gain outcomes are preserved in the CSV and summary.

## Artifacts

- `phase22_results.csv`
- `transfer_stage_usage.png`
- `insertion_window_success.png`
- `energy_error_vs_time.png`
- `angular_momentum_error_vs_time.png`
- `trajectories/phase22_representative_*.png`