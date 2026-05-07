# Phase 20 Predictive Crossing-State Planning

## Scope

- 2D Python-only explicit short-horizon predictive planner.
- Physics, CAPTURE/LOCK logic, and success definition are unchanged.
- DESCENT replans every `40` steps over a `260`-step short horizon.
- Candidate actions include coast, prograde, retrograde, radial, and diagonal low-thrust actions.
- Scoring uses target-radius crossing quality, post-cross recoverability, and safety penalties.
- Near the target radius, the planner hands off to the existing Phase 7.6 `soft_linear_3e4` controller.

## Reduced-Grid Result

| Controller | CAPTURE | Success | Crossing cases | Recoverable crossings | Overspeed | Total |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_soft_linear_3e4` | 12 | 12 | 12 | 0 | 0 | 48 |
| `phase19_minimal_transfer` | 0 | 0 | 0 | 0 | 12 | 48 |
| `phase20_predictive_planner` | 12 | 12 | 12 | 0 | 0 | 48 |

- Capture improvement vs baseline: `0`.
- Success improvement vs baseline: `0`.
- Crossing-case improvement vs baseline: `0`.
- Recoverable-crossing improvement vs baseline: `0`.
- Capture improvement vs Phase 19: `12`.
- Success improvement vs Phase 19: `12`.
- Crossing-case improvement vs Phase 19: `12`.
- Total radius crossings: baseline `15`, Phase 19 `0`, Phase 20 `15`.
- Overspeed terminations: baseline `0`, Phase 19 `12`, Phase 20 `0`.
- Mean crossing score: baseline `0.7122`, Phase 20 `0.7122`.
- Mean recoverability score: baseline `0.0237`, Phase 20 `0.0237`.
- Phase 20 replans: `82566`.
- Phase 20 candidate simulations: `743094`.
- Phase 20 predictive steps: `263240`.
- Phase 20 coast steps: `3039350`.
- Phase 20 Phase 7.6 handoff steps: `2188`.
- Phase 20 mode usage: `{"coast": 3039350, "phase76_handoff": 2188, "predictive_descent": 263240}`.

## Answers

1. Can predictive short-horizon search produce more recoverable crossings than reactive local control? `no`. Recoverable crossings change from `0` to `0`.
2. Does recoverability-aware planning outperform crossing-only planning? `not directly tested`. This script implements recoverability-aware scoring but does not include a crossing-only ablation.
3. Can planning improve no-CAPTURE scenarios without degrading stable cases? `no`. CAPTURE changes by `0`, success changes by `0`.
4. Does planning outperform reactive control? `no` on this reduced grid.

## Interpretation

Phase 20 is the first guidance-layer test that chooses actions by forecasting future crossing quality and post-cross recoverability. A positive result requires more than crossing the target radius; the planner must create states that the Phase 7.6 handoff can exploit.

## Artifacts

- `reduced_grid_results.csv`
- `summary.md`
- `capture_crossing_comparison.png`
- `trajectories/*.png`