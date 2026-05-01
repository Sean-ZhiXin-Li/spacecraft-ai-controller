# Phase 10 Failure-Conditioned Reachability Controller

## Scope

- 2D Python-only explicit controller test.
- CAPTURE/LOCK logic, physics, and success definition are unchanged.
- DESCENT chooses among `outer_orbit`, `inner_orbit`, `angle_misaligned`, and `near_window` strategies.
- `near_window` uses the existing Phase 7.6 `soft_linear_3e4` behavior unchanged.

## Reduced-Grid Result

| Controller | CAPTURE | Success | Near-miss | Total |
|---|---:|---:|---:|---:|
| `baseline_soft_linear_3e4` | 12 | 12 | 0 | 48 |
| `phase10_conditioned` | 12 | 12 | 0 | 48 |

- Capture improvement: `0`.
- Success improvement: `0`.
- Most frequently triggered failure type: `angle_misaligned`.
- Highest capture-rate triggered mode: `near_window`.

## Answers

1. Does failure-conditioned control improve reachability? It `does not improve` reachability on this reduced Phase 9 grid: CAPTURE changes by `0` and success changes by `0`.
2. Which failure type benefits most? By triggered-mode capture rate, `near_window` is strongest. By frequency, `angle_misaligned` is the dominant diagnosed mode.
3. Is geometry-aware control necessary? The Phase 8 and Phase 9 evidence says geometry diagnosis is necessary for understanding the failures. This first simple conditioned controller is not sufficient proof that the tested strategy set solves them.
4. Does this outperform single-controller design? `no` on the reduced grid. The comparison should be treated as a controller-structure diagnostic, not a global 2D solution.

## Artifacts

- `comparison.csv`
- `mode_usage.json`
- `success_by_mode.png`
- `capture_improvement_vs_baseline.png`