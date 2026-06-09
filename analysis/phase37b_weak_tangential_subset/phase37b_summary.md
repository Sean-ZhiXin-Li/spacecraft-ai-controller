# Phase37B Weak Tangential Subset Diagnostic

## Scope

- Subset diagnostic only: 4 selected Phase37A-improved non-crossing cases plus 8 Phase36B baseline crossing-producing regression cases.
- Two settings: `early_commit_low_radial_only` and `early_commit_low_plus_weak_tangential`.
- Fixed Phase34 `radius_priority` terminal/post-cross controller after first target-radius crossing.
- No full 24-case tangential grid, no MPC, no RL, and no coast-duration search.
- This remains a simplified 2D orbital-control sandbox result, not real spacecraft validation.

## Control Perturbation

- Radial overlay: `u_r = clamp(0.055 * direction_to_target - 0.16 * vr_ratio, -0.055, 0.055)`.
- Weak tangential overlay: `u_t = clamp(-0.06 * vt_error_ratio, -0.020, 0.020)`.
- Combined diagnostic action norm limit: `0.060`.
- Weak tangential shaping is gated to the four selected `r0 = 1.02`, `thrust = 10000` cases and only before first crossing while outside the target-radius corridor.

## Aggregate Results

| Setting | Rollouts | Selected crossings | Selected recoverable crossings | Regression crossings | Regression recoverable crossings | Overspeed | Instability |
|---|---:|---:|---:|---:|---:|---:|---:|
| `early_commit_low_radial_only` | 12 | 0 | 0 | 4 | 4 | 0 | 0 |
| `early_commit_low_plus_weak_tangential` | 12 | 0 | 0 | 4 | 4 | 0 | 0 |

## Required Report

- Total rollouts: `24`.
- Selected-case new crossings under weak tangential shaping: `0 / 4`.
- Selected-case recoverable crossings under weak tangential shaping: `0 / 4`.
- Regression crossing preservation under weak tangential shaping: `4 / 8` crossings and `4 / 8` recoverable crossings, compared with the Phase36B baseline requirement of `8 / 8`.
- Overspeed count: `0`.
- Instability count: `0`.
- Closest-approach comparison versus radial-only on selected cases: `3` improved, `0` worsened, `1` unchanged.

## Closest-Approach Deltas

- `selected_01`: delta `0.00000000e+00` (unchanged)
- `selected_02`: delta `-1.04197805e-08` (improved)
- `selected_03`: delta `-1.27103552e-08` (improved)
- `selected_04`: delta `-1.36119823e-08` (improved)

## Decision

- Diagnostic classification: `unsafe globally`.
- Interpretation: The diagnostic action did not preserve the Phase36B baseline regression crossing set.

Phase37B should not be expanded unless this subset result creates at least one new selected-case crossing without damaging the regression set. If it only improves closest approach, that is a weak signal requiring analysis before adding any larger search dimension.

## Artifacts

- `phase37b_results.csv`
- `phase37b_summary.md`
- `phase37b_comparison.png`
