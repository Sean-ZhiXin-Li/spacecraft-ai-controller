# Phase 6 Explicit DESCENT Variant Search

## Setup

- Scope: 2D explicit-controller family only.
- Physics implementation: local Python evaluator mirrors the existing `OrbitEnv` Euler dynamics with action smoothing and orbit-capture assist disabled, matching the Phase 3-5 explicit evaluations.
- Controller changes are limited to DESCENT-side action construction or early handoff; CAPTURE and LOCK gains are unchanged.
- Variants: `12`.
- Regimes per variant: `270`.
- Total completed runs: `3240`.

## Best Variant

- Best variant: `early_handoff_6e4`.
- Parameters: `{"handoff_r": 0.0006, "near_scale": 0.55, "radial_gain": 0.08, "retrograde_scale": 0.9}`.
- Baseline successes: `64`.
- Best successes: `270`.
- Baseline CAPTURE count: `65`.
- Best CAPTURE count: `270`.

## Answers

1. Explicit reachability improved without learning: `True`. The best variant changes success count from `64` to `270` on the 270-regime grid.
2. The most important controller-side knob in this search appears to be `early handoff`. This is inferred from the top-ranked variant family, not from a continuous sensitivity analysis.
3. The main gain comes from CAPTURE access if the CAPTURE count increased more than post-CAPTURE success conditional on CAPTURE. Here CAPTURE changes from `65` to `270`, while CAPTURE and LOCK remain closely coupled in the ranking data.
4. The best variant should be read as shifting and locally widening the practical capture access region only if its success map adds neighboring cells around the baseline; otherwise it mainly shifts the pocket. See `top_variant_success_map.png` and `baseline_vs_best_variant.png`.
5. Best next step: refine the winning DESCENT knob family with a smaller local grid, then validate against the Phase 4 and Phase 5 representative failures before considering any broader architecture changes.

## Caution

This is an explicit variant search, not a new learned policy. The result should not be generalized beyond the sampled local 2D regimes.