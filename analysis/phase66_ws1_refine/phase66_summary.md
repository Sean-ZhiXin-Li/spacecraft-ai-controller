# Phase 6.6 WS-1 Refinement Summary

## Setup

- Scope: local 2D WS-1 parameter refinement only; no learning, PPO retraining, physics changes, 3D, or multi-orbit cases.
- Variant family: original WS-1 plus one-knob-at-a-time changes for `near_band_ratio`, `reduced_retro_scale`, and `score_energy`.
- Total variants: `8`; regimes per variant: `270`.
- Ranking order: success count, CAPTURE count, lower near-miss count, lower mean minimum radius error.

## Result

- Original WS-1: success `157`, CAPTURE `157`, near-miss `42`, mean min |radius error| `1.645e+08`.
- Best variant: `retro_scale_0p80` with success `172`, CAPTURE `172`, near-miss `47`, mean min |radius error| `1.275e+08`.
- Delta vs original: success `+15`, CAPTURE `+15`, near-miss `+5`, mean min |radius error| `-3.702e+07`.

## Knob Sensitivity

- `near_band_ratio`: success range `34`, capture range `34`, near-miss range `33`; best one-factor setting `original_ws1`.
- `reduced_retro_scale`: success range `43`, capture range `43`, near-miss range `18`; best one-factor setting `retro_scale_0p80`.
- `score_energy`: success range `8`, capture range `8`, near-miss range `4`; best one-factor setting `original_ws1`.

## Answers

1. Small parameter refinement improved WS-1 on this grid. The best strict-success delta is `+15`.
2. The most consequential knob by one-factor success/capture/near-miss spread is `reduced_retro_scale`.
3. The best variant changes success by `+15`, CAPTURE by `+15`, and near-misses by `+5`; this indicates a real CAPTURE/success change rather than an inferred post-CAPTURE effect.
4. The success set is `shifted` relative to original WS-1: retained `143`, gained `29`, lost `14` regimes.
5. Best next step: stay in this same 2D Python evaluator and run a second compact refinement centered only on the best one-factor setting, with at most a few adjacent values for the dominant knob and no changes to CAPTURE/LOCK physics.

## Caution

These results are local to the Phase 6.5 270-regime grid. The script intentionally avoids broad search and does not change the environment or learned policies.