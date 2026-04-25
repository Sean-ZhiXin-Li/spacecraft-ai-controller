# Phase 7.6 Soft Hybrid Controller Summary

## Setup

- Scope: 2D Python-only continuous blend between `prewindow_radial_medium` and `adaptive_soft`.
- Grid: same 270 regimes from Phase 6.5-7.5.
- Soft rule: `action = alpha * pre_action + (1 - alpha) * ws_action`, normalized after blending.
- No hard switching, new phases, learning, or physics changes.

## Ranking

- `soft_linear_3e4` schedule `linear` mid `3.0e-04` width `0.0e+00`: success `217`, CAPTURE `217`, near-miss `8`.
- `soft_linear_2e4` schedule `linear` mid `2.0e-04` width `0.0e+00`: success `205`, CAPTURE `205`, near-miss `0`.
- `soft_sigmoid_mid3e4` schedule `sigmoid` mid `3.0e-04` width `7.0e-05`: success `180`, CAPTURE `180`, near-miss `32`.
- `soft_linear_5e4` schedule `linear` mid `5.0e-04` width `0.0e+00`: success `176`, CAPTURE `176`, near-miss `9`.
- `soft_sigmoid_mid2e4` schedule `sigmoid` mid `2.0e-04` width `5.0e-05`: success `165`, CAPTURE `165`, near-miss `32`.

## Reference Comparison

- `adaptive_soft`: success `172`, CAPTURE `172`, near-miss `44`.
- `prewindow_radial_medium`: success `209`, CAPTURE `209`, near-miss `56`.
- `hard_hybrid_1e4`: success `170`, CAPTURE `170`, near-miss `42`.
- Best soft hybrid: `soft_linear_3e4` with success `217`, CAPTURE `217`, near-miss `8`.

## Answers

1. Does soft hybrid recover the high reachability of prewindow_radial_medium? `yes`. Delta vs prewindow_radial_medium is success `+8`.
2. Does it reduce near-miss compared with prewindow_radial_medium? `yes`. Delta vs prewindow_radial_medium is near-miss `-48`; delta vs adaptive_soft is `-36`.
3. Does it beat hard hybrid? `yes`. Delta vs hard_hybrid_1e4 is success `+47`, near-miss `-34`.
4. Which alpha schedule works best? `soft_linear_3e4` (`linear`, mid `3.0e-04`, width `0.0e+00`).
5. Final project conclusion: orbit insertion control on this 2D grid is structurally sensitive to approach geometry before window entry. Pre-window radial shaping creates the largest reachability gain; adaptive WS improves local window behavior; hard boundaries are brittle; continuous action coordination is the right structure to test when combining explicit phases, but the final choice remains empirical because reachability and near-miss stability can still trade off. Current best trade-off assessment: `the soft hybrid combines the desired strengths on this grid`.