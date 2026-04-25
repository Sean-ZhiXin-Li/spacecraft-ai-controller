# PL25 - Phase 7 Pre-Window Shaping

## Goal

Improve spacecraft approach geometry before entering the WS band by adding a DESCENT-only pre-window shaping stage.

## Method

Kept the Phase 6.7 evaluator, grid, physics, WS behavior, and CAPTURE/LOCK logic. Added pre-window shaping between relative radius errors `8e-5` and `5e-4`, including radial, energy-guided, and geometry-guided variants.

## Key Result

`prewindow_radial_medium` was the best Phase 7 variant:

- `adaptive_soft`: 172 successes, 172 CAPTURE entries, 44 near-misses.
- `prewindow_radial_medium`: 209 successes, 209 CAPTURE entries, 56 near-misses.

## Interpretation

Pre-window radial shaping widened reachability and confirmed that many remaining failures were caused by poor approach geometry before WS activation.

## Limitations

The reachability gain came with more near-misses. Pre-window shaping alone did not provide the same stable window-entry behavior as `adaptive_soft`.

## Next Step

Combine pre-window shaping with adaptive WS behavior and test whether the strengths can be preserved together.
