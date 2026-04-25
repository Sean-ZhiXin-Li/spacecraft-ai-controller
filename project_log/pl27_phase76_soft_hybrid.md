# PL27 - Phase 7.6 Soft Hybrid

## Goal

Fix the Phase 7.5 hard-switch failure by continuously blending pre-window shaping with adaptive WS behavior.

## Method

Defined `pre_action` as `prewindow_radial_medium` and `ws_action` as `adaptive_soft`, then applied:

```text
action = alpha * pre_action + (1 - alpha) * ws_action
```

The blended action was normalized after blending. Five alpha schedules were tested on the same 270-regime grid.

## Key Result

Best variant: `soft_linear_3e4`.

- 217 successes out of 270.
- 217 CAPTURE entries.
- 8 near-misses.

This beats `adaptive_soft`, `prewindow_radial_medium`, and `hard_hybrid_1e4`.

## Interpretation

Continuous coordination between pre-window shaping and WS behavior is the strongest tested structure. The controller needs both approach-geometry shaping and smooth window-entry behavior.

## Limitations

The result is still scoped to the 2D local grid. It is not evidence for 3D, multi-orbit, or learned-policy transfer.

## Next Step

Freeze `soft_linear_3e4` as the current best 2D explicit controller result, document the evidence trail, and avoid further algorithm expansion until repository cleanup and reproducibility are stable.
