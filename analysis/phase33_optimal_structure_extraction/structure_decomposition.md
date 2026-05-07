# Phase 33 Structural Trajectory Decomposition

Best trajectory: `recoverability_target` / `baseline_crossing_high_angle`.

## Phase Segments

| Phase | Step range | Radius trend | vr sign | vt start | vt end | Mean thrust | Mean thrust angle | Smoothness |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Initial geometry shaping | 0-28 | increasing | mostly outward | -0.003805 | -0.152726 | 0.018092 | -30.42 | 0.00001340 |
| Radial approach | 29-61 | decreasing | mostly inward | -0.157716 | -0.312864 | 0.015174 | -8.52 | 0.00000257 |
| Crossing preparation | 62-80 | decreasing | mostly inward | -0.317392 | -0.394467 | 0.013097 | -5.04 | 0.00000004 |
| Crossing | 81-86 | decreasing | mostly inward | -0.398441 | -0.418314 | 0.012120 | -5.04 | 0.00000000 |
| Energy / tangential correction | 87-511 | decreasing | mostly inward | -0.422289 | -0.005790 | 0.007764 | 103.02 | 0.00000004 |
| Post-cross stabilization | 512-512 | flat | mostly outward | 0.000011 | 0.000011 | 0.017687 | 175.03 | 0.00000000 |

## Timing Facts

- vr sign flip steps: `[28, 512]`.
- vr flip count: `2`.
- crossing step: `81`.
- best recoverability step: `512`.
- best recoverability occurs `after first crossing`.
- minimum |vt error| step: `512`.
- vt is minimized `after crossing or no crossing`.
- crossing-state sync error: `1.676881`.
- crossing-state distance to recoverable: `2.313443`.

## Extracted Behavior

- The optimal trajectory uses very small continuous thrust rather than large impulse-like burns.
- It crosses target radius early, but the crossing state is still outside the true recoverability basin.
- It then uses a long post-cross correction arc to bring vt, vr, and radius into simultaneous alignment.
- The control law is smooth and low authority; the important structure is timing and phase alignment, not thrust magnitude.

## Implementation Note

- Phase 32 labels `recoverable_crossing` when a trajectory both crosses target radius and reaches a recoverable state somewhere on the horizon.
- For this best case, the first crossing itself is not the best recoverability state; Phase 33 therefore treats the first crossing and late recoverable endpoint separately.