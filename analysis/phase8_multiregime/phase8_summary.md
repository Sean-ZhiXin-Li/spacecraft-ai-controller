# Phase 8 Multi-Regime Generalization Map

## Setup

- Scope: 2D Python-only expanded regime map for the Phase 7.6 `soft_linear_3e4` controller.
- Controller structure, physics, CAPTURE/LOCK logic, and success definition are unchanged from Phase 7.6.
- Grid size: `1296` completed regimes.
- This is not a final project result; it is a current 2D Phase 8 generalization diagnostic.

## Aggregate Result

- Successes: `220` / `1296` (`0.170`).
- CAPTURE entries: `265` / `1296` (`0.204`).
- LOCK entries: `208` / `1296`.
- Radius crossings: `265` / `1296`.
- Near-miss failures: `68`.
- Dominant failure mode: `no_capture_access`.

## Best Slices

- Best thrust slice: `8000` with success rate `0.222`.
- Best angle slice: `180` with success rate `0.191`.
- Best r0 slice: `1` with success rate `0.840`.

## Answers

1. Does `soft_linear_3e4` generalize beyond the Phase 7.6 local grid? It generalizes only if the expanded map retains a meaningful success region. On this Phase 8 grid, success is `220` / `1296` (`0.170`), so the evidence should be read at that scope.
2. Where does it work best? The strongest one-dimensional slices are thrust `8000`, angle `180`, and r0 `1` by success rate.
3. Where does it fail? Failures concentrate in the dominant class `no_capture_access`; the plots break this down by r0, angle, and thrust.
4. Are failures mostly pre-CAPTURE access failures or post-CAPTURE instability? In this classification they are mostly `pre-CAPTURE access failures`: no_capture_access `1008`, unstable_after_capture `0`.
5. Does the result support the continuous-coordination insight? It `partially supports` the insight in 2D: the same coordinated controller was tested without structural changes, and the map shows where the structure remains useful versus where expanded initial conditions exceed its reach.
6. Phase 9 should stay in 2D and analyze the boundary of the successful Phase 8 regions before adding controller complexity: focus on failure-mode-specific diagnostics, especially whether no-capture cases need broader pre-window shaping or whether captured failures need CAPTURE/LOCK robustness checks.

## Artifacts

- `phase8_grid.csv`
- `phase8_failure_modes.json`
- `success_rate_by_thrust.png`
- `success_rate_by_angle.png`
- `success_rate_by_r0.png`
- `capture_rate_by_thrust.png`
- `failure_mode_distribution.png`
- `success_map_r0_vs_angle_thrust10000.png`
- `success_map_r0_vs_angle_thrust15000.png`
- `min_radius_error_map_r0_vs_angle_thrust10000.png`
- `capture_vs_success_summary.png`