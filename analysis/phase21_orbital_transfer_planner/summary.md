# Phase 21 Orbital-Class Transfer Planner

## Scope

- Python-only 2D explicit transfer-regime planner.
- Physics, CAPTURE/LOCK logic, success definition, and prior phase outputs are unchanged.
- DESCENT uses orbital diagnostics: specific energy, angular momentum, semi-major-axis proxy, periapsis/apoapsis proxy, radial velocity, and tangential velocity error.
- The planner uses burn/coast transfer modes instead of short-horizon local action search.

## Reduced-Grid Result

| Controller | Crossing cases | Recoverable crossings | CAPTURE | Success | Overspeed | Mean energy err | Mean h err | Burn ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 7.6 `baseline_soft_linear_3e4` | 12 | 0 | 12 | 12 | 0 | 0.7068 | 0.6533 | 0.000 |
| Phase 20 `phase20_predictive_planner` | 12 | 0 | 12 | 12 | 0 | 1.0699 | 0.5279 | 0.000 |
| Phase 21 `phase21_orbital_transfer_planner` | 12 | 0 | 12 | 12 | 0 | 0.2243 | 0.0864 | 0.031 |

## Phase 21 Diagnostics

- Mean initial |energy error|: `0.0219`; mean final |energy error|: `0.2243`.
- Mean initial |angular momentum error|: `0.0454`; mean final |angular momentum error|: `0.0864`.
- Mean crossing score: `0.7122`.
- Mean recoverability score: `0.0237`.
- Mode usage: `{"angular_momentum_shape": 74759, "coast_to_crossing": 2252700, "energy_shape": 74916, "phase76_handoff": 2188, "safety_guard": 1197625}`.
- Transfer burn/coast ratio: `0.0312`.

## Research Answers

1. Can orbital-class transfer planning create new crossings beyond Phase 7.6? `no`. Crossing cases changed from `12` to `12`.
2. Can energy/angular-momentum shaping create recoverable crossings? `no`. Recoverable crossings: `0`.
3. Does Phase 21 improve CAPTURE or success? CAPTURE improvement: `0`; success improvement: `0`.
4. Is the bottleneck now transfer design, capture design, or physical reachability? Current evidence points to `transfer design`.

## Success Criteria

- Minimum success, crossing cases > Phase 7.6: `not met`.
- Moderate success, recoverable crossings > 0: `not met`.
- Strong success, CAPTURE > Phase 7.6: `not met`.
- Major success, success > Phase 7.6: `not met`.

## Honesty Note

- Energy moved toward target values: `no`.
- Angular momentum moved toward target values: `no`.
- Negative or mixed results are treated as diagnostic evidence, not forced into a positive claim.

## Artifacts

- `phase21_results.csv`
- `transfer_mode_usage.png`
- `energy_error_vs_time.png`
- `angular_momentum_error_vs_time.png`
- `capture_crossing_comparison.png`
- `trajectories/phase21_representative_*.png`