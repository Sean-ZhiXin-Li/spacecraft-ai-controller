# Phase 35 Crossing Basin Expansion

## Scope

- Python-only 2D explicit-controller benchmark.
- Phase35 acts only before the first target-radius crossing.
- Phase34 `radius_priority` post-cross synchronization remains the terminal controller after crossing.
- Physics, CAPTURE/LOCK thresholds, reward assumptions, and recoverability thresholds are unchanged.
- Benchmark: same 24-case reduced grid used in Phase34.

## Results

| Upstream variant | Cases | Geometric crossings | Recoverable crossings | Simulator success label | Non-crossing | Mean crossing potential | Mean min radius error ratio | Overspeed | Instability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_phase34` | 24 | 8 | 8 | 8 | 16 | 0.8156 | 0.013311 | 0 | 0 |
| `radial_energy_push` | 24 | 0 | 0 | 0 | 24 | 0.8108 | 0.013321 | 5 | 0 |
| `tangential_corridor_entry` | 24 | 0 | 0 | 0 | 24 | 0.7714 | 0.013333 | 0 | 0 |
| `predictive_crossing_bias` | 24 | 8 | 8 | 8 | 16 | 0.8823 | 0.012909 | 0 | 0 |

## Scientific Questions

1. Did Phase35 increase crossing-producing cases? `no`. Best crossing count: `8 / 24` from `baseline_phase34`; baseline Phase34 count: `8 / 24`.
2. Which failure mode dominates the 16 non-crossing cases? `near_crossing, over_conservative_transfer` with `8` cases each in the baseline diagnosis.
3. Did upstream expansion preserve Phase34 downstream recoverability? `yes` under this benchmark aggregation.
4. Is the bottleneck pre-cross energy, tangential corridor, or initial geometry? Current labels point primarily to `near_crossing, over_conservative_transfer` rather than post-cross recovery.
5. Should Phase36 integrate a planner, MPC-lite, or stronger transfer family? `yes`. The pre-cross module needs a more expressive transfer planner than these local biases.

## Interpretation

Phase35 diagnosed the upstream bottleneck but did not expand the crossing basin.

Phase35 keeps the Phase34 terminal law fixed. Any change in crossing count comes from upstream routing, not from relaxed recoverability or post-cross threshold changes.

The simulator success label, CAPTURE, and LOCK are state-machine/result labels in this 2D sandbox. They are not real flight-validation states.

## Artifacts

- `phase35_results.csv`
- `non_crossing_diagnosis.csv`
- `failure_mode_summary.md`
- `phase35_vs_phase34.md`
- `crossing_count_comparison.png`
- `non_crossing_failure_modes.png`
- `crossing_potential_distribution.png`
- `min_radius_error_by_variant.png`
- `phase34_terminal_handoff_examples.png`
