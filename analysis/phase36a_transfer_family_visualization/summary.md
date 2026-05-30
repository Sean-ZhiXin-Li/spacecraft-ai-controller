# Phase36A Transfer Family Visualization

## Scope

- Visualization-first experiment, not an optimization phase.
- Phase34 `radius_priority` remains the post-cross terminal controller.
- Physics, CAPTURE/LOCK thresholds, rewards, and recoverability thresholds are unchanged.
- Default run uses three representative cases: one crossing case, one `near_crossing` non-crossing case, and one `over_conservative_transfer` case selected from Phase35 outputs.

## Representative Cases

| Case label | r0 / target | Initial velocity angle | Thrust scale |
|---|---:|---:|---:|
| `representative_crossing` | 1.00 | 150.0 | 8000 |
| `representative_near_crossing` | 0.98 | 150.0 | 10000 |
| `representative_over_conservative_transfer` | 1.02 | 175.0 | 8000 |

## Results

| Transfer family | Cases | Crossings | Recoverable crossings | Mean min radius error ratio | Mean crossing potential | Mean crossing sync | Overspeed | Instability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_phase34` | 3 | 1 | 1 | 0.013309 | 0.8209 | 3.0007 | 0 | 0 |
| `spiral_approach` | 3 | 1 | 1 | 0.013306 | 0.8613 | 4.0058 | 0 | 0 |
| `delayed_crossing` | 3 | 0 | 0 | 0.013331 | 0.8647 | N/A | 0 | 0 |
| `energy_bleed_then_cross` | 3 | 0 | 0 | 0.013333 | 0.7480 | N/A | 3 | 0 |
| `overshoot_return` | 3 | 0 | 0 | 0.013333 | 0.7481 | N/A | 3 | 0 |
| `grazing_corridor` | 3 | 1 | 1 | 0.013289 | 0.8782 | 6.1565 | 0 | 0 |
| `two_stage_transfer` | 3 | 0 | 0 | 0.013333 | 0.7550 | N/A | 3 | 0 |

## Scientific Answers

1. Which transfer families are visually distinct? `spiral_approach, delayed_crossing, energy_bleed_then_cross, overshoot_return, grazing_corridor, two_stage_transfer` show distinct geometry or metric behavior relative to the baseline in this small subset.
2. Which families approach target radius but fail to commit? `18` family-case rows stayed within 2.5% radius error without crossing.
3. Which families produce violent crossings? `1` family-case rows crossed with high radial ratio or high sync error.
4. Which families produce smoother crossing states? `1` family-case rows crossed with lower radial and tangential error by the simple Phase36A filter.
5. Which families seem worth testing in Phase36B? `delayed_crossing, grazing_corridor, spiral_approach`.
6. Does this support the hypothesis that crossing-generation is trajectory-family geometry? `yes, cautiously`. The families produce visibly different pre-cross paths and different crossing-state quality, even when they do not improve recoverable count.

## Interpretation

Phase36A clarified transfer geometry but did not improve crossing count.

The result should be read as a geometry map, not a success benchmark. A family that creates a visually distinct path or a better handoff state is useful even if it does not improve the small-subset count.

## Artifacts

- `phase36a_family_results.csv`
- `family_case_notes.md`
- `phase36a_vs_phase35.md`
- `family_trajectory_overlay.png`
- `radius_vs_time_by_family.png`
- `vr_vs_time_by_family.png`
- `vt_error_vs_time_by_family.png`
- `crossing_state_scatter_by_family.png`
- `family_geometry_map.png`