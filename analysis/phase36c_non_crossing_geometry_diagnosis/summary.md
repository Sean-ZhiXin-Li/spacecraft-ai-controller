# Phase36C Non-Crossing Geometry Diagnosis

## Scope

This diagnostic package reads existing Phase36B CSV outputs. It does not run a new controller, change physics, change thresholds, tune family gains, or modify Phase34 terminal behavior.

The analysis is scoped to the simplified 2D orbital-control sandbox.

## Which Baseline Cases Failed To Cross?

`baseline_phase34` had `16 / 24` non-crossing cases in Phase36B.

### `near_crossing`

- r0=0.98, angle=150.0, thrust=8000.0
- r0=0.98, angle=150.0, thrust=10000.0
- r0=0.98, angle=165.0, thrust=8000.0
- r0=0.98, angle=165.0, thrust=10000.0
- r0=0.98, angle=170.0, thrust=8000.0
- r0=0.98, angle=170.0, thrust=10000.0
- r0=0.98, angle=175.0, thrust=8000.0
- r0=0.98, angle=175.0, thrust=10000.0

### `over_conservative_transfer`

- r0=1.02, angle=150.0, thrust=8000.0
- r0=1.02, angle=150.0, thrust=10000.0
- r0=1.02, angle=165.0, thrust=8000.0
- r0=1.02, angle=165.0, thrust=10000.0
- r0=1.02, angle=170.0, thrust=8000.0
- r0=1.02, angle=170.0, thrust=10000.0
- r0=1.02, angle=175.0, thrust=8000.0
- r0=1.02, angle=175.0, thrust=10000.0

## Failure Label Distribution

| Label source | Failure label | Count |
|---|---|---:|
| baseline non-crossing cases | `near_crossing` | 8 |
| baseline non-crossing cases | `over_conservative_transfer` | 8 |
| all Phase36B families on baseline non-crossing set | `near_crossing` | 56 |
| all Phase36B families on baseline non-crossing set | `over_conservative_transfer` | 8 |

The baseline non-crossing set is split between near-crossing behavior and over-conservative transfer behavior. Near-crossing should be treated as useful information: the geometry approaches the target-radius event but fails to commit under the tested transfer families.

## Family Delta Summary

| Transfer family | Improved closest approach | Improved crossing potential | Worsened geometry | Changed failure label |
|---|---:|---:|---:|---:|
| `grazing_corridor` | 12 | 15 | 5 | 8 |
| `redesigned_delayed_crossing` | 8 | 16 | 8 | 8 |
| `spiral_approach` | 8 | 16 | 8 | 8 |

## Required Questions

- Are the baseline failures mostly near-crossing or over-conservative? They are tied: `near_crossing=8`, `over_conservative_transfer=8`.
- Did any non-baseline family improve closest approach without crossing? `yes`; `28` non-baseline family-case rows improved `min_abs_radius_error_ratio` relative to baseline.
- Did any family improve crossing potential without crossing? `yes`; `47` non-baseline family-case rows improved `best_crossing_potential` relative to baseline.
- Did any family worsen geometry? `yes`; `21` non-baseline family-case rows worsened either closest approach or crossing potential.
- Did any family change failure labels? `yes`; `24` non-baseline family-case rows changed the baseline diagnostic label.
- What does this imply about crossing-generation? The remaining bottleneck is upstream crossing-generation, and the Phase36B family variants changed local geometry metrics without creating new target-radius crossings.
- What exact planner-level search should be tried next? Run a small coarse grid over `coast_duration`, `radial_push_timing`, `radial_push_magnitude`, and `tangential_shaping_magnitude`, with Phase34 fixed as the terminal controller.
- Why is this not yet MPC-lite? The immediate question is not online replanning; it is whether coarse transfer timing and shaping parameters can create crossings at all in the non-crossing cases.

## Bottleneck Interpretation

Phase36B already showed that post-cross recovery is not the limiting factor for crossing-producing cases. Phase36C shows that the non-crossing cases can move in closest-approach and crossing-potential metrics without producing new Phase34-compatible crossings. That points toward a planner-level trajectory search space rather than another manually named local family.

## Artifacts

- `non_crossing_case_set.csv`
- `family_behavior_on_non_crossing_cases.csv`
- `non_crossing_family_delta.csv`
- `planner_search_space.md`
