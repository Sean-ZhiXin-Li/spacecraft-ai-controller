# Phase 30 Burn-A Endpoint Search

## Scope

- Research context written to `research_context.md` before controller conclusions.
- Python-only 2D Burn-A endpoint search benchmark on the same reduced 48-case grid.
- Physics, CAPTURE/LOCK, thresholds, reward, PPO, coast, and Burn B concept are unchanged.
- Phase 30 searches bounded Burn-A endpoint candidates over duration, thrust norm, radial bias, and tangential bias.
- Endpoint candidates mapped: `6912`. Selected endpoint previews with passive crossing: `9`.

## Controller Results

| Controller | Crossings | Near recoverable | Recoverable | CAPTURE | Success | Windows | Good windows | Dead windows | Mean sync | Burn-A abs h | Burn-A abs E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `phase30_endpoint_crossing_quality_search` | 12 | 5 | 0 | 12 | 12 | 18 | 0 | 18 | 3.5336 | 0.1203 | 0.1633 |
| `phase30_endpoint_energy_h_target` | 12 | 5 | 0 | 12 | 12 | 15 | 0 | 15 | 3.5336 | 0.1270 | 0.1759 |
| `phase30_endpoint_grid_search` | 12 | 5 | 0 | 12 | 12 | 15 | 0 | 15 | 3.5336 | 0.1183 | 0.2436 |
| `phase30_phase22_baseline` | 12 | 5 | 0 | 12 | 12 | 12 | 0 | 12 | 3.5336 | 0.5370 | 0.7408 |
| `phase30_phase29_best_baseline` | 12 | 5 | 0 | 12 | 12 | 18 | 0 | 18 | 3.5336 | 0.0000 | 0.0594 |

## Research Answers

1. Does endpoint search improve crossings? `no`. Best crossing count: `12` from `phase30_endpoint_crossing_quality_search`.
2. Does endpoint search reduce dead windows? `no`. Best dead-window count: `12` from `phase30_phase22_baseline`.
3. Does endpoint search improve near-recoverable crossings? `no`. Best near-recoverable count: `5`.
4. Does endpoint search improve CAPTURE? `no`.
5. Are some endpoint manifolds clearly better? `weak/no` by downstream metrics; best mean-quality variant `phase30_endpoint_crossing_quality_search`.
6. Is Burn-A endpoint more important than Burn B? `not proven`.
7. What is Phase 31? If Phase 30 does not move crossing-state structure, Phase 31 should test a global orbital transfer solver or Lambert-like endpoint planner rather than another local Burn-B layer.

## Success Criteria

- Minimum, Burn-A endpoint manifold mapped: `met`.
- Moderate, dead windows reduced: `not met`.
- Strong, crossing count improves: `not met`.
- Major, near recoverable improves: `not met`.
- Breakthrough, recoverable/CAPTURE improves: `not met`.

## Honest Interpretation

- Best mean quality variant: `phase30_endpoint_crossing_quality_search`.
- Compared with Phase 29 best baseline, endpoint search improvement in recoverable/CAPTURE/success: `no`.
- Endpoint search maps the post-Burn-A manifold and tests whether selected endpoints alter downstream structure.
- If performance does not move, the current bounded endpoint family is insufficient; this points toward a global transfer solver rather than more late insertion tuning.

## Artifacts

- `research_context.md`
- `phase30_results.csv`
- `phase30_endpoint_dataset.csv`
- `burn_a_endpoint_manifold.png`
- `endpoint_energy_vs_h.png`
- `endpoint_periapsis_apoapsis_map.png`
- `endpoint_quality_distribution.png`
- `phase30_vs_phase29_comparison.png`
- `controller_comparison.png`
