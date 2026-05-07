# Phase 26 Tangential Velocity Corridor Engineering

## Scope

- Python-only 2D benchmark using the same reduced 48-case grid per Phase 26 variant.
- Phase 22 Burn A and coast concepts are preserved.
- CAPTURE, LOCK, physics, PPO, reward, and recoverability thresholds are unchanged.
- Burn B is replaced with deterministic vt-corridor shaping variants.
- Phase 24 is included as the loaded baseline, not overwritten.

## Controller Comparison

| Controller | Crossings | Recoverable | CAPTURE | Success | Min pre-cross vt | Mean crossing vt | Min crossing vt | Best distance | Overspeed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `phase24_precision_insertion_geometry` | 12 | 0 | 12 | 12 | nan | 0.7889 | 0.6067 | nan | 0 |
| `phase26_vt_aware_scoring` | 12 | 0 | 12 | 12 | 0.0000 | 0.7889 | 0.6067 | 2.4482 | 0 |
| `phase26_two_step_corridor` | 12 | 0 | 12 | 12 | 0.0000 | 0.7889 | 0.6067 | 2.4482 | 0 |
| `phase26_burn_hold_burn` | 12 | 0 | 12 | 12 | 0.0000 | 0.7889 | 0.6067 | 2.4482 | 0 |
| `phase26_stronger_tangential` | 12 | 0 | 12 | 12 | 0.0000 | 0.7889 | 0.6067 | 2.4482 | 0 |

## VT Corridor Findings

- Recoverable vt threshold: `0.2500`.
- Soft corridor bands: `3.0x=0.7500`, `2.0x=0.5000`, `1.5x=0.3750`, `1.0x=0.2500`.
- Best vt variant: `phase26_two_step_corridor` with min |vt error ratio| `0.0000`.
- Best recoverability-distance variant: `phase26_vt_aware_scoring` with min distance `2.4482`.
- Strongest best-band rank reached: `4`.
- Phase 24 mean crossing |vt error ratio|: `0.7889`.
- Best Phase 26 mean crossing |vt error ratio|: `0.7889`.
- Best Phase 26 mean pre/post Burn B vt correction: `0.3717`.

## Research Answers

1. Can vt-focused shaping reduce vt mismatch? `yes locally, no at crossing`.
2. Does reduced vt mismatch create recoverable crossings? `no`.
3. Is Burn B architecture too shallow? `yes`.
4. Is multi-step shaping required? `possible`.
5. Dominant bottleneck now: `vt-radius timing synchronization`.

## Success Criteria

- Minimum, reduce vt mismatch measurably: `met locally, not at crossing`.
- Moderate, reach tighter vt corridor bands: `met pre-cross, not at crossing`.
- Strong, first recoverable crossing: `not met`.
- Major, increase CAPTURE or success: `not met`.

## Honesty Note

- Positive CAPTURE or success is not inferred from relaxed thresholds.
- vt-band progress is reported separately from recoverability and CAPTURE.
- Overspeed and unstable outcomes are retained in both CSV outputs.

## Artifacts

- `phase26_results.csv`
- `phase26_vt_corridor_analysis.csv`
- `vt_error_over_time.png`
- `crossing_vt_distribution.png`
- `vt_corridor_band_progression.png`
- `controller_comparison.png`
- `pre_post_burn_vt_correction.png`
- `recoverability_distance_progression.png`
- `trajectories/phase26_representative_*.png`