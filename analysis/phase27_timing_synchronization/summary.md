# Phase 27 VT-Radius Timing Synchronization

## Scope

- Python-only 2D reduced-grid benchmark.
- Burn A, coast, Burn B concept, physics, recoverability thresholds, CAPTURE, and LOCK are unchanged.
- Phase 26 vt-aware baseline is loaded from CSV and preserved.
- Phase 27 changes Burn B scheduling to target arrival-state synchronization at crossing.

## Controller Comparison

| Controller | Crossings | Recoverable | CAPTURE | Success | Mean crossing vt | Mean crossing vr | Mean sync error | Best sync error | Mean timing offset | Overspeed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `phase26_vt_aware_scoring` | 12 | 0 | 12 | 12 | 0.7889 | 0.0218 | 3.5336 | 2.4482 | nan | 0 |
| `phase27_predicted_cross_vt_targeting` | 12 | 0 | 12 | 12 | 0.7889 | 0.0218 | 3.5336 | 2.1197 | nan | 0 |
| `phase27_delayed_sync_burn` | 12 | 0 | 12 | 12 | 0.7889 | 0.0218 | 3.5336 | 2.1197 | nan | 0 |
| `phase27_split_phase_sync_burn` | 12 | 0 | 12 | 12 | 0.7889 | 0.0218 | 3.5336 | 2.1197 | nan | 0 |
| `phase27_adaptive_sync_corridor` | 12 | 0 | 12 | 12 | 0.7889 | 0.0218 | 3.5336 | 2.1197 | nan | 0 |

## Synchronization Findings

- `sync_error` is max-normalized against the true recoverability thresholds, so Band S means every radius/vr/vt component is within the current basin.
- Band thresholds: S `<=1.0`, A `<=1.5`, B `<=2.0`, C `<=3.0`.
- Phase 26 mean crossing sync error: `3.5336`.
- Best Phase 27 mean crossing sync error: `3.5336` from `phase27_predicted_cross_vt_targeting`.
- Strongest best pre-cross sync band rank reached: `1`.
- Recoverable crossings: Phase 26 `0`, best Phase 27 `0`.
- Timing offset was measurable: `no`.

## Research Answers

1. Can timing-aware shaping reduce crossing-state mismatch? `no`.
2. Does vt synchronization improve recoverability? `no`.
3. Is timing more important than burn strength? `not proven`.
4. Is split-burn architecture superior? `no`.
5. Dominant bottleneck now: `Burn B timing has no leverage on crossing-producing cases`.

## Success Criteria

- Minimum, reduce crossing sync error: `not met`.
- Moderate, reach tighter sync bands: `not met`.
- Strong, first recoverable crossing: `not met`.
- Major, increase CAPTURE or success: `not met`.

## Honesty Note

- The summary reports actual crossing-state sync, not local Burn B vt improvement.
- No recoverability, CAPTURE, or success is inferred from relaxed thresholds.
- Negative outcomes are retained as structural evidence.

## Artifacts

- `phase27_results.csv`
- `phase27_sync_analysis.csv`
- `sync_error_distribution.png`
- `crossing_state_alignment.png`
- `timing_offset_histogram.png`
- `predicted_vs_actual_crossing.png`
- `controller_comparison.png`
- `sync_band_progression.png`
- `trajectories/phase27_representative_*.png`