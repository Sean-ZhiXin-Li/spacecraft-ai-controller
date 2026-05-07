# Phase 25 Recoverability Basin Mapping and CAPTURE Threshold Ablation

## Scope

- CSV-first structural analysis using Phase 22, Phase 23, and Phase 24 outputs.
- No trajectories are rerun and no controller, physics, Burn A, coast arc, or Burn B logic is changed.
- Phase 22/23 insertion windows did not store full geometry; those rows are marked as `insertion_window_uninstrumented`.

## Dataset

- Total extracted states: `108`.
- Numeric geometry states: `60`.
- Original recoverable states: `0`.
- Median distance to recoverability: `3.8142`.
- Minimum distance to recoverability: `2.4531`.
- Dominant failure variable: `tangential_velocity`.
- Dominant failure cluster, all states: `uninstrumented_window`.
- Dominant failure cluster, numeric states only: `energy_mismatch`.

## Threshold Ablation

- Baseline threshold recoverable states: `0`.
- Best relaxed threshold recoverable states: `15`.
- Best relaxed threshold recoverable cases: `5`.
- Best threshold factors: r `1.0`, vr `1.5`, vt `3.0`.
- Dominant blocking threshold at best setting: `tangential_velocity`.
- Potential CAPTURE case gain from threshold-only reclassification: `0`.

## Research Answers

1. Are any prior crossings near recoverable? `yes`.
2. Which variable is most often failing? `tangential_velocity`.
3. Which threshold blocks recoverability most? `tangential_velocity` at the best relaxed setting.
4. Can CAPTURE improve via threshold architecture only? `no` under the tested 1.0-3.0 factor grid. Recoverability can improve: `yes`.
5. Current bottleneck: `recoverability basin width`.

## Success Criteria

- Minimum, identify dominant failure variable: `met`.
- Moderate, threshold relaxation produces recoverability: `met`.
- Strong, isolate exact bottleneck threshold: `met`.
- Major, prove CAPTURE architecture is bottleneck: `not met`.

## Honesty Note

- Threshold ablation reclassifies existing states only; it does not rerun trajectories.
- CAPTURE gain is reported as potential acceptance gain, not as a changed simulation result.
- If no relaxed setting works, this script reports crossing geometry as invalid for the tested basin.

## Artifacts

- `phase25_crossing_dataset.csv`
- `phase25_threshold_ablation.csv`
- `crossing_r_vs_vr.png`
- `crossing_vr_vs_vt.png`
- `recoverability_distance_histogram.png`
- `threshold_ablation_heatmap.png`
- `failure_mode_clusters.png`
- `basin_boundary_analysis.png`