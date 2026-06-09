# Public Scientific Artifact Manifest

This manifest identifies the repository artifacts that currently support public scientific claims. These files are evidence records, not scratch outputs. Do not overwrite historical outputs casually; if a benchmark is rerun, write the new run to a new phase directory or document the regeneration explicitly.

## Phase7.6 - Soft Hybrid Local Controller

Status: historical local-controller milestone.

Artifacts:

- `analysis/phase76_soft_hybrid/phase76_summary.md`
- `analysis/phase76_soft_hybrid/soft_hybrid_grid.csv`
- `analysis/phase76_soft_hybrid/soft_hybrid_ranking.csv`

Supports:

- `soft_linear_3e4` reached `217 / 270` simulator success labels on its local 2D grid.
- The same variant reached `217 / 270` CAPTURE entries and `217 / 270` LOCK entries, with `8` near-misses.

## Phase8 - Multi-Regime Generalization Map

Status: historical generalization diagnostic.

Artifacts:

- `analysis/phase8_multiregime/phase8_summary.md`
- `analysis/phase8_multiregime/phase8_grid.csv`

Supports:

- Phase7.6 did not solve broad global reachability.
- The expanded Phase8 grid produced `220 / 1296` simulator success labels and `265 / 1296` CAPTURE/crossing entries.

## Phase34 - Post-Cross Synchronization

Status: current terminal-controller foundation.

Artifacts:

- `analysis/phase34_post_cross_sync/summary.md`
- `analysis/phase34_post_cross_sync/phase34_results.csv`
- `analysis/phase34_post_cross_sync/phase34_vs_phase31_comparison.md`

Supports:

- The reduced Phase31-style reference produced `8 / 24` crossings and `0 / 24` recoverable crossings.
- Phase34 `radius_priority` preserved `8 / 24` crossings and converted those crossing-producing cases into `8 / 24` recoverable crossings.
- Crossing-case best distance improved from `3.9923` to `0.9855`.

## Phase36B - Transfer-Family Benchmark

Status: current upstream transfer-family negative result.

Artifacts:

- `analysis/phase36b_transfer_family_benchmark/summary.md`
- `analysis/phase36b_transfer_family_benchmark/phase36b_results.csv`
- `analysis/phase36b_transfer_family_benchmark/phase36b_family_summary.csv`

Supports:

- `baseline_phase34`, `spiral_approach`, `grazing_corridor`, and `redesigned_delayed_crossing` each produced `8 / 24` crossings.
- Each family produced `8 / 24` recoverable crossings, `0` overspeed cases, and `0` instability cases.
- No tested transfer family expanded the crossing basin beyond the Phase34 baseline.

## Phase36C - Non-Crossing Geometry Diagnosis

Status: current non-crossing diagnosis.

Artifacts:

- `analysis/phase36c_non_crossing_geometry_diagnosis/summary.md`
- `analysis/phase36c_non_crossing_geometry_diagnosis/non_crossing_case_set.csv`
- `analysis/phase36c_non_crossing_geometry_diagnosis/planner_search_space.md`

Supports:

- The Phase36B baseline had `16 / 24` non-crossing cases.
- Baseline non-crossing labels split into `8` `near_crossing` cases and `8` `over_conservative_transfer` cases.
- Phase36B family variants moved closest-approach and crossing-potential metrics without producing new target-radius crossings.

## Phase37A - Radial Commitment Timing Sweep

Status: recent negative result.

Artifacts:

- `analysis/phase37a_radial_commit_timing/phase37a_results.csv`
- `analysis/phase37a_radial_commit_timing/phase37a_summary.md`
- `analysis/phase37a_radial_commit_timing/phase37a_comparison.png`
- `project_log/phase37a_radial_commit_timing.md`

Supports:

- Phase37A ran `6` variants across the same `24` cases, for `144` rollouts.
- It created `0` new crossings on the `16` Phase36B baseline non-crossing cases.
- `delayed_commit_low` and `delayed_commit_medium` each preserved `8 / 24` crossings and `8 / 24` recoverable crossings.
- Early and mid commitment degraded the existing crossing set.
- No Phase37A variant produced overspeed or instability.

## Phase37B - Weak Tangential Subset Diagnostic

Status: current frontier negative diagnostic.

Artifacts:

- `analysis/phase37b_weak_tangential_subset/phase37b_results.csv`
- `analysis/phase37b_weak_tangential_subset/phase37b_summary.md`
- `analysis/phase37b_weak_tangential_subset/phase37b_comparison.png`
- `project_log/phase37b_weak_tangential_postmortem.md`

Supports:

- Phase37B ran `24` subset rollouts: 4 selected Phase37A-improved non-crossing cases and 8 Phase36B regression crossing cases across 2 settings.
- Weak tangential shaping created `0 / 4` selected-case crossings and `0 / 4` selected-case recoverable crossings.
- Weak tangential shaping preserved only `4 / 8` regression crossings and `4 / 8` regression recoverable crossings.
- Phase37B produced `0` overspeed cases and `0` instability cases.
- Closest approach improved in `3 / 4` selected cases, but this was not sufficient because no new crossings were created and regression preservation failed.

## Phase38 - Evidence-Based Search-Space Planning

Status: current planning document; not an implementation phase.

Artifacts:

- `docs/phase38_evidence_based_search_space.md`

Supports:

- Phase38 should analyze crossing-basin expansion failures before new controller code.
- The next experiment must be justified by Phase36B, Phase36C, Phase37A, and Phase37B evidence.
- The existing `8 / 24` crossing-producing cases should remain regression guards.

## Preservation Rule

These artifacts are the current public evidence trail. Keep failed or negative results visible when they explain a research decision. Prefer adding new phase directories over rewriting historical CSVs, plots, or summaries.
