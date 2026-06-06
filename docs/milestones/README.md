# Milestones

This folder is reserved for curated milestone documentation. The detailed result directories remain under `analysis/` so existing scripts and README links keep working.

## Current Milestone

The current milestone is the Phase37A radial commitment timing result built on the Phase36 transfer-family benchmark and non-crossing diagnosis:

- Phase34 is the fixed terminal/post-cross controller.
- Phase36B tested transfer-family variants and did not expand the crossing basin beyond `8 / 24` crossing-producing cases.
- Phase36C diagnosed the remaining `16 / 24` non-crossing cases and prepared the next planner-level transfer search.
- Phase37A tested radial commitment timing and bounded radial magnitude over `144` rollouts.
- Phase37A created `0` new crossings on the baseline non-crossing cases; delayed commitment preserved `8 / 24` crossings and `8 / 24` recoverable crossings, while early and mid commitment degraded the existing crossing set.
- The next step is not blind radial-timing expansion. Inspect Phase37A closest-approach deltas before deciding whether limited tangential shaping is justified.

Current references:

- [Project logs index](../project_logs_index.md)
- [Research direction](../research_direction.md)
- [Phase36B summary](../../analysis/phase36b_transfer_family_benchmark/summary.md)
- [Phase36C summary](../../analysis/phase36c_non_crossing_geometry_diagnosis/summary.md)
- [Phase37A summary](../../analysis/phase37a_radial_commit_timing/phase37a_summary.md)
- [Phase36C planner search space](../../analysis/phase36c_non_crossing_geometry_diagnosis/planner_search_space.md)
- [PL36 project log](../../project_log/pl36_transfer_family_benchmark_and_diagnosis.md)
- [PL37A project log](../../project_log/phase37a_radial_commit_timing.md)
- [Artifact manifest](../../analysis/artifact_manifest.md)

## Earlier Local-Controller Milestone

Phase7.6 remains the strongest local 2D explicit-controller result and should be treated as earlier controller evidence, not the current project frontier:

- [Phase7.6 Soft Hybrid Summary](../../analysis/phase76_soft_hybrid/phase76_summary.md)
- [PL22-PL27 project log trail](../project_logs_index.md#earlier-local-controller-milestone)
