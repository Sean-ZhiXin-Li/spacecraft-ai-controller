# Project Logs Index

This index points readers to the current 2D orbital-insertion research trail while keeping older controller milestones visible as historical evidence.

## Current Research Milestone Trail

1. [Sprint PPO22-27 - Local control cannot solve global reachability](../project_log/sprint_ppo22-27.md)
2. [Sprint PPO28-33 - From trajectory-family mapping to optimal structure extraction](../project_log/sprint_ppo28-33.md)
3. [PL34 - Crossing is transition, not success](../project_log/pl34_post_cross_sync.md)
4. [PL35 - Crossing basin expansion failed under local bias architectures](../project_log/pl35_crossing_basin_expansion.md)
5. [PL36 - Transfer-family benchmark and non-crossing diagnosis](../project_log/pl36_transfer_family_benchmark_and_diagnosis.md)

## Current Research State

- Phase34 is the fixed terminal/post-cross controller result.
- Phase36B tested transfer-family variants on the full 24-case reduced benchmark and did not expand crossings beyond the Phase34 baseline.
- Phase36C diagnosed the remaining `16 / 24` non-crossing cases and prepared the next planner-level search space.
- The next step is a small parameterized planner-level transfer search for upstream crossing-generation.

Primary current references:

- [Research direction](research_direction.md)
- [Phase36B summary](../analysis/phase36b_transfer_family_benchmark/summary.md)
- [Phase36C summary](../analysis/phase36c_non_crossing_geometry_diagnosis/summary.md)
- [Phase36C planner search space](../analysis/phase36c_non_crossing_geometry_diagnosis/planner_search_space.md)

## Earlier Local-Controller Milestone

Phase7.6 remains the strongest local 2D explicit-controller milestone:

- Controller: `soft_linear_3e4`
- Simulator success labels: `217 / 270`
- CAPTURE entries: `217 / 270`
- Near-misses: `8`

Use these files for the local-controller evidence trail:

1. [PL22 - Phase 6.5 Window-Seeking](../project_log/pl22_phase65_window_seeking.md)
2. [PL23 - Phase 6.6 WS-1 Refinement](../project_log/pl23_phase66_ws1_refine.md)
3. [PL24 - Phase 6.7 Adaptive Window-Seeking](../project_log/pl24_phase67_adaptive_ws.md)
4. [PL25 - Phase 7 Pre-Window Shaping](../project_log/pl25_phase7_prewindow_shaping.md)
5. [PL26 - Phase 7.5 Hard Hybrid](../project_log/pl26_phase75_hard_hybrid.md)
6. [PL27 - Phase 7.6 Soft Hybrid](../project_log/pl27_phase76_soft_hybrid.md)
7. [Phase7.6 summary](../analysis/phase76_soft_hybrid/phase76_summary.md)

## Historical Logs

The remaining files in `project_log/` document earlier baseline, PPO, imitation-learning, residual-control, debugging, and exploratory stages. Keep them as historical context, but use the current research milestone trail above as the primary reading path.
