# Next 30-Day Plan

## Goal

Move from phase-by-phase exploration to a stable research platform for transfer-family and recoverability analysis.

## Week 1 - Phase36B Benchmark

- Run full 24-case transfer-family benchmark.
- Keep Phase34 terminal controller fixed.
- Compare baseline, spiral, grazing, and redesigned delayed crossing.
- Report both positive and negative results.

## Week 2 - Benchmark Infrastructure

- Extract shared 2D rollout and metric utilities.
- Centralize recoverability thresholds and success-label terminology.
- Make generated markdown use precise simulator-label language.
- Add one benchmark manifest for the 24-case set.

## Week 3 - Trajectory Structure Analysis

- Build trajectory-family dataset with time-series fields.
- Cluster trajectories by behavior.
- Create state-space maps for radius error, vr ratio, vt error, energy proxy, and angular momentum proxy.
- Identify families that approach but do not cross versus families that cross with bad sync.

## Week 4 - Decision Point

Choose the next direction based on Phase36B:

- If no family improves crossing count: design a planner-level trajectory search.
- If crossing improves but recoverability does not: focus on handoff quality constraints.
- If both crossing and recoverability improve: formalize the family as the Phase34 upstream module.

## What To Delay

Delay:

- MPC-lite until family geometry is clearer
- direct optimization until family-level structure is understood
- 3D physics
- SPICE
- C++ rewrite
- larger PPO or RL scaling

## 30-Day Deliverable

The deliverable should be a clean research package:

- Phase36B full benchmark CSV
- family comparison summary
- trajectory-family dataset
- failure mode analysis
- benchmark manifest
- recommendation on whether MPC-lite is justified

The objective is not more phases. The objective is a cleaner scientific basis for the next phase.

