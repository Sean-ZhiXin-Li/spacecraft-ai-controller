# Next 7-Day Plan

## Goal

Turn Phase36A from a visualization probe into a disciplined Phase36B benchmark plan without adding unnecessary complexity.

## Day 1 - Freeze Benchmark Definitions

- Define the exact 24-case Phase36B benchmark as the same reduced grid used in Phase34 and Phase35.
- Write down the required CSV columns before coding.
- Keep Phase34 `radius_priority` as the fixed terminal controller.

## Day 2 - Select Families

Carry forward:

- `baseline_phase34`
- `spiral_approach`
- `grazing_corridor`
- redesigned `delayed_crossing`

Do not carry forward unless redesigned:

- `energy_bleed_then_cross`
- `overshoot_return`
- `two_stage_transfer`

## Day 3 - Define Primary Metrics

Primary:

- crossing count
- Phase34-compatible crossing count
- recoverable crossing count
- overspeed
- instability

Crossing-state quality:

- crossing vr ratio
- crossing vt error ratio
- crossing sync error
- best post-cross distance
- min radius error

## Day 4 - Build or Plan Shared Metric Core

Before implementing more family variants, identify duplicated code in Phase34, Phase35, and Phase36A:

- dynamics step
- state diagnostics
- termination logic
- recoverability logic
- CSV aggregation

The goal is to prepare modularization, not to refactor everything immediately.

## Day 5 - Run Full Phase36B

Run the selected families on the full 24-case benchmark.

Do not tune families after seeing results unless a clear runtime bug exists.

## Day 6 - Analyze Failure Modes

Classify non-crossing rows:

- near crossing
- over-conservative
- overspeed
- bad tangential corridor
- bad radial timing
- dead geometry

Compare against Phase35 failure labels.

## Day 7 - Write Phase36B Summary

The summary should answer:

- Did any family increase crossing count?
- Did any family preserve Phase34 recoverability?
- Which family generated the best crossing-state quality?
- Which failures are global geometry failures?
- Should MPC-lite be tried next?

## Success Criterion

The week succeeds even if no family improves crossing count, as long as the result narrows the transfer-family hypothesis space.

