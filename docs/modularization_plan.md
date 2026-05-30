# Modularization Plan

## Why Modularization Matters Scientifically

The project has reached a point where duplicated experiment code can affect scientific credibility. Phase34, Phase35, and Phase36A all repeat similar dynamics, state-machine handling, metric calculation, and CSV/report generation.

Modularization is not cosmetic. It matters because it reduces the risk that future phases silently change:

- physics
- thresholds
- termination behavior
- recoverability definitions
- crossing detection
- Phase34 terminal handoff behavior

The goal is to make future benchmark comparisons more trustworthy.

## Duplicated Areas

Repeated logic appears across Phase34, Phase35, and Phase36A:

- 2D dynamics step
- initial state construction
- burn/coast/handoff transfer skeleton
- Phase34 post-cross synchronization action
- CAPTURE and LOCK state-machine transitions
- simulator-defined success label checks
- overspeed and instability termination checks
- crossing detection
- recoverability metrics
- CSV aggregation and markdown tables
- plotting setup and output directory handling

This duplication increases the chance of drift between phases.

## Proposed Shared Modules

Create shared modules only after documenting expected behavior and validating old outputs.

Proposed modules:

- `2D dynamics step`: one function for gravitational and thrust-limited integration.
- `initial condition builder`: one function for `r0_over_target`, velocity angle, thrust scale, and target scale.
- `benchmark manifest loader`: one source of truth for the 24-case Phase34/35/36B benchmark.
- `termination logic`: one implementation for overspeed, out-of-range, too-close, radial-stall, and max-step behavior.
- `Phase34 terminal controller`: one reusable post-cross synchronization implementation for `radius_priority`.
- `recoverability metric computation`: one implementation for recoverable state, sync error, and recoverability distance.
- `crossing-state metric computation`: one implementation for crossing vr ratio, crossing vt error ratio, crossing sync error, crossing step, and min radius error.
- `CSV writer`: one stable field ordering and writer helper for benchmark outputs.

## Staged Refactor Plan

### Stage 1 - Documentation First

Write down the exact benchmark contract and expected metrics before moving code. This is now represented by `docs/benchmark_contract.md` and `docs/phase36b_plan.md`.

No behavior changes should happen in this stage.

### Stage 2 - Benchmark Manifest Second

Create a simple manifest for the 24-case reduced benchmark.

Validation:

- manifest expands to exactly 24 cases
- case ordering matches Phase34/35 expectations
- no physics or controller logic changes

### Stage 3 - Shared Metrics Third

Extract metrics before extracting rollout control:

- recoverability distance
- sync error
- recoverable state check
- crossing-state metrics
- aggregation helpers

Validation:

- recomputed Phase34 and Phase35 aggregate counts match existing CSVs
- no existing CSVs are overwritten

### Stage 4 - Shared Rollout Core Last

Only after metrics are stable, extract the rollout core:

- dynamics step
- initial condition builder
- termination logic
- Phase34 terminal controller

This is the highest-risk stage because small changes can alter benchmark results.

## Risks

Key risks:

- accidentally changing behavior while moving code
- silently changing thresholds
- changing old phase outputs
- altering Phase34 terminal behavior
- introducing a new case ordering that changes comparisons
- overwriting historical CSVs or markdown summaries

These risks are why modularization should proceed in stages.

## Validation Requirements

Before and after extraction:

- Phase34 `radius_priority` must reproduce `8 / 24` crossings and `8 / 24` recoverable crossings.
- Phase34 Phase31-style reference must remain `8 / 24` crossings and `0 / 24` recoverable crossings in the reduced comparison.
- Phase35 `baseline_phase34` must remain `8 / 24` crossings and `8 / 24` recoverable crossings.
- Phase35 `radial_energy_push` and `tangential_corridor_entry` must remain `0 / 24` crossings if rerun unchanged.
- Existing CSV outputs must not be modified during validation unless the user explicitly requests regeneration.

## Near-Term Recommendation

Do not start with a broad refactor. First implement Phase36B under the current structure if needed, then extract shared metrics and benchmark manifest once the benchmark contract is stable.

The safest order is:

1. document the benchmark contract
2. run or plan Phase36B against that contract
3. extract shared metrics
4. extract shared rollout core only after regression checks are defined
