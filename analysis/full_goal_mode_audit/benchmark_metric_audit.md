# Benchmark and Metric Audit

## Overall Assessment

The benchmark structure is useful but not yet mature. The 24-case reduced benchmark gives continuity between Phase34 and Phase35, but the project needs a stable benchmark contract before adding more transfer families.

## Strongest Metrics

The strongest metrics are:

- geometric crossing count
- recoverable crossing count
- crossing-case best distance to recoverability
- crossing sync error
- crossing radial velocity ratio
- crossing tangential velocity error ratio
- overspeed count
- instability count

These metrics reveal trajectory structure rather than only final labels.

## Weakest Metrics

The weakest metric is the generic `success` label when shown without context. It is simulator-defined and can be misleading if read as mission success.

Other weaker metrics:

- all-case mean best distance when dominated by non-crossing families
- crossing potential when used as an outcome rather than a diagnostic
- representative-subset counts when presented beside full benchmark counts without warning

## Phase34 and Phase35 Comparability

Phase34 and Phase35 are comparable because:

- both use the same 24-case reduced benchmark
- Phase35 preserves Phase34 terminal behavior after crossing
- physics and thresholds remain unchanged

This makes the Phase35 negative result meaningful.

## Phase36A Subset Caveat

Phase36A is not directly comparable to Phase35 as a full benchmark. It uses three representative cases:

- one crossing case
- one near-crossing non-crossing case
- one over-conservative-transfer case

This is appropriate for visualization, but not for general performance claims.

## Metric Inflation Risks

Potential inflation risks:

- treating CAPTURE/LOCK as real flight validation
- treating simulator success as mission success
- treating visual demos as benchmark evidence
- treating a later recoverable state as meaning the first crossing was recoverable
- treating crossing potential as equivalent to actual future crossing

The current public docs mostly avoid these errors.

## Representative Case Selection

Representative cases are useful if they are explicitly labeled as representative. They should not be described as "the benchmark".

For Phase36B, every family should run on the full 24-case set, with Phase36A plots used only to interpret why a family behaved as it did.

## Recommendation

Before adding MPC-lite or stronger planners, define a stable benchmark table schema:

- crossing count
- recoverable crossing count after Phase34 handoff
- crossing-state quality metrics
- min radius error
- overspeed
- instability
- dominant failure label
- family qualitative label

This will prevent the project from drifting into incomparable phase-specific scorecards.

