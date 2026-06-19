# Phase38A Failure Signature

Scope: evidence-mining summary from existing Phase34, Phase36B, Phase36C, Phase37A, and Phase37B CSVs. No new controller experiment was run.

## What Makes Preserved Crossing Cases Different?

- Preserved crossing cases are the only class with actual `first_crossing_step` / `crossing_step` values.
- They are the only class with crossing-state fields such as `crossing_vr_ratio`, `crossing_vt_error_ratio`, and `crossing_sync`.
- In the reduced Phase36B baseline, preserved crossing-producing cases are associated with the middle radius-ratio regime, `r0_over_target=1.00`.
- When crossing exists and the Phase34 `radius_priority` post-cross controller is fixed, recoverable-crossing preservation is strong in Phase34, Phase36B, and the delayed Phase37A variants.
- Overspeed and instability do not explain the difference in the inspected evidence because they are consistently absent in the relevant Phase34/36/37 aggregate results.

Important limitation: crossing-state metrics are not available for non-crossing rows, so they cannot explain why near-crossing or over-conservative-transfer rows fail before crossing.

## What Makes `near_crossing` Different?

- `near_crossing` is a recorded non-crossing failure label, not a successful event.
- In Phase36C baseline labels, near-crossing rows are associated with the lower radius-ratio side, `r0_over_target=0.98`.
- `near_crossing` rows include many late or max-step closest approaches, unlike the over-conservative baseline rows that cluster at early closest approach.
- `best_crossing_potential` is generally higher for near-crossing rows than for over-conservative-transfer rows in the mined data.
- Phase37A still created zero new crossings on baseline non-crossing cases, so the near-crossing label alone does not identify a sufficient radial-timing implementation variable.

## What Makes `over_conservative_transfer` Different?

- In Phase36C baseline labels, over-conservative-transfer rows are associated with the higher radius-ratio side, `r0_over_target=1.02`.
- Baseline over-conservative-transfer rows show early closest approach in the mined data.
- Their `best_crossing_potential` is lower than crossing-producing and near-crossing rows in the aggregated evidence.
- Phase37B selected cases came from this class, but weak tangential shaping created zero selected-case crossings and preserved only `4 / 8` regression crossings.

## Signatures That Appear Repeatedly

- Actual target-radius crossing remains the decisive transition. Without crossing, post-cross synchronization and crossing-state quality metrics cannot operate.
- `r0_over_target` is the strongest descriptive separator in the reduced benchmark classes.
- `closest_approach_step` appears to separate failure timing signatures between near-crossing and over-conservative-transfer cases.
- `best_crossing_potential` and `min_abs_radius_error_ratio` are useful diagnostics but not primary success metrics.
- Safety flags (`overspeed`, `instability`) are not class separators in the inspected Phase34/36/37 evidence.

## Signatures That Remain Inconclusive

- Coast duration is not directly isolated in the recorded CSV evidence.
- Angular-momentum correction is not directly isolated in the recorded CSV evidence.
- Radial timing has direct negative evidence from Phase37A: zero new crossings on baseline non-crossing cases.
- Radial magnitude has direct negative evidence from Phase37A: medium magnitude degraded crossings and low magnitude did not create new crossings.
- Weak tangential shaping has direct negative evidence from Phase37B: zero selected-case crossings and failed regression preservation.

## Bottom Line

The evidence supports failure-class mining, not immediate controller implementation. The strongest current signatures are descriptive: radius-ratio regime, closest-approach timing, crossing potential, and actual crossing-state availability. The currently tested control variables do not yet justify a Phase39 implementation.
