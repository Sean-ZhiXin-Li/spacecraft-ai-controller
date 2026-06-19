# Phase38C Registered Hypotheses

Scope: hypotheses registered from existing Phase34, Phase36B, Phase36C, Phase37A, and Phase37B evidence. These are not controller implementations.

## H1 - Failure Class Is Strongly Conditioned By Initial Radius Regime

- Claim: The reduced benchmark classes separate descriptively by `r0_over_target`.
- Supporting evidence: Phase36C baseline labels place `near_crossing` cases on the `0.98` side and `over_conservative_transfer` cases on the `1.02` side; Phase36B baseline crossing-producing cases are associated with the `1.00` regime.
- Contradicting evidence: `r0_over_target` is an initial condition, not a controller variable. It may not generalize beyond the reduced grid.
- Prediction: Future analysis should show different failure signatures when rows are grouped by radius regime.
- Rejection condition: If class separation disappears under a broader benchmark or regenerated full-benchmark evidence.
- Controller implication: Use as analysis conditioning only, not as a standalone control law.
- Confidence: Medium-high for descriptive analysis; low for implementation.

## H2 - Closest-Approach Timing Separates Failure Modes

- Claim: `closest_approach_step` distinguishes near-crossing from over-conservative-transfer failures.
- Supporting evidence: Phase38A mining found near-crossing rows include late/max-step closest approaches, while over-conservative-transfer baseline rows cluster at early closest approach.
- Contradicting evidence: Phase37B improved closest approach in selected cases but produced zero selected-case crossings.
- Prediction: Failure review grouped by closest-approach timing should separate late-approach and early-approach non-crossing modes.
- Rejection condition: If timing differences fail to persist when all non-crossing cases are evaluated under a fixed baseline.
- Controller implication: Analysis variable only until a causal link to actual crossings is shown.
- Confidence: Medium.

## H3 - Crossing Potential Is Diagnostic But Not Sufficient

- Claim: `best_crossing_potential` separates classes descriptively but is not a success criterion.
- Supporting evidence: Phase38A mining found higher values in crossing-producing rows than non-crossing rows; Phase36C recorded family rows where crossing potential moved.
- Contradicting evidence: Phase36C family variants improved crossing potential without producing new target-radius crossings.
- Prediction: Potential may rank cases by proximity but will not reliably predict actual crossing without additional conditions.
- Rejection condition: If future evidence shows no association between potential and crossing proximity.
- Controller implication: Use only as a diagnostic ranking feature.
- Confidence: Medium-low.

## H4 - Closest-Approach Error Is Diagnostic But Not Sufficient

- Claim: `min_abs_radius_error_ratio` is useful for diagnosing proximity but cannot justify implementation by itself.
- Supporting evidence: Phase36C and Phase37B record closest-approach movement.
- Contradicting evidence: Phase37B improved closest approach in `3 / 4` selected cases but produced zero selected-case crossings and damaged regression preservation.
- Prediction: Smaller radius error may identify interesting failures but will not necessarily produce crossing.
- Rejection condition: If future registered runs show closest-approach improvement with no repeated relation to crossing generation.
- Controller implication: Do not approve a controller on closest-approach improvement alone.
- Confidence: Medium.

## H5 - Radial Commitment Timing Alone Does Not Expand Crossing

- Claim: Radial timing alone is insufficient under the tested Phase37A design.
- Supporting evidence: Phase37A ran six variants over `144` rollouts and produced zero new crossings on baseline non-crossing cases.
- Contradicting evidence: Delayed variants preserved the existing `8 / 24` crossings and recoverable crossings.
- Prediction: Repeating radial timing without a new explanatory control will not create robust new crossings.
- Rejection condition: If a registered, regression-protected radial timing variant creates new crossings and preserves all regression cases.
- Controller implication: Do not expand radial timing as the next implementation variable.
- Confidence: Medium-high.

## H6 - Radial Magnitude Alone Is Unsafe Or Insufficient

- Claim: Radial magnitude alone should not be tested again as a standalone lever.
- Supporting evidence: Phase37A medium magnitude collapsed crossing counts; low magnitude created zero new crossings.
- Contradicting evidence: Low magnitude was not overspeed/instability unsafe.
- Prediction: Larger or repeated radial magnitude sweeps will either preserve baseline without new crossings or degrade the crossing set.
- Rejection condition: If a pre-registered magnitude study creates new crossings while preserving all regression cases.
- Controller implication: No standalone radial magnitude implementation.
- Confidence: Medium.

## H7 - Weak Tangential Shaping In The Phase37B Form Is Not Viable

- Claim: The tested weak tangential shaping form should not be expanded.
- Supporting evidence: Phase37B produced zero selected-case crossings and preserved only `4 / 8` regression crossings.
- Contradicting evidence: Closest approach improved slightly in `3 / 4` selected cases.
- Prediction: Expanding the same weak tangential form will risk regression damage before creating robust new crossings.
- Rejection condition: If a revised tangential hypothesis isolates the effect and preserves all regression cases while creating actual crossings.
- Controller implication: Reject Phase37B-style weak tangential shaping as the next implementation variable.
- Confidence: Medium-high.

## H8 - Coast Duration Is Not Yet Evidence-Backed

- Claim: Coast duration is not currently approved because it is not isolated in the recorded CSV evidence.
- Supporting evidence: Phase36C and logs suggest timing may matter, but no CSV isolates coast duration.
- Contradicting evidence: Phase37A already gives negative evidence for radial timing, raising the risk that coast duration repeats the same failure.
- Prediction: Without additional evidence, coast duration will be hard to distinguish from commit-window timing.
- Rejection condition: If existing or future registered analysis shows coast duration is separable from radial commitment timing and predicts actual crossing.
- Controller implication: Do not implement as Phase39 yet.
- Confidence: Unknown.

## H9 - Angular Momentum Correction Is Plausible But Unsupported

- Claim: Angular momentum correction is physically plausible but not evidence-backed by the inspected CSVs.
- Supporting evidence: Tangential state matters after crossing, and orbital control reasoning makes angular momentum relevant.
- Contradicting evidence: No inspected CSV isolates angular momentum correction; Phase37B tangential shaping has negative evidence.
- Prediction: Without explicit logged evidence, angular momentum correction will be confounded with tangential shaping.
- Rejection condition: If existing records cannot isolate it and no registered diagnostic can separate it from tangential correction.
- Controller implication: Defer.
- Confidence: Unknown.
