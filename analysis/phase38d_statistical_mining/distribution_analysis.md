# Phase38D Distribution Analysis

Scope: descriptive analysis of recorded CSV fields for three classes: `crossing_producing`, `near_crossing`, and `over_conservative_transfer`.

## Class Distribution

| Class | Rows |
|---|---:|
| `crossing_producing` | 96 |
| `near_crossing` | 200 |
| `over_conservative_transfer` | 16 |

The distribution is imbalanced. `over_conservative_transfer` has far fewer rows than `near_crossing`, so conclusions about that class should remain cautious.

## Radius-Regime Distribution

The strongest descriptive pattern is radius-regime conditioning:

- crossing-producing baseline cases are associated with the middle radius ratio;
- near-crossing baseline cases are associated with the lower radius-ratio side;
- over-conservative-transfer baseline cases are associated with the higher radius-ratio side.

This should be interpreted as a reduced-benchmark signature, not broad orbital generalization.

## Closest-Approach Timing

`closest_approach_step` separates failure timing signatures:

- over-conservative-transfer baseline rows cluster at early closest approach;
- near-crossing rows include late or max-step closest approach behavior.

This supports a timing-signature hypothesis, but it does not yet support a timing controller.

## Crossing Potential

`best_crossing_potential` has strong descriptive separation:

- crossing-producing rows have the highest mean;
- near-crossing rows are intermediate;
- over-conservative-transfer rows are lowest.

However, Phase36C already showed that crossing potential can improve without producing actual target-radius crossings.

## Post-Cross Distance

`best_post_cross_distance` strongly separates crossing-producing and non-crossing behavior. This supports the main project interpretation that post-cross recoverability and upstream crossing generation are separate problems.

## Safety Flags

Overspeed and instability do not separate the three classes in the inspected evidence. They should remain regression and safety guards, not discovery variables.
