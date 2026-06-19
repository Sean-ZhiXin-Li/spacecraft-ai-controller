# Phase38B Variable Ranking

Scope: ranks only variables supported or contradicted by existing Phase34, Phase36B, Phase36C, Phase37A, and Phase37B CSV evidence. This is not a controller design.

| Rank | Candidate variable | Confidence | Implementation priority |
|---:|---|---|---|
| 1 | Initial radius regime / `r0_over_target` conditioning | Medium-high as an analysis variable; low as a direct controller lever | P0 for analysis; not a standalone implementation variable |
| 2 | Closest-approach timing / `closest_approach_step` | Medium | P1 analysis-only before any implementation |
| 3 | Crossing potential / `best_crossing_potential` | Medium-low | P1 diagnostic only |
| 4 | Minimum radius error / `min_abs_radius_error_ratio` | Medium-low | P1 diagnostic only |
| 5 | Radial commitment timing | Low for implementation | Do not expand without new evidence |
| 6 | Radial magnitude | Low | Do not test standalone again |
| 7 | Weak tangential shaping | Low-negative | Reject as next implementation variable |
| 8 | Coast duration | Unknown | Do not implement before additional evidence mining |
| 9 | Angular momentum correction | Unknown | Defer |

## Ranked Assessments

### 1. Initial radius regime / `r0_over_target` conditioning

- Supporting evidence: strong descriptive separation in Phase36C baseline labels. Near-crossing appears on the lower radius-ratio side, over-conservative-transfer on the higher radius-ratio side, and preserved Phase36B baseline crossing-producing cases are associated with the middle radius-ratio regime.
- Contradicting evidence: this is an initial-condition descriptor, not a controller variable. Conditioning on it can overfit the reduced grid and does not create crossings by itself.
- Confidence: medium-high as an analysis variable; low as a direct controller lever.
- Implementation priority: P0 for analysis; not a standalone implementation variable.

### 2. Closest-approach timing / `closest_approach_step`

- Supporting evidence: Phase36C baseline labels show different timing signatures between near-crossing and over-conservative-transfer rows.
- Contradicting evidence: Phase37B improved closest approach in selected cases but produced zero selected-case crossings.
- Confidence: medium.
- Implementation priority: P1 analysis-only before implementation.

### 3. Crossing potential / `best_crossing_potential`

- Supporting evidence: crossing-producing rows have higher `best_crossing_potential` than non-crossing rows in the mined data, and Phase36C recorded many non-baseline rows with improved crossing potential.
- Contradicting evidence: Phase36C showed potential movement without new crossings.
- Confidence: medium-low.
- Implementation priority: diagnostic only.

### 4. Minimum radius error / `min_abs_radius_error_ratio`

- Supporting evidence: recorded across Phase36B/C/37A/37B and captures closest approach.
- Contradicting evidence: Phase37B showed tiny improvements without selected-case crossings.
- Confidence: medium-low.
- Implementation priority: diagnostic only.

### 5. Radial commitment timing

- Supporting evidence: Phase37A directly tested timing; delayed variants preserved the known `8 / 24` crossing and recoverability counts.
- Contradicting evidence: Phase37A created zero new crossings on baseline non-crossing cases, and early/mid variants degraded the crossing set.
- Confidence: low for implementation.
- Implementation priority: do not expand without new evidence.

### 6. Radial magnitude

- Supporting evidence: recorded in Phase37A and Phase37B; magnitude changes affected crossing preservation.
- Contradicting evidence: medium radial magnitude collapsed crossings in Phase37A; low magnitude created zero new crossings.
- Confidence: low.
- Implementation priority: do not test standalone again.

### 7. Weak tangential shaping

- Supporting evidence: Phase37B weak tangential setting moved closest approach slightly in three of four selected cases.
- Contradicting evidence: it produced zero selected-case crossings and preserved only four of eight regression crossings.
- Confidence: low-negative.
- Implementation priority: reject as next implementation variable.

### 8. Coast duration

- Supporting evidence: suggested in planning/log narrative as a possible timing variable after Phase36C.
- Contradicting evidence: not directly isolated in recorded CSV evidence; risks repeating radial timing under another name.
- Confidence: unknown.
- Implementation priority: do not implement before additional evidence mining.

### 9. Angular momentum correction

- Supporting evidence: physically plausible from orbital-control reasoning.
- Contradicting evidence: not directly isolated in inspected CSVs and may collapse into tangential shaping, which has weak negative evidence in Phase37B.
- Confidence: unknown.
- Implementation priority: defer.

## Bottom Line

No variable is currently strong enough to justify immediate Phase39 implementation. The most supported variables are analysis variables, not controller variables: initial-condition class, closest-approach timing, and crossing-potential diagnostics. Radial timing, radial magnitude, and weak tangential shaping have direct negative evidence as implementation levers.
