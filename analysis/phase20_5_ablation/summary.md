# Phase 20.5 Predictive Planner Failure Diagnosis

## Structural Conclusion

- Dominant bottleneck: `Structural orbital-class bottleneck`.
- Reason: No tested horizon, score, replan, or expanded local-action variant improved the composite reachability metrics over the Phase 20 baseline.
- Phase 20 baseline: crossing `12`, recoverable `0`, CAPTURE `12`, success `12`, overspeed `0`.

## Research Answers

1. Does longer horizon create new reachability? `no`. Best horizon variant is `baseline_h260_r40_recoverability_base`.
2. Does planner mostly choose coast? `yes`. Baseline coast choice rate is `92.0%`.
3. Does recoverability scoring suppress useful aggressive transfers? `no`. Crossing-only score composite change is `0`.
4. Is predictive local search fundamentally incapable of orbital-class transition? `yes`.

## Horizon Sweep

| Variant | Crossing | Recoverable | CAPTURE | Success | Overspeed | Mean action norm | Coast % | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_h260_r40_recoverability_base` | 12 | 0 | 12 | 12 | 0 | 0.0056 | 92.0 | 360.3 |
| `horizon_500` | 12 | 0 | 12 | 12 | 0 | 0.0021 | 97.0 | 647.6 |
| `horizon_1000` | 12 | 0 | 12 | 12 | 0 | 0.0001 | 99.9 | 1186.3 |
| `horizon_2000` | 12 | 0 | 12 | 12 | 0 | 0.0001 | 99.9 | 2337.2 |

## Replan Interval Sweep

| Variant | Crossing | Recoverable | CAPTURE | Success | Overspeed | Mean action norm | Coast % | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_h260_r40_recoverability_base` | 12 | 0 | 12 | 12 | 0 | 0.0056 | 92.0 | 360.3 |
| `replan_20` | 12 | 0 | 12 | 12 | 0 | 0.0001 | 99.9 | 801.5 |
| `replan_80` | 12 | 0 | 12 | 12 | 0 | 0.0020 | 97.2 | 182.4 |
| `replan_150` | 12 | 0 | 12 | 12 | 0 | 0.0000 | 100.0 | 109.5 |

## Score Ablation

| Variant | Crossing | Recoverable | CAPTURE | Success | Overspeed | Mean action norm | Coast % | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_h260_r40_recoverability_base` | 12 | 0 | 12 | 12 | 0 | 0.0056 | 92.0 | 360.3 |
| `score_crossing_only` | 12 | 0 | 12 | 12 | 0 | 0.0056 | 92.0 | 365.4 |

## Action Space Ablation

| Variant | Crossing | Recoverable | CAPTURE | Success | Overspeed | Mean action norm | Coast % | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_h260_r40_recoverability_base` | 12 | 0 | 12 | 12 | 0 | 0.0056 | 92.0 | 360.3 |
| `action_expanded` | 12 | 0 | 12 | 12 | 0 | 0.0056 | 92.0 | 478.5 |
| `action_expanded_burn_hold` | 12 | 0 | 12 | 12 | 0 | 0.0007 | 99.7 | 1028.7 |

## Action Distribution

| Variant | Coast % | Prograde % | Retrograde % | Radial inward % | Radial outward % | Diagonal % | Mean margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_h260_r40_recoverability_base` | 92.0 | 0.0 | 7.9 | 0.0 | 0.0 | 0.0 | 0.000004 |
| `horizon_500` | 97.0 | 0.0 | 2.9 | 0.0 | 0.0 | 0.0 | 0.000008 |
| `horizon_1000` | 99.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.000015 |
| `horizon_2000` | 99.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.000031 |
| `replan_20` | 99.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.000001 |
| `replan_80` | 97.2 | 0.0 | 2.8 | 0.0 | 0.0 | 0.0 | 0.000000 |
| `replan_150` | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.499998 |
| `score_crossing_only` | 92.0 | 0.0 | 7.9 | 0.0 | 0.0 | 0.0 | 0.000002 |
| `action_expanded` | 92.0 | 0.0 | 7.9 | 0.0 | 0.0 | 0.0 | 0.000003 |
| `action_expanded_burn_hold` | 99.7 | 0.0 | 0.0 | 0.1 | 0.1 | 0.1 | 0.000000 |

## Artifacts

- `ablation_results.csv`
- `horizon_sweep.png`
- `replan_interval_sweep.png`
- `action_distribution.png`
- `score_ablation_comparison.png`