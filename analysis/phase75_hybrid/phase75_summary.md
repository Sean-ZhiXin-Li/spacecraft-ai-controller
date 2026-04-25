# Phase 7.5 Hybrid Controller Summary

## Setup

- Scope: 2D Python-only hybrid switch between `prewindow_radial_medium` and `adaptive_soft`.
- Grid: same 270 regimes from Phase 6.5-7.
- Hybrid rule: if `abs(r_error_ratio) > switch_threshold`, use Phase 7 `prewindow_radial_medium`; otherwise use Phase 6.7 `adaptive_soft` WS behavior.
- No action blending, no new heuristics, no learning, and no physics changes.

## Ranking

- `hybrid_1e4` threshold `1.0e-04`: success `170`, CAPTURE `170`, near-miss `42`.
- `hybrid_3e4` threshold `3.0e-04`: success `157`, CAPTURE `157`, near-miss `31`.
- `hybrid_2e4` threshold `2.0e-04`: success `157`, CAPTURE `157`, near-miss `36`.

## Reference Comparison

- `adaptive_soft`: success `172`, CAPTURE `172`, near-miss `44`.
- `prewindow_radial_medium`: success `209`, CAPTURE `209`, near-miss `56`.
- Best hybrid: `hybrid_1e4` with success `170`, CAPTURE `170`, near-miss `42`.

## Answers

1. Does hybrid maintain the high success of prewindow_radial_medium? `no`. Delta vs prewindow_radial_medium is success `-39`.
2. Does it reduce near-miss? `yes`. Delta vs prewindow_radial_medium is near-miss `-14`; delta vs adaptive_soft is `-2`.
3. Which switch_threshold works best? `hybrid_1e4` with threshold `1.0e-04`.
4. Does hybrid truly combine strengths, or still trade off? `the hybrid improves near-miss stability but gives up reachability`. Success overlap vs prewindow is retained `164`, gained `6`, lost `45`; overlap vs adaptive is retained `157`, gained `13`, lost `15`.
5. Final conclusion of the whole project: within the 2D Python evaluator, explicit controllers can beat the learned baseline on this local orbit-lock grid; Phase 7's pre-window radial shaping is the main reachability gain, while Phase 7.5 tests whether a threshold switch can recover stability without changing physics or learning. The final controller choice should prioritize the ranked grid result and near-miss tolerance rather than adding new algorithmic scope.