# Phase 19 Minimal Transfer Planning

## Scope

- 2D Python-only explicit planned-transfer test.
- Physics, CAPTURE/LOCK logic, and success definition are unchanged.
- DESCENT uses a short injection burn, then forced coast, then Phase 18-style targeting only when a crossing is being approached inside the targeting band.
- The injection burn is small and purely tangential: norm `0.10`, with duration capped between 20 and 80 steps.
- After injection, the planner forces at least `20` no-thrust coast steps before targeting can engage.

## Reduced-Grid Result

| Controller | CAPTURE | Success | Near-miss | Total |
|---|---:|---:|---:|---:|
| `baseline_soft_linear_3e4` | 12 | 12 | 0 | 48 |
| `phase18_crossing_targeting` | 12 | 12 | 0 | 48 |
| `phase19_minimal_transfer` | 0 | 0 | 12 | 48 |

- Capture improvement vs baseline: `-12`.
- Success improvement vs baseline: `-12`.
- Crossing-case improvement vs baseline: `-12`.
- Capture improvement vs Phase 18 reactive targeting: `-12`.
- Success improvement vs Phase 18 reactive targeting: `-12`.
- Crossing-case improvement vs Phase 18 reactive targeting: `-12`.
- Total radius crossings: baseline `15`, Phase 18 `15`, Phase 19 `0`.
- Overspeed terminations: baseline `0`, Phase 18 `0`, Phase 19 `12`.
- Phase 19 injection steps: `1837`.
- Phase 19 coast steps: `3598512`.
- Phase 19 targeting steps: `0`.
- Phase 19 mode usage: `{"coast": 3598512, "injection": 1837}`.

## Answers

1. Does planned injection create new crossings? `no`. Crossing cases change from Phase 18 `12` to Phase 19 `0`.
2. Does capture increase? `no`. CAPTURE changes by `-12` versus Phase 18.
3. Is coast phase critical? `inconclusive`. Coast dominates the mode counts (`3598512` coast steps vs `1837` injection and `0` targeting), but this run produced `0` Phase 19 crossing cases and does not include a separate no-coast ablation.
4. Does planning outperform reactive control? `no` on this reduced grid. The minimal injection/coast plan only outperforms Phase 18 if CAPTURE or success increases.

## Interpretation

This phase tests whether a very small impulse-like setup plus coast can create useful target-radius crossings before local targeting. If crossings do not increase, the missing piece is still transfer design rather than local crossing correction.

## Artifacts

- `comparison.csv`
- `phase19_summary.md`
