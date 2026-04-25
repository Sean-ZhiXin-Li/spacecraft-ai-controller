# Phase 6.7 Adaptive Window-Seeking Summary

## Setup

- Scope: 2D Python-only adaptive reduced-retrograde schedules inside the WS-1 window-seeking controller.
- CAPTURE/LOCK logic, one-step lookahead scoring, candidate structure, physics, and 270-regime grid are unchanged from Phase 6.6.
- Tested variants: `original_ws1`, `fixed_best_0p80`, `adaptive_step`, `adaptive_soft`, and `adaptive_energy`.
- Note: `adaptive_step` uses the requested `2e-4` threshold. Because WS activation remains `8e-5`, its far branch is not expected to activate in this narrow test.

## Result

- Original WS-1: success `157`, CAPTURE `157`, near-miss `42`.
- Fixed Phase 6.6 best: success `172`, CAPTURE `172`, near-miss `47`, gained/lost vs original `29/14`.
- Best adaptive: `adaptive_soft` with success `172`, CAPTURE `172`, near-miss `44`, gained/lost vs original `29/14`.
- Best overall by ranking: `adaptive_soft`.

## Answers

1. Adaptive reduced-retrograde does not improve over the fixed Phase 6.6 best on this grid; best adaptive delta vs fixed best is success `+0`, CAPTURE `+0`, near-miss `-3`.
2. The shift-vs-widen trade-off is `not reduced` relative to fixed_best_0p80: fixed changed `43` success-set memberships vs original, best adaptive changed `43`.
3. The best adaptive schedule is `adaptive_soft` by the same ranking rule.
4. Any improvement is mainly from CAPTURE access if CAPTURE changes with success. Here success delta `+0` and CAPTURE delta `+0` match, so this is not evidence for post-CAPTURE stabilization gains.
5. Best next step: keep the same 2D Python evaluator and refine only the adaptive schedule thresholds/norms around the best adaptive rule, while explicitly checking gained/lost success-set membership against both original WS-1 and fixed_best_0p80.

## Caution

This is a local adaptive-WS refinement only. It does not alter environment physics, learned policies, CAPTURE/LOCK equations, or the Phase 3-6.6 output directories.