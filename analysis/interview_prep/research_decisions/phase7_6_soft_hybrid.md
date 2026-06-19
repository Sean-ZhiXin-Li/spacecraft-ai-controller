## Research Question

Could a soft phase-structured explicit controller produce reliable local simulator-defined insertion behavior across a modest 2D orbital-control grid?

---

## Why was this question important?

This phase tested whether the project had a usable explicit-control backbone before moving to broader benchmarks. It was important because later research questions only make sense if a controller can already solve some nontrivial subset of the simplified environment.

---

## Previous evidence

The audit identifies Phase7.6 as historical context in `analysis/phase76_soft_hybrid/phase76_summary.md`. The key recorded result was `soft_linear_3e4` with `217 / 270` success, `217 / 270` CAPTURE, and `8` near-miss cases.

---

## Competing hypotheses

- A soft phase blend would reduce brittle transitions compared with harder phase logic.
- The local result would be mainly due to careful hand tuning rather than a robust control principle.
- The controller would work only in the local grid and fail when the initial-condition distribution widened.
- CAPTURE/success counts would hide weaker dynamical quality that later recoverability metrics could expose.

---

## Why was this experiment designed this way?

The experiment used a 270-case local regime grid to test a named controller variant, `soft_linear_3e4`, against simulator-defined success and CAPTURE-style labels. The design was appropriate for an early architecture check because it stressed many local regimes while keeping the scope small enough for repeated controller iteration.

---

## What result was expected?

A reasonable expectation was that soft phase blending might improve continuity across controller modes and therefore increase local success relative to more brittle switching. A more cautious expectation was that success would still be distribution-specific.

---

## What actually happened?

The audit records `217 / 270` success, `217 / 270` CAPTURE, and `8` near-miss cases for `soft_linear_3e4` in Phase7.6.

---

## Interpretation

The result supports the claim that explicit phase structure was useful in a local 2D simulator setting. It does not support broad generalization, real spacecraft readiness, or a final controller architecture. Its main research value is that it created a credible local baseline that could be challenged by Phase8.

---

## Alternative explanations

- The local grid may have been easier than later benchmarks.
- The success label may have been too coarse for the later scientific question.
- Gains and phase thresholds may have been tuned to the distribution.
- The controller may have exploited simulator-specific behavior.

---

## Reviewer criticism

Reviewer #2 would likely say that a 270-case local grid is not enough to claim robust spacecraft control, and that CAPTURE/success labels do not establish recoverable orbital behavior. The reviewer could also ask whether the phase structure is a general design principle or just an engineered solution for a favorable subset.

---

## Sean's response

Phase7.6 should be presented as a historical local-controller milestone, not as the central contribution. The safe answer is: Phase7.6 showed that explicit phase structure could work locally in the simplified simulator, but Phase8 was needed because repository evidence did not justify a broad generalization claim.

---

## If you repeated this experiment

I would separate target-radius crossing, recoverable crossing, overspeed, instability, and closest approach from the beginning. I would also make the benchmark manifest mechanical so later comparisons cannot accidentally drift.

---

## Future direction

The natural next experiment was Phase8: run the controller over a much broader multi-regime map and test whether the local success survived distribution expansion.

---

## Scientific maturity score

Question quality: 7/10. It asked a real architecture question, but it was still framed around local success rather than the later recoverability distinction.

Experimental design: 7/10. The 270-case grid was useful for early controller development, but it was not a broad generalization benchmark.

Evidence quality: 7/10. The recorded `217 / 270` result is concrete, but the metric set was less mature than later phases.

Interpretation: 8/10. The safest interpretation is narrow and historically useful.

Claim safety: 7/10. Safe if described as a local milestone; unsafe if described as solved spacecraft control.

Overall: 7/10. Strong early engineering evidence, but not the main scientific result.
