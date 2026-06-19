## Research Question

What geometric or behavioral patterns characterize the baseline non-crossing cases after Phase36B?

---

## Why was this question important?

After Phase35 and Phase36B failed to expand crossing count, another controller change without diagnosis would be weak science. Phase36C asked why the remaining cases failed and whether they formed interpretable categories.

---

## Previous evidence

The audit cites `analysis/phase36b_transfer_family_benchmark/summary.md` and `analysis/phase36c_non_crossing_geometry_diagnosis/summary.md`. Phase36B found that all four families stayed at `8 / 24` crossings and `8 / 24` recoverable crossings, leaving `16 / 24` baseline non-crossing cases.

---

## Competing hypotheses

- Some non-crossing cases are near the boundary and need small corrections.
- Some non-crossing cases are over-conservative and fail to commit enough energy or geometry.
- Proxy improvements in closest approach or crossing potential may be enough to predict a future crossing variant.
- Proxy improvements may not translate into actual target-radius crossings.
- Different families may reveal different failure mechanisms even without changing the aggregate count.

---

## Why was this experiment designed this way?

Phase36C was diagnostic rather than a new controller benchmark. It analyzed the existing Phase36B outputs and classified the `16 / 24` baseline non-crossing cases. This was appropriate because the previous aggregate results were flat and needed per-case explanation.

---

## What result was expected?

A reasonable expectation was that the non-crossing cases would not be uniform. If they clustered into interpretable types, later controller tests could be narrower and safer.

---

## What actually happened?

The audit records `16 / 24` baseline non-crossing cases. Phase36C classified them as `8` near-crossing and `8` over-conservative transfer cases. The audit also notes that Phase36C was diagnostic-only.

---

## Interpretation

Phase36C supports the conclusion that the unresolved cases have at least two diagnostic patterns. It does not show that either category is solvable by a specific controller. It also does not allow proxy metric improvement to be counted as success.

---

## Alternative explanations

- The near-crossing and over-conservative labels may be heuristic rather than formally derived.
- The split may depend on the chosen diagnostic thresholds.
- Some cases may belong to both categories depending on the feature used.
- The classification may not be causal.

---

## Reviewer criticism

Reviewer #2 would likely say that classification is not an experiment unless it leads to falsifiable predictions. They would ask how the categories were defined and whether they predict later controller outcomes.

---

## Sean's response

The honest response is that Phase36C is not a positive result. It is a diagnostic step that narrowed the next research question. The repository evidence supports the `8` near-crossing and `8` over-conservative split, but not a stronger causal claim.

---

## If you repeated this experiment

I would formalize the category rules, store per-case diagnostic features in the artifact manifest, and pre-register which category each later Phase37 variant is expected to help.

---

## Future direction

The next logical experiment was Phase37A: test whether radial commitment timing and magnitude could convert the diagnosed non-crossing cases into actual crossings.

---

## Scientific maturity score

Question quality: 8/10. It paused to diagnose instead of continuing blind tuning.

Experimental design: 7/10. Diagnostic reuse of prior outputs was appropriate but not causal.

Evidence quality: 7/10. The `8` and `8` split is useful but depends on diagnostic labels.

Interpretation: 8/10. The phase is strongest when framed as hypothesis generation.

Claim safety: 8/10. Safe because the audit explicitly treats it as diagnostic-only.

Overall: 7.6/10. A mature diagnostic bridge between negative benchmarks and targeted variants.
