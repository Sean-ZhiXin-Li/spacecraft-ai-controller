## Research Question

Can broader named transfer families expand crossings beyond the Phase34 baseline while preserving recoverability and safety diagnostics?

---

## Why was this question important?

Phase35 showed that simple local upstream shaping did not expand crossings. Phase36B tested whether more structured transfer-family behavior could solve the remaining upstream access problem.

---

## Previous evidence

The audit cites `analysis/phase35_crossing_basin_expansion/summary.md` and `analysis/phase36b_transfer_family_benchmark/summary.md`. Phase35 preserved `8 / 24` with baseline and predictive bias, produced `0 / 24` with radial energy push and tangential corridor, and showed `5` overspeed cases for radial energy push.

---

## Competing hypotheses

- One transfer family would create new crossings in the `16 / 24` non-crossing cases.
- All families would preserve the same Phase34 crossing basin but fail to expand it.
- Transfer-family changes would improve proxy metrics without changing discrete crossings.
- More aggressive families would cause overspeed or instability.
- The benchmark would reveal that upstream access requires a different formulation such as search, MPC, or trajectory optimization.

---

## Why was this experiment designed this way?

Phase36B evaluated four transfer families on the 24-case reduced benchmark with Phase34 terminal behavior fixed. It tracked crossings, recoverable crossings, overspeed, instability, and related compatibility metrics. This design tested upstream strategy while protecting the known downstream result.

---

## What result was expected?

A reasonable expectation was that a broader transfer family might improve crossing count if the Phase35 variants were simply too local. The risk was that broader transfer logic might damage the `8 / 24` known crossing cases.

---

## What actually happened?

The audit records that all four families produced `8 / 24` crossings and `8 / 24` recoverable crossings. Overspeed was `0`, and instability was `0` for the families.

---

## Interpretation

Phase36B supports the conclusion that the tested transfer families did not expand the crossing basin. It also reinforces that Phase34 terminal behavior remains effective when a crossing is produced. It does not prove that no transfer family can work.

---

## Alternative explanations

- The family definitions may have been too similar in practice.
- Parameter ranges may have been too conservative.
- The benchmark may require targeted per-case timing rather than family-level behavior.
- The metrics may not capture useful intermediate differences between families.

---

## Reviewer criticism

Reviewer #2 would ask whether the four families were meaningfully different and whether the experiment had enough resolution to detect why they converged to the same `8 / 24` outcome.

---

## Sean's response

The response should be: Phase36B was a controlled negative result. It showed that the tested families preserved the known recoverable crossings but did not expand them. The project therefore needed diagnostic analysis rather than another broad controller tweak.

---

## If you repeated this experiment

I would report per-family parameter definitions, per-case closest approach and velocity errors, and a manifest-backed regression table showing exactly which eight cases stayed successful.

---

## Future direction

The next logical step was Phase36C: diagnose the `16 / 24` non-crossing cases rather than immediately inventing another controller.

---

## Scientific maturity score

Question quality: 8/10. It addressed the upstream bottleneck with a broader intervention.

Experimental design: 8/10. Four families with fixed terminal behavior and safety diagnostics were a sensible comparison.

Evidence quality: 8/10. The equal `8 / 24` result across all families is clear.

Interpretation: 8/10. The negative result is useful and appropriately scoped.

Claim safety: 8/10. Safe if described as "tested families failed to expand crossings."

Overall: 8/10. A strong negative benchmark that motivated diagnosis rather than tuning.
