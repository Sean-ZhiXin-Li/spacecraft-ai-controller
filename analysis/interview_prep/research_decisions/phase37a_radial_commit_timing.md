## Research Question

Can changing radial commitment timing and magnitude create new crossings in the Phase34/36 non-crossing cases while preserving known recoverable crossings?

---

## Why was this question important?

Phase36C suggested some cases were near-crossing or over-conservative. Phase37A tested whether a concrete upstream variable, radial commitment timing, could convert those diagnosed failures into actual target-radius crossings.

---

## Previous evidence

The audit cites `analysis/phase36c_non_crossing_geometry_diagnosis/summary.md` and `analysis/phase37a_radial_commit_timing/phase37a_summary.md`. Phase36C found `16 / 24` baseline non-crossing cases split into `8` near-crossing and `8` over-conservative transfer cases.

---

## Competing hypotheses

- Earlier radial commitment would help over-conservative cases cross.
- Delayed radial commitment would preserve known crossing cases better.
- Higher radial commitment magnitude would create new crossings.
- Radial timing alone would be insufficient without tangential shaping.
- Some variants would create new crossings but damage regression cases or safety diagnostics.

---

## Why was this experiment designed this way?

Phase37A evaluated six variants, crossing timing and magnitude settings across the 24-case benchmark, for `144` total rollouts. It kept Phase34 terminal/post-cross behavior fixed and measured new crossings on the `16` non-crossing cases plus preservation of known behavior. Overspeed and instability were also tracked.

---

## What result was expected?

If radial commitment was the missing variable, at least one variant should create new crossings among the `16` non-crossing cases. A reasonable safety expectation was that the best variant should also preserve the original `8 / 24` crossing/recoverable cases.

---

## What actually happened?

The audit records `0 / 16` new crossings across Phase37A. Delayed low and delayed medium preserved `8 / 24`. Overspeed was `0`, and instability was `0`.

---

## Interpretation

Phase37A supports the conclusion that the tested radial timing and magnitude changes were insufficient to expand the crossing basin. It also suggests that delayed variants were safer for preserving the existing `8 / 24` result. It does not prove that radial control is irrelevant in general.

---

## Alternative explanations

- The six variants may not have covered the useful timing/magnitude range.
- Radial commitment may need to be coupled with tangential shaping.
- The non-crossing cases may require a global trajectory plan rather than local radial changes.
- The fixed Phase34 terminal controller may interact with the upstream timing in a limiting way.

---

## Reviewer criticism

Reviewer #2 would say that `0 / 16` new crossings is a useful negative result, but the design may still be too narrow to reject radial strategies broadly. They would ask for parameter sweeps or sensitivity analysis.

---

## Sean's response

The safe answer is: Phase37A showed that these six radial-commitment variants did not create new crossings, and delayed variants best preserved the baseline. It does not show that all radial strategies fail.

---

## If you repeated this experiment

I would expand timing and magnitude sweeps only after defining regression criteria, and I would add per-case plots of radial velocity, target-radius gap, and tangential velocity error around the closest approach.

---

## Future direction

The next logical experiment was Phase37B: test whether a weak tangential overlay could help selected non-crossing cases after radial-only timing failed.

---

## Scientific maturity score

Question quality: 8/10. It tested a concrete variable from the Phase36C diagnosis.

Experimental design: 8/10. Six variants and `144` rollouts with safety diagnostics were appropriate.

Evidence quality: 8/10. `0 / 16` new crossings is clear evidence for this tested set.

Interpretation: 8/10. The conclusion is useful and narrow.

Claim safety: 8/10. Safe if not generalized to all radial control.

Overall: 8/10. A disciplined negative test of a specific upstream hypothesis.
