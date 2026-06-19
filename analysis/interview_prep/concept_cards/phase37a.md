## Name

Phase37A is the radial commitment timing sweep that tested whether radial timing and magnitude could create new crossings.

---

## Why does this concept exist?

It was introduced after Phase36C suggested non-crossing cases might need upstream crossing-generation changes. Phase37A tested whether radial commitment timing alone was sufficient.

---

## Repository Evidence

Evidence cited in the audit: `analysis/phase37a_radial_commit_timing/phase37a_summary.md`. Six variants across 24 cases produced `144` rollouts, `0 / 16` new crossings on baseline non-crossing cases, and `0` overspeed / `0` instability.

---

## Mathematics

The control perturbation modifies radial action timing and magnitude. It affects motion toward or away from target radius but does not by itself guarantee tangential velocity or phase alignment.

---

## Engineering

Audit points to `scripts/explicit_controller_phase37a_radial_commit_timing.py`. It keeps Phase34 `radius_priority` fixed as terminal controller.

---

## Scientific Meaning

Phase37A is a negative result that rules out the tested radial timing variants as sufficient crossing-basin expansion mechanisms.

---

## Common Misunderstandings

- Mistake: radial timing can never work. Wrong; only tested variants failed.
- Mistake: preserving `8 / 24` is improvement. Wrong; delayed variants preserved baseline but created no new crossings.

---

## Reviewer Objections

- The radial parameter search is limited.
- The timing labels may be too coarse.
- No new crossings means the hypothesis was not supported.

---

## How Sean Should Respond

Say Phase37A provides scoped negative evidence. It showed radial timing alone, in the tested variants, did not expand crossings while delayed variants preserved known crossings.

---

## Related Concepts

Phase36C -> Phase37A -> Radial velocity -> Phase37B -> Benchmark contract

---

## Difficulty

Medium

---

## Interview Probability

80%

---

## Importance

Important

---

## 30-Second Explanation

Phase37A tested six radial timing/magnitude variants over `144` rollouts. It created `0 / 16` new crossings on baseline non-crossing cases, so radial timing alone did not solve upstream crossing generation.

---

## 3-Minute Explanation

Phase37A responded to Phase36C’s diagnosis by testing a controlled radial-commitment variable while keeping Phase34 fixed after crossing. The best variants, delayed low and delayed medium, preserved `8 / 24` crossings and recoverable crossings, but no variant created new crossings on the `16` baseline non-crossing cases. This is useful because it narrows the next search.

---

## One-Sentence Safe Claim

Phase37A showed that the tested radial commitment timing variants did not create new crossings on the baseline non-crossing set.

---

## One Dangerous Overclaim

"Radial timing cannot help crossing generation." This is unsafe because Phase37A only tested a limited set of variants.

---

## Follow-Up Questions

1. Why was Phase37A tested after Phase36C?
2. How many variants and rollouts were used?
3. Which variants preserved baseline crossings?
4. Why is `0 / 16` important?
5. What would a stronger radial experiment look like?

---

## Confidence Checklist

□ I know `144` and `0 / 16`.  
□ I can explain why this is negative evidence.  
□ I know Phase34 stayed fixed.  
□ I can avoid global impossibility claims.  
□ I can explain what it taught.

