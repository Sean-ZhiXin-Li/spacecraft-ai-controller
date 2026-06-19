## Name

Phase31 is the transfer-family benchmark that showed target-radius crossings could occur without recoverable crossings.

---

## Why does this concept exist?

It exists as the key baseline showing that geometric crossing and recoverability are different. It motivated the shift toward post-cross recoverability analysis.

---

## Repository Evidence

Evidence cited in the audit: `analysis/phase31_global_transfer_solver/summary.md` and Phase34 reduced comparison files. Full Phase31 used a reduced 48-case grid; best crossings were `12`, but recoverable count was `0` for listed families. The Phase34 reduced reference had `8 / 24` crossings and `0 / 24` recoverable crossings.

---

## Mathematics

Phase31 concerns trajectory families and crossing-state quality. It exposes that satisfying `r = r_target` does not constrain `v_r` or `v_t` enough to be recoverable.

---

## Engineering

The audit identifies Phase31 result files and Phase31 reference rows used by Phase34 comparison.

---

## Scientific Meaning

Phase31 is the empirical reason the project’s central claim exists. It falsifies "crossing implies insertion" in the simplified benchmark.

---

## Common Misunderstandings

- Mistake: mixing Phase31 full 48-case numbers with Phase34 24-case comparison.
- Mistake: calling Phase31 successful because it had CAPTURE/success labels.
- Mistake: ignoring recoverable count `0`.

---

## Reviewer Objections

- Are the Phase31 and Phase34 comparisons fair?
- Why use a reduced reference rather than only the full Phase31 benchmark?
- Were transfer families sufficiently diverse?

---

## How Sean Should Respond

Keep scopes separate. Say the full Phase31 result is background, and the Phase34 reduced reference is the controlled comparison used for the main Phase34 claim.

---

## Related Concepts

Phase31 -> Target-radius crossing -> Recoverable crossing -> Phase34

---

## Difficulty

Medium

---

## Interview Probability

85%

---

## Importance

Important

---

## 30-Second Explanation

Phase31 showed that transfer-style controllers could create target-radius crossings but still produce zero recoverable crossings. It is the baseline evidence that crossing alone is not enough.

---

## 3-Minute Explanation

Phase31 tested global transfer-family ideas. The important result is not just the crossing count but the recoverability failure: crossings existed, but recoverable crossings remained zero. In the Phase34 reduced comparison, the Phase31-style reference had `8 / 24` crossings and `0 / 24` recoverable crossings. That made Phase34’s later `8 / 24` recoverable result meaningful.

---

## One-Sentence Safe Claim

Phase31 provided simulator evidence that target-radius crossings can occur without recoverable post-cross states.

---

## One Dangerous Overclaim

"Phase31 solved insertion because it produced crossings." This is unsafe because recoverable crossings were zero.

---

## Follow-Up Questions

1. What denominator belongs to Phase31 full benchmark?
2. What denominator belongs to Phase34 reduced comparison?
3. Why did Phase31 motivate Phase32/33?
4. What does recoverable count `0` mean?
5. Could Phase31 have failed because of metric choice?

---

## Confidence Checklist

□ I can state Phase31 full vs reduced comparison separately.  
□ I know recoverable count was zero.  
□ I can explain why Phase31 matters.  
□ I can avoid calling crossings insertion.  
□ I can cite the relevant files.

