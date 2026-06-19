## Name

Phase37B is the weak tangential subset diagnostic that tested a narrow tangential overlay and found no new selected crossings plus regression degradation.

---

## Why does this concept exist?

It exists because Phase37A did not create new crossings, and closest-approach changes suggested weak tangential shaping might help selected cases. Phase37B tested that idea cautiously before a larger sweep.

---

## Repository Evidence

Evidence cited in the audit: `analysis/phase37b_weak_tangential_subset/phase37b_summary.md` and `project_log/phase37b_weak_tangential_postmortem.md`. It ran `24` subset rollouts. Weak tangential shaping produced `0 / 4` selected crossings, `0 / 4` selected recoverable crossings, and preserved only `4 / 8` regression crossings and recoverable crossings. Closest approach improved in `3 / 4` selected cases.

---

## Mathematics

The tangential overlay modifies tangential velocity error with a bounded correction. It can reduce closest approach error slightly but still fail to create a discrete target-radius crossing or preserve known crossing cases.

---

## Engineering

Audit points to `scripts/explicit_controller_phase37b_weak_tangential_subset.py`. Phase34 terminal/post-cross controller remains fixed.

---

## Scientific Meaning

Phase37B is a high-value negative diagnostic. It shows proxy improvement is insufficient and regression preservation is mandatory.

---

## Common Misunderstandings

- Mistake: closest approach improvement means success. Wrong; selected crossings stayed `0 / 4`.
- Mistake: all tangential shaping is impossible. Wrong; only this weak subset diagnostic failed.
- Mistake: regression degradation is minor. Wrong; it damaged known good cases.

---

## Reviewer Objections

- Subset diagnostic is small.
- Weak tangential shaping may be too narrow.
- It cannot justify broad claims about tangential control.

---

## How Sean Should Respond

Say Phase37B is not a proof against tangential control. It is evidence that this specific weak overlay should not be expanded because it created no selected crossings and failed regression preservation.

---

## Related Concepts

Phase37B -> Tangential velocity -> Closest approach -> Regression guard -> Negative results

---

## Difficulty

Hard

---

## Interview Probability

90%

---

## Importance

Critical

---

## 30-Second Explanation

Phase37B tested weak tangential shaping on 4 selected non-crossing cases plus 8 regression crossing cases. It created `0 / 4` selected crossings and preserved only `4 / 8` regression crossings, so it was a negative diagnostic.

---

## 3-Minute Explanation

Phase37B is important because it prevents overreading proxy metrics. Weak tangential shaping improved closest approach in `3 / 4` selected cases, but it did not create a single selected crossing. Worse, it damaged regression preservation: only `4 / 8` known crossing cases remained crossing/recoverable. That means the idea was not safe to scale up without redesign.

---

## One-Sentence Safe Claim

In the Phase37B subset diagnostic, weak tangential shaping improved some closest-approach values but produced no selected crossings and failed full regression preservation.

---

## One Dangerous Overclaim

"Tangential shaping does not work." This is unsafe because Phase37B only tested one narrow weak tangential diagnostic.

---

## Follow-Up Questions

1. Why was Phase37B a subset diagnostic?
2. What does `4 / 8` regression preservation mean?
3. Why did closest approach improvement not count?
4. What would justify expanding tangential shaping?
5. How would you redesign this test?

---

## Confidence Checklist

□ I know `0 / 4`, `4 / 8`, and `3 / 4`.  
□ I can explain why it is negative.  
□ I can explain regression preservation.  
□ I can avoid saying tangential control is impossible.  
□ I can describe what would count as a positive diagnostic.

