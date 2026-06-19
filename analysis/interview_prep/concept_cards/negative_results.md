## Name

Negative results are experiments that failed to improve the target metric but narrowed the project’s scientific hypothesis space.

---

## Why does this concept exist?

They exist because the project’s maturity comes partly from not hiding failures. Negative results clarify what does not work under tested conditions.

---

## Repository Evidence

Evidence cited in the audit: PPO/BC/IL failures, Phase35, Phase36B/C, Phase37A, and Phase37B. Examples: Phase37A `0 / 16` new crossings; Phase37B `0 / 4` selected crossings and `4 / 8` regression preservation.

---

## Mathematics

Negative results are judged by the same metrics: crossing count, recoverable crossing count, overspeed, instability, closest approach, and regression preservation.

---

## Engineering

Stored in phase summaries, artifacts, and regression-checked outputs according to the audit.

---

## Scientific Meaning

Negative results prevent false progress. They motivate better questions such as evidence-backed Phase38 search.

---

## Common Misunderstandings

- Mistake: negative result means project failed. Wrong; it narrows hypotheses.
- Mistake: failed tested variant proves impossibility. Wrong; only tested conditions are rejected.

---

## Reviewer Objections

- Negative results may be due to poor implementation.
- Search spaces may be too narrow.
- Conclusions may be stronger than evidence.

---

## How Sean Should Respond

Use scoped language: "tested variants did not..." rather than "this can never work."

---

## Related Concepts

Negative results -> Phase37B -> Benchmark contract -> Scientific contribution -> Future work

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

Negative results are valuable because they show which hypotheses failed under controlled conditions. Phase37B is important because it improved closest approach but still failed crossings and regression preservation.

---

## 3-Minute Explanation

The project’s research arc depends on negative results. PPO/BC/IL did not recover explicit structure. Phase35/36/37 did not expand the crossing basin. These failures are not hidden; they define the next question. The safe way to present them is as scoped evidence, not impossibility proofs.

---

## One-Sentence Safe Claim

Negative results in the repository narrowed the search space and prevented overclaiming unsupported controller success.

---

## One Dangerous Overclaim

"These failures prove no upstream method can create crossings." This is unsafe because only specific tested variants failed.

---

## Follow-Up Questions

1. What is the most important negative result?
2. How did Phase37B change future work?
3. How can negative results be rigorous?
4. What is the difference between failure and falsification?
5. How do you avoid overgeneralizing?

---

## Confidence Checklist

□ I can explain negative results as scientific evidence.  
□ I know Phase37A/B numbers.  
□ I can avoid impossibility claims.  
□ I can connect failures to future work.  
□ I can answer reviewer skepticism.

