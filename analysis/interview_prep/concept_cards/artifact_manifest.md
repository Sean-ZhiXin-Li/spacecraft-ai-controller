## Name

Artifact manifest is the document that maps public scientific claims to evidence files.

---

## Why does this concept exist?

It exists to prevent claims from floating without evidence. It tells readers which files support which result.

---

## Repository Evidence

Evidence cited in the audit: `analysis/artifact_manifest.md`. It lists evidence for Phase7.6, Phase8, Phase34, Phase36B/C, Phase37A/B, and Phase38 planning.

---

## Mathematics

No special mathematics. It is evidence organization: claim -> file -> metric -> scope.

---

## Engineering

Implemented as a documentation artifact. It works with the benchmark contract and regression guard.

---

## Scientific Meaning

It improves scientific honesty by making the evidence chain explicit.

---

## Common Misunderstandings

- Mistake: manifest itself is experimental evidence. Wrong; it points to evidence.
- Mistake: every file in the repo is public evidence. Wrong; manifest identifies claim-supporting artifacts.

---

## Reviewer Objections

- A manifest can become stale.
- It does not replace direct inspection of data.
- Some artifacts may be untracked unless preserved.

---

## How Sean Should Respond

Say the manifest is a map, not proof by itself. It makes claims auditable.

---

## Related Concepts

Artifact manifest -> Benchmark contract -> Regression guard -> Scientific contribution

---

## Difficulty

Easy

---

## Interview Probability

60%

---

## Importance

Useful

---

## 30-Second Explanation

The artifact manifest maps claims to files. It helps show which CSVs, summaries, and phase artifacts support each public result.

---

## 3-Minute Explanation

In a large project with many phase outputs, an artifact manifest prevents selective or confusing citation. It tells me which artifacts support Phase34, Phase36B/C, Phase37A/B, and other claims. It is not a result itself, but it makes the result trail auditable.

---

## One-Sentence Safe Claim

The artifact manifest improves claim traceability by linking major results to source files.

---

## One Dangerous Overclaim

"The artifact manifest proves the claims are correct." This is unsafe because it is an index, not independent validation.

---

## Follow-Up Questions

1. Why does a large repo need a manifest?
2. What artifacts are most important?
3. How can a manifest become stale?
4. How does it relate to regression checks?
5. What would you add to improve it?

---

## Confidence Checklist

□ I can explain what the manifest does.  
□ I know it is not proof by itself.  
□ I can name key phases it maps.  
□ I can connect it to claim safety.  
□ I can state one limitation.

