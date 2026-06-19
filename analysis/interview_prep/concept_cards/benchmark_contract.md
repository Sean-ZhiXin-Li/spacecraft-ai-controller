## Name

Benchmark contract is the documented agreement that fixes case scope, metrics, terminology, and comparison rules for the main benchmark phases.

---

## Why does this concept exist?

It exists to prevent incomparable scorecards and uncontrolled tuning. The project needed stable definitions for crossing, recoverability, simulator success, overspeed, and regression preservation.

---

## Repository Evidence

Evidence cited in the audit: `docs/benchmark_contract.md`. It defines the 24-case reduced benchmark and metric wording limits.

---

## Mathematics

The benchmark grid includes combinations of initial radius ratio, initial velocity angle, and thrust scale. It standardizes metric counts such as crossing count and recoverable crossing count.

---

## Engineering

Implemented as documentation and supported by result CSV schemas and `scripts/check_phase_results.py`. GitHub issue #4 asks for a mechanical 24-case manifest.

---

## Scientific Meaning

The benchmark contract is what makes Phase34/36/37 comparisons meaningful. Without it, later results could silently change scope.

---

## Common Misunderstandings

- Mistake: benchmark contract proves broad generalization. Wrong; it defines a controlled reduced benchmark.
- Mistake: the 24-case benchmark is complete. Wrong; it is structured and limited.

---

## Reviewer Objections

- 24 cases may be too small.
- Cases are structured, not random.
- A mechanical manifest is still needed according to the audit/GitHub issue.

---

## How Sean Should Respond

Say the benchmark supports controlled comparison, not broad generalization. Acknowledge the need for a mechanical manifest and larger future tests.

---

## Related Concepts

Benchmark contract -> Regression guard -> Artifact manifest -> Generalization -> Phase34

---

## Difficulty

Medium

---

## Interview Probability

85%

---

## Importance

Critical

---

## 30-Second Explanation

The benchmark contract fixes the cases, metrics, and terminology so Phase34/36/37 comparisons are meaningful. It supports controlled comparison but not broad generalization.

---

## 3-Minute Explanation

The project has many phases, so without a contract it would be easy to compare incompatible results. The benchmark contract defines the reduced 24-case scope and tells you how to use terms like crossing, recoverable crossing, simulator success, overspeed, and instability. It is a credibility tool, but it does not solve the limitation that 24 cases are small.

---

## One-Sentence Safe Claim

The benchmark contract provides stable definitions and scope for the main reduced-benchmark comparisons.

---

## One Dangerous Overclaim

"The benchmark contract proves the controller generalizes." This is unsafe because the contract defines scope; it does not prove generalization beyond it.

---

## Follow-Up Questions

1. What values define the 24-case grid?
2. Why is a benchmark contract needed?
3. What does it not prove?
4. How does it prevent overclaiming?
5. Why is GitHub issue #4 important?

---

## Confidence Checklist

□ I can define the benchmark contract.  
□ I know why 24 cases are limited.  
□ I can explain controlled comparison.  
□ I know mechanical manifest is future cleanup.  
□ I can avoid generalization claims.

