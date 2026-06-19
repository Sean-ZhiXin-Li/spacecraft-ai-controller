## Name

Regression guard is the automated check that verifies protected aggregate results for key benchmark phases.

---

## Why does this concept exist?

It exists to prevent accidental drift in results and claims. If CSVs or scripts change, protected counts should fail rather than silently change.

---

## Repository Evidence

Evidence cited in the audit: `scripts/check_phase_results.py` passed locally for Phase34, Phase36B, Phase36C, Phase37A, and Phase37B. GitHub issue #3 concerns adding/protecting Phase37B checks.

---

## Mathematics

It checks aggregate counts, such as:

```text
Phase34 recoverable crossings = 8
Phase37A new crossings = 0
Phase37B weak selected crossings = 0 / 4
```

---

## Engineering

Implemented in `scripts/check_phase_results.py` according to the audit. It checks stored result artifacts, not full fresh reruns of every experiment.

---

## Scientific Meaning

Regression guards make the evidence trail more trustworthy by protecting central claims from accidental changes.

---

## Common Misunderstandings

- Mistake: regression guard means full reproducibility. Wrong; it validates stored outputs.
- Mistake: passing guard proves physics is correct. Wrong; it checks expected aggregate values.

---

## Reviewer Objections

- It may not rerun experiments.
- It may only check aggregates, not every row or mechanism.
- Local smoke tests did not run because `pytest` was missing.

---

## How Sean Should Respond

Say the regression guard is useful but limited. It protects key artifacts; it is not full independent replication.

---

## Related Concepts

Regression guard -> Benchmark contract -> Artifact manifest -> Reproducibility

---

## Difficulty

Medium

---

## Interview Probability

75%

---

## Importance

Critical

---

## 30-Second Explanation

The regression guard checks that the key stored benchmark results still match expected counts. It passed for Phase34, Phase36B/C, Phase37A, and Phase37B.

---

## 3-Minute Explanation

The project’s claims depend on exact counts, so `scripts/check_phase_results.py` protects them. It verifies results like Phase34 `8 / 24` recoverable crossings and Phase37B `0 / 4` selected crossings. But it is not the same as rerunning every experiment from scratch, and local pytest smoke tests could not run in the checked environments.

---

## One-Sentence Safe Claim

The regression guard validates protected aggregate result artifacts for the main benchmark phases.

---

## One Dangerous Overclaim

"The regression guard proves the whole project is fully reproducible." This is unsafe because it checks artifacts, not all full reruns.

---

## Follow-Up Questions

1. What phases does the guard check?
2. What does it not check?
3. Why did smoke tests not run locally?
4. How would you improve reproducibility?
5. Why are aggregate checks useful?

---

## Confidence Checklist

□ I know the guard passed.  
□ I know what it checks.  
□ I know what it does not check.  
□ I can explain the pytest gap.  
□ I can avoid overclaiming reproducibility.

