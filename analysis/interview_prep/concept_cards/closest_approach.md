## Name

Closest approach is a diagnostic metric measuring how near a trajectory came to the target radius.

---

## Why does this concept exist?

It exists to analyze non-crossing cases. A trajectory may not cross but may still move closer to a crossing-like geometry.

---

## Repository Evidence

Evidence cited in the audit: Phase36C and Phase37B. Phase37B closest approach improved in `3 / 4` selected cases but created `0 / 4` selected crossings.

---

## Mathematics

Closest approach is based on minimum absolute radius error:

```text
min_t |r(t) - r_target|
```

It is scalar and geometric.

---

## Engineering

Reported in Phase36C/37 diagnostic outputs and CSV fields according to the audit.

---

## Scientific Meaning

It helps diagnose whether a controller is moving in a promising direction, but it is not a success metric.

---

## Common Misunderstandings

- Mistake: closest approach improvement means crossing success. Wrong; Phase37B improved closest approach but created zero selected crossings.
- Mistake: closest approach implies recoverability. Wrong; velocity state may still be bad.

---

## Reviewer Objections

- It is only a proxy.
- It may encourage optimizing the wrong objective.
- It ignores velocity synchronization.

---

## How Sean Should Respond

Say closest approach is diagnostic-only. It may motivate further search but cannot be counted as crossing or recoverability.

---

## Related Concepts

Closest approach -> Crossing potential -> Phase36C -> Phase37B -> Target-radius crossing

---

## Difficulty

Easy

---

## Interview Probability

75%

---

## Importance

Useful

---

## 30-Second Explanation

Closest approach measures how near the trajectory got to the target radius. It is useful diagnostically, but it does not count as a target-radius crossing.

---

## 3-Minute Explanation

For non-crossing cases, closest approach helps tell whether a controller moved geometry in the right direction. But Phase37B shows the danger: weak tangential shaping improved closest approach in `3 / 4` selected cases but produced `0 / 4` selected crossings and damaged regression preservation. So it is only a proxy.

---

## One-Sentence Safe Claim

Closest approach is a diagnostic proxy for non-crossing geometry, not evidence of crossing or recoverability.

---

## One Dangerous Overclaim

"Closest approach improved, so the controller worked." This is unsafe because Phase37B shows closest approach can improve without crossings.

---

## Follow-Up Questions

1. How is closest approach computed?
2. Why is it not a crossing?
3. What did Phase37B teach about it?
4. When is it useful?
5. How could it mislead future experiments?

---

## Confidence Checklist

□ I can define closest approach.  
□ I know Phase37B numbers.  
□ I can say it is diagnostic-only.  
□ I can distinguish it from crossing.  
□ I can explain why velocity still matters.

