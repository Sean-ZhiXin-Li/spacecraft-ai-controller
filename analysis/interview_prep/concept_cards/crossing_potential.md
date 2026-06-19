## Name

Crossing potential is a diagnostic proxy that estimates whether a non-crossing trajectory moved toward crossing-like geometry.

---

## Why does this concept exist?

It exists to compare non-crossing trajectories that do not produce discrete target-radius crossings. It helps rank possible upstream shaping variables.

---

## Repository Evidence

Evidence cited in the audit: Phase36C and Phase37A/B diagnostics. The audit repeatedly warns that crossing potential is not actual crossing.

---

## Mathematics

The audit does not provide a full equation. Conceptually, it scores proximity to crossing-like conditions. Repository evidence is insufficient to support a stronger mathematical claim.

---

## Engineering

Reported in Phase36C/37 diagnostic outputs and used for upstream search interpretation according to the audit.

---

## Scientific Meaning

It supports hypothesis narrowing but cannot be used as proof of controller success.

---

## Common Misunderstandings

- Mistake: crossing potential equals crossing. Wrong.
- Mistake: improving crossing potential means recoverability improved. Wrong.

---

## Reviewer Objections

- Proxy metric may not correlate with actual crossing.
- Formula and thresholds may be heuristic.
- It can distract from event counts.

---

## How Sean Should Respond

Say it is a search diagnostic only. Actual claims must use target-radius crossing and recoverable crossing counts.

---

## Related Concepts

Crossing potential -> Closest approach -> Phase36C -> Phase37A -> Phase38 future work

---

## Difficulty

Medium

---

## Interview Probability

65%

---

## Importance

Useful

---

## 30-Second Explanation

Crossing potential is a diagnostic score for whether a non-crossing trajectory is moving toward crossing-like geometry. It is not an actual crossing.

---

## 3-Minute Explanation

The project needed diagnostics for cases that never crossed, so crossing potential helps compare whether variants moved in a promising direction. But the audit is clear: proxy metrics must not be counted as crossing. They can motivate the next experiment, but actual evidence still requires discrete crossing and recoverable crossing counts.

---

## One-Sentence Safe Claim

Crossing potential is a diagnostic search proxy and should not be treated as a target-radius crossing.

---

## One Dangerous Overclaim

"Crossing potential improved, so crossing-basin expansion succeeded." This is unsafe because the audit says proxy improvements are not crossings.

---

## Follow-Up Questions

1. What is crossing potential used for?
2. Why is it weaker than crossing count?
3. How did Phase36C use it?
4. What would validate it as a proxy?
5. How could it lead to false conclusions?

---

## Confidence Checklist

□ I can state it is diagnostic-only.  
□ I know actual crossing counts matter.  
□ I can connect it to Phase36C/37.  
□ I can explain proxy risk.  
□ I can avoid using it as success evidence.

