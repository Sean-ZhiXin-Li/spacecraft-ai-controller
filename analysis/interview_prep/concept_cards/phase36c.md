## Name

Phase36C is the non-crossing geometry diagnosis that classified unresolved baseline cases without running a new controller.

---

## Why does this concept exist?

It exists because Phase34 solved post-cross recovery only for cases that already crossed. Phase36C was introduced to understand the remaining `16 / 24` non-crossing cases.

---

## Repository Evidence

Evidence cited in the audit: `analysis/phase36c_non_crossing_geometry_diagnosis/summary.md`. Baseline non-crossing cases were `16 / 24`, split into `8` near-crossing and `8` over-conservative transfer cases.

---

## Mathematics

Phase36C interprets closest approach and crossing-potential metrics. These are scalar diagnostics about how near a trajectory came to the target-radius event, not proof of crossing or recovery.

---

## Engineering

The audit states Phase36C reads existing Phase36B CSV outputs and does not change physics, thresholds, controller gains, or terminal controller behavior.

---

## Scientific Meaning

Phase36C separates the upstream crossing-generation problem from the downstream post-cross recoverability problem. It helps define Phase38-style search variables.

---

## Common Misunderstandings

- Mistake: Phase36C is a controller improvement. Wrong; it is diagnostic-only.
- Mistake: closest approach means crossing. Wrong; Phase36C explicitly separates proxy metrics from events.

---

## Reviewer Objections

- Diagnostics may not explain causal mechanisms.
- Failure labels may depend on hand-designed thresholds.
- No new controller result means it cannot prove a solution.

---

## How Sean Should Respond

Say Phase36C narrowed the hypothesis space. Repository evidence is insufficient to claim it solved non-crossing cases or identified a definitive causal variable.

---

## Related Concepts

Phase36C -> Closest approach -> Crossing potential -> Phase37A -> Phase37B

---

## Difficulty

Medium

---

## Interview Probability

75%

---

## Importance

Important

---

## 30-Second Explanation

Phase36C diagnosed the `16 / 24` cases that Phase34 baseline did not cross. It split them into `8` near-crossing and `8` over-conservative transfer cases but did not run a new controller.

---

## 3-Minute Explanation

After Phase34, the project needed to understand why non-crossing cases remained. Phase36C analyzed existing Phase36B results and found two categories: near-crossing and over-conservative transfer. It showed that proxy metrics could move without creating actual crossings, so the next step should be evidence-backed upstream search rather than another broad heuristic.

---

## One-Sentence Safe Claim

Phase36C diagnosed the unresolved non-crossing set but did not itself improve controller performance.

---

## One Dangerous Overclaim

"Phase36C fixed the crossing basin." This is unsafe because it was diagnostic-only.

---

## Follow-Up Questions

1. What were the two Phase36C failure labels?
2. Why is Phase36C not a controller result?
3. How did it motivate Phase37A?
4. Why are proxy metrics dangerous?
5. What would make the diagnosis causal?

---

## Confidence Checklist

□ I know the `16 / 24`, `8`, `8` numbers.  
□ I can state Phase36C was diagnostic-only.  
□ I can explain closest approach vs crossing.  
□ I can connect Phase36C to Phase37A/B.  
□ I can avoid solution claims.

