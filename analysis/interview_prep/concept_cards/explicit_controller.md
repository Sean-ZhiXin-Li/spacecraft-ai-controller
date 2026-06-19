## Name

An explicit controller is a hand-built controller whose logic and feedback structure are specified directly rather than learned end-to-end.

---

## Why does this concept exist?

It exists because the project needed interpretable control structure for diagnosing failures and testing hypotheses. Explicit controllers made it possible to isolate post-cross synchronization and upstream crossing-generation behavior.

---

## Repository Evidence

Evidence cited in the audit: Phase7.6, Phase31, Phase34, Phase37A/B summaries and scripts. The strongest positive result comes from Phase34 explicit post-cross synchronization, not PPO or behavior cloning.

---

## Mathematics

Explicit controllers use feedback on state errors such as radius error, radial velocity ratio, and tangential velocity error. They can include hand-designed gains, thresholds, and phase transitions.

---

## Engineering

Audit points to explicit controller scripts including Phase34, Phase37A, and Phase37B. It also notes duplicated rollout logic as an engineering weakness.

---

## Scientific Meaning

Explicit controllers are not just implementation choices. They are experimental instruments for testing architecture hypotheses.

---

## Common Misunderstandings

- Mistake: explicit means less scientific than learned. Wrong; here it enables controlled hypothesis tests.
- Mistake: explicit means robust or general. Wrong; hand-built controllers can be brittle.

---

## Reviewer Objections

- Hand tuning may overfit.
- Duplicated logic may drift.
- Explicit controllers may not scale.

---

## How Sean Should Respond

Say explicit controllers were chosen for interpretability and diagnostic control. Do not claim they are universal or optimal.

---

## Related Concepts

Explicit controller -> Phase-structured controller -> Phase34 -> Behavior cloning -> PPO

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

The explicit controllers encode feedback rules and phases directly. They were useful because I could inspect and change specific architecture components, especially post-cross synchronization.

---

## 3-Minute Explanation

In this project, explicit controllers made the research more interpretable. Rather than asking a learned policy to solve everything, I could test whether specific structures mattered: pre-window shaping, transfer families, post-cross synchronization, radial timing, and weak tangential shaping. Their limitation is that they are hand-tuned and may not generalize.

---

## One-Sentence Safe Claim

Explicit controllers provided interpretable, hypothesis-testable control architectures in the simplified benchmark.

---

## One Dangerous Overclaim

"Explicit controllers are the correct solution to spacecraft autonomy." This is unsafe because the evidence is simplified and benchmark-limited.

---

## Follow-Up Questions

1. Why not learn the controller directly?
2. What makes explicit control interpretable?
3. Where can explicit control overfit?
4. How did explicit controllers help Phase34?
5. How could learning be combined with explicit control?

---

## Confidence Checklist

□ I can defend explicit controllers scientifically.  
□ I can state their limitations.  
□ I can explain why learning did not replace them.  
□ I know Phase34 is explicit.  
□ I can avoid universal claims.

