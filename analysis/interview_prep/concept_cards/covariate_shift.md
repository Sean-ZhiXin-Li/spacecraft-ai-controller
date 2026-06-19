## Name

Covariate shift is the closed-loop distribution mismatch that occurs when a learned policy visits states different from the expert demonstrations.

---

## Why does this concept exist?

It explains why behavior cloning can have low supervised loss but fail during rollout. Small action errors can move the trajectory away from expert states, causing compounding errors.

---

## Repository Evidence

Evidence cited in the audit: BC/PPO transfer and minimal IL results. Learned policies had low training or validation signals but did not recover crossing behavior.

---

## Mathematics

Training distribution:

```text
s ~ expert rollouts
```

Deployment distribution:

```text
s ~ learned policy rollouts
```

If these differ, one-step action accuracy may not imply trajectory success.

---

## Engineering

Appears in learned-policy rollout failures documented in `analysis/ppo_transfer_results.md` and IL result artifacts.

---

## Scientific Meaning

Covariate shift gives a plausible mechanism for why learning transfer failed without claiming the model was incapable.

---

## Common Misunderstandings

- Mistake: BC failure means network capacity was too low. Maybe, but not proven.
- Mistake: one-step validation is enough. Wrong for long-horizon closed-loop control.

---

## Reviewer Objections

- The repository does not directly isolate covariate shift as the only cause.
- Reward design, data balance, architecture, or training duration could also matter.

---

## How Sean Should Respond

Say covariate shift is a plausible explanation consistent with the evidence, not a proven sole cause.

---

## Related Concepts

Covariate shift -> Behavior cloning -> PPO -> Phase-structured controller

---

## Difficulty

Medium

---

## Interview Probability

70%

---

## Importance

Important

---

## 30-Second Explanation

Covariate shift means the learned policy drifts into states the expert data did not cover, so small imitation errors compound. That helps explain why low BC loss did not produce crossing.

---

## 3-Minute Explanation

In supervised imitation, the policy learns from expert states. But during rollout, the learned policy controls the system. If it makes small errors, the trajectory can leave the expert distribution, and the model has no reliable training coverage there. This is especially dangerous in long-horizon orbital control with phase transitions.

---

## One-Sentence Safe Claim

Covariate shift is a plausible contributor to the documented behavior-cloning rollout failures.

---

## One Dangerous Overclaim

"Covariate shift is proven to be the only reason BC failed." This is unsafe because the repository does not isolate causes experimentally.

---

## Follow-Up Questions

1. How would you test for covariate shift?
2. How does DAgger address this?
3. Why is long horizon relevant?
4. Could phase imbalance also explain failure?
5. How would PPO fine-tuning help or fail?

---

## Confidence Checklist

□ I can define train vs rollout distribution.  
□ I can connect it to BC failure.  
□ I can state it is plausible, not proven.  
□ I know one mitigation idea.  
□ I can avoid overclaiming causality.

