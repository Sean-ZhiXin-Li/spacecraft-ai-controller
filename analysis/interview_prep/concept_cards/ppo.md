## Name

PPO is a reinforcement-learning policy optimization method evaluated in the project as a continuous-control learning baseline and fine-tuning approach.

---

## Why does this concept exist?

It was introduced because the project originally explored learned controllers and later tested whether PPO fine-tuning from behavior cloning could recover missing long-horizon structure.

---

## Repository Evidence

Evidence cited in the audit: `analysis/ppo_transfer_results.md`. Explicit controller: `1` crossing and success in fixed-baseline comparison; BC: `0` crossings; PPO fine-tuned from BC: `0` crossings. Final radius error BC `375039922.79`; PPO `374964010.17`.

---

## Mathematics

PPO optimizes a policy using clipped policy-gradient updates. The audit does not require detailed PPO equations. The important control point is that policy optimization did not recover crossing behavior in the documented setup.

---

## Engineering

Audit points to `ppo_orbit/` and PPO transfer artifacts. PPO was used as a learned policy/fine-tuning path, not the main positive result.

---

## Scientific Meaning

PPO is negative learning evidence. It shows the current positive result is explicit-controller based.

---

## Common Misunderstandings

- Mistake: PPO solved spacecraft control. Wrong; documented PPO had `0` crossings.
- Mistake: PPO failure proves RL cannot work. Wrong; only this setup failed.

---

## Reviewer Objections

- Reward design may be inadequate.
- Training may be too short.
- BC initialization or data balance may be poor.
- PPO may need different state representation or curriculum.

---

## How Sean Should Respond

Say PPO was investigated but did not produce the main positive result. Do not use PPO as evidence of AI success or as proof that RL cannot solve the problem.

---

## Related Concepts

PPO -> Behavior cloning -> Covariate shift -> Explicit controller -> Negative results

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

PPO was tested as a learned-controller route, including fine-tuning from BC. In the documented fixed-baseline comparison, PPO produced `0` crossings, so it is negative evidence, not the project’s main success.

---

## 3-Minute Explanation

PPO is a reasonable continuous-control baseline, but this project’s evidence shows it did not recover the explicit controller’s long-horizon phase behavior. The fine-tuned PPO result slightly improved final radius error relative to BC but still had `0` crossings and no success. The honest conclusion is that the tested PPO setup failed, not that all RL is impossible.

---

## One-Sentence Safe Claim

PPO was evaluated but did not reproduce successful crossing behavior in the documented transfer comparison.

---

## One Dangerous Overclaim

"PPO solved spacecraft control." This is directly contradicted by the audit’s reported PPO result.

---

## Follow-Up Questions

1. Why might PPO fail here?
2. What role did BC initialization play?
3. Could reward design explain failure?
4. What would a fairer RL experiment require?
5. Why keep PPO in the paper/project at all?

---

## Confidence Checklist

□ I know PPO had `0` crossings.  
□ I can explain PPO as negative evidence.  
□ I can avoid all-RL impossibility claims.  
□ I can discuss reward/design limitations.  
□ I can state the explicit controller was the positive result.

