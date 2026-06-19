## Name

MPC, or model predictive control, is receding-horizon optimization that repeatedly plans short future control sequences and applies the first action.

---

## Why does this concept exist?

It appears as a possible future direction, but the audit says MPC should wait until upstream crossing-generation variables are clearer.

---

## Repository Evidence

Evidence cited in the audit: engineering defense and future-work notes. The audit states Phase36/37 show upstream geometry variable is still unclear, so MPC would add complexity before knowing what to optimize.

---

## Mathematics

MPC solves repeatedly:

```text
minimize over u_0:H  cost(x, u)
subject to dynamics and constraints
apply u_0
repeat at next state
```

In this project, a natural cost would include radius, radial velocity, tangential velocity, and terminal recoverability.

---

## Engineering

The audit does not identify MPC as implemented. It is future machinery, not current evidence.

---

## Scientific Meaning

MPC could formalize post-cross synchronization and handle constraints, but it should not distract from unresolved crossing-basin generation.

---

## Common Misunderstandings

- Mistake: MPC is already part of the result. Wrong; audit treats it as future work.
- Mistake: MPC automatically solves crossing. Wrong; objective and horizon design are hard.

---

## Reviewer Objections

- Why not use MPC from the start?
- Would MPC be computationally expensive?
- What objective would it optimize?

---

## How Sean Should Respond

Say MPC is a plausible next step after identifying useful upstream variables. Current repository evidence supports explicit post-cross synchronization and diagnostic search first.

---

## Related Concepts

MPC -> Trajectory optimization -> Recoverability basin -> Phase38 future work

---

## Difficulty

Hard

---

## Interview Probability

70%

---

## Importance

Useful

---

## 30-Second Explanation

MPC would repeatedly optimize short-horizon controls using the model. It is relevant future work, but the current repository does not implement it as the main result.

---

## 3-Minute Explanation

MPC could help because the task has constraints and sequential state objectives. But the audit argues it is premature: Phase36/37 still have not identified which upstream variable reliably creates crossings. Adding MPC before defining the right objective and search target could increase complexity without improving science.

---

## One-Sentence Safe Claim

MPC is a plausible future method for recoverability-aware control, but it is not a current repository result.

---

## One Dangerous Overclaim

"MPC would solve the remaining problem." This is unsafe because the repository has not tested MPC and crossing generation remains unclear.

---

## Follow-Up Questions

1. What would MPC optimize?
2. What horizon would be needed?
3. How would you include thrust limits?
4. Why not use MPC before Phase38?
5. How would MPC compare to Phase34?

---

## Confidence Checklist

□ I can define MPC.  
□ I know it is future work.  
□ I can state why it is premature.  
□ I can propose a cost function.  
□ I can avoid saying it will solve everything.

