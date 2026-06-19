## Name

Trajectory optimization is the broader class of methods that optimize a trajectory and control sequence subject to dynamics and constraints.

---

## Why does this concept exist?

It exists because Phase32 used an optimization-style probe, and future work may use more rigorous trajectory optimization or MPC after the upstream crossing variables are clearer.

---

## Repository Evidence

Evidence cited in the audit: Phase32 direct-shooting summary and engineering defense notes warning not to claim full optimal-control success.

---

## Mathematics

A generic formulation:

```text
minimize    J(x_0:N, u_0:N)
subject to  x_{k+1} = f(x_k, u_k)
            u_k within thrust limits
            x_N in terminal/recoverability set
```

The repository does not provide a full formal trajectory-optimization proof.

---

## Engineering

Audit points to Phase32 scripts and states CasADi/IPOPT was unavailable, so full direct collocation was not used.

---

## Scientific Meaning

Trajectory optimization is relevant as a stronger future formalism for recoverability, but current evidence only supports the Phase32 probe.

---

## Common Misunderstandings

- Mistake: trajectory optimization is already fully implemented. Wrong; only direct-shooting probe is supported by audit evidence.
- Mistake: fuel optimality was proven. Wrong; repository evidence is insufficient.

---

## Reviewer Objections

- Current optimization evidence is small.
- No formal solver validation.
- No broad benchmark with optimized trajectories.

---

## How Sean Should Respond

Say trajectory optimization is an important direction, but the repository only supports a limited direct-shooting probe.

---

## Related Concepts

Trajectory optimization -> Direct shooting -> MPC -> Phase32 -> Recoverability basin

---

## Difficulty

Hard

---

## Interview Probability

65%

---

## Importance

Useful

---

## 30-Second Explanation

Trajectory optimization would optimize states and controls under dynamics and constraints. In this project, only a limited direct-shooting version was used as a Phase32 probe.

---

## 3-Minute Explanation

A full trajectory-optimization formulation would define state, control, dynamics, thrust constraints, and a terminal recoverability set. The repository does not yet support claims that this was solved rigorously. Phase32 is best understood as an exploratory optimization probe that revealed post-cross structure.

---

## One-Sentence Safe Claim

Trajectory optimization is relevant future machinery, but current repository evidence only supports a limited direct-shooting probe.

---

## One Dangerous Overclaim

"The controller is fuel-optimal." This is unsafe because the audit explicitly warns there is no fuel-optimal proof.

---

## Follow-Up Questions

1. What would the state and control be?
2. What terminal set would you use?
3. Why not solve collocation now?
4. How would fuel enter the objective?
5. What constraints matter most?

---

## Confidence Checklist

□ I can write a simple optimization formulation.  
□ I know Phase32 was limited.  
□ I can avoid fuel-optimal claims.  
□ I can explain why this is future work.  
□ I can connect it to recoverability.

