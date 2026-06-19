## Name

Direct shooting is an optimal-control method that optimizes a sequence of controls by rolling the dynamics forward.

---

## Why does this concept exist?

It exists in the project as the Phase32 fallback method after CasADi/IPOPT was unavailable. It allowed the project to test whether recoverable states could be reached under simplified dynamics.

---

## Repository Evidence

Evidence cited in the audit: `analysis/phase32_direct_optimal_control/summary.md`. Phase32 used SciPy direct shooting, horizon `512` physics steps, `64` control intervals, and 4 solves per objective.

---

## Mathematics

Direct shooting chooses controls:

```text
u_0, u_1, ..., u_N
```

Then simulates:

```text
x_{k+1} = f(x_k, u_k)
```

and minimizes a cost over the resulting trajectory.

---

## Engineering

Audit points to `scripts/phase32_direct_optimal_control.py` and Phase32 output artifacts.

---

## Scientific Meaning

Direct shooting helped separate physical reachability under simplified dynamics from failure of hand-built controllers.

---

## Common Misunderstandings

- Mistake: direct shooting equals formal optimal control proof. Wrong; it is local and solver-dependent.
- Mistake: Phase32 used CasADi. Wrong; CasADi was unavailable.

---

## Reviewer Objections

- Local minima.
- Small number of solves.
- Coarse horizon/control intervals.
- No robustness guarantee.

---

## How Sean Should Respond

Say direct shooting was a probe used to generate insight, not a validated optimal controller.

---

## Related Concepts

Direct shooting -> Phase32 Direct Shooting -> Trajectory Optimization -> Phase33 Structure Extraction

---

## Difficulty

Hard

---

## Interview Probability

75%

---

## Importance

Important

---

## 30-Second Explanation

Direct shooting optimizes control inputs, simulates the trajectory forward, and evaluates a cost. In Phase32 it was used as a SciPy fallback probe, not a full optimal-control proof.

---

## 3-Minute Explanation

Direct shooting parameterizes a control sequence and repeatedly rolls out the dynamics to evaluate objective values. It is simple and compatible with the existing simulator, but it can be sensitive to initialization and local minima. In this project its role was to find whether recoverable states were possible in selected cases and to motivate Phase33/34.

---

## One-Sentence Safe Claim

Direct shooting was used as a limited Phase32 probe to identify recoverability structure in selected simplified-dynamics cases.

---

## One Dangerous Overclaim

"Direct shooting proved the globally optimal insertion controller." This is unsafe because Phase32 was limited and used a fallback solver.

---

## Follow-Up Questions

1. How does direct shooting differ from collocation?
2. Why can direct shooting be sensitive?
3. What objective did Phase32 optimize?
4. What did Phase32 teach Phase33?
5. What would make this optimization more rigorous?

---

## Confidence Checklist

□ I can explain forward rollout optimization.  
□ I know Phase32 was SciPy fallback.  
□ I know it was not proof.  
□ I can compare it to collocation.  
□ I can connect it to Phase33.

