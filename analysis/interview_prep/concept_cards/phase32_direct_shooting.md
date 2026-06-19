## Name

Phase32 direct shooting is the coarse finite-horizon optimization probe that searched for recoverable states under the simplified dynamics.

---

## Why does this concept exist?

It was introduced to test whether recoverable states were reachable at all under the project’s simplified dynamics after heuristic and transfer-family controllers failed recoverability.

---

## Repository Evidence

Evidence cited in the audit: `analysis/phase32_direct_optimal_control/summary.md`. Phase32 used SciPy direct shooting because CasADi/IPOPT was unavailable. Horizon: `512` physics steps, `64` control intervals, 4 solves per objective.

---

## Mathematics

Direct shooting optimizes a sequence of controls by simulating dynamics forward. The audit indicates objective modes such as recoverability target and sync-error minimization. It is not a proof of global optimality.

---

## Engineering

Audit points to `scripts/phase32_direct_optimal_control.py` and Phase32 summary artifacts.

---

## Scientific Meaning

Phase32 suggested that recoverable states could exist in selected simplified-dynamics cases. It motivated Phase33 structure extraction.

---

## Common Misunderstandings

- Mistake: saying CasADi/IPOPT solved the problem. Wrong; it was unavailable.
- Mistake: saying optimal control proved feasibility generally. Wrong; it was a small direct-shooting probe.
- Mistake: treating it as production controller.

---

## Reviewer Objections

- Direct shooting can be local and sensitive.
- Four solves per objective is small.
- No formal optimality or robustness proof.

---

## How Sean Should Respond

Call Phase32 a probe. Say it provided evidence that motivated Phase33/34, but repository evidence is insufficient to support claims of optimality or broad feasibility.

---

## Related Concepts

Phase32 -> Direct shooting -> Phase33 structure extraction -> Phase34 -> Trajectory optimization

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

Phase32 used SciPy direct shooting as a coarse optimization probe after CasADi was unavailable. It suggested recoverable states could be reached in selected simplified cases, but it was not a production controller or optimality proof.

---

## 3-Minute Explanation

Phase32 matters because it asked whether the recoverability failure was due to the controller architecture or the physics. Using a finite-horizon direct-shooting fallback, it found selected recoverability-targeted trajectories. The key scientific role was to generate structure for Phase33, not to claim a solved optimal controller. I should always mention the CasADi fallback and limited scope.

---

## One-Sentence Safe Claim

Phase32 provided selected direct-shooting evidence that recoverable states can exist under the simplified 2D dynamics.

---

## One Dangerous Overclaim

"Phase32 proved optimal control solved the benchmark." This is unsafe because the solver was a SciPy fallback and the evidence was limited.

---

## Follow-Up Questions

1. What is direct shooting?
2. Why did CasADi not run?
3. What was the horizon?
4. How does direct shooting differ from collocation?
5. What would make Phase32 more rigorous?

---

## Confidence Checklist

□ I can explain direct shooting.  
□ I know the Phase32 horizon and intervals.  
□ I can state the CasADi limitation.  
□ I know it motivated Phase33.  
□ I can avoid optimality overclaims.

