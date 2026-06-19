## Name

A phase-structured controller is an explicit controller organized into named behavioral regimes such as descent, capture, lock, and post-cross synchronization.

---

## Why does this concept exist?

It exists because orbital insertion-like behavior is sequential. The controller needs different feedback behavior before crossing, after crossing, and near stabilization.

---

## Repository Evidence

Evidence cited in the audit: Phase7.6, PPO/BC transfer results, Phase34, and IL result files. Learning-transfer failures suggested policies did not reproduce long-horizon phase structure.

---

## Mathematics

Phase structure can be viewed as a hybrid controller:

```text
u = f_i(x) depending on mode i
```

Switching logic determines which feedback law applies based on state events and thresholds.

---

## Engineering

Implemented across explicit controller phase scripts. The audit notes phase labels such as DESCENT, CAPTURE, LOCK, and POST_CROSS_SYNC.

---

## Scientific Meaning

It supports the hypothesis that successful behavior requires sequential architecture, not only one-step action matching.

---

## Common Misunderstandings

- Mistake: phases are just labels. Wrong; they encode different feedback objectives.
- Mistake: phase structure proves formal hybrid-system stability. Wrong; repository evidence does not prove that.

---

## Reviewer Objections

- Switching thresholds may be brittle.
- Hybrid logic may overfit the benchmark.
- Learned policies might fail because of implementation, not because phase structure is essential.

---

## How Sean Should Respond

Say phase structure is supported as useful in the tested benchmark, not proven necessary in all formulations.

---

## Related Concepts

Phase-structured controller -> Explicit controller -> Behavior cloning -> PPO -> Covariate shift

---

## Difficulty

Medium

---

## Interview Probability

80%

---

## Importance

Important

---

## 30-Second Explanation

A phase-structured controller uses different feedback behavior in different parts of the trajectory. In this project, that structure helped explain why one-step learned imitation failed to reproduce long-horizon behavior.

---

## 3-Minute Explanation

The project’s explicit controllers use phases because the task is not a single local regulation problem. Before crossing, the controller must generate a trajectory that reaches the target radius. After crossing, it must synchronize velocities and radius. Near the end, it must stabilize. Learning methods did not recover this sequence in the documented tests, which made phase structure scientifically important.

---

## One-Sentence Safe Claim

Phase-structured explicit control was useful for representing the sequential control logic required by the simplified benchmark.

---

## One Dangerous Overclaim

"Phase structure is mathematically required for all orbital insertion controllers." This is unsafe because the repository does not prove necessity.

---

## Follow-Up Questions

1. What are the main phases?
2. How are phase transitions triggered?
3. Why did learned policies struggle with phases?
4. Could a continuous policy replace phases?
5. How would you test phase-transition robustness?

---

## Confidence Checklist

□ I can define hybrid/phase control simply.  
□ I know phases are feedback regimes.  
□ I can connect this to PPO/BC failure.  
□ I can state brittleness risks.  
□ I can avoid necessity claims.

