## Name

Recoverability basin is the simulator-defined region of state space where radius, radial velocity, and tangential velocity are within recoverability thresholds.

---

## Why does this concept exist?

It exists to define what kind of post-cross state counts as dynamically usable. Without a basin-like concept, a controller could be judged by visual crossing or a loose success label.

---

## Repository Evidence

Evidence cited in the audit: metric definition table, `docs/benchmark_contract.md`, Phase34 summary, and Phase34 scripts using recoverability thresholds.

---

## Mathematics

The basin can be thought of as a thresholded set:

```text
B = {x : |r_error| <= eps_r,
         |v_r_ratio| <= eps_vr,
         |v_t_error| <= eps_vt}
```

The repository evidence treats this as a diagnostic set, not a formally invariant set.

---

## Engineering

Implemented conceptually through threshold constants and fields such as `best_post_cross_distance`, `best_post_cross_sync`, and `recoverable_state` in Phase34-related scripts.

---

## Scientific Meaning

It gives the project a state-space interpretation. Instead of optimizing only an event, the project evaluates whether the trajectory enters a useful region after the event.

---

## Common Misunderstandings

- Mistake: basin means mathematically proven basin of attraction. Wrong; the audit explicitly says this is hand-defined.
- Mistake: entering the basin means real orbit achieved. Wrong; it is simulator-defined.

---

## Reviewer Objections

- Why these thresholds?
- Is the set invariant under the controller?
- Does the basin generalize outside the 24-case benchmark?

---

## How Sean Should Respond

State that the basin is an operational metric for this simplified simulator. It is useful for comparing controllers but should be strengthened in future with threshold sensitivity, formal terminal-set analysis, or reachability methods.

---

## Related Concepts

Recoverability basin -> Recoverability -> Recoverable crossing -> Phase34 -> Control theory

---

## Difficulty

Very Hard

---

## Interview Probability

90%

---

## Importance

Critical

---

## 30-Second Explanation

The recoverability basin is the set of simulator states where radius, radial velocity, and tangential velocity are all close enough to the desired orbit-like state. It is a benchmark-defined set, not a formal proof of stability.

---

## 3-Minute Explanation

In state-space terms, the project needs a target set that is stronger than crossing a radius. The recoverability basin is that target set: it requires simultaneous alignment in radius, radial velocity, and tangential velocity. Phase34’s result is important because crossing-producing cases later entered this basin. But I should be precise: the repository does not prove the basin is invariant or that it is physically complete.

---

## One-Sentence Safe Claim

The recoverability basin is a simulator-defined terminal-region proxy used to evaluate post-cross state quality.

---

## One Dangerous Overclaim

"The recoverability basin is a proven basin of attraction." This is unsafe because the repository evidence does not include formal proof.

---

## Follow-Up Questions

1. How would you prove invariance?
2. How would threshold sensitivity affect your conclusion?
3. What state variables are missing from the basin?
4. How would this change in 3D?
5. How does the basin relate to terminal sets in MPC?

---

## Confidence Checklist

□ I can define it as a set.  
□ I can explain why it is not formally proven.  
□ I know why it matters for Phase34.  
□ I can name one future improvement.  
□ I can answer a reviewer asking "why these thresholds?"

