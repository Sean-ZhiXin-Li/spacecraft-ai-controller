## Name

Recoverable crossing is a target-radius crossing followed later by entry into the simulator-defined recoverability basin.

---

## Why does this concept exist?

It was introduced because target-radius crossing alone inflated apparent progress. The project needed a stricter metric that measured whether a crossing could be converted into a stable, insertion-like post-cross state.

---

## Repository Evidence

Evidence cited in the audit: `docs/benchmark_contract.md`, `analysis/phase34_post_cross_sync/summary.md`, and `analysis/phase34_post_cross_sync/phase34_vs_phase31_comparison.md`. Phase34 `radius_priority` reached `8 / 24` recoverable crossings; the Phase31-style reduced reference had `0 / 24`.

---

## Mathematics

Recoverable crossing combines an event condition and a state-set condition:

```text
crossing occurred
and later |r_error|, |v_r_error|, |v_t_error| are within simulator thresholds
```

The first crossing state itself does not need to be recoverable.

---

## Engineering

The audit links this to Phase34 scripts and CSV fields such as `recoverable_crossing`, `best_post_cross_distance`, and `best_post_cross_sync`.

---

## Scientific Meaning

This is the project’s central evaluation upgrade. It tests whether a controller produces a useful post-cross trajectory, not just a radius event.

---

## Common Misunderstandings

- Mistake: assuming recoverable crossing means first crossing was recoverable. Wrong; Phase34 explicitly treats later post-cross basin entry as recoverability.
- Mistake: treating it as real spacecraft insertion. Wrong; it is simulator-defined.

---

## Reviewer Objections

- The recoverability thresholds are hand-defined.
- The basin is not formally proven invariant.
- The metric depends on simplified 2D dynamics.

---

## How Sean Should Respond

Say that recoverable crossing is an operational benchmark metric. It is stronger than crossing but still simulator-defined. Repository evidence is insufficient to claim real flight insertion or formal viability.

---

## Related Concepts

Target-radius crossing -> Recoverability basin -> Post-cross synchronization -> Phase34

---

## Difficulty

Medium

---

## Interview Probability

98%

---

## Importance

Critical

---

## 30-Second Explanation

A recoverable crossing means the trajectory crosses the target radius and then reaches a simulator-defined state where radius, radial velocity, and tangential velocity are simultaneously close enough to the desired orbit-like state.

---

## 3-Minute Explanation

The project found that crossing alone was too weak. Recoverable crossing adds the requirement that, after crossing, the controller can bring the state into a basin defined by radius error, radial velocity ratio, and tangential velocity error. Phase34 matters because it preserved the same `8 / 24` crossings but changed recoverable crossings from `0 / 24` to `8 / 24` in the reduced comparison.

---

## One-Sentence Safe Claim

Recoverable crossing is a simulator-defined metric that distinguishes geometric crossing from later post-cross basin entry.

---

## One Dangerous Overclaim

"Recoverable crossing proves real orbital insertion." This is unsafe because the project uses simplified 2D dynamics and simulator-defined thresholds.

---

## Follow-Up Questions

1. Does recoverable crossing require the first crossing state to be recoverable?
2. Which variables define the recoverability basin?
3. Why did Phase34 improve this metric?
4. What would make this metric more rigorous?
5. How sensitive is it to thresholds?

---

## Confidence Checklist

□ I can define recoverable crossing precisely.  
□ I know Phase34 improved it from `0 / 24` to `8 / 24`.  
□ I can explain why it is not real mission success.  
□ I can state the first crossing does not need to be recoverable.  
□ I can identify one reviewer criticism.

