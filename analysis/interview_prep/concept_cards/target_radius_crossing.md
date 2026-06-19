## Name

Target-radius crossing is a geometric event where the simulated trajectory reaches or crosses the target orbital radius.

---

## Why does this concept exist?

It exists because early controller results could look successful when the trajectory touched the target radius, even if the spacecraft state was not dynamically usable afterward. The project introduced this concept to separate geometric access from insertion-like recovery.

---

## Repository Evidence

Evidence cited in the audit: `docs/benchmark_contract.md`, `analysis/phase31_global_transfer_solver/summary.md`, `analysis/phase34_post_cross_sync/summary.md`, and Phase36/37 summaries. Phase31-style reduced reference had `8 / 24` crossings but `0 / 24` recoverable crossings.

---

## Mathematics

Let `r = ||position||`. A target-radius crossing occurs when the sign of `r - r_target` changes or the trajectory reaches the target-radius surface. This only constrains one scalar component of state. It does not constrain radial velocity, tangential velocity, orbital energy, angular momentum, or post-cross controllability.

---

## Engineering

The audit links this to Phase34/37 scripts and benchmark CSV fields such as `crossing_occurs`, `radius_crossings_total`, and `first_crossing_step`.

---

## Scientific Meaning

This concept is the first half of the central claim: crossing is not insertion. It supports the hypothesis that a controller can satisfy a geometric milestone while failing the actual control objective.

---

## Common Misunderstandings

- Mistake: treating crossing as insertion. Wrong because Phase31-style results crossed but were not recoverable.
- Mistake: treating first crossing quality as enough. Wrong because Phase33 showed best recoverability happened much later.

---

## Reviewer Objections

- A reviewer may say crossing is too weak to be a meaningful metric.
- A reviewer may ask whether target-radius crossing is only a proxy created by the benchmark.

---

## How Sean Should Respond

Say that crossing is intentionally weak and geometric. It is useful because the project shows why it is insufficient. Repository evidence is insufficient to support a stronger claim that crossing alone represents orbital insertion.

---

## Related Concepts

Target-radius crossing -> Recoverable crossing -> Recoverability -> Phase31 -> Phase34

---

## Difficulty

Easy

---

## Interview Probability

95%

---

## Importance

Critical

---

## 30-Second Explanation

Target-radius crossing means the simulated spacecraft reaches the target radius. It is a geometric event, not an insertion result. The project uses it because Phase31-style behavior could produce crossings without recoverable post-cross states.

---

## 3-Minute Explanation

The project separates radius crossing from the full orbital state. Crossing only says `r` reached `r_target`. But for an orbit-like state, radial velocity should be near zero and tangential velocity should be near circular velocity. Phase31-style reduced evidence had `8 / 24` crossings but `0 / 24` recoverable crossings, so crossing alone was misleading. This motivated post-cross recoverability metrics and Phase34 synchronization.

---

## One-Sentence Safe Claim

In the simplified 2D benchmark, target-radius crossing is a geometric event that is necessary but not sufficient for simulator-defined recoverability.

---

## One Dangerous Overclaim

"A target-radius crossing means the spacecraft entered orbit." This should never be said because the repository evidence explicitly shows crossings without recoverability.

---

## Follow-Up Questions

1. How is crossing detected in the benchmark?
2. Why is crossing not enough to define an orbit?
3. Which experiment best shows crossing without recoverability?
4. How would crossing change if the target radius changed?
5. Is crossing a metric, a contribution, or both?

---

## Confidence Checklist

□ I can explain this without notes.  
□ I know Phase31-style reduced evidence supports it.  
□ I know it is geometric only.  
□ I can distinguish it from recoverable crossing.  
□ I can explain why it should not be called mission success.

