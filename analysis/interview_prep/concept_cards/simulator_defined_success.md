## Name

Simulator-defined success is an internal benchmark label produced under configured simulator thresholds, not real mission success.

---

## Why does this concept exist?

It exists because the project needs an internal success signal, but that signal can be misleading if interpreted as physical spacecraft success.

---

## Repository Evidence

Evidence cited in the audit: `docs/benchmark_contract.md`, Phase34/36 summaries, and audit warnings about wording drift. The audit repeatedly warns to say "simulator-defined success label."

---

## Mathematics

Success depends on simulator thresholds such as radius error, velocity error, angular alignment, and holding duration. The exact details are implementation-specific in the audit.

---

## Engineering

Implemented in environment/controller logic and reported in phase summaries as success/CAPTURE/LOCK style labels.

---

## Scientific Meaning

This concept protects scientific honesty. It distinguishes internal labels from real-world validation.

---

## Common Misunderstandings

- Mistake: simulator success equals mission success. Wrong.
- Mistake: CAPTURE/LOCK are real physical states. Wrong; they are simulator labels.
- Mistake: success alone is the main metric. Wrong; recoverability is more diagnostic.

---

## Reviewer Objections

- The success label may hide failure modes.
- Thresholds may be arbitrary.
- It can overstate performance.

---

## How Sean Should Respond

Always say "simulator-defined." Emphasize crossing, recoverability, overspeed, and instability separately rather than relying on success alone.

---

## Related Concepts

Simulator-defined success -> CAPTURE/LOCK -> Recoverability -> Benchmark contract

---

## Difficulty

Easy

---

## Interview Probability

85%

---

## Importance

Critical

---

## 30-Second Explanation

Simulator-defined success is an internal label under simplified simulator thresholds. It should not be interpreted as real spacecraft mission success.

---

## 3-Minute Explanation

The project reports success labels because the simulator needs stopping criteria and internal evaluation. But the stronger research story separates success from crossing and recoverability. CAPTURE, LOCK, and success are useful internal state-machine labels, not flight validation. This is one of the most important overclaim boundaries.

---

## One-Sentence Safe Claim

Simulator-defined success is an internal benchmark label and must be interpreted only within the simplified 2D simulator.

---

## One Dangerous Overclaim

"The spacecraft mission succeeded." This is unsafe because no real spacecraft mission or high-fidelity validation exists.

---

## Follow-Up Questions

1. Why not just report success?
2. How do CAPTURE and LOCK differ from real capture?
3. Which metrics are stronger than success alone?
4. How can wording drift create overclaims?
5. How would you design better success criteria?

---

## Confidence Checklist

□ I always say simulator-defined.  
□ I can distinguish success from recoverability.  
□ I can explain CAPTURE/LOCK.  
□ I know this is not mission success.  
□ I can identify wording risks.

