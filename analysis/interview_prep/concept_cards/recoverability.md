## Name

Recoverability is the simulator-defined ability of a post-cross state to align radius, radial velocity, and tangential velocity closely enough for insertion-like stabilization.

---

## Why does this concept exist?

It exists because the project needed a stricter notion of progress than visual target-radius contact or simulator success labels. It solves the problem of false progress: a controller can cross the target radius while still being dynamically unrecoverable.

---

## Repository Evidence

Evidence cited in the audit: `docs/benchmark_contract.md`, Phase31, Phase33, and Phase34 summaries. Phase31-style reduced reference had `8 / 24` crossings and `0 / 24` recoverable crossings. Phase34 reached `8 / 24` recoverable crossings.

---

## Mathematics

Recoverability is based on a state vector containing at least:

```text
r_error = (r - r_target) / r_target
v_r relative to circular speed
v_t error relative to circular speed
```

It is not formal viability theory in the repository. It is an operational thresholded state-set.

---

## Engineering

The audit identifies `RECOVERABLE_R_RATIO`, `RECOVERABLE_VR_RATIO`, and `RECOVERABLE_VT_RATIO` as code-level threshold constants imported by Phase34 scripts.

---

## Scientific Meaning

Recoverability is the core scientific concept. It turns the problem from "did the spacecraft touch the target radius?" into "did the state become dynamically usable after crossing?"

---

## Common Misunderstandings

- Mistake: recoverability is mathematically proven. Wrong; it is operational.
- Mistake: recoverability equals success label. Wrong; success labels are simulator state-machine outputs.
- Mistake: recoverability applies to real spacecraft. Wrong; repository evidence is simplified 2D only.

---

## Reviewer Objections

- Thresholds are hand-defined.
- No formal invariant-set proof.
- No sensitivity analysis is highlighted in the audit.
- Simplified dynamics limit physical interpretation.

---

## How Sean Should Respond

Say: "I use recoverability as a simulator-defined viability proxy, not a theorem. The evidence supports it as a better diagnostic than crossing, but repository evidence is insufficient to claim formal reachability or real spacecraft readiness."

---

## Related Concepts

Recoverability -> Recoverability basin -> Recoverable crossing -> Phase34 -> Benchmark contract

---

## Difficulty

Hard

---

## Interview Probability

100%

---

## Importance

Critical

---

## 30-Second Explanation

Recoverability means the simulated state is not only near the target radius but also has radial velocity near zero and tangential velocity near circular. It is a simulator-defined proxy for whether the post-cross state can be stabilized.

---

## 3-Minute Explanation

The project’s key shift was realizing that crossing target radius does not guarantee a useful orbital state. A recoverable state must align position and velocity: radius close to target, radial velocity small, and tangential velocity close to circular speed. Phase33 showed the best recoverability state occurred after first crossing, and Phase34 used post-cross synchronization to reach recoverability in crossing-producing cases. I should be careful: this is not a formal invariant basin or flight proof.

---

## One-Sentence Safe Claim

Recoverability is a simulator-defined proxy for post-cross dynamic viability based on radius, radial velocity, and tangential velocity alignment.

---

## One Dangerous Overclaim

"Recoverability proves the controller is stable and mission-ready." This is unsafe because the repository does not prove formal stability or flight validation.

---

## Follow-Up Questions

1. Why do all three state components matter?
2. How would you formalize recoverability as a terminal set?
3. What would a formal viability proof require?
4. How sensitive is the conclusion to thresholds?
5. What experiment would make recoverability more convincing?

---

## Confidence Checklist

□ I can explain recoverability without calling it a proof.  
□ I know the three state components.  
□ I know Phase33 and Phase34 evidence.  
□ I can state its limitations.  
□ I can connect it to viability theory without overclaiming.

