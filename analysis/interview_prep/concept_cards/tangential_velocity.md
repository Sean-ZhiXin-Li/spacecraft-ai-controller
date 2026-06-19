## Name

Tangential velocity is the component of velocity perpendicular to the radius vector and is compared to circular orbit velocity in recoverability diagnostics.

---

## Why does this concept exist?

It exists because being at the target radius is not enough; the spacecraft also needs appropriate sideways velocity to remain orbit-like rather than falling inward or escaping.

---

## Repository Evidence

Evidence cited in the audit: recoverability definitions, Phase33/34 summaries, Phase37B weak tangential diagnostic, and physics/code references.

---

## Mathematics

Tangential velocity is the velocity component orthogonal to radial direction. Its error is compared to:

```text
v_circ = sqrt(mu / r_target)
```

Recoverability requires tangential velocity near circular.

---

## Engineering

Used in diagnostic fields such as `crossing_vt_error_ratio` and Phase37B weak tangential overlay according to the audit.

---

## Scientific Meaning

Tangential velocity explains why radius crossing is physically incomplete. It is central to recoverability and Phase37B’s tested hypothesis.

---

## Common Misunderstandings

- Mistake: target radius plus low radial velocity is enough. Wrong; tangential velocity can still be wrong.
- Mistake: Phase37B proves tangential shaping cannot work. Wrong; only one weak diagnostic failed.

---

## Reviewer Objections

- Tangential thresholds are hand-defined.
- Phase37B subset is too narrow for global conclusions.

---

## How Sean Should Respond

Say tangential velocity is essential in the simplified recoverability metric, but Phase37B only tests one limited weak shaping design.

---

## Related Concepts

Tangential velocity -> Circular orbit velocity -> Phase37B -> Recoverability

---

## Difficulty

Medium

---

## Interview Probability

90%

---

## Importance

Critical

---

## 30-Second Explanation

Tangential velocity is sideways velocity around the central body. For recoverability, it should be close to circular velocity; otherwise crossing the radius may not lead to an orbit-like state.

---

## 3-Minute Explanation

Orbital state is not just position. At the target radius, the tangential component determines whether the spacecraft has the right angular motion. Phase33/34 focus on synchronizing tangential velocity along with radius and radial velocity. Phase37B tested a weak tangential correction, but it did not create selected crossings and damaged regression preservation.

---

## One-Sentence Safe Claim

Tangential velocity error is a core part of the simulator-defined recoverability metric.

---

## One Dangerous Overclaim

"Phase37B proves tangential control is useless." This is unsafe because Phase37B was a narrow subset diagnostic.

---

## Follow-Up Questions

1. How is tangential velocity different from speed?
2. Why compare it to circular velocity?
3. What did Phase37B test?
4. Why did tangential shaping improve closest approach but still fail?
5. How would you design a better tangential experiment?

---

## Confidence Checklist

□ I can define tangential velocity geometrically.  
□ I know why circular velocity is the reference.  
□ I can connect it to recoverability.  
□ I can explain Phase37B carefully.  
□ I can avoid global tangential-control claims.

