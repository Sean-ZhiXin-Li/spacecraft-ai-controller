## Name

Radial velocity is the component of velocity along the radius vector, indicating motion toward or away from the central body.

---

## Why does this concept exist?

It exists because a spacecraft can cross target radius while moving too quickly inward or outward, making the crossing dynamically unrecoverable.

---

## Repository Evidence

Evidence cited in the audit: recoverability definitions, Phase33/34 summaries, and code references to radial/tangential decomposition.

---

## Mathematics

Simple formula:

```text
v_r = dot(position, velocity) / ||position||
```

For a circular orbit-like state, radial velocity should be near zero.

---

## Engineering

The audit identifies radial velocity in environment observations and Phase34/37 diagnostic/control scripts.

---

## Scientific Meaning

Radial velocity is one reason target-radius crossing is insufficient. It captures whether the spacecraft is merely passing through the target radius.

---

## Common Misunderstandings

- Mistake: target radius implies radial velocity is okay. Wrong; a trajectory can cross at high radial speed.
- Mistake: only tangential velocity matters. Wrong; radial velocity affects recoverability.

---

## Reviewer Objections

- Radial velocity threshold is hand-defined.
- Simplified 2D dynamics limit physical interpretation.

---

## How Sean Should Respond

Say radial velocity is part of the simulator-defined recoverability proxy and a physically meaningful state component, but threshold validity is limited.

---

## Related Concepts

Radial velocity -> Recoverability -> Post-cross synchronization -> Phase37A

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

Radial velocity is how fast the spacecraft moves inward or outward. At crossing, high radial velocity can mean the spacecraft is just passing through the target radius rather than entering a recoverable state.

---

## 3-Minute Explanation

The position radius tells where the spacecraft is, but radial velocity tells whether it is moving through that radius. In a circular orbit-like state, radial velocity should be close to zero. Phase34 uses post-cross synchronization partly to reduce radial velocity mismatch after crossing.

---

## One-Sentence Safe Claim

Radial velocity is a necessary diagnostic component because radius crossing alone does not determine post-cross recoverability.

---

## One Dangerous Overclaim

"Low radial velocity alone proves recoverability." This is unsafe because tangential velocity and radius error also matter.

---

## Follow-Up Questions

1. How do you compute radial velocity?
2. Why should it be near zero?
3. What happens if radius is correct but radial velocity is large?
4. How did Phase37A manipulate radial behavior?
5. How does radial velocity relate to recoverability basin thresholds?

---

## Confidence Checklist

□ I can write the formula.  
□ I can explain the geometry.  
□ I can connect it to crossing failure.  
□ I can connect it to Phase34/37A.  
□ I can avoid saying it is sufficient alone.

