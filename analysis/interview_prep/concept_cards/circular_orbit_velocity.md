## Name

Circular orbit velocity is the ideal tangential speed for a circular orbit at radius `r`, given by `v = sqrt(mu / r)`.

---

## Why does this concept exist?

It exists because recoverability requires not only reaching the target radius but also matching the velocity state needed for an orbit-like condition.

---

## Repository Evidence

Evidence cited in the audit: physics cards and code references `simulator/physics.py`, `envs/orbit_env.py`, plus recoverability definitions in benchmark files.

---

## Mathematics

Core equation:

```text
v_circ = sqrt(mu / r)
```

where `mu` is the gravitational parameter and `r` is radius.

---

## Engineering

Implemented or used in simulator/environment logic and diagnostic scripts according to the audit.

---

## Scientific Meaning

It provides the velocity reference for tangential velocity error. Without it, target radius alone would be physically incomplete.

---

## Common Misunderstandings

- Mistake: correct radius implies correct orbit. Wrong; velocity must also match.
- Mistake: speed magnitude alone is enough. Wrong; direction and radial component also matter.

---

## Reviewer Objections

- Simplified circular velocity ignores real perturbations and 3D effects.
- The simulator is 2D point-mass, not high-fidelity astrodynamics.

---

## How Sean Should Respond

Say circular velocity is the correct reference for the simplified central-body model, but repository evidence is insufficient for real spacecraft dynamics.

---

## Related Concepts

Circular orbit velocity -> Tangential velocity -> Recoverability -> Target-radius crossing

---

## Difficulty

Medium

---

## Interview Probability

85%

---

## Importance

Critical

---

## 30-Second Explanation

Circular velocity is the speed needed for a circular orbit at a given radius in the simplified gravity model: `sqrt(mu/r)`. It is the reference for tangential velocity error.

---

## 3-Minute Explanation

In a central gravitational field, a circular orbit at radius `r` has speed `sqrt(mu/r)`. The project uses this because crossing the target radius does not mean the spacecraft has the correct velocity. Recoverability requires tangential velocity near this circular reference and radial velocity near zero.

---

## One-Sentence Safe Claim

Circular orbit velocity provides the simplified-model velocity reference used in recoverability diagnostics.

---

## One Dangerous Overclaim

"Matching circular velocity in this simulator validates a real spacecraft orbit." This is unsafe because real missions require more physics and validation.

---

## Follow-Up Questions

1. Derive or explain `sqrt(mu/r)`.
2. Why is tangential direction important?
3. What if radial velocity is nonzero?
4. How would 3D change this?
5. What if gravity parameter changes?

---

## Confidence Checklist

□ I know the equation.  
□ I can explain why radius alone is insufficient.  
□ I can connect it to tangential velocity error.  
□ I know the 2D limitation.  
□ I can avoid real-world overclaims.

