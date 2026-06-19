## Research Question

Can a weak tangential overlay create crossings in selected non-crossing cases without damaging known Phase34 recoverable crossings?

---

## Why was this question important?

Phase37A showed that radial commitment timing alone did not create new crossings. Phase37B tested the next plausible upstream variable, tangential shaping, but did so cautiously on a subset because regression damage was a real risk.

---

## Previous evidence

The audit cites `analysis/phase37a_radial_commit_timing/phase37a_summary.md`, `analysis/phase37b_weak_tangential_subset/phase37b_summary.md`, and `project_log/phase37b_weak_tangential_postmortem.md`. Phase37A produced `0 / 16` new crossings, while delayed variants preserved `8 / 24`.

---

## Competing hypotheses

- Weak tangential shaping would help selected near-crossing cases become actual crossings.
- It would improve closest approach but not discrete crossing.
- It would damage known crossing/recoverable regression cases.
- It would have no meaningful effect.
- Stronger or differently gated tangential shaping might be needed, but the weak overlay would be a safe diagnostic.

---

## Why was this experiment designed this way?

Phase37B used a subset diagnostic: `4` selected non-crossing cases plus `8` regression cases, with two settings for `24` total subset rollouts. It preserved the Phase34 terminal/post-cross logic and evaluated selected crossings, selected recoverability, regression preservation, closest approach, overspeed, and instability.

---

## What result was expected?

A positive diagnostic would have produced at least one selected crossing while preserving the known regression cases. A weaker but still useful result might have improved closest approach without regression loss.

---

## What actually happened?

The audit records `0 / 4` selected crossings and `0 / 4` selected recoverable crossings. Regression preservation was only `4 / 8`. Closest approach improved in `3 / 4` selected cases. Overspeed was `0`, and instability was `0`.

---

## Interpretation

Phase37B supports a negative conclusion for this specific weak tangential subset: proxy improvement did not become actual crossing, and regression preservation failed. It does not prove that all tangential shaping is impossible.

---

## Alternative explanations

- The weak overlay may have been too weak to create crossings.
- The gating may have applied tangential correction at the wrong time.
- The selected `4` cases may not represent all non-crossing cases.
- Regression damage may reflect implementation or interaction with Phase34 terminal logic.
- Stronger tangential shaping might work but would need stricter regression guards.

---

## Reviewer criticism

Reviewer #2 would say the subset is small and cannot justify broad claims about tangential control. They would also say closest-approach improvement is not sufficient evidence, especially when regression preservation dropped to `4 / 8`.

---

## Sean's response

The honest response is: Phase37B was a negative diagnostic. It improved closest approach in some selected cases, but it created no selected crossings and damaged regression preservation. Therefore this exact weak tangential overlay should not be scaled up without redesign.

---

## If you repeated this experiment

I would define stronger regression gates first, test tangential timing separately from magnitude, and require that any selected-case improvement preserve all known `8 / 8` regression crossings before expansion.

---

## Future direction

The most logical next experiment is a Phase38-style evidence-backed upstream search that treats closest approach as a proxy only, protects regression cases, and varies radial and tangential timing jointly.

---

## Scientific maturity score

Question quality: 8/10. It tested the next plausible variable after radial-only failure.

Experimental design: 7/10. The subset design was cautious, but small.

Evidence quality: 8/10. The `0 / 4`, `4 / 8`, and `3 / 4` numbers make the diagnostic clear.

Interpretation: 9/10. The negative result is interpreted correctly: proxy improvement is not success.

Claim safety: 9/10. The audit and concept card explicitly avoid saying all tangential control fails.

Overall: 8.2/10. A high-value negative diagnostic because it prevented an unsafe expansion.
