## Research Question

Can global transfer-family controllers produce target-radius crossings and recoverable crossings more reliably than earlier explicit-controller behavior?

---

## Why was this question important?

Phase8 showed that local explicit control did not broadly reach the capture/crossing region. Phase31 asked whether changing the upstream transfer architecture could address that access problem and whether crossings would also be dynamically recoverable.

---

## Previous evidence

The audit cites `analysis/phase8_multiregime/phase8_summary.md` as showing `220 / 1296` success and `265 / 1296` crossings/CAPTURE, with dominant `no_capture_access`. Phase31 evidence is cited from `analysis/phase31_global_transfer_solver/summary.md` and later Phase34 comparison files.

---

## Competing hypotheses

- Global transfer families would increase crossings and recoverable crossings.
- They would improve crossings but still fail recoverability.
- The transfer family search space would be too limited.
- The recoverability thresholds would reveal a downstream post-cross problem rather than an upstream access problem.
- The reduced benchmark would be too small to support broad claims.

---

## Why was this experiment designed this way?

Phase31 used a reduced 48-case grid and named transfer-family variants. It tracked crossings and recoverable behavior rather than only a single success label. The design was meant to compare upstream transfer strategies while exposing whether crossing the target radius was actually enough.

---

## What result was expected?

A reasonable prediction was that global transfer families might raise crossing count compared with earlier local control. Whether those crossings would also be recoverable was uncertain and was the core scientific question.

---

## What actually happened?

The audit records that Phase31 used a reduced 48-case grid, that the best listed families reached `12` crossings, and that recoverable count was `0` for the listed families. In the controlled Phase34 reduced reference, Phase31-style behavior produced `8 / 24` crossings and `0 / 24` recoverable crossings.

---

## Interpretation

Phase31 supports the central distinction between target-radius crossing and recoverable crossing. It shows that a controller can reach the target-radius event without entering the simulator-defined recoverability basin. It does not prove that all transfer-family controllers fail, and it should not be mixed numerically with the 24-case Phase34 benchmark without explaining the denominator change.

---

## Alternative explanations

- The listed transfer families may not have covered the right search space.
- The reduced 48-case grid may have biased the result.
- Recoverability thresholds may have been strict or hand-defined.
- A better post-cross controller might have recovered after the same crossings.

---

## Reviewer criticism

Reviewer #2 would challenge the fairness of comparing Phase31 full-grid results with Phase34 reduced-grid results. They would also ask whether `0` recoverable crossings reflects a real control gap or a metric artifact.

---

## Sean's response

Keep the scopes separate. The full Phase31 result is background showing a crossing/recoverability gap. The controlled Phase34 reduced reference is the correct comparison for the main Phase34 claim: `8 / 24` crossings existed but `0 / 24` were recoverable before post-cross synchronization.

---

## If you repeated this experiment

I would lock a manifest-backed benchmark, store every transfer-family parameter set, and report crossing-state radius, radial-velocity, and tangential-velocity errors for each crossing.

---

## Future direction

The logical next experiment was Phase32: use direct shooting as a probe to ask whether recoverable states are reachable under the simplified dynamics in selected cases.

---

## Scientific maturity score

Question quality: 8/10. It asked the right transition question after Phase8.

Experimental design: 7/10. The reduced grid and named families were useful, but benchmark scope and denominator handling required care.

Evidence quality: 8/10. The `12` crossings and `0` recoverable result are high-value evidence.

Interpretation: 8/10. The crossing/recoverability separation is scientifically meaningful.

Claim safety: 7/10. Safe if denominator differences and metric limitations are stated.

Overall: 7.5/10. Phase31 is a key negative result that created the later research direction.
