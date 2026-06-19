## Research Question

Can explicit post-cross synchronization convert crossing-producing cases into simulator-defined recoverable crossings without relaxing the simplified benchmark?

---

## Why was this question important?

Phase31 showed crossings without recoverability, and Phase33 suggested that recoverability could occur after first crossing. Phase34 tested whether that mechanism could be implemented as a closed-loop explicit controller improvement.

---

## Previous evidence

The audit cites `analysis/phase31_global_transfer_solver/summary.md`, `analysis/phase33_optimal_structure_extraction/structure_decomposition.md`, `analysis/phase34_post_cross_sync/summary.md`, and `analysis/phase34_post_cross_sync/phase34_vs_phase31_comparison.md`. Phase33 recorded first crossing at step `81` and best recoverability at step `512`, with crossing-state distance `2.313443` and best distance `0.000470`.

---

## Competing hypotheses

- Post-cross synchronization is the missing structure for crossing-producing cases.
- First-crossing quality is the real bottleneck, so post-cross logic will not help.
- Any improvement will come from accidental changes in upstream behavior rather than the post-cross mode.
- The result will improve recoverability but not create new crossings.
- The thresholds will make the result look stronger than the underlying trajectory quality.

---

## Why was this experiment designed this way?

Phase34 used a 24-case reduced benchmark with Phase31-style reference behavior. It compared post-cross controller modes including `radius_priority`. The metrics included crossings, recoverable crossings, crossing-case best distance, and overspeed. The design was appropriate because it isolated the downstream post-cross question on cases that already produced crossings.

---

## What result was expected?

If Phase33's mechanism was useful, Phase34 should preserve crossings and improve recoverable crossings. If crossing-state quality dominated, recoverable count would remain low even with post-cross synchronization.

---

## What actually happened?

The audit records that the Phase31-style reduced reference produced `8 / 24` crossings and `0 / 24` recoverable crossings. Phase34 `radius_priority` produced `8 / 24` crossings and `8 / 24` recoverable crossings. Crossing-case best distance improved from `3.9923` to `0.9855`, and overspeed was `0`.

---

## Interpretation

Phase34 supports the claim that post-cross synchronization solved the tested post-cross recoverability gap for crossing-producing cases in the reduced benchmark. It does not support the claim that all cases are solved, that upstream crossing generation is solved, or that the controller is validated for real spacecraft.

---

## Alternative explanations

- The improvement may depend on hand-tuned recoverability thresholds.
- Duplicated rollout code could weaken causal attribution if more than post-cross behavior changed.
- The 24-case benchmark may be too small for broad generalization.
- The `radius_priority` controller may exploit simplified 2D dynamics.

---

## Reviewer criticism

Reviewer #2 would focus on benchmark size, threshold validity, and causal isolation. They might ask whether Phase34 actually changed only the downstream post-cross logic and whether the result survives held-out cases.

---

## Sean's response

The safe response is: Phase34 is the strongest positive result, but it is scoped. It shows that post-cross synchronization improved simulator-defined recoverability for crossing-producing cases in the reduced 24-case benchmark. Repository evidence is insufficient for real-world, all-case, or broad generalization claims.

---

## If you repeated this experiment

I would mechanically lock the 24-case benchmark, factor shared rollout code to reduce drift, run threshold-sensitivity checks, and add held-out crossing-producing cases.

---

## Future direction

The next logical research problem was upstream crossing-basin expansion: keep Phase34 terminal behavior fixed while testing whether new transfer logic can create crossings in the remaining non-crossing cases.

---

## Scientific maturity score

Question quality: 10/10. It directly tested the mechanism extracted from Phase33.

Experimental design: 8/10. The fixed reduced benchmark and reference comparison were strong, though benchmark size and code duplication limit certainty.

Evidence quality: 9/10. The `0 / 24 -> 8 / 24` recoverability change on preserved `8 / 24` crossings is strong repository evidence.

Interpretation: 9/10. The scoped interpretation is scientifically clean.

Claim safety: 8/10. Safe if limited to crossing-producing cases and simulator-defined recoverability.

Overall: 8.8/10. This is the project's strongest positive research decision.
