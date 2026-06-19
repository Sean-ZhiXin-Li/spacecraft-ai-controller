## Research Question

Did the Phase7.6 local explicit-controller success generalize to a broader multi-regime orbital-control map?

---

## Why was this question important?

This question was important because local success can be misleading. A broader map tests whether the controller solved a general control problem or only a favorable region of initial conditions.

---

## Previous evidence

Phase7.6 produced the strongest local historical result: `soft_linear_3e4` reached `217 / 270` success and `217 / 270` CAPTURE with `8` near-misses, as cited in the audit from `analysis/phase76_soft_hybrid/phase76_summary.md`.

---

## Competing hypotheses

- The Phase7.6 controller would generalize broadly across the expanded regime map.
- The controller would retain a success pocket but fail outside it.
- The main bottleneck would be reaching the target-radius/capture region at all.
- The main bottleneck would be post-capture or post-cross stability.
- The broader benchmark would expose that simulator success labels needed more diagnostic decomposition.

---

## Why was this experiment designed this way?

Phase8 used a completed `1296`-case multi-regime map to stress the controller across a wider set of conditions. The metrics included success, crossing/CAPTURE counts, and dominant failure modes. This made it a distribution-expansion test rather than another local tuning run.

---

## What result was expected?

A reasonable researcher would expect the success rate to drop when the controller left the local Phase7.6 regime. The uncertain question was whether the drop would be modest, indicating broad structure, or severe, indicating a narrow basin.

---

## What actually happened?

The audit records `220 / 1296` success and `265 / 1296` crossings/CAPTURE in Phase8, with dominant failure mode `no_capture_access`.

---

## Interpretation

Phase8 supports the conclusion that Phase7.6 did not broadly generalize across the expanded map. It also points toward upstream access to the capture/crossing region as a major bottleneck. It does not show that explicit controllers are useless; it shows that local success was not enough.

---

## Alternative explanations

- The expanded map may have included cases outside the controller's intended operating envelope.
- The controller may have needed a different transfer stage rather than different terminal logic.
- The failure-mode label may have grouped multiple geometric causes under `no_capture_access`.
- Success thresholds may have been too coarse to reveal near-miss structure.

---

## Reviewer criticism

Reviewer #2 would ask whether Phase8 is a fair generalization benchmark or an uncontrolled distribution shift. They might also ask whether the failure modes were analyzed deeply enough before moving to later phases.

---

## Sean's response

The honest answer is that Phase8 was a broad stress test, not a proof of impossibility. Its value was showing that Phase7.6 local success did not justify a global claim, and that later work needed to distinguish upstream crossing access from downstream recoverability.

---

## If you repeated this experiment

I would log the same diagnostic metrics later used in Phase34 through Phase37: target-radius crossing, recoverable crossing, overspeed, instability, closest approach, and a manifest-backed case identity.

---

## Future direction

The logical next direction was to decompose the failure: first test broader transfer families and crossing/recoverability separation, which appears in Phase31.

---

## Scientific maturity score

Question quality: 8/10. It directly tested whether a local result generalized.

Experimental design: 8/10. The `1296`-case expansion was a meaningful stress test.

Evidence quality: 8/10. The counts and dominant failure label are concrete, though less diagnostic than later metrics.

Interpretation: 8/10. The safe conclusion is clear: local success did not transfer broadly.

Claim safety: 8/10. The phase is strong if used as a generalization check rather than a final negative theorem.

Overall: 8/10. This was an important transition from engineering success to research discipline.
