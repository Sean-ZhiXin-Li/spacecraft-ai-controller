## Research Question

What control structure distinguishes a direct-shooting trajectory that becomes recoverable from a trajectory that merely crosses the target radius?

---

## Why was this question important?

Phase32 suggested selected recoverable states were reachable, but an optimized open-loop trajectory is not a reusable controller. Phase33 asked what mechanism could be extracted and implemented explicitly.

---

## Previous evidence

The audit cites `analysis/phase32_direct_optimal_control/summary.md`, `analysis/phase33_optimal_structure_extraction/phase33_summary.md`, and `analysis/phase33_optimal_structure_extraction/structure_decomposition.md`. Phase31 had shown crossing without recoverability, and Phase32 suggested selected recoverable states could be reached by direct shooting.

---

## Competing hypotheses

- The first target-radius crossing must already be recoverable.
- Recoverability can emerge after crossing through post-cross synchronization.
- The optimized trajectory succeeds because of smooth timing rather than a specific controller mode.
- The result is case-specific and should not be generalized.
- The main missing structure is radial/tangential synchronization rather than simply crossing earlier.

---

## Why was this experiment designed this way?

Phase33 decomposed the best direct-shooting trajectory rather than introducing a new benchmark controller. It tracked crossing step, best recoverability step, synchronization error, closest distance, and crossing-state distance so the project could identify what changed after the first target-radius event.

---

## What result was expected?

If crossing itself was enough, the first crossing state would already be close to the recoverability basin. If post-cross behavior mattered, the best recoverability state would occur later than the first crossing.

---

## What actually happened?

The audit records the best case as `recoverability_target / baseline_crossing_high_angle`. First crossing occurred at step `81`, best recoverability occurred at step `512`, best sync was `0.000464`, best distance was `0.000470`, and crossing-state distance was `2.313443`.

---

## Interpretation

Phase33 supports the hypothesis that post-cross synchronization can matter more than the first crossing event itself. The key evidence is the time separation between first crossing and best recoverability. It does not prove the mechanism for all cases; it is structure extraction from selected optimized behavior.

---

## Alternative explanations

- The late best state may reflect the specific direct-shooting objective rather than a general principle.
- The selected case may be unusually favorable.
- The recoverability distance may encode the desired behavior by construction.
- A different controller could potentially make the first crossing recoverable without post-cross synchronization.

---

## Reviewer criticism

Reviewer #2 would say that one representative optimized trajectory is not enough to claim a universal mechanism. They would ask whether the extracted structure survives when implemented in a closed-loop explicit controller.

---

## Sean's response

The response should be narrow: Phase33 did not prove a theorem. It generated a testable mechanism. The mechanism was then tested in Phase34 by adding post-cross synchronization and measuring whether recoverable crossings improved in the reduced benchmark.

---

## If you repeated this experiment

I would decompose multiple optimized trajectories, report whether the same delay between first crossing and best recoverability appears across cases, and compare against trajectories where recoverability is never reached.

---

## Future direction

The logical next experiment was Phase34: implement post-cross synchronization explicitly and test it against a controlled reduced benchmark.

---

## Scientific maturity score

Question quality: 9/10. It converted optimization output into a mechanism question.

Experimental design: 8/10. Structure extraction was the right bridge between Phase32 and controller design.

Evidence quality: 7/10. The numbers are precise, but the scope is selected rather than broad.

Interpretation: 9/10. The phase is strongest when framed as hypothesis generation.

Claim safety: 8/10. Safe if described as a mechanism candidate, not a proof.

Overall: 8.2/10. Phase33 is a mature research step because it turns a result into a falsifiable controller hypothesis.
