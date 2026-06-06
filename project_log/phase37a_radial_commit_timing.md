# Phase37A - Radial Commitment Timing Sweep

## Scope

- Fixed 24-case reduced benchmark.
- Focused interpretation on the 16 Phase36B baseline non-crossing cases.
- Phase34 `radius_priority` post-cross synchronization is the fixed terminal controller after first crossing.
- No tangential search, no coast-duration variable, no MPC, and no RL.
- This is a simplified 2D orbital-control sandbox result, not real spacecraft validation.

## Experiment Purpose

- Test whether varying radial commitment timing and magnitude can convert baseline non-crossing cases into geometric crossings.
- Evaluate safety: ensure existing recoverable crossings are preserved, no overspeed or instability occurs.
- Provide evidence to guide Phase37B design.

## Parameters

| Commit timing | Radial magnitude label | Radial magnitude |
|---------------|----------------------|----------------|
| early_commit   | low                  | 0.055          |
| early_commit   | medium               | 0.105          |
| mid_commit     | low                  | 0.055          |
| mid_commit     | medium               | 0.105          |
| delayed_commit | low                  | 0.055          |
| delayed_commit | medium               | 0.105          |

- Fixed tangential shaping: conservative baseline, no coast-duration adjustment.

## Aggregate Results

| Variant               | Cases | Crossings | New crossings on baseline non-crossing | Phase34-compatible crossings | Recoverable crossings | Overspeed | Instability | Mean crossing sync |
|-----------------------|-------|-----------|---------------------------------------|-----------------------------|---------------------|-----------|------------|------------------|
| early_commit_low      | 24    | 4         | 0 / 16                                | 4                           | 4                   | 0         | 0          | 3.9797           |
| early_commit_medium   | 24    | 0         | 0 / 16                                | 0                           | 0                   | 0         | 0          | N/A              |
| mid_commit_low        | 24    | 4         | 0 / 16                                | 4                           | 4                   | 0         | 0          | 3.9797           |
| mid_commit_medium     | 24    | 0         | 0 / 16                                | 0                           | 0                   | 0         | 0          | N/A              |
| delayed_commit_low    | 24    | 8         | 0 / 16                                | 8                           | 8                   | 0         | 0          | 5.4831           |
| delayed_commit_medium | 24    | 8         | 0 / 16                                | 8                           | 8                   | 0         | 0          | 5.7453           |

**Total rollouts:** 144  
**Validation:** All requested CSV, summary, and figure outputs exist. Git diff checked.

---

## Interpretation

1. **No new crossings** were created among the 16 baseline non-crossing cases.
2. **Delayed commitment** preserved existing recoverable crossings at baseline levels (8 / 24).
3. Early and mid commitment degraded existing crossings (4 / 24 or 0 / 24).
4. No overspeed or instability observed.
5. Delayed commitment preserved the baseline crossing/recoverability count, but its mean crossing sync was higher than the Phase36B baseline.

**Conclusion:** Radial commitment timing alone is insufficient to generate new target-radius crossings. The experiment validates Phase36C evidence about the strongest signal (closest-approach timing) but confirms that additional mechanisms are needed for crossing-generation.

---

## Phase37B Design Guidance

Based on Phase37A:

- Do **not** expand radial commitment timing search blindly.
- Inspect **closest-approach metrics** from Phase37A for potential candidates.
- Phase37B may include:
  - Select baseline non-crossing cases with improved closest approach.
  - Introduce **tangential shaping** or refined radial magnitude only if Phase37A evidence indicates potential handoff improvement.
  - Keep Phase34 post-cross controller fixed.

**Decision logic:**

| Phase37A Result | Interpretation | Phase37B Action |
|-----------------|----------------|----------------|
| No new crossings, no improvement | Radial timing not the right lever | Reassess trajectory-family structure |
| No new crossings, but improved closest approach | Timing affects geometry | Phase37B: add tangential shaping for subset |
| New crossings, not recoverable | Crossing generation improved but handoff poor | Phase37B: optimize handoff metrics (vt_error, sync) |
| New recoverable crossings | Radial timing effective | Phase37B: refine around winning timing/magnitude region |
| Existing crossings degraded | Variant unsafe | Do not expand; restrict to non-crossing conditional logic |

---

## Artifacts

- `phase37a_results.csv`  
- `phase37a_summary.md`  
- `phase37a_comparison.png`

These should be archived in `analysis/phase37a_radial_commit_timing/` for reproducibility.

---

## Notes

- Experiment purpose: diagnostic parameter sweep, **not a solution to orbital insertion**.
- Fixed Phase34 terminal controller ensures reproducibility and safety.
- Provides causal evidence about radial timing as a potential upstream lever for new crossings.

---

**Next Step:** Use Phase37A insights to carefully plan Phase37B with limited tangential shaping adjustments, guided by closest-approach metrics.
