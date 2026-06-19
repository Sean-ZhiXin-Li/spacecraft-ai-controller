# Phase38 Evidence-Mining Summary

Scope: scientific data-mining summary from existing Phase34, Phase36B, Phase36C, Phase37A, and Phase37B CSV evidence. No controller experiment was rerun, no new controller was implemented, and no historical artifact was modified.

## What Do We Now Know?

- Crossing-producing cases are structurally different because they are the only cases with actual crossing events and crossing-state metrics.
- In the reduced benchmark evidence, `r0_over_target` is the clearest descriptive separator: Phase36C baseline near-crossing rows are on the `0.98` side, over-conservative-transfer rows are on the `1.02` side, and preserved Phase36B baseline crossing rows are associated with `1.00`.
- `closest_approach_step` separates near-crossing and over-conservative-transfer timing signatures in the recorded baseline failures.
- `best_crossing_potential` and `min_abs_radius_error_ratio` are useful diagnostics, but Phase36C and Phase37B show they can improve without producing target-radius crossings.
- Phase34 remains the positive downstream result: post-cross synchronization can convert already crossing-producing cases into recoverable crossings.
- Phase37A and Phase37B provide direct negative evidence against blindly expanding radial timing, radial magnitude, or weak tangential shaping.

## What Do We Still Not Know?

- We do not know whether coast duration is an independent variable or only a renamed form of radial commitment timing.
- We do not know whether angular momentum correction is a useful isolated variable because it is not directly recorded in the inspected CSVs.
- We do not know whether any diagnostic threshold in closest approach or crossing potential predicts actual crossings.
- We do not know whether the observed radius-regime separation generalizes beyond the reduced 24-case benchmark.
- We do not know how to create new target-radius crossings without damaging the known `8 / 24` crossing-producing regression set.

## Which Hypotheses Became Stronger?

- H1: Failure classes differ by initial-condition regime, especially `r0_over_target`. Stronger as a descriptive hypothesis.
- H2: Near-crossing and over-conservative-transfer failures have different closest-approach timing signatures. Stronger as an analysis hypothesis.
- H3: Diagnostic geometry metrics alone are insufficient success criteria. Stronger because Phase36C and Phase37B both show metric movement without crossing.
- H4: Post-cross recovery and upstream crossing generation are separate problems. Stronger because Phase34 succeeds downstream while Phase36/37 fail upstream.

## Which Hypotheses Became Weaker?

- H5: Radial commitment timing alone can expand the crossing basin. Weaker after Phase37A produced zero new crossings.
- H6: Weak tangential shaping is a promising next implementation lever. Weaker after Phase37B produced zero selected crossings and preserved only `4 / 8` regression crossings.
- H7: Closest-approach improvement should justify implementation. Weaker because improvements did not produce crossings.
- H8: Coast duration is ready for implementation. Weaker/unknown because it is not directly isolated in the recorded CSV evidence.

## Which Variables Should Never Be Tested Again?

At the current evidence level, the following should not be repeated as standalone implementation searches:

- Broad radial commitment timing sweeps without a new explanatory control.
- Standalone radial magnitude sweeps using the same low/medium logic from Phase37A.
- Weak tangential shaping in the Phase37B form, because it created no selected crossings and damaged regression preservation.
- Any experiment that treats closest-approach improvement or crossing-potential movement as success without actual target-radius crossing.

Variables that should be deferred, not permanently banned:

- Coast duration, until evidence shows it is independent of radial timing.
- Angular momentum correction, until existing or future logged fields can isolate it from tangential shaping.

## Should Phase39 Implementation Be Approved?

No. Phase39 implementation should not be approved from the current evidence. The mining pass identifies descriptive signatures and diagnostic variables, but no controller variable has enough direct support to justify implementation without repeating prior failures or introducing hidden coupling.

Recommended next step: remain in Phase38 evidence analysis. If any implementation is later proposed, it must start from a registered hypothesis that identifies an existing recorded feature, a causal prediction, a regression guard, and a rejection condition.
