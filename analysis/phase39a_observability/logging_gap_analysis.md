# Phase39A Logging Gap Analysis

## What Current CSVs Can Explain

Current CSVs can explain:

- whether target-radius crossing occurred;
- when first crossing occurred, if it occurred;
- whether a crossing became recoverable under simulator-defined criteria;
- whether overspeed or instability occurred;
- which benchmark case was run;
- which controller family, mode, variant, or subset setting produced the row;
- closest-approach proximity using `min_abs_radius_error_ratio`;
- closest-approach timing using `closest_approach_step`;
- crossing-potential proxy using `best_crossing_potential`;
- crossing-state quality after crossing using `crossing_vr_ratio`, `crossing_vt_error_ratio`, and `crossing_sync`;
- coarse failure class using `dominant_failure_label` or `failure_label`.

These are sufficient for the current paper claim: crossing and recoverability are distinct, Phase34 improves post-cross recoverability, and Phase36/37 tested upstream variants did not expand the crossing basin.

## What Current CSVs Cannot Explain

Current CSVs cannot adequately explain:

- whether non-crossing failures are energy-limited;
- whether non-crossing failures are angular-momentum-limited;
- whether closest-approach improvement is dynamically meaningful or only a geometric proxy;
- whether coast duration is independent of radial timing;
- whether tangential shaping failed because of magnitude, timing, phase gating, angular momentum mismatch, or regression damage;
- how control effort accumulates before closest approach;
- whether a controller phase transition happened at a scientifically interpretable state;
- how state variables evolved over time before crossing or failure;
- whether over-conservative-transfer cases are truly conservative in energy, angular momentum, phase timing, or only in radius approach.

## Why Phase38 Concluded No Phase39 Controller Variable Is Approved

Phase38 found descriptive signatures but no implementation-ready controller variable.

Evidence:

- `r0_over_target` strongly separates classes descriptively, but it is an initial condition, not a controller variable.
- `closest_approach_step` separates timing signatures, but timing evidence is not causal.
- `best_crossing_potential` and `min_abs_radius_error_ratio` are useful proxies, but Phase36C and Phase37B showed proxy movement without new crossings.
- radial commitment timing was tested directly in Phase37A and created zero new crossings.
- weak tangential shaping was tested directly in Phase37B and failed both selected-case crossing and regression preservation.
- coast duration and angular momentum correction are not isolated in recorded CSV evidence.

## Why Improved Observability Is Needed Before New Controller Implementation

New implementation without better observability would repeat the Phase37 problem: it might move a proxy metric without explaining the physics of failure or protecting regression cases.

Improved observability should make it possible to answer:

- Is failure energy-limited?
- Is failure angular-momentum-limited?
- Is closest approach happening with the wrong velocity geometry?
- Are selected non-crossing cases truly different from regression crossing cases before handoff?
- Does a proposed variable change the right physical quantity or only move a proxy?

Phase39 should therefore begin as instrumentation and observability, not controller implementation.
