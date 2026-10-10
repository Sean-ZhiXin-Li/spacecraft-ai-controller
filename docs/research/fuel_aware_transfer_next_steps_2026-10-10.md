# Research Follow-Up — Robust Fuel-Aware Transfer Planning

**Decision recorded:** 2026-10-10
**Input evidence:** [Development experiment](../experiments/fuel_aware_transfer_and_pareto_dev_v0.md)
**Relation to existing roadmap:** An exploratory parallel branch; **does not supersede** frozen Phase38/Stage2 hypotheses or authorize modifications to their evaluation contracts.

## 1. What Today's Data Supports

For one Phase35-like initial state (`r0/R=0.98`, `angle=150 deg`), the tested two-burn family produced six finite-thrust replay candidates satisfying both a near-circular terminal predicate and a specified residual-propellant threshold. A 20 kg reserve case arrived at the target radius in `613.679425 days`, compared with `825.584095 days` for a no-departure-burn reference. A six-row reserve sweep produced a clear, *restricted-family* time/fuel tradeoff.

The solver finding no candidate for a 150 kg reserve is an unresolved numerical-search result, not a proof of infeasibility. These results do not constitute an improvement over the original Phase35 `8/24` crossing benchmark or a general autonomous-capture policy.

## 2. Primary Next Experiment — Corrective Guidance Under Uncertainty

**Starting candidate:** 80 kg minimum-reserve row from the current sweep (`710.50 day` crossing, `80.50 kg` remaining in deterministic replay). This leaves more propellant for correction than the fastest near-dry-mass solution.

**Question:** If execution differs slightly from the ideal predicted state, can one bounded midcourse state update and replan improve final capture reliability without exhausting the requested reserve?

Initial proposed implementation scope:

1. Start from the exact saved control proposal for the 80 kg row; keep the propulsion assumptions unchanged.
2. Perturb actual departure-state radial and tangential velocity within predeclared small bounds; optionally provide a separately marked measured-state offset to the planner.
3. Simulate the nominal open-loop departure and coast; allow one midcourse feedback checkpoint for the alternative branch.
4. Replan only the remaining transfer/capture budget using the actual simulated mass and estimated state. Do not grant a new fuel tank or reset the clock.
5. Compare terminal eccentricity, radius error, reserve margin, capture achieved, and actual time across the same initial perturbations.

No results or success rates are claimed for this proposed experiment; the perturbation range, timing and sample count must be fixed before running it.

## 3. Secondary Questions

- **Optimizer completeness:** Repeat with additional numerical initializations and alternative optimization algorithms only if justified by a concrete missed-feasibility example (especially the 150 kg row). Avoid reporting solver non-convergence as a physical impossibility theorem.
- **Model fidelity:** Add realistic third-body perturbations / ephemeris-driven propagation after the feedback experiment establishes what is being optimized. Keep model-class differences explicit.
- **Time objective:** Distinguish first crossing time from time at completed powered capture. In the present runs the difference is minutes, but they are not identical quantities.
- **Operational fuel reserve:** Avoid using the 0 kg reserve row as a real mission plan; investigate how reserve policy changes feasibility under failures and estimation errors.
- **Cross-case behavior:** Before any formal comparison to Phase35, repeat on predeclared initial conditions and restore the original benchmark's time, controller, safety and mass-model comparability—or clearly define a new benchmark.

## 4. Decision Boundary

**Keep:** The restricted two-burn development result and its six returned feasible reserve levels, with the negative 150 kg search outcome.

**Do not claim:** Global Pareto optimality, robust flight readiness, original Phase35 benchmark improvement, or validated autonomy under uncertain measurements.

**Build next:** A feedback/replanning capability that can correct deviations in the *same* fuel-constrained continuous dynamics while preserving deterministic baseline results as a fixed reference.
