# Scientific Honesty Audit

## Overall Assessment

Scientific honesty is one of the repository's stronger areas. The current public narrative is mostly careful: it says the work is a simplified 2D sandbox, not real spacecraft readiness, and it distinguishes geometric crossing from recoverable insertion.

The key claims are grounded in CSV-backed results:

- Phase31-style reference in the Phase34 comparison: `8 / 24` crossings, `0 / 24` recoverable crossings.
- Phase34 `radius_priority`: `8 / 24` crossings, `8 / 24` recoverable crossings.
- Phase35 local upstream variants did not improve crossing count beyond `8 / 24`.
- Phase36A did not improve crossing count in the representative subset.

## Scoped Correctly

The repo correctly scopes the central result to the current simulator:

- It calls the environment a 2D physics-based orbital control sandbox.
- It does not claim real spacecraft deployment.
- It does not claim full operational GNC.
- It does not claim universal success across all initial conditions.
- It states that non-crossing trajectory families remain unsolved.

This is appropriate and should remain the public framing.

## Negative Results

Negative results are reported honestly:

- PPO and imitation-learning paths are not framed as secretly successful.
- Phase31 transfer families are credited for crossings but not recoverability.
- Phase35 is described as a negative structural result.
- Phase36A is framed as visualization-first, not a benchmark win.

This is scientifically valuable. The repository improves when it treats failed hypotheses as constraints on the architecture, not as embarrassment to hide.

## Terminology

Current terminology is mostly rigorous:

- `crossing` means target-radius crossing, a geometric event.
- `recoverable_crossing` means a crossing occurred and a later state entered the recoverability basin.
- `CAPTURE` and `LOCK` are simulator state-machine labels.
- `success` is increasingly scoped as a simulator-defined success label, not mission success.

The main remaining risk is generator drift. Some script-generated markdown templates still use older wording such as `Success` rather than `Simulator success label`. Current public files have been cleaned, but rerunning older scripts could reintroduce ambiguity.

## Overclaim Risks

No major current public claim appears unsupported by the inspected CSVs and markdown summaries. The remaining risks are subtle:

- "Insertion" can sound stronger than the simulator supports unless it is repeatedly scoped to the sandbox.
- "Breakthrough" language should remain avoided or limited; "architecture result" is better.
- Demo visuals should stay clearly separated from benchmark results.
- Phase36A should never be used as proof that a family works across the benchmark.

## Verdict

The repository is scientifically honest enough to present as an independent simulation-based control research project. It is not yet mature enough to present as validated astrodynamics or spacecraft autonomy software.

