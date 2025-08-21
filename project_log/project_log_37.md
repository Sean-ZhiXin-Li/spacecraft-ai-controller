# Project Log 37 — Baseline Complex Controller Evaluation
**Date:** 2025-08-21  
**Author:** (you)

## Summary
Ran the `run_baseline_complex` suite across circular and elliptical regimes to validate controller behavior and logging. Results match the intended shape: expert controllers generally succeed with small radial errors (≈1e-4–1e-3), rewards scale with radius/energy targets, and the *wire* layer correctly switches between `elliptic_ecc` and `transfer_2phase` with the expected parameters. Random policy shows notably higher variance and more failures, especially at tighter regimes.

## Objectives
- Sanity-check success behavior of expert families (`elliptic`, `elliptic_strong`, `elliptic_ecc`, `transfer`, `transfer_2phase`) vs `random`.
- Verify reward scaling and error magnitudes across radii from `5e11` up to `1e13`, and selected elliptical (rp/ra) pairs.
- Confirm wire-time configuration messages and parameter propagation (e.g., `circ_tol`, `fire`).

## What I Ran
- Circular tasks: `circ_r_{5e11, 1e12, 2e12, 5e12, 7.5e12, 1e13}_{various}`
- Elliptical tasks: `elli_rp_*_ra_*_{various}`
- Transfer sweeps: `transfer_{r_small}_to_{r_large}_{various}` and downscales

Controllers per task:
- `expert:elliptic`, `expert:elliptic_strong`, `expert:elliptic_ecc`
- `expert:transfer`, `expert:transfer_2phase`
- `random`

## Key Observations
- **Experts succeed consistently at moderate+ radii.**  
  For `r ∈ {1e12, 2e12, 5e12, 7.5e12, 1e13}`, expert families almost always report `succ=1` with `r_err` typically in `2e-4–8e-4`.  
- **Random is noisier and fails more often.**  
  `succ=0` is common for `random` in several regimes, with `r_err` often drifting toward `1e-3` or higher.
- **Reward scaling looks correct.**  
  Circular: `-480` (1e12), `-960` (2e12), `-2400` (5e12), `-3600` (7.5e12), `-4800` (1e13).  
  Elliptical and transfer variants follow the expected step-ups (e.g., `-600`, `-675`, `-750`, …; or `-3000`, `-4500`, etc., depending on target).  
- **Wire-time routing is correct.**  
  Log lines show:
  - `using elliptic_ecc with circ_tol=0.10, fire=0.9`
  - `using transfer_2phase (timered) with circ_tol=0.12, fire=1.0`
- **Lower-radius corner (`5e11`) is the toughest.**  
  Multiple controllers (including experts) show `succ=0` there with elevated `r_err` (≥1e-3). This is consistent with tighter tolerances / higher sensitivity at that scale.

## Representative Snippets (from today’s run)
- **High radius (1e13):** experts `succ=1`, `r_err≈1.7e-4–6.2e-4`; random sometimes fails; rewards near `-6000±`.  
- **Mid-high (7.5e12 & 5e12):** experts stable; random intermittently `succ=0`; expected transfer rewards around `-4500` and `-3000`.  
- **Mid (2e12 & 1e12):** experts largely stable; random failures more frequent when angles are perturbed.  
- **Low (5e11):** elevated failures across several controllers; consider mitigation or relaxed tolerance if this regime must be robust.

## Quality/Health Checks
- **Determinism:** same task+controller pairs reproduce similar `r_err` bands and rewards.  
- **Telemetry:** controller-family tags and `wire` banners present and readable; good for downstream parsing.  
- **Error bands:** clustered within the intended 1e-4–1e-3 range for experts.

## Issues / Anomalies
- **Instability at `5e11`:** frequent `succ=0` across multiple controllers. Might be due to tighter geometric sensitivity or current tolerance settings.
- **Random success on some large-radius tasks:** acceptable (by design), but if desired we can further reduce random’s success probability.

## Decisions (Today)
- **No code changes applied now.**  
  Deferred the previously proposed noise retuning and the CLI `--noise_scale` knob.
- Kept current `circ_tol` and `fire` defaults as-is.

## Next Steps
1. (Optional, deferred) Add a global `--noise_scale` to quickly tune difficulty and spread between expert vs random.
2. (Optional, deferred) Slightly reduce expert radial perturbation or increase random noise if we want starker separation.
3. Add a CSV summarizer (already drafted previously) to quantify success rate and error distribution per controller over a whole run.
4. Investigate the `5e11` corner case: try (a) slightly relaxed `circ_tol`, (b) more iterations, or (c) better warm-start for velocity alignment.

## Action Items
- [ ] Land `script/summarize_results.py` and run it on the latest CSV to get per-controller aggregates.  
- [ ] Design a short ablation on `circ_tol` vs success at `5e11`.  
- [ ] Decide target success bands (e.g., Experts ≥95%, Random ≤40%) before any noise retuning.

## Artifacts
- Raw logs: captured from today’s full suite (see run output pasted earlier in the discussion).  
- Codebase: unchanged today.

