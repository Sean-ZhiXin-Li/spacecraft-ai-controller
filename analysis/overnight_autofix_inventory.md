# Overnight Autofix Inventory

## Trusted Baseline

- `dt = 100`
- `max_steps = 100000`
- `r0_over_target = 1.00005`
- `thrust_scale = 10000`

## Active Presentation Path

- `README.md`
- `main.py`
- `scripts/generate_orbit_demo.py`
- `analysis/demo/orbit_demo.gif`
- `analysis/demo/orbit_demo_full.png`
- `analysis/demo/orbit_demo_trajectory.png`
- `analysis/demo/orbit_demo_summary.json`

This is the current README-facing demo path and it is aligned: `main.py` imports `scripts.generate_orbit_demo.main`, the script writes into `analysis/demo`, and the README assets exist.

## Active Validation And Learning Scripts

- `scripts/generate_orbit_demo.py`
- `scripts/orbit_lock_validation.py`
- `scripts/day21_validation.py`
- `scripts/orbit_lock_benchmark.py`
- `scripts/orbit_lock_generalization.py`
- `scripts/minimal_il_test.py`
- `scripts/phase_aware_il_test.py`
- `scripts/train_behavior_cloning.py`
- `scripts/eval_bc_policy.py`

These are the scripts most directly connected to the current explicit-controller narrative, benchmark path, and learning-transfer path.

## Current Summary Docs

- `analysis/final_project_summary.md`
- `analysis/orbit_lock_benchmark.md`
- `analysis/orbit_lock_generalization.md`
- `analysis/orbit_lock_validation.md`
- `analysis/orbit_lock_phase_controller.md`
- `analysis/day21_summary.md`
- `analysis/ppo_transfer_results.md`
- `analysis/phase_aware_il_result.md`
- `analysis/phase_controller_dataset.md`

## Current Learning-Result Assets

- `analysis/minimal_il/minimal_il_summary.json`
- `analysis/phase_aware_il/phase_aware_il_summary.json`
- `analysis/phase_aware_il_result.md`
- `analysis/phase_controller_dataset/phase_controller_dataset_balanced.npz`
- `analysis/phase_controller_dataset/phase_controller_dataset_balanced_metadata.json`
- `models/minimal_il_policy.pth`
- `models/phase_aware_il_policy.pth`

## Generated Output Areas

- `analysis/demo/`
- `analysis/figs/day21_validation/`
- `analysis/figs/orbit_lock_validation/`
- `analysis/minimal_il/`
- `analysis/phase_aware_il/`
- `models/`

These directories contain rerunnable generated assets, summaries, plots, or learned checkpoints that are produced by the current validation scripts.

## Historical Or Secondary Artifacts

- `README_reproduce.md`
- `analysis/ONE_PAGE_SUMMARY.md`
- `analysis/WEEK*_*.md`
- `analysis/NEW_WEEK_*`
- `analysis/PROJECT_LOG_2026-02-01_RESTART_DAY.md`
- `scripts/day15_*`
- `scripts/day20_*`
- `scripts/week0*`
- `scripts/pl_*`

These files document older phases of the project or side analyses. They are useful context, but they are not the current headline evidence chain described by `README.md`.

## Likely Stale Or Potentially Misleading References

- `README_reproduce.md` was presenting a Day5 ablation path as a "key result" even though the active README presents a later explicit-controller result. This was corrected during this pass by marking it historical.
- `analysis/ONE_PAGE_SUMMARY.md` remains a historical summary for the earlier action-interface ablation path and should not be read as the current project conclusion.
- `analysis/final_project_summary.md` is a manually maintained presentation summary rather than an automatically regenerated output from a single script. It was left unchanged because its narrative still matches the validated baseline and this pass avoided speculative editorial rewrites.

## Trust Notes From This Pass

- README-facing demo assets exist and were successfully regenerated.
- The active baseline constants are consistent across the priority scripts that were rerun.
- No broken Markdown links were found in the checked Markdown set during this pass.
- The main overnight trust improvement was reproducibility of the IL evaluation path and clearer disclosure around phase-oracle use.
