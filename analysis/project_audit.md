# Project Audit

## Findings

| Issue | Severity | Why it matters | Fix applied or not applied | Files involved |
| --- | --- | --- | --- | --- |
| Orbit-lock validation summaries contradicted their own metrics by saying no controller maintained tail crossings even when the explicit controller had `tail_crosses_target_radius = true` | high | This directly weakens trust in a current validation summary and blurs the distinction between single-crossing success and repeated cycling | Fixed in the generator and regenerated the affected summaries | `scripts/orbit_lock_validation.py`, `analysis/orbit_lock_validation.md`, `analysis/orbit_lock_phase_controller.md` |
| `minimal_il_test.py` trained without explicit seeds or deterministic loader state | high | Re-running the same script could silently shift losses and evaluation metrics | Fixed by seeding Python, NumPy, and PyTorch, making the DataLoader shuffle deterministic, and recording seed and device metadata | `scripts/minimal_il_test.py`, `analysis/minimal_il/minimal_il_summary.json` |
| `phase_aware_il_test.py` trained without explicit seeds or deterministic loader state | high | A current learning-result artifact had already drifted across reruns, which is a direct reproducibility problem | Fixed by adding the same deterministic seeding path and verifying stable repeated metrics after the fix | `scripts/phase_aware_il_test.py`, `analysis/phase_aware_il/phase_aware_il_summary.json`, `analysis/phase_aware_il_result.md` |
| Phase-aware IL reporting did not clearly foreground oracle phase usage at evaluation time | high | Readers could otherwise over-credit the learned policy and misread the result as fully self-contained | Fixed by strengthening the generated note and preserving explicit metadata in the JSON summary | `scripts/phase_aware_il_test.py`, `analysis/phase_aware_il/phase_aware_il_summary.json`, `analysis/phase_aware_il_result.md` |
| `README_reproduce.md` described an older Day5 ablation path like a current key-result entry | medium | This created a stale README-facing route that could confuse current project positioning | Fixed by marking it historical and pointing to the current explicit-controller evidence trail | `README_reproduce.md` |

## What Was Checked

- Repository structure and current project narrative
- `README.md`, `README_reproduce.md`, `main.py`
- Current summary docs under `analysis/`
- Priority scripts:
  - `scripts/generate_orbit_demo.py`
  - `scripts/orbit_lock_validation.py`
  - `scripts/day21_validation.py`
  - `scripts/minimal_il_test.py`
  - `scripts/phase_aware_il_test.py`
- README-facing demo assets and priority generated outputs
- Markdown links in the checked Markdown set
- Baseline constant consistency across the priority scripts

## What Was Intentionally Not Changed

- Core physics equations
- PPO training or checkpoints
- Controller logic and reward design
- Historical analysis scripts that are not part of the active presentation path
- The current validated baseline:
  - `dt = 100`
  - `max_steps = 100000`
  - `r0_over_target = 1.00005`
  - `thrust_scale = 10000`

## Residual Risks

- `analysis/final_project_summary.md` remains a manual narrative summary rather than the output of a single regeneration script. It still matches the active evidence chain, so it was left unchanged.
- Historical docs in `analysis/WEEK*`, `analysis/NEW_WEEK_*`, and `analysis/ONE_PAGE_SUMMARY.md` remain in the repository. They are useful context, but they should not be treated as the active headline result unless explicitly marked as historical.
