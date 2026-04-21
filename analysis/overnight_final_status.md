# Overnight Final Status

## What Was Checked

- Repository structure, entrypoint, and README-facing assets
- Current summary and learning-result docs
- Priority validation and demo scripts
- Baseline consistency for the active validated setup
- Markdown link integrity in the checked Markdown set

## What Was Fixed

- Corrected contradictory orbit-lock summary wording in the validation generator and regenerated the affected summaries.
- Added deterministic seeding and summary metadata to `scripts/minimal_il_test.py`.
- Added deterministic seeding and summary metadata to `scripts/phase_aware_il_test.py`.
- Strengthened phase-aware IL caveats so oracle phase usage is explicit in the generated note and summary.
- Marked `README_reproduce.md` as a historical reproduction path instead of a current headline result.

## What Was Rerun

- `scripts/generate_orbit_demo.py`
- `scripts/orbit_lock_validation.py`
- `scripts/day21_validation.py`
- `scripts/minimal_il_test.py`
- `scripts/phase_aware_il_test.py`
- `scripts/orbit_lock_validation.py` again after the summary-generator fix
- `scripts/minimal_il_test.py` again after the reproducibility fix
- `scripts/phase_aware_il_test.py` twice after the reproducibility fix to confirm stable repeated metrics

## Outputs Regenerated

- Demo assets in `analysis/demo/`
- Orbit-lock validation outputs in `analysis/figs/orbit_lock_validation/`
- Day21 validation outputs in `analysis/figs/day21_validation/`
- Minimal IL outputs in `analysis/minimal_il/` and `models/minimal_il_policy.pth`
- Phase-aware IL outputs in `analysis/phase_aware_il/`, `analysis/phase_aware_il_result.md`, and `models/phase_aware_il_policy.pth`

## What Remains Intentionally Unchanged

- Core physics meaning
- PPO training or checkpoint content
- Controller structure and reward logic
- Historical artifacts that are not on the active presentation path
- Manual final narrative summaries whose claims still match the regenerated evidence

## Current Trusted Baseline

- `dt = 100`
- `max_steps = 100000`
- `r0_over_target = 1.00005`
- `thrust_scale = 10000`

## Current Strongest Project Conclusions

- The explicit phase-structured controller remains the strongest verified controller on the current validated baseline.
- The explicit controller achieves successful single-crossing insertion on the baseline, but the repository still does not show repeated orbit-lock cycling as a general capability.
- PPO still fails to recover first crossing on the main validated baseline.
- Minimal imitation learning still fails to recover crossing on the validated baseline.
- Phase-aware imitation learning can recover crossing under oracle phase input, but this is not yet a fully autonomous learned-controller result.
