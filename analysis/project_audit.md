# Project Audit

This audit focused on low-risk fixes that improve correctness, reproducibility, and trust without changing physics, controller structure, or PPO training.

## Issues

| Issue | Severity | Why it matters | Fix applied (or why not fixed) | File(s) involved |
|---|---|---|---|---|
| `day21` summary file was missing even though later work referenced it | high | This breaks the documentation trail and makes the validation stage look unverified | Re-ran the validation script and restored `analysis/day21_summary.md`; also updated the generator so future reruns preserve the radius-error and imitation-learning subsections | [scripts/day21_validation.py](../scripts/day21_validation.py), [analysis/day21_summary.md](./day21_summary.md) |
| `orbit_lock_validation.py` still used an older baseline (`dt=50`, `thrust_scale=20000`) while the current verified benchmark/demo baseline uses `dt=100`, `thrust_scale=10000` | high | This created a baseline mismatch across scripts, summaries, and README-facing outputs | Aligned the validation script to the verified benchmark/demo baseline and regenerated the validation outputs | [scripts/orbit_lock_validation.py](../scripts/orbit_lock_validation.py), [analysis/orbit_lock_validation.md](./orbit_lock_validation.md), [analysis/orbit_lock_phase_controller.md](./orbit_lock_phase_controller.md) |
| Phase-aware imitation results did not clearly disclose that evaluation used controller phases as an oracle input | medium | Without this note, the result can be misread as a fully self-contained learned controller | Added explicit metadata to the summary JSON and note; clarified that the learned model outputs the action but receives online phase labels from an explicit-controller phase oracle | [scripts/phase_aware_il_test.py](../scripts/phase_aware_il_test.py), [analysis/phase_aware_il_result.md](./phase_aware_il_result.md), [analysis/phase_aware_il/phase_aware_il_summary.json](./phase_aware_il/phase_aware_il_summary.json) |
| Minimal IL summary did not record the exact baseline setup | medium | This made it harder to compare minimal IL and phase-aware IL on equal footing | Added `baseline_setup` metadata and regenerated the summary | [scripts/minimal_il_test.py](../scripts/minimal_il_test.py), [analysis/minimal_il/minimal_il_summary.json](./minimal_il/minimal_il_summary.json) |
| Shared PPO checkpoint loader emitted noisy `torch.load` warnings in evaluation scripts | low | Warnings clutter output and make evaluation logs look unstable even when behavior is correct | Made checkpoint loading explicit with `weights_only=False` in shared evaluation paths used by PPO checkpoints | [scripts/day20_policy_surface.py](../scripts/day20_policy_surface.py), [scripts/eval_bc_policy.py](../scripts/eval_bc_policy.py), [scripts/recover_ppo_rollout.py](../scripts/recover_ppo_rollout.py) |
| Phase-controller wording could overstate what “stabilize after crossing” means | low | The controller reaches strict success on the verified setup, but it still does not show repeated crossing cycles | Softened the phrasing to “stabilize after crossing into the target band” | [scripts/orbit_lock_validation.py](../scripts/orbit_lock_validation.py), [analysis/orbit_lock_phase_controller.md](./orbit_lock_phase_controller.md) |

## Highest-Priority Findings

1. The most important inconsistency was baseline drift across evaluation scripts. The project now has one clear validated comparison baseline for orbit-lock validation:
   - `dt = 100`
   - `max_steps = 100000`
   - `r0_over_target = 1.00005`
   - `thrust_scale = 10000`

2. The most important learning-pipeline caveat was under-documented structural help in phase-aware IL. The current result is meaningful, but it depends on explicit phase labels at evaluation time and should be described that way.

3. The documentation chain is now complete again for the Day21 validation outputs:
   - figures exist
   - summary exists
   - the summary now retains its diagnostic subsections on rerun

## What Was Intentionally Left Unchanged

- Historical experiment scripts with older baselines were left in place. They document the project’s search process, and normalizing all of them to one baseline would blur provenance.
- Physics, reward, controller structure, PPO training logic, and demo behavior were not changed.
- Old output assets that are no longer central were not deleted. They are part of the experiment record and do not currently break README or the main demo path.

## Current Trust Status

- README-linked assets and analysis files referenced in the current presentation path exist.
- The main demo path is consistent with the verified successful explicit-controller setup.
- The benchmark, generalization, transfer, minimal IL, and phase-aware IL outputs now all describe their baseline or structural assumptions more clearly.
