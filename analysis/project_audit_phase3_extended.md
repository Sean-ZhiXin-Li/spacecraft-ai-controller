# Phase 3 Extended Project Audit

## Findings

| Issue | Severity | Why it matters | Fix applied or not applied | Files involved |
| --- | --- | --- | --- | --- |
| README did not expose the new same-scenario deep-dive diagnostics. | medium | The repository front page would otherwise omit the latest stability, timestep, robustness, and physics evidence for the current explicit controller. | Fixed. Added a `Current 2D Deep-Dive Diagnostics` section linking the generated CSV, plot, JSON, and summary artifacts. | `README.md`, `analysis/local_stability_map.csv`, `analysis/dt_sensitivity.csv`, `analysis/robustness_results.csv`, `analysis/phase_statistics.json` |
| Active explicit-controller and residual scripts duplicate baseline constants locally. | low | Repeated constants can drift across scripts over time, which would make comparisons less trustworthy. | Not changed. The active scripts were checked and currently agree on `dt=100`, `max_steps=100000`, `r0_over_target=1.00005`, and `thrust_scale=10000`. Centralizing constants would be a broader refactor than this cleanup pass should perform. | `scripts/orbit_lock_validation.py`, `scripts/residual_explicit_il_test.py`, `scripts/residual_explicit_alpha_sweep.py`, `scripts/residual_explicit_tune.py`, `scripts/residual_explicit_magnitude_only_test.py`, `scripts/explicit_controller_deep_dive.py` |
| The default shell `python` environment lacks plotting dependencies. | low | Running analysis scripts outside the project Conda environment can fail before producing outputs. | Not changed. The documented and verified path is `conda run -n spacecraft ...`; the deep-dive script was run successfully there. | `environment.yml`, `README.md` |
| Local Markdown links could have become stale after adding new analysis references. | low | Broken analysis links weaken reproducibility and reviewability. | Checked `README.md`, `README_reproduce.md`, and top-level `analysis/*.md`; no missing local targets found. | `README.md`, `README_reproduce.md`, `analysis/*.md` |
| Deep-dive output files needed existence verification. | low | The final summary depends on these artifacts being present. | Verified all required outputs exist: stability CSV/PNG, dt CSV/PNG, robustness CSV/MD, energy plot, angular momentum plot, and phase statistics JSON. | `analysis/local_stability_map.*`, `analysis/dt_sensitivity.*`, `analysis/robustness_*`, `analysis/energy_vs_time.png`, `analysis/angular_momentum.png`, `analysis/phase_statistics.json` |
| Script syntax needed validation after adding the deep-dive script. | low | Syntax errors would make the new analysis non-reproducible. | Verified with `compileall` under the `spacecraft` Conda environment. | `scripts/explicit_controller_deep_dive.py`, `scripts/residual_explicit_magnitude_only_test.py`, `main.py` |

## Checks Performed

- Read current explicit-controller, environment, and Phase 3 scripts.
- Ran the full explicit-controller deep-dive script.
- Verified generated artifacts required by the task.
- Checked local Markdown links.
- Checked active baseline constants across current explicit/residual scripts.
- Compiled the new script and main entry point.
- Reviewed generated stability, dt sensitivity, robustness, and phase-statistics outputs.

## Cleanup Applied

- Updated `README.md` to include links to the current 2D deep-dive diagnostics.

## Cleanup Deliberately Not Applied

- No environment physics changes.
- No explicit controller redesign.
- No PPO retraining.
- No broad consolidation of duplicated constants.
- No deletion of historical outputs or notes.

## Residual Risks

- The controller's validated success remains highly local to the tested baseline.
- Some historical scripts and notes use older experiment framing; current README sections now identify the active evidence trail.
- Centralized experiment configuration would improve long-term maintainability, but should be done as a focused refactor with regression checks rather than inside this analysis pass.
