# Deep Repository Bug And Consistency Audit

Date: 2026-04-25

Scope: lightweight repository audit for the current 2D Phase 7 milestone. No heavy experiments, PPO retraining, physics changes, or script refactors were performed.

## Issues

| Severity | File path | Problem | Why it matters | Recommended fix | Fixed |
|---|---|---|---|---|---|
| High | `.gitignore`, README-linked artifacts | Broad ignore rules (`*.csv`, `*.png`, `*.gif`, `analysis/*.json`) can hide result artifacts linked from README. | GitHub readers may see broken links even though files exist locally. | Add narrow unignore exceptions for README-linked milestone/demo artifacts and force-add any still-untracked linked artifacts. | Yes, narrow exceptions added; `analysis/phase75_hybrid/hybrid_ranking.csv` was force-added. |
| Medium | `README.md`, Phase 7 summaries | Some text used "final" language that could imply the whole project is permanently complete. | The user clarified that Phase 7.6 is only the current 2D milestone. | Replace with "current 2D Phase 7 milestone", "current best 2D result", or "current Phase 7 conclusion". | Yes for README, Phase 7.5, Phase 7.6, and the current repo audit. Historical logs were not globally rewritten. |
| Medium | `script/` and `scripts/` | Both directories exist and contain overlapping script names. `scripts/` is active, while `script/` appears historical. | New contributors may run stale scripts or import the wrong helper. | Keep both for now; later add a deprecation README inside `script/` or migrate with import tests. | No, documented only. |
| Medium | `controller/` and `controllers/` | `controller/` contains active code; `controllers/` is effectively empty except `__init__.py`. | Directory duplication creates import ambiguity. | Keep `controller/` as active; later remove or document `controllers/` after checking old references. | No, documented only. |
| Medium | `environment.yml`, local execution history | README recommends `conda activate spacecraft`; recent Phase 7.6 verification used an existing local `orbittools` Python because the default launcher was broken in this sandbox. | Reproducibility may be confusing if users expect the exact local env name from execution logs. | Keep `environment.yml` as the GitHub setup source; optionally add a short note that any Python 3.10 env with numpy/matplotlib is enough for Phase 7 explicit scripts. | No code change; documented here. |
| Medium | `controller/imitation_controller.py` | Default `state_scaler_V5.joblib` path is root-relative, while a scaler exists under `controller/state_scaler_V5.joblib`. | Running this controller from repo root may fail unless the scaler is also present at root or caller overrides the path. | Make default paths relative to `Path(__file__).parent` in a future targeted fix. | No, not part of Phase 7 milestone path. |
| Medium | `controller/ppo_controller.py` | Default PPO model path is `ppo_orbit/ppo_best_model.pth`; audit did not verify that file exists. | PPO examples may fail for users without checkpoints, even though PPO is not the current best result. | Document expected checkpoint setup or guard with clearer error messages. | No, documented only. |
| Low | `README.md` | Some linked files are generated artifacts and may be ignored unless explicitly tracked. | Broken links reduce trust on GitHub. | Check README links and Git tracking before pushing. | Yes for path existence; tracking audit created separately. |
| Low | `analysis/final_*` and older project logs | Historical filenames and headings contain "final". | This can confuse readers if viewed out of context. | Keep as historical records; README now labels older final summaries as older milestone summaries. | Partially; not globally renamed to preserve history and links. |
| Low | `envs/multi_orbit_env.py.bak`, `envs/multi_orbit_env.py.bak2` | Backup files are present in source directories. | They add clutter and may confuse code search. | Move to a historical/backup folder or delete after manual confirmation. | No, user requested not deleting non-cache files. |
| Low | `scripts/explicit_controller_phase75_hybrid.py`, `scripts/explicit_controller_phase76_soft_hybrid.py` | These scripts read Phase 7 reference CSVs for comparison. | If a user runs only Phase 7.5/7.6 from a fresh clone without Phase 7 artifacts, comparison sections can be incomplete. | README should tell users to keep phase result artifacts or rerun prerequisite phase scripts if comparisons are missing. | No, documented only. |

## Additional Notes

- README local links were checked and currently resolve on disk.
- Key Phase 6.5-7.6 scripts compile under the available Python 3.10 conda environment.
- The active Phase 7.6 result remains scoped to the 2D local 270-regime grid.
- No physics, PPO training, learning experiments, or 3D/multi-orbit code were modified.
