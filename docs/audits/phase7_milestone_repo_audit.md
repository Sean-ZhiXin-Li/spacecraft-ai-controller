# 2D Phase 7 Milestone Repository Audit

Date: 2026-04-25

Scope: repository cleanup and documentation pass at the end of the current 2D Phase 7 stage. This audit does not change environment physics, retrain PPO, run learning experiments, or expand beyond the 2D Python controller line.

## 1. Primary Files To Keep

- `README.md` - top-level status, current best 2D controller result, reading order, and key links.
- `scripts/explicit_controller_phase76_soft_hybrid.py` - current best 2D controller evaluation script.
- `analysis/phase76_soft_hybrid/phase76_summary.md` - Phase 7.6 milestone summary.
- `analysis/phase76_soft_hybrid/soft_hybrid_ranking.csv` - current milestone ranking table.
- `analysis/phase76_soft_hybrid/soft_hybrid_grid.csv` - current 270-regime result grid for all soft hybrid variants.
- `analysis/phase76_soft_hybrid/soft_hybrid_comparison.png` - current milestone comparison plot.
- `analysis/phase76_soft_hybrid/soft_hybrid_success_map.png` - current best-controller success map.
- `project_log/pl22_phase65_window_seeking.md` through `project_log/pl27_phase76_soft_hybrid.md` - concise evidence trail for Phases 6.5-7.6.
- `controller/orbit_lock_controller.py` - CAPTURE/LOCK gain definitions used by the explicit-controller family.
- `scripts/explicit_controller_phase7_pre_window_shaping.py` and `analysis/phase7_pre_window_shaping/` - Phase 7 reachability reference.
- `scripts/explicit_controller_phase75_hybrid.py` and `analysis/phase75_hybrid/` - hard-switch negative control/reference.
- `scripts/explicit_controller_phase67_adaptive_ws.py` and `analysis/phase67_adaptive_ws/` - adaptive WS reference.
- `scripts/explicit_controller_phase66_ws1_refine.py` and `analysis/phase66_ws1_refine/` - WS-1 refinement reference.
- `scripts/explicit_controller_phase65_window_seeking.py` and `analysis/phase65_window_seeking/` - original WS-1 reference.

## 2. Historical Files To Keep But Not Emphasize

- Earlier Phase 3-6 explicit-controller scripts and outputs, including `explicit_controller_phase4_regime_sweep.py`, `explicit_controller_phase5_reachability.py`, `explicit_controller_phase6_variant_search.py`, phase maps, robustness checks, and mechanism comparisons.
- Learning-transfer and imitation-learning scripts/results, including `train_behavior_cloning.py`, `eval_bc_policy.py`, `learned_phase_*`, `phase_conditioned_*`, `soft_phase_conditioned_*`, and residual explicit experiments. These are useful negative/diagnostic evidence but are not the primary current 2D milestone result.
- PPO implementation and checkpoints under `ppo_orbit/`, `models/`, and related summaries. They explain why the current 2D milestone result is explicit and phase-structured rather than learned.
- Older project logs in `project_log/`, including numbered and sprint logs. They preserve development history but should not be used as the primary reading path.
- Demo artifacts under `analysis/demo/` and earlier final-project plots. They remain useful illustrations, but Phase 7.6 is now the current quantitative 2D milestone result.
- IDE folders such as `.vscode/` and `.idea/` are present locally and ignored. Keep or remove according to local workflow; they are not part of the current milestone narrative.

## 3. Files Recommended For Deletion

Safe cleanup targets:

- `__pycache__/` directories throughout the repository.
- `*.pyc` bytecode files throughout the repository.
- `.matplotlib/` cache directories under phase analysis outputs.
- `.ipynb_checkpoints/` directories.
- `.pytest_cache/` directories.
- Empty accidental root files such as `echo`, `git`, `Send`, `set`, `ssh-keygen`, and `Test-NetConnection`.

Recommended but not automatically deleted:

- Potentially obsolete duplicate script families under both `script/` and `scripts/`. These should be reviewed manually because some may be historical evidence rather than exact duplicates.
- Large legacy model/data artifacts such as `imitation_policy_model*.joblib`, `ppo_traj.npy`, and `orekit-data.zip`. They may be useful for reproducibility, so do not delete without a separate artifact-retention decision.
- `.idea_backup/`, `checkpoints_backup/`, and local IDE/config backups. Keep if they contain needed local state; otherwise archive or remove outside the current repo organization pass.

Cleanup already performed before this organization pass:

- Removed cache directories and bytecode where accessible.
- Removed empty accidental root files: `echo`, `git`, `Send`, `set`, `ssh-keygen`, and `Test-NetConnection`.

## 4. README Issues Fixed

- Added current Phase 7.6 milestone status.
- Added the current best 2D controller: `soft_linear_3e4`.
- Added current 2D milestone result numbers: 217 / 270 successes, 217 CAPTURE entries, 8 near-misses.
- Added a current result table comparing `adaptive_soft`, `prewindow_radial_medium`, `hard_hybrid_1e4`, and `soft_linear_3e4`.
- Added links to the Phase 7.6 summary, ranking, comparison plot, and success map.
- Updated the recommended reading order to put Phase 7.6 and PL22-PL27 first.
- Clarified the core conclusion: 2D orbit insertion requires phase-structured continuous coordination, not reactive learning or static gain tuning alone.

## 5. Remaining Risks

- The current milestone result is scoped to the 2D local 270-regime grid. It should not be presented as 3D, multi-orbit, or broad orbital-mechanics generalization.
- The repository still contains many historical scripts and outputs. This is useful for traceability but can confuse readers unless README and project logs are treated as the primary navigation path.
- `.gitignore` contains broad patterns such as `*.csv`, `*.png`, and `*.npy`. Narrow exceptions were added for README-linked milestone artifacts, but future generated outputs may still need explicit tracking decisions.
- `git status` may require a safe-directory override because the sandbox user differs from the repository owner.

## 6. Suggested Current Milestone Repo Structure

```text
spacecraft_ai_project/
├── README.md
├── docs/
│   ├── project_logs_index.md
│   └── audits/
│       └── phase7_milestone_repo_audit.md
├── project_log/
│   ├── pl22_phase65_window_seeking.md
│   ├── pl23_phase66_ws1_refine.md
│   ├── pl24_phase67_adaptive_ws.md
│   ├── pl25_phase7_prewindow_shaping.md
│   ├── pl26_phase75_hard_hybrid.md
│   └── pl27_phase76_soft_hybrid.md
├── scripts/
│   ├── explicit_controller_phase65_window_seeking.py
│   ├── explicit_controller_phase66_ws1_refine.py
│   ├── explicit_controller_phase67_adaptive_ws.py
│   ├── explicit_controller_phase7_pre_window_shaping.py
│   ├── explicit_controller_phase75_hybrid.py
│   └── explicit_controller_phase76_soft_hybrid.py
├── controller/
│   └── orbit_lock_controller.py
└── analysis/
    ├── phase65_window_seeking/
    ├── phase66_ws1_refine/
    ├── phase67_adaptive_ws/
    ├── phase7_pre_window_shaping/
    ├── phase75_hybrid/
    └── phase76_soft_hybrid/
```

Do not move scripts or analysis result directories without a separate migration pass; the current cleanup keeps executable and result paths stable.
