# Repository Organization Recommendations

Date: 2026-04-25

Context: this is the end of the current 2D Phase 7 milestone, not the end of the long-term project. The current best 2D result remains `soft_linear_3e4` from Phase 7.6: 217 / 270 successes, 217 CAPTURE entries, and 8 near-misses.

## 1. Files/Folders Safe To Delete Now

These are generated local cache or accidental files and do not carry scientific content:

- `__pycache__/`
- `*.pyc`
- `.pytest_cache/`
- `.matplotlib/`
- `.ipynb_checkpoints/`
- empty accidental command-name files at repo root, if they reappear: `echo`, `git`, `Send`, `set`, `ssh-keygen`, `Test-NetConnection`

Safe cleanup was performed for these classes during the documentation/audit pass. New `__pycache__/` directories may reappear after `py_compile` checks and can be removed again before commit.

## 2. Keep Local But Usually Do Not Track On GitHub

These are useful locally but should be tracked only when there is a clear reproducibility reason:

- large PPO checkpoints and model weights under `ppo_orbit/checkpoints/`, `models/`, and `controller/*.pth`
- large imitation model artifacts such as `imitation_policy_model*.joblib`
- raw trajectory/data arrays such as `*.npy`, raw `.npz` traces, and generated datasets
- `analysis/runs/`
- raw training logs and rollout logs
- local IDE folders such as `.idea/`, `.idea_backup/`, `.vscode/`
- local environment/build folders such as `spacecraft_ai_project.egg-info/`, `conda_envs/`, and backup checkpoint folders

The current `.gitignore` broadly ignores CSV/PNG/NPY/log outputs. Narrow exceptions are now present for README-linked milestone artifacts.

## 3. Keep As Historical Development Path

Do not delete these automatically:

- Phase 4-7.6 explicit-controller scripts in `scripts/`
- Phase 6.5-7.6 result directories under `analysis/`
- PL22-PL27 logs in `project_log/`
- PPO failure analysis and transfer summaries
- imitation-learning negative results
- residual explicit-controller experiments
- mechanism, robustness, and phase-map diagnostics

These files explain why the current 2D Phase 7 milestone converged on explicit phase-structured control rather than PPO retraining or static gain tuning.

## 4. Duplicated Or Obsolete-Looking Files Not To Delete Automatically

- `script/` and `scripts/` both exist, and several filenames overlap. Treat `scripts/` as the active directory, but keep `script/` until a separate import/path audit confirms it is unused.
- `controller/` and `controllers/` both exist. `controller/` is active; `controllers/` currently appears mostly empty but should not be removed without checking old imports.
- `envs/multi_orbit_env.py.bak` and `.bak2` look like backups. Do not delete during this pass because the user explicitly asked to preserve development path and avoid risky deletion.
- Older `analysis/final_*` and `project_audit_*` files use historical language. They should be treated as milestone-era records, not current project status.
- Large local assets such as `orekit-data.zip` may be removable or externalizable later, but not without an artifact-retention decision.

## 5. Suggested Long-Term Repo Layout

```text
spacecraft_ai_project/
├── README.md
├── docs/
│   ├── project_logs_index.md
│   ├── milestones/
│   └── audits/
├── project_log/
│   └── historical and current milestone logs
├── controller/
│   └── active controller implementations
├── envs/
│   └── active 2D environment code
├── scripts/
│   └── active evaluation and analysis scripts
├── analysis/
│   └── curated milestone result directories
├── models/
│   └── tracked only if lightweight or explicitly needed
└── historical/
    └── optional future home for old exploratory scripts after a separate migration
```

Near-term recommendation: keep script and analysis paths stable until the Phase 7.6 result is fully staged/tagged. Then do a separate migration PR for historical folders and large artifact policy.
