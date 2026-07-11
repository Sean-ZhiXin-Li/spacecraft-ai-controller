# Repository Implementation Readiness Audit

## Status

Audit date: 2026-07-11

Source commit: `a12edda8085b10b4f635356fbe3471ec9e19489a`

Branch: `main`

Scope: implementation readiness for the recoverability-aware autonomous-control research platform and the bounded Final Veto overspeed ablation prerequisite.

This is an evidence-based repository audit, not an architecture roadmap. It does not alter controllers, physics, thresholds, historical artifacts, README content, CI, dependencies, or experiment results.

## Executive Summary

The repository is ready to freeze the Final Veto experiment contract, but it is not yet ready to implement the monitor by directly importing or copying rollout physics. The current scientific baseline is protected and internally consistent at the aggregate level. The Week 1-8 contracts are coherent, and the new manifest is machine-valid. The main implementation blocker is that the exact Phase34-37 one-step transition is nested inside phase rollout functions, while the existing shared dynamics helper has a different interface and edge behavior. A monitor that owns a copied dynamics equation could silently diverge from the rollout it is meant to supervise.

The tracked tree was clean before this task. Eleven unrelated top-level paper, submission, print, release, and research-workspace entries were untracked and were not touched. After `git fetch --prune`, local `main` remained aligned with `origin/main` at `a12edda`. The pre-change protected regression guard passed all Phase34/36/37 checks.

Repository scale and environment drift are material implementation risks. The checkout contains 1,187 tracked files totaling 501.01 MiB. The full working tree is 11.899 GiB, dominated by an ignored 10.73 GiB NPY file. Current tracked data accounts for 367.06 MiB, and no current file is managed by Git LFS even though 105 historical LFS entries exist. The active `python` is Python 3.13.9 with NumPy 2.5.0 and lacks `pytest`, `torch`, `gymnasium`, and most declared scientific dependencies. CI declares Python 3.10 and NumPy below 2.

The manifest prerequisite is complete:

- `analysis/final_veto_ablation_v0/manifest.json` freezes 8 preservation cases and 5 diagnostic stress cases.
- `scripts/check_final_veto_manifest.py` passes 31 standard-library checks.
- `Tests/test_final_veto_manifest.py` passes 16 tests through `unittest`.
- No monitor, paired runner, result row, decision log, plot, or experiment summary was created.

Finding counts:

| Severity | Count |
| --- | ---: |
| P0 | 1 |
| P1 | 8 |
| P2 | 11 |
| P3 | 6 |
| Informational | 8 |

## Audit Method And Evidence Boundary

The audit inspected the Git index and object database, local and remote branch state, tracked and ignored files, repository size, active source, Phase34-37 scripts, shared simulator modules, tests, CI, environments, historical CSV headers, Week 1-8 contracts, artifact policies, current narratives, security-sensitive loading paths, and lightweight GitHub maintenance state.

Finding language follows these rules:

- **Confirmed** means directly observed in the current checkout, command output, or connected GitHub metadata.
- **Suspicion** means the code or layout creates a plausible risk that was not executed or reproduced during this audit.
- Severity uses P0, P1, P2, P3, and Informational exactly as requested.
- Confidence is high, medium, or low.

Audit limitations:

- The `gh` executable is not installed, so `gh workflow list` and `gh run list --limit 10` could not run. The GitHub connector returned no PR-triggered workflow run or commit-status record for the current direct-push head; this does not prove that no Actions run exists.
- The active Python and the named local conda environments do not contain `pytest`. Requested pytest commands therefore stopped before collection. Static discovery and source inspection were used to characterize the full-suite boundary.
- Untracked paper, submission, print, release, and research workspaces were inventoried by Git status but not opened or modified.
- No historical experiment was rerun because the phase scripts write to protected output directories.

## Repository Inventory

| Item | Confirmed value | Evidence |
| --- | --- | --- |
| Current branch | `main` | `git branch --show-current` |
| Current commit | `a12edda8085b10b4f635356fbe3471ec9e19489a` | `git rev-parse HEAD` |
| Upstream status | `main...origin/main`, no divergence after fetch | `git fetch --prune`; `git status -sb` |
| Staged files before task | 0 | `git diff --cached --name-only` |
| Tracked modifications before task | 0 | `git status --short` |
| Untracked top-level entries before task | 11 | `git status --short` |
| Tracked files | 1,187 | `git ls-files` |
| Tracked working-tree bytes | 525,350,930 bytes, 501.01 MiB | size aggregation over `git ls-files` |
| Full working-tree size excluding `.git` | 12,776,910,398 bytes, 11.899 GiB | recursive file inventory |
| Full working-tree file count | 3,510 | recursive file inventory |
| Git pack size | 399.49 MiB | `git count-objects -vH` |
| Loose Git objects | 307 objects, 3.26 MiB | `git count-objects -vH` |
| Git integrity | no errors | `git fsck --full` |
| Current Git LFS entries | 0 | `git lfs ls-files` |
| Historical Git LFS entries | 105 | `git lfs ls-files --all` |
| `.gitattributes` | absent | root inspection |
| Symlinks, submodules, executable modes | none in index | `git ls-files --stage` |
| Nested repositories | none | recursive `.git` directory inspection |
| Case-insensitive path collisions | 0 | lowercased tracked-path grouping |

### Tracked Top-Level Concentration

| Location | Tracked files | Tracked size | Interpretation |
| --- | ---: | ---: | --- |
| `analysis/` | 577 | 86.52 MiB | Mixed protected evidence and generated analysis |
| `scripts/` | 131 | 2.65 MiB | Active and historical experiment source |
| `project_log/` | 119 | 0.36 MiB | Historical research evidence |
| `data/` | 88 | 367.06 MiB | Dominant tracked data payload |
| `ppo_orbit/` | 42 | 15.13 MiB | Learning source, checkpoints, and artifacts |
| `tools/` | 32 | 0.16 MiB | Diagnostics, plotting, replay, and publishing tools |
| `docs/` | 28 at audited HEAD | 0.52 MiB | Contracts, architecture, and plans |
| `controller/` | 21 | 0.55 MiB | Active and legacy controllers plus serialized models |
| `script/` | 12 | 0.05 MiB | Older overlapping command namespace |

### Suspicious Location Classification

| Location | Classification | Confirmed basis |
| --- | --- | --- |
| `scripts/` | active source | Contains the protected guard and current Phase34-37 experiment scripts |
| `script/` | probable duplicate | Contains older names also present under `scripts/`, including `eval_orbit.py`, `gen_tasks.py`, and `replay_worst.py` |
| `controller/` | active source | Imported by current smoke and phase scripts |
| `controllers/` | probable duplicate | Contains only `__init__.py` while active imports use `controller` |
| `Tests/` | active source | Contains the two CI smoke tests and the new manifest test |
| Root `test_day_48.py`, `test_day_49.py`, `test_day_50.py` | uncertain and requires manual review | Pytest-discoverable historical scripts; some write logs and only one defines a static `test_` function |
| `controller/test_expert_controller.py` | uncertain and requires manual review | Pytest-discoverable file executes a 10,000,000-step loop and plotting at import |
| `scripts/*_test.py` | historical evidence | Learning experiments use test-like names but train or evaluate models rather than act as unit tests |
| `analysis/phase34_post_cross_sync/` through `analysis/phase37b_weak_tangential_subset/` | active evidence | Protected public scientific baseline |
| Other tracked `analysis/` phase and figure directories | historical evidence | Retain earlier positive and negative experiment trail |
| Ignored untracked `analysis/` outputs | generated output | `.gitignore` suppresses broad CSV, NPY, PNG, and run output patterns |
| `project_log/` | historical evidence | Research decision and phase history |
| `ab/` | historical evidence | Earlier ablation evidence and local ignored run outputs |
| `data/dataset/` | historical evidence | Thirty large tracked expert CSV datasets dominate current repository size |
| Root `.joblib`, `.npy`, `.png`, and paper PDFs | historical evidence | Tracked legacy models, training artifacts, and public presentation files |
| `orekit-data.zip` | uncertain and requires manual review | Largest tracked file; no active import reference was established in the audited implementation path |
| `checkpoints_backup/`, `models/`, `logs/`, `.idea/`, `.pytest_cache/` | local-only workspace | Ignored or local generated state |
| `Research_Paper/`, `print_library/`, `print_pdf_theme_validation/`, `release_print/`, `docs/submission/`, `docs/research_workspace/` | local-only workspace | Untracked at task start and explicitly excluded from modification |
| `.agents/`, `analysis/notebooks/`, `ab/day38/replay/` | uncertain and requires manual review | Empty at audit time; no move or deletion attempted |

## Confirmed Strengths

- The tracked working tree started clean, with no staged changes and no divergence from `origin/main`.
- The protected Phase34/36/37 regression guard passes before implementation work.
- Historical negative results remain tracked and described instead of being overwritten by later narratives.
- Week 1-8 contracts consistently distinguish crossing, recoverability, simulator success, safety, diagnostic subsets, and non-claims.
- Phase34-37 import shared constants and recoverability helpers from earlier phase modules instead of redefining every constant independently.
- CI already invokes the two bounded smoke tests, controller import/action smoke, and protected historical guard.
- Git object integrity is clean; there are no nested repositories, symlinks, submodules, executable-mode anomalies, or case-insensitive path collisions.
- No tracked credential, private-key, API-token, email-address, or private-lab-evidence pattern was confirmed.
- The Final Veto plan predeclares hazard, threshold, fallback, preservation set, diagnostic stress set, counterfactual pairing, and prohibited claims.
- The new manifest validator uses only the standard library and exercises negative mutations without changing the real manifest.

## P0 Findings

### P0-1 - No importable exact rollout predictor boundary

- **Confidence:** high
- **Fact:** confirmed
- **Evidence:** Exact scalar `env_step` implementations are nested inside `rollout_phase34_case` at `scripts/explicit_controller_phase34_post_cross_sync.py:294`, `rollout_phase35_case` at `scripts/explicit_controller_phase35_crossing_basin_expansion.py:349`, `rollout_family_case` at `scripts/explicit_controller_phase36b_transfer_family_benchmark.py:237`, `rollout_variant_case` at `scripts/explicit_controller_phase37a_radial_commit_timing.py:262`, and `rollout_case` at `scripts/explicit_controller_phase37b_weak_tangential_subset.py:244`.
- **Additional evidence:** `simulator/physics.py:24` exposes a pure `step_dynamics`, but it accepts thrust rather than normalized action, does not own the phase action clamp, and has a different radius singularity rule at line 38. `envs/orbit_env.py:429` adds optional action smoothing and capture assist, so it is not the Phase35 rollout contract.
- **Consequence:** A monitor cannot currently import the exact transition used by the future stress rollout. Copying the equation into the monitor would create two physics implementations and could turn predictor drift into false veto or false-negative evidence.
- **Affected dimensions:** scientific validity, reproducibility, maintenance, Final Veto correctness.
- **Bounded remediation:** Make the monitor accept a pure predictor callback. In the next issue, expose one exact action-to-next-state predictor at the runner boundary and prove one-step equality against the actual rollout over nominal, clamped, zero-action, and near-threshold cases. Do not put gravitational or integration equations in the monitor.
- **Regression checks:** protected guard; one-step equality tests; action-clamp edge cases; threshold comparator tests; no historical artifact changes.
- **Blast radius:** medium.
- **Dedicated branch/PR:** yes.

## P1 Findings

| ID | Confidence | Confirmed evidence | Consequence and affected dimensions | Bounded remediation |
| --- | --- | --- | --- | --- |
| P1-1 | high | At least 34 nested `env_step` definitions exist under `scripts/`; Phase34-37 repeat the same transition and benchmark grid. Crossing, success, and termination blocks are also repeated. | Silent scientific drift across phases; scientific validity and maintenance. | Extract only a tested one-step predictor seam first; defer broad rollout modularization. |
| P1-2 | high | Phase writers call `OUTPUT_DIR.mkdir` and open fixed historical result paths for writing, for example Phase34 lines 730-735 and Phase35 lines 817-824. | An accidental rerun can overwrite protected CSVs and summaries; artifact integrity and reproducibility. | Add refusal guards to future runners and require fresh output directories; do not edit historical runners in the monitor issue. |
| P1-3 | high | `scripts/check_phase_results.py` checks aggregate row counts and booleans only. It does not hash protected files, validate headers, constants, ordering, or row identity. | Changed rows can retain aggregate counts and still pass; scientific validity and artifact integrity. | Add a separate future protected-artifact checksum/schema manifest without weakening current checks. |
| P1-4 | high | Active `python` is 3.13.9 with NumPy 2.5.0 and lacks pytest, torch, gymnasium, SciPy, and most declared packages. CI declares Python 3.10 and `numpy<2`. | Requested local validation cannot run in the active shell; reproducibility and CI parity. | Establish one documented implementation environment and preflight command; do not upgrade dependencies in this change. |
| P1-5 | high | Nineteen files match default pytest naming, but static inspection finds only three `test_` functions. `controller/test_expert_controller.py` runs ten million steps and plotting at import; several `scripts/*_test.py` files train models. | A full pytest run is unsafe, slow, dependency-heavy, and difficult to interpret; CI and maintenance. | Add pytest collection boundaries or rename historical experiment scripts in a dedicated cleanup after preserving history. |
| P1-6 | high | `.gitignore:261` ignores all CSV files and line 264 ignores all PNG files. Future `results.csv`, `paired_results.csv`, and `comparison.png` are currently ignored. | A completed ablation can omit primary evidence from Git without warning; artifact integrity and public presentation. | Before running the ablation, add narrow negations for only the declared Final Veto outputs or enforce an explicit publication step. |
| P1-7 | high | No Python source or current artifact contains `terminal_label`, `accepted_as_progress`, `decision_schema_version`, or `veto_status`. | Result Schema v1, taxonomy priority, progress acceptance, and decision logging remain unenforced contracts; scientific validity and auditability. | Implement only the narrow Result Schema and Decision Log fields required by the paired runner, plus validation. |
| P1-8 | high | Tracked content is 501.01 MiB; `data/` is 367.06 MiB; the Git pack is 399.49 MiB; current LFS entries are zero. | Clone, CI, review, and archival costs are high; reproducibility and maintenance. | Inventory dataset necessity and plan a non-destructive LFS/Release/archive migration on a separate branch; do not rewrite history casually. |

## P2 Findings

| ID | Confidence | Confirmed evidence | Consequence and affected dimensions | Bounded remediation |
| --- | --- | --- | --- | --- |
| P2-1 | high | Phase34 line 569 and Phase35 line 681 allow `success` to terminate directly, while Phase36B line 589, Phase37A line 559, and Phase37B line 551 require a first crossing. | Legacy success semantics are not fully uniform; scientific interpretation. | Preserve legacy fields and make new `final_simulator_success` plus crossing/recoverability fields explicit. |
| P2-2 | high | `SAFE_SPEED_RATIO = 1.65` at `scripts/explicit_controller_phase22_two_burn_transfer.py:80` is an internal guard, while realized termination uses `> 1.90` in every current phase and `envs/orbit_env.py:524`. | Future code can confuse advisory action screening with the realized hazard boundary; scientific validity. | Keep both names explicit and test that the manifest hazard remains the realized `> 1.90` signal. |
| P2-3 | high | Phase21, Phase22, and Phase34-37 create `.matplotlib` output directories at import before importing Matplotlib. Imports cascade from Phase35 to Phase34 and earlier phases. | Importing a predictor source mutates the filesystem and loads plotting dependencies; maintenance and artifact hygiene. | Put future predictor and monitor interfaces in import-pure modules; do not import phase scripts from the monitor. |
| P2-4 | high | Both `script/` and `scripts/` are tracked; `controller/` and `controllers/` coexist; tests exist under `Tests/`, root, controller, and scripts. | Ambiguous ownership and discovery; maintenance and CI. | Publish a directory ownership map, then consolidate only after compatibility checks. |
| P2-5 | high | `README.md:228` says current status is June 2026 and `README.md:266`, `docs/milestones/README.md:15`, and `docs/research_direction.md:104` still point to Phase38 rather than the Final Veto prerequisite. | Public narrative does not reflect the completed Week 1-8 transition; public presentation. | Open one bounded README/current-status refresh after the manifest commit; do not mix it with monitor code. |
| P2-6 | high | `pyproject.toml` declares only NumPy and Gymnasium; `environment.yml` and `conda_envs/spacecraft_linux.yml` differ; active tools import PDF/Markdown packages absent from both. | Installation method determines available behavior; reproducibility and maintenance. | Define core, experiment, and publishing dependency groups after inventory; retain CI's baseline until tested. |
| P2-7 | high | CI runs two smoke tests, controller smoke, and the historical guard only. It has no manifest/schema validation, lint, type check, or safe full-suite boundary. Action references use major tags rather than commit SHAs. | New contracts can regress without CI detection; CI and supply-chain hygiene. | Smallest next CI addition: run `python scripts/check_final_veto_manifest.py` and the manifest test after pytest availability is confirmed. |
| P2-8 | high | `.gitattributes` is absent, `core.autocrlf=true`, 28 tracked files have mixed working-tree endings, and 395 index-LF files are CRLF locally. | Cross-platform diffs and generated-file churn are likely; maintenance and review quality. | Add a narrow line-ending policy in a dedicated normalization change, never mixed with scientific edits. |
| P2-9 | high | Multiple paths use `torch.load` without `weights_only=True`, `joblib.load`, or `np.load(..., allow_pickle=True)`, including `controller/ppo_controller.py:39` and `tools/diagnostics/diag_orbit.py:35`. | Untrusted model or NumPy artifacts can execute code; security and supply chain. | Document trusted-artifact-only loading and migrate supported loads to safer modes where compatible. |
| P2-10 | high | Historical CSVs lack commit SHA, environment, stable cross-phase case IDs, normalized schema version, and deterministic seed fields. Phase scripts write fixed directories. | Exact rerun provenance and row-level comparison are incomplete; reproducibility. | Require these fields for new artifacts only; do not migrate historical CSVs in place. |
| P2-11 | medium | `Tests/conftest.py` and `Tests/compat_shims.py` patch environment APIs at import and contain many broad exception handlers. | Smoke tests can pass through compatibility behavior that production code does not expose; CI confidence. | Add direct API assertions beside shimmed smoke tests before removing any compatibility layer. |

## P3 Findings

| ID | Confidence | Confirmed evidence | Consequence and affected dimensions | Bounded remediation |
| --- | --- | --- | --- | --- |
| P3-1 | high | Markdown-link scan found 11 missing historical image links in `project_log/project_log_9.md`, `project_log/project_log_18.md`, `project_log/project_log_33.md`, and `project_log/sprint_ppo21.md`. | Broken historical navigation; public presentation. | Repair links only when source artifacts are confirmed; otherwise label them unavailable. |
| P3-2 | high | Tracked names include `controller/muti_orbit_expert_controller.py`, `project_log/NEW_WEEK_PROJECT_LOG_0 .md`, multiple files with spaces, and mixed phase numbering such as `phase20_5`. | Search and cross-platform usability friction; maintenance. | Create a rename map and compatibility-import plan before any rename. |
| P3-3 | medium | `.agents/`, `analysis/notebooks/`, and `ab/day38/replay/` were empty at audit time. | Minor directory noise; maintenance. | Manual review only; remove later only with explicit approval. |
| P3-4 | high | `.github/` contains only the workflow; no CONTRIBUTING, CODEOWNERS, issue template, or PR template was found. | Lightweight traceability guidance is absent; maintenance. | Add one concise contribution/PR checklist if collaboration expands. |
| P3-5 | high | Multiple older roadmap and logging documents remain active-looking without a superseded marker even though Week 8 closes the documentation phase. | Readers can select conflicting next steps; public presentation. | Add status banners in a bounded documentation-only change. |
| P3-6 | high | Eleven unrelated untracked top-level workspaces/files make status output broad, including paper, print, submission, release, and duplicate PDF material. | Accidental staging risk; repository hygiene. | Keep them untouched; decide track/archive/ignore policy in a separate manual cleanup. |

## Informational Findings

| ID | Confidence | Confirmed fact |
| --- | --- | --- |
| I-1 | high | `main` and `origin/main` both point to `a12edda`; no staged or tracked modifications existed before the task. |
| I-2 | high | The protected guard passes all required Phase34, Phase36B, Phase36C, Phase37A, and Phase37B facts. |
| I-3 | high | `git fsck --full` is clean; no nested repository, symlink, submodule, executable-mode anomaly, or case-insensitive collision was found. |
| I-4 | high | No confirmed tracked secret, private key, email address, or private lab evidence was found. One token-assignment pattern in `tools/replay_player.py:57` is a function-call result, not a literal credential. |
| I-5 | high | Git LFS is installed locally. There are 105 historical LFS entries but zero current LFS-managed paths and no current `.gitattributes`. |
| I-6 | high | Connected GitHub metadata shows a public repository, no open issues, five closed issues, two merged PRs, four remote branches, and two local tags. |
| I-7 | high | CI statically includes Linux smoke tests, the controller smoke command, and the protected result guard. |
| I-8 | high | Phase34, Phase35, Phase36B, and Phase37A iterate cases deterministically in thrust-angle-radius order. The frozen manifest records seed `0` even for deterministic controller contexts. |

## Top Ten Repository Risks

| Rank | Finding | Why it matters now |
| ---: | --- | --- |
| 1 | P0-1: no exact pure predictor boundary | Blocks scientifically defensible monitor implementation |
| 2 | P1-1: duplicated scientific logic | A small edit can change the monitor and rollout differently |
| 3 | P1-2: protected runners overwrite fixed paths | A validation rerun can destroy the evidence being protected |
| 4 | P1-3: aggregate-only regression guard | Row or constant drift can pass while counts remain unchanged |
| 5 | P1-4: unusable active validation environment | Required tests cannot run under the active Python |
| 6 | P1-5: unsafe pytest discovery | Full test invocation can import training or ten-million-step scripts |
| 7 | P1-6: Final Veto CSV/PNG outputs ignored | Primary future evidence can be absent from commits |
| 8 | P1-7: schemas are not code-enforced | Claims and decision evidence can drift from contracts |
| 9 | P1-8: 501 MiB tracked repository | Clone, CI, and review costs impede repeatability |
| 10 | P2-1: inconsistent legacy success semantics | Cross-phase `success` values cannot be treated as one normalized field |

## Scientific Drift Risk Register

| Risk | Duplicated locations | Protected behavior | Detection currently available | Missing protection | Severity |
| --- | --- | --- | --- | --- | --- |
| One-step integration | Nested `env_step` in Phase21, Phase22, Phase34, Phase35, Phase36B, Phase37A, Phase37B, and many earlier scripts | Semi-implicit velocity-then-position update | Aggregate historical counts only | One-step equality test and single callable boundary | P0 |
| Gravity and thrust acceleration | Nested phase steps; `simulator/physics.py`; `envs/orbit_env.py` | `-MU*r/r^3 + thrust_scale*action/MASS` for protected phases | No constant/formula check | Predictor/rollout equality and constant manifest | P1 |
| Action clamp | Each nested step clamps components to `[-1,1]`; `envs/orbit_env.py:439` also clamps and may smooth | Protected phase action execution | No edge-case test | Clamp equality test and nominal/executed action log | P1 |
| Mass | `MASS=722.0` in Phase21; `envs/orbit_env.py:36` uses 722.0; `envs/orbit_presets.py:25` uses 721.9 | Protected phase value 722.0 | Aggregate counts | Explicit experiment constant block | P2 |
| Time step | `DT=100.0` in Phase21; simulator defaults differ (`simulator/config.py`, `envs/orbit_env.py:37`) | Protected phase `DT=100.0` | No explicit guard | Manifest/runner constant hash | P1 |
| Target radius | `DEFAULT_TARGET_RADIUS=7.5e12` in Phase21 and defaults elsewhere | Protected 2D benchmark target | No constant check | Frozen run config and recorded units | P1 |
| Circular-speed normalization | Recomputed as `sqrt(MU/target_radius)` in phase rollouts and environment | Overspeed ratio and recoverability ratios | Output max ratio only | Unit test against runner state and manifest normalization | P1 |
| Crossing detection | Repeated sign-change tests on previous/current radius error in Phase34-37 | Eight protected crossings | Aggregate crossing count | First-step identity and edge-condition tests | P1 |
| Recoverability criteria | Defined in Phase34 lines 167-188 from Phase21 tolerances and imported downstream | Eight protected recoverable crossings | Aggregate recoverable count | Direct boundary tests and versioned criteria ID | P1 |
| Post-cross synchronization | `MODES` in Phase34 and imported downstream | `radius_priority` behavior | Protected counts | Controller parameter fingerprint and fresh preservation rerun | P1 |
| CAPTURE/LOCK | `OrbitLockConfig` plus copied phase state transitions | Simulator-specific state-machine behavior | Some legacy fields | Transition tests and normalized logging | P2 |
| Overspeed | Internal `SAFE_SPEED_RATIO=1.65`; realized `>1.90` repeated in phases and environment | Zero protected Phase36/37 overspeed; Week 7 hazard | Aggregate booleans | Named constants, comparator test, predictor equality | P1 |
| Instability | Out-range, too-close, and radial-stall logic repeated; labels vary | Zero protected Phase36/37 instability | Aggregate booleans | Shared mechanism definitions and terminal priority tests | P1 |
| Termination priority | Success, out-range, overspeed, too-close, stall order copied; crossing requirement differs | Historical outcome labels | Aggregate counts | Explicit priority test and taxonomy mapping | P1 |
| Case grid and ordering | Three lists repeated in Phase34, Phase35, Phase36B, Phase37A | 24-case membership and counts | Row counts, not exact order | Machine-readable benchmark manifest and stable IDs | P1 |
| Seeds | Deterministic phases do not record row seeds; older stochastic paths seed inconsistently | Current explicit rollouts appear deterministic | No row-level seed check | Required seed field and run manifest | P2 |
| Success persistence | `STRICT_CFG` in Phase21 uses 200 steps; `envs/orbit_env.py` defaults to 40 | Protected phase success semantics | No threshold assertion | Explicit simulator profile ID and threshold test | P2 |

## Final Veto Physics-Consistency Audit

| Question | Answer |
| --- | --- |
| Exact state representation | Protected phase rollouts use scalar `(x, y, vx, vy)` in SI-like simulator units. Historical environment observations may include additional radial velocity. |
| Exact action representation | Two normalized Cartesian components `(action_x, action_y)`, component-clamped to `[-1, 1]`. |
| Action clamp source | Local nested `env_step` via imported `clamp`; no standalone protected transition function. |
| One-step integration order | Compute gravity and thrust acceleration from current position; update velocity; update position with the new velocity. |
| Gravitational acceleration | `-MU * position / (radius^3 + 1e-12)`. |
| Thrust acceleration | `thrust_scale * clamped_action / MASS`. |
| Target circular speed | `sqrt(MU / target_radius)`. |
| Speed ratio | Next-state speed divided by target circular speed. |
| Realized overspeed | Executed next-state speed ratio `> 1.90`, not `>= 1.90`. |
| Zero-action fallback | Execute normalized action `[0.0, 0.0]` for one step, then reevaluate; it is not proven safe. |
| Can rollout dynamics be imported safely? | No. The exact functions are nested, and importing phase modules has filesystem and plotting side effects. |
| Does importing Phase35 have side effects? | Yes. Phase35 creates its `.matplotlib` directory at import and transitively imports Phase34 and earlier phase modules that do the same. Main experiment execution remains guarded, but import is not pure. |
| Is a pure shared dynamics function present? | `simulator.physics.step_dynamics` is pure and uses the same broad integration order, but its thrust interface, clamp ownership, state shape, and singularity handling are not the protected Phase35 contract. |
| Would copied monitor physics drift? | Yes. It would create an unprotected duplicate and is prohibited by the recommended boundary. |
| Smallest safe boundary | Monitor accepts `predict_next_state(state, nominal_action, context)` as a callback and owns only threshold comparison and allow/veto output. The runner owns the exact predictor adapter and executed transition. |
| Required equality proof | Parameterized unit tests must compare predictor and rollout one-step state at exact tolerance for nominal, clamped, zero-action, and near-threshold cases using the same constants and integration order. |

The manifest freezes this interface requirement but does not implement it.

## Test And CI Coverage Matrix

| Check | Environment | Result | Count/runtime | Boundary |
| --- | --- | --- | --- | --- |
| `python scripts/check_phase_results.py` before changes | Active Python 3.13.9 | PASS | All protected checks; about 4 s | Aggregate historical facts only |
| `python -m pytest -q Tests/test_env_smoke.py Tests/test_quickrun_smoke.py` | Active Python 3.13.9 | NOT RUN | 0 collected; about 4 s | `pytest` missing |
| `python -m scripts.test_all_controllers` | Active Python 3.13.9 | NOT RUN | Import failure under 1 s | `torch` missing |
| `E:\conda3\envs\spacecraft\python.exe -m scripts.test_all_controllers` | Python 3.12.12 environment | PASS | Expert and PPO actions finite; about 13.4 s | Supplemental, not the requested active `python` |
| `python -m pytest -q` | Active Python 3.13.9 | NOT RUN | 0 collected; about 4 s | `pytest` missing, so unsafe collection boundary was not reached |
| Static pytest discovery | Source inspection | 19 discoverable files | 3 static `test_` functions | Many modules are experiments or import-side-effect scripts |
| `python scripts/check_final_veto_manifest.py` | Active Python 3.13.9 | PASS | 31 checks; under 1 s | Manifest only, no result validation |
| `python -m unittest -q Tests.test_final_veto_manifest` | Active Python 3.13.9 | PASS | 16 tests in 0.205 s | Standard-library fallback because pytest is absent |
| CI workflow | Ubuntu, declared Python 3.10 | Statically configured | Two smoke tests, controller smoke, historical guard | No local `gh` run history available |

Scientifically important behavior with no direct unit test:

- predictor/rollout one-step equality;
- exact action-clamp equivalence;
- crossing detector edge cases;
- recoverability boundary values;
- termination priority and `>` comparator behavior;
- Phase34 post-cross parameter fingerprint;
- protected artifact byte identity;
- Result Schema and Failure Label Taxonomy enforcement;
- monitor pair completeness and counterfactual metrics;
- output-directory refusal for protected paths.

### CI Assessment

The workflow triggers on pushes and pull requests to `main`, uses `actions/checkout@v4`, `mamba-org/setup-micromamba@v1`, and `conda_envs/spacecraft_linux.yml`, then runs the intended smoke, controller, and historical guard commands. This is a useful bounded baseline.

Missing CI coverage includes manifest validation, result/schema validation, linting, formatting, type checking, artifact path protection, full-suite collection boundaries, and dependency-lock verification. Failures should be reasonably diagnosable because commands are separate comments within one run step, but separate named steps would localize failures better.

The smallest next CI addition is the manifest validator and manifest unit test, after ensuring pytest is present in the actual CI environment. CI workflow modification was not allowed today.

## Environment Consistency Matrix

| Environment source | Python | NumPy | Key packages | Assessment |
| --- | --- | --- | --- | --- |
| Active `python`, `E:\conda3\python.exe` | 3.13.9 | 2.5.0 | pandas 3.0.3, matplotlib 3.11.0; pytest/torch/gymnasium/SciPy absent | Not safe as the implementation environment; outside declared Python and NumPy ranges |
| Local `spacecraft` env | 3.12.12 | 2.3.5 | torch 2.9.1, gymnasium 1.2.2, SciPy 1.16.3; pytest absent | Can run controller smoke but does not match CI pins |
| Local `orbittools` env | 3.10.6 | 1.23.4 | SciPy 1.9.3, matplotlib 3.6.2; torch/gymnasium/pytest absent | Useful for older phase scripts, not complete for repository tests |
| `environment.yml` | 3.10 | `<2` | pandas, matplotlib, CasADi, OSQP, do-mpc, Gymnasium, SPICE, torch 2.2 pip | Missing pytest, SciPy, scikit-learn, joblib, PyYAML, Pillow, tqdm used elsewhere |
| `conda_envs/spacecraft_linux.yml` and CI | 3.10 | `<2` | Broad scientific stack, pytest, PyTorch 2.2 CPU | Best declared baseline; mostly unpinned transitive versions |
| `pyproject.toml` | `>=3.10` | unpinned | NumPy and Gymnasium only | Insufficient to run active controllers, phase scripts, tests, or publishing tools |

Python 3.13 should not be treated as supported merely because the standard-library guard and manifest validator run. Current declared PyTorch 2.2 and NumPy below 2 target a Python 3.10 baseline. Compatibility must be established in a complete environment, not inferred from partial imports.

## Artifact And Schema Governance Findings

### Legacy Field Mapping

| Result Schema v1 concept | Clean legacy mapping | Conflict or gap |
| --- | --- | --- |
| `r0_over_target`, angle, thrust | Present in Phase34-37 | Stable cross-phase `case_id` absent except Phase37B's index-based IDs |
| `crossed_target_radius` | `crossing_occurs` | Name differs; Phase37B omits first crossing step |
| `first_crossing_step` | `crossing_step` in Phase34/35; `first_crossing_step` in Phase36B/37A | Inconsistent names; absent from Phase37B output |
| `recoverable_crossing` | Present | Recoverable-state details and time are not normalized |
| `final_simulator_success` | `success` or `simulator_success_label` | Legacy success termination semantics differ by phase |
| `overspeed`, `instability` | Present in Phase35-37; overspeed in Phase34 | Phase34 lacks normalized instability |
| `max_speed` | `max_speed_ratio` | Schema name does not state whether value is speed or ratio |
| `terminal_label` | `termination_reason`, `dominant_failure_label`, or `failure_label` | Controlled taxonomy priority is not enforced and legacy labels differ |
| `diagnostic_labels` | Phase36C labels and legacy failure labels | No normalized list representation |
| subset/regression metadata | Phase37A baseline flag and Phase37B `group` | No common fields across phases |
| `accepted_as_progress` | none | Exists only in documentation |
| Decision Log v0 fields | none | No decision logs are implemented |

### Output Tracking

| Future artifact | Current Git ignore state | Manifest record |
| --- | --- | --- |
| `analysis/final_veto_ablation_v0/manifest.json` | trackable | current prerequisite |
| `results.csv` | ignored by `.gitignore:261` | `currently_ignored_by_gitignore=true` |
| `paired_results.csv` | ignored by `.gitignore:261` | `currently_ignored_by_gitignore=true` |
| `decision_log.jsonl` | trackable | `currently_ignored_by_gitignore=false` |
| `summary.md` | trackable | `currently_ignored_by_gitignore=false` |
| `comparison.png` | ignored by `.gitignore:264` | `currently_ignored_by_gitignore=true` |

The new manifest and validator ensure future paths remain under `analysis/final_veto_ablation_v0/` and do not overlap protected directories. They do not change `.gitignore` or guarantee that future ignored outputs are published.

## Documentation Consistency Findings

Week 1-8 milestone metadata is consistent with actual completion dates. The benchmark, taxonomy, result schema, regression policy, decision-log schema, cross-embodiment note, Final Veto plan, and transition report agree on the protected evidence and evidence limits.

The main inconsistency is the public/current narrative outside those contracts:

- `README.md:228` still labels the current status as June 2026.
- `README.md:258-270` points to Phase38 analysis as the next step.
- `docs/milestones/README.md:7-15` calls Phase37B/Phase38 the current milestone.
- `docs/research_direction.md:104` and line 141 retain a Phase38-before-code next direction.
- `docs/modularization_plan.md` still recommends planning Phase37B after Phase37A.

These older documents are not scientifically false about Phase34-37, but their current/next labels are superseded by the Week 8 transition report. The README also remains spacecraft-centered and does not state the newer cross-embodiment identity. It does not overclaim implementation; the gap is currency and framing.

A separate bounded README/current-status issue is recommended. It should update status and links without changing protected scientific counts or expanding cross-domain claims.

## Large-File And Repository-Size Findings

Tracked size is 501.01 MiB, while the checkout excluding `.git` is 11.899 GiB. The ignored file `data/data/preprocessed/merged_expert_dataset.npy` alone is approximately 10.73 GiB. Three ignored imitation trajectory NPY files are approximately 122.07 MiB each. The tracked repository is dominated by thirty expert dataset CSVs and the 19.756 MiB `orekit-data.zip`.

The 30 largest tracked files are:

| Rank | Path | MiB |
| ---: | --- | ---: |
| 1 | `orekit-data.zip` | 19.756 |
| 2 | `data/dataset/expert_dataset_01.csv` | 8.749 |
| 3 | `data/dataset/expert_dataset_02.csv` | 8.700 |
| 4 | `data/dataset/expert_dataset_03.csv` | 8.697 |
| 5 | `data/dataset/expert_dataset_05.csv` | 8.696 |
| 6 | `data/dataset/expert_dataset_04.csv` | 8.696 |
| 7 | `data/dataset/expert_dataset_06.csv` | 8.695 |
| 8 | `data/dataset/expert_dataset_27.csv` | 8.695 |
| 9 | `data/dataset/expert_dataset_07.csv` | 8.695 |
| 10 | `data/dataset/expert_dataset_28.csv` | 8.695 |
| 11 | `data/dataset/expert_dataset_29.csv` | 8.695 |
| 12 | `data/dataset/expert_dataset_26.csv` | 8.695 |
| 13 | `data/dataset/expert_dataset_08.csv` | 8.695 |
| 14 | `data/dataset/expert_dataset_30.csv` | 8.695 |
| 15 | `data/dataset/expert_dataset_18.csv` | 8.694 |
| 16 | `data/dataset/expert_dataset_23.csv` | 8.694 |
| 17 | `data/dataset/expert_dataset_24.csv` | 8.694 |
| 18 | `data/dataset/expert_dataset_22.csv` | 8.694 |
| 19 | `data/dataset/expert_dataset_25.csv` | 8.694 |
| 20 | `data/dataset/expert_dataset_21.csv` | 8.694 |
| 21 | `data/dataset/expert_dataset_19.csv` | 8.694 |
| 22 | `data/dataset/expert_dataset_20.csv` | 8.694 |
| 23 | `data/dataset/expert_dataset_16.csv` | 8.694 |
| 24 | `data/dataset/expert_dataset_11.csv` | 8.694 |
| 25 | `data/dataset/expert_dataset_12.csv` | 8.694 |
| 26 | `data/dataset/expert_dataset_10.csv` | 8.694 |
| 27 | `data/dataset/expert_dataset_17.csv` | 8.694 |
| 28 | `data/dataset/expert_dataset_09.csv` | 8.694 |
| 29 | `data/dataset/expert_dataset_14.csv` | 8.694 |
| 30 | `data/dataset/expert_dataset_15.csv` | 8.694 |

The largest Git objects also show historical path duplication such as `data/data/dataset/` and earlier root-level dataset paths. A future migration should first determine which datasets are required for published reproducibility. Git LFS, Releases, external storage with checksums, or an archive branch may be appropriate. No migration or history rewrite was performed.

## Security And Privacy Findings

- No confirmed tracked API key, token literal, private key, password, email address, or private lab artifact was found.
- The repository is public according to connected GitHub metadata, so tracked serialized models and PDFs should be treated as publicly downloadable.
- `torch.load` without a safe weights-only mode, `joblib.load`, pickle-compatible NumPy loads, and explicit `weights_only=False` paths must accept only trusted repository artifacts.
- No dependency lock is present; conda channels and most packages float within broad constraints. CI action references use major tags.
- Several historical scripts delete specific old plots or output files. No broad repository-destructive command was found in the active Final Veto path, but future runners should never accept protected output directories.
- No untracked paper, submission, or lab workspace was inspected for content, and no private material was copied into the new files.

## Reproducibility Findings

Confirmed positive properties:

- Phase34-37A use deterministic case ordering.
- The explicit phase rollouts inspected do not depend on stochastic policy sampling for the protected cases.
- The protected guard can re-read historical CSVs without importing controller or plotting code.
- The manifest freezes source commit, repository, case IDs, seed fields, threshold, comparator, pairing, and output locations.

Remaining gaps:

- Historical rows do not record commit SHA or environment.
- Multiple environment definitions compete, and no lock file records resolved versions.
- Phase runners write fixed historical output paths and do not isolate reruns.
- Plot versions and rendering backends are not recorded in historical artifacts.
- Units are embedded in code and prose rather than a machine-readable simulator profile.
- Missing optional fields were not normalized before Result Schema v1.
- Historical results cannot be safely regenerated by invoking current phase `main()` functions without risking overwrite.

## Issue And Maintenance Workflow Audit

Connected GitHub metadata reports:

- no open issues;
- five closed issues, including the Phase37B guard, benchmark manifest, Phase38 search-space, postmortem, and Linux smoke-test work;
- two merged pull requests, both from February 2026;
- remote branches `main`, `before-action-fix`, `whpl09-radial-pd`, and `whpl10-gated`;
- tags `v1.0-phase34` and `v2.0-phase34-post-cross-sync`.

Recent documentation changes were committed directly on `main`. No local CONTRIBUTING guide, issue template, PR template, or CODEOWNERS file exists. Branch protection could not be verified with available tools. For a solo research repository, the lightweight recommendation is one branch and PR for each implementation milestone that can affect physics, controllers, schemas, or protected evidence. Enterprise process is not required.

## Prioritized Remediation Backlog

| Finding | Severity | Evidence / affected files | Proposed bounded change | Regression checks | Blast radius | Dedicated branch/PR |
| --- | --- | --- | --- | --- | --- | --- |
| P0-1 | P0 | Nested Phase34-37 `env_step`; non-equivalent shared helpers | Add monitor predictor callback and exact one-step equality tests | Guard, equality, clamp, comparator | Medium | Yes |
| P1-1 | P1 | 34 nested `env_step` definitions; copied crossing/termination | Extract only the active one-step seam after characterization | Guard plus state equality | Medium | Yes |
| P1-2 | P1 | Fixed Phase34-37 output writers | Add future-run output refusal and new-directory policy | Protected path snapshot, guard | Low | Yes |
| P1-3 | P1 | `scripts/check_phase_results.py` aggregate-only checks | Add separate protected artifact schema/checksum validator | Current guard plus checksum fixtures | Low | Yes |
| P1-4 | P1 | Active Python 3.13 lacks required packages | Define and document one implementation preflight environment | Smoke, controller, guard, manifest | Low | Yes |
| P1-5 | P1 | 19 pytest-discoverable files, import-side-effect tests | Add collection boundaries; reclassify historical experiments | `pytest --collect-only`, smoke suite | Medium | Yes |
| P1-6 | P1 | `.gitignore:261`, `.gitignore:264` | Add narrow Final Veto artifact negations before experiment run | `git check-ignore`, artifact validator | Low | Yes |
| P1-7 | P1 | No schema/label/decision fields in code | Implement narrow ablation writer and validators | Schema, taxonomy, pair tests | Medium | Yes |
| P1-8 | P1 | 501 MiB tracked, 367 MiB data, no current LFS | Plan non-destructive storage migration | Clone test, checksums, guard | High | Yes |
| P2-1 | P2 | Phase success termination differences | Preserve legacy fields; normalize only new outputs | Schema consistency tests | Low | Yes |
| P2-2 | P2 | 1.65 advisory versus 1.90 realized thresholds | Introduce explicit names in new code only | Threshold freeze tests | Low | Yes |
| P2-3 | P2 | Import-time `.matplotlib` directory creation | Keep predictor and monitor modules import-pure | Import test with clean temp cwd | Low | Yes |
| P2-4 | P2 | `script/`/`scripts/`, controller namespaces, test locations | Publish ownership map before consolidation | Imports and CLI smoke | Medium | Yes |
| P2-5 | P2 | README and current-status docs point to Phase38 | Bounded README/current status refresh | Link check and guard | Low | Yes |
| P2-6 | P2 | Three dependency contracts differ | Define core/experiment/publishing groups | Fresh environment smoke | Medium | Yes |
| P2-7 | P2 | CI lacks manifest/schema checks | Add manifest validation as smallest next CI step | CI green on PR and push | Low | Yes |
| P2-8 | P2 | No `.gitattributes`; mixed EOL | Add explicit text policy in isolated normalization commit | `git diff --check`, no content diffs | Medium | Yes |
| P2-9 | P2 | Unsafe deserialization call sites | Document trust boundary and adopt safe modes where compatible | Model load smoke and hash checks | Medium | Yes |
| P2-10 | P2 | Historical provenance gaps | Require provenance on new artifacts only | Manifest/result schema checks | Low | Yes |
| P2-11 | P2 | Test compatibility shims patch APIs | Add direct API tests, then narrow shims | Smoke on Windows/Linux | Medium | Yes |
| P3-1 | P3 | 11 missing Markdown links | Repair or mark unavailable | Link scanner | Low | Optional |
| P3-2 | P3 | Typo/space/inconsistent names | Create rename and compatibility map | Import/path link checks | Medium | Yes |
| P3-3 | P3 | Empty local directories | Manual review and explicit cleanup approval | `git status`, no tracked deletion | Low | No |
| P3-4 | P3 | No contribution/templates | Add concise research-change checklist | Documentation review | Low | Optional |
| P3-5 | P3 | Superseded plans lack banners | Add documentation-only status banners | Link and claim review | Low | Optional |
| P3-6 | P3 | Broad unrelated untracked workspaces | Decide archive/track/ignore policy manually | Full status review | High if moved | Yes |

## Do Now / Do Next / Postpone

| Timing | Work | Rationale |
| --- | --- | --- |
| Do now | Freeze and validate the Final Veto experiment manifest | Completed today; no physics or results required |
| Do now | Keep the protected guard passing and preserve unrelated workspaces | Maintains evidence integrity |
| Do next | Add the minimal overspeed monitor with a pure predictor interface and tests proving one-step equality with rollout dynamics | Resolves P0-1 before intervention logic can be trusted |
| Do next | Keep monitor logic limited to threshold comparison and allow/veto evidence | Prevents physics ownership from entering Runtime Assurance code |
| Do next | Add narrow output isolation and action logging required by the monitor test | Makes later paired evidence auditable |
| Postpone | Paired runner and formal preservation/stress experiment | Monitor and equality tests do not exist yet |
| Postpone | Broad physics modularization | High blast radius before active seam is proven |
| Postpone | Full schema migration or historical CSV rewrite | Historical evidence must remain immutable |
| Postpone | README, `.gitignore`, CI, and dependency changes | Each deserves an isolated bounded issue |
| Postpone | LFS/history migration | Requires storage and reproducibility plan |
| Postpone | Decision Manager, trust manager, multi-hazard assurance, other embodiments, and hardware | Not needed for the first falsifiable simulator monitor |

## Explicit Non-Actions Taken

- Did not implement `runtime_assurance/final_veto_monitor.py`.
- Did not implement `scripts/run_final_veto_ablation.py`.
- Did not run preservation or stress rollouts.
- Did not create result CSVs, paired CSVs, decision logs, plots, or experiment summaries.
- Did not modify controller code, simulator physics, action clamp, integration order, thresholds, or case grids.
- Did not rewrite or regenerate any Phase34/35/36/37 artifact.
- Did not modify README, `.gitignore`, CI workflows, environment files, or dependencies.
- Did not stage, commit, move, delete, or inspect the content of unrelated untracked paper, submission, print, release, or research-workspace files.
- Did not run Git garbage collection, history rewriting, LFS migration, or destructive cleanup.
- Did not claim formal safety, Runtime Assurance verification, hardware readiness, or cross-domain validation.

## Audit Conclusion

The repository has a credible protected evidence base and sufficiently precise Week 1-8 contracts to begin bounded implementation. The experiment manifest is now frozen and validated. Safe monitor implementation remains blocked until the transition predictor is an explicit pure interface and one-step equality with the rollout is tested. The next implementation issue should address only that boundary and the minimal overspeed decision rule. The paired runner and experiment must remain deferred.
