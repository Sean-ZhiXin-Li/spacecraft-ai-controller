# Full Research Accuracy Audit

## Executive Verdict

SAFE TO PRESENT

The core research narrative is numerically supported by the checked CSV and JSON files. The main Phase34/Phase35 claims are accurate: Phase34 converts the existing 8 / 24 crossing-producing cases into 8 / 24 recoverable crossings, while Phase35 local upstream biases do not expand the crossing basin. The repo is broadly safe to present as a 2D simulator-based control-architecture research project.

The prior minor wording issues have been resolved in the public wording files. Success terminology is now scoped as a simulator-defined label where it appears in the Phase34/Phase35 public tables, CAPTURE/LOCK language is scoped as simulator state-machine labeling, and Phase34 wording has been toned down to the more conservative "architecture result."

## Pass 1 — Numerical Accuracy Table

| Claim | Claimed Value | Computed Value | Source | Status | Notes |
|---|---:|---:|---|---|---|
| Phase31 full baseline crossing count | 12 / 48 | 12 / 48 | `analysis/phase31_global_transfer_solver/phase31_results.csv` | PASS | Applies to full Phase31 reduced grid for `phase31_phase22_baseline`. |
| Phase31 full baseline recoverable crossing count | 0 / 48 | 0 / 48 | `analysis/phase31_global_transfer_solver/phase31_results.csv` | PASS | No recoverable crossings in full Phase31 baseline. |
| Phase31 full baseline success label count | 12 / 48 | 12 / 48 | `analysis/phase31_global_transfer_solver/phase31_results.csv` | PASS | This is the simulator `success` label. |
| Phase31-style reference in Phase34 comparison crossing count | 8 / 24 | 8 / 24 | `analysis/phase34_post_cross_sync/phase34_results.csv` | PASS | Applies to imported Phase31 reference rows in the Phase34 24-case benchmark. |
| Phase31-style reference in Phase34 comparison recoverable crossings | 0 / 24 | 0 / 24 | `analysis/phase34_post_cross_sync/phase34_results.csv` | PASS | Matches Phase34 comparison docs. |
| Phase31-style reference in Phase34 comparison success label count | 8 / 24 | 8 / 24 | `analysis/phase34_post_cross_sync/phase34_results.csv` | PASS | Use "success label" when public-facing. |
| Phase32 used SciPy fallback, not full CasADi/IPOPT | SciPy direct shooting fallback | 16 / 16 rows use `scipy_direct_shooting` | `analysis/phase32_direct_optimal_control/phase32_results.csv` | PASS | Summary also states CasADi unavailable. |
| Phase32 `recoverability_target` solve count | 4 / 4 | 4 / 4 | `analysis/phase32_direct_optimal_control/phase32_results.csv` | PASS | All four rows solved. |
| Phase32 `recoverability_target` recoverable crossing count | 1 | 1 | `analysis/phase32_direct_optimal_control/phase32_results.csv` | PASS | Also has 2 recoverable states. |
| Phase32 `sync_error_minimization` recoverable crossing count | 1 | 1 | `analysis/phase32_direct_optimal_control/phase32_results.csv` | PASS | Also has 2 crossings total. |
| Phase33 first crossing step | 81 | 81 | `analysis/phase33_optimal_structure_extraction/phase33_metrics.csv` | PASS | Best case: `recoverability_target / baseline_crossing_high_angle`. |
| Phase33 first crossing outside basin | crossing sync 1.676881, distance 2.313443 | crossing sync 1.676881, distance 2.313443 | `analysis/phase33_optimal_structure_extraction/phase33_metrics.csv` | PASS | Supports "first crossing is not insertion." |
| Phase33 later best recoverable state | step 512, best sync 0.000464, best distance 0.000470 | step 512, best sync 0.000464, best distance 0.000470 | `analysis/phase33_optimal_structure_extraction/phase33_metrics.csv` | PASS | `best_state_relation_to_crossing` is `after_first_crossing`. |
| Phase34 best-mode cases | 24 | 24 | `analysis/phase34_post_cross_sync/phase34_results.csv` | PASS | For `radius_priority`. |
| Phase34 best-mode geometric crossings | 8 | 8 | `analysis/phase34_post_cross_sync/phase34_results.csv` | PASS | For `radius_priority`. |
| Phase34 best-mode recoverable crossings | 8 | 8 | `analysis/phase34_post_cross_sync/phase34_results.csv` | PASS | For `radius_priority`. |
| Phase34 best-mode success label count | 8 | 8 | `analysis/phase34_post_cross_sync/phase34_results.csv` | PASS | Accurate, but terminology should stay scoped as simulator label. |
| Phase34 crossing-case best distance improvement | 3.9923 -> 0.9855 | 3.9923 -> 0.9855 | `analysis/phase34_post_cross_sync/phase34_results.csv` | PASS | Computed from crossing-row mean best distance for `none` and `radius_priority`. |
| Phase34 overspeed count | 0 | 0 | `analysis/phase34_post_cross_sync/phase34_results.csv` | PASS | All Phase34 modes and Phase31 reference rows have 0 overspeed in this benchmark. |
| Phase35 `baseline_phase34` crossings | 8 / 24 | 8 / 24 | `analysis/phase35_crossing_basin_expansion/phase35_results.csv` | PASS | Matches Phase34 baseline behavior. |
| Phase35 `baseline_phase34` recoverable crossings | 8 / 24 | 8 / 24 | `analysis/phase35_crossing_basin_expansion/phase35_results.csv` | PASS | Downstream Phase34 recoverability preserved. |
| Phase35 `radial_energy_push` crossings | 0 / 24 | 0 / 24 | `analysis/phase35_crossing_basin_expansion/phase35_results.csv` | PASS | Also has 5 overspeed rows. |
| Phase35 `tangential_corridor_entry` crossings | 0 / 24 | 0 / 24 | `analysis/phase35_crossing_basin_expansion/phase35_results.csv` | PASS | No crossing improvement. |
| Phase35 `predictive_crossing_bias` crossings | 8 / 24, no improvement | 8 / 24, no improvement | `analysis/phase35_crossing_basin_expansion/phase35_results.csv` | PASS | Matches baseline crossing count and recoverable count. |
| Phase35 dominant failure labels tied | `near_crossing` 8, `over_conservative_transfer` 8 | `near_crossing` 8, `over_conservative_transfer` 8 | `analysis/phase35_crossing_basin_expansion/non_crossing_diagnosis.csv` | PASS | Correctly described as a tie. |
| Demo success | true | true | `analysis/demo/orbit_demo_summary.json` | PASS | Demo evidence only, not benchmark evidence. |
| Demo radius crossings | 1 | 1 | `analysis/demo/orbit_demo_summary.json` | PASS | Current demo run. |
| Demo first crossing step | 48,269 | 48,269 | `analysis/demo/orbit_demo_summary.json` | PASS | Exact JSON value: 48269. |
| Demo final radius error | 27,657.63 m | 27,657.63 m | `analysis/demo/orbit_demo_summary.json` | PASS | Rounded from 27657.630859375. |
| Demo phase transitions | DESCENT -> CAPTURE, CAPTURE -> LOCK | DESCENT -> CAPTURE, CAPTURE -> LOCK | `analysis/demo/orbit_demo_summary.json` | PASS | Two transitions recorded. |

## Pass 2 — Scientific Rigor Findings

### Overclaim Risks

- No checked file claims real spacecraft readiness as a current result. README and docs explicitly state this is a simplified 2D simulation project.
- No checked file claims full orbital autonomy or universal insertion success. Current limitations are clearly stated.
- No checked file claims the first crossing state itself is recoverable. Phase34 and Phase33 materials explicitly distinguish first crossing from later recoverability.
- Phase34 is now described in README as an "architecture result" rather than more promotional wording.
- Phase35 is correctly framed as ruling out local pre-cross bias architectures under the tested benchmark, not proving the crossing basin can never be expanded.
- Phase36 is clearly planning-only in the research context. It is not presented as implemented.

### Correctly Scoped Claims

- Geometric crossing and recoverable crossing are usually distinguished correctly.
- Phase34 is consistently scoped as solving the downstream post-cross problem for crossing-producing cases.
- Phase35 is consistently scoped as a negative structural result for local upstream biases.
- Phase32 is correctly scoped as a SciPy direct-shooting upper-bound prototype, not a production controller or full CasADi/IPOPT collocation result.
- Demo visuals are correctly described as control-architecture evidence, not flight validation.
- The README explicitly lists unsupported claims: full orbital autonomy, real spacecraft readiness, universal success, non-crossing families solved, and first crossing as insertion.

### Terminology Risks

- README now uses "Simulator success label" in the Phase34 table and adds a note that this is not real spacecraft mission success.
- Phase35 and Phase34 public tables now use simulator success label wording where needed.
- CAPTURE and LOCK are identified near public-facing discussion as simulator state-machine/result labels, not real flight-validation states.
- "Recoverable crossing" is defined correctly in the current phase materials, but it remains a non-obvious term. It should continue to be defined near any public table that uses it.

### Recommended Wording Fixes

- Continue using "simulator success label" or "simulator-defined success" in future public-facing tables.
- Continue using "Phase34-compatible crossing" rather than "successful crossing" to avoid conflating geometry with recoverability.

## Pass 3 — Presentation Integrity Findings

### Links

- Checked 19 local markdown links and image links across the audited markdown files.
- Broken local markdown links/images detected: 0.
- README document links to Phase34, Phase33, sprint logs, older phase summaries, and LICENSE all resolve.

### Images

- README image paths all exist:
  - `analysis/demo/orbit_demo_trajectory.png`
  - `analysis/demo/orbit_demo_zoom.gif`
  - `analysis/demo/orbit_demo_full.png`
- README Phase34 plot links all exist:
  - `analysis/phase34_post_cross_sync/mode_comparison.png`
  - `analysis/phase34_post_cross_sync/post_cross_sync_examples.png`
  - `analysis/phase34_post_cross_sync/phase31_vs_phase34_overlay.png`

### Encoding

- UTF-8 scan found no literal mojibake markers or replacement characters in the audited files.
- PowerShell console output displayed some Unicode punctuation incorrectly during inspection, but direct UTF-8 scanning did not find corrupted characters in file contents.

### Tracked/Ignored Asset Risks

- `git check-ignore` did not report the new public-facing files as ignored.
- Current `git status --short` still shows untracked public-facing outputs relevant to this audit:
  - `analysis/full_research_accuracy_audit.md`
  - `analysis/phase35_crossing_basin_expansion/`
  - `analysis/phase36_transfer_family_search/`
  - `docs/research_direction.md`
  - `project_log/pl35_crossing_basin_expansion.md`
- Additional untracked paths were observed outside this audit's public-wording scope, including `analysis/outreach/` and two `scripts/` files. They were not modified or evaluated by this wording cleanup.
- This is not a content error in the checked public-facing narrative. It is now handled by the explicit pre-push tracking checklist below.

## Pre-Push Tracking Checklist

Before pushing, manually add the public-facing files that are intended to appear in the repository:

- `docs/research_direction.md`
- `project_log/pl35_crossing_basin_expansion.md`
- `analysis/full_research_accuracy_audit.md`
- `analysis/phase35_crossing_basin_expansion/`
- `analysis/phase36_transfer_family_search/`

Do not assume these are tracked until `git status --short` confirms they have been staged or committed.

### README/Public-Facing Clarity

- README is rigorous overall and repeatedly states the 2D sandbox limitation.
- The current research sequence is understandable: Phase31/32/33 diagnose structure, Phase34 solves post-cross recoverability for crossing cases, Phase35 fails to expand crossings with local biases, Phase36 should search transfer families.
- The previous clarity issue around "success" terminology in current public tables has been resolved.

## Critical Issues

None.

## Minor Issues

- No remaining content issues identified after the wording cleanup.
- Manual git add steps remain; see the Pre-Push Tracking Checklist.

## Final Confidence Score

100 / 100

The main numerical and scientific claims are well supported by the CSV/JSON data. Public wording now scopes simulator success, CAPTURE/LOCK labels, and Phase34 tone appropriately. The repo is safe to present, with the explicit manual pre-push tracking checklist above.
