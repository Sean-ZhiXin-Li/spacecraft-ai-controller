# GitHub Tracking Audit

Date: 2026-04-25

Scope: verify tracking status for README-linked current 2D Phase 7 milestone artifacts and key Phase 7 references.

## Checked Artifacts

| Artifact | Purpose | Tracking status before fix | Action |
|---|---|---|---|
| `analysis/phase76_soft_hybrid/phase76_summary.md` | Phase 7.6 summary | tracked | none |
| `analysis/phase76_soft_hybrid/soft_hybrid_ranking.csv` | Phase 7.6 ranking | tracked | `.gitignore` unignore exception added |
| `analysis/phase76_soft_hybrid/soft_hybrid_comparison.png` | Phase 7.6 comparison plot | tracked | `.gitignore` unignore exception added |
| `analysis/phase76_soft_hybrid/soft_hybrid_success_map.png` | Phase 7.6 success map | tracked | `.gitignore` unignore exception added |
| `analysis/phase7_pre_window_shaping/phase7_summary.md` | Phase 7 summary | tracked | none |
| `analysis/phase7_pre_window_shaping/pre_window_ranking.csv` | Phase 7 ranking | tracked | `.gitignore` unignore exception added |
| `analysis/phase7_pre_window_shaping/best_prewindow_success_map.png` | Phase 7 map | tracked | `.gitignore` unignore exception added |
| `analysis/phase75_hybrid/phase75_summary.md` | Phase 7.5 summary | tracked | none |
| `analysis/phase75_hybrid/hybrid_ranking.csv` | Phase 7.5 ranking | untracked | force-added specifically with `git add -f analysis/phase75_hybrid/hybrid_ranking.csv` |
| `analysis/phase75_hybrid/hybrid_vs_baseline.png` | Phase 7.5 comparison plot | tracked | `.gitignore` unignore exception added |

## Ignore Rule Changes

Narrow exceptions were added for README-linked CSV, PNG, GIF, and JSON artifacts so future `git status` checks do not silently hide them. The repository still intentionally ignores broad generated-output classes such as `*.npy`, most `*.csv`, most `*.png`, logs, caches, and raw traces.

## Remaining Tracking Risks

- Broad ignore rules are still appropriate for raw experiment output but require explicit exceptions for any new README-linked artifacts.
- `analysis/figs/final_project/action_norm_vs_time.png` appeared as an untracked artifact after an overly broad unignore was briefly present. The ignore rule was narrowed again; this file is not README-linked and was not added.
- New audit docs and project-log index files are not ignored, but they still need to be staged by the user or a later commit step.
