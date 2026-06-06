# Phase37A Radial Commitment Timing Sweep

## Scope

- Fixed 24-case reduced benchmark.
- Focused interpretation on the 16 Phase36B baseline non-crossing cases.
- Phase34 `radius_priority` post-cross synchronization is the fixed terminal controller after first crossing.
- No tangential search, no coast-duration variable, no MPC, and no RL.
- This is a simplified 2D orbital-control sandbox result, not real spacecraft validation.

## Parameters

| Commit timing | Radial magnitude label | Radial magnitude |
|---|---|---:|
| `early_commit` | `low` | 0.055 |
| `early_commit` | `medium` | 0.105 |
| `mid_commit` | `low` | 0.055 |
| `mid_commit` | `medium` | 0.105 |
| `delayed_commit` | `low` | 0.055 |
| `delayed_commit` | `medium` | 0.105 |

## Aggregate Results

| Variant | Cases | Crossings | New crossings on baseline non-crossing cases | Phase34-compatible crossings | Recoverable crossings | Simulator success label | Overspeed | Instability | Mean crossing sync |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `early_commit_low` | 24 | 4 | 0 | 4 | 4 | 4 | 0 | 0 | 3.9797 |
| `early_commit_medium` | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | N/A |
| `mid_commit_low` | 24 | 4 | 0 | 4 | 4 | 4 | 0 | 0 | 3.9797 |
| `mid_commit_medium` | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | N/A |
| `delayed_commit_low` | 24 | 8 | 0 | 8 | 8 | 8 | 0 | 0 | 5.4831 |
| `delayed_commit_medium` | 24 | 8 | 0 | 8 | 8 | 8 | 0 | 0 | 5.7453 |

## Required Report

- Total rollout count: `144`.
- Best total crossing variant: `delayed_commit_low` with `8 / 24` crossings.
- Best new-crossing variant: `early_commit_low` with `0 / 16` new crossings on baseline non-crossing cases.
- Best recoverable-crossing variant: `delayed_commit_low` with `8 / 24` recoverable crossings.
- Overspeed count across all variants: `0`.
- Instability count across all variants: `0`.

## Interpretation

Phase37A did not create new target-radius crossings on the Phase36B baseline non-crossing set.

The result should be read only as evidence about radial commitment timing in the current 2D sandbox. It does not change Phase34, does not introduce a planner, and does not validate real spacecraft control.

## Phase37B Decision

Do not expand radial commitment timing blindly; inspect whether closest-approach metrics shifted before considering Phase37B.

## Artifacts

- `phase37a_results.csv`
- `phase37a_summary.md`
- `phase37a_comparison.png`