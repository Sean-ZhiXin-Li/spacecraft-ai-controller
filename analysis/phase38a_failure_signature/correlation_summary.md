# Phase38A Correlation Summary

Scope: evidence-oriented association summary from recorded CSV fields. No machine-learning model was fit.

| Feature | Evidence category | Summary |
|---|---|---|
| `crossing_occurs` | Strong evidence | Defines the crossing-producing class and remains the primary event metric. It is not an explanatory input. |
| `r0_over_target` | Strong evidence | Clearly separates recorded Phase36C baseline classes in the reduced benchmark. This is descriptive, not proof that radius ratio alone causes crossing. |
| `closest_approach_step` | Strong evidence | Separates failure timing signatures between near-crossing and over-conservative-transfer rows in the recorded baseline evidence. |
| `first_crossing_step` | Strong evidence | Exists only for crossing rows. Useful for crossing handoff timing, not for non-crossing causal separation. |
| `recoverable_crossing` | Strong evidence | Primary post-cross outcome. Supports Phase34 as a recoverability result after crossing exists. |
| `best_crossing_potential` | Moderate evidence | Higher in crossing-producing and near-crossing rows than over-conservative rows, but it can move without crossings. |
| `min_abs_radius_error_ratio` | Moderate evidence | Useful closest-approach diagnostic, but Phase37B showed improvement without selected-case crossings. |
| `best_post_cross_distance` | Moderate evidence | Supports post-cross recoverability analysis. Less useful for selecting upstream crossing-generation variables. |
| `crossing_vr_ratio` | Moderate evidence | Important crossing-state quality field, but unavailable for non-crossing rows. |
| `crossing_vt_error_ratio` | Moderate evidence | Important crossing-state quality field, but unavailable for non-crossing rows. |
| `crossing_sync` | Moderate evidence | Useful after crossing; not a pre-cross class separator. |
| `simulator_success_label` / `success` | Moderate evidence | Useful recorded label but must not be overread as mission success. |
| `commit_timing` | Weak evidence | Phase37A shows timing affects preservation of crossing cases but creates zero new crossings. |
| `radial_magnitude` | Weak evidence | Phase37A shows medium magnitude can degrade crossings and low magnitude does not create new crossings. |
| `weak_tangential_setting` | Weak/negative evidence | Phase37B shows tiny closest-approach movement but zero selected crossings and poor regression preservation. |
| `overspeed` | No evidence as separator | Consistently false in inspected evidence; keep as safety guard. |
| `instability` | No evidence as separator | Consistently false in inspected evidence; keep as safety guard. |
| `coast_duration` | Unknown | Not directly tested or recorded as an isolated variable in inspected CSVs. |
| `angular_momentum_correction` | Unknown | Not isolated as a recorded variable in inspected CSVs. |
