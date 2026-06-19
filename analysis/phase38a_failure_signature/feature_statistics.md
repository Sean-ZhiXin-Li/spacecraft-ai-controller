# Phase38A Failure Signature Feature Statistics

Scope: descriptive mining of existing CSV artifacts from Phase34, Phase36B, Phase36C, Phase37A, and Phase37B. No new controller experiment was run. No new physics metric was derived.

## Class Counts

Rows are classified as:

- `crossing_producing`: row has `crossing_occurs=True`.
- `near_crossing`: non-crossing row with recorded failure label `near_crossing`.
- `over_conservative_transfer`: non-crossing row with recorded failure label `over_conservative_transfer`.

| Class | Rows |
|---|---:|
| `crossing_producing` | 96 |
| `near_crossing` | 200 |
| `over_conservative_transfer` | 16 |

## Class Counts By Phase

| Phase | Crossing-producing | Near crossing | Over conservative |
|---|---:|---:|---:|
| Phase34 | 32 | 0 | 0 |
| Phase36B | 32 | 56 | 8 |
| Phase36C | 0 | 8 | 8 |
| Phase37A | 24 | 120 | 0 |
| Phase37B | 8 | 16 | 0 |

## Numeric Feature Statistics

### `first_crossing_step`

| Class | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| `crossing_producing` | 88 | 1159.181818 | 182 | 10 | 29076 |
| `near_crossing` | 0 | N/A | N/A | N/A | N/A |
| `over_conservative_transfer` | 0 | N/A | N/A | N/A | N/A |

Interpretation: `first_crossing_step` is a crossing-only field. It separates actual crossing rows from non-crossing rows but does not explain why non-crossing cases fail.

### `closest_approach_step`

| Class | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| `crossing_producing` | 64 | 1 | 1 | 1 | 1 |
| `near_crossing` | 200 | 34682.095 | 117 | 1 | 100000 |
| `over_conservative_transfer` | 16 | 1 | 1 | 1 | 1 |

Interpretation: `closest_approach_step` provides a timing signature. In the mined rows, over-conservative rows cluster at early closest approach, while near-crossing rows include many late or max-step approaches.

### `min_abs_radius_error_ratio`

| Class | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| `crossing_producing` | 64 | 0 | 0 | 0 | 0 |
| `near_crossing` | 200 | 0.016781 | 0.019984 | 0 | 0.02 |
| `over_conservative_transfer` | 16 | 0.02 | 0.02 | 0.02 | 0.02 |

Interpretation: this is useful as a diagnostic closest-approach field, but it is not a success metric. Phase37B already showed that tiny closest-approach improvements can occur without new crossings.

### `best_crossing_potential`

| Class | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| `crossing_producing` | 56 | 0.95913 | 0.943731 | 0.943729 | 0.999445 |
| `near_crossing` | 184 | 0.817806 | 0.820018 | 0.772068 | 0.822413 |
| `over_conservative_transfer` | 16 | 0.683357 | 0.691791 | 0.651392 | 0.698454 |

Interpretation: `best_crossing_potential` has moderate class separation in the recorded rows, but Phase36C showed it can move without creating target-radius crossings.

### `crossing_vr_ratio`

| Class | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| `crossing_producing` | 96 | -0.040717 | -0.013726 | -0.153813 | -0.000146 |
| `near_crossing` | 0 | N/A | N/A | N/A | N/A |
| `over_conservative_transfer` | 0 | N/A | N/A | N/A | N/A |

Interpretation: crossing-state velocity fields are only available after crossing. They are useful for handoff quality, not for pre-cross failure-class separation.

### `crossing_vt_error_ratio`

| Class | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| `crossing_producing` | 96 | -0.74801 | -0.923337 | -1.120147 | -0.000395 |
| `near_crossing` | 0 | N/A | N/A | N/A | N/A |
| `over_conservative_transfer` | 0 | N/A | N/A | N/A | N/A |

Interpretation: useful after crossing, but unavailable for non-crossing rows.

### `crossing_sync`

| Class | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| `crossing_producing` | 96 | 4.190102 | 3.933694 | 2.752643 | 7.690632 |
| `near_crossing` | 0 | N/A | N/A | N/A | N/A |
| `over_conservative_transfer` | 0 | N/A | N/A | N/A | N/A |

Interpretation: crossing synchronization explains handoff quality among crossing rows, not why non-crossing rows fail.

### `best_post_cross_distance`

| Class | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| `crossing_producing` | 96 | 1.799893 | 0.990288 | 0.62947 | 8.013551 |
| `near_crossing` | 192 | 8.250742 | 8.995866 | 4.026824 | 9.816804 |
| `over_conservative_transfer` | 8 | 15.499169 | 13.481946 | 9.248438 | 25.806354 |

Interpretation: supports the existing distinction between post-cross recoverability and upstream crossing generation. It should not be treated as a direct controller variable.

### `max_speed_ratio`

| Class | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| `crossing_producing` | 88 | 1.000406 | 1 | 1 | 1.008434 |
| `near_crossing` | 176 | 1.123883 | 1.006375 | 0.929264 | 1.484433 |
| `over_conservative_transfer` | 8 | 0.970572 | 0.971122 | 0.96751 | 0.974185 |

Interpretation: speed differs descriptively across some classes, but overspeed remains false in the inspected Phase34/36/37 evidence. This is not enough to justify a new speed-based controller variable.

### `radial_magnitude`

| Class | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| `crossing_producing` | 24 | 0.071667 | 0.055 | 0.055 | 0.105 |
| `near_crossing` | 120 | 0.081667 | 0.105 | 0.055 | 0.105 |
| `over_conservative_transfer` | 0 | N/A | N/A | N/A | N/A |

Interpretation: Phase37A directly tested radial magnitude with timing variants and created zero new crossings. This feature has negative implementation evidence.

## Boolean Outcome Summary

| Feature | Evidence summary |
|---|---|
| `crossing_occurs` | Strong outcome separator by definition. Not an explanatory variable. |
| `recoverable_crossing` | Strong post-cross outcome metric; supports Phase34 as downstream recovery result. |
| `simulator_success_label` / `success` | Useful recorded label, but unsafe as a primary scientific claim without the simulator-defined qualifier. |
| `overspeed` | No useful class separation in inspected evidence; keep as safety guard. |
| `instability` | No useful class separation in inspected evidence; keep as safety guard. |
| `new_crossing_on_baseline_non_crossing_case` | Directly important in Phase37A; recorded count was zero. |
