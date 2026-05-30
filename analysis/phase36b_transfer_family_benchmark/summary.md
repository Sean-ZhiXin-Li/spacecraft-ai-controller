# Phase36B Transfer Family Benchmark

## Scope

- Full 24-case reduced benchmark used for Phase34 and Phase35.
- Simplified 2D orbital-control sandbox only; this is not real spacecraft validation.
- Phase34 `radius_priority` post-cross synchronization is the fixed terminal controller.
- Physics, CAPTURE/LOCK thresholds, recoverability thresholds, and termination checks are unchanged.
- Phase36A representative-subset results are not used as full benchmark evidence.

## Families

- `baseline_phase34`: Phase34 reference transfer.
- `spiral_approach`: gradual low-thrust radius shaping.
- `grazing_corridor`: near-target loiter with soft crossing windows.
- `redesigned_delayed_crossing`: bounded delayed commitment before crossing.

Excluded unless redesigned: `energy_bleed_then_cross`, `overshoot_return`, and `two_stage_transfer`.

## Aggregate Results

| Transfer family | Cases | Crossings | Phase34-compatible crossings | Recoverable crossings | Simulator success label | Overspeed | Instability | Mean crossing sync | Mean best post-cross distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_phase34` | 24 | 8 | 8 | 8 | 8 | 0 | 0 | 3.2940 | 0.9855 |
| `spiral_approach` | 24 | 8 | 8 | 8 | 8 | 0 | 0 | 4.2017 | 0.9818 |
| `grazing_corridor` | 24 | 8 | 8 | 8 | 8 | 0 | 0 | 4.7752 | 0.8107 |
| `redesigned_delayed_crossing` | 24 | 8 | 8 | 8 | 8 | 0 | 0 | 5.0664 | 0.8229 |

## Required Questions

- Did any family improve geometric crossing count over `baseline_phase34`? `no`. Best crossing family: `baseline_phase34` with `8 / 24` crossings; baseline has `8 / 24`.
- Did any family improve recoverable crossing count over `baseline_phase34`? `no`. Best recoverable family: `baseline_phase34` with `8 / 24`; baseline has `8 / 24`.
- Did any new crossing remain Phase34-compatible? `no`.
- Which family produced the best crossing-state quality? `baseline_phase34` by mean crossing sync among families with crossings.
- Which family caused overspeed or instability? Overspeed: `none`. Instability: `none`.
- Which failures were near-crossing or over-conservative? `near_crossing=56`, `over_conservative_transfer=8`.
- Does the result support or weaken the transfer-family geometry hypothesis? Phase36B narrowed the transfer-family hypothesis space but did not expand the crossing basin under the tested family set.
- Should the next step be planner-level search, MPC-lite, or family redesign? `planner-level transfer-family search before MPC-lite`.

## Failure Mode Counts

| Failure label | Count |
|---|---:|
| `near_crossing` | 56 |
| `over_conservative_transfer` | 8 |

## Artifacts

- `phase36b_results.csv`
- `phase36b_family_summary.csv`
- `phase36b_failure_modes.csv`
- `phase36b_family_comparison.png`