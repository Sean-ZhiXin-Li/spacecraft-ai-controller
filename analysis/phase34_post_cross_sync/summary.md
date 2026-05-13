# Phase 34 Post-Cross Smooth Synchronization Controller

## Scope

- Python-only 2D explicit controller.
- Physics, reward, thresholds, CAPTURE, and LOCK rules are unchanged.
- Phase 22/31 early transfer behavior is preserved; Phase 34 inserts a post-cross synchronization mode after first target-radius crossing.
- Benchmark: 24 representative reduced-grid cases across three post-cross modes, plus Phase31 baseline reference rows from existing CSV.

## Results

| Controller/mode | Cases | Crossings | Recoverable states | Recoverable crossings | CAPTURE | LOCK | Success | Mean crossing sync | Crossing-case best sync | Crossing-case best distance | Finite-row best distance | Overspeed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 24 | 8 | 0 | 0 | 8 | 8 | 8 | 3.8769 | nan | 3.9923 | 3.9923 | 0 |
| `radius_priority` | 24 | 8 | 8 | 8 | 8 | 8 | 8 | 3.2940 | 0.9855 | 0.9855 | 8.3901 | 0 |
| `sync_balanced` | 24 | 8 | 8 | 8 | 8 | 8 | 8 | 3.2940 | 0.9902 | 0.9902 | 8.3917 | 0 |
| `vt_priority_then_sync` | 24 | 8 | 8 | 8 | 8 | 8 | 8 | 3.2940 | 0.9891 | 0.9891 | 8.3914 | 0 |

## Research Answers

1. Does post-cross sync improve recoverability? `yes` by recoverable-crossing count; crossing-case distance improvement: `yes`.
2. Which matters more, crossing quality or post-cross correction quality? Post-cross correction dominates on crossing-producing cases: crossing sync remains outside basin, but best post-cross sync enters the recoverable threshold.
3. Can heuristic smooth steering approximate optimal structure? `partially` based on best-distance comparison.
4. Which mode best reproduces Phase 32 motif? `radius_priority` by mean best distance.
5. Is architecture gap reduced? `yes` under this bounded hand-built controller.

## Honesty Note

- This script does not claim the first crossing itself is recoverable. `recoverable_crossing` means crossing occurred and the post-cross synchronization arc later reached a recoverable state.
- For Phase34 modes, finite-row best distance includes all 24 cases. For the imported Phase31/`none` reference, only the 8 crossing rows have finite best-distance values, so its finite-row value is effectively a crossing-row mean.
- Crossing-case best distance is the primary Phase34 diagnostic. Non-crossing families remain outside this phase's scope.
- CAPTURE and LOCK thresholds are not relaxed.
- If recoverable crossings do not improve, the conclusion is that optimal structure exists but this heuristic imitation is insufficient.
