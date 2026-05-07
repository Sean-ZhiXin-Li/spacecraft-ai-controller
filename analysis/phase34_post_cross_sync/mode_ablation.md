# Phase 34 Mode Ablation

| Mode | Duration | Max norm | Crossing-case best sync | Crossing-case best distance | All-case best distance | Recoverable crossings | Mean control smoothness |
|---|---:|---:|---:|---:|---:|---:|---:|
| `radius_priority` | 360 | 0.026 | 0.9855 | 0.9855 | 8.3901 | 8 | 0.00000122 |
| `sync_balanced` | 520 | 0.021 | 0.9902 | 0.9902 | 8.3917 | 8 | 0.00000027 |
| `vt_priority_then_sync` | 560 | 0.020 | 0.9891 | 0.9891 | 8.3914 | 8 | 0.00000017 |

Best ablation mode: `radius_priority`.
The ablation is interpreted by basin distance first, then recoverable crossings. Smoothness is diagnostic, not a success metric by itself.