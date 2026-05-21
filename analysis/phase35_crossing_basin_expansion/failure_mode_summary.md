# Phase 35 Non-Crossing Failure Mode Summary

Baseline Phase34 non-crossing cases are diagnosed using the Phase35 metric set, not by changing Phase34 behavior.

| Failure label | Cases |
|---|---:|
| `near_crossing` | 8 |
| `over_conservative_transfer` | 8 |

## Interpretation

- Baseline non-crossing cases diagnosed: `16`.
- Dominant label: `near_crossing, over_conservative_transfer`.
- `near_crossing` means the trajectory came close enough that better local timing may matter.
- `over_conservative_transfer` means the trajectory stayed near the target-radius boundary but did not commit to crossing.
- `wrong_tangential_corridor` means angular momentum or tangential velocity alignment is the main blocker.
- `insufficient_radial_energy` means the trajectory did not carry enough useful radial motion toward the target radius.
- `bad_initial_geometry` and `dead_geometry` are stronger signs that a local bias is not enough.
