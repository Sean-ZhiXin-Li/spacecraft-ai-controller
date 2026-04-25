# Boundary Refinement v2 Summary

## Detection

- Automatic frontier detection from `phase_map_v2.csv` found success/failure transitions at `dt=50`, `90`, `100`, `130`, `140`, and `150`.
- The adaptive v2 refinement focused on the validated transition and neighboring successful bands: `dt=85..155` and `r0_over_target` near `1.00003..1.00010`.

## Validated dt=100 Boundary

- On the validated `dt=100` line, success is first lost at `r0_over_target=1.00006`.
- The validated baseline `r0_over_target=1.00005` remains successful.

## Boundary Shape

- The `r0` boundary is mostly monotonic within each fixed-`dt` row in the focused refinement.
- Success is **not monotonic across `dt`**. The controller fails for the focused band around `dt=105`, regains a broad success region around `dt=130..150`, then fails again at `dt=155`.
- Isolated or narrow success pockets are present, especially around `dt=90`.
- Best successful refined point by final radius error: `dt=145`, `r0_over_target=1.00003`, final_radius_error `3.523e+03`.

## Per-dt Frontier

- `dt=85`: no success in the focused range.
- `dt=90`: isolated success at `r0_over_target=1.00003`; first failure at `1.00004`.
- `dt=95`: success through approximately `1.000075`; failure begins by approximately `1.000080`.
- `dt=100`: success through `1.00005`; first failure at `1.00006`.
- `dt=105`: no success in the focused range.
- `dt=125`: success through approximately `1.000035`; failure begins by approximately `1.000040`.
- `dt=130`: success through approximately `1.000085`; failure begins by approximately `1.000090`.
- `dt=135`, `140`, `145`: success across the full focused range through `1.00010`.
- `dt=150`: success through approximately `1.000085`; failure begins by approximately `1.000090`.
- `dt=155`: no success in the focused range.
