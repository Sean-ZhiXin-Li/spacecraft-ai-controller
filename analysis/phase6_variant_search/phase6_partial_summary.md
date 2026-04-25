# Phase 6 Partial Summary

## Data Status

- Source file: `analysis/phase6_variant_search/variant_grid.csv`.
- Completed rows available: `2250`.
- Variants with any data: `9`.
- Fully completed variants (`270/270` runs): `8`.
- This is a partial ranking only. Several variants have not been run yet, so the snapshot should not be treated as the final Phase 6 result.

## Partial Ranking

- Current top-ranked variant by the requested rule: `baseline`.
- Baseline partial result: success `64/270`, capture `65/270`, mean minimum abs radius error `2.557e+08` m.
- No completed variant is clearly ahead of baseline yet under the requested ranking rule.

## Interpretation

- moderately reduced retrograde authority is the strongest completed altered variant so far, improving reachability relative to the other completed non-baseline descent changes.
- too much retrograde reduction hurts reachability relative to both baseline and retro_085.
- radial damping variants reduce mean closest-approach error strongly, but none of the completed ones have converted that into CAPTURE entry yet.
- Some improvement is visible in partial data, but not enough to beat baseline under the requested ranking rule. `retro_085` remains behind baseline on both success and capture count (`57` vs `64/65` on matched `270`-run coverage), while still outperforming `retro_070` and the completed damping/scheduled variants.
- Baseline and `retro_115` are effectively tied at the top of the completed set, which suggests that increasing retrograde authority alone does not improve reachability over the current validated controller.
- The completed radial-damping and scheduled variants have not entered CAPTURE yet, despite some lower mean minimum-radius-error values. That points to a narrow access window where getting somewhat closer is not enough by itself.
- Because some variants are still incomplete, any statement about the best overall knob family should remain provisional.