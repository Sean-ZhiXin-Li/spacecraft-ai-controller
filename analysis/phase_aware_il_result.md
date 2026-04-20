# Phase-Aware Imitation Learning Result

## Main Answer

- Does adding phase information recover crossing? `Yes`

## Interpretation

- Adding phase information is enough to recover first crossing under this baseline, which implies that structural representation is the missing ingredient.
- What is still missing is post-crossing stabilization quality, not the first transition itself.
- Evaluation uses the explicit controller only as a phase oracle to provide the phase one-hot input online; the learned model still produces the action itself.

## Metrics

- crossing_occurs `True`
- radius_crossings_total `1`
- success `False`
- final_radius_error `3.167e+08`