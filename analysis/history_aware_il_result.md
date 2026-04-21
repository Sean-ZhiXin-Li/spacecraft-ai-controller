# History-Aware Imitation Learning Result

## Main Answer

- Does short-term history recover crossing without oracle phase input? `Yes`

## Interpretation

- Short-term state history is enough to recover target-radius crossing without oracle phase input under this baseline.
- What is still missing is stronger implicit phase inference and post-crossing stabilization quality over the full insertion horizon.
- This test used a fixed history length of `4` normalized states and no oracle phase label.

## Metrics

- crossing_occurs `True`
- radius_crossings_total `1`
- first_crossing_step `40969`
- success `False`
- final_radius_error `5.392e+08`