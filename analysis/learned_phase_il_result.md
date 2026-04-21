# Learned Phase Imitation Learning Result

## Main Answer

- Does learned phase representation recover crossing without oracle phase input? `Yes`
- Does it improve over history-aware imitation? `No.`
- Does it achieve success? `No`

## Interpretation

- Evaluation input is only the normalized state history; no explicit-controller phase is provided online.
- The learned internal phase signal still does not close the loop into a stable post-crossing insertion; the missing piece is reliable stabilization after the first radius transition.

## Metrics

- crossing_occurs `True`
- radius_crossings_total `5`
- first_crossing_step `36689`
- success `False`
- final_radius_error `2.392e+09`

## History-Aware Reference

- crossing_occurs `True`
- radius_crossings_total `1`
- first_crossing_step `40969`
- success `False`
- final_radius_error `5.392e+08`