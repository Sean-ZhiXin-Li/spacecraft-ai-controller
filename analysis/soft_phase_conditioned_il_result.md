# Soft Phase-Conditioned Imitation Learning Result

## Main Answer

- Does soft phase conditioning recover crossing? `No`
- Does it improve over hard phase conditioning? `Yes`
- Does it improve over learned_phase_il? `No`
- Does it achieve success? `No`
- What remains missing if it still fails? The learned closed loop still fails to trigger a reliable descent-to-capture transition without oracle phase input.

## Interpretation

- Training teacher-forces the action branch with ground-truth phase converted to one-hot probabilities.
- Evaluation uses no oracle phase; the phase head softmax probabilities are fed directly into the action head.

## Metrics

- crossing_occurs `False`
- radius_crossings_total `0`
- first_crossing_step `None`
- success `False`
- final_radius_error `3.793e+08`
- predicted_phase_transitions `1`

## Hard Phase-Conditioned Reference

- crossing_occurs `False`
- radius_crossings_total `0`
- first_crossing_step `None`
- success `False`
- final_radius_error `3.794e+08`

## Learned-Phase Reference

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