# Learned Phase Stabilized Evaluation Result

## Main Answer

- Does phase stabilization improve over learned_phase_il? `No`
- Does it reduce phase oscillation? `Yes`
- Does it improve crossing or final radius error? `No`
- Does it achieve success? `No`

## Interpretation

- The raw phase head is smoothed with consecutive-step hysteresis, but the loaded policy action head does not consume that stabilized phase signal.
- As expected for this architecture, phase stabilization reduces the diagnostic phase oscillation but does not materially change closed-loop behavior.
- What remains missing is a way for the stabilized internal phase state to affect action selection, or a temporally consistent action policy that learns stable post-crossing dynamics directly.

## Metrics

- crossing_occurs `True`
- radius_crossings_total `5`
- first_crossing_step `36689`
- success `False`
- final_radius_error `2.392e+09`
- raw_phase_transitions `18518`
- stabilized_phase_transitions `3`

## Learned-Phase Reference

- crossing_occurs `True`
- radius_crossings_total `5`
- first_crossing_step `36689`
- success `False`
- final_radius_error `2.392e+09`