# Phase-Conditioned Imitation Learning Result

## Main Answer

- Does conditioning action on phase improve over learned_phase_il? `No`
- Does it reduce oscillation? `Yes`
- Does it achieve success? `No`

## Interpretation

- Training uses ground-truth explicit-controller phase labels for the action branch.
- Evaluation uses only predicted phase argmax; no oracle phase is provided online.
- What gap remains vs explicit controller? The explicit controller still provides a stable hand-coded phase transition law and phase-conditioned action law; this learned policy still has to infer phase online and keep the action branch stable over long horizons.

## Metrics

- crossing_occurs `False`
- radius_crossings_total `0`
- first_crossing_step `None`
- success `False`
- final_radius_error `3.794e+08`
- predicted_phase_transitions `1`

## Learned-Phase Reference

- crossing_occurs `True`
- radius_crossings_total `5`
- first_crossing_step `36689`
- success `False`
- final_radius_error `2.392e+09`
- predicted_phase_transitions `18518`