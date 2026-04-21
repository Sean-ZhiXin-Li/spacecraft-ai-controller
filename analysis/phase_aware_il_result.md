# Phase-Aware Imitation Learning Result

## Main Answer

- Does adding phase information recover crossing? `Yes`

## Interpretation

- With oracle phase labels provided online, the learned policy recovers target-radius crossing under this baseline.
- What is still missing is self-contained phase inference and post-crossing stabilization quality, not only the first transition itself.
- Evaluation uses the explicit controller only as a phase oracle to provide the phase one-hot input online; the learned model still produces the action itself.
- This is therefore not a fully autonomous learned controller result yet.

## Metrics

- crossing_occurs `True`
- radius_crossings_total `2`
- success `False`
- final_radius_error `1.018e+08`