# Final Veto Paired Diagnostic Summary

This summary separates declared-hazard evidence from task completion and does not make a formal-safety claim.

## Declared hazard reduction

- Complete valid pairs avoiding the declared overspeed hazard: 5.

## Task outcome

- Pairs recovering the declared task after hazard avoidance: 0.
- Pairs preserving the explicit recoverable-crossing and simulator-success tuple: 13.

## Intervention burden

- Monitor evaluations: 511327.
- Vetoes: 499877.
- Aggregate intervention rate: 0.97760728.

## Performance cost

- Monitor-on minus monitor-off step deltas: [0, 0, 0, 0, 0, 0, 0, 0, 99978, 99972, 99975, 99974, 99973].
- Step deltas are reported without automatically labeling their sign as beneficial or harmful.

## Terminal failure-mode transition

- overspeed -> max_steps.
- success -> success.

Declared hazard avoidance does not by itself mean the task recovered or completed.
