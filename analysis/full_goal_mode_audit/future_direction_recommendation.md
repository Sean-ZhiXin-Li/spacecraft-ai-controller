# Future Direction Recommendation

## Direction Choice

Direction B is scientifically stronger:

- trajectory geometry
- recoverability structure
- transfer-family science
- crossing-state quality
- insertion architecture
- clean benchmark discipline

Direction A should not be the next step:

- larger RL systems
- bigger planners
- stronger optimizers
- more complexity
- more phases

The repo does not currently need more complexity. It needs clearer trajectory-family evidence.

## Should Phase36B Continue?

Yes. Phase36B is the correct next phase if it is disciplined.

It should not be another local gain search. It should be a full 24-case transfer-family benchmark with Phase34 fixed as the terminal controller.

## Full Benchmark or Subset?

Phase36B should use the full 24-case benchmark.

Phase36A already served the subset visualization role. Continuing with subsets would risk overfitting interpretation to a few hand-picked examples.

## MPC-Lite Timing

MPC-lite should wait.

The project should first identify which transfer-family geometry is promising. MPC-lite is more useful after the family space shows a structure worth exploiting.

## 3D, C++, and SPICE

Delay all of these:

- 3D orbital mechanics
- C++ simulation core
- SPICE integration
- high-fidelity perturbations
- large RL systems

These are not wrong long-term, but they would currently add complexity before the 2D architecture is understood.

## Safest Next Technical Step

Phase36B should test:

- `baseline_phase34`
- `spiral_approach`
- `grazing_corridor`
- redesigned `delayed_crossing`

Primary outcomes:

- crossing count
- Phase34-compatible crossing count
- recoverable crossing count
- crossing vr ratio
- crossing vt error ratio
- crossing sync error
- min radius error
- overspeed
- instability

## Final Recommendation

Continue toward transfer-family science, not PPO scaling. The next 30 days should produce a clean transfer-family benchmark and a shared rollout/metric core, not another stack of one-off phase scripts.

