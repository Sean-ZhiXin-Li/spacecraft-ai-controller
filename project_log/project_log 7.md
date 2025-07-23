# Project Log – Day 7

## Date
2025-07-22

## Focus
Today focused on completing the full integration and validation of the imitation controller into the orbital simulation framework. The controller was fully encapsulated, tested in a closed-loop setting, and error analysis and visualization tools were implemented for future experiments.

---

## What I Did Today

### 1. Built `ImitationController` as a standalone module
- Created `controller/imitation_controller.py`
- Unified interface: `__call__(t, pos, vel)`
- Optional parameters:
  - `clip=True`: ensures output stays in [-1, 1]
  - `verbose=False`: allows debug printing if needed
- Integrated into simulation as `thrust_vector`

### 2. Implemented full closed-loop test
- Created `main_imitation.py`
- Used Voyager parameters for long-range escape trajectory
- Ran simulation with imitation controller trained via MLPRegressor
- Saved trajectory as `.npy` for future reuse

### 3. Visualized controller behavior
- Plotted 2D orbit path using `plot_trajectory()`
- Plotted radius over time with `plot_radius_vs_time()`

### 4. Added radial error analysis function
- Appended `plot_radius_error()` to `orbit_analysis.py`
- Plots `r(t) - target_radius` over time
- Supports both interactive display and file saving with `save_path` parameter

### 5. Evaluated controller performance
- Mean radial error ≈ `4.72e+12 m`
- Std deviation ≈ `2.39e+12 m`
- Trajectory escaped solar gravity and did not stabilize — controller learned to thrust outward but failed to orbit or regulate

---

## ️ Files Created / Updated

- `controller/imitation_controller.py` ✅
- `main_imitation.py` ✅
- `simulator/orbit_analysis.py`: added `plot_radius_error()` ✅
- `data/logs/imitation_traj.npy` ✅
- `data/logs/imitation_error_curve.png` ✅

---

## Observations

- The imitation controller successfully runs in closed loop and outputs thrust continuously.
- However, the learned behavior is more like “constant acceleration” rather than “orbit correction”.
- This suggests expert demonstrations were likely biased toward escape-style thrust (radial only).
- MLP controller lacks feedback or convergence logic — not surprising without reward shaping.

---

## Issues Encountered Today

- The closed-loop simulation ran but resulted in **runaway trajectories**. The AI did not learn to stay in orbit.
- The radial error graph had extreme values at early timesteps — required normalization and axis tuning.
- A small typo (`acton` instead of `action`) caused confusion and a prediction crash early on.
- The class docstring in `ImitationController` was originally misindented, which caused it to appear outside the class.
- Needed to clarify `.reshape(1, -1)` and `np.clip()` usage from first principles to fully understand preprocessing and action output.

---

## Notes

- The current system is fully modular and compatible with future controllers (e.g., PPO, DQN).
- `verbose=True` in the imitation controller is useful for per-step debugging.
- The error analysis tool can now be reused across experiments for standardized evaluation.
