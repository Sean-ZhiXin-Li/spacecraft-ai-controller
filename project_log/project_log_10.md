# Project Log – Day 10

**Date:** 2025-07-25  
**Project:** AI Thruster Control – Imitation Controller (V3 vs V3.1)

---

## Today’s Progress

1. **Trained and Compared Two Imitation Controllers**
   -  Trained **V3** model with high deviation from the target orbit:  
     `Mean radial error: 6.06e+12 m, Std: 2.77e+12 m`  
     The spacecraft failed to stay in orbit and drifted outward rapidly.
   -  Trained **V3.1** model with improved performance:  
     `Test MSE: 1.79e-1`  
     `Mean radial error: 5.06e+12 m, Std: 1.47e+12 m`  
     Demonstrated better trajectory stabilization and closer matching with expert control.

2. **Generated Orbit Analysis Visualizations**
   - Plotted and saved the following figures:
     - `r(t)` curve over time
     - Orbit trajectory with target circle overlay
     - **Enhanced radial error plot** (`enhanced_error_v3.1.png`)
     - **Radial error histogram** (`error_hist_v3.1.png`)
     - **Thrust field quiver plot** (`thrust_field_v3.1.png`)

3. **Training Monitoring**
   - Logged and analyzed each epoch’s `loss` and `validation score` for V3.1;
   - Observed large initial losses converging toward stability after ~15 iterations.

4. **Implemented Thrust Vector Field Visualization**
   - Successfully plotted thrust vectors using `plot_thrust_quiver()` function;
   - Tuned `scale`, `sampling step`, and `vector spacing` to make the arrows visible and interpretable.

---

## Issues Encountered

- **V3 model was ineffective**
  - Output trajectory drifted significantly from target radius.

- **Training was time-consuming**
  - V3.1 took several minutes due to large dataset size and deep MLP structure.

- **Data dimension mismatch**
  - Reshaping `.npy` file raised `ValueError: cannot reshape array of size 360000 into shape (7)`; root cause: time column was not recorded in `ThrustDataset`.

- **Missing file error**
  - Initial attempt to load `data/logs/thrust_log.npy` failed because the actual saved file was in `data/dataset/`.

- **Empty or distorted quiver plots**
  - Plot appeared empty when `scale` was too large or sampling `step` too high;
  - Adjustments were required to ensure proper arrow visibility.

---

## Artifacts Produced

- `imitation_policy_model_v3.1.joblib`
- `enhanced_error_v3.1.png`
- `error_hist_v3.1.png`
- `thrust_field_v3.1.png`
- `radial_decay_with_noise.npy/.csv`
- `radial_noise_decay.npy/.csv`
- `saved_trajectories/expert_traj.npy/.csv`

---

> All related files have been committed to GitHub and are ready for further analysis or reuse.
