# Project Log Day 8 – Imitation Controller Evaluation & Visualization Fix

## Date
2025-07-23

## Today’s Progress

### 1. Model Evaluation & Comparison (V1 vs V2)
- Successfully ran the V2 model `imitation_policy_model_V2.joblib`.
- Output mean radial error and standard deviation:
  - **V2**: `Mean radial error: 4.84e+12 m, Std: 1.59e+12 m`
  - **V1**: `Mean radial error: 4.72e+12 m, Std: 2.39e+12 m`
- Analysis:
  - While V2 has a slightly higher mean error, its **standard deviation is significantly lower** → more stable thrust behavior.
  - Visual comparison shows V2 trajectory is more stable and doesn’t significantly drift from the Sun.

---

### 2. Visualization Module Fixes & Enhancements
- Fixed the issue where the target orbit circle didn’t display due to insufficient axis limits.
- Implemented dynamic plot bounds logic:
  ```python
  max_radius = np.max(np.linalg.norm(trajectory, axis=1))
  if target_radius is not None:
      max_radius = max(max_radius, target_radius)
  buffer = 0.2 * max_radius
  ax.set_xlim(-max_radius - buffer, max_radius + buffer)
  ax.set_ylim(-max_radius - buffer, max_radius + buffer)
  ```
- Improved `plot_radius_vs_time()` to show consistent growth in radial distance clearly.

---

## Problems Encountered

1. **Target orbit not visible**
   - Cause: target orbit exceeded static axis limits.
   - Fix: made plot bounds automatically expand to include `target_radius`.

2. **ImportError: Cannot import `plot_radius_vs_time`**
   - Cause: the function was mistakenly defined inside another function.
   - Fix: moved `plot_radius_vs_time()` outside and it imported correctly.

3. **Slow convergence in MLPRegressor**
   - Studied the effect of `max_iter`, activation functions, and error convergence.
   - Gained better understanding of model iteration and training limits.

4. **Thrust vector visualization mismatch**
   - Likely due to uneven distribution in training data.
   - Will consider improving expert dataset sampling strategy in the future.

---

## ️ Summary

-  Completed error-based V1/V2 comparison.
-  All visualization bugs fixed and plots restored.
-  Evaluated model quality using both mean and standard deviation.
-  Gained hands-on experience with practical debugging: ImportErrors, visibility bugs, and neural training issues.
