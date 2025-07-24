# Project Log Day 9: Controller Comparison + Error Analysis

**Date:** 2025-07-24
**Title:** Imitation vs Expert Controller – Performance Comparison + Enhanced Error Analysis

---

## Objective

The objective today was to **compare the orbital performance** of the learned `ImitationController` against the handcrafted `ExpertController`, and to **deeply analyze the radial error** of the AI-controlled trajectory.

---

## Methods

- Simulated two trajectories under:
  - `ExpertController` (rule-based)
  - `ImitationController` (V2 MLP model)
- Used identical initial conditions and parameters.
- Evaluated via:
  - Trajectory shape
  - Radius over time `r(t)`
  - Radial error `r(t) - target_radius`
  - Mean, Std, Max error
  - Histogram + Moving Average + Max Error analysis

---

## Results

### Trajectory Comparison

![Trajectory](../data/logs/imitation_traj.png)

- **Blue**: Imitation Controller  
- **Orange dashed**: Expert Controller  
- **Gray Circle**: Target Orbit

---

###  r(t) vs Time

![r(t)](../data/logs/comparison_radius.png)

- Imitation controller (red) shows steady increase but fails to reach target.
- Expert controller (blue) climbs faster and overshoots.

---

###  Radial Error vs Time

![Error](../data/logs/comparison_error.png)

- Red = Imitation Error  
- Blue = Expert Error  
- Shows how close the spacecraft stayed near target radius

---

###  Enhanced Radial Error Analysis

![Enhanced Error](../data/logs/enhanced_error_imitation.png)

- Highlighted **max error**: `7.35e+12 m`  
- Moving average (window = 1000) for error smoothing  
- Demonstrates stable error slope

---

### Error Histogram

![Histogram](../data/logs/error_hist_imitation.png)

- Distribution of **absolute radial error**  
- Majority of values between 4e12 – 7e12 m  
- Peak near `7.35e+12 m`

---

## Quantitative Summary

| Metric                  | Imitation Controller | Expert Controller |
|-------------------------|----------------------|-------------------|
| Mean Radial Error (m)   | **4.84 × 10¹²**       | **4.45 × 10¹²**    |
| Std Radial Error (m)    | 1.59 × 10¹²           | 2.14 × 10¹²        |
| Max Absolute Error (m)  | **7.35 × 10¹²**       | *[Not evaluated]*  |

---

## Observations

- Imitation model is **stable but conservative**.
- Expert controller overshoots target but reaches orbit faster.
- Standard deviation of error is **lower for imitation**, indicating smoother control.

---

## Files Generated

- `compare_controllers.py`
- `comparison_radius.png`
- `comparison_error.png`
- `enhanced_error_imitation.png`
- `error_hist_imitation.png`
- `project_log_9.md`

---
