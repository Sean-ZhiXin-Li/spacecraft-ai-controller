
#  Project Log – Day 2: Thrust Controller and Orbit Evaluation

**Date:** 2025-07-17  
**Author:** Sean Li (Zhixin)  
**Project:** AI-Controlled Spacecraft Orbit Simulation  
**Day:** 2  
**Focus:** Dynamic thrust controller, simulation error evaluation, modularization

---

##  What I Did Today

### 1. Implemented Dynamic Thrust Controller
- Developed `velocity_direction_controller(t, pos, vel)` to apply thrust aligned with the velocity vector after 3 seconds.
- Ensures thrust is applied only after initial orbital stabilization.
- More realistic orbit expansion compared to static upward thrust.

### 2. Refactored Controller into a Separate Module
- Created `controller/velocity_controller.py`
- Made the code modular and scalable for testing multiple thrust strategies.
- In `main.py`, used:
  ```python
  from controller.velocity_controller import velocity_direction_controller
  ```

### 3. Saved Trajectory to Files
- Stored trajectory in both binary (`.npy`) and readable (`.csv`) formats:
  ```python
  np.save("data/saved_trajectories/main_traj.npy", main_traj)
  np.savetxt("data/saved_trajectories/main_traj.csv", main_traj, delimiter=",")
  ```
- Enables reuse for future plotting, comparison, or machine learning.

### 4. Added Orbit Error Evaluation Function
- Created `evaluate_orbit_error(trajectory, target_radius)`
- Calculates:
  - **Mean radial error**: average distance from desired orbit radius
  - **Standard deviation**: measures orbital stability
- Usage:
  ```python
  mean_error, std_error = evaluate_orbit_error(main_traj, target_radius)
  print(f"Mean radial error: {mean_error:.4f}, Std: {std_error:.4f}")
  ```

---

##  Results

- **Target radius:** 100.0  
- **Mean radial error:** `22.4969`  
- **Standard deviation:** `15.7424`

 The orbit spirals outward due to continuous thrust. However, the deviation is still large, indicating unstable radius. Further improvements to the controller are needed.

---

### Challenges Encountered

### 1. Initial Confusion About `thrust_vector` Logic
Although `thrust_vector=None` had default logic in `simulate_orbit`, early code mistakenly used:

```python
total_force = gravity_force + thrust_vector
```

This caused a `TypeError` when `thrust_vector` was `None`.

 **Fix:**  
Explicitly compute `thrust` using type checking:
```python
if callable(thrust_vector):
    thrust = thrust_vector(t, pos, vel)
elif isinstance(thrust_vector, np.ndarray):
    thrust = thrust_vector
else:
    thrust = np.array([0.0, 0.0])
```

---

### 2. Ineffective Thrust Direction
A static thrust of `[0.0, 0.002]` did not align with the velocity vector, causing unrealistic orbit distortion.

 **Fix:**  
Thrust was aligned with velocity direction:
```python
unit_direction = vel / np.linalg.norm(vel)
return 0.002 * unit_direction
```
This produced a more physically realistic orbit expansion.

---

### 3. Controller Modularity
Initially, the controller logic was embedded directly in `main.py`, making reuse and testing inconvenient.

 **Fix:**  
Moved logic to:
```
controller/velocity_controller.py
```
Improved code readability and reusability.

---

### 4. Unfamiliarity with Error Metrics
The concept of mean radial error and standard deviation was initially confusing. After studying the formulas:

- **Mean radial error**:  
  \[
  \mu = rac{1}{N} \sum_{i=1}^{N} |r_i - r_0|
  \]

- **Standard deviation**:  
  \[
  \sigma = \sqrt{ rac{1}{N} \sum_{i=1}^{N} (|r_i - r_0| - \mu)^2 }
  \]

 **Understanding:**  
- \( r_i \): Distance to origin at step i  
- \( r_0 \): Desired target orbit radius  
- Helps evaluate how accurate and smooth the orbit is.

---

##  Files Modified Today

```
main.py
simulator/simulate_orbit.py
simulator/visualize.py
controller/velocity_controller.py
analysis/evaluate_orbit_error.py
data/saved_trajectories/main_traj.npy
data/saved_trajectories/main_traj.csv
```

---

##  Next Steps (Day 3 Goals)

- Animate orbit evolution (GIF or live matplotlib)
- Plot radius vs. time for stability insight
- Experiment with pulse-based or periodic thrust controllers
- Simulate actuator noise or thrust limitations
- (Optional) Prototype a simple reinforcement learning thrust agent

---
