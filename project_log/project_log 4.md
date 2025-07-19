#  Project Log – Day 4

 **Date**: 2025-07-19  
 **Author**: Zhixin Li (Sean)  
 **Log ID**: Day 4  
 **Location**: AI Spacecraft Propulsion Simulation

---

##  Summary of Today’s Progress

### 1. Added Attitude Noise Simulation
- Implemented `add_attitude_noise()` to introduce thrust direction error.
- Simulates real-world spacecraft actuator/attitude imprecision.
- Controlled by `max_angle_deg`, applied randomly per thrust command.
- Verified effect by comparing perturbed vs. clean trajectories.
- Integrated with controller using `add_noise=True`.

### 2. Refactored to Class-Based `Controller`
- Replaced loose functions with a configurable `Controller` class.
- All thrust logic (radial, tangential, impulse, decay, noise) now unified.
- Supports `__call__()` method for easy lambda use.
- Parameterized via `__init__()` with clean configuration interface.

### 3. Added Thrust Logging Support
- Enabled recording of:
  - `time`, `position`, `velocity`, `thrust`
- Automatically appends to `self.log` if `enable_logging=True`.
- Implemented `save_log()` to export both `.npy` and `.csv` formats to `data/logs/`.

### 4. Enabled Batch Simulation
- Wrote utility to run multiple simulations with varied modes (e.g., tangential-only, noisy, decaying).
- Outputs are saved with unique `name` prefix (e.g., `tangential_decay`, `radial_noise`, etc.).
- Supports sweeping through hyperparameters (alpha/beta, decay, etc.).

### 5. Created `ThrustDataset` Class
- Designed to collect (pos, vel, thrust) data per timestep.
- Useful for future ML model training.
- Supports `add()` and `save()` methods.
- Implements `__call__()` to integrate directly with simulation loop.

---

## Code Modules Affected

- `controller/combined_controller.py`
- `controller/perturb.py`
- `controller/velocity_controller.py`
- `visualize/plot.py`
- `data/thrust_dataset.py`
- `main.py`

---

## Demo Config (Example)

```python
controller = Controller(
    impulse=True,
    enable_radial=True,
    enable_tangential=True,
    alpha=17,
    beta=17,
    thrust_decay_type='exponential',
    decay_rate=1e-7,
    add_noise=True,
    noise_deg=10,
    enable_logging=True
)
```

---

 *All code changes committed to GitHub with tag `Day 4`.*


## Problems Encountered Today
- Attitude noise (±3° to ±10°) had little visible effect on the orbit at first, requiring us to increase the noise level and add visual comparisons.
- `os.makedirs` raised an error due to mistaken import from numpy instead of Python standard library.
- Logging and dataset saving logic had to be carefully placed to avoid performance or logic issues.
- Understanding object-oriented principles like `class`, `self`, and `__init__` required foundational explanations.
- It was initially unclear why encapsulating into a class was better than using standalone functions.
