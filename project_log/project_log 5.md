#  Project Log Day 5 - Expert Controller + RL Environment

##  Date
2025-07-20

##  Today’s Progress

### 1. Expert Controller Implemented
- Completed `ExpertController` class:
  - Corrects both radial error and adds tangential thrust for orbital motion.
  - Caps the thrust magnitude to ensure physical realism.
- Successfully integrated with `ThrustDataset` to collect expert demonstration data.

### 2. Dataset Generation Script Completed
- Created `generate_dataset.py` that generates 10 variations of initial velocity angles.
- Used lambda wrapping to log controller behavior:
  ```python
  lambda t, pos, vel: dataset(t, pos, vel, controller)
  ```
- Saved `.npy` and `.csv` datasets to `data/dataset/` as `expert_dataset_01.npy`, ..., `expert_dataset_10.npy`.

### 3. RL Environment: OrbitEnv
- Designed and implemented a custom `OrbitEnv` class based on OpenAI Gym API.
- Supports `reset()` and `step(action)` interfaces compatible with RL libraries (DQN, PPO, etc.).
- Models 2D orbital physics:
  - Gravitational pull
  - Agent-applied thrust (action)
  - Position and velocity updates
  - Reward: negative radial error from target orbit
- Verified correctness using a standalone `test.py` script.

### 4. AI-Assisted Understanding
- Today’s environment code was co-designed with AI assistance.
- I went through every line of code and now fully understand each part’s function and purpose.

---

##  Issues Faced and Solutions

### 1. `main.py` Unintentionally Triggered
- Initially imported `vel_init` from `main.py`, which caused full simulation + plots to run.
-  Fixed by removing unnecessary imports and redefining local variables inside `generate_dataset.py`.

### 2. Gym Interface Confusion
- Adjusted to Gym’s new `reset()` signature which returns `(obs, info)`.
- Ensured `step()` returns full `(obs, reward, done, info)` tuple as required.
-  Followed Gym API documentation closely to resolve.

### 3. Simplistic Reward Function
- Current reward: `-abs(r - target_r) / target_r`.
- Too simple for meaningful learning; will explore reward shaping in future versions.

---

##  Additional Reflections

- Learned how to structure a proper Gym environment and RL-compatible interface.
- Identified that single-target orbit is not challenging enough — AI doesn’t show clear advantage yet.
- Planning to introduce:
  - Multi-phase transfer orbits
  - Fuel constraints
  - Partial observability
  - Robust reward shaping
- Baseline controller and agent integration (e.g., PPO) will be important next steps.
