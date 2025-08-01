# Project Log – Day 17
 Title: Expert Dataset Generation with Tuned Insertion Profile

 Date: 2025-08-01  
 Artifacts: expert_dataset_01.npy to expert_dataset_30.npy, expert_controller.py, generate_dataset.py
 Controller: ExpertController  
 Angle range: -30° to +30°

---

##  Summary

Today I successfully generated a complete set of 30 expert datasets using a refined version of the expert controller. These trajectories simulate spacecraft insertion from a boosted suborbital starting point to a large circular orbit.

Each dataset captures state-action pairs from the controller, and will be used for training imitation learning models starting tomorrow.

---

## ️ Simulation Setup

- **Central Mass**: Sun  
- **G**: 6.67430e-11  
- **M**: 1.989e30  
- **Target Orbit Radius**: 7.5e12 m  
- **Initial Position**: [0, 2.5e12]  
- **Initial Speed**: 1.2 × circular velocity at r₀  
- **Initial Angle**: −30° to +30°, evenly spaced (N = 30)  
- **Time Step**: 2000 seconds  
- **Total Steps**: 1,200,000  
- **Mass**: 721.9 kg (Voyager)

---

##  Expert Controller Settings

```python
ExpertController(
    target_radius = 7.5e12,
    G = 6.67430e-11,
    M = 1.989e30,
    mass = 722,
    radial_gain = 4.0,
    tangential_gain = 5.0,
    damping_gain = 6.0,
    thrust_limit = 20.0,
    enable_damping = True
)
```

---

##  Output

Each dataset file includes:
- State vectors: [x, y, vx, vy]
- Action vectors: [thrust_x, thrust_y]

Saved in:
```
data/dataset/expert_dataset_01.npy
...
data/dataset/expert_dataset_30.npy
```

---

##  Issues Encountered

Before finalizing the expert controller configuration, I tested multiple gain and thrust limit values.  
- Low `thrust_limit` (<10) caused slow convergence  
- High gains introduced severe oscillations  
- Boost factor >1.2 caused overshoots and unstable trajectories

After tuning:
- Stable orbit insertion was achieved within 20–25 simulated days  
- Damping gain = 6.0 proved critical for suppressing post-insertion oscillations  
- Final parameters produced smooth, accurate and efficient captures

This marks a **major milestone** in the project: the entire expert dataset is now ready and validated.

---

 Tomorrow begins imitation learning. Let’s teach the AI to fly.
