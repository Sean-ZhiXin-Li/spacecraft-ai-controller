# Day 16 – Back to Expert v3.1: Circular Insertion Revisited

## Context
Today I reverted from PPO experiments back to a physically realistic and interpretable controller — **ExpertController v3.1**. This version is designed to perform clean, non-spiral orbit insertion based on radial + tangential thrust logic with damping and shutdown logic.

The goal: create a controller that uses physical rules — not learned policies — to achieve stable circular orbit capture at Voyager-1–scale distances.

---

## Implementation Notes

### ExpertController v3.1
- **Tangential control:** Controls speed along orbital direction to reach circular velocity.
- **Radial control:** Brings spacecraft toward the target radius.
- **Damping:** Reduces radial oscillation using radial velocity.
- **Shutdown logic:** If `|r - r_target| < 1%` and `|v_t - v_circular| < 1%`, set thrust = 0.

### Parameters used:
- `target_radius = 7.5e12 m` (Voyager-1 scale)
- `mass = 1000 kg`  (manually corrected to Voyager probe scale)
- `radial_gain = 12.0`
- `tangential_gain = 8.0`
- `damping_gain = 4.0`
- `thrust_limit = 1.0`
- `enable_damping = True`

---

## Results

### Old controller (failure case)
- Radius over time `r(t)` shows large oscillation from 0.25e13 to 1.25e13.
- No stable orbit achieved — appears sinusoidal.
- Orbit path is elongated and non-circular.

### Expert v3.1 result:
- **Radius vs Time** plot: Flat radius at ~5.95e12 m — stable!
- **Trajectory plot:** Near-perfect circular orbit around the Sun.
- **Thrust stopped after capture**, verifying the shutdown condition works.

### Screenshots saved:
- `radius_vs_time.png` (flat)
- `orbit_debug_plot.png` (circle)
- Marked difference compared to oscillating previous controller.

---

## Issues Encountered

- Initial confusion about Voyager mass: fixed to `1000 kg` instead of unrealistic defaults (like `1e7 kg`).
- Old plots still showed spiral or broken orbits — but the new one works cleanly.
- Had to double-check the shutdown logic — turns out both `radial_error` and `delta_v` thresholds matter.

---

## Reflections

Returning to a physics-based controller gave me control, interpretability, and realistic behavior — everything PPO failed to deliver with my current training curve.

The orbital match with Voyager-scale radius now feels **correct in both dynamics and units**.

-
