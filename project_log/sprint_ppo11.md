# Project Log — Day X (PPO Orbit Control)

## Overview
Today’s goal was to improve the PPO-based orbital control system so that the agent can move from unstable radial motion toward stable orbital behavior. The focus was on fixing reward shaping, stabilizing training, and introducing physical constraints (especially angular momentum).

---

## What Was Done

### 1. Reward Function Improvements
- Added **angular momentum term (h_term)** to encourage real orbital motion instead of straight-line drifting.
- Introduced **overspeed penalties** to prevent the agent from exploiting reward by continuously accelerating.
- Split reward into:
  - **Approach phase** (reach target radius)
  - **Stabilization phase** (maintain orbit)
- Used a **smooth blending (w)** between phases instead of a hard switch.

---

### 2. Physics Consistency Enhancements
- Ensured that thrust is applied **before integration**, so it actually affects motion.
- Added a **minimum tangential thrust fallback** to help the agent escape purely radial motion.

---

### 3. PPO Training Stability Fixes
- Fixed rollout reward clipping and logging issues.
- Ensured correct use of:
  - raw actions (for PPO ratio)
  - clipped actions (for environment)
- Verified that training loop runs without NaNs or shape issues.

---

### 4. Observed Training Behavior

#### Trajectory
- The trajectory changed from a **straight line → piecewise (step-like) path**.
- Indicates that the agent is starting to adjust direction, but control is still discontinuous.

#### Alignment (cos angle)
- Decreased from ~1 toward ~0.25 over time.
- This shows the agent is beginning to move from **radial → tangential motion**.

#### Radius
- Changes in a step-like pattern rather than smoothly.
- Suggests unstable or discontinuous control policy.

#### Speed
- Continues increasing over time.
- Indicates the agent is still exploiting reward via acceleration.

#### Reward Curve
- Highly unstable with large fluctuations.
- Training has not converged yet.

---

## Key Problems Identified

### 1. Velocity Explosion (Critical)
- Angular momentum reward unintentionally encourages higher speed.
- Agent exploits reward by accelerating instead of forming orbit.

### 2. Discontinuous Control
- Policy outputs vary sharply between steps.
- Leads to **zig-zag / staircase trajectory** instead of smooth orbit.

### 3. Reward Exploitation
- Agent prioritizes:
  - maintaining radius
  - increasing speed
- Instead of achieving stable circular motion.

---

## Insights

- The system has moved from:
  - ❌ “no control” (straight escape)
  - ➝ ⚠️ “partial control” (piecewise turning)
- This is a **transitional phase** where the agent begins to learn orbital structure.

- The biggest missing constraint is:
  > Proper coupling between **angular momentum and velocity magnitude**

---

## Next Steps (Planned)

1. **Fix angular momentum reward**
   - Clip or normalize to prevent speed exploitation.

2. **Strengthen speed penalty**
   - Penalize deviation from target orbital velocity more aggressively.

3. **Add action smoothness penalty**
   - Reduce sudden changes in thrust direction.

4. **Reduce minimum thrust fallback**
   - Gradually remove artificial assistance.

5. **Goal for next iteration**
   - Transition from “piecewise path” → **smooth curved trajectory**
   - Achieve stable tangential motion (cos ≈ 0)

---

## Conclusion

Today’s progress marks a significant step forward.  
The agent is no longer purely drifting and has started to exhibit directional control.

However, true orbital behavior has not yet emerged due to:
- reward imbalance
- excessive acceleration
- lack of smooth control

The system is now in a **critical refinement stage**, where correct reward shaping will determine whether stable orbit learning is achieved.

---