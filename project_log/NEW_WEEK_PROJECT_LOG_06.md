# NEW_WEEK_PROJECT_LOG_06 — Project Log  
**Project:** Spacecraft AI Propulsion Control  
**Focus:** Verification & Maintenance Week  
**Status:** Light update (USACO + TOEFL priority)

---

## Summary

Week 6 was intentionally kept **lightweight**, with no major feature additions or architectural changes.  
Due to ongoing **USACO preparation** and **TOEFL exam training**, development effort was limited to **verification, diagnostics, and maintenance**, similar in scope to Week 5.

The primary goal of this week was to **confirm earlier conclusions** about controller behavior and environment constraints, rather than to push new functionality.

---

## Work Completed

### 1. Action Saturation Verification (Follow-up from Week 5)

- Confirmed that **action clipping at the environment boundary** is the dominant factor in several scenarios.
- Measured **action saturation rate** (`saturation_rate`) across scenarios:
  - `weak_thrust_far`: ~0.37–0.38
  - Other scenarios (`default`, `misaligned_entry`, `oscillation_noise`): ~0.10
- Verified via `thrust_vec` logs that thrust commands frequently hit physical limits (e.g. `[-800, -800] N`).

**Conclusion:**  
Differences between `ExpertV3` and `ExpertImproved` are **masked at the environment interface** when actions saturate, explaining nearly identical trajectory-level metrics.

---

### 2. Shadow-Step Sanity Check (No New Logic Added)

- Used a **shadow-step diagnostic**:
  - Compared `step(action)` vs `step(0)` from the same state.
- Observed clear differences in:
  - Next observation
  - Reward
  - Thrust-related diagnostic outputs

**Conclusion:**  
The environment **does respond to actions correctly**; the issue is not a missing action hook but **over-saturation**.

---

### 3. Logging & Code Hygiene

- No changes to core dynamics or controller logic.
- No new reward shaping or physics parameters added.
- Logging output was reviewed for consistency and clarity.
- Week 6 intentionally avoids introducing new experimental variables.

---

## What Was *Not* Done (By Design)

- No controller redesign.
- No new scenarios or noise models.
- No reward function changes.
- No hyperparameter tuning.
- No training or learning-based updates.

This was a **maintenance and validation week**, not an expansion week.

---

## Rationale for Limited Updates

During Week 6, priority was given to:
- **USACO contest preparation**
- **TOEFL exam training**

Given the time constraints, the decision was made to:
- Keep project work aligned with Week 5
- Avoid partial or rushed feature additions
- Preserve experimental stability

---

## Current Project State

- Controller behavior is **well-understood** under saturation conditions.
- Environment dynamics and action pathways are **verified and functioning**.
- The system is ready for future work in **non-saturated regimes** or **controller–environment interface refinement**.

---

**End of Week 6 Log**
