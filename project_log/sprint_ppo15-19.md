# Project Log (Day 15–19)

## Overview

This phase focused on stabilizing PPO-based spacecraft control behavior and diagnosing why the learned policy fails to achieve true orbital lock despite exhibiting stable long-horizon dynamics.

The key transition in this stage was moving away from reward engineering as the primary lever, and instead shifting toward:

* behavior-based checkpoint selection
* deterministic evaluation
* failure mode diagnosis (control collapse, drift bias)

---

## Day 15 — Stability Domain & Behavior Characterization

* Completed stability-domain style evaluation of controllers
* Compared multiple controller variants under identical physics
* Identified that PPO can achieve **long survival (20,000 steps)** but not orbit lock

Key observations:

* PPO shows **stable but biased trajectories**
* Radius does not converge to target
* Behavior resembles **drift stabilization rather than closed-loop control**

Insight:

> Stability ≠ Control correctness

---

## Day 16 — Mechanism-Level Understanding

Shifted focus from results to mechanisms.

Analyzed rollout metrics:

* radius vs time
* radial velocity (v_r)
* thrust alignment

Findings:

* v_r reduces quickly but **does not oscillate around zero**
* policy minimizes radial motion but does not correct residual bias
* no sign change in v_r → no closed-loop behavior

Conclusion:

> The policy learns to "stop radial motion" instead of "lock orbit"

---

## Day 17 — PPO Integration & Diagnostic Metrics

Integrated PPO fully into the unified evaluation pipeline.

Added structured analysis:

* tail_mean_abs_vr
* final_radius_error
* tail_avg_radius_error
* alignment metrics

Established a **behavior-based evaluation standard**.

Key realization:

> Trainer-selected checkpoints (best_mean_hold) are not behavior-optimal

---

## Day 18 — Reward Experiments (Failure Phase)

Explored multiple reward modifications:

* directional radius correction
* gated radius penalties
* speed-refine terms
* anti-coasting penalties
* keep-alive terms

Results:

* Most modifications **destroyed stability**
* Some improved local behavior but caused:

  * oscillation explosion
  * alignment degradation
  * survival drop

Critical insight:

> Adding more reward terms does not fix structural policy issues

Decision:

> Stop reward search

---

## Day 19 — Behavior-Based Checkpoint Selection & Failure Diagnosis

### 1. Behavior-Based Checkpoint Selection

Replaced trainer-based selection with composite metrics:

Priority:

1. survival (must be ~20000)
2. tail_mean_abs_vr ↓
3. final_radius_error ↓
4. tail_avg_radius_error ↓
5. alignment (low)

Selected checkpoint:

```
ppo_orbit/speed_refine_50/ppo_epoch_300.pth
```

---

### 2. Deterministic Evaluation (Deployment Mode)

Changed evaluation to use:

* policy mean (mu)
* no sampling

Results:

* behavior becomes fully deterministic
* rollout-to-rollout variance → 0
* lower v_r compared to stochastic evaluation
* improved alignment and smoothness

Conclusion:

> Stochasticity was masking true policy behavior

---

### 3. Multi-Initial Condition Testing

Tested controller under:

* r0 = 0.98
* r0 = 1.00
* r0 = 1.02
* r0 = 1.05

Findings:

* full survival in all cases
* consistent directional drift:

  * below target → outward drift
  * above target → inward drift
* **no symmetric oscillation**

Conclusion:

> Policy is a drift corrector, not an orbit controller

---

### 4. Action Collapse Diagnosis

Performed episode-level analysis of action behavior.

Detected collapse condition:

* action_norm → ~0 at step ~90–100

At collapse boundary:

* v_r is small but non-zero
* v_t error is still large
* radius error not resolved

Post-collapse:

* policy stops acting
* system drifts passively

Key insight:

> Policy interprets "low radial velocity" as task completion

---

### 5. Root Cause

The learned implicit rule:

```
if |v_r| is small → stop acting
```

Missing condition:

```
and orbit is actually correct
```

Result:

* premature shutdown
* no closed-loop correction

---

## Day 15-19 Milestone Conclusion

The PPO controller achieves:

* long-horizon stability
* deterministic repeatable behavior
* correct drift direction

But fails at:

* orbit locking
* symmetric error correction
* sustained feedback control

---

## Key Insight

> The failure is NOT due to reward magnitude,
> but due to missing state information and incorrect implicit stopping logic.

---

## Next Direction

Instead of adding reward terms:

* improve state representation (e.g., explicit v_r)
* prevent premature shutdown
* enforce true closed-loop behavior

---

## Philosophy

> The system has learned when to stop moving,
> but not when it has actually arrived.
