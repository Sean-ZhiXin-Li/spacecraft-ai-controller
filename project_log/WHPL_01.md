# WHPL_01 — Winter Holiday Project Log 01  
**Date:** 2026-02-02  
**Status:** System Unfreeze · Action–Thrust Decoupling Verified

---

## Context

This log marks the restart of the spacecraft AI control project after a long interruption caused by:
- Final exams
- USACO season competition
- Winter holiday schedule shift

The goal of today was **not performance improvement**, but to verify whether the control → action → physics pipeline was *structurally alive* and *correctly parameterized*.

This log serves as the **first checkpoint of the Winter Holiday phase**.

---

## Primary Objective (Today)

> Verify that thrust magnitude in Newtons is **no longer silently canceled** by the action normalization / clipping pipeline, and that physical thrust strength genuinely affects acceleration and dynamics.

This is a **Day-0 correctness task**, not an optimization task.

---

## Key Changes Introduced

### 1. Action–Thrust Decoupling

Previously, thrust magnitude changes were largely neutralized due to:
- Action normalization coupled to `env.thrust_scale`
- Implicit dependence between controller output scale and environment actuator scale

**Fix applied:**
```python
REF_THRUST_SCALE = 3000.0
action, ainfo = thrust_to_action(thrust_intent, REF_THRUST_SCALE, cfg_ai)
```

This decouples:
- Controller intent (Newtons)
- Action normalization
- Environment actuator strength

Result:  
**Changing thrust strength now produces proportional physical acceleration changes.**

---

### 2. Environment Variable Clarification

Resolved a conceptual confusion around:

```bash
$env:THRUST_NEWTON
```

Clarified and enforced:
- `THRUST_NEWTON` is **environment-side physics only**
- It should **not** influence action normalization
- It is treated as a single-run physical override, not a learning signal

This removed a hidden mental and technical coupling.

---

## Experimental Verification

### Test Setup

- Scenarios tested:
  - `weak_thrust_far`
  - `oscillation_noise` (fallback)
  - `misaligned_entry`
  - `default`
- Controllers:
  - ExpertV3
  - ExpertImproved
- Runs compared under:
  - `THRUST_NEWTON = 200`
  - `THRUST_NEWTON = 2000`

---

### Observed Results

#### Physical Scaling (Key Result)

| Thrust (N) | a_eff (m/s²) | dv_eff (m/s) |
|-----------:|-------------:|-------------:|
| 200        | ~1.87e-01    | ~1.87        |
| 2000       | ~1.87e+00    | ~18.7        |

- Acceleration scales **linearly** with thrust
- No evidence of action clipping cancellation
- `clip_norm_mean` remains stable and interpretable

This confirms the pipeline is **physically responsive**.

---

### Saturation & Stability

- `saturation_rate_mean ≈ 0.000`
- No hidden saturation or silent clipping
- Jitter remains numerically negligible (`~1e-10` to `1e-13`)

The system is **stable, responsive, and correctly wired**.

---

## What Did *Not* Change (Intentionally)

- No reward redesign
- No controller logic modification
- No gain tuning
- No PPO / learning integration

This day was strictly **structural verification**, not behavioral improvement.

---

## Final Assessment

✅ Action normalization is no longer canceling thrust magnitude  
✅ Physical thrust strength correctly affects dynamics  
✅ Metrics (`raw_norm`, `clip_norm`, `a_eff`) are now meaningful  
✅ System is safe to build learning on top of

This concludes the **unfreeze phase**.

---

## Next Logical Step (Deferred)

- Introduce learning (imitation or RL) **only after** locking this interface
- Freeze action interface contract
- Promote this setup as the baseline for Phase B training

No further changes today.

---

*End of WHPL_01*
