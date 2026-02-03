# Project Log — WHPL_02 (Day 2)

**Date**: 2026-02-03  
**Focus**: Action Interface Compression Diagnosis  
**Status**: Completed (Evidence Collected, No Design Changes)

---

## Objective

Confirm whether the current **action interface** is compressing differences between
`ExpertV3` and `ExpertImproved`, rather than differences being absent at the controller level.

Key question:

> Are controller-level differences being erased by action normalization / clipping,
> or are the controllers already equivalent under current conditions?

---

## Scope & Constraints (Respected)

- ❌ No changes to `OrbitEnv`
- ❌ No changes to controller logic
- ❌ No reward modification
- ❌ No RL / learning logic
- ✅ Only diagnostic logging and controlled comparisons
- ✅ Single-variable, reproducible runs

---

## Experiments Run

### 1. Baseline (Default Scenario)

Command:
```bash
python src/quick_compare_v3_v4.py --no_sweep --thrust 200 --scenario default
```

**Observations**:
- `shadow_step` confirms action → dynamics coupling is active
- `thrust_intent_norm_mean` identical for V3 and Improved
- `thrust_intent_dir_var` ~1e-13 for both
- `action_norm_mean` identical
- Orbital metrics (`r`, energy, angular momentum) evolve identically
- Rewards and final states identical to print precision

**Conclusion**:  
No evidence of action interface compression.  
Controllers are behaviorally equivalent under default initial conditions.

---

### 2. Weak Thrust, Far Orbit Scenario

Command:
```bash
python src/quick_compare_v3_v4.py --no_sweep --thrust 200 --scenario weak_thrust_far
```

**Validation Checks**:
- `phys={'thrust_newton': 200.0}` confirms physical override is applied
- `shadow_step` shows large state divergence vs zero action
- Action clearly affects dynamics (not masked)

**Key Statistics**:

| Metric | ExpertV3 | ExpertImproved |
|------|---------|----------------|
| thrust_intent_norm_mean | ~6.19999 | ~6.19991 |
| thrust_intent_dir_var | ~4.7e-10 | ~4.7e-10 |
| action_norm_mean | ~0.619999 | ~0.619991 |
| total_reward | identical | identical |
| final_r | identical | identical |

Differences are within floating-point noise.

**Conclusion**:  
Even under stressed (weak thrust) conditions, both controllers produce
effectively identical thrust intents and actions.

---

## Resulting Conclusion (WHPL_02)

**Evidence strongly suggests**:

- The action interface is **not** compressing controller diversity
- Physical overrides are correctly applied
- Actions influence dynamics as expected
- `ExpertV3` and `ExpertImproved` are **functionally equivalent** in the tested state space

Therefore:

> Any intended improvements in `ExpertImproved` are not being triggered
> by the current initial conditions or state trajectory.

---

## Next Step (Deferred to WHPL_03)

Not a logging or environment issue.

Next diagnostic layer should move **inside controller logic**:
- Instrument early-step branch decisions in `act()`
- Log which phase / mode / conditionals are activated
- Identify why improved logic is never entered

No further runs are required today.

---

## End of Day Status

- WHPL_02 objectives met
- Diagnostics complete
- Safe to freeze Day 2 and transition to controller-level analysis next session
