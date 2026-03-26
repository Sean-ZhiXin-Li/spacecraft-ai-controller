## Day 1 – PPO Integration: Fixing Scaling and Pipeline Consistency

### Objective

The goal of today was to integrate PPO into the existing W03 control pipeline without creating a separate system. The focus was on ensuring that PPO behaves as a new controller under the same action interface, enabling fair comparison with existing controllers (always-on and gated).

---

### Problem Identified

During initial integration, PPO outputs were incorrectly interpreted due to a **scaling mismatch**:

* PPO policy outputs actions in the range **[-1, 1]**
* The W03 pipeline expects **thrust in physical units (Newtons)**
* As a result, PPO actions were:

  * Treated as thrust
  * Then re-normalized again by `thrust_to_action`

This led to **double scaling**, causing:

* Extremely small effective control signals (~1e-3)
* Near-zero actuation in the environment
* Misleading evaluation results

---

### Fix Implemented

#### 1. Enforced Single Scaling Rule

Ensured that scaling occurs **exactly once** in the pipeline:

```python
# PPO returns normalized action → convert to thrust
if label.lower() == "ppo":
    thrust_intent = thrust_intent * REF_THRUST_SCALE
```

This converts PPO output into the same unit (Newtons) as other controllers.

---

#### 2. Preserved Unified Action Interface

All controllers now follow:

```text
Controller → thrust_intent (N) → thrust_to_action → env.step(action)
```

* Expert / Gated: directly output thrust (N)
* PPO: outputs normalized action → converted to thrust in runner

No changes were made to:

* `thrust_to_action`
* environment dynamics
* existing controllers

---

#### 3. Removed Implicit Scaling from PPO Controller

The PPO controller was corrected to:

* Output only normalized action in [-1, 1]
* Avoid any internal multiplication (e.g., `*1000`)

This prevents hidden scaling bugs and keeps the controller clean.

---

### Result

After the fix:

* PPO actions produce **non-trivial thrust magnitudes**
* `raw_norm_mean` is now in a realistic range
* `saturation_rate_mean` is no longer near zero
* PPO behavior becomes observable in rollout

---

### Key Insight

> PPO is not a separate system—it must conform to the same physical interface as existing controllers.

This required explicitly **lifting PPO outputs into the thrust domain** before passing through the shared pipeline.

---

### Status

 Scaling bug resolved
 PPO compatible with W03 pipeline
 Ready for controller unification (Day 2)

---

### Next Step (Day 2)

* Refactor PPO controller to match unified interface:

  ```python
  controller.act(obs)
  ```
* Register PPO alongside existing controllers
* Enable seamless switching in the runner

---

### Reflection

This step was critical: without fixing the scaling issue, any comparison involving PPO would be invalid. The correction ensures that all controllers operate under the same physical constraints, making future analysis meaningful.
