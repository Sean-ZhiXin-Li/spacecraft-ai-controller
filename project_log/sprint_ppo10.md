# Day 10 — PPO Orbit Control Debug

## Key Takeaways (Day 9)

The current issue is **NOT reward design**, but:

> Policy collapse (the agent stopped exploring)

Observed behaviors:

* Thrust output ≈ 0 (agent does almost nothing)
* Trajectory = near free-fall
* Alignment stuck at ~0.68
* Radius monotonically decreasing

 Root cause: agent found a local optimum — *"do nothing is safest"*

---

# Day 10 Objective

> Break policy collapse and force the agent to **explore + actively apply thrust**

---

# Task 1: Increase Exploration (MOST IMPORTANT)

Modify PPO parameter:

```python
ent_coef = 0.1
```

If still inactive:

```python
ent_coef = 0.2
```

### Purpose

* Prevent premature convergence
* Encourage diverse thrust actions

---

# Task 2: Enforce Minimum Thrust (Critical Trick)

Add in environment step:

```python
if np.linalg.norm(thrust) < 0.05:
    thrust = thrust + 0.05 * unit_t
```

### Purpose

* Prevent "do nothing" behavior
* Force continuous control signal

---

# Task 3: Restart Training (MANDATORY)

 Do NOT load old model

```python
model = PPO(...)  # fresh initialization
```

### Reason

* Current policy is stuck
* Hard to escape local optimum

---

# Task 4: Keep Current Reward (Do NOT change yet)

Your reward already includes:

* Tangential velocity guidance
* Radius constraint
* Thrust direction reward

 Focus on fixing exploration first

---

 Expected Changes (Success Indicators)

## Trajectory

* ❌ Before: straight falling curve
* ✅ Target: visibly curved (starting to orbit)

## Radius

* ❌ Before: monotonic decrease
* ✅ Target: flattening / slowing decay

## Alignment

* ❌ Before: stuck ~0.68
* ✅ Target: > 0.75 → toward 0.8+

## Speed

* ❌ Before: monotonically increasing
* ✅ Target: controlled (no runaway growth)

---

# Notes & Expected Side Effects

### 1. Possible oscillation

 Normal due to increased exploration

### 2. Reward may temporarily drop

 Normal during exploration phase

### 3. Trajectory may look messy

 Fine — as long as it is NOT straight anymore

---

# Core Idea of Day 10

> ❗ First make the agent "move"
> Then make it "move correctly"

---

# Definition of Done (Day 10)

Any of the following:

* Trajectory becomes clearly curved
* Radius is no longer strictly decreasing
* Alignment > 0.75

---

# Next Step (Preview Day 11)

If successful:
 Start **orbit stabilization (circularization)**

If not:
 Further increase exploration + action shaping

---

 Tomorrow’s goal:

> See the first "orbit-like" trajectory
