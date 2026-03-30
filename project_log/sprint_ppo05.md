# Day 5 (Integration + Behavior Discovery)

## Core Objective

Integrate PPO into the W03 pipeline as a **comparable controller**, and directly analyze its behavior alongside always_on and gated under a unified evaluation system.

---

# Day 5 — PPO Integration + First Behavior Insight

> Day 5 is not just integration — it is the moment PPO becomes an **experimental object**.

---

## Part 1 — Unified Controller Integration

### Unified Interface

```python
class Controller:
    def act(self, obs):
        return action
```

### Controllers Included

* always_on
* gated
* PPO

### PPO Design Choice

* Inference only (no training loop)
* Deterministic action (mean of policy)

### Unified Execution

All controllers run through:

```python
obs = env.reset()
for t in range(T):
    action = controller.act(obs)
    obs, reward, done, _ = env.step(action)
```

---

## Part 2 — Standardized Evaluation

### ✔ Metrics (Same for all controllers)

* total_reward
* avg_radius_error
* saturation_rate
* jitter

### Output

* JSON / CSV
* All results from **real rollout (no training logs)**

---

## Part 3 — Behavior Instrumentation (Key Upgrade)

Added per-step thrust direction decomposition:

* `cos(thrust, radial)` → cos_tr
* `cos(thrust, tangential)` → cos_tt

Stored in `traj.npz` for all controllers.

---

## Part 4 — Comparative Results

| Controller | cos_tr  | cos_tt | Variance | Behavior                    |
| ---------- | ------- | ------ | -------- | --------------------------- |
| PPO        | ~ -0.95 | ~ 0.31 | Very low | Near-constant inward thrust |
| Gated      | ~ -0.93 | ~ 0.36 | Medium   | Structured control          |
| Always_on  | ~ -0.90 | ~ 0.41 | High     | Aggressive / adaptive       |

---

## Core Finding (Day 5 Output)

> PPO converges to a **low-variance, near-constant inward thrust strategy**,
> while expert controllers maintain higher directional flexibility.

---

## Interpretation

* PPO does not learn a highly dynamic feedback controller
* Instead, it discovers a **simple, stable control structure**
* This structure is:

  * Strongly radial inward
  * Slightly tangential
  * Nearly constant over time

---

## Key Insight

> PPO behaves more like a **fixed-structure controller** than a dynamic feedback controller.

---

## Day 5 One-Line Summary

> PPO is not optimizing complexity — it is collapsing to a simple, stable control law.

---

## Next Step (Day 6)

* Validate under different thrust regimes
* Confirm whether PPO consistently exhibits this structure
* Extend comparison across parameter sweeps
