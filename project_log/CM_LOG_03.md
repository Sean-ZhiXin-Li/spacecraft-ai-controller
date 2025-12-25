# CM Day3 — Week7 Ablation (Evidence, Not Opinions)

**Date:** 2025-12-25
**Session Theme:** Ablation Evidence, Not Opinions

---

## Objective

Convert an implementation detail (ACTION_IF_MODE: raw vs prescale) into **explicit experimental evidence**.

Constraints were strict:

* No controller changes
* No environment / physics changes
* No reward changes
* Same scenario, same steps

Only allowed actions: re-run, record metrics, save results, plot one figure.

---

## Experiment Setup

* **Script:** `src/quick_compare_v3_v4.py`
* **Scenario:** `default`
* **Steps:** 2000
* **Controllers:** ExpertV3, ExpertImproved
* **Variable under test:** `ACTION_IF_MODE ∈ {raw, prescale}`

All other factors were frozen.

---

## Metrics Collected

Metrics were copied directly from existing summary/log lines (no recomputation):

* `saturation_rate`
* `avg_radius_error`
* `total_reward`

---

## Results

### Raw

* ExpertV3:

  * saturation_rate = 0.50
  * avg_radius_error = 1.875e12
  * total_reward = -1.553e04

* ExpertImproved:

  * saturation_rate = 0.50
  * avg_radius_error = 1.875e12
  * total_reward = -1.553e04

### Prescale

* ExpertV3:

  * saturation_rate = 0.50
  * avg_radius_error = 1.875e12
  * total_reward = -1.553e04

* ExpertImproved:

  * saturation_rate = 0.50
  * avg_radius_error = 1.875e12
  * total_reward = -1.553e04

---

## Figure

* `analysis/fig_sat_rate_raw_vs_prescale.png`
* Content: saturation_rate comparison (raw vs prescale)

Both controllers produce identical curves; lines overlap.

---

## Conclusion (Evidence-Level)

Under `scenario=default` and 2000 steps, switching ACTION_IF_MODE from **raw** to **prescale** produces **no observable change** in:

* action saturation behavior
* long-term orbital error
* total reward

This indicates that, in this configuration, **action scaling is not a confounding factor**.

---

## Why This Matters

This session establishes a reusable **ablation template**:

* isolate one variable
* freeze everything else
* let the system produce evidence

Future controller improvements can now be evaluated against a verified interface baseline, preventing false attribution of gains to implementation details.

---

## Deliverables

* `analysis/WEEK7_ablation_results.json`
* `analysis/fig_sat_rate_raw_vs_prescale.png`

---

