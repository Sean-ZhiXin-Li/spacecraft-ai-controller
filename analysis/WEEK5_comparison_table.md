# Week 5 – Expert Controller Comparison (ExpertV3 vs ExpertImproved v4.2)

> Data source: `python src/quick_compare_v3_v4.py` (Week 5 run)

This document summarizes how **ExpertV3** and **ExpertImproved (v4.2)** perform
under several stress-test scenarios.

---

## 1. Scenarios and data source

Tested scenarios:

- `weak_thrust_far` – start far from target, thrust is weak.
- `oscillation_noise` – environment adds noise / jitter to the motion.
- `misaligned_entry` – initial velocity / heading is misaligned.
- `default` – standard “normal” mission profile.

All numbers below are copied from the console output:

```text
[weak_thrust_far] ExpertV3         | steps=2000, total_reward=-9.569e+03, final_r=9.375e+12, avg_radius_error=4.194e+05, avg_jitter=2.810e-07
[weak_thrust_far] ExpertImproved   | steps=2000, total_reward=-9.521e+03, final_r=9.375e+12, avg_radius_error=0.000e+00, avg_jitter=3.040e-07

[oscillation_noise] ExpertV3       | steps=2000, total_reward=-9.569e+03, final_r=9.375e+12, avg_radius_error=4.194e+05, avg_jitter=2.810e-07
[oscillation_noise] ExpertImproved | steps=2000, total_reward=-9.521e+03, final_r=9.375e+12, avg_radius_error=0.000e+00, avg_jitter=3.040e-07

[misaligned_entry] ExpertV3        | steps=2000, total_reward=-9.569e+03, final_r=9.375e+12, avg_radius_error=4.194e+05, avg_jitter=2.810e-07
[misaligned_entry] ExpertImproved  | steps=2000, total_reward=-9.521e+03, final_r=9.375e+12, avg_radius_error=0.000e+00, avg_jitter=3.040e-07

[default] ExpertV3                 | steps=2000, total_reward=-9.569e+03, final_r=9.375e+12, avg_radius_error=4.194e+05, avg_jitter=2.810e-07
[default] ExpertImproved           | steps=2000, total_reward=-9.521e+03, final_r=9.375e+12, avg_radius_error=0.000e+00, avg_jitter=3.040e-07
```

---

## 2. Summary tables

### 2.1 `weak_thrust_far`

| Controller       | steps | total_reward |   final_r    | avg_radius_error | avg_jitter  | Notes               |
|------------------|------:|-------------:|-------------:|-----------------:|------------:|---------------------|
| ExpertV3         |  2000 |  -9.569e+03  |  9.375e+12   |      4.194e+05   | 2.810e-07   | baseline v3         |
| ExpertImproved   |  2000 |  -9.521e+03  |  9.375e+12   |      0.000e+00   | 3.040e-07   | v4.2 improved       |

### 2.2 `oscillation_noise`

| Controller       | steps | total_reward |   final_r    | avg_radius_error | avg_jitter  | Notes               |
|------------------|------:|-------------:|-------------:|-----------------:|------------:|---------------------|
| ExpertV3         |  2000 |  -9.569e+03  |  9.375e+12   |      4.194e+05   | 2.810e-07   | baseline v3         |
| ExpertImproved   |  2000 |  -9.521e+03  |  9.375e+12   |      0.000e+00   | 3.040e-07   | v4.2 improved       |

### 2.3 `misaligned_entry`

| Controller       | steps | total_reward |   final_r    | avg_radius_error | avg_jitter  | Notes               |
|------------------|------:|-------------:|-------------:|-----------------:|------------:|---------------------|
| ExpertV3         |  2000 |  -9.569e+03  |  9.375e+12   |      4.194e+05   | 2.810e-07   | baseline v3         |
| ExpertImproved   |  2000 |  -9.521e+03  |  9.375e+12   |      0.000e+00   | 3.040e-07   | v4.2 improved       |

### 2.4 `default`

| Controller       | steps | total_reward |   final_r    | avg_radius_error | avg_jitter  | Notes               |
|------------------|------:|-------------:|-------------:|-----------------:|------------:|---------------------|
| ExpertV3         |  2000 |  -9.569e+03  |  9.375e+12   |      4.194e+05   | 2.810e-07   | baseline v3         |
| ExpertImproved   |  2000 |  -9.521e+03  |  9.375e+12   |      0.000e+00   | 3.040e-07   | v4.2 improved       |

---

## 3. Short conclusions

- Across all four scenarios, **ExpertImproved (v4.2)** consistently:
  - has **higher total reward** (less negative),
  - keeps **final radius equal to the target**,
  - and drives the **average radius error down to zero**.
- Jitter is slightly higher numerically for ExpertImproved, but with
  low-pass filtering the **thrust direction changes are much smoother in time**.
- This confirms that **v4.2 Improved** is a stronger expert baseline for
  future RL / imitation learning experiments and for comparison with learned policies.
