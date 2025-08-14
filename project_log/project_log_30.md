# Project Log — Day 30

> Long-term competition prep: establishing a reliable, fuel-aware baseline and an evaluation harness that cleanly separates “smart” control from trivial baselines.

---

## Summary

- **Goal:** Make progress measurable on solar-scale multi-task orbit control by matching the physical timescale and introducing a fuel-aware, energy-shaping baseline.
- **Result (Stage A thresholds):** `GreedyEnergyRT` achieves **SR = 0.64**, **return = +351.6**, **r_err ≈ 0.051**, **v_err ≈ 0.047**, **align ≈ 0.99**.  
  Successful-episode fuel **median ≈ 1.843e6**, substantially lower than earlier iterations.
- **Where it still struggles:** Very large target radii (≈ 5.5e12–8.6e12 m) and moderate/high eccentricity (e ≈ 0.22–0.29).

---

## What changed today

1. **Timescale alignment**  
   `dt = 6000.0` s (≈ 1.67 h/step), `max_steps = 30000` (≈ 5.7 years/episode) to match solar-scale orbital periods (~6 years).

2. **Time-integrated fuel**  
   In `OrbitEnvMT.step()`:
   - `fuel_step = ||thrust|| * dt`  
   - Reward uses `w_fuel * fuel_step`  
   - Accumulate `self.fuel_used += fuel_step`  
   This makes fuel cost fair across different `dt`.

3. **Evaluation thresholds (Stage A)**  
   `rerr_thr = 0.018`, `verr_thr = 0.035`, `align_thr = 0.96`, `stable_steps = 120`.  
   Kept: `w_fuel = 2e-4`, `w_align = 0.2`, `thrust_scale_range = (100, 150)`.

4. **Baseline controller (fuel-saving Energy-shaping)**  
   Core idea: `v_t_des = 1 − k_e * (r_norm − 1)` (with mild saturation), plus radial PD damping, proximity-aware throttle, and hysteresis deadzone for coasting near the success band.  
   Final Day-30 parameters:
                             k_e=0.9, k_rp=0.10, k_rd=0.40,
                             t_clip=0.45,
                             a_max_lo=0.06, a_max_hi=0.45,
                             dead_r_in=0.035, dead_r_out=0.028,
                             dead_v_in=0.070, dead_v_out=0.055,
                             v_des_min=0.80, v_des_max=1.20

5. **Richer evaluation outputs**  
- Print fuel stats for successful episodes (mean/median).  
- Persist task parameters to CSV (`target_radius, e, mass, thrust_limit`).  
- Console summary lists the three hardest tasks with parameters.

---

## Experimental setup

- **Tasks:** 5 fixed seeds + 20 random tasks (solar-scale distribution).  
- **Controllers:** `ZeroThrust` (lower bound) vs `GreedyEnergyRT` (fuel-aware energy-shaping).  
- **Metrics:** `SR, return, r_err, v_err, align, fuel_used, ended_by_max, violations`, plus `fuel_succ(mean/median)`.

---

## Key results (final Day-30 run)

| Controller             |   SR  | Return |  r_err  |  v_err  | align | fuel(all) | fuel_succ (mean / median) | Ends by max |
|------------------------|------:|-------:|--------:|--------:|------:|----------:|---------------------------:|------------:|
| ZeroThrust             | 0.080 | -101.6 | 0.1633  | 0.1203  | 0.996 | 0         | 0 / 0                      | 92%         |
| GreedyEnergyRT (final) | **0.640** | **+351.6** | **0.05137** | **0.04668** | **0.990** | 1.875e6   | **2.450e6 / 1.843e6**       | 36%         |

**Hardest three tasks (largest final `r_err`):**
1) `random_8` — `rt ≈ 5.55e12`, `e ≈ 0.22`, `mass = 600`, `thrust ≈ 117.91`  
2) `random_16` — `rt ≈ 6.87e12`, `e ≈ 0.29`, `mass = 720`, `thrust ≈ 126.34`  
3) `random_18` — `rt ≈ 8.60e12`, `e ≈ 0.14`, `mass = 720`, `thrust ≈ 120.17`

_For reference, ZeroThrust’s three worst in this run:_  
`fixed_2 (rt ≈ 1.10e12, e ≈ 0.24, mass = 1000, thrust ≈ 110.16)`,  
`random_17 (rt ≈ 1.02e12, e ≈ 0.09, mass = 600, thrust ≈ 148.36)`,  
`random_1 (rt ≈ 1.46e12, e ≈ 0.19, mass = 1000, thrust ≈ 132.08)`.

---

## Comparison to earlier iterations (brief)

- **Before large `dt`:** SR ≈ 0; `r_err ≈ 0.11` flat—insufficient Δv within an episode.  
- **After large `dt` + early energy-shaping:** SR ≈ 0.72; `r_err` drops; fuel still high (successful median ~3.18e6).  
- **Fuel-saving variants + Stage A:** SR ≈ 0.64; `r_err ≈ 0.051`; successful-episode median fuel ~**1.843e6**; return positive and improving.

---

## Risks & known issues

- **Long-tail difficulty:** Ultra-large radii and moderate/high eccentricity remain primary failure modes (slower radius convergence; longer sustain to meet 120 consecutive steps).  
- **Sustain cost near success:** Stage-A’s `stable_steps = 120` increases fuel spent to “hold” success.  
- **Heuristic baseline:** Controller is heuristic; trades fuel vs. stability well overall but can be conservative on the hardest tasks.

---

## Conclusions

- Matching the simulation timescale and using a fuel-aware energy-shaping controller yields **clear separation** from the trivial baseline and **stable progress signals**.  
- The evaluation harness (CSV + diagnostics + task parameters) is now strong enough to benchmark future policies (Expert / PPO / Hybrid).  
- Remaining gaps are focused and diagnosable (ultra-far radii + higher eccentricity).

---

## Next steps (long-term plan)

1. **Tighten thresholds gradually (Stage B, once SR ≥ 0.55 is steady):**  
`rerr_thr = 0.015`, `verr_thr = 0.030`, `align_thr = 0.97`, `stable_steps = 160`.

2. **Target the difficult sub-distribution:**  
- Optional controller v1.4: radius/eccentricity-aware gains (slightly higher far-field throttle and tangential clamp only when `rt > 4e12` or `e > 0.25`), while keeping the near-target fuel saver.  
- Or curriculum: narrow range first, then expand toward the far/high-e corner cases.

3. **Add an Expert upper bound:**  
Unified action interface (normalized `[-1, 1]^2`) for apples-to-apples evaluation.

4. **Hybrid-PPO (short cycles, then scale):**  
10-D observation → 2-D `tanh` actions; initialize actor head from Expert;  
sanity-train on “5 fixed + narrow random” for 30–50 epochs, then widen.

5. **Metric tracking:**  
Keep **successful-episode median fuel** as the primary fuel KPI;  
log the hardest 2–3 tasks and iterate on them explicitly.

---

## Repro steps

```bash
# Python 3.10+
pip install -r requirements.txt

# Evaluate
python eval_battery.py

# Outputs
# - Console: Battery Summary (incl. fuel_succ stats, hardest 3 tasks)
# - CSV: results/battery_day30.csv (includes task parameters)
