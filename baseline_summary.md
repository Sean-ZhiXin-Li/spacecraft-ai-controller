# Baseline Summary (Day35)

## 1. Average Metrics (N=20 random tasks, seed=999)

| Controller       | SR    | Return  | r_err   | v_err   | Align  | Fuel(all)   | Fuel(succ mean/median) |
|------------------|-------|---------|---------|---------|--------|-------------|-------------------------|
| Zero             | 0.000 | 790.3   | 0.1220  | 0.0707  | 0.995  | 0.0         | n/a                     |
| Greedy (energy)  | 0.560 | 1493.9  | 0.0280  | 0.0317  | 0.990  | 1.65e6      | 2.30e6 / 2.08e6         |
| Expert-eco       | 0.480 | 364.6   | 0.0377  | 0.0366  | 0.980  | 3.69e6      | 4.52e6 / 5.17e6         |
| Expert-fast      | 0.520 | **49.2**| 0.0372  | 0.0370  | 0.978  | 4.16e6      | 4.82e6 / 6.09e6         |

Notes:
- **SR** = Success Rate
- **Return** = cumulative reward (higher is better)
- **r_err, v_err, Align** = final orbital error metrics
- **Fuel(all)** = total fuel used across all runs
- **Fuel(succ)** = mean/median fuel on successful runs

---

## 2. Worst-3 Tasks (by return)

- fixed_3
- random_7
- random_15  

*(appeared consistently across Greedy / Expert controllers)*

---

## 3. Replay Plots

- ΔReturn (vs. Zero)  
  ![](ab/day35_replay/plot_delta_ret.png)

- Δr_err (position error evolution)  
  ![](ab/day35_replay/plot_delta_rerr.png)

- ΔFuel (consumption trajectory)  
  ![](ab/day35_replay/plot_delta_fuel.png)

*(plots generated via `replay_worst.py --csv results/battery_day35_final.csv --out ab/day35_replay`)*

---

## 4. Conclusions

- **Upper Bounds (Reference Experts):**  
  - *Expert-eco*: fuel-aware reference, stable SR≈0.48  
  - *Expert-fast*: reward-oriented, SR≈0.52, **positive return (>0)** achieved

- **Lower Bound:**  
  - *Zero*: trivial baseline (always fail, no fuel)  

- **Greedy Controller:**  
  - Practical middle ground: high return, moderate SR, relatively efficient fuel usage.  
  - Can be used as **training baseline** or ablation benchmark.  

- **Summary:**  
  - *Expert controllers* serve as **performance upper bound**.  
  - *Zero* is the **lower bound**.  
  - *Greedy* is a **reasonable practical baseline** with strong return but lower success reliability.
