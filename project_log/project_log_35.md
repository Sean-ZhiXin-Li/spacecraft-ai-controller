# Project Log - Day35

## Overview
Today marks the completion of the **baseline experiment phase (Day30–Day34)**.  
All baseline controllers (Zero / Greedy / Expert) have been fully evaluated and summarized, closing the loop for Phase 1.  

This provides a solid reference point for upcoming Phase 2 experiments in more complex environments.

---

## Work Completed
- Consolidated results from Day30–34 runs into a unified baseline summary.
- Verified and saved key plots:
  - Return vs. episodes
  - r_err evolution
  - Fuel consumption curves
- Exported `baseline_summary.md` containing:
  - Averaged metrics for each controller
  - Worst-3 tasks list
  - Replay visualization
  - Conclusions on baseline upper/lower bounds
- Confirmed reproducibility by saving CSV logs (`results/battery_day31.csv`).

---

## Key Takeaways
- **Zero** serves as a strict lower bound (SR=0).
- **Greedy** shows strong return and moderate fuel use, competitive as a mid-level baseline.
- **Expert-eco** balances SR and fuel efficiency → interpretable as a “safe” upper bound.
- **Expert-fast** reaches higher SR but consumes more fuel, representing a more aggressive upper bound.

---

## Reflections
- Having a **closed-loop baseline** ensures robustness: even if Phase 2 experiments fail, fallback comparisons are available.
- The separation of **eco vs. fast expert** gives flexibility: one emphasizes efficiency, the other emphasizes success rate.
- Visualizations and worst-case analysis increase clarity for future report writing.

---

## Next Steps
- Move to Phase 2: extend experiments to complex environments.
- Use baselines as anchor points for benchmarking.
- Automate future logging/export (baseline + advanced controllers) to keep project tracking consistent.
- Prepare Day36 kickoff with environment extension and early ablation tests.


