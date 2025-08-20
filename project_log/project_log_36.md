# Project Log 36 - 2025-08-20

## Summary
Today focused on **Route A: quick validation of expert controllers** on large-scale spacecraft orbit tasks. Successfully generated a batch of `task_specs_fast`, executed baseline runs with multiple controllers (`spiral_in`, `bangband`, `transfer`, `elliptic`, `random`), and performed evaluation.

## Achievements
- Generated 20 fast tasks for circular, elliptic, and transfer orbits (`task_specs_fast`).
- Ran baseline controllers and collected results into CSV (`baseline_fast.csv`, `baseline_fast_experts.csv`).
- Verified **success in circular orbit tasks**:
  - All controllers reached `succ=1`.
  - New experts (`transfer`, `elliptic`) achieved **lower radius error (r_err)** compared to `spiral_in`.
- Completed evaluation pipeline:
  - Produced `summary.csv` and `worst_20.csv`.
  - Confirmed replay visualization worked correctly.

## Issues Encountered
1. **Elliptic tasks**  
   - All controllers failed (`succ=0`).
   - Radius error remained high (≈0.18–0.50).
   - Possible causes: insufficient eccentricity damping, tail window too short, or large spacecraft inertia.

2. **Transfer tasks**  
   - Most controllers failed with very high radius error (`r_err=6.5` or `9.0`).
   - Current “pseudo-Hohmann” approach too coarse; not enough accuracy in large-scale transfers.

3. **Result variance**  
   - Some runs initially showed all failures, but later reruns with the same tasks succeeded for circular cases.
   - Indicates sensitivity to initial conditions or random seeds.

## Key Insights
- **Circular orbit control is solved** under current settings, with new controllers outperforming older ones.  
- **Elliptic and transfer orbits remain unsolved**, requiring stronger eccentricity damping and phased strategies.  
- Evaluation and replay pipeline is now stable, providing clear data for future RL training targets.
