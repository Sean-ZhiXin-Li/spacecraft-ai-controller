# PL_W01 — First Stability Surface from Trajectory Summary

## Date

2026-03-07

---

## Objective

Construct the first 2D stability surface from trajectory summary data.

This week introduces no new simulations and no controller modifications. The goal is to transform existing trajectory summary results into a structured two‑dimensional representation so that controller behavior can be compared directly across thrust magnitude and controller structure.

---

## Data Interface

Input file:

`analysis/results/day17_dynamics_summary.csv`

The summary dataset contains four completed experiment runs from the near‑field comparison set. Available metrics include:

* `min_r_err`
* `max_r_err`
* `min_vr`
* `t_flip`
* `t_cross`
* `delta_r`

Each run corresponds to one configuration defined by:

* controller structure (`always_on` or `gated`)
* thrust magnitude (800 N or 2000 N)

---

## Method

A heatmap‑based stability surface was constructed.

Axes:

* **X‑axis:** thrust magnitude
* **Y‑axis:** controller structure

The summary metric is mapped into a 2D matrix and visualized using matplotlib.

Two surfaces were generated during PL_W01:

1. **Primary metric:** `min_r_err`
2. **Secondary metric:** `min_vr`

Output artifacts:

* `analysis/scripts/week01_heatmap.py`
* `analysis/figs/stability_surface_v1.png`
* `analysis/figs/stability_surface_min_vr_v1.png`

---

## Observation

### Surface using `min_r_err`

All four configurations produced identical `min_r_err` values. The resulting surface showed no separation across controller structures or thrust levels.

This indicates that the minimum radial error in the current near‑field setup is dominated by the initial condition rather than controller dynamics.

### Surface using `min_vr`

When switching the metric to `min_vr`, small but visible differences appeared across the surface.

At **800 N**, the two controllers behave almost identically, suggesting that control structure has little influence in the low‑thrust regime.

At **2000 N**, the two controllers diverge:

* The **gated** controller produces a lower `min_vr`
* The **always_on** controller produces a higher `min_vr`

This suggests that controller logic begins to interact with thrust magnitude as actuation strength increases.

Although the dataset is small, the surface already hints at a regime‑dependent controller effect.

---

## Conclusion

PL_W01 successfully established the first stability‑surface analysis pipeline.

The initial metric (`min_r_err`) was found to be insensitive under the current initial conditions, while `min_vr` provided weak but observable separation between controller behaviors.

The key outcome of this week is methodological: the project now has a working 2D parameter‑space visualization for controller comparison.

---

## Next Step

PL_W02 will extend the surface along the thrust axis by performing a **thrust sweep** across multiple intermediate values.

The goal is to determine whether a clearer controller‑dependent regime boundary emerges as thrust magnitude increases.
