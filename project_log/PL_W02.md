# PL_W02 — Thrust Sweep and Controller Separation

Date: 2026-3-14

---

## Objective

Extend the stability surface exploration by sweeping thrust magnitude and observing how system dynamics change under different controller structures.

Core question:

Does increasing thrust amplify the instability pattern, and how does the controller variant influence this amplification?

Controllers tested:

* `always_on`
* `gated`

Primary metric:

* `min_vr` (minimum radial velocity)

Secondary metrics:

* `t_flip`
* `delta_r`

---

## Experimental Setup

Environment parameters remain fixed:

* target_radius = 7.5e12
* initial orbit ≈ 1.005 × target

Thrust sweep range:

500 N
800 N
1000 N
1200 N
1500 N
1800 N
2000 N
2500 N

For each thrust level:

1. Run simulation
2. Extract trajectory summary
3. Compute dynamics metrics
4. Append results to sweep table

Result file:

analysis/results/week02_thrust_sweep_summary.csv

---

## Results

The thrust sweep produced a consistent pattern across the full range.

Observation 1

Both controllers show monotonic deepening of `min_vr` as thrust increases.

This indicates that increasing thrust amplifies the inward plunge dynamics.

Observation 2

`always_on` consistently produces a more negative `min_vr` than `gated`.

This pattern holds for every thrust level tested.

Observation 3

The gap between the two controllers grows gradually as thrust increases.

At low thrust the difference is moderate, but it becomes significantly larger in the high-thrust regime.

Observation 4

`t_flip` decreases with thrust for both controllers, meaning that inward dynamics begin earlier when thrust increases.

---

## Visualization

A line plot was generated to visualize the sweep:

analysis/figs/thrust_vs_min_vr_week02.png

The plot reveals two nearly parallel curves:

* both curves descend with increasing thrust
* `always_on` lies consistently below `gated`
* the separation between curves gradually widens

This structure suggests that thrust controls the overall instability growth, while the controller shifts the system onto different dynamical branches.

---

## Interpretation

The experiment indicates two interacting effects:

1. Thrust amplification

Higher thrust increases the magnitude of inward radial motion.

2. Controller structural effect

The `always_on` controller continuously injects thrust, which produces stronger inward excursions compared with the gated strategy.

The combination results in a persistent separation between controller behaviors across the thrust axis.

---

## Conclusion

PL_W02 confirms that thrust is a major driver of instability growth in the system.

Both controllers become more unstable as thrust increases, but `always_on` consistently produces deeper inward plunges and slightly earlier instability onset.

The difference between the controllers expands gradually with thrust rather than appearing abruptly at a single threshold.

---

## Next Step (W03)

Refine the stability surface by exploring additional parameters.

Possible directions:

* refine the thrust region around mid-range values
* explore sensitivity to initial orbit error (`r0_over_target`)
* analyze how controller behavior changes across different initial conditions

Goal:

Move from a thrust-only sweep toward a broader stability surface mapping.
