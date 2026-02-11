# WHPL_06 — Reproducible 2D Evidence Loop

Date: 2026-02-11
Environment: spacecraft (conda)

---

## 1. Objective

Turn the 2D research goal (thrust × difficulty) into a reproducible experimental pipeline:

* 1 validated data source
* 1 heatmap figure
* 1 explicit conclusion statement
* Fully rerunnable scripts

No physics, reward, or controller logic modifications.

---

## 2. Freeze Line (Strict)

* OrbitEnv physics unchanged
* Reward function unchanged
* ExpertImproved controller unchanged
* Only: data validation, plotting, experiment hygiene

---

## 3. Data Source

Single source of truth:

analysis/results/ablation_thrust_x_difficulty_clean.csv

Current snapshot:

* thrust_newton = 800 N
* difficulty_tag = Hard
* r0_over_target ≈ 1.25
* final_radius_error ≈ 1.874958e12
* total_reward ≈ -2.0795e4

Rows = 1

---

## 4. Engineering Additions

### 4.1 CSV Validation Guardrail

New script:

scripts/validate_ablation_csv.py

Checks:

* Required columns exist
* No NaN in critical metrics
* dedup_key fully populated
* No duplicate dedup_key

Result:

[OK] rows=1 dedup_unique=1 nan_total_reward=0 dedup_key_non_null_ratio=1.000

This converts the dataset into a controlled experimental artifact.

---

### 4.2 Heatmap Plotting Pipeline

New script:

scripts/plot_ablation_heatmap.py

Output:

analysis/figs/whpl06_thrust_x_difficulty_heatmap.png

Heatmap definition:

* x-axis: thrust_newton
* y-axis: difficulty_tag
* color: final_radius_error

Pipeline works even with sparse grid (single pixel case).

---

## 5. Result Interpretation

Observed point:

( thrust = 800 N , difficulty = Hard , r0/rt ≈ 1.25 )

Final radius error remains ≈ 1.87e12.

This is on the same order as the initial deviation.

Interpretation:

The controller does NOT converge to the target orbit under this thrust–difficulty combination.

System is outside stable control regime at this point.

This serves as the Hard-regime anchor point for future 2D grid expansion.

---

## 6. Scientific Status

Current grid density: 1 pixel.

Information gained:

* 800N is insufficient for Hard regime recovery.
* Stable region boundary is not yet observed.

This is a lower-bound failure confirmation.

---

## 7. Next Step (WHPL_07)

Minimal grid expansion strategy:

thrust ∈ {200, 800, 2000}
difficulty ∈ {Easy, Hard}

Objective:

Detect regime boundary with minimum additional runs.

Keep all system components frozen.
Only append new rows to clean CSV.

---

## 8. Definition of Done (WHPL_06)

* CSV validation script operational
* Heatmap script operational
* Figure generated
* Explicit structural conclusion written

WHPL_06 complete.

This is the first reproducible 2D experimental artifact in the project.
