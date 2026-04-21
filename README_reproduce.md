# Reproducing the Historical Day5 Ablation Result

This document is kept for the earlier action-interface ablation path.
It does not describe the repository's current headline result.

For the current validated explicit-controller result, use:

- `README.md`
- `analysis/final_project_summary.md`
- `analysis/orbit_lock_benchmark.md`
- `analysis/orbit_lock_generalization.md`

This document provides a zero-ambiguity path to reproduce the key result reported in `analysis/ONE_PAGE_SUMMARY.md`.
The goal is to allow any reviewer, judge, or collaborator to rerun the experiment and obtain the same conclusion.

---

## Environment

* OS: Linux (tested under WSL / Ubuntu)
* Python: see `conda_envs/spacecraft.yml`
* No code modifications are required

---

## Configuration

All experiments are fully defined by a single configuration file:

* `config/default.yaml`

This file specifies the physics parameters, reward definition, and numerical integration settings.
No configuration values are changed between runs.

The **only independent variable** is the action interface mode, controlled via the environment variable `ACTION_IF_MODE`.

---

## Reproduction Command

Run the following commands from the project root:

```bash
# Raw action interface
export ACTION_IF_MODE=raw
python src/quick_compare_v3_v4.py

# Prescaled action interface
export ACTION_IF_MODE=prescale
python src/quick_compare_v3_v4.py
```

---

## Expected Output

Both runs should complete a 2000-step episode for each controller and report:

* Mean `saturation_rate` ≈ **0.50** for both `raw` and `prescale`
* Identical average radius error and total reward across interface modes
* Console summaries consistent with the values recorded in:

  * `analysis/WEEK7_ablation_results.json`

The absence of measurable differences between the two modes reproduces the key finding described in Day4.

---

## Generated Artifacts

Running the commands above will generate or update:

* `analysis/WEEK7_ablation_results.json`
* `analysis/fig_sat_rate_raw_vs_prescale.png`

These artifacts correspond directly to the evidence cited in the one-page research summary.

---

## Notes

* No physics, reward, or controller logic is modified between runs.
* All results rely on the fixed configuration in `config/default.yaml`.
* The reproduction path is intentionally minimal to reduce ambiguity and support contest-level credibility.
    
