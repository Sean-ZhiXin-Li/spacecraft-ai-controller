
# NEW_WEEK-PROJECT_LOG_03 — Week 3 Report

## Overview

Week 3 focused on systematic stress‑testing of the expert controller across multiple challenging scenarios.  
The goal was to identify failure modes, quantify them with energy‑based metrics, and build a structured failure catalog to guide controller improvements for Week 4.

This week marks the first time the project reached a full *failure‑mode engineering* cycle:  
running multiple edge‑case scenarios → capturing .npz logs → generating diagnostic plots → extracting metrics → synthesizing insights into a formal document.

## What Was Completed

### 1. Stress Battery (Week 3)
Three main scenarios were executed:

1. **weak_thrust_far**  
   Spacecraft begins far outside target orbit with insufficient thrust authority.

2. **oscillation_noise**  
   Controller receives noisy/perturbed inputs, causing unstable thrust decisions.

3. **misaligned_entry**  
   Spacecraft approaches target orbit with a large angular mismatch and non‑circular entry velocity.

All three scenarios ran 3 episodes each (`run_01 ~ run_03`) and produced:
- orbit trajectory plots  
- radial‑error vs time curves  
- thrust‑magnitude histograms  
- energy‑summary text blocks  
- extracted metrics (.json)

### 2. Failure Catalog
All results were consolidated into **NEW_WEEK_3_failure_catalog.md**, containing:

- Detailed descriptions of each failure mode  
- Metric‑based signatures (energy drift, angular‑momentum error, oscillation index…)  
- Visual references  
- Root‑cause analysis  
- Controller‑level interpretation  
- Implications for Week 4’s design

The failure catalog now functions as the foundation for controller upgrades.

### 3. Verification & File Pipeline
Week 3 also validated the full toolchain:

- `stress_battery.py` now stable with OrbitEnv v3  
- Energy‑view pipeline produces reproducible metrics  
- Figures save correctly into:  
  `logs/new_week_3/<scenario>/run_xx/*.png`

This confirms the environment/tooling is reliable enough for Week 4’s algorithmic changes.

## Key Findings

### weak_thrust_far
- Energy drift extremely high (~3000%)  
- Controller consistently under‑thrusting  
- Orbit remains far from target  
- Indicates need for thrust‑aware scaling or adaptive switching strategy

### oscillation_noise
- Very low convergence step (7) but **unstable energy behavior**  
- Thrust magnitudes fluctuate heavily  
- Shows controller is sensitive to angular misalignment + input noise  
- Requires either low‑pass filtering or angle‑penalty shaping

### misaligned_entry
- Controller eventually stabilizes (convergence ~141 steps)  
- Energy drift near zero → good  
- But radial error shows staircase pattern  
- Suggests angle correction logic is correct but too discretized  
- Calls for smoothing or continuous modulation

## What Worked Well

- Diagnostic pipeline is fully mature  
- Energy‑based metrics successfully reveal controller failure signatures  
- All plots rendered correctly  
- Failure catalog readable and academically structured  
- Stress test runs quickly without hangs after fixes

## What Still Needs Improvement

- Expert controller lacks robustness in:
  - high‑eccentricity states  
  - far‑field radial corrections  
  - noisy angle regimes

- Thrust magnitude patterns show strong mode collapse  
- No switching logic between “enter orbit” vs “maintain orbit” modes  
- Needs better mapping between angular‑momentum errors and thrust response

These insights directly shape next week's upgrade plan.

## Summary

Week 3 achieved a complete stress‑analysis cycle:
simulation → visualization → failure interpretation → metrics → catalog writing.

have a formal failure‑mode document, stable tooling, and clear guidance for controller redesign. Week 4 is primed to deliver the first significantly improved version of the controller.

