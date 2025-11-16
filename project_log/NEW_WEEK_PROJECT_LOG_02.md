# NEW_WEEK_2 — Visual Diagnostics Pipeline

## 🎯 Objective
Extend Week1’s energy-view analysis into a complete visual-diagnostics system.
The goal is to standardize how any replay (expert / RL / imitation) is inspected:
orbit geometry, radial convergence, thrust usage, and energy metrics all plotted
in a unified format.

This matches real research workflows used in Stanford SLAB, NASA JPL, and ESA ACT.

---

## ✅ Progress Summary

### 1. `diag_orbit.py` — Unified Visual Diagnostic Entry
A new CLI tool has been fully implemented:

```
python -m tools.diagnostics.diag_orbit   --replay logs/.../replay.npz   --metrics logs/.../metrics_energy.json   --outdir figures/...   --save --png-dpi 180   --target-radius 7.5e12   --title "NEW_WEEK_1 high_thrust"
```

It produces **four standard figures**:
- Orbit (x,y)
- Radial error vs time
- Thrust magnitude histogram
- Energy summary sheet

---

### 2. Diagnostic Figures (NEW_WEEK_1 / high_thrust)
Figures generated and saved under:

```
figures/new_week_2/high_thrust/
```

The high-thrust expert controller shows:
- clean spiral-in trend  
- monotonic radial convergence with stepwise thrust usage  
- dual-mode thrust histogram (low-power drift + high-power correction)  
- stable energy metrics (drift < 1%)  

---

### 3. Module Structure Improvements
Diagnostics folder is now a fully functional Python package.

---

### 4. Visual PASS / FAIL Criteria (Week2 Standard)
Full criteria in:

```
analysis/NEW_WEEK_2_visual_criteria.md
```

This defines how to judge stability, convergence, oscillation, and thrust behavior.

---

## 📌 Next Steps (NEW_WEEK_3 Preview)
- Add multi-run comparison  
- Add trajectory overlay (Expert vs RL)  
- Add thrust-angle visualization  
- Experiment with energy-driven controller adjustments  

---

## 🔖 Commit Message
```
WEEK2: add visual diagnostics (orbit/error/thrust/energy) and criteria
```
