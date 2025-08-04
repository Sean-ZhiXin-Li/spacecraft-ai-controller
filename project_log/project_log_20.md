# Project Log – Day 20

**Date:** August 4, 2025  
**Author:** Zhixin Li  
**Project:** AI-Controlled Spacecraft Orbital Simulator  
**Focus:** Dataset Generation Recovery & PyCharm Environment Fix

---

## Summary

Today marks Day 20 of the project. Despite a major power outage in the afternoon, significant progress was achieved.

### Main Progress

- Resolved a persistent `ModuleNotFoundError: No module named 'simulator'` issue that blocked execution of dataset generation scripts.
- Deleted the `.idea/` directory to reset the PyCharm project environment and reimported the project from scratch.
- Marked `controller/`, `data/`, `simulator/`, and other relevant folders as **Source Root** to correctly enable module-level imports in PyCharm.
- Successfully ran `generate_dataset.py`, confirming the datasets were properly saved to `.npy` and `.csv` formats without import errors.

---

## Unexpected Issues

### Power Outage

From approximately **1:00 PM to 7:15 PM**, a sudden power outage halted all development and coding activity.  
This significant delay affected the day’s schedule and forced a later recovery in the evening.

### IDE Module Recognition Bug

- PyCharm was persistently showing all folders in **yellow**, indicating modules were not properly recognized.
- `Open Module Settings` (F4) option was **unresponsive** when right-clicking the root directory.
- Solved by full IDE reset: delete `.idea/`, close project, and re-open using `Open > spacecraft_ai_project`.

---

## Files Updated

- `generate_dataset.py` executed successfully
- Dataset files:
  - `expert_dataset_01.csv`
  - `expert_dataset_01.npy`

---

## Reflection

Though this day was interrupted by external issues, the IDE environment is now fully stable and configured for all future module development and training tasks.

Tomorrow, work can now resume with:

- Training ImitationController V6
- Running full orbit simulation with learned models
- Beginning PPO environment integration (if time allows)

---

