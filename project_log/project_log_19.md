# Project Log – Day 19

 **Date:** 2025-08-03  
 **Focus:** Dataset cleanup & V6 imitation controller preparation

---

## Task Summary

Today I began preparing the foundation for **Imitation Controller V6**.  
The main focus was cleaning up data files, revisiting model training scripts, and preparing for a clean training pipeline.

### Completed:

- Cleaned up `.npy`, `.joblib` and `.pth` files from Git tracking
- Added a `.gitignore` file to exclude:
  - `*.npy`, `*.joblib`, `*.pth`, `*.png`
  - Cached or intermediate training/visualization outputs
- Ensured all dataset files (`expert_dataset_*.npy`) are ready for training
- Wrote and debugged the `train_imitation.py` script, covering:
  - Dataset loading and stacking via `np.vstack`
  - Input/output split (`[pos, vel]` → `[thrust]`)
  - Scaling with `StandardScaler`
  - Training pipeline with `MLPRegressor`

---

## Problems Encountered

### Memory Error During Training

While attempting to train the V6 model using the current structure:

```python
hidden_layer_sizes = (256, 128)
dtype = float64 (default)
dataset_size = ~24 million rows
```

The process crashed with the following error:

```text
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 22.9 GiB for an array with shape (24000000, 128)
```

### Causes:

- Dataset too large to fit in memory for training
- No batching or partial loading used
- `float64` precision doubles memory use
- Large layer sizes in MLP

---

## Lessons Learned

- Future training should consider:
  - Reducing hidden layer sizes (e.g., (128, 64) or (64, 32))
  - Downsampling dataset (e.g., 1/5 or 1/10)
  - Switching to `float32` before model training
  - Considering incremental/batch training options if using very large datasets

---

## Output / Changes

- `train_imitation.py` (new script) created for MLP training
- `.gitignore` now prevents large generated files from cluttering version control
- All old joblib/PPO model files cleared from tracking

---

## Progress Recap

```
️ V6 imitation controller: dataset prep complete
 Training not started due to OOM error
 Git hygiene improved: outputs now ignored
```

---
