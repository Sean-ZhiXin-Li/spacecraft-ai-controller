# Project Log – Day 28

## Date
2025-08-12

## Progress
Today I continued debugging and monitoring the PPO training loop for the orbital control environment.

- **Training behavior**:  
  - The adaptive learning rate now consistently reaches the current maximum of `8e-5`.  
  - KL divergence stays stable in the `0.003 – 0.010` range.  
  - Policy updates remain clean without sudden collapses.
- **Performance**:  
  - Mean episode return has stabilized in the range of `-64k ~ -57k`.  
  - Indicates that the policy has reached a plateau and further tuning might be required to break through.
- **Code inspection**:  
  - Verified adaptive LR logic and KL-based update scaling.  
  - Checked gradient flow, optimizer step, and advantage calculation — no anomalies found.  
  - Logging and reward curve visualization are working as intended.

## Observations
- The consistently low KL values suggest that the policy could afford larger update steps without risking instability.  
- Current exploration and learning dynamics are too conservative — limiting potential performance gains.  
- Reward stagnation points to either:
  1. Reward shaping limitations.
  2. Policy expressiveness bottleneck.
  3. Insufficient exploration early in training.

## Issues Encountered
- **Reward plateau** despite stable KL and clean updates.  
- **Under-utilized LR capacity**: KL never approaches critical thresholds.
- **Exploration decay**: Possible insufficient variance in actions during early epochs.

## Next Ideas (Not Implemented Today)
- Raise actor LR cap from `8e-5` → `1.2e-4`.  
- Add KL “fast lane”: +1–2 `TRAIN_ITERS` when KL is very low.  
- Increase min log_std in first 150 epochs from `0.08` → `0.10`.  
- Reduce `vf_coef` from `1.0` → `0.8` after 200 epochs.  
- Increase `TRAIN_ITERS` max from `25` → `28`.

## Output Snapshot
- **KL Range**: `0.003 – 0.010`
- **Return Range**: `-64000 ~ -57000`
- **Actor LR (adaptive)**: `→ 8e-5 (cap reached)`

---

**Commit Message Suggestion**:
