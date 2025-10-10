import os, json, numpy as np
root = r"logs/day54/spiral_in/smoke"
npz  = os.path.join(root, "replay.npz")
meta = os.path.join(root, "meta.json")

d = np.load(npz)
print("keys:", list(d.files))

obs = d.get("obs"); act = d.get("act"); rew = d.get("rew"); info = d.get("info")
if obs is not None: print("obs.shape:", obs.shape, "obs[-1]:", obs[-1])
if act is not None: print("act.shape:", act.shape, "act last 3:", act[-3:])
if rew is not None: print("rew.shape:", rew.shape, "rew.sum:", float(np.nan_to_num(rew).sum()))
if info is not None: print("info.shape:", info.shape)

with open(meta, "r", encoding="utf-8") as f:
    m = json.load(f)
print({k: m.get(k) for k in ["steps_recorded","seed","env_factory","policy"]})
