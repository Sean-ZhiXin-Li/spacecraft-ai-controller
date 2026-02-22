import numpy as np

p = r"analysis/runs/run_750091/traj.npz"
d = np.load(p)
r = d["r"].astype(float)
vr = d["vr"].astype(float)
target_r = float(d["target_r"])

# indicators
r_err = r - target_r
min_vr = float(np.nanmin(vr))
min_r_err = float(np.nanmin(r_err))

# first time v_r flips + -> -
t_flip = None
for i in range(1, len(vr)):
    if np.isfinite(vr[i-1]) and np.isfinite(vr[i]) and (vr[i-1] > 0) and (vr[i] <= 0):
        t_flip = i
        break

# first time crosses target from above to below
t_cross = None
for i in range(1, len(r)):
    if (r[i-1] >= target_r) and (r[i] < target_r):
        t_cross = i
        break

r_err = r - target_r
print("min_r_err =", float(np.min(r_err)))
print("max_r_err =", float(np.max(r_err)))
print("min_vr =", min_vr)
print("min_r_err =", min_r_err)
print("t_flip =", t_flip)
print("t_cross =", t_cross)
print("delta_r =", float(r[-1] - r[0]))
print("rel_delta_r =", float((r[-1] - r[0]) / target_r))