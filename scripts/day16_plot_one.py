import numpy as np
import matplotlib.pyplot as plt

p = r"analysis/runs/run_200825/traj.npz"
d = np.load(p)
r = d["r"].astype(float)
vr = d["vr"].astype(float)
target_r = float(d["target_r"])

t = np.arange(len(r))
r_err = r - target_r

plt.figure()
plt.plot(t, r_err)
plt.axhline(0.0, linestyle="--")
plt.xlabel("step")
plt.ylabel("r - target_r")
plt.title("radius error (r - target)")
plt.tight_layout()
plt.savefig("analysis/figs/day16_rerr_t.png", dpi=200)

plt.figure()
plt.plot(t, vr)
plt.axhline(0.0, linestyle="--")
plt.xlabel("step")
plt.ylabel("radial velocity v_r")
plt.title("v_r(t)")
plt.tight_layout()
plt.savefig("analysis/figs/day16_vr_t.png", dpi=200)

print("saved: analysis/figs/day16_rerr_t.png")
print("saved: analysis/figs/day16_vr_t.png")