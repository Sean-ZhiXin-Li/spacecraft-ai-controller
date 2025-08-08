import numpy as np
import matplotlib.pyplot as plt
import glob

files = glob.glob("data/data/processed/merge_expert_dataset.npy")

for path in files:
    traj = np.load(path)
    r = np.linalg.norm(traj[:, :2], axis=1)
    plt.plot(r)

plt.axhline(y=7.5e12, linestyle='--', color='gray')
plt.title("Expert Trajectories Radius Check")
plt.xlabel("Timestep")
plt.ylabel("Radius")
plt.show()