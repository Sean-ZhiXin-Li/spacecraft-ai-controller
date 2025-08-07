import numpy as np
import matplotlib.pyplot as plt
import os

# Load trajectory
traj = np.load("ppo_traj.npy")  # Make sure this path is correct
x, y = traj[:, 0], traj[:, 1]
r = np.sqrt(x**2 + y**2)
t = np.arange(len(r))  # Time step index

# Optional parameters
target_radius = 7.5e12
star_pos = (0, 0)

# Print debug info
print(f"Trajectory length: {len(traj)}")
print(f"Start: ({x[0]:.2e}, {y[0]:.2e})")
print(f"End:   ({x[-1]:.2e}, {y[-1]:.2e})")
print(f"Max radius: {np.max(r):.2e}")
print(f"Min radius: {np.min(r):.2e}")

# Create output folder
os.makedirs("plots", exist_ok=True)

# Plot 1: 2D Orbit Trajectory
plt.figure(figsize=(6, 6))
plt.plot(x, y, 'b-', linewidth=1, label="Trajectory Line")
plt.plot(x, y, 'b.', markersize=1, label="Trajectory Points")

# Target orbit (dashed circle)
target_circle = plt.Circle(star_pos, target_radius, color='gray', linestyle='--', fill=False, label="Target Orbit")
plt.gca().add_patch(target_circle)

# Star, Start, End
plt.plot(0, 0, 'yo', label="Star", markersize=8)
plt.plot(x[0], y[0], 'go', label="Start", markersize=8)
plt.plot(x[-1], y[-1], 'ro', label="End", markersize=8)

plt.xlabel("x position (m)")
plt.ylabel("y position (m)")
plt.title("Final Orbit Controlled by PPO")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/ppo_orbit_final.png")
plt.show()

# Plot 2: Radius over Time
plt.figure(figsize=(8, 4))
plt.plot(t, r, label="Radius r(t)")
plt.hlines(target_radius, 0, len(r), colors='gray', linestyles='--', label="Target Radius")
plt.xlabel("Time Step")
plt.ylabel("Radius (m)")
plt.title("Orbit Radius Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/ppo_radius_over_time.png")
plt.show()
