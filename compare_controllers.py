import numpy as np
import matplotlib.pyplot as plt
import os

from simulator.simulate_orbit import simulate_orbit
from simulator.visualize import plot_trajectory, plot_radius_vs_time
from simulator.orbit_analysis import plot_radius_error, evaluate_orbit_error
from controller.expert_controller import ExpertController
from controller.imitation_controller import ImitationController
from simulator.orbit_analysis import (
    plot_radius_error_with_analysis,
    plot_error_histogram,
    analyze_error_stats
)

# Config
steps = 600000
dt = 200
G = 6.67430e-11
M = 1.989e30
mass = 721.9
target_radius = 7.5e12
pos_init = np.array([0.0, target_radius * (1 / 3)])
r0 = np.linalg.norm(pos_init)
v_circular = np.sqrt(G * M / r0)
angle_deg = 30
angle_rad = np.deg2rad(angle_deg)
vel_direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
boost_factor = 1.2
vel_init = boost_factor * v_circular * vel_direction

# Load imitation trajectory
imitation_traj = np.load("data/logs/imitation_traj_V5_long.npy")
imitation_traj = imitation_traj[:steps]

print("Loaded imitation_traj steps:", imitation_traj.shape[0])
print("Expected steps:", steps)

if imitation_traj.shape[0] > steps:
    imitation_traj = imitation_traj[:steps]
elif imitation_traj.shape[0] < steps:

    pad_len = steps - imitation_traj.shape[0]
    pad_value = imitation_traj[-1]
    imitation_traj = np.vstack([imitation_traj, np.repeat([pad_value], pad_len, axis=0)])
    print(f"Warning: imitation_traj was shorter than steps, padded last value for {pad_len} steps.")

# Generate expert trajectory
expert_controller = ExpertController(
    target_radius=target_radius,
    G=G,
    M=M
)
expert_traj = simulate_orbit(
    steps=steps,
    dt=dt,
    G=G,
    M=M,
    mass=mass,
    pos_init=pos_init,
    vel_init=vel_init,
    thrust_vector=expert_controller
)

assert imitation_traj.shape[0] == expert_traj.shape[0], "Trajectory length mismatch!"

# Plot combined trajectory
plot_trajectory(
    trajectory=imitation_traj,
    title="Trajectory Comparison: Imitation vs Expert",
    target_radius=target_radius,
    arrows=True,
    others=[(expert_traj, "Expert Controller")]
)


# Plot r(t) comparison
def plot_radius_vs_time_comparison(traj1, traj2, dt, label1, label2, title):
    t = np.arange(len(traj1)) * dt
    r1 = np.linalg.norm(traj1, axis=1)
    r2 = np.linalg.norm(traj2, axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(t, r1, label=label1, color="red")
    plt.plot(t, r2, label=label2, color="blue")
    plt.axhline(target_radius, color='gray', linestyle='--', label="Target Radius")
    plt.xlabel("Time (s)")
    plt.ylabel("r(t) (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("comparison_radius.png")
    plt.show()


plot_radius_vs_time_comparison(
    imitation_traj, expert_traj, dt,
    label1="Imitation Controller",
    label2="Expert Controller",
    title="r(t) vs Time: Imitation vs Expert"
)


# Plot radial error curves
def plot_radial_error_comparison(traj1, traj2, label1, label2):
    r1 = np.linalg.norm(traj1, axis=1)
    r2 = np.linalg.norm(traj2, axis=1)
    error1 = r1 - target_radius
    error2 = r2 - target_radius

    plt.figure(figsize=(10, 5))
    plt.plot(error1, label=label1, color="red")
    plt.plot(error2, label=label2, color="blue")
    plt.axhline(0, color='gray', linestyle='--', label="Target Radius")
    plt.xlabel("Time Step")
    plt.ylabel("Radial Error (m)")
    plt.title("Radial Error: Imitation vs Expert")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("comparison_error.png")
    plt.show()


plot_radial_error_comparison(
    imitation_traj, expert_traj,
    label1="Imitation Controller",
    label2="Expert Controller"
)

# Evaluate and print errors
mean_imitation, std_imitation = evaluate_orbit_error(imitation_traj, target_radius)
mean_expert, std_expert = evaluate_orbit_error(expert_traj, target_radius)

print("\n=== Error Summary ===")
print(f"[Imitation Controller] Mean: {mean_imitation:.2e}, Std: {std_imitation:.2e}")
print(f"[Expert   Controller] Mean: {mean_expert:.2e}, Std: {std_expert:.2e}")
plot_radius_error_with_analysis(imitation_traj, target_radius, save_path="enhanced_error_imitation.png")
plot_error_histogram(imitation_traj, target_radius, save_path="error_hist_imitation.png")
analyze_error_stats(imitation_traj, target_radius)

# Save trajectories
os.makedirs("data/logs", exist_ok=True)
np.save("data/logs/expert_traj.npy", expert_traj)
np.save("data/logs/imitation_traj.npy", imitation_traj)
