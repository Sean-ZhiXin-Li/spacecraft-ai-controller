import numpy as np
import os
from simulator.simulate_orbit import simulate_orbit
from simulator.visualize import plot_trajectory
from controller.velocity_controller import velocity_direction_controller
from simulator.orbit_analysis import evaluate_orbit_error

# Simulation Setup
steps = 6000
dt = 0.05
G = 1.0
M = 1000.0
mass = 1.0
target_radius = 100.0

# Initial Conditions
pos_init = np.array([target_radius, 0.0])
vel_init = None  # Let simulate_orbit compute orbital velocity

# Baseline (No thrust)
baseline_traj = simulate_orbit(
    steps=steps,
    dt=dt,
    G=G,
    M=M,
    mass=mass,
    pos_init=pos_init,
    vel_init=vel_init,
    thrust_vector=None
)

# --- Main controlled trajectory ---
main_traj = simulate_orbit(
    steps=steps,
    dt=dt,
    G=G,
    M=M,
    mass=mass,
    pos_init=pos_init,
    vel_init=vel_init,
    thrust_vector=velocity_direction_controller
)

# --- Visualization ---
plot_trajectory(
    trajectory=main_traj,
    title="Orbit with Velocity-Direction Thrust",
    target_radius=target_radius,
    arrows=True,
    others=[(baseline_traj, "No Thrust")]
)

# Save trajectory data
np.save("data/saved_trajectories/main_traj.npy", main_traj)
np.savetxt("data/saved_trajectories/main_traj.csv", main_traj, delimiter=",")

# Analyze the orbital distance error
mean_error, std_error = evaluate_orbit_error(main_traj, target_radius)
print(f"Mean radial error: {mean_error:.4f}, Std: {std_error:.4f}")