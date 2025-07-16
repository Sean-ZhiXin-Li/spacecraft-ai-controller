import numpy as np
from simulator.simulate_orbit import simulate_orbit
from simulator.visualize import plot_trajectory

# Simulation Setup
steps = 6000
dt = 0.05
G = 1.0
M = 1000.0
mass = 1.0
target_radius = 100.0

# Initial Conditions
pos_init = np.array([target_radius, 0.0])
vel_init = None  # Let simulate_orbit compute orbital velocity automatically

# Baseline: No thrust
baseline_traj = simulate_orbit(
    steps=steps,
    dt=dt,
    G=G,
    M=M,
    mass=mass,
    pos_init=pos_init,
    vel_init=vel_init,
    thrust_vector=np.array([0.0, 0.0])
)

# Thrust Scenario: Constant thrust up
main_traj = simulate_orbit(
    steps=steps,
    dt=dt,
    G=G,
    M=M,
    mass=mass,
    pos_init=pos_init,
    vel_init=vel_init,
    thrust_vector=np.array([0.0, 0.002])  # constant upward thrust
)

# Plot both trajectories
plot_trajectory(
    trajectory=main_traj,
    title="Orbit with Constant Thrust vs No Thrust",
    target_radius=target_radius,
    arrows=True,
    others=[(baseline_traj, "No Thrust")]
)
