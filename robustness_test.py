import numpy as np
import matplotlib.pyplot as plt
from controller.combined_controller import smart_combined_controller
from simulator.simulate_orbit import simulate_orbit
from simulator.orbit_analysis import evaluate_orbit_error

# Robustness Test Script
# This script compares the performance of a thrust controller with and without
# attitude noise (directional thrust deviation). It evaluates the difference
# in radial error and visualizes the effect over time.

# Simulation parameters
steps = 6000
dt = 3600  # 1 hour per step
G = 6.67430e-11         # Gravitational constant
M = 1.989e30            # Mass of the Sun
mass = 722              # Mass of spacecraft (Voyager)
target_radius = 1.5e11  # Target orbital radius (1 AU)
pos_init = np.array([0.0, target_radius])
vel_init = np.array([17000.0, 0.0])  # Initial outward velocity (m/s)

# Simulation: No Attitude Noise
traj_clean = simulate_orbit(
    steps=steps, dt=dt, G=G, M=M, mass=mass,
    pos_init=pos_init, vel_init=vel_init,
    thrust_vector=lambda t, pos, vel: smart_combined_controller(
        t, pos, vel,
        continuous=False,
        impulse=True,
        impulse_period=5.0,
        impulse_duration=1.0,
        enable_radial=True,
        enable_tangential=False,
        thrust_decay_type='exponential',
        decay_rate=1e-7,
        add_noise=False
    )
)

# Simulation: With Attitude Noise
traj_noisy = simulate_orbit(
    steps=steps, dt=dt, G=G, M=M, mass=mass,
    pos_init=pos_init, vel_init=vel_init,
    thrust_vector=lambda t, pos, vel: smart_combined_controller(
        t, pos, vel,
        continuous=False,
        impulse=True,
        impulse_period=5.0,
        impulse_duration=1.0,
        enable_radial=True,
        enable_tangential=False,
        thrust_decay_type='exponential',
        decay_rate=1e-7,
        add_noise=True,
        noise_deg=45
    )
)

# Evaluate radial error
mean_clean, std_clean = evaluate_orbit_error(traj_clean, target_radius)
mean_noisy, std_noisy = evaluate_orbit_error(traj_noisy, target_radius)

print(" No Attitude Noise:")
print(f"  Mean Radial Error: {mean_clean:.4e}")
print(f"  Std  Radial Error: {std_clean:.4e}")

print(" With Attitude Noise:")
print(f"  Mean Radial Error: {mean_noisy:.4e}")
print(f"  Std  Radial Error: {std_noisy:.4e}")

print(" Difference:")
print(f"  ΔMean: {mean_noisy - mean_clean:.4e}")
print(f"  ΔStd:  {std_noisy - std_clean:.4e}")

# Plot radial error comparison between clean and noisy trajectories
r_clean = np.linalg.norm(traj_clean, axis=1)
r_noisy = np.linalg.norm(traj_noisy, axis=1)
errors_clean = np.abs(r_clean - target_radius)
errors_noisy = np.abs(r_noisy - target_radius)

plt.figure(figsize=(10, 4))
plt.plot(errors_clean, label="No Noise", linewidth=2)
plt.plot(errors_noisy, label="With Noise", linewidth=2)
plt.title("Radial Error Comparison Over Time")
plt.xlabel("Time Step")
plt.ylabel("Radial Error (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
