import numpy as np
import os
from simulator.simulate_orbit import simulate_orbit
from simulator.visualize import plot_trajectory
from controller.velocity_controller import radial_controller
from simulator.orbit_analysis import evaluate_orbit_error
from controller.velocity_controller import get_thrust
from simulator.visualize import plot_radius_vs_time
from controller.velocity_controller import tangential_controller
from controller.combined_controller import Controller
from data.thrust_dataset import ThrustDataset

use_voyager = True
# Simulation Setup
if use_voyager:
    steps = 6000
    dt = 3600  # Each step is 1 hour (3600 seconds)
    G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
    M = 1.989e30   # Mass of the Sun (kg)
    mass = 722  # Mass of Voyager 1 (kg)
    target_radius = 1.5e11 # The distance from the Sun to the Earth (meters)
    # Voyager 1 Initial Conditions (leaving Earth)
    pos_init = np.array([0.0, 1.5e11])  # 1 AU from the Sun (m)
    vel_init = np.array([17000.0, 0.0])  # 17 km/s outward (m/s)

else:
        G = 1.0
        M = 1000.0
        mass = 1.0
        dt = 0.05
        steps = 6000
        target_radius = 100.0
        pos_init = np.array([target_radius, 0.0])
        vel_init = None


# Mode selection
mode = "impulse"

# Baseline (No thrust)
baseline_traj = simulate_orbit(
    steps = steps,
    dt = dt,
    G = G,
    M = M,
    mass = mass,
    pos_init = pos_init,
    vel_init = vel_init,
    thrust_vector = None
)

# Main controlled trajectory(using selected mode)
controller = Controller(
    continuous = False,
    impulse = True,
    impulse_period = 5.0,
    impulse_duration = 1.0,
    enable_radial=  True,
    enable_tangential = True,
    alpha = 17,
    beta = 17,
    thrust_decay_type = 'exponential',
    decay_rate = 1e-7,
    add_noise = True,
    noise_deg = 10
)

dataset = ThrustDataset()

main_traj = simulate_orbit(
    steps = steps,
    dt = dt,
    G = G,
    M = M,
    mass = mass,
    pos_init = pos_init,
    vel_init = vel_init,
    thrust_vector = lambda  t, pos, vel: dataset(t, pos, vel, controller)
)

dataset.save("radial_decay_with_noise")

# Visualization
plot_trajectory(
    trajectory = main_traj,
    title = "OIntegrated continuous + pulse thrust + radial + tangential + thrust decay",
    target_radius = target_radius,
    arrows = True,
    others = [(baseline_traj, "No Thrust")]
)

# Visualize the r(t) curve graph
plot_radius_vs_time(main_traj, dt, title=f"r(t) vs Time - Mode: {mode}")

# save thrust_log
controller.save_log("radial_noise_decay")

# Save trajectory data
save_path = f"data/saved_trajectories/{mode}_traj"
os.makedirs("data/saved_trajectories", exist_ok=True)
np.save(f"{save_path}.npy", main_traj)
np.savetxt(f"{save_path}.csv", main_traj, delimiter=",")

# Analyze the orbital distance error
mean_error, std_error = evaluate_orbit_error(main_traj, target_radius)
print(f"Mean radial error: {mean_error:.4f}, Std: {std_error:.4f}")