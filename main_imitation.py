import numpy as np
import os
from simulator.simulate_orbit import simulate_orbit
from simulator.visualize import plot_trajectory, plot_radius_vs_time
from simulator.orbit_analysis import evaluate_orbit_error
#from controller.imitation_controller import ImitationController
from simulator.orbit_analysis import (
    plot_radius_error_with_analysis,
    plot_error_histogram)
from controller.imitation_controller_V6_1 import ImitationController

def run_imitation():
    # Physical constants and mission settings
    G = 6.67430e-11                  # Gravitational constant
    M = 1.989e30                     # Mass of the Sun (kg)
    target_radius = 7.5e12          # Target circular orbit radius (m)
    mass = 721.9                    # Voyager 1 mass (kg)
    dt = 2000                       # Smaller time step (for better resolution)
    steps = 8000000              # Longer simulation to allow spiral out

    # Initial position and boosted velocity
    pos_init = np.array([0.0, target_radius * (1 / 3)])
    r0 = np.linalg.norm(pos_init)
    v_circular = np.sqrt(G * M / r0)
    boost_factor = 1.2
    vel_init = np.array([v_circular * boost_factor, 0.0])

    # Load imitation controller
    controller = ImitationController(
        model_path="controller/mimic_model_V6_1.pth",
    )

    # Simulate orbit
    trajectory = simulate_orbit(
        steps=steps,
        dt=dt,
        G=G,
        M=M,
        mass=mass,
        pos_init=pos_init,
        vel_init=vel_init,
        thrust_vector=lambda t, pos, vel: controller(t, pos, vel)
    )

    # Save trajectory
    os.makedirs("data/logs", exist_ok=True)
    np.save("data/logs/imitation_traj_V6.1_long.npy", trajectory)

    # Visualizations
    plot_trajectory(
        trajectory,
        title="V6.1 Controller – Long Simulation (Spiral Transfer)",
        target_radius=target_radius
    )
    plot_radius_vs_time(
        trajectory,
        dt,
        title="r(t) vs Time – V6.1 Imitation Controller"
    )

    os.makedirs("plots", exist_ok=True)
    plot_radius_error_with_analysis(
        trajectory,
        target_radius,
        save_path="plots/error_v6.1_long.png"
    )
    plot_error_histogram(
        trajectory,
        target_radius,
        save_path="plots/hist_v6.1_long.png"
    )

    # Evaluate final performance
    mean_error, std_error = evaluate_orbit_error(trajectory, target_radius)
    print(f"[V6.1 Long Run] Mean radial error: {mean_error:.2e} m, Std: {std_error:.2e} m")

if __name__ == "__main__":
    run_imitation()

