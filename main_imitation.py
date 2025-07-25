import numpy as np
import os

from data.generate_dataset import steps
from simulator.simulate_orbit import simulate_orbit
from simulator.visualize import plot_trajectory, plot_radius_vs_time
from simulator.orbit_analysis import evaluate_orbit_error
from controller.imitation_controller import ImitationController
from simulator.orbit_analysis import (
    plot_radius_error_with_analysis,
    plot_error_histogram)
def run_imitation():
    # Set simulation parameters
    steps = 60000
    dt = 3600
    G = 6.67430e-11
    M = 1.989e30
    mass = 722
    target_radius = 7.5e12
    pos_init = np.array([0.0, 1.5e11])
    vel_init = np.array([20000.0, 0.0])

    # Load my imitation-based controller
    controller = ImitationController("imitation_policy_model_V3.1.joblib")

    # Simulate the orbit under AI control
    trajectory = simulate_orbit(
        steps = steps,
        dt = dt,
        G = G,
        M = M,
        mass = mass,
        pos_init = pos_init,
        vel_init = vel_init,
        thrust_vector = lambda t, pos, vel: controller(t, pos,vel)
    )

    # Save the trajectory to a file
    os.makedirs("data/logs", exist_ok=True)
    np.save("data/logs/imitation_traj.npy", trajectory)

    # Visualize the results
    plot_trajectory(
        trajectory,
        title="Orbit with Imitation Controller",
        target_radius=target_radius
    )
    plot_radius_vs_time(
        trajectory,
        dt,
        title="r(t) vs Time – Imitation Controller"
    )

    plot_radius_error_with_analysis(trajectory, target_radius, save_path="enhanced_error_v3.1.png")
    plot_error_histogram(trajectory, target_radius, save_path="error_hist_v3.1.png")

    # Evaluate the orbit error
    mean_error, std_error = evaluate_orbit_error(trajectory, target_radius)
    print(f" AI Control Result] Mean radial error: {mean_error:.2e} m, Std: {std_error:.2e} m")


if __name__ == "__main__":
    run_imitation()
