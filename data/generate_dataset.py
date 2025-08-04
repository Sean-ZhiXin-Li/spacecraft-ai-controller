import numpy as np
import os
from simulator.simulate_orbit import simulate_orbit
from controller.expert_controller import ExpertController
from data.thrust_dataset import ThrustDataset

# Create output directory if not exists
os.makedirs("data/dataset", exist_ok=True)

# == Physical constants and mission settings ==
G = 6.67430e-11                # Gravitational constant
M = 1.989e30                   # Mass of the Sun (kg)
target_radius = 7.5e12        # Target circular orbit radius (m)
mass = 721.9                  # Mass of spacecraft (Voyager 1 approx., kg)
dt = 2000                     # Simulation time step (seconds)
steps = 10000000             # Total simulation steps

# Initial position: 1/3 of target orbit
pos_init = np.array([0.0, target_radius * (1 / 3)])
r0 = np.linalg.norm(pos_init)
v_circular = np.sqrt(G * M / r0)  # Circular orbital velocity at initial radius
boost_factor = 1.2                # Initial velocity boost factor (>1 to reach higher orbit)

# == Generate N trajectories with different launch angles ==
N = 30
angles_deg = np.linspace(-30, 30, N)  # From -30° to +30°

for i, angle_deg in enumerate(angles_deg):
    angle_rad = np.deg2rad(angle_deg)
    vel_direction = np.array([
        np.cos(angle_rad),
        np.sin(angle_rad)
    ])
    vel_init = boost_factor * v_circular * vel_direction

    # Initialize the expert controller with tuned parameters
    controller = ExpertController(
        target_radius=target_radius,
        G=G,
        M=M,
        mass=mass,
        radial_gain=4.0,
        tangential_gain=5.0,
        damping_gain=6.0,
        thrust_limit=20.0,
        enable_damping=True
    )

    # Create dataset wrapper to record thrust actions
    dataset = ThrustDataset()

    # Run simulation with expert controller
    simulate_orbit(
        steps=steps,
        dt=dt,
        G=G,
        M=M,
        mass=mass,
        pos_init=pos_init,
        vel_init=vel_init,
        thrust_vector=lambda t, pos, vel: dataset(t, pos, vel, controller)
    )

    # Save the expert trajectory dataset
    filename = f"expert_dataset_{i + 1:02d}"
    dataset.save(filename)
    print(f"Saved: data/dataset/{filename}.npy")

print("All expert datasets successfully generated.")
