import numpy as np
import os
from simulator.simulate_orbit import simulate_orbit
from controller.expert_controller import ExpertController
from data.thrust_dataset import ThrustDataset

os.makedirs("data/dataset", exist_ok= True)

# Simulation settings
steps = 60000
dt = 3600  # 1 hour per step
G = 6.67430e-11
M = 1.989e30  # Sun
mass = 722  # Voyager
target_radius = 7.5e12
pos_init = np.array([0.0, 1.5e11])
speed = 20000.0  # 20 km/s

# Generate N trajectories with varied initial velocity angle
N = 30
angles_deg = np.linspace(-30, 30, N)

for i, angle_deg, in enumerate(angles_deg):
    angle_rad = np.deg2rad(angle_deg)
    vel_direction = np.array([
        np.cos(angle_rad),
        np.sin(angle_rad)
    ])
    vel_init = speed * vel_direction

    # Create controller and dataset
    controller = ExpertController(target_radius= target_radius)
    dataset = ThrustDataset()

    # Run simulation with controller wrapped by dataset
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

    # Save dataset
    filename = f"expert_dataset_{i + 1:02d}"
    dataset.save(filename)
    print(f"Saved: data/dataset/{filename}.npy")

print(" All expert datasets generated.")