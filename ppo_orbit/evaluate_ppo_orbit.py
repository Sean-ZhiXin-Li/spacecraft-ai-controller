import torch
import numpy as np
import matplotlib.pyplot as plt
from envs.orbit_env import OrbitEnv
from simulator.visualize import (
    plot_trajectory,
    plot_radius_vs_time,
    plot_thrust_quiver
)
from simulator.orbit_analysis import evaluate_orbit_error
from ppo_orbit.ppo import ActorCritic

# Load trained PPO model
model = ActorCritic()
model.load_state_dict(torch.load("ppo_best_model.pth", weights_only=True))
model.eval()
print("PPO model loaded successfully.")

# Initialize orbital simulation environment
env = OrbitEnv()
state, _ = env.reset()  # Correct unpacking
state = np.array(state)  # Ensure it’s a flat array
state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

# Run simulation with PPO controller
states = []  # To record full states [x, y, vx, vy, Tx, Ty]
for _ in range(8000):  # Match simulation length (steps)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        dist, _ = model(state_tensor)
        action = dist.sample().squeeze().numpy()
    next_state, reward, done, _ = env.step(action)

    # Record current state + action (used for visualization)
    full_state = np.concatenate([state, action])  # [x, y, vx, vy, Tx, Ty]
    states.append(full_state)

    state = next_state
    if done:
        break

states = np.array(states)
trajectory = states[:, :2]  # Extract positions [x, y]

# Plot orbital trajectory
plot_trajectory(
    trajectory,
    title="PPO-Controlled Orbit",
    target_radius=env.target_radius,
    arrows=True
)

# Plot radius over time (r(t))
plot_radius_vs_time(
    trajectory,
    dt=env.dt,
    title="Radius vs Time (PPO)"
)

# === Step 6: Visualize thrust vector field ===
plot_thrust_quiver(
    states,
    title="Thrust Vector Field (PPO)",
    step=200  # Plot every N-th step for clarity
)

# Evaluate orbital performance (mean error, std deviation)
evaluate_orbit_error(trajectory, env.target_radius)
