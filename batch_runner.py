from controller.combined_controller import Controller
from simulator.simulate_orbit import simulate_orbit
import numpy as np

# General simulation parameters
steps = 6000
dt = 10
G = 6.67430e-11
M = 1.989e30
mass = 1000
pos_init = np.array([1.5e11, 0.0])         # Initial position (e.g., Earth distance from Sun)
vel_init = np.array([0.0, 29780.0])        # Initial velocity for near-circular orbit

def run_simulation(config):
    name = config["name"]
    print(f"\n Running simulation: {name}")

    # Create a controller with custom parameters
    controller = Controller(
        continuous=config.get("continuous", False),
        impulse=config.get("impulse", True),
        impulse_period=config.get("impulse_period", 5.0),
        impulse_duration=config.get("impulse_duration", 1.0),
        enable_radial=config.get("enable_radial", True),
        enable_tangential=config.get("enable_tangential", True),
        alpha=config.get("alpha", 17),
        beta=config.get("beta", 17),
        thrust_decay_type=config.get("thrust_decay_type", "none"),
        decay_rate=config.get("decay_rate", 0),
        add_noise=config.get("add_noise", False),
        noise_deg=config.get("noise_deg", 3),
        enable_logging=True
    )

    # Simulate orbit with this controller
    trajectory = simulate_orbit(
        steps = steps,
        dt = dt,
        G = G,
        M = M,
        mass = mass,
        pos_init = pos_init,
        vel_init = vel_init,
        thrust_vector = controller
    )

    # Save thrust log to CSV/NPY
    controller.save_log(path_prefix=name)

if __name__ == "__main__":
    # Define multiple test configurations
    configs = [
        {"name": "radial_fixed", "enable_radial": True, "enable_tangential": False},
        {"name": "tangential_fixed", "enable_radial": False, "enable_tangential": True},
        {"name": "combo_decay", "enable_radial": True, "enable_tangential": True, "thrust_decay_type": "exponential", "decay_rate": 1e-7},
        {"name": "combo_noise", "enable_radial": True, "enable_tangential": True, "add_noise": True, "noise_deg": 8}
    ]

    # Run simulation for each configuration
    for config in configs:
        run_simulation(config)
