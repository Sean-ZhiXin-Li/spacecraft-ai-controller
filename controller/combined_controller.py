import numpy as np
from controller.velocity_controller import radial_controller
from controller.velocity_controller import tangential_controller

def smart_combined_controller(
        t, pos, vel,
        continuous = False,
        impulse = True,
        impulse_period = 5.0,
        impulse_duration = 1.0,
        enable_radial = True,
        enable_tangential = True,
        alpha = 17,
        beta = 17,
        thrust_decay_type = 'none',
        decay_rate = 1e-6
):
    """
        General controller that supports:
        - Continuous mode
        - Impulse mode (if enabled)
        - Radial + Tangential thrust combinations

        Args:
            t (float): Current simulation time.
            pos (np.ndarray): Current position vector [x, y].
            vel (np.ndarray): Current velocity vector [vx, vy].
            continuous (bool): If True, always apply thrust.
            impulse (bool): If True, apply periodic impulse thrust.
            impulse_period (float): Period of impulse mode.
            impulse_duration (float): Active duration within each impulse period.
            enable_radial (bool): If True, apply radial thrust component.
            enable_tangential (bool): If True, apply tangential thrust component.
            alpha (float): Radial thrust scaling factor.
            beta (float): Tangential thrust scaling factor.
            thrust_decay_type (str): Type of thrust decay over time.
            - 'none': No decay (constant thrust).
            - 'linear': Linearly decreasing thrust: max(0, 1 - decay_rate * t).
            - 'exponential': Exponentially decreasing thrust: exp(-decay_rate * t).

                decay_rate (float): Rate of decay. Determines how fast the thrust decreases over time.
                Ignored if thrust_decay_type is 'none'.

        Returns:
            np.ndarray: Final thrust vector [Tx, Ty].
        """
    if not continuous and not impulse:
        return np.array([0.0, 0.0])

    if  impulse and (t % impulse_period >= impulse_duration):
        return np.array([0.0, 0.0])

    thrust = np.array([0.0, 0.0])
    if enable_radial:
        thrust += alpha * radial_controller(t, pos, vel)
    if enable_tangential:
        thrust += beta * tangential_controller(t, pos, vel)

    if thrust_decay_type == 'linear':
        decay_factor = max(0.0, 1.0 - decay_rate * t)
        thrust *= decay_factor
    elif thrust_decay_type == 'exponential':
        decay_factor = np.exp(-decay_rate * t)
        thrust *= decay_factor

    return thrust