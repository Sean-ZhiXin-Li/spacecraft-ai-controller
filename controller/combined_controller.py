import numpy as np
import os
from controller.velocity_controller import radial_controller
from controller.velocity_controller import tangential_controller
from controller.perturb import add_attitude_noise

class Controller:
    def __init__(self,
                 continuous = False,
                 impulse = True,
                 impulse_period = 5.0,
                 impulse_duration = 1.0,
                 enable_radial = True,
                 enable_tangential = True,
                 alpha = 17,
                 beta = 17,
                 thrust_decay_type = 'none',
                 decay_rate = 1e-6,
                 add_noise = False,
                 noise_deg = 3,
                 enable_logging=False):
        self.continuous = continuous
        self.impulse = impulse
        self.impulse_period = impulse_period
        self.impulse_duration = impulse_duration
        self.enable_radial = enable_radial
        self.enable_tangential = enable_tangential
        self.alpha = alpha
        self.beta = beta
        self.thrust_decay_type = thrust_decay_type
        self.decay_rate = decay_rate
        self.add_noise = add_noise
        self.noise_deg = noise_deg
        self.enable_logging = enable_logging
        self.log = []
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
            add_noise: Add a perturbation to the attitude.
            noise_deg: angle of deviation in degrees.

        Returns:
            np.ndarray: Final thrust vector [Tx, Ty].
        """
    def get_thrust(self, t, pos, vel):
        if not self.continuous and not self.impulse:
            return np.array([0.0, 0.0])

        if  self.impulse and (t % self.impulse_period >= self.impulse_duration):
            return np.array([0.0, 0.0])

        thrust = np.array([0.0, 0.0])
        if self.enable_radial:
            thrust += self.alpha * radial_controller(t, pos, vel)
        if self.enable_tangential:
            thrust += self.beta * tangential_controller(t, pos, vel)

        if self.thrust_decay_type == 'linear':
            decay_factor = max(0.0, 1.0 - self.decay_rate * t)
            thrust *= decay_factor
        elif self.thrust_decay_type == 'exponential':
            decay_factor = np.exp(-self.decay_rate * t)
            thrust *= decay_factor

        if self.add_noise:
            thrust = add_attitude_noise(thrust, max_angle_deg = self.noise_deg)

        if self.enable_logging:
            self.log.append([
                t,
                pos[0], pos[1],
                vel[0], vel[1],
                thrust[0], thrust[1]
            ])

        return thrust

    def __call__(self, t, pos, vel):
        return self.get_thrust(t, pos, vel)

    def save_log(self,path_prefix = "thrust_log"):
        log_array = np.array(self.log)
        os.makedirs("data/logs", exist_ok = True)
        np.save(f"data/logs/{path_prefix}.npy", log_array)
        np.savetxt(
            f"data/logs/{path_prefix}.csv",
            log_array,
            delimiter = ",",
            header = "t, pos_x, pos_y, vel_x, vel_y, thrust_x, thrust_y",
            comments = ''
        )
        print(f"[✅] Thrust log saved to: data/logs/{path_prefix}.csv/.npy")