import numpy as np

class ExpertController:
    def __init__(self,
                 target_radius,
                 radial_gain = 0.4,
                 tangential_gain = 0.12,
                 thrust_cap = 0.6):
        """
        Expert controller that corrects radial and tangential errors.
        :param target_radius: desired orbit radius (scalar).
        :param radial_gain: how strongly to correct radial error.
        :param tangential_gain: thrust to maintain circular motion.
        :param thrust_cap: maximum allowed thrust magnitude.
        """
        self.target_radius = target_radius
        self.radial_gain = radial_gain
        self.tangential_gain = tangential_gain
        self.thrust_cap = thrust_cap

    def __call__(self, t, pos, vel):
        r = np.linalg.norm(pos)
        if r == 0:
            return np.array([0.0, 0.0])

        radial_dir = pos / r
        radial_error = r - self.target_radius
        thrust_r = -self.radial_gain * radial_error  # Negative to push inward if too far

        tangential_dir = np.array([-radial_dir[1], radial_dir[0]])
        thrust_t = self.tangential_gain

        thrust = thrust_r * radial_dir + thrust_t * tangential_dir

        # Limit thrust magnitude
        mag = np.linalg.norm(thrust)
        if mag > self.thrust_cap:
            thrust = (thrust / mag) * self.thrust_cap

        return thrust

