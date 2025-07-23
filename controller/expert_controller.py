import numpy as np

class ExpertController:
    def __init__(self,
                 target_radius,
                 radial_gain = 0.4,
                 tangential_gain = 0.12,
                 thrust_cap = 0.6,
                 enable_error_feedback = True,
                 enable_turn_penalty = True,
                 enable_slowdown = True):
        """
        Expert controller with enhanced behavior:
        - Dynamic feedback based on radial error.
        - Slowdown near target orbit.
        - Penalize thrust in direction opposite tp velocity.
        :param target_radius: desired orbit radius (scalar).
        :param radial_gain: how strongly to correct radial error.
        :param tangential_gain: thrust to maintain circular motion.
        :param thrust_cap: maximum allowed thrust magnitude.
        :param enable_error_feedback: Dynamically scale thrust with radial error.
        :param enable_turn_penalty: Penalize thrust opposite to velocity.
        :param enable_slowdown: Reduce thrust when close to target radius
        """
        self.target_radius = target_radius
        self.radial_gain = radial_gain
        self.tangential_gain = tangential_gain
        self.thrust_cap = thrust_cap
        self.enable_error_feedback = enable_error_feedback
        self.enable_turn_penalty = enable_turn_penalty
        self.enable_slowdown = enable_slowdown

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

        if self.enable_error_feedback:
            error_ratio = min(2.0, abs(radial_error) / self.target_radius)
            thrust *= (0.5 + error_ratio)

        if self.enable_slowdown and abs(radial_error) < 0.05 * self.target_radius:
            thrust *= 0.5

        if self.enable_turn_penalty:
            v_norm = np.linalg.norm(vel)
            if v_norm > 1e-3 and np.linalg.norm(thrust) > 1e-5:
                angle_cos = np.dot(thrust, vel) / (np.linalg.norm(thrust) * v_norm + 1e-8)
                if angle_cos < 0:
                    thrust *= 0.3


        # Limit thrust magnitude
        mag = np.linalg.norm(thrust)
        if mag > self.thrust_cap:
            thrust = (thrust / mag) * self.thrust_cap

        return thrust

