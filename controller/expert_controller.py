import numpy as np

class ExpertController:
    def __init__(self,
                 target_radius,
                 G,
                 M,
                 radial_gain=0.3,
                 tangential_gain=0.05,
                 thrust_cap=0.2,
                 enable_error_feedback=True,
                 enable_slowdown=True,
                 enable_turn_penalty=True):
        self.target_radius = target_radius
        self.G = G
        self.M = M
        self.radial_gain = radial_gain
        self.tangential_gain = tangential_gain
        self.thrust_cap = thrust_cap
        self.enable_error_feedback = enable_error_feedback
        self.enable_slowdown = enable_slowdown
        self.enable_turn_penalty = enable_turn_penalty

    def __call__(self, t, pos, vel):
        r_vec = pos
        v_vec = vel
        r = np.linalg.norm(r_vec)
        v_mag = np.linalg.norm(v_vec)

        radial_dir = r_vec / (r + 1e-8)
        tangential_dir = np.array([-radial_dir[1], radial_dir[0]])

        v_circular = np.sqrt(self.G * self.M / r)

        # === Radial thrust ===
        radial_error = r - self.target_radius
        thrust_r = -self.radial_gain * np.tanh(radial_error / (0.05 * self.target_radius))

        # === Tangential thrust ===
        delta_v = v_circular - v_mag
        thrust_t = self.tangential_gain * np.tanh(delta_v / (0.1 * v_circular))

        # === Combine thrust ===
        thrust = thrust_r * radial_dir + thrust_t * tangential_dir

        # === Error feedback boost ===
        if self.enable_error_feedback:
            error_ratio = np.clip(abs(radial_error) / self.target_radius, 0.0, 2.0)
            thrust *= (0.4 + error_ratio)

        # === Near-target slowdown ===
        if self.enable_slowdown and abs(radial_error) < 0.02 * self.target_radius:
            thrust *= 0.4

        # === Turn penalty ===
        if self.enable_turn_penalty:
            v_norm = np.linalg.norm(vel)
            thrust_norm = np.linalg.norm(thrust)
            if v_norm > 1e-3 and thrust_norm > 1e-5:
                angle_cos = np.dot(thrust, vel) / (v_norm * thrust_norm + 1e-8)
                penalty = 1.0 + 0.5 * angle_cos
                thrust *= max(0.1, penalty)

        # === Cap thrust ===
        thrust_mag = np.linalg.norm(thrust)
        if thrust_mag > self.thrust_cap:
            thrust = thrust / thrust_mag * self.thrust_cap

        return thrust

