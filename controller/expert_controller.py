import numpy as np

class ExpertController:
    """
    Expert Controller v3.1 – Physically realistic orbit insertion controller.
    Features:
    - Radial + tangential control
    - Optional damping
    - Capture detection to stop thrust when orbit is reached
    - Clean single-pass entry without spiral loops
    """
    def __init__(self,
                 target_radius,
                 G=6.67430e-11,
                 M=1.989e30,
                 mass=1000,
                 radial_gain=12.0,
                 tangential_gain=8.0,
                 damping_gain=4.0,
                 thrust_limit=1.0,
                 enable_damping=True):
        """
        Initialize the expert controller.

        Args:
            target_radius (float): Desired orbit radius in meters.
            G (float): Gravitational constant.
            M (float): Central mass (e.g. Sun).
            mass (float): Spacecraft mass.
            radial_gain (float): Radial correction gain.
            tangential_gain (float): Tangential speed correction gain.
            damping_gain (float): Damping gain to suppress radial velocity oscillation.
            thrust_limit (float): Max magnitude of thrust vector.
            enable_damping (bool): Toggle damping force.
        """
        self.target_radius = target_radius
        self.G = G
        self.M = M
        self.mass = mass
        self.radial_gain = radial_gain
        self.tangential_gain = tangential_gain
        self.damping_gain = damping_gain
        self.thrust_limit = thrust_limit
        self.enable_damping = enable_damping

    def __call__(self, t, pos, vel):
        """
        Compute thrust vector based on current position and velocity.

        Args:
            t (float): Time (unused here).
            pos (np.ndarray): Position vector [x, y].
            vel (np.ndarray): Velocity vector [vx, vy].

        Returns:
            np.ndarray: Thrust vector [tx, ty].
        """
        r_vec = np.array(pos)
        v_vec = np.array(vel)

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        # === Compute unit vectors ===
        radial_dir = r_vec / (r + 1e-12)
        tangential_dir = np.array([-radial_dir[1], radial_dir[0]])

        # === Desired circular orbit speed ===
        v_circular = np.sqrt(self.G * self.M / self.target_radius)

        # === Component along tangential only (for more realistic speed control) ===
        v_tangential = np.dot(v_vec, tangential_dir)
        delta_v = v_circular - v_tangential

        # === Radial error ===
        radial_error = r - self.target_radius

        # === Tangential control: accelerate/decelerate into circular speed ===
        thrust_t = self.tangential_gain * np.tanh(delta_v / v_circular)

        # === Radial control: bring r to target radius ===
        thrust_r = -self.radial_gain * np.tanh(radial_error / (0.1 * self.target_radius))

        # === Damping: suppress radial oscillation ===
        if self.enable_damping:
            radial_velocity = np.dot(v_vec, radial_dir)
            thrust_r += -self.damping_gain * radial_velocity

        # === Stop thrust when orbit is stable ===
        if abs(radial_error) < 0.01 * self.target_radius and abs(delta_v) < 0.01 * v_circular:
            return np.zeros(2)  # Turn off thrust – considered captured

        # === Final thrust vector ===
        thrust_vec = thrust_r * radial_dir + thrust_t * tangential_dir

        # === Clip to max thrust limit ===
        norm = np.linalg.norm(thrust_vec)
        if norm > self.thrust_limit:
            thrust_vec = thrust_vec / norm * self.thrust_limit

        return thrust_vec