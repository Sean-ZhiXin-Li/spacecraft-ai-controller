import gym
from gym import spaces
import numpy as np
from typing import Optional, Tuple
from gym.core import ObsType

class OrbitEnv(gym.Env):
    """
    Custom Gym environment simulating 2D orbital control for a spacecraft.
    The agent applies thrust to maintain or reach a target orbital radius.
    """

    def __init__(self,
                 G = 6.67430e-11,            # Gravitational constant (m^3 kg^-1 s^-2)
                 M = 1.989e30,               # Mass of the central body (e.g., the Sun)
                 mass = 722,                 # Spacecraft mass (kg)
                 dt = 600,                   # Simulation time step (600 seconds = 10 minutes)
                 max_steps = 8000,           # Max steps per episode
                 target_radius = 7.5e12,     # Target orbital radius (meters)
                 thrust_scale = 10.0):        # Scaling factor for thrust vector
        super().__init__()

        self.G = G
        self.M = M
        self.mass = mass
        self.dt = dt
        self.max_steps = max_steps
        self.target_radius = target_radius
        self.thrust_scale = thrust_scale

        # Action space: 2D continuous thrust vector in range [-1, 1]^2
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: [x, y, vx, vy] — position and velocity in 2D
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Initialize state
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        """
        Resets the environment to the initial orbital state.
        The spacecraft starts near the target orbit with circular velocity.
        """
        super().reset(seed=seed)
        self.steps = 0

        # Start closer to the target orbit for better convergence
        self.pos = np.array([0.0, 1.1 * self.target_radius], dtype=np.float64)


        r = np.linalg.norm(self.pos)
        v_orbit = np.sqrt(self.G * self.M / r)

        # Initial velocity orthogonal to position (circular orbit)
        self.vel = np.array([v_orbit, 0.0])

        return self._get_obs(), {}

    def _get_obs(self) -> ObsType:
        """
        Returns the current observation: position and velocity vector [x, y, vx, vy].
        """
        return np.concatenate([self.pos, self.vel]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, dict]:
        """
        Advances the environment one time step using the given thrust vector.
        """
        self.steps += 1

        # Apply thrust vector (scaled)
        thrust = self.thrust_scale * np.clip(action, -1.0, 1.0)
        acc_thrust = thrust / self.mass

        # Compute gravitational acceleration
        r_vec = self.pos
        r = np.linalg.norm(r_vec)
        acc_gravity = -self.G * self.M * r_vec / ((r ** 3) + 1e-8)
        acc_gravity = np.clip(acc_gravity, -1e-2, 1e-2)

        # Euler integration for velocity and position
        self.vel += (acc_gravity + acc_thrust) * self.dt
        self.pos += self.vel * self.dt

        # Compute orbital error metrics
        curr_radius = np.linalg.norm(self.pos)
        radius_error = abs(curr_radius - self.target_radius)

        v_actual = np.linalg.norm(self.vel)
        v_circular = np.sqrt(self.G * self.M / curr_radius)
        speed_error = abs(v_actual - v_circular)

        escape_speed = np.sqrt(2 * self.G * self.M / self.target_radius)

        # Reward Function
        reward = 0.00

        # Penalize radius and speed deviation
        reward -= 5.0 * (radius_error / self.target_radius)
        reward -= 0.3 * (speed_error / v_circular)

        # Bonus for being close to circular orbit
        if radius_error < 1e10 and speed_error < 300:
            reward += 20.0

        reward += 2.0 * np.exp(-radius_error / 1e11)

        # Mild penalty for leaving target orbit range
        if curr_radius > 2.5 * self.target_radius or v_actual > escape_speed:
            reward -= 10.0

        fuel_penalty = 0.05 * np.linalg.norm(thrust)
        reward -= fuel_penalty

        # Angular alignment bonus (cosine similarity)
        cos_theta = np.dot(self.pos, self.vel) / (np.linalg.norm(self.pos) * np.linalg.norm(self.vel) + 1e-8)
        reward += 0.3 * cos_theta

        # Optional logging every 1000 steps
        if self.steps % 200 == 0:
            print(f"[Step {self.steps}] r = {curr_radius:.2e}, v = {v_actual:.2e}, reward = {reward:.2f}")

        # Episode done after max_steps
        done = self.steps >= self.max_steps

        if self.steps % 100 == 0:
            print(
                f"[Step {self.steps}] radius error = {radius_error:.2e}, speed error = {speed_error:.2f}, reward = {reward:.2f}")

        return self._get_obs(), reward, done, {}

    def render(self, mode="human"):
        print(f"Step {self.steps}: pos={self.pos}, vel={self.vel}")

