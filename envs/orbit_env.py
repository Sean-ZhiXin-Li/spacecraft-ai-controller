import gym
from gym import spaces
import numpy as np
from typing import Optional, Tuple
from gym.core import ObsType

class OrbitEnv(gym.Env):
    """
    A custom Gym environment simulating 2D spacecraft orbital control.
    The agent applies thrust to maintain or reach a desired orbital radius.
    """

    def __init__(self,
                 G=6.67430e-11,           # Gravitational constant (m^3 kg^-1 s^-2)
                 M=1.989e30,              # Mass of the central body (e.g., Sun)
                 mass=722,                # Spacecraft mass in kg
                 dt=3600,                 # Time step in seconds (1 hour)
                 max_steps=120000,         # Maximum simulation steps per episode
                 target_radius=7.5e12,    # Target orbital radius (meters)
                 thrust_scale=0.2):       # Scaling factor for thrust magnitude
        super().__init__()

        # Physical constants
        self.G = G
        self.M = M
        self.mass = mass
        self.dt = dt
        self.max_steps = max_steps
        self.target_radius = target_radius
        self.thrust_scale = thrust_scale

        # Observation space: [x, y, vx, vy]
        # No bounds on position or velocity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Action space: 2D thrust vector in range [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Initialize simulation state
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        """
        Reset the environment to its initial state.

        Returns:
            observation (np.ndarray): The initial state [x, y, vx, vy]
            info (dict): Additional info (empty here)
        """
        super().reset(seed=seed)
        self.steps = 0

        self.pos = np.array([0.0, 2.0e12], dtype=np.float64)

        v_orbit = np.sqrt(self.G * self.M / np.linalg.norm(self.pos))
        self.vel = np.array([v_orbit * 0.95, 0.0])

        # Return initial observation
        return self._get_obs(), {}

    def _get_obs(self) -> ObsType:
        """
        Generate current observation.

        Returns:
            observation (np.ndarray): Flattened array [x, y, vx, vy]
        """
        return np.concatenate([self.pos, self.vel]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, dict]:
        """
        Perform one simulation step given an agent's action.

        Args:
            action (np.ndarray): Thrust vector in range [-1, 1]^2

        Returns:
            observation (np.ndarray): New state
            reward (float): Reward signal
            done (bool): Whether the episode is over
            info (dict): Additional debug info
        """
        self.steps += 1
        done = False

        thrust = self.thrust_scale * np.clip(action, -1.0, 1.0)
        acc_thrust = thrust / self.mass

        r = np.linalg.norm(self.pos)
        acc_gravity = -self.G * self.M * self.pos / r ** 3 if r != 0 else np.zeros(2)

        self.vel += (acc_gravity + acc_thrust) * self.dt
        self.pos += self.vel * self.dt

        curr_radius = np.linalg.norm(self.pos)
        radius_error = abs(curr_radius - self.target_radius)

        v_actual = np.linalg.norm(self.vel)
        v_circular = np.sqrt(self.G * self.M / curr_radius)
        speed_error = abs(v_actual - v_circular)

        reward = 0.0
        reward -= 1.5 * (radius_error / self.target_radius)
        reward -= 0.3 * (speed_error / v_circular)
        reward += 0.01  # time bonus

        if radius_error < 2e10 and speed_error < 1000:
            reward += 10.0  # close to target

        escape_speed = np.sqrt(2 * self.G * self.M / self.target_radius)
        if curr_radius > 2.0 * self.target_radius:
            reward -= 50.0  # too far
        if v_actual > escape_speed:
            reward -= 80.0  # likely escaping

        if self.steps == self.max_steps and radius_error < 5e10:
            reward += 30.0  # bonus for stable ending

        # cosine alignment
        cos_theta = np.dot(self.pos, self.vel) / (np.linalg.norm(self.pos) * np.linalg.norm(self.vel) + 1e-8)
        reward += 2.0 * cos_theta

        # 👇 debug print (optional)
        if self.steps % 1000 == 0:
            print(f"[Step {self.steps}] Radius = {curr_radius:.2e}, Speed = {v_actual:.2e}, Reward = {reward:.3f}")

        done = self.steps >= self.max_steps
        return self._get_obs(), reward, done, {}