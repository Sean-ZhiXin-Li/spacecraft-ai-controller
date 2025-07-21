
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
                 max_steps=60000,         # Maximum simulation steps per episode
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

        # Start at the target radius directly "above" the Sun (positive y-axis)
        self.pos = np.array([0.0, self.target_radius], dtype=np.float64)

        # Initial tangential velocity (to the right)
        speed = 20000.0  # in m/s
        self.vel = speed * np.array([1.0, 0.0], dtype=np.float64)

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

        # Convert normalized action to physical thrust
        thrust = self.thrust_scale * np.clip(action, -1.0, 1.0)
        acc_thrust = thrust / self.mass  # F = ma → a = F / m

        # Calculate gravitational acceleration
        r = np.linalg.norm(self.pos)
        if r == 0:
            acc_gravity = np.zeros(2)
        else:
            acc_gravity = -self.G * self.M * self.pos / r**3  # Central gravitational force

        # Update velocity and position using Euler integration
        total_acc = acc_gravity + acc_thrust
        self.vel += total_acc * self.dt
        self.pos += self.vel * self.dt

        # Calculate reward: negative normalized distance to target radius
        radius_error = np.abs(np.linalg.norm(self.pos) - self.target_radius)
        reward = -radius_error / self.target_radius

        # Check termination conditions
        done = self.steps >= self.max_steps or r > 10 * self.target_radius

        # Return observation, reward, done flag, and empty info
        return self._get_obs(), reward, done, {}
