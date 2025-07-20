import gym
from gym import spaces
import numpy as np
from typing import Optional, Tuple
from gym.core import ObsType

class OrbitEnv(gym.Env):
    """
    A custom Gym environment for spacecraft orbital control.
    The agent applies 2D thrust to control the orbit.
    """

    def __init__(self,
                 G=6.67430e-11,
                 M=1.989e30,
                 mass=722,
                 dt=3600,
                 max_steps=60000,
                 target_radius=7.5e12,
                 thrust_scale=0.2):
        super().__init__()

        # Physical constants
        self.G = G  # Gravitational constant
        self.M = M  # Central body mass (e.g., the Sun)
        self.mass = mass  # Spacecraft mass
        self.dt = dt  # Time step size (seconds)
        self.max_steps = max_steps
        self.target_radius = target_radius  # Desired orbital radius (meters)
        self.thrust_scale = thrust_scale  # Max physical thrust for action range [-1, 1]

        # Observation: [x, y, vx, vy]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Action: 2D thrust vector (tx, ty) in range [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Initialize the simulation state
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)
        self.steps = 0

        # Start at the target radius directly "above" the Sun (positive y-axis)
        self.pos = np.array([0.0, self.target_radius], dtype=np.float64)

        # Initial velocity (tangential, pointing right)
        speed = 20000.0  # m/s
        self.vel = speed * np.array([1.0, 0.0], dtype=np.float64)

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> ObsType:
        """
        Return current observation as a flat array: [x, y, vx, vy]
        """
        return np.concatenate([self.pos, self.vel]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, dict]:
        """
        Advance the environment one time step given an action.
        """
        self.steps += 1

        # Convert normalized thrust [-1, 1] into physical thrust
        thrust = self.thrust_scale * np.clip(action, -1.0, 1.0)
        acc_thrust = thrust / self.mass  # a = F / m

        # Compute gravitational acceleration
        r = np.linalg.norm(self.pos)
        if r == 0:
            acc_gravity = np.zeros(2)
        else:
            acc_gravity = -self.G * self.M * self.pos / r**3  # Central force field

        # Total acceleration and state update
        total_acc = acc_gravity + acc_thrust
        self.vel += total_acc * self.dt
        self.pos += self.vel * self.dt

        # Reward: negative normalized distance from target orbit
        radius_error = np.abs(np.linalg.norm(self.pos) - self.target_radius)
        reward = -radius_error / self.target_radius

        # Done: either max steps reached or trajectory escaped too far
        done = self.steps >= self.max_steps or r > 10 * self.target_radius

        # Standard Gym return
        return self._get_obs(), reward, done, {}

