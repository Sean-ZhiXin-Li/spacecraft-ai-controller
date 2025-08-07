import gym
from gym import spaces
import numpy as np
from typing import Optional, Tuple
from gym.core import ObsType
from ppo_orbit.rewards_utils import compute_reward  # Ensure this path is correct

class OrbitEnv(gym.Env):
    """
    A 2D orbital environment simulating Newtonian gravitational physics
    with thrust-based control for a spacecraft.
    """

    def __init__(self,
                 G=6.67430e-11,                # Gravitational constant
                 M=1.989e30,                   # Central mass (e.g., the Sun)
                 mass=722,                     # Mass of the spacecraft
                 dt=10.0,                      # Time step (seconds)
                 max_steps=60000,              # Max number of simulation steps
                 target_radius=7.5e12,         # Desired circular orbit radius (meters)
                 thrust_scale=3000,            # Multiplier for action → thrust vector
                 success_threshold=300,        # Number of steps within target to count as success
                 verbose=False):               # Whether to print debug info
        super().__init__()

        # Environment parameters
        self.G = G
        self.M = M
        self.mass = mass
        self.dt = dt
        self.max_steps = max_steps
        self.target_radius = target_radius
        self.thrust_scale = thrust_scale
        self.success_threshold = success_threshold
        self.verbose = verbose

        # Action space: 2D continuous thrust vector, normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: position (x, y) and velocity (vx, vy)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, start_mode: str = "default") -> Tuple[ObsType, dict]:
        """
        Resets the environment to an initial state based on the selected mode.

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Unused in this implementation.
            start_mode (str): Initial condition setup. Supported: "default", "spiral".

        Returns:
            observation (np.ndarray): Initial observation [x, y, vx, vy]
            info (dict): Additional info (empty)
        """
        super().reset(seed=seed)

        self.steps = 0
        self.success_counter = 0

        if start_mode == "default":
            # Start just outside the target orbit, with a 30-degree velocity offset
            self.pos = np.array([0.0, 1.25 * self.target_radius], dtype=np.float64)
            v_mag = np.sqrt(self.G * self.M / np.linalg.norm(self.pos))
            angle = np.deg2rad(30)
            self.vel = v_mag * np.array([np.cos(angle), np.sin(angle)])

        elif start_mode == "spiral":
            # Start well inside the orbit, with reduced tangential velocity
            self.pos = np.array([0.0, 0.6 * self.target_radius], dtype=np.float64)
            v_mag = 0.8 * np.sqrt(self.G * self.M / np.linalg.norm(self.pos))
            angle = np.deg2rad(60)
            self.vel = v_mag * np.array([np.cos(angle), np.sin(angle)])

        else:
            raise ValueError(f"Unknown start_mode: {start_mode}")

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """
        Returns the current observation: position and velocity combined.

        Returns:
            np.ndarray: [x, y, vx, vy] as float32
        """
        return np.concatenate([self.pos, self.vel]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, dict]:
        """
        Applies the given thrust action and updates the spacecraft's state.

        Args:
            action (np.ndarray): Normalized thrust vector in [-1, 1]^2

        Returns:
            observation (np.ndarray): Next state [x, y, vx, vy]
            reward (float): Shaped reward value
            done (bool): Whether the episode is over
            info (dict): Diagnostic information
        """
        self.steps += 1

        # Convert normalized action to real thrust force
        action = np.clip(action, -1.0, 1.0)
        thrust = self.thrust_scale * action
        acc_thrust = thrust / self.mass

        # Compute gravitational acceleration
        r_vec = self.pos
        r = np.linalg.norm(r_vec)
        acc_gravity = -self.G * self.M * r_vec / ((r ** 3) + 1e-8)
        acc_gravity = np.clip(acc_gravity, -1e-2, 1e-2)  # Clamp for stability

        # Update velocity and position using Euler integration
        self.vel += (acc_gravity + acc_thrust) * self.dt
        self.pos += self.vel * self.dt

        # Check episode termination
        done = self.steps >= self.max_steps or r > 2.5 * self.target_radius

        # Compute shaped reward
        reward, shaping, bonus, penalty, r_err, v_err = compute_reward(
            pos=self.pos,
            vel=self.vel,
            thrust=thrust,
            target_radius=self.target_radius,
            fuel_used=np.linalg.norm(thrust),
            G=self.G,
            M=self.M,
            step_count=self.steps,
            done=done
        )

        info = {
            "reward": reward,
            "shaping": shaping,
            "bonus": bonus,
            "penalty": penalty,
            "radius_error": r_err,
            "speed_error": v_err,
            "steps": self.steps,
            "success_counter": self.success_counter
        }

        return self._get_obs(), reward, done, info

    def render(self, mode="human"):
        """
        Prints the current position and velocity of the spacecraft.
        """
        print(f"Step {self.steps} | pos: {self.pos}, vel: {self.vel}")
