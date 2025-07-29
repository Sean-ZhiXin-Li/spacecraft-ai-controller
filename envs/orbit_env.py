import gym
from gym import spaces
import numpy as np
from typing import Optional, Tuple
from gym.core import ObsType
from ppo_orbit.rewards_utils import compute_reward  # Custom reward shaping

class OrbitEnv(gym.Env):
    """
    Custom 2D orbital mechanics environment for reinforcement learning.
    The agent controls a spacecraft by applying continuous 2D thrust to reach and maintain a target orbit.
    """

    def __init__(self,
                 G=6.67430e-11,                # Gravitational constant
                 M=1.989e30,                   # Central body mass (e.g. the Sun)
                 mass=722,                     # Spacecraft mass in kg
                 dt=600,                       # Time step in seconds
                 max_steps=8000,               # Max episode length
                 target_radius=7.5e12,         # Desired orbit radius (meters)
                 thrust_scale=8.0,             # Thrust scale factor
                 success_threshold=300,        # Steps near target to declare success
                 verbose=False):               # Whether to print logs
        super().__init__()

        # Simulation parameters
        self.G = G
        self.M = M
        self.mass = mass
        self.dt = dt
        self.max_steps = max_steps
        self.target_radius = target_radius
        self.thrust_scale = thrust_scale
        self.success_threshold = success_threshold
        self.verbose = verbose

        # Initialize counters and logs
        self.steps = 0
        self.success_counter = 0
        self.history = []

        # Action space: continuous 2D thrust [-1, 1] for x and y directions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: [x, y, vx, vy]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Initialize the environment
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        """
        Reset the environment to a new episode. Spacecraft is initialized below the target orbit with angled velocity.
        """
        super().reset(seed=seed)
        self.steps = 0
        self.success_counter = 0
        self.history.clear()

        # Start just inside the target orbit (position along x-axis)
        self.pos = np.array([self.target_radius, 0.0])

        # Compute initial circular orbit velocity magnitude
        v_mag = np.sqrt(self.G * self.M / self.target_radius)

        # Launch with angle 30 degrees from x-axis
        angle_rad = np.deg2rad(30)
        self.vel = v_mag * np.array([np.cos(angle_rad), np.sin(angle_rad)])

        return self._get_obs(), {}

    def _get_obs(self) -> ObsType:
        """
        Return current state observation: [x, y, vx, vy]
        """
        return np.concatenate([self.pos, self.vel]).astype(np.float32)

    def _out_of_bounds(self) -> bool:
        """
        Check if the spacecraft is too far from valid orbital space.
        """
        max_radius = 1.5 * self.target_radius
        return np.linalg.norm(self.pos) > max_radius

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, dict]:
        """
        Apply a thrust action and simulate one time step of physics.
        """
        self.steps += 1

        # Clip and scale thrust force
        thrust = self.thrust_scale * np.clip(action, -1.0, 1.0)
        acc_thrust = thrust / self.mass

        # Compute gravity acceleration
        r_vec = self.pos
        r = np.linalg.norm(r_vec)
        acc_gravity = -self.G * self.M * r_vec / ((r ** 3) + 1e-8)
        acc_gravity = np.clip(acc_gravity, -1e-2, 1e-2)  # Numerical safety clip

        # Euler integration
        self.vel += (acc_gravity + acc_thrust) * self.dt
        self.pos += self.vel * self.dt

        # Record history for visualization/debugging
        self.history.append((self.pos.copy(), self.vel.copy(), thrust.copy()))

        # Compute shaped reward and diagnostics
        reward, shaping, bonus, penalty, r_err, v_err = compute_reward(
            pos=self.pos,
            vel=self.vel,
            target_radius=self.target_radius,
            thrust=thrust,
            success_counter=self.success_counter,
            G=self.G,
            M=self.M,
            mass=self.mass
        )

        # Update success counter
        if r_err < 0.05 * self.target_radius:
            self.success_counter += 1
        else:
            self.success_counter = 0

        # Check terminal condition
        done = (
            self.steps >= self.max_steps or
            self._out_of_bounds() or
            self.success_counter >= self.success_threshold
        )

        # Optional logging
        if self.verbose and self.steps % 500 == 0:
            print(f"[Step {self.steps}] r_err = {r_err:.2e}, v_err = {v_err:.2f}, reward = {reward:.2f}, bonus = {bonus:.2f}")

        # Pack diagnostic info
        info = {
            "reward": reward,
            "shaping": shaping,
            "bonus": bonus,
            "penalty": penalty,
            "r_error": r_err,
            "v_error": v_err,
            "steps": self.steps
        }

        return self._get_obs(), reward, done, info

    def render(self, mode="human"):
        """
        Print current state for debugging.
        """
        print(f"Step {self.steps}: pos={self.pos}, vel={self.vel}")



