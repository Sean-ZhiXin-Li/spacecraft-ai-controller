import gym
from gym import spaces
import numpy as np
from typing import Optional, Tuple
from gym.core import ObsType
from ppo_orbit.rewards_utils import compute_reward  # 确保此模块路径正确

class OrbitEnv(gym.Env):
    """
    2D orbital environment simulating Newtonian gravitational physics with thrust control.
    """

    def __init__(self,
                 G=6.67430e-11,
                 M=1.989e30,
                 mass=722,
                 dt=10.0,
                 max_steps=60000,
                 target_radius=7.5e12,
                 thrust_scale=3000,
                 success_threshold=300,
                 verbose=False):
        super().__init__()

        self.G = G
        self.M = M
        self.mass = mass
        self.dt = dt
        self.max_steps = max_steps
        self.target_radius = target_radius
        self.thrust_scale = thrust_scale
        self.success_threshold = success_threshold
        self.verbose = verbose

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)

        self.steps = 0
        self.success_counter = 0

        # Initial position: above circular target radius
        self.pos = np.array([0.0, 1.25 * self.target_radius], dtype=np.float64)

        # Initial velocity: slightly off from ideal tangential (30° angle)
        v_mag = np.sqrt(self.G * self.M / np.linalg.norm(self.pos))
        angle = np.deg2rad(30)
        self.vel = v_mag * np.array([np.cos(angle), np.sin(angle)])

        return self._get_obs(), {}

    def _get_obs(self) -> ObsType:
        return np.concatenate([self.pos, self.vel]).astype(np.float32)

    def _out_of_bounds(self) -> bool:
        return np.linalg.norm(self.pos) > 2.5 * self.target_radius

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, dict]:
        self.steps += 1

        # Thrust control
        action = np.clip(action, -1.0, 1.0)
        thrust = self.thrust_scale * action
        acc_thrust = thrust / self.mass

        # Gravitational acceleration
        r_vec = self.pos
        r = np.linalg.norm(r_vec)
        acc_gravity = -self.G * self.M * r_vec / ((r ** 3) + 1e-8)
        acc_gravity = np.clip(acc_gravity, -1e-2, 1e-2)  # Avoid numerical instability

        # Physics update
        self.vel += (acc_gravity + acc_thrust) * self.dt
        self.pos += self.vel * self.dt

        # Orbital error metrics
        r_err = abs(r - self.target_radius)
        v_err = abs(np.linalg.norm(self.vel) - np.sqrt(self.G * self.M / self.target_radius))

        if r_err < 0.05 * self.target_radius:
            self.success_counter += 1
        else:
            self.success_counter = 0

        # Terminal condition
        done = (self.steps >= self.max_steps) or self._out_of_bounds()

        # Compute reward using modular shaping
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

        if self.verbose and self.steps % 500 == 0:
            print(f"[{self.steps}] r={r_err:.2e}, v={v_err:.2f}, reward={reward:.2f}, bonus={bonus:.2f}")

        return self._get_obs(), reward, done, info

    def render(self, mode="human"):
        print(f"Step {self.steps} | pos: {self.pos}, vel: {self.vel}")

