import gym
from gym import spaces
import numpy as np
from typing import Optional, Tuple
from gym.core import ObsType

# 与同目录 rewards_utils.py 对齐
from rewards_utils import compute_reward

class OrbitEnv(gym.Env):
    """
    2D Newtonian orbital environment with thrust control.
    Includes success early-stop and terminal reward.
    """

    def __init__(self,
                 G=6.67430e-11,
                 M=1.989e30,
                 mass=722,
                 dt=10.0,
                 max_steps=60000,
                 target_radius=7.5e12,
                 thrust_scale=3000,              # keep consistent with PPO THRUST_SCALE
                 success_threshold=120,          # consecutive steps inside tolerance
                 tol_r=2e-3, tol_v=2e-3, tol_ang=0.08,  # relaxed early phase
                 term_reward_success=1000.0,     # terminal success bonus
                 term_reward_fail=-50.0,         # mild failure penalty
                 verbose=False):
        super().__init__()

        self.G = G; self.M = M
        self.mass = mass
        self.dt = dt
        self.max_steps = max_steps
        self.target_radius = target_radius
        self.thrust_scale = thrust_scale
        self.success_threshold = success_threshold
        self.tol_r = tol_r
        self.tol_v = tol_v
        self.tol_ang = tol_ang
        self.term_reward_success = term_reward_success
        self.term_reward_fail = term_reward_fail
        self.verbose = verbose

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, start_mode: str = "default") -> Tuple[ObsType, dict]:
        super().reset(seed=seed)
        self.steps = 0
        self.success_counter = 0

        if start_mode == "default":
            # start slightly outside target orbit with 30-degree velocity offset
            self.pos = np.array([0.0, 1.25 * self.target_radius], dtype=np.float64)
            v_mag = np.sqrt(self.G * self.M / np.linalg.norm(self.pos))
            angle = np.deg2rad(30)
            self.vel = v_mag * np.array([np.cos(angle), np.sin(angle)])
        elif start_mode == "spiral":
            self.pos = np.array([0.0, 0.6 * self.target_radius], dtype=np.float64)
            v_mag = 0.8 * np.sqrt(self.G * self.M / np.linalg.norm(self.pos))
            angle = np.deg2rad(60)
            self.vel = v_mag * np.array([np.cos(angle), np.sin(angle)])
        else:
            raise ValueError(f"Unknown start_mode: {start_mode}")

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.pos, self.vel]).astype(np.float32)

    def _inside_tolerance(self, pos, vel) -> bool:
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        v_target = np.sqrt(self.G * self.M / self.target_radius)

        r_err = abs(r - self.target_radius) / self.target_radius
        v_err = abs(v - v_target) / v_target

        ur = pos / (r + 1e-8)
        uv = vel / (v + 1e-8)
        ang = abs(np.dot(ur, uv))  # want near 0

        return (r_err < self.tol_r) and (v_err < self.tol_v) and (ang < self.tol_ang)

    def step(self, action: np.ndarray):
        self.steps += 1

        # thrust
        action = np.clip(action, -1.0, 1.0)
        thrust = self.thrust_scale * action
        acc_thrust = thrust / self.mass

        # gravity
        r_vec = self.pos
        r = np.linalg.norm(r_vec)
        acc_gravity = -self.G * self.M * r_vec / ((r ** 3) + 1e-8)
        acc_gravity = np.clip(acc_gravity, -1e-2, 1e-2)  # numerical safety

        # integrate (Euler)
        self.vel += (acc_gravity + acc_thrust) * self.dt
        self.pos += self.vel * self.dt

        # success window counter
        if self._inside_tolerance(self.pos, self.vel):
            self.success_counter += 1
        else:
            self.success_counter = 0

        # termination
        r_now = np.linalg.norm(self.pos)
        time_up   = self.steps >= self.max_steps
        out_range = r_now > 2.5 * self.target_radius
        success   = self.success_counter >= self.success_threshold
        done = time_up or out_range or success

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

        # terminal bonuses/penalties
        term_bonus = 0.0
        if done:
            if success:
                term_bonus += self.term_reward_success
            elif out_range:
                term_bonus += self.term_reward_fail
        reward += term_bonus

        info = {
            "reward": reward,
            "shaping": shaping,
            "bonus": bonus,
            "penalty": penalty,
            "radius_error": r_err,
            "speed_error": v_err,
            "steps": self.steps,
            "success_counter": self.success_counter,
            "terminal_bonus": term_bonus,
            "success": bool(success)
        }

        return self._get_obs(), float(reward), bool(done), info

    def render(self, mode="human"):
        print(f"Step {self.steps} | pos: {self.pos}, vel: {self.vel}")
