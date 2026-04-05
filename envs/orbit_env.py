import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import os

# Keep aligned with local rewards_utils.py
from ppo_orbit.rewards_utils import compute_reward


class OrbitEnv(gym.Env):
    """
    2D Newtonian orbital environment with thrust control.
    The agent outputs a 2D action in [-1, 1], which is scaled to a thrust vector.
    Dynamics: gravity (point mass) + thrust acceleration, integrated with Euler.

    This env supports:
      - early success detection within a tolerance window
      - terminal bonus/penalty
      - runtime injection of task parameters (mass, thrust, target radius, etc.)
      - runtime initialization of the orbital state (position + velocity)

    NOTE:
    - Thrust is computed as: thrust_vec = thrust_scale * action
    - Acceleration from thrust is: a_thrust = thrust_vec / mass
    - If you want to simulate a mega-mass interstellar spacecraft (e.g., 1e9–1e10 kg),
      either increase thrust_scale, or use a larger dt, or accept slower maneuvers.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
            self,
            G: float = 6.67430e-11,
            M: float = 1.989e30,
            mass: float = 722.0,
            dt: float = 2.0,
            max_steps: int = 60000,
            target_radius: float = 7.5e12,
            thrust_scale: float = 3000.0,
            success_threshold: int = 120,
            tol_r: float = 2e-3,
            tol_v: float = 2e-3,
            tol_ang: float = 0.08,
            term_reward_success: float = 1000.0,
            term_reward_fail: float = -50.0,
            verbose: bool = False,
            reward_mode: str = "base",
            w_radius: float = 0.0,
            w_progress: float = 0.0,
            w_speed: float = 0.0,
    ) -> None:
        super().__init__()

        # Physical constants and runtime parameters
        self.G = G
        self.M = M
        self.mu = self.G * self.M
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
        self.reward_mode = reward_mode
        self.w_radius = w_radius
        self.w_progress = w_progress

        # Day 8 reward controls
        self.reward_mode = reward_mode
        self.w_radius = w_radius
        self.w_progress = w_progress
        self.w_speed = w_speed

        # Optional acceleration cap derived from thrust/mass
        self.a_cap: Optional[float] = None
        try:
            if self.mass > 0.0:
                self.a_cap = self.thrust_scale / self.mass
        except Exception:
            self.a_cap = None

        # Action/observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Internal state buffers
        self.steps = 0
        self.success_counter = 0
        self.pos = np.zeros(2, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)

        # Initialize
        self.reset()

    def set_physical_params(
        self,
        mass: Optional[float] = None,
        thrust_newton: Optional[float] = None,
        max_steps: Optional[int] = None,
        r_target: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Inject physical/task parameters at runtime.
        """
        if mass is not None:
            self.mass = float(mass)

        if thrust_newton is not None:
            self.thrust_scale = float(thrust_newton)

        if max_steps is not None:
            self.max_steps = int(max_steps)

        if r_target is not None:
            self.target_radius = float(r_target)

        self.mu = self.G * self.M

        try:
            if getattr(self, "mass", None) and getattr(self, "thrust_scale", None):
                if self.mass > 0.0:
                    self.a_cap = self.thrust_scale / self.mass
        except Exception:
            self.a_cap = None

        if seed is not None:
            import random

            try:
                random.seed(int(seed))
            except Exception:
                random.seed()
            try:
                np.random.seed(int(seed))
            except Exception:
                pass

    def set_initial_state(self, init_state: dict) -> None:
        """
        Initialize the environment's state from a task spec.
        """
        pos = init_state.get("pos", [self.target_radius, 0.0])
        vx_vy = init_state.get("vel", None)
        vel_angle_deg = float(init_state.get("vel_angle_deg", 0.0))
        vel_scale = float(init_state.get("vel_scale", 1.0))

        self.pos = np.array([float(pos[0]), float(pos[1])], dtype=np.float64)

        if vx_vy is not None:
            self.vel = np.array([float(vx_vy[0]), float(vx_vy[1])], dtype=np.float64)
        else:
            r_ref = float(self.target_radius if self.target_radius > 0.0 else np.linalg.norm(self.pos))
            r_ref = max(1e-12, r_ref)
            v_circ = np.sqrt(self.mu / r_ref)

            ang = np.deg2rad(vel_angle_deg)
            vx = vel_scale * v_circ * np.cos(ang)
            vy = vel_scale * v_circ * np.sin(ang)
            self.vel = np.array([vx, vy], dtype=np.float64)

        self.steps = 0
        self.success_counter = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        start_mode: str = "default",
        **cfg: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment state.
        """
        if "mu" in cfg and cfg["mu"] is not None:
            try:
                self.mu = float(cfg["mu"])
                self.M = float(self.mu / self.G) if self.G != 0.0 else self.M
            except Exception:
                pass

        if "sc_mass" in cfg and cfg["sc_mass"] is not None:
            try:
                self.mass = float(cfg["sc_mass"])
            except Exception:
                pass

        if "r0" in cfg and cfg["r0"] is not None:
            try:
                self.target_radius = float(cfg["r0"])
            except Exception:
                pass

        self.mu = self.G * self.M if not hasattr(self, "mu") else (self.mu if self.mu == self.G * self.M else self.mu)

        super().reset(seed=seed)
        self.steps = 0
        self.success_counter = 0

        try:
            r0_mul = float(os.environ.get("R0_OVER_TARGET", "1.25"))
        except Exception:
            r0_mul = 1.25

        if start_mode == "default":
            self.pos = np.array([0.0, r0_mul * self.target_radius], dtype=np.float64)
            v_mag = np.sqrt(self.mu / np.linalg.norm(self.pos))
            angle = np.deg2rad(30.0)
            self.vel = v_mag * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)

        elif start_mode == "spiral":
            self.pos = np.array([0.0, 0.6 * self.target_radius], dtype=np.float64)
            v_mag = 0.8 * np.sqrt(self.mu / np.linalg.norm(self.pos))
            angle = np.deg2rad(60.0)
            self.vel = v_mag * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)

        else:
            raise ValueError(f"Unknown start_mode: {start_mode}")

        info: Dict[str, Any] = {
            "start_mode": start_mode,
            "seed": seed,
        }
        return self._get_obs(), info

    def apply_delta_v(self, dv: float) -> None:
        """
        Apply an instantaneous tangential impulse of magnitude `dv` [m/s] at the current state.
        """
        r_vec = self.pos.astype(float)
        v_vec = self.vel.astype(float)
        r = np.linalg.norm(r_vec)
        if r < 1e-12:
            return

        ur = r_vec / r
        t = np.array([-ur[1], ur[0]], dtype=float)

        if np.dot(v_vec, t) < 0.0:
            t = -t

        self.vel = self.vel + float(dv) * t

    def action_space_sample(self) -> np.ndarray:
        """
        Convenience helper so wrappers can derive a zero-like action safely.
        """
        try:
            a = self.action_space.sample()
            return np.zeros_like(a)
        except Exception:
            return np.zeros(2, dtype=np.float32)

    def get_state(self) -> dict:
        """
        Return a snapshot of env internal state.
        """
        return {
            "steps": int(self.steps),
            "success_counter": int(self.success_counter),
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "G": float(self.G),
            "M": float(self.M),
            "mu": float(self.mu),
            "mass": float(self.mass),
            "dt": float(self.dt),
            "max_steps": int(self.max_steps),
            "target_radius": float(self.target_radius),
            "thrust_scale": float(self.thrust_scale),
            "success_threshold": int(self.success_threshold),
            "tol_r": float(self.tol_r),
            "tol_v": float(self.tol_v),
            "tol_ang": float(self.tol_ang),
            "term_reward_success": float(self.term_reward_success),
            "term_reward_fail": float(self.term_reward_fail),
            "verbose": bool(self.verbose),
            "a_cap": float(self.a_cap) if self.a_cap is not None else None,
            "reward_mode": str(self.reward_mode),
            "w_radius": float(self.w_radius),
            "w_progress": float(self.w_progress),
            "w_speed": float(self.w_speed),
        }

    def set_state(self, state: dict) -> None:
        """
        Restore a snapshot produced by get_state().
        """
        if not isinstance(state, dict):
            return

        self.steps = int(state.get("steps", self.steps))
        self.success_counter = int(state.get("success_counter", self.success_counter))

        pos = state.get("pos", None)
        vel = state.get("vel", None)
        if pos is not None:
            self.pos = np.array(pos, dtype=np.float64, copy=True)
        if vel is not None:
            self.vel = np.array(vel, dtype=np.float64, copy=True)

        self.G = float(state.get("G", self.G))
        self.M = float(state.get("M", self.M))
        self.mu = float(state.get("mu", self.mu))
        self.mass = float(state.get("mass", self.mass))
        self.dt = float(state.get("dt", self.dt))
        self.max_steps = int(state.get("max_steps", self.max_steps))
        self.target_radius = float(state.get("target_radius", self.target_radius))
        self.thrust_scale = float(state.get("thrust_scale", self.thrust_scale))

        self.success_threshold = int(state.get("success_threshold", self.success_threshold))
        self.tol_r = float(state.get("tol_r", self.tol_r))
        self.tol_v = float(state.get("tol_v", self.tol_v))
        self.tol_ang = float(state.get("tol_ang", self.tol_ang))

        self.term_reward_success = float(state.get("term_reward_success", self.term_reward_success))
        self.term_reward_fail = float(state.get("term_reward_fail", self.term_reward_fail))

        self.verbose = bool(state.get("verbose", self.verbose))

        self.reward_mode = str(state.get("reward_mode", self.reward_mode))
        self.w_radius = float(state.get("w_radius", self.w_radius))
        self.w_progress = float(state.get("w_progress", self.w_progress))
        self.w_speed = float(state.get("w_speed", self.w_speed))

        a_cap_saved = state.get("a_cap", None)
        if a_cap_saved is not None:
            try:
                self.a_cap = float(a_cap_saved)
            except Exception:
                self.a_cap = None
        else:
            try:
                self.a_cap = self.thrust_scale / self.mass if self.mass > 0.0 else None
            except Exception:
                self.a_cap = None

    def _get_obs(self) -> np.ndarray:
        """Return observation as [x, y, vx, vy]."""
        return np.concatenate([self.pos, self.vel]).astype(np.float32)

    def _inside_tolerance(self, pos: np.ndarray, vel: np.ndarray) -> bool:
        """
        Check whether the current state is within the success tolerance window.
        """
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        v_target = np.sqrt(self.mu / self.target_radius)


        r_err = abs(r - self.target_radius) / self.target_radius
        v_err = abs(v - v_target) / v_target

        ur = pos / (r + 1e-8)
        uv = vel / (v + 1e-8)
        ang = abs(np.dot(ur, uv))

        return (r_err < self.tol_r) and (v_err < self.tol_v) and (ang < self.tol_ang)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advance one simulation step with the given action.
        """
        self.steps += 1

        # Save previous position for progress reward
        prev_pos = self.pos.copy()

        # Thrust from action
        action = np.clip(action, -1.0, 1.0)
        thrust = self.thrust_scale * action
        acc_thrust = thrust / max(1e-12, self.mass)

        # Gravity from point mass at the origin
        r_vec = self.pos
        r = np.linalg.norm(r_vec)
        acc_gravity = -self.mu * r_vec / ((r ** 3) + 1e-12)

        # Numerical safety clamp
        acc_gravity = np.clip(acc_gravity, -1e-2, 1e-2)

        r_norm = np.linalg.norm(r_vec) + 1e-12
        r_hat = r_vec / r_norm
        t_hat = np.array([-r_hat[1], r_hat[0]], dtype=np.float64)

        if np.linalg.norm(thrust) < 0.05 * self.thrust_scale:
            thrust = thrust + 0.05 * self.thrust_scale * t_hat
            acc_thrust = thrust / max(1e-12, self.mass)

        # Integrate with simple Euler
        self.vel = self.vel + (acc_gravity + acc_thrust) * self.dt
        self.pos = self.pos + self.vel * self.dt

        # Success window tracking
        if self._inside_tolerance(self.pos, self.vel):
            self.success_counter += 1
        else:
            self.success_counter = 0

        # Termination logic
        r_now = np.linalg.norm(self.pos)
        v_now = np.linalg.norm(self.vel)
        v_target = np.sqrt(self.mu / self.target_radius)

        time_up = self.steps >= self.max_steps
        out_range = r_now > 2.5 * self.target_radius
        success = self.success_counter >= self.success_threshold
        overspeed = v_now > 1.45 * v_target
        too_close = r_now < 0.55 * self.target_radius

        terminated = bool(success or out_range or overspeed or too_close)
        truncated = bool(time_up)
        done = bool(terminated or truncated)

        # compute radial velocity
        r_vec = self.pos
        v_vec = self.vel

        v_r = float(np.dot(v_vec, r_hat))

        # Reward shaping
        reward_dict = compute_reward(
            pos=self.pos,
            vel=self.vel,
            thrust=thrust,
            target_radius=self.target_radius,
            fuel_used=np.linalg.norm(thrust),
            G=self.G,
            M=self.M,
            step_count=self.steps,
            done=done,
            prev_pos=prev_pos,
            reward_mode=self.reward_mode,
            w_radius=self.w_radius,
            w_progress=self.w_progress,
            w_speed=self.w_speed,
            v_r=v_r,
            thrust_scale=self.thrust_scale,
        )

        reward = reward_dict["reward"]
        shaping = reward_dict["shaping"]
        bonus = reward_dict["bonus"]
        penalty = reward_dict["penalty"]
        r_err = reward_dict["r_error"]
        v_err = reward_dict["v_error"]

        # Terminal bonus/penalty
        term_bonus = 0.0
        if done:
            if success:
                term_bonus += self.term_reward_success
            elif out_range or overspeed or too_close:
                term_bonus += self.term_reward_fail

        reward += term_bonus

        info: Dict[str, Any] = {
            "reward": float(reward),
            "shaping": float(shaping),
            "bonus": float(bonus),
            "penalty": float(penalty),
            "radius_error": float(r_err),
            "speed_error": float(v_err),
            "progress": float(reward_dict["progress"]),
            "reward_radius_term": float(reward_dict["radius_term"]),
            "reward_progress_term": float(reward_dict["progress_term"]),
            "angle_cos": float(reward_dict["angle_cos"]),
            "r_term": float(reward_dict["r_term"]),
            "v_term": float(reward_dict["v_term"]),
            "angle_term": float(reward_dict["angle_term"]),
            "steps": int(self.steps),
            "success_counter": int(self.success_counter),
            "terminal_bonus": float(term_bonus),
            "success": bool(success),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "a_cap_ref": float(self.a_cap) if self.a_cap is not None else None,
            "mass": float(self.mass),
            "thrust_scale": float(self.thrust_scale),
            "dt": float(self.dt),
            "reward_mode": str(self.reward_mode),
            "w_radius": float(self.w_radius),
            "w_progress": float(self.w_progress),
            "action_clipped": action.astype(np.float32),
            "thrust_vec": thrust.astype(np.float64),
            "acc_thrust": acc_thrust.astype(np.float64),
            "acc_gravity": acc_gravity.astype(np.float64),
            "acc_total": (acc_gravity + acc_thrust).astype(np.float64),
            "w_speed": float(self.w_speed),
            "reward_speed_term": float(reward_dict["speed_term"]),
            "stage": reward_dict.get("stage", "unknown"),
            "tangential_alignment": float(reward_dict.get("tangential_alignment", 0.0)),
            "radial_alignment": float(reward_dict.get("radial_alignment", 0.0)),
            "stop_bonus": float(reward_dict.get("stop_bonus", 0.0)),
            "v_r_raw": float(reward_dict.get("v_r", v_r)),
            "overspeed_penalty": float(reward_dict.get("overspeed_penalty", 0.0)),
            "thrust_norm": float(reward_dict.get("thrust_norm", 0.0)),
            "h_norm": float(reward_dict.get("h_norm", 0.0)),
            "angular_momentum": float(reward_dict.get("angular_momentum", 0.0)),
            "overspeed": bool(overspeed),
            "v_now": float(v_now),
            "v_target": float(v_target),
            "too_close": bool(too_close),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def set_reward_config(
            self,
            reward_mode: Optional[str] = None,
            w_radius: Optional[float] = None,
            w_progress: Optional[float] = None,
            w_speed: Optional[float] = None,
    ) -> None:
        """
        Update reward shaping configuration at runtime.
        """
        if reward_mode is not None:
            self.reward_mode = str(reward_mode)
        if w_radius is not None:
            self.w_radius = float(w_radius)
        if w_progress is not None:
            self.w_progress = float(w_progress)
        if w_speed is not None:
            self.w_speed = float(w_speed)

    def render(self):
        """Minimal text renderer for quick debugging."""
        print(f"Step {self.steps} | pos: {self.pos}, vel: {self.vel}")

