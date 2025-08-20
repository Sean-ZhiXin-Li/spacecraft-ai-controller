import gym
from gym import spaces
import numpy as np
from typing import Optional, Tuple
from gym.core import ObsType

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

    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 G: float = 6.67430e-11,
                 M: float = 1.989e30,
                 mass: float = 722.0,
                 dt: float = 10.0,
                 max_steps: int = 60000,
                 target_radius: float = 7.5e12,
                 thrust_scale: float = 3000.0,       # keep consistent with PPO THRUST_SCALE
                 success_threshold: int = 120,       # consecutive steps inside tolerance
                 tol_r: float = 2e-3,
                 tol_v: float = 2e-3,
                 tol_ang: float = 0.08,              # angle tolerance (rad), want near 0
                 term_reward_success: float = 1000.0,
                 term_reward_fail: float = -50.0,
                 verbose: bool = False) -> None:
        super().__init__()

        # Physical constants and runtime parameters
        self.G = G
        self.M = M
        self.mu = self.G * self.M  # cached GM for convenience
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

        # Optional acceleration cap derived from thrust/mass; updated by setters too.
        self.a_cap: Optional[float] = None
        try:
            if self.mass > 0.0:
                # This is a soft reference value; we do not clip with it by default.
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

    # --------------------------------------------------------------------------
    # RUNTIME INJECTION SETTERS
    # --------------------------------------------------------------------------
    def set_physical_params(self,
                            mass: Optional[float] = None,
                            thrust_newton: Optional[float] = None,
                            max_steps: Optional[int] = None,
                            r_target: Optional[float] = None,
                            seed: Optional[int] = None) -> None:
        """
        Inject physical/task parameters at runtime.

        Args:
            mass: spacecraft mass [kg]. For mega-mass craft (e.g., 5e9 kg), pass it here.
            thrust_newton: if provided, we interpret it as the magnitude scale of thrust
                           and set `thrust_scale = thrust_newton`. The policy still outputs
                           a vector in [-1,1]^2 which is scaled by `thrust_scale`.
            max_steps: maximum simulation steps before time-up termination.
            r_target: target orbital radius used by success/tolerance and reward shaping.
            seed: random seed to improve determinism (Python + NumPy only).

        Notes:
            - We also refresh `a_cap = thrust_scale / mass` when both are known.
            - We do NOT reset() here. Call reset() afterwards to apply new params.
        """
        if mass is not None:
            self.mass = float(mass)

        if thrust_newton is not None:
            # In this env, `thrust_scale` is the vector scale (N) for action ∈ [-1,1]^2.
            self.thrust_scale = float(thrust_newton)

        if max_steps is not None:
            self.max_steps = int(max_steps)

        if r_target is not None:
            self.target_radius = float(r_target)

        # Keep mu cached; if G/M are ever changed elsewhere, user should update self.mu accordingly.
        self.mu = self.G * self.M

        # Optional derived cap for diagnostics/safety (not enforced unless you choose to).
        try:
            if getattr(self, "mass", None) and getattr(self, "thrust_scale", None):
                if self.mass > 0.0:
                    self.a_cap = self.thrust_scale / self.mass
        except Exception:
            self.a_cap = None

        # Seed basic RNGs (framework RNGs like torch need to be seeded externally if used there)
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

        Expected fields in `init_state`:
            - "pos": [x0, y0] in meters
            - "vel_angle_deg": velocity direction in degrees (0° along +x)
            - "vel_scale": scalar multiplied by circular speed at target radius

        Effect:
            Writes to self.pos, self.vel, and makes the observation reflect [x,y,vx,vy].

        If your outer code wants a different initialization (e.g., elliptical velocity at rp),
        you can extend this function to compute that analytically.
        """
        # Fallbacks with robust parsing
        pos = init_state.get("pos", [self.target_radius, 0.0])
        vx_vy = init_state.get("vel", None)  # optional direct velocity override
        vel_angle_deg = float(init_state.get("vel_angle_deg", 0.0))
        vel_scale = float(init_state.get("vel_scale", 1.0))

        # Position
        self.pos = np.array([float(pos[0]), float(pos[1])], dtype=np.float64)

        if vx_vy is not None:
            # If explicit velocity is provided, trust it.
            self.vel = np.array([float(vx_vy[0]), float(vx_vy[1])], dtype=np.float64)
        else:
            # Build velocity from circular-speed reference at target radius.
            # For large target radius, circular speed is sqrt(mu / r_target).
            r_ref = float(self.target_radius if self.target_radius > 0.0 else np.linalg.norm(self.pos))
            r_ref = max(1e-12, r_ref)
            v_circ = np.sqrt(self.mu / r_ref)

            ang = np.deg2rad(vel_angle_deg)
            vx = vel_scale * v_circ * np.cos(ang)
            vy = vel_scale * v_circ * np.sin(ang)
            self.vel = np.array([vx, vy], dtype=np.float64)

        # Reset step counters since we are effectively re-initializing
        self.steps = 0
        self.success_counter = 0

    # --------------------------------------------------------------------------
    # GYM API
    # --------------------------------------------------------------------------
    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
              start_mode: str = "default") -> Tuple[ObsType, dict]:
        """
        Reset the environment state.

        Args:
            seed: optional seed forwarded to Gym base reset (does not seed NumPy here).
            options: reserved for future use.
            start_mode: "default" or "spiral" (demo presets). For task-driven setups,
                        prefer calling set_initial_state() and then reset().

        Returns:
            observation (np.ndarray), info (dict)
        """
        super().reset(seed=seed)
        self.steps = 0
        self.success_counter = 0

        if start_mode == "default":
            # Start slightly outside target orbit with a 30-degree velocity offset.
            self.pos = np.array([0.0, 1.25 * self.target_radius], dtype=np.float64)
            v_mag = np.sqrt(self.mu / np.linalg.norm(self.pos))
            angle = np.deg2rad(30.0)
            self.vel = v_mag * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)

        elif start_mode == "spiral":
            # Start well inside target radius with lower-than-circular speed (spiral-up demo).
            self.pos = np.array([0.0, 0.6 * self.target_radius], dtype=np.float64)
            v_mag = 0.8 * np.sqrt(self.mu / np.linalg.norm(self.pos))
            angle = np.deg2rad(60.0)
            self.vel = v_mag * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)

        else:
            raise ValueError(f"Unknown start_mode: {start_mode}")

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Return observation as [x, y, vx, vy]."""
        return np.concatenate([self.pos, self.vel]).astype(np.float32)

    def _inside_tolerance(self, pos: np.ndarray, vel: np.ndarray) -> bool:
        """
        Check whether the current state is within the success tolerance window.
        Tolerances compare current radius/speed/flight-angle to target circular-orbit references.
        """
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        v_target = np.sqrt(self.mu / self.target_radius)

        r_err = abs(r - self.target_radius) / self.target_radius
        v_err = abs(v - v_target) / v_target

        # Angle between position vector (radial) and velocity vector.
        # For circular orbit, we want them perpendicular (dot -> 0, |cos| -> 0).
        ur = pos / (r + 1e-8)
        uv = vel / (v + 1e-8)
        ang = abs(np.dot(ur, uv))  # want near 0

        return (r_err < self.tol_r) and (v_err < self.tol_v) and (ang < self.tol_ang)

    def step(self, action: np.ndarray):
        """
        Advance one simulation step with the given action.
        Action is a 2D vector in [-1, 1]; we scale it to a thrust vector in Newtons.

        Returns:
            obs (np.ndarray), reward (float), done (bool), info (dict)
        """
        self.steps += 1

        # Thrust from action
        action = np.clip(action, -1.0, 1.0)
        thrust = self.thrust_scale * action  # [Nx, Ny], Newtons
        acc_thrust = thrust / max(1e-12, self.mass)  # m/s^2

        # Gravity from point mass at the origin
        r_vec = self.pos
        r = np.linalg.norm(r_vec)
        acc_gravity = -self.mu * r_vec / ((r ** 3) + 1e-12)

        # Numerical safety clamp (prevents rare blow-ups with tiny r)
        acc_gravity = np.clip(acc_gravity, -1e-2, 1e-2)

        # Integrate with simple (explicit) Euler
        self.vel = self.vel + (acc_gravity + acc_thrust) * self.dt
        self.pos = self.pos + self.vel * self.dt

        # Success window tracking
        if self._inside_tolerance(self.pos, self.vel):
            self.success_counter += 1
        else:
            self.success_counter = 0

        # Termination conditions
        r_now = np.linalg.norm(self.pos)
        time_up = self.steps >= self.max_steps
        out_range = r_now > 2.5 * self.target_radius
        success = self.success_counter >= self.success_threshold
        done = bool(time_up or out_range or success)

        # Reward shaping (delegated to rewards_utils.compute_reward)
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

        # Terminal bonus/penalty
        term_bonus = 0.0
        if done:
            if success:
                term_bonus += self.term_reward_success
            elif out_range:
                term_bonus += self.term_reward_fail
        reward += term_bonus

        info = {
            "reward": float(reward),
            "shaping": float(shaping),
            "bonus": float(bonus),
            "penalty": float(penalty),
            "radius_error": float(r_err),
            "speed_error": float(v_err),
            "steps": int(self.steps),
            "success_counter": int(self.success_counter),
            "terminal_bonus": float(term_bonus),
            "success": bool(success),
            # Diagnostics that can help with massive-ship scenarios:
            "a_cap_ref": float(self.a_cap) if self.a_cap is not None else None,
            "mass": float(self.mass),
            "thrust_scale": float(self.thrust_scale),
            "dt": float(self.dt),
        }

        return self._get_obs(), float(reward), bool(done), info

    def render(self, mode="human"):
        """Minimal text renderer for quick debugging."""
        print(f"Step {self.steps} | pos: {self.pos}, vel: {self.vel}")
