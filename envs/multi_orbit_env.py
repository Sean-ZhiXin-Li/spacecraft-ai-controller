# envs/multi_orbit_env.py
# Day 43 — MultiOrbitEnv with cfg-mapping ctor + default TaskSampler
from __future__ import annotations
from typing import Optional, Any, Dict
import numpy as np

from envs.orbit_env import OrbitEnv
from envs.task_sampler import TaskSampler, TaskSpec


class _DefaultTaskSampler(TaskSampler):
    """
    Minimal default sampler:
    - Generates a near-circular orbit at r_target with random angle.
    - Fills required TaskSpec fields with safe defaults for smoke tests.
    """

    def __init__(
        self,
        mu: float,
        *,
        r_min: float = 6.7e6,
        r_max: float = 8.5e6,
        mass: float = 720.0,
        thrust_newton: float = 0.4,
        max_steps: int = 200,
        name_prefix: str = "auto_circ",
    ):
        self.mu = float(mu)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.mass = float(mass)
        self.thrust_newton = float(thrust_newton)
        self.max_steps = int(max_steps)
        self.name_prefix = str(name_prefix)
        self._counter = 0

    def _vcirc(self, r: float) -> float:
        r = max(1e-12, float(r))
        return float(np.sqrt(self.mu / r))

    def sample(self, *, seed: Optional[int] = None) -> TaskSpec:
        rng = np.random.default_rng(seed)
        r = float(rng.uniform(self.r_min, self.r_max))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        v = self._vcirc(r)

        x, y = r * np.cos(theta), r * np.sin(theta)
        # velocity tangential to position (clockwise orthogonal)
        vx, vy = -v * np.sin(theta), v * np.cos(theta)

        self._counter += 1
        name = f"{self.name_prefix}_{self._counter}"

        return TaskSpec(
            name=name,
            mass=self.mass,
            thrust_newton=self.thrust_newton,
            r_target=r,
            max_steps=self.max_steps,
            seed=seed,
            init_state=np.array([x, y, vx, vy], dtype=np.float64),
        )


class MultiOrbitEnv:
    """
    A meta-environment that:
      1) Pulls a TaskSpec from TaskSampler at reset()
      2) Injects the task's physical params and initial state into OrbitEnv
      3) Delegates step() and observation handling to the underlying OrbitEnv

    Keeps OrbitEnv physics untouched while enabling multi-task training/eval.
    """

    def __init__(
        self,
        base_env_or_cfg: OrbitEnv | Any,
        task_sampler: Optional[TaskSampler] = None,
        normalize_obs: bool = False,
        eps: float = 1e-12,
    ):
        """
        Args:
            base_env_or_cfg: either an OrbitEnv instance OR a cfg-like object
                             (with mu/dt/max_steps and optionally target_radius/r_target).
            task_sampler: where tasks are loaded/sampled. If None, a safe default is used.
            normalize_obs: if True, normalize observations by (r_target, v_circ(r_target)).
            eps: small epsilon to avoid division by zero.
        """
        # Accept OrbitEnv directly, or construct from a cfg-like object by explicit mapping.
        if isinstance(base_env_or_cfg, OrbitEnv):
            self.base = base_env_or_cfg
        else:
            cfg = base_env_or_cfg
            has_mu = hasattr(cfg, "mu")
            has_dt = hasattr(cfg, "dt")
            has_ms = hasattr(cfg, "max_steps")
            if not (has_mu and has_dt and has_ms):
                raise TypeError(
                    "MultiOrbitEnv: cfg-like object must expose mu, dt, and max_steps."
                )

            # Ensure G * M == mu. Pick M=1.0 so G=mu, which matches your SimConfig's μ.
            target_radius = None
            if hasattr(cfg, "target_radius"):
                target_radius = float(getattr(cfg, "target_radius"))
            elif hasattr(cfg, "r_target"):
                target_radius = float(getattr(cfg, "r_target"))

            kwargs = dict(
                G=float(cfg.mu),
                M=1.0,
                dt=float(cfg.dt),
                max_steps=int(cfg.max_steps),
            )
            if target_radius is not None:
                kwargs["target_radius"] = target_radius

            self.base = OrbitEnv(**kwargs)

        self.normalize_obs = bool(normalize_obs)
        self.eps = float(eps)

        # Cache mu for normalization; fallback to G*M if base doesn't store mu.
        G = getattr(self.base, "G", 6.67430e-11)
        M = getattr(self.base, "M", 1.989e30)
        self.mu: float = float(getattr(self.base, "mu", G * M))
        if not hasattr(self.base, "mu"):
            # Keep base.mu consistent for downstream code if missing.
            try:
                setattr(self.base, "mu", self.mu)
            except Exception:
                pass

        # Default sampler if none is provided.
        self.sampler: TaskSampler = task_sampler or _DefaultTaskSampler(self.mu)

        self.task: Optional[TaskSpec] = None

    # ---------------- Internals / helpers ----------------

    def _v_circ(self, r: float) -> float:
        """Circular speed at radius r."""
        r_safe = max(self.eps, float(r))
        return float(np.sqrt(self.mu / r_safe))

    def _ensure_vector_obs(self, obs: Any) -> np.ndarray:
        """
        Ensure we return a 1-D observation vector.
        - If base.step/reset already returns a 1-D array-like, use it.
        - Otherwise fall back to concatenating base.pos/base.vel.
        """
        try:
            arr = np.asarray(obs, dtype=np.float32).reshape(-1)
            if arr.size >= 4:
                return arr
        except Exception:
            pass

        # Fallback: derive observation from base attributes
        if not (hasattr(self.base, "pos") and hasattr(self.base, "vel")):
            raise RuntimeError(
                "Cannot derive observation: base env does not expose .pos/.vel "
                "and returned obs is not a flat vector."
            )
        pos = np.asarray(self.base.pos, dtype=np.float32).reshape(-1)
        vel = np.asarray(self.base.vel, dtype=np.float32).reshape(-1)
        return np.concatenate([pos, vel], dtype=np.float32)

    def _norm_obs(self, obs_vec: np.ndarray) -> np.ndarray:
        """
        Normalize [x, y, vx, vy] by r_target and v_circ(r_target), if enabled.
        If obs has more than 4 dims, only the first four are normalized.
        """
        if (not self.normalize_obs) or (self.task is None):
            return obs_vec.astype(np.float32, copy=False)

        if obs_vec.ndim != 1 or obs_vec.size < 4:
            return obs_vec.astype(np.float32, copy=False)

        ps = max(self.eps, float(self.task.r_target))
        vs = max(self.eps, self._v_circ(self.task.r_target))

        out = obs_vec.astype(np.float32, copy=True)
        out[0] /= ps
        out[1] /= ps
        out[2] /= vs
        out[3] /= vs
        return out

    def _validate_task(self, t: TaskSpec) -> None:
        """Lightweight sanity checks to fail fast on bad tasks."""
        if hasattr(t, "mass") and not (t.mass > 0):
            raise ValueError(f"TaskSpec.mass must be > 0, got {t.mass}")
        if hasattr(t, "thrust_newton") and not (t.thrust_newton >= 0):
            raise ValueError(f"TaskSpec.thrust_newton must be >= 0, got {t.thrust_newton}")
        if hasattr(t, "r_target") and not (t.r_target > 0):
            raise ValueError(f"TaskSpec.r_target must be > 0, got {t.r_target}")
        if hasattr(t, "max_steps") and not (t.max_steps and t.max_steps > 0):
            raise ValueError(f"TaskSpec.max_steps must be > 0, got {t.max_steps}")
        if hasattr(t, "init_state"):
            init = np.asarray(t.init_state, dtype=np.float64).reshape(-1)
            if init.size < 4:
                raise ValueError(
                    f"TaskSpec.init_state must have at least 4 dims [x, y, vx, vy], got {init.shape}"
                )

    def _attach_task_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Attach task metadata into info for easier downstream analysis."""
        info = dict(info) if info is not None else {}
        if self.task is not None:
            task_id = getattr(self.task, "name", None) or getattr(self.task, "id", None)
            if task_id is not None:
                info.setdefault("task_id", task_id)
            info.setdefault(
                "task_params",
                {
                    "mass": getattr(self.task, "mass", None),
                    "thrust_newton": getattr(self.task, "thrust_newton", None),
                    "r_target": getattr(self.task, "r_target", None),
                    "max_steps": getattr(self.task, "max_steps", None),
                },
            )
        return info

    # ---------------- Gym/Gymnasium-like API ----------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """
        Sample a task, reset base env, then overwrite state from task.

        Returns:
            (obs, info)  # Gymnasium-style
        """
        # Sample a task (forward seed if your sampler supports it)
        if seed is not None and hasattr(self.sampler, "sample"):
            try:
                self.task = self.sampler.sample(seed=seed)  # optional signature
            except TypeError:
                self.task = self.sampler.sample()
        else:
            self.task = self.sampler.sample()

        self._validate_task(self.task)

        # Inject physical parameters first, if the method exists.
        if hasattr(self.base, "set_physical_params"):
            self.base.set_physical_params(
                mass=getattr(self.task, "mass", None),
                thrust_newton=getattr(self.task, "thrust_newton", None),
                max_steps=getattr(self.task, "max_steps", None),
                r_target=getattr(self.task, "r_target", None),
                seed=getattr(self.task, "seed", None),
            )
        else:
            # Soft fallback: write attributes if present.
            for k in ("mass", "thrust_newton", "max_steps", "r_target", "seed"):
                if hasattr(self.task, k):
                    setattr(self.base, k, getattr(self.task, k))

        # Reset base env (try Gymnasium signature; fallback to old one).
        try:
            reset_out = self.base.reset(seed=seed, options=options)  # (obs, info)
        except TypeError:
            reset_out = self.base.reset()  # obs OR (obs, info)

        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            base_obs, info = reset_out
        else:
            base_obs, info = reset_out, {}

        # Now write the task's initial state.
        init_state = getattr(self.task, "init_state", None)
        if init_state is not None:
            if hasattr(self.base, "set_initial_state"):
                self.base.set_initial_state(init_state)
            else:
                # Best-effort fallback: write pos/vel if the base exposes them.
                arr = np.asarray(init_state, dtype=np.float64).reshape(-1)
                if arr.size >= 4 and hasattr(self.base, "pos") and hasattr(self.base, "vel"):
                    self.base.pos = np.array(arr[:2], dtype=np.float64)
                    self.base.vel = np.array(arr[2:4], dtype=np.float64)

        # Build observation from the just-written state.
        obs_vec = self._ensure_vector_obs(base_obs)
        obs_vec = self._norm_obs(obs_vec)

        info = self._attach_task_info(info)
        return obs_vec, info

    def step(self, action: np.ndarray):
        """
        Forward the action to the base env and normalize obs if needed.

        Returns a tuple compatible with both Gym (4-tuple) and Gymnasium (5-tuple).
        """
        out = self.base.step(action)

        if not isinstance(out, tuple):
            raise RuntimeError("base.step(action) must return a tuple.")

        if len(out) == 4:
            base_obs, reward, done, info = out
            obs_vec = self._ensure_vector_obs(base_obs)
            obs_vec = self._norm_obs(obs_vec)
            info = self._attach_task_info(info)
            return obs_vec, float(reward), bool(done), info

        elif len(out) == 5:
            base_obs, reward, terminated, truncated, info = out
            obs_vec = self._ensure_vector_obs(base_obs)
            obs_vec = self._norm_obs(obs_vec)
            info = self._attach_task_info(info)
            return obs_vec, float(reward), bool(terminated), bool(truncated), info

        else:
            raise RuntimeError(
                f"Unexpected base.step() return length {len(out)}. "
                "Expected 4-tuple (Gym) or 5-tuple (Gymnasium)."
            )

    # ---------------- Optional passthroughs ----------------

    @property
    def action_space(self):
        return getattr(self.base, "action_space", None)

    @property
    def observation_space(self):
        return getattr(self.base, "observation_space", None)

    def close(self):
        """Pass-through close if base env supports it."""
        if hasattr(self.base, "close"):
            self.base.close()

    @property
    def unwrapped(self):
        """Return the underlying OrbitEnv for advanced access."""
        return self.base
