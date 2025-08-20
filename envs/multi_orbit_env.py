from typing import Optional, Tuple
import numpy as np

from envs.orbit_env import OrbitEnv
from envs.task_sampler import TaskSampler, TaskSpec

class MultiOrbitEnv:
    """
    A meta-environment that:
      1) Pulls a TaskSpec from TaskSampler at reset()
      2) Injects the task's physical params and initial state into OrbitEnv
      3) Delegates step() and observation handling to the underlying OrbitEnv

    This keeps OrbitEnv physics untouched while enabling multi-task training/eval.
    """

    def __init__(self,
                 base_env: OrbitEnv,
                 task_sampler: TaskSampler,
                 normalize_obs: bool = False):
        """
        Args:
            base_env: the inner OrbitEnv instance (physics + reward).
            task_sampler: where tasks are loaded/sampled.
            normalize_obs: if True, normalize observations by (r_target, v_circ).
        """
        self.base = base_env
        self.sampler = task_sampler
        self.normalize_obs = normalize_obs

        # Cache mu for normalization; fallback to Sun if missing.
        self.mu = getattr(self.base, "mu", (getattr(self.base, "G", 6.67430e-11) * getattr(self.base, "M", 1.989e30)))
        self.task: Optional[TaskSpec] = None

    # -------------------------- helpers --------------------------
    def _v_circ(self, r: float) -> float:
        """Circular speed at radius r."""
        return float(np.sqrt(self.mu / max(1e-12, r)))

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize [x, y, vx, vy] by r_target and v_circ(r_target), if enabled.
        """
        if (not self.normalize_obs) or (self.task is None):
            return obs.astype(np.float32)
        x, y, vx, vy = obs
        ps = float(self.task.r_target)
        vs = self._v_circ(self.task.r_target)
        return np.array([x/ps, y/ps, vx/vs, vy/vs], dtype=np.float32)

    # gym-like API
    def reset(self):
        """
        Sample a task, reset base env, then overwrite state from task.
        """
        self.task = self.sampler.sample()

        # Inject physical parameters first
        self.base.set_physical_params(
            mass=self.task.mass,
            thrust_newton=self.task.thrust_newton,
            max_steps=self.task.max_steps,
            r_target=self.task.r_target,
            seed=self.task.seed
        )

        # IMPORTANT: reset first, then apply initial state to avoid being overwritten
        _, info = self.base.reset()

        # Now write the task's initial state (overrides any preset from reset)
        self.base.set_initial_state(self.task.init_state)

        # Build observation from the just-written state
        obs = np.concatenate([self.base.pos, self.base.vel])
        return self._norm_obs(obs), info

    def step(self, action: np.ndarray):
        """
        Forward the action to the base env and normalize obs if needed.
        """
        obs, rew, done, info = self.base.step(action)
        return self._norm_obs(obs), rew, done, info

    # Optional passthroughs
    @property
    def action_space(self):
        return self.base.action_space

    @property
    def observation_space(self):
        return self.base.observation_space
