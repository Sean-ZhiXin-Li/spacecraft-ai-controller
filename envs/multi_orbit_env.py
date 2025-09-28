from dataclasses import asdict
from typing import Dict, Any, Optional, Tuple
import math
import numpy as np

from .orbit_presets import PRESET_MAP, OrbitParams, v_circ
from .orbit_env import OrbitEnv  # your existing env

class MultiOrbitEnv:
    def __init__(self,
                 scenario: str = "circular",
                 preset_overrides: Optional[Dict[str, Any]] = None):
        """
        Wrapper that selects a scenario preset and configures the inner OrbitEnv.
        - scenario: 'circular' | 'elliptic' | 'transfer'
        - preset_overrides: shallow dict to override fields in the preset
        """
        if scenario not in PRESET_MAP:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario_name = scenario
        params = PRESET_MAP[scenario]
        if preset_overrides:
            params = OrbitParams(**{**asdict(params), **preset_overrides})
        self.params = params
        self.env = OrbitEnv()
        self.episode_stats: Dict[str, Any] = {}

    # --- Hohmann transfer helper (Δv1 at r1) ---
    def _hohmann_delta_v1(self, mu: float, r1: float, r2: float) -> float:
        a_t = 0.5 * (r1 + r2)
        v_c1 = math.sqrt(mu / r1)
        v_peri_transfer = math.sqrt(mu * (2.0 / r1 - 1.0 / a_t))
        return v_peri_transfer - v_c1

    def _infer_r2(self) -> float:
        default_r2 = 8.0e12
        notes = self.params.notes or ""
        if "r2=" in notes:
            frag = notes.split("r2=")[1]
            buf = ""
            for ch in frag:
                if ch in "0123456789.+-eE":
                    buf += ch
                else:
                    break
            try:
                return float(buf)
            except Exception:
                return default_r2
        return default_r2

    def reset(self, seed: Optional[int] = None):
        """
        Pass Day48 preset keys to your OrbitEnv.reset via **cfg mapping:
        - mu -> GM (your reset maps mu to M,mu already)
        - sc_mass -> mass
        - r0 -> target_radius
        """
        cfg = asdict(self.params)
        try:
            obs, info = self.env.reset(seed=seed, start_mode="default", **cfg)
        except TypeError:
            # Fallback if your reset signature rejects 'seed' or 'start_mode'
            obs, info = self.env.reset(**cfg)

        dv1 = 0.0
        if self.scenario_name == "transfer":
            r1 = self.params.r0
            r2 = self._infer_r2()
            dv1 = self._hohmann_delta_v1(self.params.mu, r1, r2)
            # Try tangential impulse (you implemented apply_delta_v in Step 1)
            try:
                self.env.apply_delta_v(dv1)
            except AttributeError:
                pass

        self.episode_stats = {"scenario": self.scenario_name,
                              "steps": 0, "fuel_used": 0.0, "dv1": float(dv1)}
        return obs  # your env returns (obs, info); wrappers often only need obs

    def _step_unpack(self, out) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Support both classic Gym 4-tuple and Gymnasium 5-tuple.
        """
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
            return obs, float(reward), done, info or {}
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            return obs, float(reward), bool(done), info or {}
        else:
            raise RuntimeError("Unexpected env.step() return signature")

    def step(self, action: np.ndarray):
        out = self.env.step(action)
        obs, reward, done, info = self._step_unpack(out)
        self.episode_stats["steps"] += 1
        if isinstance(info, dict) and "fuel" in info:
            self.episode_stats["fuel_used"] = info["fuel"]
        return obs, reward, done, info

    def rollout(self, max_steps: int = 2048):
        """
        Zero-thrust baseline rollout for logging/comparison.
        """
        obs = self.reset()
        total_r = 0.0
        last_info: Dict[str, Any] = {}
        for _ in range(max_steps):
            # Create a zero-like action with correct shape
            try:
                a = self.env.action_space_sample()
                a = np.zeros_like(a)
            except Exception:
                a = np.zeros(2, dtype=np.float32)
            obs, r, done, info = self.step(a)
            last_info = info if isinstance(info, dict) else {}
            total_r += r
            if done:
                break
        metrics = {
            **self.episode_stats,
            "total_reward": float(total_r),
            "final_orbit_error": float(
                last_info.get("radius_error", np.nan)
            ),
        }
        return metrics
