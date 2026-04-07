import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.orbit_env import OrbitEnv
from controller.stable_orbit_controller import StableOrbitController


def run_once(seed: int = 0, max_steps: int = 6000):
    env = OrbitEnv(thrust_scale=120.0, dt=6000.0, max_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    ctrl = StableOrbitController(target_radius=env.target_radius, mu=env.mu)

    for _ in range(max_steps):
        action = ctrl.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("success", False):
            return True, info
        if terminated or truncated:
            return False, info
    return False, info


if __name__ == "__main__":
    ok, info = run_once(seed=0, max_steps=6000)
    print("success:", ok)
    print(
        "radius_error:",
        float(info.get("radius_error", np.nan)),
        "speed_error:",
        float(info.get("speed_error", np.nan)),
        "steps:",
        int(info.get("steps", -1)),
    )
    raise SystemExit(0 if ok else 1)

