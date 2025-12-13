# Smoke test for OrbitEnv

import numpy as np
from src.utils.load_config import load_config

from envs.orbit_env import OrbitEnv


START_MODES = ["default", "spiral"]


def run_smoke(start_mode: str, steps: int = 100) -> None:
    cfg = load_config()

    env = OrbitEnv(
        mass=cfg.spacecraft_mass,
        dt=cfg.simulation.dt,
        max_steps=cfg.simulation.max_steps,
        target_radius=cfg.target_radius,
    )

    obs, info = env.reset(start_mode=start_mode)

    for _ in range(steps):
        action = np.zeros(2, dtype=np.float32)
        obs, reward, done, info = env.step(action)

        assert np.all(np.isfinite(obs)), "Observation has NaN/Inf"
        assert np.all(np.isfinite(action)), "Action has NaN/Inf"

        if done:
            break

    print(f"[start_mode={start_mode}] OK")


if __name__ == "__main__":
    for m in START_MODES:
        run_smoke(m)

    print("\nALL SYSTEM SMOKE TESTS PASSED")
