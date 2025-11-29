import os
import sys
import numpy as np

# Make project root importable so that we can import envs/, controller/

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)  # one level above src/

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from envs.orbit_env import OrbitEnv
from controller.expert_controller_improved import ExpertController


def build_controller_from_obs(obs):
    """
    Build ExpertController v4 using the initial orbit radius inferred
    from the observation vector.

    We assume obs = [pos..., vel...] (1D), so target_radius ≈ ||pos0||.
    This is enough for a smoke test.
    """
    x = np.asarray(obs)
    if x.ndim != 1 or x.size % 2 != 0:
        raise ValueError(
            f"Unexpected obs shape for ExpertController v4: {x.shape}. "
            "Expected 1D vector [pos..., vel...]."
        )
    n = x.size // 2
    pos0 = x[:n]
    target_radius = float(np.linalg.norm(pos0))
    return ExpertController(target_radius=target_radius)


def main():
    env = OrbitEnv()

    # reset(): support both (obs) and (obs, info)
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
    else:
        obs, info = reset_result, {}

    controller = build_controller_from_obs(obs)

    total_steps = 0
    for step in range(2000):
        action = controller.act(obs)

        # step(): support both
        # (obs, reward, done, info)  [older Gym style]
        # and
        # (obs, reward, terminated, truncated, info)  [newer Gymnasium style]
        step_result = env.step(action)

        if isinstance(step_result, tuple) and len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        elif isinstance(step_result, tuple) and len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated, truncated = done, False
        else:
            raise ValueError(
                f"Unexpected env.step() return format: len={len(step_result)}"
            )

        total_steps += 1
        if done:
            break

    print(f"[smoke_test_v4] finished after {total_steps} steps.")


if __name__ == "__main__":
    main()
