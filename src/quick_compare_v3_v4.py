import os
import sys
import numpy as np

# Make project root importable so we can import envs/, controller/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from envs.orbit_env import OrbitEnv
from controller.expert_controller import ExpertController as ExpertV3
from controller.expert_controller_improved import ExpertController as ExpertV4


def build_controller_from_obs(ControllerCls, obs):
    """
    Build controller instance using initial radius inferred from obs.
    obs = [pos..., vel...] (1D)
    """
    x = np.asarray(obs)
    if x.ndim != 1 or x.size % 2 != 0:
        raise ValueError(
            f"Unexpected obs shape: {x.shape}. Expected 1D [pos..., vel...]."
        )
    n = x.size // 2
    pos0 = x[:n]
    target_radius = float(np.linalg.norm(pos0))
    return ControllerCls(target_radius=target_radius)


def make_env_for_scenario(scenario_name=None):
    """
    If your OrbitEnv supports set_scenario(name), use it.
    """
    env = OrbitEnv()
    if scenario_name is not None and hasattr(env, "set_scenario"):
        env.set_scenario(scenario_name)
    return env


def run_episode(scenario, label, ControllerCls, max_steps=2000):
    """
    Run one episode and collect several metrics:
    - steps
    - total_reward
    - average radius error
    - average thrust direction jitter (1 - cos(theta))
    """
    env = make_env_for_scenario(scenario)

    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
    else:
        obs, info = reset_result, {}

    controller = build_controller_from_obs(ControllerCls, obs)

    # infer target radius from initial obs
    x0 = np.asarray(obs)
    n = x0.size // 2
    pos0 = x0[:n]
    target_radius = float(np.linalg.norm(pos0))

    total_reward = 0.0
    final_radius = None

    radius_error_sum = 0.0
    jitter_sum = 0.0
    prev_action_dir = None
    effective_steps = 0

    steps = 0
    for _ in range(max_steps):
        action = controller.act(obs)

        step_result = env.step(action)
        # support both 4- and 5-element step outputs
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        total_reward += float(reward)

        # radius error
        x = np.asarray(obs)
        n = x.size // 2
        pos = x[:n]
        r = float(np.linalg.norm(pos))
        radius_error = abs(r - target_radius)
        radius_error_sum += radius_error
        final_radius = r
        effective_steps += 1

        # thrust direction jitter
        a = np.asarray(action)
        a_norm = np.linalg.norm(a)
        if a_norm > 1e-8:
            cur_dir = a / a_norm
            if prev_action_dir is not None:
                cos_theta = float(np.clip(np.dot(prev_action_dir, cur_dir), -1.0, 1.0))
                jitter_sum += 1.0 - cos_theta
            prev_action_dir = cur_dir

        steps += 1
        if done:
            break

    avg_radius_error = radius_error_sum / max(effective_steps, 1)
    avg_jitter = jitter_sum / max(effective_steps - 1, 1)

    print(
        f"[{scenario or 'default'}] {label} | "
        f"steps={steps}, total_reward={total_reward:.3e}, "
        f"final_r={final_radius:.3e}, "
        f"avg_radius_error={avg_radius_error:.3e}, "
        f"avg_jitter={avg_jitter:.3e}"
    )

    return steps, total_reward, final_radius, avg_radius_error, avg_jitter


def main():
    # Week3 failure scenarios + default
    scenarios = [
        "weak_thrust_far",
        "oscillation_noise",
        "misaligned_entry",
        None,  # default
    ]

    for s in scenarios:
        run_episode(s, "ExpertV3", ExpertV3)
        run_episode(s, "ExpertImproved", ExpertV4)


if __name__ == "__main__":
    main()
