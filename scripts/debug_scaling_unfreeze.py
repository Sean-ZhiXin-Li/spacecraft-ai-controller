# scripts/debug_scaling_unfreeze.py
# Purpose: verify thrust_scale -> a_eff monotonicity under fixed action/state

import numpy as np

from envs.orbit_env import OrbitEnv

def one_step_a_eff(env, action):
    obs, reward, terminated, truncated, info = env.step(action)
    a_eff = float(np.linalg.norm(info["acc_thrust"]))
    thrust_norm = float(np.linalg.norm(info["thrust_vec"]))
    return a_eff, thrust_norm, info


def scaling_unfreeze_check(env, base_action=None):
    if base_action is None:
        base_action = np.array([1.0, 0.0], dtype=np.float32)

    base_action = np.clip(base_action, -1.0, 1.0).astype(np.float32)

    # snapshot identical state
    s0 = env.get_state()

    # thrust = 200
    env.set_state(s0)
    env.set_physical_params(thrust_newton=200.0)
    a1, t1, _ = one_step_a_eff(env, base_action)

    # thrust = 2000
    env.set_state(s0)
    env.set_physical_params(thrust_newton=2000.0)
    a2, t2, _ = one_step_a_eff(env, base_action)

    print("[CHECK] action =", base_action)
    print(f"[CHECK] thrust=200   thrust_norm={t1:.6g}  a_eff={a1:.6g}")
    print(f"[CHECK] thrust=2000  thrust_norm={t2:.6g}  a_eff={a2:.6g}")
    print(f"[RATIO] thrust_norm ratio = {t2/(t1+1e-12):.6g}")
    print(f"[RATIO] a_eff ratio        = {a2/(a1+1e-12):.6g}")


if __name__ == "__main__":
    env = OrbitEnv()
    env.reset()
    scaling_unfreeze_check(env)
