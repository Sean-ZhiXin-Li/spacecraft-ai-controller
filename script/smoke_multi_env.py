import numpy as np
from envs.orbit_env import OrbitEnv
from envs.task_sampler import TaskSampler
from envs.multi_orbit_env import MultiOrbitEnv

def main():
    sampler = TaskSampler("ab/day36/task_specs", mode="sequential")
    base = OrbitEnv()
    env = MultiOrbitEnv(base, sampler, normalize_obs=False)

    obs, info = env.reset()
    print("[reset] obs shape:", obs.shape, "| mass:", env.base.mass, "| thrust_scale:", env.base.thrust_scale)

    steps, ret = 0, 0.0
    while steps < 200:
        # random action in [-1,1]^2
        a = np.random.uniform(-1.0, 1.0, size=(2,))
        obs, r, done, info = env.step(a)
        ret += r
        steps += 1
        if done:
            break

    print(f"[rollout] steps={steps}, return={ret:.2f}, done={done}, success={info.get('success', False)}")

if __name__ == "__main__":
    main()
