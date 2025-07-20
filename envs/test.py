from orbit_env import OrbitEnv

env = OrbitEnv()
obs, _ = env.reset()
print("Initial obs:", obs)

for _ in range(5):
    action = env.action_space.sample()  # 随机动作
    obs, reward, done, info = env.step(action)
    print(f"obs: {obs}, reward: {reward:.5f}, done: {done}")
