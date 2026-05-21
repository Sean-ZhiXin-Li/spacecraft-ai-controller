from envs.orbit_env import OrbitEnv


def main():
    env = OrbitEnv()

    obs, info = env.reset(seed=0)
    print("Reset obs shape:", obs.shape)
    print("Reset info:", info)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    print("Action:", action)
    print("Action shape:", action.shape)
    print("Next obs shape:", next_obs.shape)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info keys:", list(info.keys())[:20])


if __name__ == "__main__":
    main()