import numpy as np
from controller.ppo_controller import PPOController


def main():
    # Create controller
    controller = PPOController()

    # Fake observation
    obs = np.array([7.5e12, 0.0, 0.0, 3.0e4], dtype=np.float32)

    # Test act()
    action1 = controller.act(obs)

    # Test __call__()
    action2 = controller(0.0, obs[:2], obs[2:])

    print("=== PPO Controller Test ===")
    print("action from act     :", action1)
    print("action from __call__:", action2)
    print("shape:", action1.shape)
    print("nan check:", np.isnan(action1).any())
    print("range:", action1.min(), action1.max())
    print("same:", np.allclose(action1, action2))


if __name__ == "__main__":
    main()