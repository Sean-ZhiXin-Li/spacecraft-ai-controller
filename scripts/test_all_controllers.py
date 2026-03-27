import numpy as np
from controller.factory import get_controller


def main():
    print("=== All Controllers Test ===")

    obs = np.array([7.5e12, 0.0, 0.0, 3.0e4], dtype=np.float32)

    for name in ["expert", "ppo"]:
        print(f"\ncontroller = {name}")

        controller = get_controller(name)
        action = controller.act(obs)

        print("action:", action)
        print("shape:", action.shape)
        print("nan:", np.isnan(action).any())


if __name__ == "__main__":
    main()