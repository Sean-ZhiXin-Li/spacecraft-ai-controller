import numpy as np
from controller.expert_controller_improved import ExpertController


def main():
    print("=== Expert Controller Test ===")

    controller = ExpertController(target_radius=7.5e12)

    obs = np.array([7.5e12, 0.0, 0.0, 3.0e4], dtype=np.float32)

    action = controller.act(obs)

    print("action:", action)
    print("shape:", action.shape)
    print("nan check:", np.isnan(action).any())
    print("range:", action.min(), action.max())


if __name__ == "__main__":
    main()