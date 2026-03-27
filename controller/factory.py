from controller.expert_controller_improved import ExpertController
from controller.ppo_controller import PPOController


def get_controller(name):
    """
    Factory function to create controllers.

    Args:
        name (str): controller name

    Returns:
        controller instance
    """
    if name == "expert":
        return ExpertController(target_radius=7.5e12)

    elif name == "ppo":
        return PPOController()

    else:
        raise ValueError(f"Unknown controller: {name}")