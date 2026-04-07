from controller.expert_controller_improved import ExpertController
from controller.ppo_controller import PPOController
from controller.stable_orbit_controller import StableOrbitController


def get_controller(name):
    """
    Factory function to create controllers.

    Args:
        name (str): controller name

    Returns:
        controller instance
    """
    if name == "expert":
        return StableOrbitController(target_radius=7.5e12)
    elif name == "expert_improved":
        return ExpertController(target_radius=7.5e12)
    elif name == "stable":
        return StableOrbitController(target_radius=7.5e12)

    elif name == "ppo":
        return PPOController()

    else:
        raise ValueError(f"Unknown controller: {name}")