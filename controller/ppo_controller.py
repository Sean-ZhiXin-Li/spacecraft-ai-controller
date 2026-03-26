import torch
import numpy as np
from ppo_orbit.ppo import ActorCritic


class PPOController:
    """
    A PPO-based controller that maps current position and velocity
    to a normalized action vector using a trained neural network policy.

    Compatible with the controller interface: __call__(t, pos, vel)

    IMPORTANT:
    - This controller returns normalized action in [-1, 1].
    - It does NOT apply thrust scaling internally.
    - The environment / runner should perform the only physical scaling.
    """

    def __init__(
        self,
        model_path="ppo_orbit/ppo_best_model.pth",
        normalize=True,
        device="cpu",
        verbose=False,
    ):
        """
        Initialize the PPO controller.

        Args:
            model_path (str): Path to the saved PyTorch PPO model weights.
            normalize (bool): Whether to normalize the state inputs (recommended).
            device (str): Device to load the model on (e.g., "cpu" or "cuda").
            verbose (bool): Print debug info if True.
        """
        self.device = device
        self.model = ActorCritic().to(self.device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Support both:
        # 1) raw state_dict
        # 2) checkpoint dict with "model_state"
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        self.normalize = normalize
        self.verbose = verbose

        # Constants for normalization (must match training)
        self.pos_scale = 7.5e12
        self.vel_scale = 3e4

    def _normalize_state(self, pos, vel):
        """
        Normalize the orbital state to match the training scale.
        """
        return np.array(
            [
                pos[0] / self.pos_scale,
                pos[1] / self.pos_scale,
                vel[0] / self.vel_scale,
                vel[1] / self.vel_scale,
            ],
            dtype=np.float32,
        )

    def __call__(self, t, pos, vel):
        """
        Compute normalized action based on the current time, position, and velocity.

        Args:
            t (float): Current simulation time (not used in PPO).
            pos (np.array): Current position [x, y].
            vel (np.array): Current velocity [vx, vy].

        Returns:
            np.array: Normalized action vector in range [-1, 1].
        """
        # Prepare state
        state = (
            self._normalize_state(pos, vel)
            if self.normalize
            else np.concatenate([pos, vel]).astype(np.float32)
        )

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            # ActorCritic.forward() returns (mu, value), not a distribution
            mu, _ = self.model(state_tensor)

            # Deterministic inference: use mean action instead of sampling
            action = torch.clamp(mu, -1.0, 1.0)

            # Convert to numpy
            action_np = action.squeeze(0).cpu().numpy().astype(np.float32)

        if self.verbose:
            print(
                f"[PPOController] t={t:.2f}, "
                f"pos={pos}, vel={vel}, action={action_np}"
            )

        return action_np


