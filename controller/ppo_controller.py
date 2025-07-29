import torch
import numpy as np
from ppo_orbit.ppo import ActorCritic


class PPOController:
    """
    A PPO-based controller that maps current position and velocity
    to a thrust vector using a trained neural network policy.

    Compatible with the controller interface: __call__(t, pos, vel)
    """

    def __init__(self,
                 model_path="ppo_orbit/ppo_best_model.pth",
                 normalize=True,
                 device="cpu",
                 verbose=False):
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
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.normalize = normalize
        self.verbose = verbose

        # Constants for normalization (should match training)
        self.pos_scale = 7.5e12
        self.vel_scale = 3e4

    def _normalize_state(self, pos, vel):
        """
        Normalize the orbital state to match the training scale.
        """
        return np.array([
            pos[0] / self.pos_scale,
            pos[1] / self.pos_scale,
            vel[0] / self.vel_scale,
            vel[1] / self.vel_scale
        ], dtype=np.float32)

    def __call__(self, t, pos, vel):
        """
        Compute thrust based on the current time, position, and velocity.

        Args:
            t (float): Current simulation time (not used in PPO).
            pos (np.array): Current position [x, y].
            vel (np.array): Current velocity [vx, vy].

        Returns:
            np.array: Thrust vector [Tx, Ty] predicted by PPO.
        """
        state = self._normalize_state(pos, vel) if self.normalize else np.concatenate([pos, vel])
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_mean, _ = self.model(state_tensor)  # Gaussian mean only

        thrust = action_mean.squeeze().cpu().numpy()

        if self.verbose:
            print(f"[PPOController] t={t:.1f}, pos={pos}, vel={vel} → thrust={thrust}")

        return thrust
