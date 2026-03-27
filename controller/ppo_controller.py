import torch
import numpy as np
from ppo_orbit.ppo_infer_model import ActorCriticInference


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
        self.model = ActorCriticInference().to(self.device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Support both:
        # 1) raw state_dict
        # 2) checkpoint dict with "model_state"
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

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

    def act(self, obs):
        """
        Compute normalized action from observation.

        Args:
            obs (np.ndarray): [x, y, vx, vy]

        Returns:
            np.ndarray: action in [-1, 1]
        """
        obs = np.asarray(obs, dtype=np.float32).ravel()
        n = obs.size // 2

        pos = obs[:n]
        vel = obs[n:]

        # Use existing normalization logic
        state = (
            self._normalize_state(pos, vel)
            if self.normalize
            else obs.astype(np.float32)
        )

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            mu, _ = self.model(state_tensor)
            action = torch.clamp(mu, -1.0, 1.0)

            action_np = action.squeeze(0).cpu().numpy().astype(np.float32)

        if self.verbose:
            print(
                f"[PPOController.act] obs={obs}, action={action_np}"
            )

        return action_np

    def __call__(self, t, pos, vel):
        """
        Backward-compatible wrapper.

        Converts (pos, vel) into obs and calls act().
        """
        obs = np.concatenate([pos, vel]).astype(np.float32)
        return self.act(obs)


