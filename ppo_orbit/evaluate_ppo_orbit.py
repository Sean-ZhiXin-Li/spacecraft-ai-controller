import torch
import torch.nn as nn
import numpy as np

class PPOActor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2):
        super(PPOActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class PPOController:
    """
    PPOController uses a trained PPO policy network to generate thrust vectors
    for spacecraft orbital control.
    """

    def __init__(self, model_path, normalize=True, device="cpu", verbose=False):
        """
        Initialize the PPO controller.

        Args:
            model_path (str): Path to the trained PPO model (.pth file).
            normalize (bool): Whether to normalize input state.
            device (str): Device to load the model on ("cpu" or "cuda").
            verbose (bool): Whether to print debug info.
        """
        self.normalize = normalize
        self.device = device
        self.verbose = verbose

        # Load model
        self.model = PPOActor()
        try:
            # Use safe loading with weights_only=True (requires PyTorch ≥ 2.2)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, t, pos, vel):
        """
        Compute thrust based on current state.

        Args:   
            t (float): Current time (not used).
            pos (np.ndarray): Position vector [x, y].
            vel (np.ndarray): Velocity vector [vx, vy].

        Returns:
            np.ndarray: Thrust vector in 2D space, range [-1, 1].
        """
        state = np.concatenate([pos, vel]).astype(np.float32)

        if self.normalize:
            pos_scale = 7.5e12
            vel_scale = 3e4
            state = np.array([
                state[0] / pos_scale,
                state[1] / pos_scale,
                state[2] / vel_scale,
                state[3] / vel_scale
            ], dtype=np.float32)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.model(state_tensor).squeeze(0).cpu().numpy()

        if self.verbose:
            print(f"[PPO] t={t:.1f}, pos={pos}, vel={vel}, action={action}")

        return np.clip(action, -1.0, 1.0)