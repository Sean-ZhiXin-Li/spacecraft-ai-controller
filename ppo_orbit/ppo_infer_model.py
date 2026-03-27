import torch
import torch.nn as nn


class ActorCriticInference(nn.Module):
    """
    Actor-Critic model for PPO inference.

    IMPORTANT:
    - This architecture is reconstructed from the saved checkpoint.
    - It must exactly match the training-time model structure.
    - Do not modify this class unless the PPO model is retrained.
    """

    def __init__(self):
        super().__init__()

        # Actor network reconstructed from checkpoint keys:
        # actor.0 -> Linear(4, 256)
        # actor.1 -> LayerNorm(256)
        # actor.3 -> Linear(256, 256)
        # actor.5 -> Linear(256, 2)
        self.actor = nn.Sequential(
            nn.Linear(4, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 2),
        )

        # Critic network reconstructed from checkpoint keys:
        # critic.0 -> Linear(4, 256)
        # critic.2 -> Linear(256, 256)
        # critic.4 -> Linear(256, 1)
        self.critic = nn.Sequential(
            nn.Linear(4, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # Global log standard deviation parameter saved in checkpoint.
        # It is not used for deterministic inference, but must exist so the
        # checkpoint can be loaded correctly.
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input state tensor with shape (batch_size, 4).

        Returns:
            tuple:
                mu (torch.Tensor): Action mean with shape (batch_size, 2).
                value (torch.Tensor): Value estimate with shape (batch_size, 1).
        """
        mu = self.actor(x)
        value = self.critic(x)
        return mu, value
