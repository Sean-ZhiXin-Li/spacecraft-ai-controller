import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_dim = 4, action_dim = 2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128,1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x)

    def act(self, state):
        with torch.no_grad():
            policy_logits, _ = self.forward(state)
            action = torch.tanh(policy_logits)  # action in [-1,1]
        return action

