import torch
import torch.nn as nn
import numpy as np

class MimicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

class ImitationController:
    def __init__(self, model_path="controller/mimic_model_V6_1.pth", clip=True):
        self.model = MimicNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.clip = clip

    def __call__(self, t, pos, vel):
        # Convert state to normalized torch tensor
        state = np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float32)
        state_tensor = torch.tensor(state).unsqueeze(0)  # Add batch dim

        with torch.no_grad():
            thrust = self.model(state_tensor).squeeze(0).numpy()

        if self.clip:
            thrust = np.clip(thrust, -1.0, 1.0)

        return thrust
