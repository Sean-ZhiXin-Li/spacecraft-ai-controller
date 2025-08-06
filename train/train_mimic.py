import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
from tqdm import tqdm

# Custom Dataset for Expert Trajectories
class ThrustDataset(Dataset):
    def __init__(self, file_paths, downsample=40):
        """
        Load multiple expert trajectory files, merge, downsample, and normalize them.

        Args:
            file_paths (list): List of .npy file paths.
            downsample (int): Step size to sample (e.g. 10 = take 1 every 10 steps).
        """
        self.data = []
        for path in file_paths:
            arr = np.load(path)[::downsample]
            self.data.append(arr)
        self.data = np.concatenate(self.data, axis=0)

    def normalize_state(self, state):
        pos_scale = 7.5e12
        vel_scale = 3e4
        return np.array([
            state[0] / pos_scale,
            state[1] / pos_scale,
            state[2] / vel_scale,
            state[3] / vel_scale
        ], dtype=np.float32)

    def normalize_thrust(self, thrust):
        thrust_scale = 1.0
        return np.clip(thrust / thrust_scale, -1.0, 1.0).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        state = self.normalize_state(sample[:4])
        thrust = self.normalize_thrust(sample[4:])
        return torch.tensor(state, dtype=torch.float32), torch.tensor(thrust, dtype=torch.float32)


# Define a simple MLP model
class MimicNet(nn.Module):
    def __init__(self):
        """
        MLP with 2 hidden layers to predict thrust from orbital state.
        Input:  [pos_x, pos_y, vel_x, vel_y]
        Output: [thrust_x, thrust_y]
        """
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

# Train the model
def train(model, dataloader, epochs=20, lr=1e-3):
    """
    Train the MLP model on expert data.

    Args:
        model:       MimicNet instance
        dataloader:  PyTorch DataLoader
        epochs:      Number of training epochs
        lr:          Learning rate
    """
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()       # Reset gradients
            output = model(x_batch)     # Forward pass
            loss = criterion(output, y_batch)  # Compute loss
            loss.backward()             # Backpropagation
            optimizer.step()            # Update weights
            total_loss += loss.item()   # Track epoch loss

            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.6f}")

# Main execution block
if __name__ == "__main__":
    # List the merge expert dataset
    file_paths = glob.glob("E:/spacecraft_ai_project/data/data/preprocessed/merged_expert_dataset.npy")
    print("Matched files:", file_paths)

    # Initialize dataset and data loader
    dataset = ThrustDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize model
    model = MimicNet()

    # Train model
    train(model, dataloader, epochs=20, lr=1e-3)

    # Save model weights
    os.makedirs("controller", exist_ok=True)
    torch.save(model.state_dict(), "controller/mimic_model_V6_1.pth")
    print("Model saved to controller/mimic_model_V6_1.pth")
