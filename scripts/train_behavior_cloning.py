from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo_orbit.ppo import normalize_state


DATASET_PATH = PROJECT_ROOT / "analysis" / "phase_controller_dataset" / "phase_controller_dataset_balanced.npz"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "bc_policy.pth"
SUMMARY_PATH = PROJECT_ROOT / "analysis" / "bc_training_summary.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BCPolicy(nn.Module):
    def __init__(self, hidden1: int = 256, hidden2: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(5, hidden1),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.shared(x))


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(DATASET_PATH)
    obs_np = np.asarray(data["observations"], dtype=np.float32)
    obs_norm = np.stack([normalize_state(obs) for obs in obs_np], axis=0).astype(np.float32)
    observations = torch.tensor(obs_norm, dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.float32)

    dataset = TensorDataset(observations, actions)
    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

    model = BCPolicy().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = []
    epochs = 200
    patience = 20
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for obs_batch, action_batch in train_loader:
            obs_batch = obs_batch.to(DEVICE)
            action_batch = action_batch.to(DEVICE)
            pred = model(obs_batch)
            loss = loss_fn(pred, action_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * obs_batch.size(0)
        train_loss /= max(1, len(train_set))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_batch, action_batch in val_loader:
                obs_batch = obs_batch.to(DEVICE)
                action_batch = action_batch.to(DEVICE)
                pred = model(obs_batch)
                loss = loss_fn(pred, action_batch)
                val_loss += loss.item() * obs_batch.size(0)
        val_loss /= max(1, len(val_set))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(json.dumps(history[-1]))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Behavior cloning did not produce a valid model state.")

    torch.save(
        {
            "model_state": best_state,
            "best_val_loss": best_val,
            "history": history,
            "model_type": "bc_policy",
        },
        MODEL_PATH,
    )
    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "dataset": DATASET_PATH.as_posix(),
                "model_path": MODEL_PATH.as_posix(),
                "best_val_loss": best_val,
                "epochs_ran": len(history),
                "train_size": n_train,
                "val_size": n_val,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved BC model to: {MODEL_PATH}")
    print(f"Saved training summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
