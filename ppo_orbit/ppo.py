import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from envs.orbit_env import OrbitEnv
import numpy as np
from simulator.visualize import plot_trajectory
import matplotlib.pyplot as plt
import csv
import os

# PPO Hyperparameters
GAMMA = 0.99           # Discount factor
LAMBDA = 0.90          # GAE lambda
LR = 5e-5              # Learning rate
EPOCHS = 1000          # Total training epochs
TRAIN_ITERS = 20       # Policy update iterations per epoch
THRUST_SCALE = 8.0     # Scaling factor for thrust magnitude
CHECKPOINT_EVERY = 50  # Save model every N epochs
ENTROPY_COEF = 0.003

# Use float32 and configure GPU/CPU device
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalize state for better PPO performance
def normalize_state(state):
    pos_scale = 7.5e12
    vel_scale = 3e4
    return np.array([
        state[0] / pos_scale,
        state[1] / pos_scale,
        state[2] / vel_scale,
        state[3] / vel_scale
    ], dtype=np.float32)

# PPO Actor-Critic model with Gaussian policy
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.log_std = nn.Parameter(torch.ones(act_dim) * 0.0)  # Log standard deviation for Gaussian

    def act(self, obs):
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(axis=-1)
        value = self.critic(obs).squeeze(-1)
        return action, logprob, value

    def forward(self, obs):
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        value = self.critic(obs).squeeze(-1)
        return dist, value

# Compute GAE Advantage
def compute_advantages(rewards, values, gamma=GAMMA, lam=LAMBDA):
    advantages = []
    gae = 0
    values = values + [0.0]  # Append final value estimate
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

# PPO Clip Loss Function
def ppo_clip_loss(old_logprobs, new_logprobs, advantages, clip=0.2):
    ratio = (new_logprobs - old_logprobs).exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
    return -torch.min(unclipped, clipped).mean()

# Collect one episode (trajectory)
def collect_trajectory(env, model):
    state = env.reset()[0]
    done = False
    states, actions, rewards, values, logprobs, positions = [], [], [], [], [], []

    while not done:
        norm_state = normalize_state(state)
        state_tensor = torch.tensor(norm_state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action, logprob, value = model.act(state_tensor)

        next_state, reward, done, _ = env.step(action.squeeze().cpu().numpy())

        # Save trajectory data
        states.append(state_tensor.squeeze(0))
        actions.append(action.squeeze(0))
        logprobs.append(logprob)
        values.append(value)
        rewards.append(torch.tensor(reward, dtype=torch.float32))
        positions.append(env.pos.copy())

        state = next_state

    return states, actions, rewards, values, logprobs, positions

# === Evaluate and visualize PPO trajectory ===
def evaluate_and_plot(model, thrust_scale=THRUST_SCALE):
    env = OrbitEnv(max_steps=8000, thrust_scale=thrust_scale)
    state = env.reset()[0]
    done = False
    trajectory = []

    while not done:
        norm_state = normalize_state(state)
        state_tensor = torch.tensor(norm_state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action, _, _ = model.act(state_tensor)

        state, _, done, _ = env.step(action.squeeze().cpu().numpy())
        trajectory.append(env.pos.copy())

    trajectory = np.array(trajectory)
    plot_trajectory(trajectory, title="Final PPO Trajectory", target_radius=env.target_radius)
    plt.show()

# PPO Training Loop
def train():
    best_return = -np.inf
    env = OrbitEnv(max_steps=8000, thrust_scale=THRUST_SCALE)
    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    reward_sums = []

    os.makedirs("checkpoints", exist_ok=True)
    with open("loss_log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Return", "Loss_pi", "Loss_v"])

    for epoch in range(EPOCHS):
        result = collect_trajectory(env, model)
        if len(result[0]) == 0:
            print(f"Epoch {epoch + 1}: Empty trajectory.")
            continue

        states, actions, rewards, values, logprobs, _ = result
        advantages = compute_advantages(rewards, [v.item() for v in values])
        returns = [a + v.item() for a, v in zip(advantages, values)]

        advantages = (torch.tensor(advantages) - torch.mean(torch.tensor(advantages))) / \
                     (torch.std(torch.tensor(advantages)) + 1e-8)

        try:
            # Convert to tensors
            states = torch.stack(states).to(device)
            actions = torch.stack(actions).to(device)
            logprobs = torch.stack(logprobs).to(device)
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            advantages = advantages.to(device)

            # Update policy and value network
            for _ in range(TRAIN_ITERS):
                dist, values_pred = model(states)
                new_logprobs = dist.log_prob(actions).sum(axis=-1)

                loss_pi = ppo_clip_loss(logprobs, new_logprobs, advantages)
                loss_v = ((returns - values_pred) ** 2).mean()
                loss = loss_pi + 0.5 * loss_v

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_return = sum([r.item() for r in rewards])
            reward_sums.append(total_return)
            print(f"[Epoch {epoch + 1}] Return = {total_return:.2f}, Loss_pi = {loss_pi.item():.4f}, Loss_v = {loss_v.item():.4f}")

            if total_return > best_return:
                best_return = total_return
                torch.save(model.state_dict(), "ppo_best_model.pth")
                print(f"[Model Saved] Best Return: {best_return:.2f} at Epoch {epoch + 1}")

            with open("loss_log.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, total_return, loss_pi.item(), loss_v.item()])

            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                torch.save(model.state_dict(), f"checkpoints/ppo_epoch_{epoch + 1}.pth")

        except Exception as e:
            print(f"Epoch {epoch + 1}: Error {e}. Skipping.")

    torch.save(model.state_dict(), "ppo_controller_gaussian.pth")

    # Plot training reward curve
    plt.figure(figsize=(8, 5))
    plt.plot(reward_sums, marker='o', color='royalblue')
    plt.xlabel("Epoch")
    plt.ylabel("Total Return")
    plt.title("PPO Training Reward Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_reward_curve_gaussian.png", dpi=300)
    plt.show()

    try:
        print("\n[Evaluation] Loading best model and generating trajectory...\n")
        model.load_state_dict(torch.load("ppo_best_model.pth", weights_only=True))
        torch.save(model.state_dict(), "ppo_orbit.pth")
        print(" Best model weights loaded and saved to ppo_orbit.pth")

        evaluate_and_plot(model)
    except Exception as e:
        print(f"[️Evaluation Error] Failed to evaluate or plot: {e}")


if __name__ == "__main__":
    train()

