import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from envs.orbit_env import OrbitEnv

# Enable CUDA memory expansion
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.95
LR = 1e-5
EPOCHS = 300
TRAIN_ITERS = 10
THRUST_SCALE = 3000

# Normalize observation
def normalize_state(state):
    pos_scale = 7.5e12
    vel_scale = 3e4
    return np.array([
        state[0] / pos_scale,
        state[1] / pos_scale,
        state[2] / vel_scale,
        state[3] / vel_scale
    ], dtype=np.float32)

# PPO Actor-Critic with Gaussian policy
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            mu, _ = self.forward(state)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        action = dist.sample()
        return action.cpu().numpy(), dist.log_prob(action).sum()

    def evaluate(self, states, actions):
        mu, value = self.forward(states)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        return log_probs, entropy, value.squeeze()

# Generalized Advantage Estimation
def compute_gae(rewards, values, masks, gamma=GAMMA, lam=LAMBDA):
    gae = 0
    returns = []
    values = values + [0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lam * masks[i] * gae
        returns.insert(0, gae + values[i])
    return returns

# PPO Main Training Loop
def train():
    env = OrbitEnv()
    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    all_rewards = []
    trajectory_data = []

    for epoch in range(EPOCHS):
        obs, _ = env.reset()
        state = normalize_state(obs)

        log_probs, values, states, actions, rewards, masks = [], [], [], [], [], []
        total_reward = 0
        traj = []

        for _ in range(2048):
            action, log_prob = model.get_action(state)
            next_obs, reward, done, info = env.step(action * THRUST_SCALE)
            norm_next_state = normalize_state(next_obs)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            masks.append(0 if done else 1)
            log_probs.append(log_prob)

            with torch.no_grad():
                _, value = model.forward(torch.tensor(state, dtype=torch.float32).to(device))
                values.append(value.item())

            traj.append(env.pos.copy())
            state = norm_next_state
            total_reward += reward

            if done:
                break

        all_rewards.append(total_reward)
        trajectory_data.append(np.array(traj))

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        old_log_probs = torch.stack(log_probs).detach().to(device)
        returns = compute_gae(rewards, values, masks)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # PPO update
        for _ in range(TRAIN_ITERS):
            new_log_probs, entropy, value = model.evaluate(states, actions)
            advantage = returns - value.detach()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - value).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{EPOCHS} | Reward: {total_reward:.2f}")

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Save reward curve
    plt.plot(all_rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Curve")
    plt.grid()
    plt.savefig("training_curve.png")
    plt.show()

    # Save last trajectory
    if trajectory_data:
        np.save("ppo_traj.npy", trajectory_data[-1])

if __name__ == "__main__":
    train()


