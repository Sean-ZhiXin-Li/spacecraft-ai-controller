import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from envs.orbit_env import OrbitEnv
from controller.expert_controller import ExpertController

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO hyperparameters
GAMMA = 0.995
LAMBDA = 0.97
LR = 3e-5
EPOCHS = 800
TRAIN_ITERS = 20
THRUST_SCALE = 5000

# Normalize state
def normalize_state(state):
    pos_scale = 7.5e12
    vel_scale = 3e4
    return np.array([
        state[0] / pos_scale,
        state[1] / pos_scale,
        state[2] / vel_scale,
        state[3] / vel_scale
    ], dtype=np.float32)

# PPO Actor-Critic Network
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
        std = self.log_std.exp()
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return log_probs, entropy, value.squeeze(-1)

# GAE (Generalized Advantage Estimation)
def compute_gae(rewards, values, masks, gamma=GAMMA, lam=LAMBDA):
    gae = 0
    returns = []
    values = values + [0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lam * masks[i] * gae
        returns.insert(0, gae + values[i])
    return returns

# Expert Initialization
def load_expert_into_actor_critic(model, expert_controller, env, samples=10000):
    states, targets = [], []
    for _ in range(samples):
        obs, _ = env.reset()
        pos, vel = obs[:2], obs[2:]
        action = expert_controller(0, pos, vel)
        states.append(normalize_state(obs))
        targets.append(np.clip(action / THRUST_SCALE, -1.0, 1.0))
    states = torch.tensor(states, dtype=torch.float32).to(device)
    targets = torch.tensor(targets, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(500):
        mu, _ = model.forward(states)
        loss = nn.MSELoss()(mu, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("PPO actor initialized from ExpertController")

# PPO Main Training Loop
def train():
    env = OrbitEnv()
    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    expert = ExpertController(target_radius=7.5e12)
    load_expert_into_actor_critic(model, expert, env)

    all_rewards = []

    for epoch in range(EPOCHS):
        obs, _ = env.reset()
        state = normalize_state(obs)

        log_probs, values, states, actions, rewards, masks = [], [], [], [], [], []
        total_reward = 0

        for _ in range(2048):
            action, log_prob = model.get_action(state)
            next_obs, reward, done, _ = env.step(action * THRUST_SCALE)
            norm_next_state = normalize_state(next_obs)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            masks.append(0 if done else 1)
            log_probs.append(log_prob)

            with torch.no_grad():
                _, value = model.forward(torch.tensor(state, dtype=torch.float32).to(device))
                values.append(value.item())

            state = norm_next_state
            total_reward += reward
            if done:
                break

        all_rewards.append(total_reward)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Reward: {total_reward:.2f}")

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        old_log_probs = torch.stack(log_probs).detach().to(device)
        returns = compute_gae(rewards, values, masks)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # PPO updates
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

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Save reward curve
    plt.plot(all_rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Curve (Expert Init)")
    plt.grid()
    plt.savefig("training_curve_expert.png")
    plt.show()

if __name__ == "__main__":
    train()
