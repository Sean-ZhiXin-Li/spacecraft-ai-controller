import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from envs.orbit_env import OrbitEnv
import numpy as np
from simulator.visualize import plot_trajectory
import matplotlib.pyplot as plt

# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.95
LR = 1e-4
EPOCHS = 50
TRAIN_ITERS = 5

# Normalize state
def normalize_state(state):
    return np.array([
        state[0] / 1e13,
        state[1] / 1e13,
        state[2] / 3e4,
        state[3] / 3e4
    ], dtype=np.float32)

# Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # Learnable log-std

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

# GAE Advantage Computation
def compute_advantages(rewards, values, gamma=GAMMA, lam=LAMBDA):
    advantages = []
    gae = 0
    values = values + [0.0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

# PPO Loss
def ppo_clip_loss(old_logprobs, new_logprobs, advantages, clip=0.2):
    ratio = (new_logprobs - old_logprobs).exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
    return -torch.min(unclipped, clipped).mean()

# Collect a trajectory from one rollout
def collect_trajectory(env, model):
    state = env.reset()[0]
    done = False

    states, actions, rewards, values, logprobs, positions = [], [], [], [], [], []

    while not done:
        # state normalization if needed
        norm_state = normalize_state(state)
        state_tensor = torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action, logprob, value = model.act(state_tensor)

        next_state, reward, done, _ = env.step(action.squeeze().numpy())

        states.append(state_tensor.squeeze(0))
        actions.append(action.squeeze(0))
        logprobs.append(logprob)
        values.append(value)
        rewards.append(torch.tensor(reward, dtype=torch.float32))
        positions.append(env.pos.copy())

        state = next_state

    return states, actions, rewards, values, logprobs, positions
# Main Training Loop
def train():
    env = OrbitEnv()
    env = OrbitEnv(max_steps=10000)
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    reward_sums = []

    for epoch in range(EPOCHS):
        result = collect_trajectory(env, model)
        if len(result[0]) == 0:
            print(f" Epoch {epoch+1}: No valid trajectory collected. Skipping.")
            continue

        states, actions, rewards, values, logprobs, positions = result
        advantages = compute_advantages(rewards, [v.item() for v in values])
        returns = [a + v.item() for a, v in zip(advantages, values)]

        # Convert to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        logprobs = torch.stack(logprobs)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

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
        print(f"Epoch {epoch+1}: Return = {total_return:.2f}, Loss_pi = {loss_pi.item():.4f}, Loss_v = {loss_v.item():.4f}")

    # Save Results
    torch.save(model.state_dict(), "ppo_controller_gaussian.pth")
    plt.figure(figsize=(8, 5))
    plt.plot(reward_sums, marker='o', linestyle='-', color='royalblue')
    plt.xlabel("Epoch")
    plt.ylabel("Return")
    plt.title("PPO (Gaussian) Return Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_reward_curve_gaussian.png", dpi=300)
    plt.show()

    trajectory = np.array(positions)
    plot_trajectory(trajectory, title="Final PPO (Gaussian) Trajectory", target_radius=env.target_radius)

if __name__ == "__main__":
    train()
