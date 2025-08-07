import os
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from envs.orbit_env import OrbitEnv
import argparse
from hybrid_init import load_mimic_pth_into_actor_critic

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO hyperparameters
GAMMA = 0.995
LAMBDA = 0.97
LR = 3e-5
EPOCHS = 800
TRAIN_ITERS = 20
THRUST_SCALE = 5000

# Normalize the raw state (position + velocity)
def normalize_state(state):
    pos_scale = 7.5e12  # scale for position
    vel_scale = 3e4     # scale for velocity
    return np.array([
        state[0] / pos_scale,
        state[1] / pos_scale,
        state[2] / vel_scale,
        state[3] / vel_scale
    ], dtype=np.float32)

# Actor-Critic network with Gaussian policy
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
        self.log_std = nn.Parameter(torch.zeros(2))  # Learnable log_std

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

# Evaluate current policy by running one rollout and logging reward components
def evaluate(env, model):
    state, _ = env.reset()
    total_reward = 0
    shaping_sum, bonus_sum, penalty_sum = 0, 0, 0
    r_error_sum, v_error_sum = 0, 0
    steps = 0

    while True:
        state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32).to(device)
        with torch.no_grad():
            mu, _ = model.forward(state_tensor)
        std = model.log_std.exp()
        dist = Normal(mu, std)
        action = dist.sample().cpu().numpy()

        next_state, reward, done, info = env.step(action * THRUST_SCALE)

        # Accumulate reward components
        total_reward += reward
        shaping_sum += info.get("shaping", 0)
        bonus_sum += info.get("bonus", 0)
        penalty_sum += info.get("penalty", 0)
        r_error_sum += info.get("r_error", 0)
        v_error_sum += info.get("v_error", 0)

        state = next_state
        steps += 1
        if done:
            break

    return total_reward, shaping_sum / steps, bonus_sum / steps, penalty_sum / steps, r_error_sum / steps, v_error_sum / steps

# Compute GAE returns for advantage estimation
def compute_gae(rewards, values, masks, gamma=GAMMA, lam=LAMBDA):
    gae = 0
    returns = []
    values = values + [0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lam * masks[i] * gae
        returns.insert(0, gae + values[i])
    return returns

# Main training loop
def train(args):
    env = OrbitEnv()
    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Optional hybrid warm-start
    if args.mode == 'hybrid' and args.init_mimic_path is not None:
        load_mimic_pth_into_actor_critic(model, args.init_mimic_path)
        print(f"Loaded mimic model from {args.init_mimic_path}")

    all_rewards = []
    trajectory_data = []

    for epoch in range(EPOCHS):
        obs, _ = env.reset(start_mode=args.start_mode)
        state = normalize_state(obs)

        log_probs, values, states, actions, rewards, masks = [], [], [], [], [], []
        total_reward = 0
        traj = []

        for _ in range(2048):
            action, log_prob = model.get_action(state)
            next_obs, reward, done, info = env.step(action * THRUST_SCALE)
            norm_next_state = normalize_state(next_obs)

            # Collect experience
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

        print(f"Epoch {epoch + 1}/{EPOCHS} | Reward: {total_reward:.2f}")

        # Log reward breakdown every epoch (if hybrid mode)
        if args.mode == 'hybrid':
            eval_reward, shaping, bonus, penalty, r_err, v_err = evaluate(env, model)
            os.makedirs("logs", exist_ok=True)
            with open("logs/reward_breakdown.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, eval_reward, shaping, bonus, penalty, r_err, v_err])

        # Convert buffers to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        old_log_probs = torch.stack(log_probs).detach().to(device)
        returns = compute_gae(rewards, values, masks)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # PPO update (train actor and critic)
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

        # Free GPU memory
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Plot reward curve
    plt.plot(all_rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Curve")
    plt.grid()
    plt.savefig("training_curve.png")
    plt.show()

    # Save final trajectory
    if trajectory_data:
        np.save("ppo_traj.npy", trajectory_data[-1])

# Entry point for script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='vanilla', help='vanilla | hybrid')
    parser.add_argument('--init_mimic_path', type=str, default=None, help='Path to mimic_model_V6_1.pth')
    parser.add_argument('--start_mode', type=str, default='spiral', help='spiral | circular')
    args = parser.parse_args()

    train(args)

