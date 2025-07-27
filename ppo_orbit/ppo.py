import torch
import torch.nn as nn
import torch.optim as optim
from envs.orbit_env import OrbitEnv
import numpy as np
from simulator.visualize import plot_trajectory
import matplotlib.pyplot as plt

# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.95
LR = 1e-4
EPOCHS = 10
TRAIN_ITERS = 5


def normalize_state(state):
    return np.array([
        state[0] / 1e13,  # x
        state[1] / 1e13,  # y
        state[2] / 3e4,  # vx
        state[3] / 3e4  # vy
    ], dtype=np.float32)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.1)
        nn.init.constant_(m.bias, 0)

# Actor-Critic Neural Network
class ActorCritic(nn.Module):

    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(4, 64),        # Input: [x, y, vx, vy]
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(64, 2)   # Output: thrust vector [Tx, Ty]
        self.value_head = nn.Linear(64, 1)    # Output: state value V(s)

    def forward(self, x):
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x)

# Compute Generalized Advantage Estimation (GAE)
def compute_advantages(rewards, values, gamma=GAMMA, lam=LAMBDA):
    advantages = []
    gae = 0
    values = values + [0.0]  # Append dummy value for easier computation
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

# PPO Clipped Surrogate Objective
def ppo_clip_loss(old_logprobs, new_logprobs, advantages, clip=0.2):
    ratio = (new_logprobs - old_logprobs).exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
    return -torch.min(unclipped, clipped).mean()

# Rollout a single trajectory using current policy
def collect_trajectory(env, model):
    state, _ = env.reset()  # Unpack observation from reset()
    states, actions, rewards, values, logprobs = [], [], [], [], []

    positions =[]

    done = False
    while not done:
        positions.append(state[:2])
        s = torch.tensor(normalize_state(state), dtype=torch.float32)
        logits, value = model(s)
        tanh_logits = torch.tanh(logits)
        action = tanh_logits.detach().numpy()
        action = np.clip(action, -1.0, 1.0)

        new_state, reward, done, _ = env.step(action)

        logprob = -((tanh_logits - torch.tensor(action)) ** 2).sum()

        states.append(s)
        actions.append(torch.tensor(action))
        rewards.append(reward)
        values.append(value.item())
        logprobs.append(logprob.detach())

        state = new_state

    return states, actions, rewards, values, logprobs, positions

# Main PPO Training Loop
def train():
    env = OrbitEnv()
    model = ActorCritic()
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    reward_sums = []

    for epoch in range(EPOCHS):
        # Rollout a trajectory
        states, actions, rewards, values, logprobs, positions = collect_trajectory(env, model)
        advantages = compute_advantages(rewards, values)
        returns = [a + v for a, v in zip(advantages, values)]

        # Convert to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_logprobs = torch.stack(logprobs)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # PPO Optimization
        for _ in range(TRAIN_ITERS):
            logits, values = model(states)
            new_logprobs = -((logits - actions) ** 2).sum(dim=1)

            loss_pi = ppo_clip_loss(old_logprobs, new_logprobs, advantages)
            loss_v = ((returns - values.squeeze()) ** 2).mean()

            loss = loss_pi + 0.5 * loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Return Sum = {sum(rewards):.2f}, Loss_pi = {loss_pi.item():.4f}, Loss_v = {loss_v.item():.4f}")
        reward_sums.append(sum(rewards))
    torch.save(model.state_dict(), "ppo_controller.pth")
    plt.figure(figsize=(8, 5))
    plt.plot(reward_sums, marker='o', linestyle='-', color='royalblue')
    plt.xlabel("Epoch")
    plt.ylabel("Return Sum")
    plt.title("PPO Training Return Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_reward_curve.png", dpi=300)
    plt.show()

    print("PPO model saved to ppo_controller.pth")
    # Visualize final trajectory
    trajectory = np.array(positions)
    plot_trajectory(trajectory, title="Final PPO Trajectory", target_radius=env.target_radius)
if __name__ == "__main__":
    train()

