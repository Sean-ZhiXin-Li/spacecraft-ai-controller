import torch

def compute_advantages(rewards, values,gamma = 0.99, lam = 0.95):
    advantages = []
    gae = 0
    values = values +[0.0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma *lam * gae
        advantages.insert(0, gae)
    return advantages

def ppo_clip_loss(old_logprobs, new_logprobs, advantages, clip = 0.2):
    ratio = (new_logprobs - old_logprobs).exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
    return -torch.min(unclipped, clipped).mean()
