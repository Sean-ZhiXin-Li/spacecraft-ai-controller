import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

from envs.orbit_env import OrbitEnv
from controller.expert_controller import ExpertController

# ==========================================================
# Device
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ==========================================================
# Hyperparameters (stability-oriented defaults)
# ==========================================================
GAMMA = 0.995                  # reward discount
LAMBDA = 0.97                  # GAE lambda
EPOCHS = 800                   # total PPO epochs
TRAIN_ITERS = 20               # PPO update iters per epoch (minibatch)
THRUST_SCALE = 5000.0          # maps [-1,1] action to physical thrust
BATCH_STEPS = 4096             # rollout steps per epoch

# PPO specifics
CLIP_EPS = 0.30                # PPO clip epsilon (can tune later)
VF_COEF = 0.5                  # value loss coefficient
ENT_COEF = 0.001               # entropy off initially; enable later if needed (0.001~0.002)
MAX_GRAD_NORM = 0.5            # gradient clipping
VAL_CLIP_RANGE = 0.2           # value function clipping range
MB_SIZE = 128                  # PPO minibatch size

# Optimizer learning rates (parameter groups)
LR_ACTOR = 5e-5
LR_CRITIC = 2e-4

# Logging / I/O
LOG_DIR = "ppo_orbit"
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "loss_log.csv")
FINAL_CKPT = os.path.join(LOG_DIR, f"ppo_epoch_{EPOCHS}.pth")

# === Your dataset path (set to your file) ===
DATASET_PATH = os.path.join("data", "data", "preprocessed", "merged_expert_dataset.npy")

# ==========================================================
# State normalization (scales position and velocity to ~[-1,1])
# ==========================================================
def normalize_state(state):
    pos_scale = 7.5e12
    vel_scale = 3e4
    return np.array([
        state[0] / pos_scale,
        state[1] / pos_scale,
        state[2] / vel_scale,
        state[3] / vel_scale
    ], dtype=np.float32)

# ==========================================================
# CSV logging helper
# ==========================================================
def log_to_csv(path, row_dict, header_order):
    """Append a row to CSV; create header if file is new."""
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if not exists:
            w.writeheader()
        w.writerow(row_dict)

# ==========================================================
# Evaluation with mean action (deterministic)
# ==========================================================
@torch.no_grad()
def evaluate_policy(env, model, episodes=2):
    """Run episodes using mean action; return average return."""
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        s = normalize_state(obs)
        done = False
        ep_ret = 0.0
        while not done:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            mu, _ = model.forward(st)           # mean action
            a = np.clip(mu.squeeze(0).cpu().numpy(), -1.0, 1.0)
            next_obs, r, done, _ = env.step(a * THRUST_SCALE)
            s = normalize_state(next_obs)
            ep_ret += r
        rewards.append(ep_ret)
    return float(np.mean(rewards))

# ==========================================================
# Actor-Critic with a shared trunk
# ==========================================================
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh()
        )
        # Actor head -> mean action (size 2)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )
        # Critic head -> state value
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Conservative initial std (~0.1) to stabilize early training
        self.log_std = nn.Parameter(torch.log(torch.ones(2) * 0.1))

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

    @torch.no_grad()
    def get_action(self, state):
        """
        Sample an action and return (action_np, log_prob_tensor).
        Sampling is used for exploration during rollout collection.
        """
        st = torch.tensor(state, dtype=torch.float32, device=device)
        mu, _ = self.forward(st)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        a = dist.sample()
        logp = dist.log_prob(a).sum()
        a_np = np.clip(a.cpu().numpy(), -1.0, 1.0)  # safety clamp before env.step
        return a_np, logp

    def evaluate(self, states, actions):
        """
        Compute log_probs, entropy, and value for given (state, action) batch.
        Used in PPO loss.
        """
        mu, value = self.forward(states)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, value.squeeze(-1)

# ==========================================================
# GAE with bootstrap using the last state's value
# ==========================================================
def compute_gae(rewards, values, masks, gamma=GAMMA, lam=LAMBDA, last_value=0.0):
    """
    Args:
        rewards: list[float], r_t
        values:  list[float], V(s_t) for t=0..T-1
        masks:   list[int], 1 if not done else 0
        last_value: float, V(s_T) for bootstrap
    Returns:
        returns: list[float], target returns for critic
    """
    masks = [float(m) for m in masks]  # ensure float math
    gae = 0.0
    returns = []
    values_t = values + [last_value]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values_t[i + 1] * masks[i] - values_t[i]
        gae = delta + gamma * lam * masks[i] * gae
        returns.insert(0, gae + values_t[i])
    return returns

# ==========================================================
# Expert warm start from .npy (streamed mini-batch -> avoids CUDA OOM)
# ==========================================================
def load_expert_from_npy(model, npy_path, epochs=1, batch_size=4096, shuffle=True,
                         max_samples=200_000, progress_every=20):
    """
    Memory-safe expert pretraining from .npy with visible progress:
      - Supports dict or [N,6+] array
      - Optional uniform subsampling (max_samples) to avoid long warmup
      - Streams mini-batches to GPU; prints progress every few batches
    """
    import torch.utils.data as tud
    import time

    print(f"[Init] Loading dataset from {npy_path} ...", flush=True)
    if not os.path.exists(npy_path):
        print(f"[Warn] Dataset not found: {npy_path}. Fallback to online expert.", flush=True)
        return False

    # ---- Quick peek (structure & shape) ----
    try:
        data = np.load(npy_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            d = data.item()
            print(f"[Init] Detected dict with keys: {list(d.keys())}", flush=True)
            k0 = next(iter(d))
            print(f"[Init] Example value shape for '{k0}': {np.array(d[k0]).shape}", flush=True)
        else:
            print(f"[Init] Detected array with shape: {np.array(data).shape}", flush=True)
    except Exception as e:
        print(f"[Init] Peek failed: {e}", flush=True)
        data = np.load(npy_path, allow_pickle=True)  # fallback

    # ---- Parse states/actions ----
    states_np, actions_np = None, None
    if isinstance(data, np.ndarray) and data.dtype == object:
        d = data.item()
        keys = {k.lower(): k for k in d.keys()}
        s_key = keys.get("states") or keys.get("state") or keys.get("obs") or keys.get("x")
        a_key = keys.get("actions") or keys.get("action") or keys.get("act") or keys.get("y")
        if s_key is None or a_key is None:
            print("[Error] Dict dataset must contain states/obs and actions/act keys.", flush=True)
            return False
        states_np = np.asarray(d[s_key], dtype=np.float32)
        actions_np = np.asarray(d[a_key], dtype=np.float32)
    else:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 6:
            print(f"[Error] Unsupported array shape {arr.shape}. Expect [N,6+] with first4=state last2=action.", flush=True)
            return False
        states_np = arr[:, :4]
        actions_np = arr[:, -2:]

    N = len(states_np)
    print(f"[Init] Raw dataset size: {N}", flush=True)

    # ---- Optional subsample for fast warmstart ----
    if (max_samples is not None) and (N > max_samples):
        np.random.seed(0)  # reproducible subsampling
        idx = np.random.choice(N, size=max_samples, replace=False)
        states_np = states_np[idx]
        actions_np = actions_np[idx]
        N = max_samples
        print(f"[Init] Subsampled to {N} samples for faster warmstart.", flush=True)

    # ---- Normalize states if needed ----
    if np.max(np.abs(states_np[:, :2])) > 10 or np.max(np.abs(states_np[:, 2:])) > 1.0:
        states_np = np.stack([normalize_state(s) for s in states_np], axis=0).astype(np.float32)
    else:
        states_np = states_np.astype(np.float32)

    # ---- Scale actions to [-1,1] if not already ----
    if np.max(np.abs(actions_np)) > 1.5:
        actions_np = (actions_np / THRUST_SCALE).astype(np.float32)
    actions_np = np.clip(actions_np, -1.0, 1.0).astype(np.float32)

    # ---- DataLoader on CPU; move batch-by-batch to GPU ----
    ds = tud.TensorDataset(torch.from_numpy(states_np), torch.from_numpy(actions_np))
    dl = tud.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)

    # Train only Actor (shared + actor + log_std)
    actor_params = list(model.shared.parameters()) + list(model.actor.parameters()) + [model.log_std]
    opt = torch.optim.Adam(actor_params, lr=1e-4)
    mse = nn.MSELoss()

    print(f"[Init] Start expert pretraining: epochs={epochs}, batch_size={batch_size}", flush=True)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        running = 0.0
        for step, (s_cpu, a_cpu) in enumerate(dl, 1):
            s = s_cpu.to(device, non_blocking=True).float()
            a_tgt = a_cpu.to(device, non_blocking=True).float()
            mu, _ = model.forward(s)
            loss = mse(mu, a_tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor_params, max_norm=1.0)
            opt.step()

            running += loss.item() * s.size(0)
            if step % progress_every == 0:
                print(f"[Init][ep {ep}] step {step}  MSE={loss.item():.6f}", flush=True)

        epoch_loss = running / N
        print(f"[Init] Expert pretrain epoch {ep}/{epochs} | MSE: {epoch_loss:.6f}", flush=True)

    print(f"[Init] Done expert pretraining in {int(time.time()-t0)}s.", flush=True)
    with torch.no_grad():
        w = model.actor[0].weight[:2, :6].detach().cpu().numpy()
        b = model.actor[0].bias[:2].detach().cpu().numpy()
        print("[Sanity] Actor first layer W[:2,:6]:", np.round(w, 4), flush=True)
        print("[Sanity] Actor first layer b[:2]:", np.round(b, 4), flush=True)
    return True

# ==========================================================
# Fallback online expert warm start (if dataset missing)
# ==========================================================
def load_expert_online(model, expert_controller, env, samples=10000):
    states, targets = [], []
    for _ in range(samples):
        obs, _ = env.reset()
        pos, vel = obs[:2], obs[2:]
        action = expert_controller(0, pos, vel)
        states.append(normalize_state(obs))
        targets.append(np.clip(action / THRUST_SCALE, -1.0, 1.0))

    states  = torch.tensor(np.array(states,  dtype=np.float32), device=device)
    targets = torch.tensor(np.array(targets, dtype=np.float32), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()
    for _ in range(500):
        mu, _ = model.forward(states)
        loss = mse(mu, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("[Init] Actor initialized from ExpertController (online).")
    w = model.actor[0].weight[:2, :6].detach().cpu().numpy()
    b = model.actor[0].bias[:2].detach().cpu().numpy()
    print("[Sanity] Actor first layer W[:2,:6]:", np.round(w, 4))
    print("[Sanity] Actor first layer b[:2]:", np.round(b, 4))

# ==========================================================
# Quick expert-match report
# ==========================================================
@torch.no_grad()
def expert_match_report(env, model, expert, n=256):
    """
    Compare Actor mean vs Expert action on n reset states.
    Lower MSE & higher cosine mean better imitation.
    """
    from numpy.linalg import norm
    expert_a, actor_mu = [], []
    for _ in range(n):
        obs, _ = env.reset()
        s = normalize_state(obs)
        pos, vel = obs[:2], obs[2:]
        a_exp = expert(0, pos, vel) / THRUST_SCALE
        s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        mu, _ = model.forward(s_t)
        a_mu = mu.squeeze(0).cpu().numpy()
        expert_a.append(np.clip(a_exp, -1, 1))
        actor_mu.append(a_mu)
    expert_a = np.array(expert_a, dtype=np.float32)
    actor_mu = np.array(actor_mu, dtype=np.float32)
    mse = float(np.mean((expert_a - actor_mu) ** 2))
    cos = float(np.mean([
        np.dot(expert_a[i], actor_mu[i]) / ((norm(expert_a[i])+1e-8) * (norm(actor_mu[i])+1e-8))
        for i in range(n)
    ]))
    print(f"[Expert-Match] MSE={mse:.6f}  Cosine={cos:.4f}  (n={n})")

# ==========================================================
# Plot helpers (reward & loss curves)
# ==========================================================
def plot_curves(reward_hist, csv_path, out_dir):
    # Reward curve
    plt.figure()
    plt.plot(reward_hist)
    plt.xlabel("Epoch"); plt.ylabel("Total Reward")
    plt.title("PPO Training Curve (Expert Init)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "reward_curve.png"), dpi=150)
    plt.close()

    # Loss curves from CSV
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        plt.figure()
        plt.plot(df["epoch"], df["actor_loss"], label="Actor loss")
        plt.plot(df["epoch"], df["critic_loss"], label="Critic loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Loss vs Epoch"); plt.grid(True); plt.legend()
        plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
        plt.close()
    except Exception as e:
        print("Plot loss curves skipped:", e)

# ==========================================================
# Orbit visualization (after training)
# ==========================================================
@torch.no_grad()
def evaluate_and_plot_orbit(env, model, out_path=os.path.join(LOG_DIR, "orbit_trajectory.png")):
    """
    Runs one episode with mean action and plots the XY position trajectory.
    Saves to out_path and prints that episode return.
    """
    obs, _ = env.reset()
    s = normalize_state(obs)
    done = False
    ep_ret = 0.0
    xs, ys = [], []
    while not done:
        xs.append(obs[0]); ys.append(obs[1])
        st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        mu, _ = model.forward(st)
        a = np.clip(mu.squeeze(0).cpu().numpy(), -1.0, 1.0)
        obs, r, done, _ = env.step(a * THRUST_SCALE)
        s = normalize_state(obs); ep_ret += r
    plt.figure()
    plt.plot(np.array(xs), np.array(ys))
    plt.scatter([xs[0]],[ys[0]], marker="o", label="start")
    plt.scatter([xs[-1]],[ys[-1]], marker="x", label="end")
    plt.xlabel("x position"); plt.ylabel("y position")
    plt.title("Orbit Trajectory (mean action)")
    plt.legend(); plt.grid(True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[Evaluate] One-episode return (mean action): {ep_ret:.2f}")
    print(f"[Plot] Saved orbit trajectory to {out_path}")

# ==========================================================
# Train loop (minibatch PPO, value clipping, diagnostics, logging)
# ==========================================================
def train():
    global TRAIN_ITERS  # allow adaptive tuning of PPO update iters

    env = OrbitEnv()
    model = ActorCritic().to(device)

    # One optimizer with parameter groups:
    #   group 0: shared trunk + actor head + log_std (LR_ACTOR)
    #   group 1: critic head (LR_CRITIC)
    optimizer = optim.Adam([
        {"params": list(model.shared.parameters()) + list(model.actor.parameters()) + [model.log_std], "lr": LR_ACTOR},
        {"params": list(model.critic.parameters()), "lr": LR_CRITIC},
    ])

    # ===== Expert warm start: dataset first, fallback to online expert =====
    print(f"[Init] Using dataset at {DATASET_PATH}", flush=True)
    expert = ExpertController(target_radius=7.5e12)

    # More saturated warm start: 2 epochs, 1,000,000 samples per epoch
    ok = load_expert_from_npy(
        model,
        DATASET_PATH,
        epochs=2,             # was 1 -> 2
        batch_size=4096,      # safe for ~12GB GPUs; tune if needed
        max_samples=1_000_000,# was 200_000 -> 1,000,000 per epoch
        progress_every=20
    )
    if not ok:
        # Fallback to online expert imitation if dataset missing or malformed
        load_expert_online(model, expert, env)

    # Quick imitation report (Actor mean vs Expert on reset states)
    expert_match_report(env, model, expert, n=256)

    # Pre-evaluation with mean action
    pre_eval = evaluate_policy(env, model, episodes=2)
    print(f"[Pre-Eval] Avg return: {pre_eval:.2f}")

    all_rewards = []
    header = ["epoch", "reward", "actor_loss", "critic_loss"]

    for epoch in range(1, EPOCHS + 1):
        obs, _ = env.reset()
        state = normalize_state(obs)

        log_probs, values, states, actions, rewards, masks = [], [], [], [], [], []
        total_reward = 0.0

        # -------- Rollout collection --------
        for _ in range(BATCH_STEPS):
            action, log_prob = model.get_action(state)
            next_obs, reward, done, _ = env.step(action * THRUST_SCALE)
            ns = normalize_state(next_obs)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            masks.append(0 if done else 1)
            log_probs.append(log_prob.detach().to(device))

            with torch.no_grad():
                _, v = model.forward(torch.tensor(state, dtype=torch.float32, device=device))
                values.append(float(v.item()))

            state = ns
            total_reward += reward
            if done:
                break

        all_rewards.append(total_reward)
        print(f"Epoch {epoch}/{EPOCHS} | Reward: {total_reward:.2f}")

        # -------- Bootstrap value for the last state --------
        with torch.no_grad():
            _, last_v_t = model.forward(torch.tensor(state, dtype=torch.float32, device=device))
            last_v = float(last_v_t.item())

        # -------- Tensorize batch --------
        states_t   = torch.tensor(np.array(states),  dtype=torch.float32, device=device)
        actions_t  = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        old_logp_t = torch.stack(log_probs).detach()
        returns_np = compute_gae(rewards, values, masks, last_value=last_v)
        returns_t  = torch.tensor(returns_np, dtype=torch.float32, device=device)

        # -------- Advantage (normalize) --------
        with torch.no_grad():
            _, v_pred = model.forward(states_t)
            v_pred = v_pred.squeeze(-1)
            adv = returns_t - v_pred
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            v_old = v_pred.clone()  # cache for value clipping

        # -------- PPO updates (minibatch + value clipping + KL/clip_frac) --------
        idx_all = np.arange(len(states_t))
        last_actor_loss, last_critic_loss = 0.0, 0.0
        clip_fracs, kls = [], []

        for _ in range(TRAIN_ITERS):
            np.random.shuffle(idx_all)
            for start in range(0, len(idx_all), MB_SIZE):
                mb_idx = idx_all[start:start + MB_SIZE]
                mb = torch.tensor(mb_idx, dtype=torch.long, device=device)

                mb_states   = states_t[mb]
                mb_actions  = actions_t[mb]
                mb_adv      = adv[mb]
                mb_returns  = returns_t[mb]
                mb_old_logp = old_logp_t[mb]
                mb_v_old    = v_old[mb]

                new_logp, entropy, v_pred_mb = model.evaluate(mb_states, mb_actions)
                ratio  = torch.exp(new_logp - mb_old_logp)
                surr1  = ratio * mb_adv
                surr2  = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                actor_loss  = -torch.min(surr1, surr2).mean()

                # Value clipping for critic
                v_clip = mb_v_old + (v_pred_mb - mb_v_old).clamp(-VAL_CLIP_RANGE, VAL_CLIP_RANGE)
                critic_loss = 0.5 * torch.mean((mb_returns - v_clip) ** 2)

                # Total loss
                total_loss = actor_loss + critic_loss - ENT_COEF * entropy.mean()

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                # Keep policy std within a reasonable range
                with torch.no_grad():
                    model.log_std.data.clamp_(min=np.log(0.05), max=np.log(1.0))
                    approx_kl = (mb_old_logp - new_logp).mean().item()
                    clip_frac = (torch.abs(ratio - 1.0) > CLIP_EPS).float().mean().item()
                kls.append(approx_kl)
                clip_fracs.append(clip_frac)

                last_actor_loss = float(actor_loss.item())
                last_critic_loss = float(critic_loss.item())

        # -------- Diagnostics --------
        if epoch % 10 == 0:
            print(f"[Diag] clip_frac={np.mean(clip_fracs):.3f}  KL={np.mean(kls):.4f}")

        # -------- NEW: Adaptive PPO iters based on KL (per-epoch) --------
        # If KL is too small: updates are too conservative -> increase TRAIN_ITERS slightly (up to 25)
        # If KL is too large: updates are too aggressive   -> decrease TRAIN_ITERS slightly (down to 10)
        target_kl = 0.015
        mean_kl = float(np.mean(kls)) if len(kls) else 0.0
        if mean_kl < 0.5 * target_kl:
            TRAIN_ITERS = min(TRAIN_ITERS + 1, 25)
        elif mean_kl > 1.5 * target_kl:
            TRAIN_ITERS = max(TRAIN_ITERS - 1, 10)
        print(f"[Adapt] mean_KL={mean_kl:.4f} -> next TRAIN_ITERS={TRAIN_ITERS}")

        # -------- CSV logging --------
        log_to_csv(CSV_PATH, {
            "epoch": epoch,
            "reward": total_reward,
            "actor_loss": last_actor_loss,
            "critic_loss": last_critic_loss
        }, header)

        # -------- Periodic checkpoint (every 100 epochs) --------
        if epoch % 100 == 0:
            ckpt_path = os.path.join(LOG_DIR, f"ppo_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": {
                    "GAMMA": GAMMA, "LAMBDA": LAMBDA,
                    "CLIP_EPS": CLIP_EPS, "VF_COEF": VF_COEF,
                    "ENT_COEF": ENT_COEF, "THRUST_SCALE": THRUST_SCALE
                }
            }, ckpt_path)
            print(f"[Checkpoint] Saved to {ckpt_path}")

        # -------- Plot every 10 epochs --------
        if epoch % 10 == 0:
            plot_curves(all_rewards, CSV_PATH, LOG_DIR)

    # -------- Final checkpoint --------
    torch.save({
        "epoch": EPOCHS,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": {
            "GAMMA": GAMMA, "LAMBDA": LAMBDA,
            "CLIP_EPS": CLIP_EPS, "VF_COEF": VF_COEF,
            "ENT_COEF": ENT_COEF, "THRUST_SCALE": THRUST_SCALE
        }
    }, FINAL_CKPT)
    print(f"[Checkpoint] Saved final model to {FINAL_CKPT}")

    # -------- Final plots & orbit visualization --------
    plot_curves(all_rewards, CSV_PATH, LOG_DIR)
    evaluate_and_plot_orbit(env, model, out_path=os.path.join(LOG_DIR, "orbit_trajectory.png"))
    print("Training finished.")


if __name__ == "__main__":
    train()
