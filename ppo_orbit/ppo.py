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
# Switches (optional knobs)
# ==========================================================
USE_WIDE_NET   = True   # Wider hidden sizes for actor/critic
HIGH_VF_COEF   = True   # If True -> VF_COEF=1.0 else 0.5
HIGH_LR_CRITIC = True   # If True -> LR_CRITIC=5e-4 else 2e-4
WIDE_CRITIC = True

# ==========================================================
# Hyperparameters (stability-oriented defaults)
# ==========================================================
GAMMA       = 0.995
LAMBDA      = 0.97
EPOCHS      = 800
TRAIN_ITERS = 20
THRUST_SCALE = 3000.0          # << per your request

BATCH_STEPS = 4096

# PPO specifics
CLIP_EPS   = 0.25              # slightly larger step size
VF_COEF    = 1.0
ENT_COEF_0 = 0.005             # early exploration
ENT_COEF_1 = 0.001             # later reduced entropy
ENT_SWITCH_EPOCH = 250         # epoch to switch ENT_COEF
MAX_GRAD_NORM  = 0.5
VAL_CLIP_RANGE = 1.0
MB_SIZE   = 128

# Optimizer learning rates (parameter groups)
LR_ACTOR  = 5e-5
LR_CRITIC = 5e-4

# Logging / I/O
LOG_DIR = "ppo_orbit"
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH   = os.path.join(LOG_DIR, "loss_log.csv")
FINAL_CKPT = os.path.join(LOG_DIR, f"ppo_epoch_{EPOCHS}.pth")

# Dataset path
DATASET_PATH = os.path.join("data", "data", "preprocessed", "merged_expert_dataset.npy")

# ==========================================================
# Helpers
# ==========================================================
def normalize_state(state):
    """Scale position/velocity to ~[-1, 1]."""
    pos_scale = 7.5e12
    vel_scale = 3e4
    return np.array([
        state[0] / pos_scale,
        state[1] / pos_scale,
        state[2] / vel_scale,
        state[3] / vel_scale
    ], dtype=np.float32)

def log_to_csv(path, row_dict, header_order):
    """Append a row to CSV; create header if file is new."""
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if not exists:
            w.writeheader()
        w.writerow(row_dict)

def current_ent_coef(epoch:int)->float:
    """Piecewise entropy schedule."""
    return ENT_COEF_0 if epoch <= ENT_SWITCH_EPOCH else ENT_COEF_1

# ==========================================================
# Evaluation (deterministic & stochastic)
# ==========================================================
@torch.no_grad()
def evaluate_policy(env, model, episodes=2, max_steps=20000):
    totals, lens = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        s = normalize_state(obs)
        done = False
        ep_ret, steps = 0.0, 0
        while not done and steps < max_steps:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            mu, _ = model.forward(st)
            a_env = np.clip(mu.squeeze(0).cpu().numpy(), -1.0, 1.0)
            next_obs, r, done, _ = env.step(a_env * THRUST_SCALE)
            s = normalize_state(next_obs)
            ep_ret += r
            steps += 1
        totals.append(ep_ret); lens.append(steps)
    mean_ret = float(np.mean(totals))
    mean_len = float(np.mean(lens))
    print(f"[Eval] mean_return={mean_ret:.2f} | mean_len={mean_len:.0f} | per_step={mean_ret/max(1,mean_len):.6f}")
    return mean_ret

@torch.no_grad()
def evaluate_stochastic(env, model, episodes=2, max_steps=20000):
    totals = []
    for _ in range(episodes):
        obs, _ = env.reset()
        s = normalize_state(obs)
        done, ep_ret, steps = False, 0.0, 0
        while not done and steps < max_steps:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            mu, _ = model.forward(st)
            std = model.log_std.exp()
            a_raw = Normal(mu, std).sample().squeeze(0).cpu().numpy()
            a_env = np.clip(a_raw, -1.0, 1.0)
            obs, r, done, _ = env.step(a_env * THRUST_SCALE)
            s = normalize_state(obs); ep_ret += r; steps += 1
        totals.append(ep_ret)
    print(f"[Eval/Stoch] mean_return={float(np.mean(totals)):.2f}")

# ==========================================================
# Actor-Critic
# ==========================================================
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        h1 = 256 if USE_WIDE_NET else 128
        h2 = 128 if USE_WIDE_NET else 64

        self.shared = nn.Sequential(
            nn.Linear(4, h1),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, 1)
        )
        # Stronger initial exploration
        self.log_std = nn.Parameter(torch.log(torch.ones(2, device=device) * 0.2))

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

    @torch.no_grad()
    def get_action(self, state):
        """Return (raw_sample_np, clamped_env_np, log_prob_tensor)."""
        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        mu, _ = self.forward(st)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        a_raw = dist.rsample()
        logp  = dist.log_prob(a_raw).sum(dim=-1)
        a_env = torch.clamp(a_raw, -1.0, 1.0)
        return (a_raw.squeeze(0).cpu().numpy(),
                a_env.squeeze(0).cpu().numpy(),
                logp.squeeze(0))

    def evaluate(self, states, actions):
        """Log-probs/entropy/value for PPO on the *raw* actions sampled at rollout."""
        mu, value = self.forward(states)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, value.squeeze(-1)

# ==========================================================
# GAE
# ==========================================================
def compute_gae(rewards, values, masks, gamma=GAMMA, lam=LAMBDA, last_value=0.0):
    masks = [float(m) for m in masks]
    gae = 0.0
    returns = []
    values_t = values + [last_value]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values_t[i + 1] * masks[i] - values_t[i]
        gae = delta + gamma * lam * masks[i] * gae
        returns.insert(0, gae + values_t[i])
    return returns

# ==========================================================
# Expert warm start (from .npy)
# ==========================================================
def load_expert_from_npy(model, npy_path, epochs=2, batch_size=4096,
                         shuffle=True, max_samples=1_000_000, progress_every=20):
    import torch.utils.data as tud
    import time

    print(f"[Init] Loading dataset from {npy_path} ...", flush=True)
    if not os.path.exists(npy_path):
        print(f"[Warn] Dataset not found: {npy_path}. Fallback to online expert.", flush=True)
        return False

    try:
        data = np.load(npy_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            d = data.item()
            print(f"[Init] Detected dict with keys: {list(d.keys())}", flush=True)
        else:
            print(f"[Init] Detected array with shape: {np.array(data).shape}", flush=True)
    except Exception as e:
        print(f"[Init] Peek failed: {e}", flush=True)
        data = np.load(npy_path, allow_pickle=True)

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
        states_np  = arr[:, :4]
        actions_np = arr[:, -2:]

    N = len(states_np)
    print(f"[Init] Raw dataset size: {N}", flush=True)

    if (max_samples is not None) and (N > max_samples):
        np.random.seed(0)
        idx = np.random.choice(N, size=max_samples, replace=False)
        states_np  = states_np[idx]
        actions_np = actions_np[idx]
        N = max_samples
        print(f"[Init] Subsampled to {N} samples for faster warmstart.", flush=True)

    # normalize states and scale actions
    states_np = np.stack([normalize_state(s) for s in states_np], axis=0).astype(np.float32)
    if np.max(np.abs(actions_np)) > 1.5:
        actions_np = (actions_np / THRUST_SCALE).astype(np.float32)
    actions_np = np.clip(actions_np, -1.0, 1.0).astype(np.float32)

    ds = tud.TensorDataset(torch.from_numpy(states_np), torch.from_numpy(actions_np))
    dl = tud.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)

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
# Online expert fallback
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
        opt.zero_grad(); loss.backward(); opt.step()
    print("[Init] Actor initialized from ExpertController (online).")

# ==========================================================
# Diagnostics / plots
# ==========================================================
def plot_curves(reward_hist, csv_path, out_dir):
    # reward curve
    plt.figure()
    plt.plot(reward_hist)
    plt.xlabel("Epoch"); plt.ylabel("Total Reward")
    plt.title("PPO Training Curve (Expert Init)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "reward_curve.png"), dpi=150)
    plt.close()

    # loss curves
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

@torch.no_grad()
def evaluate_and_plot_orbit(env, model, out_path=os.path.join(LOG_DIR, "orbit_trajectory.png"), max_steps=20000):
    """Run one episode with mean action and plot XY trajectory."""
    obs, _ = env.reset()
    s = normalize_state(obs)
    done = False
    ep_ret = 0.0
    xs, ys = [], []
    steps = 0
    while not done and steps < max_steps:
        xs.append(obs[0]); ys.append(obs[1])
        st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        mu, _ = model.forward(st)
        a_env = np.clip(mu.squeeze(0).cpu().numpy(), -1.0, 1.0)
        obs, r, done, _ = env.step(a_env * THRUST_SCALE)
        s = normalize_state(obs); ep_ret += r; steps += 1
    plt.figure()
    plt.plot(np.array(xs), np.array(ys))
    plt.scatter([xs[0]],[ys[0]], marker="o", label="start")
    plt.scatter([xs[-1]],[ys[-1]], marker="x", label="end")
    plt.xlabel("x position (m)"); plt.ylabel("y position (m)")
    plt.title("Orbit Trajectory (mean action)")
    plt.legend(); plt.grid(True)
    plt.savefig(out_path, dpi=160); plt.close()
    print(f"[Evaluate] One-episode return (mean action): {ep_ret:.2f} | steps={steps}")
    print(f"[Plot] Saved orbit trajectory to {out_path}")

@torch.no_grad()
def plot_state_timeseries(env, model, out_path=os.path.join(LOG_DIR, "state_timeseries.png"), max_steps=20000):
    """Plot r(t), v(t), |cos(angle)| for a single evaluation rollout."""
    obs, _ = env.reset()
    s = normalize_state(obs)
    done = False
    rs, vs, coss, ts = [], [], [], []
    t = 0.0
    while not done and len(ts) < max_steps:
        pos = obs[:2]; vel = obs[2:]
        r = float(np.linalg.norm(pos))
        v = float(np.linalg.norm(vel))
        ur = pos / (r + 1e-8); uv = vel / (v + 1e-8)
        c = float(abs(np.dot(ur, uv)))
        rs.append(r); vs.append(v); coss.append(c); ts.append(t)

        st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        mu, _ = model.forward(st)
        a_env = np.clip(mu.squeeze(0).cpu().numpy(), -1.0, 1.0)
        obs, _, done, _ = env.step(a_env * THRUST_SCALE)
        s = normalize_state(obs); t += env.dt

    # Make three separate plots (clear and simple)
    # r(t)
    plt.figure()
    plt.plot(ts, rs)
    plt.xlabel("time (s)"); plt.ylabel("radius r (m)")
    plt.title("Radius over time")
    plt.grid(True)
    plt.savefig(out_path.replace(".png", "_r.png"), dpi=150)
    plt.close()

    # v(t)
    plt.figure()
    plt.plot(ts, vs)
    plt.xlabel("time (s)"); plt.ylabel("speed v (m/s)")
    plt.title("Speed over time")
    plt.grid(True)
    plt.savefig(out_path.replace(".png", "_v.png"), dpi=150)
    plt.close()

    # |cos(angle)| (angle between r and v)
    plt.figure()
    plt.plot(ts, coss)
    plt.xlabel("time (s)"); plt.ylabel("|cos(angle)|")
    plt.title("Alignment over time")
    plt.grid(True)
    plt.savefig(out_path.replace(".png", "_cos.png"), dpi=150)
    plt.close()
    print(f"[Plot] Saved state timeseries to {out_path.replace('.png','_{r,v,cos}.png')}")

def explained_variance(y_true_t, y_pred_t):
    y_true = y_true_t.detach().cpu().numpy()
    y_pred = y_pred_t.detach().cpu().numpy()
    var_y = np.var(y_true)
    return float(1 - np.var(y_true - y_pred) / (var_y + 1e-8))

# ==========================================================
# Train
# ==========================================================
def train():
    global TRAIN_ITERS

    env = OrbitEnv()
    model = ActorCritic().to(device)

    optimizer = optim.Adam([
        {"params": list(model.shared.parameters()) + list(model.actor.parameters()) + [model.log_std], "lr": LR_ACTOR},
        {"params": list(model.critic.parameters()), "lr": LR_CRITIC},
    ])

    # Warm start
    print(f"[Init] Using dataset at {DATASET_PATH}", flush=True)
    expert = ExpertController(target_radius=7.5e12)
    ok = load_expert_from_npy(
        model, DATASET_PATH, epochs=2, batch_size=4096,
        max_samples=1_000_000, progress_every=20
    )
    if not ok:
        load_expert_online(model, expert, env)

    # Quick imitation check
    evaluate_policy(env, model, episodes=1)

    all_rewards = []
    header = ["epoch", "reward", "actor_loss", "critic_loss"]

    for epoch in range(1, EPOCHS + 1):
        obs, _ = env.reset()
        state = normalize_state(obs)

        log_probs, values, states, actions, rewards, masks = [], [], [], [], [], []
        total_reward = 0.0
        ep_steps = 0

        # Rollout
        for _ in range(BATCH_STEPS):
            a_raw_np, a_env_np, log_prob = model.get_action(state)
            next_obs, reward, done, _ = env.step(a_env_np * THRUST_SCALE)
            ns = normalize_state(next_obs)

            states.append(state)
            actions.append(a_raw_np)   # raw actions for PPO math
            rewards.append(reward)
            masks.append(0 if done else 1)
            log_probs.append(log_prob.detach())

            with torch.no_grad():
                _, v = model.forward(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                values.append(float(v.item()))

            state = ns
            total_reward += reward
            ep_steps += 1
            if done:
                break

        all_rewards.append(total_reward)
        print(f"Epoch {epoch}/{EPOCHS} | Reward: {total_reward:.2f} | Steps: {ep_steps}")

        # Bootstrap
        with torch.no_grad():
            _, last_v_t = model.forward(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
            last_v = float(last_v_t.item())

        # Tensors
        states_t   = torch.tensor(np.array(states),  dtype=torch.float32, device=device)
        actions_t  = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        old_logp_t = torch.stack(log_probs).detach()
        returns_np = compute_gae(rewards, values, masks, last_value=last_v)
        returns_t  = torch.tensor(returns_np, dtype=torch.float32, device=device)

        # Advantage
        with torch.no_grad():
            _, v_pred = model.forward(states_t)
            v_pred = v_pred.squeeze(-1)
            adv = returns_t - v_pred
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            v_old = v_pred.clone()

        # PPO update
        idx_all = np.arange(len(states_t))
        last_actor_loss, last_critic_loss = 0.0, 0.0
        clip_fracs, kls = [], []

        ENT_COEF = current_ent_coef(epoch)

        for _ in range(TRAIN_ITERS):
            if len(idx_all) == 0:
                break
            np.random.shuffle(idx_all)
            for start in range(0, len(idx_all), MB_SIZE):
                mb_idx = idx_all[start:start + MB_SIZE]
                if len(mb_idx) == 0:
                    continue
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

                v_clip = mb_v_old + (v_pred_mb - mb_v_old).clamp(-VAL_CLIP_RANGE, VAL_CLIP_RANGE)
                critic_loss = 0.5 * torch.mean((mb_returns - v_clip) ** 2)

                total_loss = actor_loss + VF_COEF * critic_loss - ENT_COEF * entropy.mean()

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                with torch.no_grad():
                    # keep std in a healthy range
                    model.log_std.data.clamp_(min=np.log(0.08), max=np.log(1.5))
                    approx_kl = (mb_old_logp - new_logp).mean().item()
                    clip_frac = (torch.abs(ratio - 1.0) > CLIP_EPS).float().mean().item()
                kls.append(approx_kl)
                clip_fracs.append(clip_frac)

                last_actor_loss = float(actor_loss.item())
                last_critic_loss = float(critic_loss.item())

        EV = explained_variance(returns_t, v_pred)
        if epoch % 10 == 0:
            print(f"[Diag] clip_frac={np.mean(clip_fracs):.3f}  KL={np.mean(kls):.4f}  EV={EV:.3f}")

        # Adaptive train iters (slightly higher target KL)
        target_kl = 0.02
        mean_kl = float(np.mean(kls)) if len(kls) else 0.0
        if mean_kl < 0.5 * target_kl:
            TRAIN_ITERS = min(TRAIN_ITERS + 1, 25)
        elif mean_kl > 1.5 * target_kl:
            TRAIN_ITERS = max(TRAIN_ITERS - 1, 10)
        print(f"[Adapt] mean_KL={mean_kl:.4f} -> next TRAIN_ITERS={TRAIN_ITERS}")

        if mean_kl > 1.5 * target_kl:
            print(f"[EarlyStop] KL {mean_kl:.4f} > {1.5*target_kl:.4f} (epoch {epoch})")

        log_to_csv(CSV_PATH, {
            "epoch": epoch,
            "reward": total_reward,
            "actor_loss": last_actor_loss,
            "critic_loss": last_critic_loss
        }, header)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(LOG_DIR, f"ppo_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": {
                    "GAMMA": GAMMA, "LAMBDA": LAMBDA,
                    "CLIP_EPS": CLIP_EPS, "VF_COEF": VF_COEF,
                    "THRUST_SCALE": THRUST_SCALE
                }
            }, ckpt_path)
            print(f"[Checkpoint] Saved to {ckpt_path}")

        if epoch % 10 == 0:
            plot_curves(all_rewards, CSV_PATH, LOG_DIR)

    # Final checkpoint + plots
    torch.save({
        "epoch": EPOCHS,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": {
            "GAMMA": GAMMA, "LAMBDA": LAMBDA,
            "CLIP_EPS": CLIP_EPS, "VF_COEF": VF_COEF,
            "THRUST_SCALE": THRUST_SCALE
        }
    }, FINAL_CKPT)
    print(f"[Checkpoint] Saved final model to {FINAL_CKPT}")

    plot_curves(all_rewards, CSV_PATH, LOG_DIR)
    evaluate_and_plot_orbit(env, model, out_path=os.path.join(LOG_DIR, "orbit_trajectory.png"))
    plot_state_timeseries(env, model, out_path=os.path.join(LOG_DIR, "state_timeseries.png"))
    print("Training finished.")

if __name__ == "__main__":
    train()
