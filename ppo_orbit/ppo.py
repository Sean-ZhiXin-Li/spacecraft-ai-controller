import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

# Local import
from envs.orbit_env import OrbitEnv

# ==========================================================
# Device
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ==========================================================
# Hyperparameters (CLI-overridable)
# ==========================================================
GAMMA        = 0.995
LAMBDA       = 0.97
EPOCHS       = 800
TRAIN_ITERS  = 20
THRUST_SCALE = 3000.0
BATCH_STEPS  = 4096

# PPO specifics (stabilized & slightly more aggressive)
CLIP_EPS         = 0.20       # ↑ from 0.18
VF_COEF          = 1.2        # base; will be scheduled by current_vf_coef()
ENT_COEF_0       = 0.010      # ↑ from 0.008
ENT_COEF_1       = 0.001
ENT_SWITCH_EPOCH = 200        # ↓ from 300
MAX_GRAD_NORM    = 0.5
MB_SIZE          = 128

# Optimizer learning rates
LR_ACTOR   = 3e-5
LR_CRITIC  = 1e-3

# KL target (for adapt)
TARGET_KL_BASE = 0.017

# Logging / I/O
LOG_DIR   = "ppo_orbit"
PLOTS_DIR = LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH  = os.path.join(LOG_DIR, "loss_log.csv")

# Optional offline dataset (skipped if near-zero actions)
DATASET_PATH = os.path.join("data", "data", "preprocessed", "merged_expert_dataset.npy")


# ==========================================================
# Gym >=0.26 compatibility shims (also support legacy/custom env)
# ==========================================================
def reset_env(env, **kwargs):
    """Return only obs. Works for Gym>=0.26 (obs, info) and legacy/custom (obs)."""
    out = env.reset(**kwargs)
    if isinstance(out, tuple):
        return out[0]
    return out

def step_env(env, action):
    """
    Unify step return:
      - New Gym: (obs, reward, terminated, truncated, info) -> done = terminated or truncated
      - Legacy/custom: (obs, reward, done, info)
    Returns (obs, reward, done, info).
    """
    out = env.step(action)
    if isinstance(out, tuple):
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
            return obs, reward, done, info
        elif len(out) == 4:
            return out
    raise RuntimeError("Unsupported env.step return format.")


# ==========================================================
# Utils
# ==========================================================
def set_seed(seed: int):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_state(state):
    """Roughly scale position/velocity to [-1, 1] for PPO stability."""
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

def current_vf_coef(epoch:int)->float:
    """Slightly reduce value loss weight later to let policy step bigger."""
    return 1.2 if epoch <= 200 else 1.0


def dataset_action_stats(npy_path, thrust_scale, sample=500_000):
    """
    Inspect action distribution in the offline dataset after scaling to [-1,1].
    If actions are nearly zero (std and 95th percentile tiny), skip warmstart.
    """
    if not os.path.exists(npy_path):
        print(f"[Dataset] Not found: {npy_path}")
        return None

    try:
        data = np.load(npy_path, allow_pickle=True, mmap_mode='r')
    except Exception as e:
        print(f"[Dataset] Failed to mmap: {e}")
        return None

    if isinstance(data, np.ndarray) and data.dtype == object:
        try:
            d = np.load(npy_path, allow_pickle=True).item()
        except Exception as e:
            print(f"[Dataset] Object load failed: {e}")
            return None
        keys = {k.lower(): k for k in d.keys()}
        a_key = keys.get("actions") or keys.get("action") or keys.get("act") or keys.get("y")
        if a_key is None:
            print("[Dataset] No action key in dict.")
            return None
        acts = np.asarray(d[a_key], dtype=np.float32)
    else:
        arr = np.asarray(data)
        if arr.ndim != 2 or arr.shape[1] < 6:
            print(f"[Dataset] Unexpected shape {arr.shape}")
            return None
        N = arr.shape[0]
        n = min(sample, N)
        idx = np.random.choice(N, size=n, replace=False)
        acts = arr[idx, -2:].astype(np.float32)

    # Scale to [-1,1] if raw actions are likely in Newtons
    if np.max(np.abs(acts)) > 1.5:
        acts_scaled = acts / float(thrust_scale)
    else:
        acts_scaled = acts
    acts_scaled = np.clip(acts_scaled, -1.0, 1.0)

    mean = acts_scaled.mean(axis=0)
    std  = acts_scaled.std(axis=0)
    p95  = np.percentile(acts_scaled, 95, axis=0)
    amax = np.max(np.abs(acts_scaled), axis=0)

    print(f"[Dataset] action stats (scaled): mean={mean}, std={std}, p95={p95}, absmax={amax}")
    return {"mean": mean, "std": std, "p95": p95, "absmax": amax}


# ==========================================================
# Evaluation helpers
# ==========================================================
@torch.no_grad()
def evaluate_policy(env, model, thrust_scale, episodes=2, max_steps=20000):
    totals, lens = [], []
    for _ in range(episodes):
        obs = reset_env(env)
        s = normalize_state(obs)
        done = False
        ep_ret, steps = 0.0, 0
        while not done and steps < max_steps:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            mu, _ = model.forward(st)
            a_env = np.clip(mu.squeeze(0).detach().cpu().numpy(), -1.0, 1.0)
            next_obs, r, done, _ = step_env(env, a_env * thrust_scale)
            s = normalize_state(next_obs)
            ep_ret += r
            steps += 1
        totals.append(ep_ret); lens.append(steps)
    mean_ret = float(np.mean(totals))
    mean_len = float(np.mean(lens))
    print(f"[Eval] mean_return={mean_ret:.2f} | mean_len={mean_len:.0f} | per_step={mean_ret/max(1,mean_len):.6f}")
    return mean_ret


@torch.no_grad()
def evaluate_stochastic(env, model, thrust_scale, episodes=2, max_steps=20000):
    totals = []
    for _ in range(episodes):
        obs = reset_env(env)
        s = normalize_state(obs)
        done, ep_ret, steps = False, 0.0, 0
        while not done and steps < max_steps:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            mu, _ = model.forward(st)
            std = model.log_std.exp()
            a_raw = Normal(mu, std).sample().squeeze(0).detach().cpu().numpy()
            a_env = np.clip(a_raw, -1.0, 1.0)
            obs, r, done, _ = step_env(env, a_env * thrust_scale)
            s = normalize_state(obs); ep_ret += r; steps += 1
        totals.append(ep_ret)
    print(f"[Eval/Stoch] mean_return={float(np.mean(totals)):.2f}")


# ==========================================================
# Actor-Critic
# ==========================================================
class ActorCritic(nn.Module):
    def __init__(self, hidden1=256, hidden2=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(4, hidden1),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 1)
        )
        # initial exploration std
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
        """Return (log_probs, entropy, value) given states and *raw* sampled actions."""
        mu, value = self.forward(states)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, value.squeeze(-1)


# ==========================================================
# Returns normalization for critic (running stats)
# ==========================================================
class ValueNorm:
    """
    Track running mean/var of returns. Critic is trained in normalized space,
    while EV reporting uses de-normalized predictions for interpretability.
    """
    def __init__(self, eps=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: torch.Tensor):
        x = x.detach()
        batch_mean = x.mean().item()
        batch_var  = x.var(unbiased=False).item()
        batch_count = x.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
        new_var = M2 / max(total_count, 1.0)

        self.mean = new_mean
        self.var = max(new_var, 1e-8)
        self.count = total_count

    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / (self.var**0.5 + 1e-8)

    def denormalize(self, x: torch.Tensor):
        return x * (self.var**0.5 + 1e-8) + self.mean


# ==========================================================
# GAE
# ==========================================================
def compute_gae(rewards, values, masks, gamma=GAMMA, lam=LAMBDA, last_value=0.0):
    """Generalized Advantage Estimation with terminal bootstrap."""
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
# Expert warm start (offline or online)
# ==========================================================
def load_expert_from_npy(model, npy_path, thrust_scale, epochs=2, batch_size=4096,
                         shuffle=True, max_samples=1_000_000, progress_every=20):
    """
    Supervised warmstart of the actor from an offline (state, action) dataset.
    Skips if dataset actions are near-zero after scaling.
    """
    import torch.utils.data as tud
    import time

    print(f"[Init] Loading dataset from {npy_path} ...", flush=True)

    # Quick distribution check; skip if actions are tiny
    stats = dataset_action_stats(npy_path, thrust_scale)
    if stats is not None:
        if (np.max(stats["std"]) < 0.02) and (np.max(stats["p95"]) < 0.05):
            print("[Init] Actions in dataset are too small (near-zero). Skip offline warmstart.")
            return False

    if not os.path.exists(npy_path):
        print(f"[Warn] Dataset not found: {npy_path}. Skip offline warmstart.", flush=True)
        return False

    # Peek shape/keys
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

    # Parse states/actions
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

    # Normalize states and scale actions
    states_np = np.stack([normalize_state(s) for s in states_np], axis=0).astype(np.float32)
    if np.max(np.abs(actions_np)) > 1.5:
        actions_np = (actions_np / thrust_scale).astype(np.float32)
    actions_np = np.clip(actions_np, -1.0, 1.0).astype(np.float32)

    # Torch dataset/loader
    import torch.utils.data as tud
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


def _basic_expert_action(pos: np.ndarray, vel: np.ndarray, target_radius: float, thrust_scale: float) -> np.ndarray:
    """
    Minimal physics expert: radial correction + tangential speed correction (tanh-smoothed).
    Returns env action in [-1, 1]^2.
    """
    r = np.linalg.norm(pos) + 1e-8
    v = np.linalg.norm(vel) + 1e-8
    ur = pos / r
    # tangential direction: rotate radial 90°
    t_dir = np.array([-ur[1], ur[0]], dtype=np.float64)

    v_circ = np.sqrt(6.67430e-11 * 1.989e30 / target_radius)
    dv = v_circ - v
    radial_err = r - target_radius

    thrust_r = -8.0 * np.tanh(radial_err / (0.1 * target_radius))
    thrust_t =  6.0 * np.tanh(dv / max(v_circ, 1e-8))

    thrust_vec = thrust_r * ur + thrust_t * t_dir
    a_env = np.clip(thrust_vec / thrust_scale, -1.0, 1.0)
    return a_env.astype(np.float32)


def load_expert_online(model, env, thrust_scale, samples=20000):
    """
    Generate (state → action) pairs from the basic physics expert
    and fit actor with a small supervised warmstart.
    """
    states, targets = [], []
    for _ in range(samples):
        obs = reset_env(env)
        pos, vel = obs[:2], obs[2:]
        a_env = _basic_expert_action(pos, vel, env.target_radius, thrust_scale)
        states.append(normalize_state(obs))
        targets.append(a_env)

    states  = torch.tensor(np.array(states,  dtype=np.float32), device=device)
    targets = torch.tensor(np.array(targets, dtype=np.float32), device=device)
    actor_params = list(model.shared.parameters()) + list(model.actor.parameters()) + [model.log_std]
    opt = torch.optim.Adam(actor_params, lr=1e-4)
    mse = nn.MSELoss()

    for _ in range(600):
        mu, _ = model.forward(states)
        loss = mse(mu, targets)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(actor_params, 1.0)
        opt.step()
    print("[Init] Actor initialized from basic physics expert (online).")


# ==========================================================
# Plotting
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
def evaluate_and_plot_orbit(env, model, thrust_scale, out_path=os.path.join(LOG_DIR, "orbit_trajectory.png"), max_steps=20000):
    """Run one episode with mean action and plot XY trajectory."""
    obs = reset_env(env)
    s = normalize_state(obs)
    done = False
    ep_ret = 0.0
    xs, ys = [], []
    steps = 0
    while not done and steps < max_steps:
        xs.append(obs[0]); ys.append(obs[1])
        st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        mu, _ = model.forward(st)
        a_env = np.clip(mu.squeeze(0).detach().cpu().numpy(), -1.0, 1.0)
        obs, r, done, _ = step_env(env, a_env * thrust_scale)
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
def plot_state_timeseries(env, model, thrust_scale, out_path=os.path.join(LOG_DIR, "state_timeseries.png"), max_steps=20000):
    """Plot r(t), v(t), |cos(angle)| for a single evaluation rollout."""
    obs = reset_env(env)
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
        a_env = np.clip(mu.squeeze(0).detach().cpu().numpy(), -1.0, 1.0)
        obs, _, done, _ = step_env(env, a_env * thrust_scale)
        s = normalize_state(obs); t += env.dt

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

    # |cos(angle)|
    plt.figure()
    plt.plot(ts, coss)
    plt.xlabel("time (s)"); plt.ylabel("|cos(angle)|")
    plt.title("Alignment over time")
    plt.grid(True)
    plt.savefig(out_path.replace(".png", "_cos.png"), dpi=150)
    plt.close()
    print(f"[Plot] Saved state timeseries to {out_path.replace('.png','_{r,v,cos}.png')}")


def explained_variance(y_true_t, y_pred_t):
    """1 - Var[y - yhat] / Var[y]; robust to constant shifts."""
    y_true = y_true_t.detach().cpu().numpy()
    y_pred = y_pred_t.detach().cpu().numpy()
    var_y = np.var(y_true)
    return float(1 - np.var(y_true - y_pred) / (var_y + 1e-8))


# ==========================================================
# Train
# ==========================================================
def train(args):
    global TRAIN_ITERS, EPOCHS, THRUST_SCALE

    set_seed(args.seed)

    # CLI overrides
    EPOCHS       = int(args.epochs or EPOCHS)
    TRAIN_ITERS  = int(args.train_iters or TRAIN_ITERS)
    THRUST_SCALE = float(args.thrust_scale or THRUST_SCALE)

    env = OrbitEnv(thrust_scale=int(THRUST_SCALE))
    model = ActorCritic().to(device)

    optimizer = optim.Adam([
        {"params": list(model.shared.parameters()) + list(model.actor.parameters()) + [model.log_std], "lr": LR_ACTOR},
        {"params": list(model.critic.parameters()), "lr": LR_CRITIC},
    ])

    # Critic loss (Huber) is a bit more stable than MSE under reward scale drift
    huber = nn.SmoothL1Loss()
    value_norm = ValueNorm()

    # ---- Warm start: offline (if healthy) else online physics expert ----
    ok = load_expert_from_npy(
        model, DATASET_PATH, thrust_scale=THRUST_SCALE, epochs=2, batch_size=4096,
        max_samples=1_000_000, progress_every=20
    )
    if not ok:
        load_expert_online(model, env, thrust_scale=THRUST_SCALE, samples=20000)

    # Quick imitation check
    evaluate_policy(env, model, thrust_scale=THRUST_SCALE, episodes=1)

    # ------------------------------------------------------------------
    # Optional but stabilizing: one-epoch TD(lambda) critic bootstrap
    # ------------------------------------------------------------------
    critic_opt = optim.Adam(model.critic.parameters(), lr=LR_CRITIC)
    for _bootstrap_epoch in range(1):  # 1 quick epoch
        states, rewards, masks, values = [], [], [], []
        obs = reset_env(env)
        state = normalize_state(obs)
        steps = 0
        while steps < BATCH_STEPS:
            # rollout with mean action (less noisy for value fitting)
            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu, v = model.forward(st)
            a_env = np.clip(mu.squeeze(0).detach().cpu().numpy(), -1.0, 1.0)
            next_obs, r, done, _ = step_env(env, a_env * THRUST_SCALE)
            ns = normalize_state(next_obs)
            states.append(state); rewards.append(r); masks.append(0 if done else 1); values.append(float(v.item()))
            state = ns; steps += 1
            if done:
                obs = reset_env(env); state = normalize_state(obs)

        with torch.no_grad():
            st_last = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_v_t = model.forward(st_last)
            last_v = float(last_v_t.item())

        returns_np = compute_gae(rewards, values, masks, last_value=last_v)
        states_t   = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        returns_t  = torch.tensor(returns_np,        dtype=torch.float32, device=device)

        value_norm.update(returns_t)
        returns_norm = value_norm.normalize(returns_t)

        for _it in range(8):
            _, v_pred = model.forward(states_t)
            v_pred = v_pred.squeeze(-1)
            loss_v = huber(returns_norm, v_pred)
            critic_opt.zero_grad(set_to_none=True)
            loss_v.backward()
            nn.utils.clip_grad_norm_(model.critic.parameters(), MAX_GRAD_NORM)
            critic_opt.step()
    print("[Bootstrap] 1-epoch TD(lambda) critic warmup done.")

    all_rewards = []
    header = ["epoch", "reward", "actor_loss", "critic_loss"]

    for epoch in range(1, EPOCHS + 1):
        # === Rollout: collect exactly BATCH_STEPS across multiple episodes ===
        states, actions, rewards, masks, log_probs, values = [], [], [], [], [], []
        total_reward = 0.0

        # Curriculum (same逻辑，延长前期探索比例可通过 ENT 调整)
        def pick_mode(ep):
            if ep <= 300 and np.random.rand() < 0.6:
                return "spiral"
            return "default"

        obs = reset_env(env, start_mode=pick_mode(epoch))
        state = normalize_state(obs)

        steps_collected = 0
        while steps_collected < BATCH_STEPS:
            a_raw_np, a_env_np, log_prob = model.get_action(state)  # @torch.no_grad()
            next_obs, reward, done, _ = step_env(env, a_env_np * THRUST_SCALE)
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
            steps_collected += 1

            if done:
                obs = reset_env(env, start_mode=pick_mode(epoch))
                state = normalize_state(obs)

        all_rewards.append(total_reward)
        print(f"Epoch {epoch}/{EPOCHS} | Reward: {total_reward:.2f} | Steps: {steps_collected}")

        # Bootstrap last value
        with torch.no_grad():
            _, last_v_t = model.forward(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
            last_v = float(last_v_t.item())

        # Tensors
        states_t   = torch.tensor(np.array(states),  dtype=torch.float32, device=device)
        actions_t  = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        old_logp_t = torch.stack(log_probs).detach()
        returns_np = compute_gae(rewards, values, masks, last_value=last_v)
        returns_t  = torch.tensor(returns_np, dtype=torch.float32, device=device)

        # ---- Update running stats & get normalized returns for critic training
        value_norm.update(returns_t)
        returns_norm = value_norm.normalize(returns_t.detach())

        # Advantage (in normalized space)
        with torch.no_grad():
            _, v_pred_full = model.forward(states_t)
            v_pred_full = v_pred_full.squeeze(-1)
            # For EV reporting, de-normalize predictions
            v_pred_eval = value_norm.denormalize(v_pred_full.detach())

        adv = returns_norm - v_pred_full
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        v_old = v_pred_full.clone()

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

                mb_states        = states_t[mb]
                mb_actions       = actions_t[mb]
                mb_adv           = adv[mb]
                mb_returns       = returns_t[mb]
                mb_returns_norm  = returns_norm[mb]
                mb_old_logp      = old_logp_t[mb]
                mb_v_old         = v_old[mb]

                new_logp, entropy, v_pred_mb = model.evaluate(mb_states, mb_actions)
                ratio  = torch.exp(new_logp - mb_old_logp)
                surr1  = ratio * mb_adv
                surr2  = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                actor_loss  = -torch.min(surr1, surr2).mean()

                # Unclipped value loss in normalized space (more stable with ValueNorm)
                critic_loss = huber(mb_returns_norm, v_pred_mb)

                # ---- joint step (actor + critic) ----
                vf_coef = current_vf_coef(epoch)
                total_loss = actor_loss + vf_coef * critic_loss - ENT_COEF * entropy.mean()
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                with torch.no_grad():
                    # keep std in a healthy range (upper bound tightened)
                    model.log_std.data.clamp_(min=np.log(0.08), max=np.log(1.2))
                    approx_kl = (mb_old_logp - new_logp).mean().item()
                    ratio_now = torch.exp(new_logp - mb_old_logp)
                    clip_frac = (torch.abs(ratio_now - 1.0) > CLIP_EPS).float().mean().item()
                kls.append(approx_kl)
                clip_fracs.append(clip_frac)

                last_actor_loss  = float(actor_loss.item())
                last_critic_loss = float(critic_loss.item())

        # Report EVs: raw-scale (denorm) & normalized-scale (training space)
        EV_raw  = explained_variance(returns_t, v_pred_eval)
        EV_norm = explained_variance(value_norm.normalize(returns_t), v_pred_full.detach())
        if epoch % 10 == 0:
            print(f"[Diag] clip_frac={np.mean(clip_fracs):.3f}  KL={np.mean(kls):.4f}  EV_raw={EV_raw:.3f}  EV_norm={EV_norm:.3f}")

        # ===== KL-driven adaptation: iters + actor LR (more permissive ceiling) =====
        target_kl = TARGET_KL_BASE
        mean_kl = float(np.mean(kls)) if len(kls) else 0.0

        # 1) adjust iters
        if mean_kl < 0.5 * target_kl:
            TRAIN_ITERS = min(TRAIN_ITERS + 1, 25)
        elif mean_kl > 1.5 * target_kl:
            TRAIN_ITERS = max(TRAIN_ITERS - 1, 10)

        # 2) adjust actor LR (param_group 0), ↑ ceiling to 8e-5, ↑ growth to 1.03x
        actor_pg = optimizer.param_groups[0]
        old_lr = actor_pg["lr"]
        if mean_kl < 0.5 * target_kl:
            actor_pg["lr"] = min(old_lr * 1.03, 8e-5)   # gentle increase
        elif mean_kl > 1.5 * target_kl:
            actor_pg["lr"] = max(old_lr * 0.90, 1e-5)   # gentle decrease

        print(f"[Adapt] mean_KL={mean_kl:.4f} -> next TRAIN_ITERS={TRAIN_ITERS} | actor_lr={actor_pg['lr']:.2e}")
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
            plot_curves(all_rewards, CSV_PATH, PLOTS_DIR)

    # Final checkpoint + plots
    final_ckpt = os.path.join(LOG_DIR, f"ppo_epoch_{EPOCHS}.pth")
    torch.save({
        "epoch": EPOCHS,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": {
            "GAMMA": GAMMA, "LAMBDA": LAMBDA,
            "CLIP_EPS": CLIP_EPS, "VF_COEF": VF_COEF,
            "THRUST_SCALE": THRUST_SCALE
        }
    }, final_ckpt)
    print(f"[Checkpoint] Saved final model to {final_ckpt}")

    plot_curves(all_rewards, CSV_PATH, PLOTS_DIR)
    evaluate_policy(env, model, thrust_scale=THRUST_SCALE, episodes=2)
    evaluate_and_plot_orbit(env, model, thrust_scale=THRUST_SCALE, out_path=os.path.join(PLOTS_DIR, "orbit_trajectory.png"))
    plot_state_timeseries(env, model, thrust_scale=THRUST_SCALE, out_path=os.path.join(PLOTS_DIR, "state_timeseries.png"))
    print("Training finished.")


def parse_args():
    p = argparse.ArgumentParser(description="Day27 PPO (Hybrid Init + Conservative→Faster) [Gym>=0.26]")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="training epochs")
    p.add_argument("--train-iters", type=int, default=TRAIN_ITERS, help="PPO update iters per epoch")
    p.add_argument("--thrust-scale", type=float, default=THRUST_SCALE, help="env/PPO thrust scaling")
    p.add_argument("--seed", type=int, default=42, help="random seed (None to disable)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
