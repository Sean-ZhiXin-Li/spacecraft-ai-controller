import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

from envs.orbit_env_mt import OrbitEnvMT
from baselines.zero_thrust import ZeroThrustController
from baselines.greedy_energy_rt import GreedyEnergyRTController
from controller.expert_controller import ExpertController
import pandas as pd


# Expert Adapter (index-driven; uses raw env fields if available)
class ExpertAdapter:
    """
    Wrap ExpertController so that it can act on OrbitEnvMT by reading raw
    position/velocity from the env. If not available, slice obs by indices.
    """
    def __init__(self, expert_ctrl, pos_idx=(0, 1), vel_idx=(2, 3), quiet=True):
        self.ctrl = expert_ctrl
        self.env = None
        self.pos_idx = tuple(pos_idx)
        self.vel_idx = tuple(vel_idx)
        self.quiet = quiet
        self._warned = False

    def bind_env(self, env):
        self.env = env

    def act(self, obs):
        r, v = self._extract_rv(obs)
        # ExpertController signature: __call__(t, pos, vel) -> thrust(2,)
        return self.ctrl(0.0, r, v)

    def _extract_rv(self, obs):
        import numpy as _np

        # Prefer dedicated methods/attrs on env
        if self.env is not None:
            for meth in ("get_raw_rv", "get_state_vectors", "raw_rv"):
                if hasattr(self.env, meth):
                    try:
                        r, v = getattr(self.env, meth)()
                        return _np.asarray(r).reshape(-1)[:2], _np.asarray(v).reshape(-1)[:2]
                    except Exception:
                        pass

            pairs = [("pos", "vel"), ("r", "v"), ("position", "velocity"),
                     ("x", "v"), ("x", "xdot"), ("r_vec", "v_vec")]
            for pa, va in pairs:
                if hasattr(self.env, pa) and hasattr(self.env, va):
                    r = _np.asarray(getattr(self.env, pa)).reshape(-1)[:2]
                    v = _np.asarray(getattr(self.env, va)).reshape(-1)[:2]
                    return r, v

            if hasattr(self.env, "state"):
                st = getattr(self.env, "state")
                if isinstance(st, dict):
                    for pa, va in pairs:
                        if pa in st and va in st:
                            r = _np.asarray(st[pa]).reshape(-1)[:2]
                            v = _np.asarray(st[va]).reshape(-1)[:2]
                            return r, v

        # Fallback: slice obs
        obs = _np.asarray(obs).reshape(-1)
        try:
            r = obs[list(self.pos_idx)]
            v = obs[list(self.vel_idx)]
        except Exception:
            r = obs[:2]
            v = obs[2:4] if obs.shape[0] >= 4 else obs[2:]
        if not self.quiet and not self._warned:
            print("[replay][warn] Using obs indices for pos/vel:", self.pos_idx, self.vel_idx)
            self._warned = True
        return r, v


# Env/Controller factories
def make_env():
    """Create OrbitEnvMT with the same settings used in eval_battery.py."""
    return OrbitEnvMT(
        rerr_thr=0.015, verr_thr=0.030, align_thr=0.97, stable_steps=160,
        w_fuel=2e-4, w_align=0.2,
        dt=6000.0, max_steps=30000,
        thrust_scale_range=(100.0, 150.0),
    )


def make_agent_controller(name: str):
    """Return an agent controller by name; extend as needed."""
    name = name.lower()
    if name == "zero":
        return ZeroThrustController()
    if name == "greedy_energy_rt":
        return GreedyEnergyRTController(
            k_e=0.9, k_rp=0.10, k_rd=0.60,
            t_clip=0.41, a_max_lo=0.048, a_max_hi=0.43,
            dead_r_in=0.020, dead_r_out=0.017,
            dead_v_in=0.040, dead_v_out=0.033,
            v_des_min=0.82, v_des_max=1.19
        )
    raise ValueError(f"Unknown agent controller: {name}")


def make_expert_adapter(task_cfg, pos_idx=(0, 1), vel_idx=(2, 3)):
    """Build Expert wrapped by ExpertAdapter. task_cfg supplies rt/mass/thrust."""
    expert = ExpertController(
        target_radius=task_cfg["rt"],
        mass=task_cfg.get("mass", 721.9),
        thrust_limit=task_cfg.get("thrust", 20.0),
        radial_gain=4.0, tangential_gain=5.0, damping_gain=6.0,
        enable_damping=True,
    )
    return ExpertAdapter(expert, pos_idx=pos_idx, vel_idx=vel_idx, quiet=True)


# Rollout
def run_one_episode(env, controller, task_cfg, seed=None):
    """
    Reset env with a task spec and run a single episode.
    Returns: (ep_ret, steps, info)
    """
    # Map ab_tools task spec -> env expected keys
    env_task = dict(
        name=task_cfg.get("name", "task"),
        target_radius=task_cfg["rt"],
        e=task_cfg.get("e", 0.0),
        mass=task_cfg.get("mass", 720.0),
        thrust_limit=task_cfg.get("thrust", 1.0),
    )
    obs = env.reset(task=env_task, seed=seed)
    if hasattr(controller, "set_task"):
        controller.set_task(env.task)
    if hasattr(controller, "bind_env"):
        controller.bind_env(env)

    done = False
    ep_ret, steps = 0.0, 0
    info = {}
    while not done:
        if hasattr(controller, "act"):
            a = controller.act(obs)
        else:
            a = controller.get_action(obs)  # ExpertAdapter path
        obs, r, done, info = env.step(a)
        ep_ret += float(r)
        steps += 1
    return ep_ret, steps, info


def run_controller_on_tasks(controller, task_list, seed_base=12345):
    rows = []
    env = make_env()
    for i, t in enumerate(task_list):
        ret, steps, info = run_one_episode(env, controller, t, seed=seed_base + i)
        rows.append(dict(
            name=t["name"], ret=ret, steps=steps,
            r_err=float(info.get("r_err", np.nan)),
            v_err=float(info.get("v_err", np.nan)),
            align=float(info.get("align", np.nan)),
            fuel=float(info.get("fuel_used", np.nan)),
            success=bool(info.get("success", False)),
        ))
    return rows


# I/O helpers
def load_worst_task_specs(worst_dir):
    specs = []
    for path in sorted(glob.glob(os.path.join(worst_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            t = json.load(f)
        # normalize keys for Expert/env
        specs.append(dict(
            name=t.get("name", os.path.splitext(os.path.basename(path))[0]),
            rt=float(t["rt"]) if "rt" in t else float(t["target_radius"]),
            e=float(t.get("e", 0.0)),
            mass=float(t.get("mass", 720.0)),
            thrust=float(t.get("thrust", t.get("thrust_limit", 1.0))),
            seed=int(t.get("seed", -1)),
        ))
    return specs


def save_csv(rows_agent, rows_expert, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        cols = ["name",
                "agent_ret", "expert_ret",
                "agent_r_err", "expert_r_err",
                "agent_v_err", "expert_v_err",
                "agent_align", "expert_align",
                "agent_fuel", "expert_fuel",
                "agent_success", "expert_success",
                "agent_steps", "expert_steps"]
        f.write(",".join(cols) + "\n")
        # join by name
        m = {r["name"]: r for r in rows_expert}
        for a in rows_agent:
            e = m.get(a["name"], {})
            vals = [
                a["name"],
                a["ret"], e.get("ret", np.nan),
                a["r_err"], e.get("r_err", np.nan),
                a["v_err"], e.get("v_err", np.nan),
                a["align"], e.get("align", np.nan),
                a["fuel"], e.get("fuel", np.nan),
                int(a["success"]), int(e.get("success", False)),
                a["steps"], e.get("steps", np.nan),
            ]
            f.write(",".join(str(v) for v in vals) + "\n")
    return out_csv


def plot_bars(rows_agent, rows_expert, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Order by agent ret ascending (worst first)
    order = np.argsort([r["ret"] for r in rows_agent])
    names = [rows_agent[i]["name"] for i in order]

    def _bar(metric, ylabel):
        xa = [rows_agent[i][metric] for i in order]
        xe = [rows_expert[i][metric] for i in order]
        plt.figure(figsize=(10, 5))
        x = np.arange(len(names))
        w = 0.4
        plt.bar(x - w/2, xa, width=w, label=f"Agent ({metric})")
        plt.bar(x + w/2, xe, width=w, label=f"Expert ({metric})")
        plt.xticks(x, names, rotation=45, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"Agent vs Expert on worst tasks — {metric}")
        plt.legend()
        outp = os.path.join(out_dir, f"ab_{metric}.png")
        plt.tight_layout()
        plt.savefig(outp, dpi=180)
        plt.close()
        return outp

    paths = []
    paths.append(_bar("ret", "return"))
    paths.append(_bar("r_err", "r_err"))
    paths.append(_bar("fuel", "fuel"))
    return paths


# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Replay worst-K tasks and compare Agent vs Expert.")
    p.add_argument("--worst-dir", type=str, default="ab/day32/worst/task_specs",
                   help="Directory of per-task JSON specs exported by ab_tools")
    p.add_argument("--out", type=str, default="ab/day32/replay",
                   help="Output directory for CSV and figures")
    p.add_argument("--agent", type=str, default="greedy_energy_rt",
                   choices=["greedy_energy_rt", "zero"], help="Agent controller to replay")
    p.add_argument("--pos-idx", type=str, default="0,1",
                   help="Indices for position in obs, e.g., '0,1'")
    p.add_argument("--vel-idx", type=str, default="2,3",
                   help="Indices for velocity in obs, e.g., '2,3'")
    return p.parse_args()


def main():
    args = parse_args()
    pos_idx = tuple(int(i) for i in args.pos_idx.split(","))
    vel_idx = tuple(int(i) for i in args.vel_idx.split(","))

    tasks = load_worst_task_specs(args.worst_dir)
    if not tasks:
        print(f"[replay] No task specs found in {args.worst_dir}")
        return

    # Agent
    agent_ctrl = make_agent_controller(args.agent)
    rows_agent = run_controller_on_tasks(agent_ctrl, tasks)

    # Expert
    rows_expert = []
    env = make_env()
    for i, t in enumerate(tasks):
        expert = make_expert_adapter(t, pos_idx=pos_idx, vel_idx=vel_idx)
        ret, steps, info = run_one_episode(env, expert, t, seed=9999 + i)
        rows_expert.append(dict(
            name=t["name"], ret=ret, steps=steps,
            r_err=float(info.get("r_err", np.nan)),
            v_err=float(info.get("v_err", np.nan)),
            align=float(info.get("align", np.nan)),
            fuel=float(info.get("fuel_used", np.nan)),
            success=bool(info.get("success", False)),
        ))

    # Save CSV, compare and plots
    out_csv = os.path.join(args.out, "replay_worst_ab.csv")
    save_csv(rows_agent, rows_expert, out_csv)
    figs = plot_bars(rows_agent, rows_expert, os.path.join(args.out, "figs"))

    print("[replay] CSV ->", out_csv)
    print("[replay] Figures ->", figs)
    df = pd.read_csv("ab/day32/replay/replay_worst_ab.csv")
    df["Δret"] = df["expert_ret"] - df["agent_ret"]
    df["Δr_err"] = df["expert_r_err"] - df["agent_r_err"]
    df["Δfuel"] = df["expert_fuel"] - df["agent_fuel"]
    print(
        df[["name", "agent_success", "expert_success", "Δret", "Δr_err", "Δfuel"]].sort_values("Δret", ascending=True))


if __name__ == "__main__":
    main()
