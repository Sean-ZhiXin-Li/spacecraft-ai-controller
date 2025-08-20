import os, csv, argparse, numpy as np
from typing import Dict, Any

from envs.orbit_env import OrbitEnv
from envs.task_sampler import TaskSampler, TaskSpec
from envs.multi_orbit_env import MultiOrbitEnv
from controller.muti_orbit_expert_controller import ExpertController

def make_env(tasks_dir: str, normalize_obs: bool = False) -> MultiOrbitEnv:
    base = OrbitEnv()
    sampler = TaskSampler(tasks_dir, mode="sequential")
    return MultiOrbitEnv(base, sampler, normalize_obs=normalize_obs)

def make_controller(name: str):
    if name == "random":
        def policy(state, task):
            a = (task.thrust_newton / task.mass)
            theta = np.random.uniform(0, 2*np.pi)
            return np.array([a*np.cos(theta), a*np.sin(theta)]), {}
        return policy
    elif name == "expert:spiral_in":
        ctrl = ExpertController(mode="spiral_in", fire_frac=0.35)
        return lambda s, t: ctrl.act(s, t)
    elif name == "expert:bangband":
        ctrl = ExpertController(mode="bangband", band=(1.25, 1.8), fire_frac=1.0, bang_bang=True)
        return lambda s, t: ctrl.act(s, t)
    elif name == "expert:transfer":
        # Full throttle for decisive burns
        ctrl = ExpertController(mode="transfer", fire_frac=1.0, bang_bang=False, circ_tol=0.05)
        return lambda s, t: ctrl.act(s, t)
    elif name == "expert:elliptic":
        # Stronger damping for eccentricity
        ctrl = ExpertController(mode="elliptic_circ", fire_frac=0.8, bang_bang=False, circ_tol=0.05)
        return lambda s, t: ctrl.act(s, t)

    else:
        raise ValueError(f"Unknown controller: {name}")

def success_heuristic(r_over_rt, tail: int = 800, tol: float = 0.05) -> int:
    """Return 1 if mean |r/rt - 1| over the tail is < tol."""
    if len(r_over_rt) < tail:
        return 0
    tail_vals = r_over_rt[-tail:]
    return int(np.mean(np.abs(tail_vals - 1.0)) < tol)

def run_one_task(env, task, ctrl_name):
    ctrl = make_controller(ctrl_name)

    # Inject physical params
    env.base.set_physical_params(
        mass=task.mass, thrust_newton=task.thrust_newton,
        max_steps=task.max_steps, r_target=task.r_target, seed=task.seed
    )

    # Reset FIRST to clear counters and any presets
    env.base.reset()

    # THEN apply the task's initial state so it won't be overwritten
    env.base.set_initial_state(task.init_state)

    # Build initial obs from the just-set state
    obs = np.concatenate([env.base.pos, env.base.vel])

    r_over_rt = []
    ret, steps, done = 0.0, 0, False
    while not done and steps < task.max_steps:
        state = env.base.pos.tolist() + env.base.vel.tolist()

        # Controller returns acceleration command (m/s^2)
        accel_cmd, _ = ctrl(state, task)

        # Map acceleration -> action fraction in [-1,1]^2
        m = float(task.mass)
        F = float(task.thrust_newton)
        action = (np.array(accel_cmd, dtype=np.float64) * m) / max(F, 1e-12)
        action = np.clip(action, -1.0, 1.0)

        obs, r, done, info = env.step(action)
        radius = float(np.linalg.norm(env.base.pos))
        r_over_rt.append(radius / float(task.r_target))
        ret += float(r)
        steps += 1

    return {
        "task_name": task.name,
        "orbit_type": task.orbit_type,
        "controller": ctrl_name,
        "success": success_heuristic(np.array(r_over_rt)),
        "return": ret,
        "r_err": float(np.mean(np.abs(np.array(r_over_rt) - 1.0))) if r_over_rt else 1e9,
        "mass": task.mass,
        "thrust": task.thrust_newton,
        "r_target": task.r_target,
        "max_steps": task.max_steps,
        "seed": task.seed
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--controllers", nargs="+",
                    default=["expert:spiral_in", "expert:bangband", "random"])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    env = make_env(args.tasks_dir, normalize_obs=False)

    # Load all task paths once for deterministic sequential traversal
    sampler = env.sampler
    tasks = [sampler._load(p) for p in sampler.paths]

    rows = []
    for task in tasks:
        for cname in args.controllers:
            res = run_one_task(env, task, cname)
            rows.append(res)
            print(f"[{cname}] {task.name} | succ={res['success']} r_err={res['r_err']:.3e} ret={res['return']:.1f}")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[baseline] wrote {len(rows)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()
