import os, glob, json, argparse, numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from envs.orbit_env import OrbitEnv
from envs.task_sampler import TaskSampler, TaskSpec
from envs.multi_orbit_env import MultiOrbitEnv
from controller.muti_orbit_expert_controller import ExpertController

def load_task_by_name(tasks_dir: str, name: str) -> TaskSpec:
    for p in glob.glob(os.path.join(tasks_dir, "*.json")):
        with open(p,"r") as f:
            data = json.load(f)
        if data.get("name") == name:
            # Convert to TaskSpec dataclass
            return TaskSpec(**data)
    raise FileNotFoundError(f"Task not found by name: {name}")

def run_traj(env: MultiOrbitEnv, task: TaskSpec, ctrl: ExpertController, max_steps=None):
    # Inject task and reset
    env.base.set_physical_params(mass=task.mass, thrust_newton=task.thrust_newton,
                                 max_steps=task.max_steps, r_target=task.r_target, seed=task.seed)
    env.base.set_initial_state(task.init_state)
    env.base.reset()
    xs, ys = [], []
    steps = 0
    while True:
        xs.append(env.base.pos[0])
        ys.append(env.base.pos[1])
        action, _ = ctrl.act(env.base.pos.tolist() + env.base.vel.tolist(), task)
        _, _, done, _ = env.step(action)
        steps += 1
        if done or (max_steps and steps >= max_steps):
            break
    return np.array(xs), np.array(ys), steps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_dir", required=True)
    ap.add_argument("--worst_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.worst_csv)
    base = OrbitEnv()
    sampler = TaskSampler(args.tasks_dir, mode="sequential")
    env = MultiOrbitEnv(base, sampler, normalize_obs=False)
    ctrl = ExpertController(mode="spiral_in", fire_frac=0.35)

    for i, row in df.iterrows():
        name = row["task_name"]
        task = load_task_by_name(args.tasks_dir, name)
        xs, ys, steps = run_traj(env, task, ctrl, max_steps=task.max_steps)

        plt.figure()
        plt.plot(xs, ys, linewidth=1.0)
        plt.scatter([0.0],[0.0], s=10)  # central body
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(f"{name} (steps={steps})")
        fp = os.path.join(args.out_dir, f"{i:02d}_{name}.png")
        plt.savefig(fp, dpi=150)
        plt.close()
        print(f"[replay] {name} -> {fp}")

if __name__ == "__main__":
    main()
