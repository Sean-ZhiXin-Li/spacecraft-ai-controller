from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, List, Dict, Any, Tuple
import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class BatteryRecord:
    """
    Unified per-task evaluation record. Keep it minimal-but-sufficient for A/B.
    """
    # Core metrics (you should be able to provide these from your eval loop)
    name: str                  # task id, e.g., "random_13"
    ret: float                 # episodic return (higher is better)
    r_err: float               # radial error (lower is better)
    v_err: float               # velocity error (lower is better)
    align: float               # alignment score (higher is better)
    steps: int                 # steps used
    violation: float           # violation rate or 0/1

    # Task meta for reproduction
    rt: float                  # target radius
    e: float                   # eccentricity or difficulty param (if any)
    mass: float
    thrust: float
    seed: int                  # random seed; -1 if unknown

    # Optional: full task config for one-click replay
    task_cfg: Dict[str, Any] | None = None


def _ensure_dir(path: str) -> None:
    """Create directory if missing."""
    os.makedirs(path, exist_ok=True)


def _call_expert(expert: Any, state: np.ndarray) -> np.ndarray:
    """
    Call the Expert to get a thrust action (2,).

    Supported interfaces:
      1) expert.get_action(state) -> thrust(2,)
      2) expert.compute_thrust(r, v) -> thrust(2,)
      3) expert(pos, vel) OR expert(t, pos, vel) -> thrust(2,)

    Notes:
      - If the env observation contains more than 4 dims, we assume the layout
        starts with [x, y, vx, vy, ...] and only take the first 4 entries.
    """
    # 1) get_action(state)
    if hasattr(expert, "get_action"):
        return expert.get_action(state)

    # Ensure we only pass 2D pos and 2D vel to the Expert
    # Heuristic: assume the first 4 dims are [x, y, vx, vy]
    state = np.asarray(state).reshape(-1)
    if state.shape[0] >= 4:
        r = state[:2]
        v = state[2:4]
    else:
        # Fallback: try best-effort split
        r = state[:2]
        v = state[2:]

    # 2) compute_thrust(r, v)
    if hasattr(expert, "compute_thrust"):
        return expert.compute_thrust(r, v)

    # 3) __call__(pos, vel) or __call__(t, pos, vel)
    if callable(expert):
        try:
            return expert(r, v)         # signature: (pos, vel)
        except TypeError:
            return expert(0.0, r, v)    # signature: (t, pos, vel)

    raise TypeError("Unsupported expert interface (expected get_action / compute_thrust / __call__).")


def run_expert_on_battery(
    make_env_fn: Callable[[Dict[str, Any]], Any],
    make_expert_fn: Callable[[Dict[str, Any]], Any],
    task_list: List[Dict[str, Any]],
    max_steps: int = 20000,
) -> List[BatteryRecord]:
    """
    Run Expert on the *same* task list as the Agent to form an upper-bound reference.

    Requirements:
      - make_env_fn(task_cfg) -> env
        env.reset(task_cfg) -> np.ndarray state
        env.step(action) -> (next_state, reward, done, info)
        info should contain 'r_err', 'v_err', 'align', 'violation' (or compute them yourself)
      - make_expert_fn(task_cfg) -> expert (one of the supported interfaces)
    """
    results: List[BatteryRecord] = []

    for t in task_list:
        env = make_env_fn(t)
        state = env.reset(task=t)
        expert = make_expert_fn(t)

        # NEW: allow adapters to see the env (to extract raw pos/vel etc.)
        if hasattr(expert, "bind_env"):
            expert.bind_env(env)

        ep_ret = 0.0
        ep_steps = 0
        last_info: Dict[str, Any] = {}

        for _ in range(max_steps):
            action = _call_expert(expert, state)
            state, reward, done, info = env.step(action)
            ep_ret += float(reward)
            ep_steps += 1
            last_info = info
            if done:
                break

        rec = BatteryRecord(
            name=t.get("name", f"task_{t.get('seed', -1)}"),
            ret=ep_ret,
            r_err=float(last_info.get("r_err", np.nan)),
            v_err=float(last_info.get("v_err", np.nan)),
            align=float(last_info.get("align", np.nan)),
            steps=ep_steps,
            violation=float(last_info.get("violation", 0.0)),
            rt=float(t.get("rt", np.nan)),
            e=float(t.get("e", np.nan)),
            mass=float(t.get("mass", np.nan)),
            thrust=float(t.get("thrust", np.nan)),
            seed=int(t.get("seed", -1)),
            task_cfg=t,
        )
        results.append(rec)

    return results


def _rank(records: List[BatteryRecord], by: str, ascending: bool) -> List[BatteryRecord]:
    """Sort records by a field name."""
    return sorted(records, key=lambda x: getattr(x, by), reverse=not ascending)


def export_worst_k(
    agent_records: List[BatteryRecord],
    k: int = 3,
    by: str = "ret",
    outdir: str = "ab/worst3",
) -> List[BatteryRecord]:
    """
    Export worst-K tasks for A/B (CSV + JSON + per-task spec).

    Ranking rule:
      - For metrics where bigger is better ('ret', 'align'), pick the *smallest* values.
      - For error-like metrics ('r_err', 'v_err', 'violation'), pick the *largest* values.
    """
    _ensure_dir(outdir)

    # Decide ascending or descending based on metric semantics
    if by in ("ret", "align"):
        ranked = _rank(agent_records, by=by, ascending=True)   # pick smallest
    else:
        ranked = _rank(agent_records, by=by, ascending=False)  # pick largest

    worst = ranked[:k]

    # CSV dump
    csv_path = os.path.join(outdir, f"worst_{k}_by_{by}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name","ret","r_err","v_err","align","steps","violation","rt","e","mass","thrust","seed"])
        for r in worst:
            w.writerow([r.name, r.ret, r.r_err, r.v_err, r.align, r.steps, r.violation, r.rt, r.e, r.mass, r.thrust, r.seed])

    # JSON dump (with task_cfg for one-click replay)
    json_path = os.path.join(outdir, f"worst_{k}_by_{by}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in worst], f, ensure_ascii=False, indent=2)

    # Individual task specs (for replay scripts)
    specs_dir = os.path.join(outdir, "task_specs")
    _ensure_dir(specs_dir)
    for r in worst:
        spec_path = os.path.join(specs_dir, f"{r.name}.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(r.task_cfg if r.task_cfg is not None else asdict(r), f, ensure_ascii=False, indent=2)

    return worst


def plot_reference_curve(
    agent_records: List[BatteryRecord],
    expert_records: List[BatteryRecord],
    metric: str = "ret",
    outdir: str = "ab/plots",
    title: str = "Agent vs Expert (Upper-Bound Reference)",
) -> str:
    """
    Plot a reference curve on the same task set (Agent vs Expert).
    Tasks are sorted by Agent performance from 'bad' to 'good' for the chosen metric.

    metric ∈ {'ret', 'r_err', 'v_err', 'align'}
    """
    _ensure_dir(outdir)

    # Sort by Agent performance:
    #   - 'ret'/'align': higher is better → sort ascending so left side is 'worse'
    #   - errors: lower is better → sort descending so left side is 'worse'
    if metric in ("ret", "align"):
        agent_sorted = sorted(agent_records, key=lambda x: getattr(x, metric))
    else:
        agent_sorted = sorted(agent_records, key=lambda x: getattr(x, metric), reverse=True)

    # Build name → expert record mapping
    expert_map = {r.name: r for r in expert_records}

    x = np.arange(len(agent_sorted))
    y_agent = np.array([getattr(r, metric) for r in agent_sorted])
    y_expert = np.array([
        getattr(expert_map.get(r.name, r), metric)  # fallback to Agent metric if missing
        for r in agent_sorted
    ])

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_agent, label=f"Agent ({metric})", linewidth=2)
    plt.plot(x, y_expert, label=f"Expert Upper-Bound ({metric})", linewidth=2, linestyle="--")
    plt.xlabel("Tasks (sorted by Agent performance)")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    out_path = os.path.join(outdir, f"refcurve_{metric}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def run_ab_compare(
    agent_records: List[BatteryRecord],
    task_list: List[Dict[str, Any]],
    make_env_fn: Callable[[Dict[str, Any]], Any],
    make_expert_fn: Callable[[Dict[str, Any]], Any],
    export_k: int = 3,
    worst_by: str = "ret",
    out_root: str = "ab/day32",
    metrics_for_curve: Tuple[str, ...] = ("ret", "r_err"),
) -> Dict[str, Any]:
    """
    One-shot pipeline:
      1) Run Expert on the same task list
      2) Plot reference curves for given metrics
      3) Export worst-K tasks for A/B
    """
    _ensure_dir(out_root)

    # 1) Expert evaluation
    expert_records = run_expert_on_battery(make_env_fn, make_expert_fn, task_list)

    # 2) Curves
    plot_dir = os.path.join(out_root, "plots")
    plot_paths = []
    for m in metrics_for_curve:
        plot_paths.append(plot_reference_curve(agent_records, expert_records, metric=m, outdir=plot_dir))

    # 3) Worst-K export
    worst_dir = os.path.join(out_root, "worst")
    worst = export_worst_k(agent_records, k=export_k, by=worst_by, outdir=worst_dir)

    return {
        "plots": plot_paths,
        "worst": worst,
        "expert_records": expert_records,
    }
