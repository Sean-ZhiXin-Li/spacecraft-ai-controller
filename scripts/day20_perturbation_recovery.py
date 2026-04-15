from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.day20_policy_surface import (
    DEFAULT_CHECKPOINTS,
    LoadedPolicy,
    LightweightOrbitEnv,
    compute_vr,
    compute_vt,
    detect_shutdown_step,
    ensure_dir,
    norm2,
    save_json,
    series_bounds,
    svg_header,
    to_points,
)

OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figs" / "day20_policy_surface"


@dataclass
class BranchSummary:
    label: str
    reactivated: bool
    max_action_after: float
    mean_action_after: float


def capture_rollout(policy: LoadedPolicy, env: LightweightOrbitEnv) -> Dict[str, object]:
    obs = env.reset()
    states: List[Dict[str, object]] = []
    obs_list: List[List[float]] = []
    action_norm: List[float] = []
    r_error: List[float] = []
    v_r_list: List[float] = []
    v_t_error: List[float] = []
    v_circ = math.sqrt(env.mu / env.target_radius)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        states.append(env.get_state())
        obs_list.append(list(obs))
        action = policy.act(obs)
        obs, terminated, truncated = env.step(action)
        action_norm.append(norm2(action[0], action[1]))
        r_error.append(norm2(env.pos[0], env.pos[1]) - env.target_radius)
        v_r_list.append(compute_vr(env.pos, env.vel))
        v_t_error.append(compute_vt(env.pos, env.vel) - v_circ)

    return {
        "states": states,
        "obs": obs_list,
        "action_norm": action_norm,
        "r_error": r_error,
        "v_r": v_r_list,
        "v_t_error": v_t_error,
        "target_radius": env.target_radius,
        "v_circ": v_circ,
    }


def apply_vr_nudge(state: Dict[str, object], delta_vr: float) -> Dict[str, object]:
    new_state = dict(state)
    pos = [float(v) for v in state["pos"]]
    vel = [float(v) for v in state["vel"]]
    r = norm2(pos[0], pos[1]) + 1e-12
    r_hat = [pos[0] / r, pos[1] / r]
    new_state["vel"] = [vel[0] + delta_vr * r_hat[0], vel[1] + delta_vr * r_hat[1]]
    return new_state


def continue_branch(
    policy: LoadedPolicy,
    state: Dict[str, object],
    obs: Sequence[float],
    horizon: int,
    manual_action: Optional[Sequence[float]] = None,
    pulse_steps: int = 0,
) -> Dict[str, List[float]]:
    env = LightweightOrbitEnv(
        thrust_scale=float(state["thrust_scale"]),
        r0_over_target=1.05,
        max_steps=int(state["max_steps"]),
    )
    env.set_state(state)
    current_obs = list(obs)
    v_circ = math.sqrt(env.mu / env.target_radius)

    action_norm: List[float] = []
    r_error: List[float] = []
    v_r_list: List[float] = []
    v_t_error: List[float] = []

    for step in range(horizon):
        if manual_action is not None and step < pulse_steps:
            action = [float(manual_action[0]), float(manual_action[1])]
        else:
            action = policy.act(current_obs)
        current_obs, terminated, truncated = env.step(action)
        action_norm.append(norm2(action[0], action[1]))
        r_error.append(norm2(env.pos[0], env.pos[1]) - env.target_radius)
        v_r_list.append(compute_vr(env.pos, env.vel))
        v_t_error.append(compute_vt(env.pos, env.vel) - v_circ)
        if terminated or truncated:
            break

    return {
        "action_norm": action_norm,
        "r_error": r_error,
        "v_r": v_r_list,
        "v_t_error": v_t_error,
    }


def summarize_branch(label: str, branch: Dict[str, List[float]], threshold: float = 0.08) -> BranchSummary:
    if not branch["action_norm"]:
        return BranchSummary(label=label, reactivated=False, max_action_after=0.0, mean_action_after=0.0)
    max_action = max(branch["action_norm"])
    mean_action = sum(branch["action_norm"]) / len(branch["action_norm"])
    return BranchSummary(
        label=label,
        reactivated=bool(max_action > threshold),
        max_action_after=float(max_action),
        mean_action_after=float(mean_action),
    )


def save_recovery_svg(
    path: Path,
    checkpoint_tag: str,
    shutdown_step: int,
    baseline_prefix: Dict[str, List[float]],
    branches: Dict[str, Dict[str, List[float]]],
) -> None:
    width, height = 980, 1180
    left, plot_w = 110, 700
    top0, panel_h, gap = 70, 210, 45
    panels = [
        ("action_norm", "action_norm"),
        ("r_error", "r_error [m]"),
        ("v_r", "v_r [m/s]"),
        ("v_t_error", "v_t_error [m/s]"),
    ]
    colors = {
        "baseline_pre": "#000000",
        "baseline_post": "#1f77b4",
        "vr_nudge_post": "#d62728",
        "thrust_pulse_post": "#2ca02c",
    }
    lines = svg_header(width, height)
    lines.append(f'<text x="{width / 2:.2f}" y="32" font-size="22" text-anchor="middle">Day20 perturbation recovery | {checkpoint_tag} | shutdown_step={shutdown_step}</text>')

    after_len = 1
    for branch in branches.values():
        after_len = max(after_len, len(branch["action_norm"]))
    before_len = max(1, len(baseline_prefix["action_norm"]))
    x_min = -before_len
    x_max = after_len - 1 if after_len > 1 else 1
    x_values = [float(v) for v in range(x_min, x_max + 1)]

    for panel_idx, (metric, ylabel) in enumerate(panels):
        top = top0 + panel_idx * (panel_h + gap)
        bottom = top + panel_h
        series_map: Dict[str, List[float]] = {}
        series_xs: Dict[str, List[float]] = {}

        pre_values = baseline_prefix[metric]
        pre_xs = [float(v) for v in range(-len(pre_values), 0)]
        series_map["baseline_pre"] = pre_values
        series_xs["baseline_pre"] = pre_xs
        for label, branch in branches.items():
            xs = [float(v) for v in range(0, len(branch[metric]))]
            series_map[label] = branch[metric]
            series_xs[label] = xs

        y0, y1 = series_bounds(series_map.values())
        lines.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{panel_h}" fill="none" stroke="black" stroke-width="1.5"/>')
        if y0 <= 0.0 <= y1:
            zero_y = top + panel_h - (0.0 - y0) / (y1 - y0 + 1e-12) * panel_h
            lines.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_w}" y2="{zero_y:.2f}" stroke="#777" stroke-dasharray="6,4"/>')
        if x_min <= 0.0 <= x_max:
            zero_x = left + (0.0 - x_min) / (x_max - x_min + 1e-12) * plot_w
            lines.append(f'<line x1="{zero_x:.2f}" y1="{top}" x2="{zero_x:.2f}" y2="{bottom}" stroke="#777" stroke-dasharray="6,4"/>')

        for label, ys in series_map.items():
            xs = series_xs[label]
            if not ys:
                continue
            pts = to_points(xs, ys, x_min, x_max, y0, y1, left, top, plot_w, panel_h)
            lines.append(f'<polyline points="{pts}" fill="none" stroke="{colors[label]}" stroke-width="2"/>')

        for idx in range(5):
            xt = idx / 4.0
            x = left + plot_w * xt
            value = x_min + (x_max - x_min) * xt
            lines.append(f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom + 6}" stroke="black"/>')
            lines.append(f'<text x="{x:.2f}" y="{bottom + 22}" font-size="12" text-anchor="middle">{value:.0f}</text>')
        for idx in range(5):
            yt = idx / 4.0
            y = bottom - panel_h * yt
            value = y0 + (y1 - y0) * yt
            lines.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="black"/>')
            lines.append(f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{value:.3f}</text>')
        lines.append(f'<text x="28" y="{top + panel_h / 2:.2f}" font-size="15" text-anchor="middle" transform="rotate(-90 28 {top + panel_h / 2:.2f})">{ylabel}</text>')

    legend_y = 1020
    legend_x = 160
    for idx, label in enumerate(["baseline_pre", "baseline_post", "vr_nudge_post", "thrust_pulse_post"]):
        x = legend_x + idx * 180
        lines.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 24}" y2="{legend_y}" stroke="{colors[label]}" stroke-width="4"/>')
        lines.append(f'<text x="{x + 32}" y="{legend_y + 5}" font-size="13">{label}</text>')

    lines.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 28}" font-size="16" text-anchor="middle">steps relative to perturbation</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 20 PPO perturbation recovery diagnostic.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--thrust-scale", type=float, default=20000.0)
    parser.add_argument("--r0-over-target", type=float, default=1.05)
    parser.add_argument("--window-before", type=int, default=200)
    parser.add_argument("--window-after", type=int, default=400)
    parser.add_argument("--vr-nudge", type=float, default=200.0)
    parser.add_argument("--pulse-action", type=float, default=0.30)
    parser.add_argument("--pulse-steps", type=int, default=3)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    summary: Dict[str, object] = {}

    for checkpoint_tag, checkpoint_path in DEFAULT_CHECKPOINTS.items():
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        policy = LoadedPolicy(checkpoint_path)
        env = LightweightOrbitEnv(
            thrust_scale=args.thrust_scale,
            r0_over_target=args.r0_over_target,
            max_steps=args.max_steps,
        )
        captured = capture_rollout(policy, env)
        shutdown_step = detect_shutdown_step(
            captured["action_norm"],
            captured["r_error"],
            captured["v_t_error"],
            float(captured["target_radius"]),
            float(captured["v_circ"]),
        )

        state = captured["states"][shutdown_step]
        obs = captured["obs"][shutdown_step]
        start_idx = max(0, shutdown_step - args.window_before)
        baseline_prefix = {
            "action_norm": captured["action_norm"][start_idx:shutdown_step],
            "r_error": captured["r_error"][start_idx:shutdown_step],
            "v_r": captured["v_r"][start_idx:shutdown_step],
            "v_t_error": captured["v_t_error"][start_idx:shutdown_step],
        }

        pos = [float(v) for v in state["pos"]]
        r = norm2(pos[0], pos[1]) + 1e-12
        radial_pulse = [args.pulse_action * pos[0] / r, args.pulse_action * pos[1] / r]

        baseline_branch = continue_branch(policy, state, obs, args.window_after)
        vr_branch = continue_branch(policy, apply_vr_nudge(state, args.vr_nudge), obs, args.window_after)
        pulse_branch = continue_branch(policy, state, obs, args.window_after, manual_action=radial_pulse, pulse_steps=args.pulse_steps)

        branches = {
            "baseline_post": baseline_branch,
            "vr_nudge_post": vr_branch,
            "thrust_pulse_post": pulse_branch,
        }
        save_recovery_svg(
            args.output_dir / f"perturbation_recovery_{checkpoint_tag}.svg",
            checkpoint_tag,
            shutdown_step,
            baseline_prefix,
            branches,
        )

        summaries = [
            summarize_branch("baseline_post", baseline_branch),
            summarize_branch("vr_nudge_post", vr_branch),
            summarize_branch("thrust_pulse_post", pulse_branch),
        ]
        summary[checkpoint_tag] = {
            "checkpoint_path": str(checkpoint_path),
            "shutdown_step": int(shutdown_step),
            "state_at_shutdown": {
                "r_error": float(captured["r_error"][shutdown_step]),
                "v_r": float(captured["v_r"][shutdown_step]),
                "v_t_error": float(captured["v_t_error"][shutdown_step]),
                "action_norm": float(captured["action_norm"][shutdown_step]),
            },
            "branches": [asdict(item) for item in summaries],
        }

    save_json(args.output_dir / "perturbation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
