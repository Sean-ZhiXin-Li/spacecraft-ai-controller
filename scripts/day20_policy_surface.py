from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figs" / "day20_policy_surface"
DEFAULT_CHECKPOINTS = {
    "speed_refine_50": PROJECT_ROOT / "ppo_orbit" / "speed_refine_50" / "ppo_epoch_300.pth",
    "state_vr_nonlinear_100": PROJECT_ROOT / "ppo_orbit" / "state_vr_nonlinear_100" / "ppo_best.pth",
}
DEVICE = torch.device("cpu")


@dataclass
class CollapseState:
    checkpoint_tag: str
    checkpoint_path: str
    step: int
    r_error: float
    v_r: float
    v_t_error: float
    action_norm: float
    target_radius: float
    v_circ: float


class LegacyActorCritic(nn.Module):
    def __init__(self, hidden1: int = 256, hidden2: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(4, hidden1),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 1),
        )
        self.log_std = nn.Parameter(torch.log(torch.ones(2, device=DEVICE) * 0.35))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(x)
        return self.actor(x), self.critic(x)


class FiveDimActorCritic(nn.Module):
    def __init__(self, hidden1: int = 256, hidden2: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(5, hidden1),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 1),
        )
        self.log_std = nn.Parameter(torch.log(torch.ones(2, device=DEVICE) * 0.35))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(x)
        return self.actor(x), self.critic(x)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def norm2(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def compute_vr(pos: Sequence[float], vel: Sequence[float]) -> float:
    r = norm2(pos[0], pos[1])
    if r < 1e-12:
        return 0.0
    return (pos[0] * vel[0] + pos[1] * vel[1]) / r


def compute_vt(pos: Sequence[float], vel: Sequence[float]) -> float:
    r = norm2(pos[0], pos[1])
    if r < 1e-12:
        return 0.0
    r_hat_x = pos[0] / r
    r_hat_y = pos[1] / r
    t_hat_x = -r_hat_y
    t_hat_y = r_hat_x
    return vel[0] * t_hat_x + vel[1] * t_hat_y


def normalize_rollout_state(state: Sequence[float], input_dim: int) -> List[float]:
    pos_scale = 7.5e12
    vel_scale = 3.0e4
    if input_dim == 4:
        return [
            state[0] / pos_scale,
            state[1] / pos_scale,
            state[2] / vel_scale,
            state[3] / vel_scale,
        ]
    vr_norm = state[4] / vel_scale
    vr_scaled = vr_norm * (0.1 + 0.9 * math.exp(-abs(vr_norm) * 10.0))
    return [
        state[0] / pos_scale,
        state[1] / pos_scale,
        state[2] / vel_scale,
        state[3] / vel_scale,
        vr_scaled,
    ]


class LoadedPolicy:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = str(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        shared_weight = state_dict.get("shared.0.weight")
        if shared_weight is None:
            raise KeyError(f"{checkpoint_path} missing shared.0.weight")
        self.input_dim = int(shared_weight.shape[1])
        if self.input_dim == 4:
            self.model = LegacyActorCritic().to(DEVICE)
        elif self.input_dim == 5:
            self.model = FiveDimActorCritic().to(DEVICE)
        else:
            raise ValueError(f"Unsupported input dimension: {self.input_dim}")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def act(self, obs: Sequence[float]) -> List[float]:
        state = normalize_rollout_state(obs[: self.input_dim], self.input_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.model(state_tensor)
            action = torch.clamp(mu, -1.0, 1.0).squeeze(0).cpu().tolist()
        return [float(action[0]), float(action[1])]


class LightweightOrbitEnv:
    def __init__(
        self,
        *,
        thrust_scale: float = 20000.0,
        r0_over_target: float = 1.05,
        max_steps: int = 4000,
    ) -> None:
        self.G = 6.67430e-11
        self.M = 1.989e30
        self.mu = self.G * self.M
        self.mass = 722.0
        self.dt = 2.0
        self.max_steps = int(max_steps)
        self.target_radius = 7.5e12
        self.thrust_scale = float(thrust_scale)
        self.r0_over_target = float(r0_over_target)
        self.success_threshold = 40
        self.tol_r = 1.8e-2
        self.tol_v = 1.8e-2
        self.tol_ang = 0.16
        self.reset()

    def reset(self) -> List[float]:
        self.steps = 0
        self.success_counter = 0
        self.radial_stall_counter = 0
        self.pos = [0.0, self.r0_over_target * self.target_radius]
        radius = norm2(self.pos[0], self.pos[1])
        v_mag = math.sqrt(self.mu / radius)
        angle = math.radians(170.0)
        self.vel = [v_mag * math.cos(angle), v_mag * math.sin(angle)]
        return self.get_obs()

    def get_obs(self) -> List[float]:
        v_r = compute_vr(self.pos, self.vel)
        return [self.pos[0], self.pos[1], self.vel[0], self.vel[1], v_r]

    def get_state(self) -> Dict[str, object]:
        return {
            "steps": int(self.steps),
            "success_counter": int(self.success_counter),
            "radial_stall_counter": int(self.radial_stall_counter),
            "pos": list(self.pos),
            "vel": list(self.vel),
            "target_radius": float(self.target_radius),
            "thrust_scale": float(self.thrust_scale),
            "max_steps": int(self.max_steps),
        }

    def set_state(self, state: Dict[str, object]) -> None:
        self.steps = int(state["steps"])
        self.success_counter = int(state.get("success_counter", 0))
        self.radial_stall_counter = int(state.get("radial_stall_counter", 0))
        self.pos = [float(v) for v in state["pos"]]
        self.vel = [float(v) for v in state["vel"]]
        self.target_radius = float(state.get("target_radius", self.target_radius))
        self.thrust_scale = float(state.get("thrust_scale", self.thrust_scale))
        self.max_steps = int(state.get("max_steps", self.max_steps))

    def step(self, action: Sequence[float]) -> Tuple[List[float], bool, bool]:
        self.steps += 1
        action_x = clamp(float(action[0]), -1.0, 1.0)
        action_y = clamp(float(action[1]), -1.0, 1.0)
        thrust_x = self.thrust_scale * action_x
        thrust_y = self.thrust_scale * action_y
        acc_thrust_x = thrust_x / self.mass
        acc_thrust_y = thrust_y / self.mass

        r = norm2(self.pos[0], self.pos[1]) + 1e-12
        acc_gravity_x = -self.mu * self.pos[0] / (r ** 3)
        acc_gravity_y = -self.mu * self.pos[1] / (r ** 3)

        self.vel[0] += (acc_gravity_x + acc_thrust_x) * self.dt
        self.vel[1] += (acc_gravity_y + acc_thrust_y) * self.dt
        self.pos[0] += self.vel[0] * self.dt
        self.pos[1] += self.vel[1] * self.dt

        r_now = norm2(self.pos[0], self.pos[1])
        v_now = norm2(self.vel[0], self.vel[1])
        v_target = math.sqrt(self.mu / self.target_radius)
        r_err_now = abs(r_now - self.target_radius) / (self.target_radius + 1e-12)
        v_err_now = abs(v_now - v_target) / (v_target + 1e-12)
        if r_now > 1e-12 and v_now > 1e-12:
            ur_x = self.pos[0] / r_now
            ur_y = self.pos[1] / r_now
            uv_x = self.vel[0] / v_now
            uv_y = self.vel[1] / v_now
            ang_abs_now = abs(ur_x * uv_x + ur_y * uv_y)
        else:
            ang_abs_now = 1.0

        radial_stall = (r_err_now < 0.10) and (v_err_now < 0.12) and (ang_abs_now > 0.85)
        if radial_stall:
            self.radial_stall_counter += 1
        else:
            self.radial_stall_counter = 0

        time_up = self.steps >= self.max_steps
        out_range = r_now > 2.5 * self.target_radius
        overspeed = v_now > 1.90 * v_target
        too_close = r_now < 0.35 * self.target_radius
        radial_stall_fail = self.radial_stall_counter >= 800

        terminated = bool(out_range or overspeed or too_close or radial_stall_fail)
        truncated = bool(time_up)
        return self.get_obs(), terminated, truncated


def detect_shutdown_step(
    action_norm: Sequence[float],
    r_error: Sequence[float],
    v_t_error: Sequence[float],
    target_radius: float,
    v_circ: float,
) -> int:
    low_action_threshold = 0.035
    unresolved_r_threshold = 0.002 * target_radius
    unresolved_vt_threshold = 0.01 * v_circ
    sustain = 20

    for idx in range(sustain - 1, len(action_norm)):
        window = action_norm[idx - sustain + 1 : idx + 1]
        sorted_window = sorted(window)
        median = sorted_window[len(sorted_window) // 2]
        if median >= low_action_threshold:
            continue
        if abs(r_error[idx]) <= unresolved_r_threshold and abs(v_t_error[idx]) <= unresolved_vt_threshold:
            continue
        return idx

    start = min(50, max(0, len(action_norm) - 1))
    min_idx = start
    min_val = action_norm[start] if len(action_norm) > start else 0.0
    for idx in range(start, len(action_norm)):
        if action_norm[idx] < min_val:
            min_val = action_norm[idx]
            min_idx = idx
    return min_idx


def build_obs_from_local_errors(
    target_radius: float,
    mu: float,
    r_error: float,
    v_r: float,
    v_t_error: float,
) -> List[float]:
    radius = target_radius + r_error
    pos_x = 0.0
    pos_y = radius
    r = abs(radius) + 1e-12
    r_hat_x = pos_x / r
    r_hat_y = pos_y / r
    t_hat_x = -r_hat_y
    t_hat_y = r_hat_x
    v_circ = math.sqrt(mu / target_radius)
    v_t = v_circ + v_t_error
    vel_x = v_r * r_hat_x + v_t * t_hat_x
    vel_y = v_r * r_hat_y + v_t * t_hat_y
    return [pos_x, pos_y, vel_x, vel_y, v_r]


def rollout_trace(policy: LoadedPolicy, env: LightweightOrbitEnv) -> Dict[str, List[float]]:
    obs = env.reset()
    action_norm: List[float] = []
    r_error: List[float] = []
    v_r_list: List[float] = []
    v_t_error: List[float] = []
    v_circ = math.sqrt(env.mu / env.target_radius)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy.act(obs)
        obs, terminated, truncated = env.step(action)
        action_norm.append(norm2(action[0], action[1]))
        r_error.append(norm2(env.pos[0], env.pos[1]) - env.target_radius)
        v_r_list.append(compute_vr(env.pos, env.vel))
        v_t_error.append(compute_vt(env.pos, env.vel) - v_circ)

    return {
        "action_norm": action_norm,
        "r_error": r_error,
        "v_r": v_r_list,
        "v_t_error": v_t_error,
    }


def detect_collapse_state(checkpoint_tag: str, checkpoint_path: Path, trace: Dict[str, List[float]], env: LightweightOrbitEnv) -> CollapseState:
    step = detect_shutdown_step(
        action_norm=trace["action_norm"],
        r_error=trace["r_error"],
        v_t_error=trace["v_t_error"],
        target_radius=env.target_radius,
        v_circ=math.sqrt(env.mu / env.target_radius),
    )
    return CollapseState(
        checkpoint_tag=checkpoint_tag,
        checkpoint_path=str(checkpoint_path),
        step=int(step),
        r_error=float(trace["r_error"][step]),
        v_r=float(trace["v_r"][step]),
        v_t_error=float(trace["v_t_error"][step]),
        action_norm=float(trace["action_norm"][step]),
        target_radius=float(env.target_radius),
        v_circ=float(math.sqrt(env.mu / env.target_radius)),
    )


def linspace(start: float, stop: float, num: int) -> List[float]:
    if num <= 1:
        return [float(start)]
    step = (stop - start) / float(num - 1)
    return [float(start + step * i) for i in range(num)]


def evaluate_surface(
    policy: LoadedPolicy,
    r_errors: Sequence[float],
    v_rs: Sequence[float],
    fixed_vt_error: float,
    target_radius: float,
    mu: float,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    action_norm: List[List[float]] = []
    action_x: List[List[float]] = []
    action_y: List[List[float]] = []
    for v_r in v_rs:
        row_norm: List[float] = []
        row_x: List[float] = []
        row_y: List[float] = []
        for r_error in r_errors:
            obs = build_obs_from_local_errors(target_radius, mu, r_error, v_r, fixed_vt_error)
            action = policy.act(obs)
            row_norm.append(norm2(action[0], action[1]))
            row_x.append(action[0])
            row_y.append(action[1])
        action_norm.append(row_norm)
        action_x.append(row_x)
        action_y.append(row_y)
    return action_norm, action_x, action_y


def pick_color(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
    t = clamp(t, 0.0, 1.0)
    r = int(30 + 225 * t)
    g = int(60 + 160 * (1.0 - abs(t - 0.5) * 2.0))
    b = int(180 + 60 * (1.0 - t))
    return f"rgb({r},{g},{b})"


def svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def save_heatmap_svg(
    path: Path,
    data: Sequence[Sequence[float]],
    x_values: Sequence[float],
    y_values: Sequence[float],
    title: str,
    colorbar_label: str,
) -> None:
    width, height = 840, 680
    left, top, plot_w, plot_h = 100, 80, 560, 460
    right = left + plot_w
    bottom = top + plot_h
    lines = svg_header(width, height)
    flat = [value for row in data for value in row]
    vmin = min(flat)
    vmax = max(flat)
    cols = max(1, len(x_values))
    rows = max(1, len(y_values))
    cell_w = plot_w / cols
    cell_h = plot_h / rows

    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            x = left + col_idx * cell_w
            y = bottom - (row_idx + 1) * cell_h
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w + 0.4:.2f}" height="{cell_h + 0.4:.2f}" fill="{pick_color(value, vmin, vmax)}" stroke="none"/>'
            )

    lines.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="black" stroke-width="1.5"/>')
    for idx in range(5):
        xt = idx / 4.0
        x = left + plot_w * xt
        value = x_values[0] + (x_values[-1] - x_values[0]) * xt
        lines.append(f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom + 6}" stroke="black"/>')
        lines.append(f'<text x="{x:.2f}" y="{bottom + 24}" font-size="12" text-anchor="middle">{value:.0f}</text>')
    for idx in range(5):
        yt = idx / 4.0
        y = bottom - plot_h * yt
        value = y_values[0] + (y_values[-1] - y_values[0]) * yt
        lines.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="black"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{value:.0f}</text>')

    cb_x = right + 35
    cb_y = top
    cb_h = plot_h
    cb_w = 30
    for idx in range(100):
        frac = idx / 99.0
        value = vmin + (vmax - vmin) * (1.0 - frac)
        y = cb_y + cb_h * frac
        lines.append(
            f'<rect x="{cb_x}" y="{y:.2f}" width="{cb_w}" height="{cb_h / 100.0 + 1:.2f}" fill="{pick_color(value, vmin, vmax)}" stroke="none"/>'
        )
    lines.append(f'<rect x="{cb_x}" y="{cb_y}" width="{cb_w}" height="{cb_h}" fill="none" stroke="black"/>')
    lines.append(f'<text x="{cb_x + cb_w / 2:.2f}" y="{cb_y - 12}" font-size="12" text-anchor="middle">{colorbar_label}</text>')
    lines.append(f'<text x="{cb_x + cb_w + 10}" y="{cb_y + 12}" font-size="12">{vmax:.3f}</text>')
    lines.append(f'<text x="{cb_x + cb_w + 10}" y="{cb_y + cb_h}" font-size="12">{vmin:.3f}</text>')

    lines.append(f'<text x="{width / 2:.2f}" y="32" font-size="20" text-anchor="middle">{title}</text>')
    lines.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 24}" font-size="15" text-anchor="middle">r_error [m]</text>')
    lines.append(f'<text x="28" y="{top + plot_h / 2:.2f}" font-size="15" text-anchor="middle" transform="rotate(-90 28 {top + plot_h / 2:.2f})">v_r [m/s]</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def series_bounds(series: Iterable[Sequence[float]]) -> Tuple[float, float]:
    values = [value for seq in series for value in seq]
    if not values:
        return (0.0, 1.0)
    ymin = min(values)
    ymax = max(values)
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0
    pad = 0.05 * (ymax - ymin)
    return (ymin - pad, ymax + pad)


def to_points(xs: Sequence[float], ys: Sequence[float], x0: float, x1: float, y0: float, y1: float, left: float, top: float, plot_w: float, plot_h: float) -> str:
    pts: List[str] = []
    for x, y in zip(xs, ys):
        px = left + (x - x0) / (x1 - x0 + 1e-12) * plot_w
        py = top + plot_h - (y - y0) / (y1 - y0 + 1e-12) * plot_h
        pts.append(f"{px:.2f},{py:.2f}")
    return " ".join(pts)


def save_line_plot_svg(
    path: Path,
    x_values: Sequence[float],
    series: Dict[str, Sequence[float]],
    title: str,
    y_label: str,
) -> None:
    width, height = 860, 620
    left, top, plot_w, plot_h = 100, 70, 660, 420
    bottom = top + plot_h
    x0 = min(x_values)
    x1 = max(x_values) if max(x_values) > min(x_values) else min(x_values) + 1.0
    y0, y1 = series_bounds(series.values())
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    lines = svg_header(width, height)
    lines.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="black" stroke-width="1.5"/>')

    if y0 <= 0.0 <= y1:
        zero_y = top + plot_h - (0.0 - y0) / (y1 - y0 + 1e-12) * plot_h
        lines.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_w}" y2="{zero_y:.2f}" stroke="#777" stroke-dasharray="6,4"/>')
    if x0 <= 0.0 <= x1:
        zero_x = left + (0.0 - x0) / (x1 - x0 + 1e-12) * plot_w
        lines.append(f'<line x1="{zero_x:.2f}" y1="{top}" x2="{zero_x:.2f}" y2="{bottom}" stroke="#777" stroke-dasharray="6,4"/>')

    for idx, (label, ys) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        points = to_points(x_values, ys, x0, x1, y0, y1, left, top, plot_w, plot_h)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_y = top + idx * 22
        legend_x = left + plot_w + 20
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 18}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 24}" y="{legend_y + 4}" font-size="12">{label}</text>')

    for idx in range(5):
        xt = idx / 4.0
        x = left + plot_w * xt
        value = x0 + (x1 - x0) * xt
        lines.append(f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom + 6}" stroke="black"/>')
        lines.append(f'<text x="{x:.2f}" y="{bottom + 24}" font-size="12" text-anchor="middle">{value:.0f}</text>')
    for idx in range(5):
        yt = idx / 4.0
        y = bottom - plot_h * yt
        value = y0 + (y1 - y0) * yt
        lines.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="black"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{value:.3f}</text>')

    lines.append(f'<text x="{width / 2:.2f}" y="32" font-size="20" text-anchor="middle">{title}</text>')
    lines.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 26}" font-size="15" text-anchor="middle">v_r [m/s]</text>')
    lines.append(f'<text x="28" y="{top + plot_h / 2:.2f}" font-size="15" text-anchor="middle" transform="rotate(-90 28 {top + plot_h / 2:.2f})">{y_label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 20 PPO policy response surface diagnostics.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--thrust-scale", type=float, default=20000.0)
    parser.add_argument("--r0-over-target", type=float, default=1.05)
    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--r-error-frac", type=float, default=0.01)
    parser.add_argument("--vr-max", type=float, default=1200.0)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    collapse_summary: Dict[str, Dict[str, float]] = {}
    compare_curves: Dict[str, Dict[str, List[float]]] = {}

    probe_env = LightweightOrbitEnv(
        thrust_scale=args.thrust_scale,
        r0_over_target=args.r0_over_target,
        max_steps=args.max_steps,
    )
    target_radius = probe_env.target_radius
    mu = probe_env.mu
    r_errors = linspace(-args.r_error_frac * target_radius, args.r_error_frac * target_radius, args.grid_size)
    v_rs = linspace(-args.vr_max, args.vr_max, args.grid_size)

    for checkpoint_tag, checkpoint_path in DEFAULT_CHECKPOINTS.items():
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        policy = LoadedPolicy(checkpoint_path)
        env = LightweightOrbitEnv(
            thrust_scale=args.thrust_scale,
            r0_over_target=args.r0_over_target,
            max_steps=args.max_steps,
        )
        trace = rollout_trace(policy, env)
        collapse = detect_collapse_state(checkpoint_tag, checkpoint_path, trace, env)
        collapse_summary[checkpoint_tag] = asdict(collapse)

        for vt_suffix, fixed_vt_error in [("collapse", collapse.v_t_error), ("near_target", 0.0)]:
            action_norm, action_x, action_y = evaluate_surface(policy, r_errors, v_rs, fixed_vt_error, target_radius, mu)
            save_heatmap_svg(
                args.output_dir / f"action_norm_heatmap_{checkpoint_tag}_{vt_suffix}.svg",
                action_norm,
                r_errors,
                v_rs,
                f"Day20 action_norm | {checkpoint_tag} | vt={vt_suffix}",
                "action_norm",
            )
            save_heatmap_svg(
                args.output_dir / f"action_x_heatmap_{checkpoint_tag}_{vt_suffix}.svg",
                action_x,
                r_errors,
                v_rs,
                f"Day20 action_x | {checkpoint_tag} | vt={vt_suffix}",
                "action_x",
            )
            save_heatmap_svg(
                args.output_dir / f"action_y_heatmap_{checkpoint_tag}_{vt_suffix}.svg",
                action_y,
                r_errors,
                v_rs,
                f"Day20 action_y | {checkpoint_tag} | vt={vt_suffix}",
                "action_y",
            )

        settings = [
            ("r=0,vt=0", (0.0, 0.0)),
            ("r=-0.5%,vt=0", (-0.005 * target_radius, 0.0)),
            ("r=0,vt=collapse", (0.0, collapse.v_t_error)),
            ("collapse state", (collapse.r_error, collapse.v_t_error)),
        ]
        curves: Dict[str, List[float]] = {}
        for label, (r_error, v_t_error) in settings:
            ys: List[float] = []
            for v_r in v_rs:
                obs = build_obs_from_local_errors(target_radius, mu, r_error, v_r, v_t_error)
                action = policy.act(obs)
                ys.append(norm2(action[0], action[1]))
            curves[label] = ys
        compare_curves[checkpoint_tag] = curves
        save_line_plot_svg(
            args.output_dir / f"action_norm_vs_vr_{checkpoint_tag}.svg",
            v_rs,
            curves,
            f"Day20 action_norm vs v_r | {checkpoint_tag}",
            "action_norm",
        )

    merged: Dict[str, List[float]] = {}
    for checkpoint_tag, lines in compare_curves.items():
        for label, values in lines.items():
            merged[f"{checkpoint_tag} | {label}"] = values
    save_line_plot_svg(
        args.output_dir / "action_norm_vs_vr_compare.svg",
        v_rs,
        merged,
        "Day20 action_norm vs v_r | checkpoint comparison",
        "action_norm",
    )
    save_json(args.output_dir / "collapse_states.json", collapse_summary)
    print(json.dumps(collapse_summary, indent=2))
    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
