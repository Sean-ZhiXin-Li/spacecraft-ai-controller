from __future__ import annotations

import argparse
import json
import math
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.collections import LineCollection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from envs.orbit_env import OrbitEnv


OUTPUT_DIR = PROJECT_ROOT / "analysis" / "demo"
GIF_PATH = OUTPUT_DIR / "orbit_demo.gif"
ZOOM_GIF_PATH = OUTPUT_DIR / "orbit_demo_zoom.gif"
PREVIEW_PATH = OUTPUT_DIR / "orbit_demo_preview.png"
TRAJ_PATH = OUTPUT_DIR / "orbit_demo_trajectory.png"
FULL_TRAJ_PATH = OUTPUT_DIR / "orbit_demo_full.png"
SUMMARY_PATH = OUTPUT_DIR / "orbit_demo_summary.json"

TRAJ_COLOR = "#1565C0"
TARGET_COLOR = "#333333"
START_COLOR = "#2E7D32"
END_COLOR = "#C62828"
CROSS_COLOR = "#6A1B9A"
CAPTURE_COLOR = "#FB8C00"
LOCK_COLOR = "#00897B"

BEST_SETUP = {
    "r0_over_target": 1.00005,
    "dt": 100.0,
    "max_steps": 100000,
    "thrust_scale": 10000.0,
    "tol_r": 1.0e-3,
    "tol_v": 1.0e-3,
    "tol_ang": 0.02,
    "success_threshold": 200,
}

MAIN_WINDOW_BEFORE = 15000
MAIN_WINDOW_AFTER = 25000
ZOOM_WINDOW_BEFORE = 5000
ZOOM_WINDOW_AFTER = 12000


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_env() -> OrbitEnv:
    return OrbitEnv(
        thrust_scale=BEST_SETUP["thrust_scale"],
        dt=BEST_SETUP["dt"],
        max_steps=BEST_SETUP["max_steps"],
        reward_mode="orbit_circular_minimal",
        tol_r=BEST_SETUP["tol_r"],
        tol_v=BEST_SETUP["tol_v"],
        tol_ang=BEST_SETUP["tol_ang"],
        success_threshold=BEST_SETUP["success_threshold"],
        use_action_smoothing=False,
        use_orbit_capture_assist=False,
        verbose=False,
    )


def set_default_start(env: OrbitEnv, r0_over_target: float) -> None:
    env.reset(start_mode="default")
    env.steps = 0
    env.success_counter = 0
    env.radial_stall_counter = 0
    env.orbit_lock_counter = 0
    env.prev_action = np.zeros(2, dtype=np.float64)
    env.pos = np.array([0.0, r0_over_target * env.target_radius], dtype=np.float64)
    v_mag = math.sqrt(env.mu / np.linalg.norm(env.pos))
    angle = math.radians(170.0)
    env.vel = v_mag * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)


def run_rollout() -> tuple[Dict[str, float | int | bool | None], Dict[str, np.ndarray]]:
    env = make_env()
    controller = OrbitLockController(target_radius=env.target_radius, mu=env.mu, config=OrbitLockConfig())
    set_default_start(env, BEST_SETUP["r0_over_target"])
    obs = env._get_obs()

    xs: List[float] = []
    ys: List[float] = []
    radii: List[float] = []
    vrs: List[float] = []
    phases: List[str] = []
    successes: List[bool] = []

    success = False
    terminated = False
    truncated = False
    first_crossing_step: int | None = None
    prev_r_error: float | None = None
    crossings = 0

    for step in range(BEST_SETUP["max_steps"]):
        action = np.asarray(controller.act(obs), dtype=np.float64)
        obs, _, terminated, truncated, info = env.step(action)

        radius = float(np.linalg.norm(env.pos))
        r_hat = env.pos / (radius + 1e-12)
        v_r = float(np.dot(env.vel, r_hat))
        r_error = radius - env.target_radius

        if prev_r_error is not None and np.signbit(prev_r_error) != np.signbit(r_error):
            crossings += 1
            if first_crossing_step is None:
                first_crossing_step = step
        prev_r_error = r_error

        xs.append(float(env.pos[0]))
        ys.append(float(env.pos[1]))
        radii.append(radius)
        vrs.append(v_r)
        phases.append(str(controller.last_info.get("phase", controller.phase)))
        success = success or bool(info.get("success", False))
        successes.append(success)
        if terminated or truncated:
            break

    trace = {
        "x": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
        "radius": np.asarray(radii, dtype=np.float64),
        "v_r": np.asarray(vrs, dtype=np.float64),
        "phase": np.asarray(phases, dtype=object),
        "success": np.asarray(successes, dtype=bool),
    }
    metrics = {
        "success": bool(success),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "steps": int(len(xs)),
        "radius_crossings_total": int(crossings),
        "first_crossing_step": first_crossing_step,
        "final_radius_error": float(abs(radii[-1] - env.target_radius)) if radii else 0.0,
        "target_radius": float(env.target_radius),
        "phase_transition_count": int(len(controller.phase_transitions)),
        "main_demo_window": {
            "before": MAIN_WINDOW_BEFORE,
            "after": MAIN_WINDOW_AFTER,
        },
        "phase_transitions": [
            {
                "from": old,
                "to": new,
                "r_error": float(r_error),
                "v_r": float(v_r),
                "v_t_error": float(v_t_error),
            }
            for old, new, r_error, v_r, v_t_error in controller.phase_transitions
        ],
        "setup": BEST_SETUP,
    }
    return metrics, trace


def _transition_indices(metrics: Dict[str, float | int | bool | None], trace: Dict[str, np.ndarray]) -> tuple[int | None, int | None]:
    capture_idx = None
    lock_idx = None
    transitions = metrics.get("phase_transitions", [])
    if not isinstance(transitions, list):
        return capture_idx, lock_idx
    phase = trace["phase"]
    for item in transitions:
        if not isinstance(item, dict):
            continue
        to_phase = item.get("to")
        target_idx = next((i for i, p in enumerate(phase) if p == to_phase), None)
        if to_phase == "CAPTURE" and capture_idx is None:
            capture_idx = target_idx
        if to_phase == "LOCK" and lock_idx is None:
            lock_idx = target_idx
    return capture_idx, lock_idx


def _window_indices(trace: Dict[str, np.ndarray], first_crossing_step: int | None, before: int, after: int) -> np.ndarray:
    total_points = len(trace["x"])
    if total_points == 0:
        raise RuntimeError("Rollout produced no trajectory points.")
    if first_crossing_step is None:
        lo = 0
        hi = min(total_points - 1, before + after)
    else:
        lo = max(0, first_crossing_step - before)
        hi = min(total_points - 1, first_crossing_step + after)
    return np.arange(lo, hi + 1, dtype=int)


def _zoom_limits(
    trace: Dict[str, np.ndarray],
    window_idx: np.ndarray,
    *,
    extra_indices: list[int | None],
    min_span: float,
    padding_ratio: float,
) -> tuple[float, float, float, float]:
    include = list(window_idx.astype(int))
    for idx in extra_indices:
        if idx is not None:
            include.append(int(idx))
    include_idx = np.asarray(sorted(set(include)), dtype=int)

    xw = trace["x"][include_idx]
    yw = trace["y"][include_idx]

    x_min = float(np.min(xw))
    x_max = float(np.max(xw))
    y_min = float(np.min(yw))
    y_max = float(np.max(yw))

    x_span = max(x_max - x_min, min_span)
    y_span = max(y_max - y_min, min_span)
    x_pad = x_span * padding_ratio
    y_pad = y_span * padding_ratio
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _plot_target_arc(ax, target_radius: float, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, 2048)
    circle_x = target_radius * np.cos(theta)
    circle_y = target_radius * np.sin(theta)
    mask = (
        (circle_x >= xlim[0] - 0.05 * (xlim[1] - xlim[0]))
        & (circle_x <= xlim[1] + 0.05 * (xlim[1] - xlim[0]))
        & (circle_y >= ylim[0] - 0.05 * (ylim[1] - ylim[0]))
        & (circle_y <= ylim[1] + 0.05 * (ylim[1] - ylim[0]))
    )
    ax.plot(circle_x[mask], circle_y[mask], "--", linewidth=2.0, color=TARGET_COLOR, alpha=0.35, label="target orbit", zorder=1)


def save_static_plots(metrics: Dict[str, float | int | bool | None], trace: Dict[str, np.ndarray]) -> None:
    target_radius = float(metrics["target_radius"])
    theta = np.linspace(0.0, 2.0 * math.pi, 512)
    circle_x = target_radius * np.cos(theta)
    circle_y = target_radius * np.sin(theta)
    first_crossing_step = metrics.get("first_crossing_step")
    first_crossing_step = int(first_crossing_step) if isinstance(first_crossing_step, int) else None
    capture_idx, lock_idx = _transition_indices(metrics, trace)
    window_idx = _window_indices(trace, first_crossing_step, before=MAIN_WINDOW_BEFORE, after=MAIN_WINDOW_AFTER)
    xmin, xmax, ymin, ymax = _zoom_limits(
        trace,
        window_idx,
        extra_indices=[first_crossing_step, capture_idx, lock_idx],
        min_span=target_radius * 8e-5,
        padding_ratio=0.18,
    )

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    _plot_target_arc(ax, target_radius, (xmin, xmax), (ymin, ymax))

    points = np.column_stack((trace["x"][window_idx], trace["y"][window_idx]))
    segments = np.stack([points[:-1], points[1:]], axis=1)
    color_progress = np.linspace(0.15, 1.0, max(1, len(segments)))
    collection = LineCollection(
        segments,
        cmap="Blues",
        array=color_progress,
        linewidths=3.0,
        alpha=0.95,
        zorder=3,
    )
    ax.add_collection(collection)
    ax.plot([], [], color=TRAJ_COLOR, linewidth=3.0, label="explicit controller")

    start_idx = int(window_idx[0])
    end_idx = int(window_idx[-1])
    ax.scatter(trace["x"][start_idx], trace["y"][start_idx], s=140, color=START_COLOR, edgecolors="white", linewidths=1.4, label="window start", zorder=12)
    ax.scatter(trace["x"][end_idx], trace["y"][end_idx], s=140, color=END_COLOR, edgecolors="white", linewidths=1.4, label="window end", zorder=13)
    if first_crossing_step is not None:
        ax.scatter(trace["x"][first_crossing_step], trace["y"][first_crossing_step], s=150, color=CROSS_COLOR, edgecolors="white", linewidths=1.4, label="first crossing", zorder=14)
    if capture_idx is not None:
        ax.scatter(trace["x"][capture_idx], trace["y"][capture_idx], s=130, marker="s", color=CAPTURE_COLOR, edgecolors="white", linewidths=1.2, label="CAPTURE", zorder=15)
    if lock_idx is not None:
        ax.scatter(trace["x"][lock_idx], trace["y"][lock_idx], s=130, marker="D", color=LOCK_COLOR, edgecolors="white", linewidths=1.2, label="LOCK", zorder=16)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Orbit insertion event | explicit phase controller")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(TRAJ_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(circle_x, circle_y, "--", linewidth=2.0, color=TARGET_COLOR, alpha=0.35, label="target orbit", zorder=1)
    ax.plot(trace["x"], trace["y"], linewidth=3.0, color=TRAJ_COLOR, alpha=0.95, label="explicit controller", zorder=3)
    ax.scatter(trace["x"][0], trace["y"][0], s=130, color=START_COLOR, edgecolors="white", linewidths=1.2, label="start", zorder=12)
    ax.scatter(trace["x"][-1], trace["y"][-1], s=130, color=END_COLOR, edgecolors="white", linewidths=1.2, label="end", zorder=13)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Orbit insertion trajectory | full reference")
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FULL_TRAJ_PATH, dpi=180)
    plt.close(fig)


def _render_gif(
    path: Path,
    title: str,
    metrics: Dict[str, float | int | bool | None],
    trace: Dict[str, np.ndarray],
    *,
    before: int,
    after: int,
    padding_scale: float,
    frame_count: int,
    duration_ms: int,
) -> None:
    target_radius = float(metrics["target_radius"])
    total_points = len(trace["x"])
    if total_points == 0:
        raise RuntimeError("Rollout produced no trajectory points.")
    first_crossing_step = metrics.get("first_crossing_step")
    first_crossing_step = int(first_crossing_step) if isinstance(first_crossing_step, int) else None
    capture_idx, lock_idx = _transition_indices(metrics, trace)
    window_idx = _window_indices(trace, first_crossing_step, before=before, after=after)
    xmin, xmax, ymin, ymax = _zoom_limits(
        trace,
        window_idx,
        extra_indices=[first_crossing_step, capture_idx, lock_idx],
        min_span=target_radius * 8e-5,
        padding_ratio=0.12 if padding_scale <= 1.3 else 0.18,
    )

    if len(window_idx) > frame_count:
        frame_indices = np.linspace(window_idx[0], window_idx[-1], frame_count, dtype=int)
    else:
        frame_indices = window_idx

    frames: List[Image.Image] = []
    for idx in frame_indices:
        fig, ax = plt.subplots(figsize=(6.6, 6.6))
        _plot_target_arc(ax, target_radius, (xmin, xmax), (ymin, ymax))
        ax.plot(trace["x"][window_idx], trace["y"][window_idx], linewidth=1.8, color=TRAJ_COLOR, alpha=0.16, zorder=2)
        active_start = int(window_idx[0])
        ax.plot(trace["x"][active_start : idx + 1], trace["y"][active_start : idx + 1], linewidth=3.2, color=TRAJ_COLOR, alpha=0.98, label="explicit controller", zorder=4)
        ax.scatter(trace["x"][active_start], trace["y"][active_start], s=130, color=START_COLOR, edgecolors="white", linewidths=1.2, zorder=11, label="window start")
        if first_crossing_step is not None:
            ax.scatter(trace["x"][first_crossing_step], trace["y"][first_crossing_step], s=150, color=CROSS_COLOR, edgecolors="white", linewidths=1.4, zorder=13, label="first crossing")
        if capture_idx is not None:
            ax.scatter(trace["x"][capture_idx], trace["y"][capture_idx], s=130, marker="s", color=CAPTURE_COLOR, edgecolors="white", linewidths=1.2, zorder=14, label="CAPTURE")
        if lock_idx is not None:
            ax.scatter(trace["x"][lock_idx], trace["y"][lock_idx], s=130, marker="D", color=LOCK_COLOR, edgecolors="white", linewidths=1.2, zorder=15, label="LOCK")
        ax.scatter(trace["x"][idx], trace["y"][idx], s=150, color=END_COLOR, edgecolors="white", linewidths=1.3, zorder=16, label="vehicle")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.text(
            0.02,
            0.02,
            (
                f"step={idx}\n"
                f"phase={trace['phase'][idx]}\n"
                f"r_error={trace['radius'][idx] - target_radius:.3e} m\n"
                f"v_r={trace['v_r'][idx]:.3f} m/s"
            ),
            transform=ax.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P"))

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def render_animation(metrics: Dict[str, float | int | bool | None], trace: Dict[str, np.ndarray], frame_count: int = 75) -> None:
    _render_gif(
        GIF_PATH,
        "Orbit insertion process | medium-context explicit-controller demo",
        metrics,
        trace,
        before=MAIN_WINDOW_BEFORE,
        after=MAIN_WINDOW_AFTER,
        padding_scale=1.55,
        frame_count=min(frame_count, 85),
        duration_ms=20,
    )
    _render_gif(
        ZOOM_GIF_PATH,
        "Orbit insertion event | zoomed explicit-controller demo",
        metrics,
        trace,
        before=ZOOM_WINDOW_BEFORE,
        after=ZOOM_WINDOW_AFTER,
        padding_scale=1.25,
        frame_count=min(frame_count, 65),
        duration_ms=18,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the explicit-controller orbit demo assets.")
    parser.add_argument("--frame-count", type=int, default=75, help="number of GIF frames")
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)
    metrics, trace = run_rollout()
    save_static_plots(metrics, trace)
    render_animation(metrics, trace, frame_count=args.frame_count)
    SUMMARY_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved GIF: {GIF_PATH}")
    print(f"Saved zoom GIF: {ZOOM_GIF_PATH}")
    print(f"Saved zoomed trajectory plot: {TRAJ_PATH}")
    print(f"Saved full reference plot: {FULL_TRAJ_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
