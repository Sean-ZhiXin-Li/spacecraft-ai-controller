from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from controller.ppo_anti_shutdown_wrapper import AntiShutdownConfig, PPOAntiShutdownWrapper
from scripts.day20_policy_surface import (
    DEFAULT_CHECKPOINTS,
    LoadedPolicy,
    LightweightOrbitEnv,
    compute_vr,
    compute_vt,
    detect_shutdown_step,
    ensure_dir,
)

OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figs" / "day23_orbit_lock"
SUMMARY_PATH = PROJECT_ROOT / "analysis" / "day23_summary.md"
CSV_PATH = OUTPUT_DIR / "aggregate_results.csv"
JSON_PATH = OUTPUT_DIR / "aggregate_results.json"


@dataclass
class EvalCase:
    label: str
    r0_over_target: float
    vr_nudge: float = 0.0


def norm2(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def apply_initial_vr_nudge(env: LightweightOrbitEnv, delta_vr: float) -> None:
    if abs(delta_vr) < 1e-12:
        return
    r = math.sqrt(env.pos[0] * env.pos[0] + env.pos[1] * env.pos[1]) + 1e-12
    r_hat_x = env.pos[0] / r
    r_hat_y = env.pos[1] / r
    env.vel[0] += delta_vr * r_hat_x
    env.vel[1] += delta_vr * r_hat_y


def tail_slice(length: int, tail_len: int = 1000) -> slice:
    use_len = min(tail_len, length)
    return slice(length - use_len, length)


def rollout_trace(controller, env: LightweightOrbitEnv, *, max_steps: int, apply_nudge: float = 0.0) -> Dict[str, object]:
    obs = env.reset()
    apply_initial_vr_nudge(env, apply_nudge)
    obs = env.get_obs()
    v_circ = math.sqrt(env.mu / env.target_radius)

    trace: Dict[str, List[float]] = {
        "step": [],
        "radius": [],
        "r_error": [],
        "v_r": [],
        "v_t_error": [],
        "alignment": [],
        "speed": [],
        "action_norm": [],
        "impulse_fired": [],
        "impulse_radius_delta": [],
    }

    terminated = False
    truncated = False
    while not (terminated or truncated) and len(trace["step"]) < max_steps:
        pre_radius = math.sqrt(env.pos[0] * env.pos[0] + env.pos[1] * env.pos[1])
        if hasattr(controller, "act_with_info"):
            info = controller.act_with_info(obs)
            action = [float(info["final_action"][0]), float(info["final_action"][1])]
            impulse_fired = bool(info.get("impulse_fired", False))
        else:
            action = [float(v) for v in controller.act(obs)]
            impulse_fired = False
        action_norm = norm2(action[0], action[1])
        obs, terminated, truncated = env.step(action)

        radius = math.sqrt(env.pos[0] * env.pos[0] + env.pos[1] * env.pos[1])
        r_error = radius - env.target_radius
        v_r = compute_vr(env.pos, env.vel)
        v_t_error = compute_vt(env.pos, env.vel) - v_circ
        speed = math.sqrt(env.vel[0] * env.vel[0] + env.vel[1] * env.vel[1])

        if action_norm > 1e-12:
            r_hat_x = env.pos[0] / (radius + 1e-12)
            r_hat_y = env.pos[1] / (radius + 1e-12)
            a_hat_x = action[0] / action_norm
            a_hat_y = action[1] / action_norm
            alignment = abs(a_hat_x * r_hat_x + a_hat_y * r_hat_y)
        else:
            alignment = 0.0

        trace["step"].append(float(len(trace["step"])))
        trace["radius"].append(radius)
        trace["r_error"].append(r_error)
        trace["v_r"].append(v_r)
        trace["v_t_error"].append(v_t_error)
        trace["alignment"].append(alignment)
        trace["speed"].append(speed)
        trace["action_norm"].append(action_norm)
        trace["impulse_fired"].append(1.0 if impulse_fired else 0.0)
        trace["impulse_radius_delta"].append(radius - pre_radius if impulse_fired else 0.0)

    trace["num_steps"] = len(trace["step"])
    trace["terminated"] = terminated
    trace["truncated"] = truncated
    trace["target_radius"] = env.target_radius
    trace["v_circ"] = v_circ
    return trace


def summarize_trace(trace: Dict[str, object]) -> Dict[str, object]:
    action_norm = [float(v) for v in trace["action_norm"]]
    r_error = [float(v) for v in trace["r_error"]]
    v_t_error = [float(v) for v in trace["v_t_error"]]
    radius = np.asarray(trace["radius"], dtype=np.float64)
    v_r = np.asarray(trace["v_r"], dtype=np.float64)
    alignment = np.asarray(trace["alignment"], dtype=np.float64)
    speed = np.asarray(trace["speed"], dtype=np.float64)
    impulse_fired = np.asarray(trace.get("impulse_fired", []), dtype=np.float64)
    impulse_radius_delta = np.asarray(trace.get("impulse_radius_delta", []), dtype=np.float64)
    target_radius = float(trace["target_radius"])
    t_slice = tail_slice(len(radius))
    re_tail = radius[t_slice] - target_radius
    vr_tail = v_r[t_slice]
    shutdown_step = detect_shutdown_step(
        action_norm=action_norm,
        r_error=r_error,
        v_t_error=v_t_error,
        target_radius=target_radius,
        v_circ=float(trace["v_circ"]),
    )
    crossings_full = int(np.sum(np.signbit((radius - target_radius)[1:]) != np.signbit((radius - target_radius)[:-1]))) if len(radius) > 1 else 0
    impulse_trigger_count = int(np.sum(impulse_fired > 0.5)) if len(impulse_fired) else 0
    immediate_delta_radius_after_impulse = float(np.sum(impulse_radius_delta)) if len(impulse_radius_delta) else 0.0
    return {
        "num_steps": int(trace["num_steps"]),
        "terminated": bool(trace["terminated"]),
        "truncated": bool(trace["truncated"]),
        "shutdown_step": int(shutdown_step),
        "shutdown_occurred": bool(action_norm and shutdown_step < len(action_norm) - 1),
        "mean_action_norm": float(np.mean(action_norm)) if action_norm else 0.0,
        "final_radius_error": float(abs(radius[-1] - target_radius)) if len(radius) else 0.0,
        "tail_mean_abs_vr": float(np.mean(np.abs(vr_tail))) if len(vr_tail) else 0.0,
        "tail_avg_radius_error": float(np.mean(np.abs(re_tail))) if len(re_tail) else 0.0,
        "tail_radius_std": float(np.std(radius[t_slice])) if len(radius) else 0.0,
        "tail_mean_alignment": float(np.mean(alignment[t_slice])) if len(alignment) else 0.0,
        "tail_speed_cv": float(np.std(speed[t_slice]) / (np.mean(speed[t_slice]) + 1e-12)) if len(speed) else 0.0,
        "tail_vr_sign_changes": bool(np.any(np.signbit(vr_tail[1:]) != np.signbit(vr_tail[:-1]))) if len(vr_tail) > 1 else False,
        "tail_crosses_target_radius": bool(np.any(np.signbit(re_tail[1:]) != np.signbit(re_tail[:-1]))) if len(re_tail) > 1 else False,
        "radius_crossings_total": crossings_full,
        "impulse_trigger_count": impulse_trigger_count,
        "immediate_delta_radius_after_impulse": immediate_delta_radius_after_impulse,
        "amplitude_shrinks": bool(np.std((radius - target_radius)[len(radius)//2:]) < np.std((radius - target_radius)[:len(radius)//2])) if len(radius) >= 20 else False,
        "orbit_lock_like": bool(
            len(vr_tail) > 1
            and np.any(np.signbit(vr_tail[1:]) != np.signbit(vr_tail[:-1]))
            and np.any(np.signbit(re_tail[1:]) != np.signbit(re_tail[:-1]))
            and (np.std((radius - target_radius)[len(radius)//2:]) < np.std((radius - target_radius)[:len(radius)//2]))
        ),
    }


def save_timeseries_plot(path: Path, title: str, ylabel: str, traces: Dict[str, Dict[str, object]], key: str) -> None:
    plt.figure(figsize=(10, 4.8))
    for label, trace in traces.items():
        plt.plot(trace["step"], trace[key], label=label, linewidth=1.5)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_summary_chart(path: Path, checkpoint_tag: str, rows: List[Dict[str, object]]) -> None:
    variants = ["baseline", "wrapped", "explicit"]
    cases = [row["case_label"] for row in rows if row["variant"] == "baseline"]
    x = np.arange(len(cases))
    w = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for idx, variant in enumerate(variants):
        vals = [row["tail_mean_abs_vr"] for row in rows if row["variant"] == variant]
        axes[0].bar(x + (idx - 1) * w, vals, width=w, label=variant)
    axes[0].set_xticks(x, cases, rotation=20)
    axes[0].set_title("tail_mean_abs_vr")
    axes[0].legend()

    for idx, variant in enumerate(variants):
        vals = [row["final_radius_error"] for row in rows if row["variant"] == variant]
        axes[1].bar(x + (idx - 1) * w, vals, width=w, label=variant)
    axes[1].set_xticks(x, cases, rotation=20)
    axes[1].set_title("final_radius_error")
    axes[1].legend()

    fig.suptitle(f"Day23 orbit-lock summary | {checkpoint_tag}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def classify_vs_wrapper(wrapper_row: Dict[str, object], explicit_row: Dict[str, object]) -> str:
    if (
        explicit_row["tail_vr_sign_changes"]
        and explicit_row["final_radius_error"] <= wrapper_row["final_radius_error"]
        and explicit_row["tail_mean_abs_vr"] <= wrapper_row["tail_mean_abs_vr"]
    ):
        return "true_mechanism_improvement"
    if explicit_row["mean_action_norm"] > wrapper_row["mean_action_norm"]:
        return "forced_activity_only"
    return "no_gain"


def run_eval(args: argparse.Namespace) -> None:
    ensure_dir(OUTPUT_DIR)
    checkpoint_specs = {
        "speed_refine_50": DEFAULT_CHECKPOINTS["speed_refine_50"],
        "state_vr_nonlinear_100": DEFAULT_CHECKPOINTS["state_vr_nonlinear_100"],
    }
    cases = [
        EvalCase("r0_1.01", 1.01, 0.0),
        EvalCase("r0_1.03", 1.03, 0.0),
        EvalCase("r0_1.05", 1.05, 0.0),
        EvalCase("r0_1.05_vr_plus20", 1.05, 20.0),
        EvalCase("r0_1.05_vr_minus20", 1.05, -20.0),
    ]
    anti_config = AntiShutdownConfig()
    explicit_config = OrbitLockConfig()
    aggregate_rows: List[Dict[str, object]] = []
    figure_paths: List[str] = []
    summary: Dict[str, object] = {
        "anti_shutdown_config": anti_config.to_dict(),
        "orbit_lock_config": explicit_config.to_dict(),
        "checkpoints": {},
    }

    for checkpoint_tag, checkpoint_path in checkpoint_specs.items():
        base_policy = LoadedPolicy(checkpoint_path)
        checkpoint_rows: List[Dict[str, object]] = []
        nominal_traces: Dict[str, Dict[str, object]] = {}
        summary["checkpoints"][checkpoint_tag] = {"checkpoint_path": str(checkpoint_path), "cases": []}

        for case in cases:
            env_template = LightweightOrbitEnv(
                thrust_scale=args.thrust_scale,
                r0_over_target=case.r0_over_target,
                max_steps=args.max_steps,
            )
            controllers = {
                "baseline": base_policy,
                "wrapped": PPOAntiShutdownWrapper(
                    base_policy,
                    target_radius=env_template.target_radius,
                    mu=env_template.mu,
                    config=anti_config,
                ),
                "explicit": OrbitLockController(
                    target_radius=env_template.target_radius,
                    mu=env_template.mu,
                    config=explicit_config,
                ),
            }

            for variant, controller in controllers.items():
                env = LightweightOrbitEnv(
                    thrust_scale=args.thrust_scale,
                    r0_over_target=case.r0_over_target,
                    max_steps=args.max_steps,
                )
                trace = rollout_trace(controller, env, max_steps=args.max_steps, apply_nudge=case.vr_nudge)
                metrics = summarize_trace(trace)
                row = {
                    "checkpoint_tag": checkpoint_tag,
                    "checkpoint_path": str(checkpoint_path),
                    "case_label": case.label,
                    "variant": variant,
                    **metrics,
                }
                checkpoint_rows.append(row)
                aggregate_rows.append(row)
                summary["checkpoints"][checkpoint_tag]["cases"].append(row)
                if case.label == "r0_1.05":
                    nominal_traces[variant] = trace

        for case in cases:
            wrap_row = next(row for row in checkpoint_rows if row["case_label"] == case.label and row["variant"] == "wrapped")
            explicit_row = next(row for row in checkpoint_rows if row["case_label"] == case.label and row["variant"] == "explicit")
            explicit_row["effect_vs_wrapper"] = classify_vs_wrapper(wrap_row, explicit_row)

        radius_path = OUTPUT_DIR / f"{checkpoint_tag}_radius_vs_time.svg"
        vr_path = OUTPUT_DIR / f"{checkpoint_tag}_v_r_vs_time.svg"
        action_path = OUTPUT_DIR / f"{checkpoint_tag}_action_norm_vs_time.svg"
        summary_path = OUTPUT_DIR / f"{checkpoint_tag}_summary.svg"
        save_timeseries_plot(radius_path, f"{checkpoint_tag} | radius comparison", "radius [m]", nominal_traces, "radius")
        save_timeseries_plot(vr_path, f"{checkpoint_tag} | v_r comparison", "v_r [m/s]", nominal_traces, "v_r")
        save_timeseries_plot(action_path, f"{checkpoint_tag} | action_norm comparison", "action_norm", nominal_traces, "action_norm")
        save_summary_chart(summary_path, checkpoint_tag, checkpoint_rows)
        figure_paths.extend([str(radius_path), str(vr_path), str(action_path), str(summary_path)])

    write_csv(CSV_PATH, aggregate_rows)
    JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    diagnosis_lines: List[str] = []
    for checkpoint_tag in checkpoint_specs:
        rows = [row for row in aggregate_rows if row["checkpoint_tag"] == checkpoint_tag]
        explicit_rows = [row for row in rows if row["variant"] == "explicit"]
        wrapper_rows = [row for row in rows if row["variant"] == "wrapped"]
        sign_change_cases = sum(1 for row in explicit_rows if row["tail_vr_sign_changes"])
        better_cases = 0
        for explicit_row in explicit_rows:
            wrap_row = next(row for row in wrapper_rows if row["case_label"] == explicit_row["case_label"])
            if (
                explicit_row["final_radius_error"] <= wrap_row["final_radius_error"]
                and explicit_row["tail_mean_abs_vr"] <= wrap_row["tail_mean_abs_vr"]
            ):
                better_cases += 1
        if sign_change_cases > 0 and better_cases > 0:
            diagnosis = "The explicit controller showed orbit-lock-like correction on at least part of the sweep."
        elif sign_change_cases > 0:
            diagnosis = "The explicit controller produced sign-changing v_r, but not a clean metric win over the wrapper."
        else:
            diagnosis = "The explicit controller did not reach true orbit-lock-like behavior in this sweep."
        diagnosis_lines.append(f"- `{checkpoint_tag}`: {diagnosis}")

    lines = [
        "# Day 23 Orbit-Lock Prototype",
        "",
        "## Added Files",
        "",
        "- `controller/orbit_lock_controller.py`",
        "- `scripts/day23_orbit_lock_eval.py`",
        "",
        "## Controller Design",
        "",
        "The explicit controller is a small interpretable PD-like mechanism in radial/tangential coordinates. It uses `r_error`, `v_r`, and `v_t_error`, applies bounded radial and tangential corrections, and only scales down near a tight lock region instead of shutting off early.",
        "",
        "## Gains",
        "",
    ]
    for key, value in explicit_config.to_dict().items():
        lines.append(f"- `{key}` = `{value}`")
    lines.extend([
        "",
        "## Figures Generated",
        "",
    ])
    for path in figure_paths:
        lines.append(f"- `{Path(path).as_posix()}`")
    lines.extend([
        "",
        "## Aggregate Outputs",
        "",
        f"- `{CSV_PATH.as_posix()}`",
        f"- `{JSON_PATH.as_posix()}`",
        "",
        "## Main Answers",
        "",
    ])
    lines.extend(diagnosis_lines)
    lines.extend([
        "",
        "## One-Sentence Diagnosis",
        "",
        "Day23 is only a true mechanism improvement if the explicit controller produces controlled sign-changing `v_r` together with better orbit metrics than the Day21/22 wrapper.",
        "",
        "## Day24 Focus",
        "",
        "Tune the explicit radial/tangential gains and near-lock damping to turn sign-changing correction into controlled shrinking oscillation rather than drift or persistent forced activity.",
        "",
    ])
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 23 explicit orbit-lock controller evaluation.")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--thrust-scale", type=float, default=20000.0)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
