from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

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

OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figs" / "day21_day22_anti_shutdown"
SUMMARY_PATH = PROJECT_ROOT / "analysis" / "day21_day22_summary.md"
CSV_PATH = OUTPUT_DIR / "aggregate_results.csv"
JSON_PATH = OUTPUT_DIR / "aggregate_results.json"


@dataclass
class EvalCase:
    label: str
    r0_over_target: float
    vr_nudge: float = 0.0


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


def norm2(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def rollout_trace(policy, env: LightweightOrbitEnv, *, max_steps: int, apply_nudge: float = 0.0) -> Dict[str, object]:
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
        "raw_action_norm": [],
        "final_action_norm": [],
        "rescue_action_norm": [],
        "intervene": [],
    }

    terminated = False
    truncated = False
    while not (terminated or truncated) and len(trace["step"]) < max_steps:
        if hasattr(policy, "act_with_info"):
            info = policy.act_with_info(obs)
            action = [float(info["final_action"][0]), float(info["final_action"][1])]
            raw_action = [float(info["raw_action"][0]), float(info["raw_action"][1])]
            raw_action_norm = float(info["raw_action_norm"])
            final_action_norm = float(info["final_action_norm"])
            rescue_action_norm = float(info["rescue_action_norm"])
            intervene = bool(info["intervene"])
        else:
            action = [float(v) for v in policy.act(obs)]
            raw_action = list(action)
            raw_action_norm = norm2(raw_action[0], raw_action[1])
            final_action_norm = raw_action_norm
            rescue_action_norm = 0.0
            intervene = False

        obs, terminated, truncated = env.step(action)

        radius = math.sqrt(env.pos[0] * env.pos[0] + env.pos[1] * env.pos[1])
        speed = math.sqrt(env.vel[0] * env.vel[0] + env.vel[1] * env.vel[1])
        r_error = radius - env.target_radius
        v_r = compute_vr(env.pos, env.vel)
        v_t_error = compute_vt(env.pos, env.vel) - v_circ

        if final_action_norm > 1e-12:
            r_hat_x = env.pos[0] / (radius + 1e-12)
            r_hat_y = env.pos[1] / (radius + 1e-12)
            a_hat_x = action[0] / final_action_norm
            a_hat_y = action[1] / final_action_norm
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
        trace["raw_action_norm"].append(raw_action_norm)
        trace["final_action_norm"].append(final_action_norm)
        trace["rescue_action_norm"].append(rescue_action_norm)
        trace["intervene"].append(1.0 if intervene else 0.0)

    trace["num_steps"] = len(trace["step"])
    trace["terminated"] = terminated
    trace["truncated"] = truncated
    trace["target_radius"] = env.target_radius
    trace["v_circ"] = v_circ
    return trace


def summarize_trace(trace: Dict[str, object], *, shutdown_threshold_action: str) -> Dict[str, object]:
    action_key = shutdown_threshold_action
    action_norm = [float(v) for v in trace[action_key]]
    r_error = [float(v) for v in trace["r_error"]]
    v_t_error = [float(v) for v in trace["v_t_error"]]
    v_r = np.asarray(trace["v_r"], dtype=np.float64)
    radius = np.asarray(trace["radius"], dtype=np.float64)
    speed = np.asarray(trace["speed"], dtype=np.float64)
    alignment = np.asarray(trace["alignment"], dtype=np.float64)
    final_action_norm = np.asarray(trace["final_action_norm"], dtype=np.float64)
    raw_action_norm = np.asarray(trace["raw_action_norm"], dtype=np.float64)
    rescue_action_norm = np.asarray(trace["rescue_action_norm"], dtype=np.float64)
    intervene = np.asarray(trace["intervene"], dtype=np.float64)
    t_slice = tail_slice(len(radius))

    shutdown_step = detect_shutdown_step(
        action_norm=action_norm,
        r_error=r_error,
        v_t_error=v_t_error,
        target_radius=float(trace["target_radius"]),
        v_circ=float(trace["v_circ"]),
    )
    shutdown_occured = bool(action_norm and shutdown_step < len(action_norm) - 1)
    collapse_region = slice(shutdown_step, min(len(action_norm), shutdown_step + 200))
    vr_tail = v_r[t_slice]
    re_tail = radius[t_slice] - float(trace["target_radius"])

    return {
        "num_steps": int(trace["num_steps"]),
        "terminated": bool(trace["terminated"]),
        "truncated": bool(trace["truncated"]),
        "shutdown_step": int(shutdown_step),
        "shutdown_occurred": shutdown_occured,
        "min_action_norm_after_transient": float(np.min(final_action_norm[50:])) if len(final_action_norm) > 50 else float(np.min(final_action_norm)),
        "mean_action_norm_in_collapse_region": float(np.mean(final_action_norm[collapse_region])) if len(final_action_norm) else 0.0,
        "mean_raw_action_norm_in_collapse_region": float(np.mean(raw_action_norm[collapse_region])) if len(raw_action_norm) else 0.0,
        "final_radius_error": float(abs(radius[-1] - float(trace["target_radius"]))) if len(radius) else 0.0,
        "tail_mean_abs_vr": float(np.mean(np.abs(vr_tail))) if len(vr_tail) else 0.0,
        "tail_avg_radius_error": float(np.mean(np.abs(re_tail))) if len(re_tail) else 0.0,
        "tail_radius_std": float(np.std(radius[t_slice])) if len(radius) else 0.0,
        "tail_mean_alignment": float(np.mean(alignment[t_slice])) if len(alignment) else 0.0,
        "tail_speed_cv": float(np.std(speed[t_slice]) / (np.mean(speed[t_slice]) + 1e-12)) if len(speed) else 0.0,
        "tail_vr_sign_changes": bool(np.any(np.signbit(vr_tail[1:]) != np.signbit(vr_tail[:-1]))) if len(vr_tail) > 1 else False,
        "tail_crosses_target_radius": bool(np.any(np.signbit(re_tail[1:]) != np.signbit(re_tail[:-1]))) if len(re_tail) > 1 else False,
        "remains_active_after_shutdown": bool(np.mean(final_action_norm[collapse_region]) > 0.03) if len(final_action_norm[collapse_region]) else False,
        "intervention_rate": float(np.mean(intervene)) if len(intervene) else 0.0,
        "mean_rescue_action_norm": float(np.mean(rescue_action_norm)) if len(rescue_action_norm) else 0.0,
        "raw_policy_still_shutdown": bool(np.mean(raw_action_norm[collapse_region]) < 0.02) if len(raw_action_norm[collapse_region]) else False,
        "amplitude_shrinks": bool(np.std((radius - float(trace["target_radius"]))[len(radius)//2:]) < np.std((radius - float(trace["target_radius"]))[:len(radius)//2])) if len(radius) >= 20 else False,
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


def save_checkpoint_summary(path: Path, checkpoint_tag: str, rows: List[Dict[str, object]]) -> None:
    cases = [row["case_label"] for row in rows if row["variant"] == "baseline"]
    base_final_r = [row["final_radius_error"] for row in rows if row["variant"] == "baseline"]
    wrap_final_r = [row["final_radius_error"] for row in rows if row["variant"] == "wrapped"]
    base_tail_vr = [row["tail_mean_abs_vr"] for row in rows if row["variant"] == "baseline"]
    wrap_tail_vr = [row["tail_mean_abs_vr"] for row in rows if row["variant"] == "wrapped"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(cases))
    w = 0.38
    axes[0].bar(x - w / 2, base_final_r, width=w, label="baseline")
    axes[0].bar(x + w / 2, wrap_final_r, width=w, label="wrapped")
    axes[0].set_xticks(x, cases, rotation=20)
    axes[0].set_title("final_radius_error")
    axes[0].legend()

    axes[1].bar(x - w / 2, base_tail_vr, width=w, label="baseline")
    axes[1].bar(x + w / 2, wrap_tail_vr, width=w, label="wrapped")
    axes[1].set_xticks(x, cases, rotation=20)
    axes[1].set_title("tail_mean_abs_vr")
    axes[1].legend()

    fig.suptitle(f"Day21/22 anti-shutdown summary | {checkpoint_tag}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def classify_effect(base: Dict[str, object], wrapped: Dict[str, object]) -> str:
    action_helped = wrapped["shutdown_step"] > base["shutdown_step"] or wrapped["mean_action_norm_in_collapse_region"] > base["mean_action_norm_in_collapse_region"]
    orbit_helped = (
        wrapped["final_radius_error"] < base["final_radius_error"]
        and wrapped["tail_mean_abs_vr"] < base["tail_mean_abs_vr"]
    )
    if action_helped and orbit_helped:
        return "real_improvement"
    if action_helped and not orbit_helped:
        return "forced_activity_only"
    return "no_meaningful_help"


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    config = AntiShutdownConfig()
    aggregate_rows: List[Dict[str, object]] = []
    figure_paths: List[str] = []
    summary: Dict[str, object] = {"config": config.to_dict(), "checkpoints": {}}

    for checkpoint_tag, checkpoint_path in checkpoint_specs.items():
        base_policy = LoadedPolicy(checkpoint_path)
        nominal_traces: Dict[str, Dict[str, object]] = {}
        checkpoint_rows: List[Dict[str, object]] = []
        summary["checkpoints"][checkpoint_tag] = {"checkpoint_path": str(checkpoint_path), "cases": []}

        for case in cases:
            for variant in ("baseline", "wrapped"):
                env = LightweightOrbitEnv(
                    thrust_scale=args.thrust_scale,
                    r0_over_target=case.r0_over_target,
                    max_steps=args.max_steps,
                )
                policy = base_policy if variant == "baseline" else PPOAntiShutdownWrapper(
                    base_policy,
                    target_radius=env.target_radius,
                    mu=env.mu,
                    config=config,
                )
                trace = rollout_trace(policy, env, max_steps=args.max_steps, apply_nudge=case.vr_nudge)
                metrics = summarize_trace(trace, shutdown_threshold_action="final_action_norm")
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

            base_row = checkpoint_rows[-2]
            wrap_row = checkpoint_rows[-1]
            summary["checkpoints"][checkpoint_tag]["cases"][-1]["effect_class"] = classify_effect(base_row, wrap_row)

        radius_path = OUTPUT_DIR / f"{checkpoint_tag}_radius_vs_time.svg"
        vr_path = OUTPUT_DIR / f"{checkpoint_tag}_v_r_vs_time.svg"
        action_path = OUTPUT_DIR / f"{checkpoint_tag}_action_norm_vs_time.svg"
        save_timeseries_plot(radius_path, f"{checkpoint_tag} | baseline vs wrapped radius", "radius [m]", nominal_traces, "radius")
        save_timeseries_plot(vr_path, f"{checkpoint_tag} | baseline vs wrapped v_r", "v_r [m/s]", nominal_traces, "v_r")
        save_timeseries_plot(action_path, f"{checkpoint_tag} | baseline vs wrapped action_norm", "action_norm", nominal_traces, "final_action_norm")
        figure_paths.extend([str(radius_path), str(vr_path), str(action_path)])

        summary_path = OUTPUT_DIR / f"{checkpoint_tag}_summary.svg"
        save_checkpoint_summary(summary_path, checkpoint_tag, checkpoint_rows)
        figure_paths.append(str(summary_path))

    write_csv(CSV_PATH, aggregate_rows)
    JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    diagnosis_lines = []
    for checkpoint_tag in checkpoint_specs:
        rows = [row for row in aggregate_rows if row["checkpoint_tag"] == checkpoint_tag]
        baseline_rows = {row["case_label"]: row for row in rows if row["variant"] == "baseline"}
        wrapped_rows = {row["case_label"]: row for row in rows if row["variant"] == "wrapped"}
        real_count = 0
        forced_count = 0
        for case_label, base_row in baseline_rows.items():
            effect = classify_effect(base_row, wrapped_rows[case_label])
            if effect == "real_improvement":
                real_count += 1
            elif effect == "forced_activity_only":
                forced_count += 1
        if real_count > 0:
            diagnosis = "The wrapper produced localized real improvement on some cases, but not a clean orbit-lock solution."
        elif forced_count > 0:
            diagnosis = "The wrapper mostly forced extra activity without consistent orbit-quality improvement."
        else:
            diagnosis = "The wrapper did not meaningfully improve the shutdown band behavior."
        diagnosis_lines.append(f"- `{checkpoint_tag}`: {diagnosis}")

    md = [
        "# Day 21 / Day 22 Anti-Shutdown Evaluation",
        "",
        "## Wrapper Added",
        "",
        "- `controller/ppo_anti_shutdown_wrapper.py`",
        "- `scripts/day21_day22_anti_shutdown_eval.py`",
        "",
        "The wrapper acts as a minimal safety layer on top of PPO. It only intervenes when the PPO action is abnormally small while radius or tangential-speed error is still unresolved and `|v_r|` is locally quiet.",
        "",
        "## Thresholds Used",
        "",
    ]
    for key, value in config.to_dict().items():
        md.append(f"- `{key}` = `{value}`")
    md.extend([
        "",
        "## Figures Generated",
        "",
    ])
    for path in figure_paths:
        md.append(f"- `{Path(path).as_posix()}`")
    md.extend([
        "",
        "## Aggregate Outputs",
        "",
        f"- `{CSV_PATH.as_posix()}`",
        f"- `{JSON_PATH.as_posix()}`",
        "",
        "## Shutdown Effect",
        "",
    ])
    md.extend(diagnosis_lines)
    md.extend([
        "",
        "## One-Sentence Diagnosis",
        "",
        "The anti-shutdown wrapper is effective only if it improves both action activity and orbit quality; otherwise it is just masking PPO shutdown with forced control.",
        "",
    ])
    SUMMARY_PATH.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 21/22 anti-shutdown wrapper evaluation.")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--thrust-scale", type=float, default=20000.0)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
