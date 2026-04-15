from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.orbit_lock_controller import OrbitLockConfig, OrbitLockController
from envs.orbit_env import OrbitEnv


GENERALIZATION_CSV = PROJECT_ROOT / "analysis" / "figs" / "orbit_lock_generalization" / "aggregate_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "phase_controller_dataset"
NPZ_PATH = OUTPUT_DIR / "phase_controller_dataset.npz"
METADATA_PATH = OUTPUT_DIR / "phase_controller_dataset_metadata.json"
SUMMARY_PATH = PROJECT_ROOT / "analysis" / "phase_controller_dataset.md"
MAX_STEPS = 100000
STRICT_CFG = {
    "tol_r": 1.0e-3,
    "tol_v": 1.0e-3,
    "tol_ang": 0.02,
    "success_threshold": 200,
}
PHASE_TO_ID = {
    "DESCENT": 0,
    "CAPTURE": 1,
    "LOCK": 2,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_env(dt: float, thrust_scale: float) -> OrbitEnv:
    return OrbitEnv(
        thrust_scale=thrust_scale,
        dt=dt,
        max_steps=MAX_STEPS,
        reward_mode="orbit_circular_minimal",
        tol_r=STRICT_CFG["tol_r"],
        tol_v=STRICT_CFG["tol_v"],
        tol_ang=STRICT_CFG["tol_ang"],
        success_threshold=STRICT_CFG["success_threshold"],
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


def load_dataset_setups() -> List[Dict[str, float]]:
    rows: List[Dict[str, str]] = []
    with GENERALIZATION_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    selected = [
        row
        for row in rows
        if row["success"] == "True" or int(row["radius_crossings_total"]) > 0
    ]
    dedup: Dict[tuple[float, float, float], Dict[str, float]] = {}
    for row in selected:
        key = (float(row["r0_over_target"]), float(row["dt"]), float(row["thrust_scale"]))
        dedup[key] = {
            "r0_over_target": key[0],
            "dt": key[1],
            "thrust_scale": key[2],
        }
    return list(dedup.values())


def build_dataset() -> tuple[Dict[str, np.ndarray], List[Dict[str, float]]]:
    setups = load_dataset_setups()
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    phase_ids: List[int] = []
    done_flags: List[int] = []
    setup_ids: List[int] = []
    step_ids: List[int] = []
    metadata: List[Dict[str, float]] = []

    for setup_idx, setup in enumerate(setups):
        env = make_env(setup["dt"], setup["thrust_scale"])
        set_default_start(env, setup["r0_over_target"])
        controller = OrbitLockController(target_radius=env.target_radius, mu=env.mu, config=OrbitLockConfig())
        obs = np.asarray(env._get_obs(), dtype=np.float32)
        steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated) and steps < MAX_STEPS:
            info = controller.act_with_info(obs)
            action = np.asarray(info["final_action"], dtype=np.float32)
            phase = str(info["phase"])
            next_obs, _, terminated, truncated, _ = env.step(action)
            observations.append(obs.copy())
            actions.append(action.copy())
            phase_ids.append(PHASE_TO_ID[phase])
            done_flags.append(1 if (terminated or truncated) else 0)
            setup_ids.append(setup_idx)
            step_ids.append(steps)
            obs = np.asarray(next_obs, dtype=np.float32)
            steps += 1

        metadata.append(
            {
                "setup_id": setup_idx,
                "r0_over_target": setup["r0_over_target"],
                "dt": setup["dt"],
                "thrust_scale": setup["thrust_scale"],
                "num_steps": steps,
                "phase_transition_count": len(controller.phase_transitions),
                "success": float(env.success_counter >= env.success_threshold),
            }
        )

    arrays = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "phase_ids": np.asarray(phase_ids, dtype=np.int64),
        "done_flags": np.asarray(done_flags, dtype=np.int64),
        "setup_ids": np.asarray(setup_ids, dtype=np.int64),
        "step_ids": np.asarray(step_ids, dtype=np.int64),
    }
    return arrays, metadata


def write_summary(arrays: Dict[str, np.ndarray], metadata: List[Dict[str, float]]) -> None:
    phase_counts = {name: int(np.sum(arrays["phase_ids"] == idx)) for name, idx in PHASE_TO_ID.items()}
    lines = [
        "# Phase Controller Dataset",
        "",
        "## Outputs",
        "",
        f"- `{NPZ_PATH.as_posix()}`",
        f"- `{METADATA_PATH.as_posix()}`",
        "",
        "## Dataset Contents",
        "",
        f"- observation count: `{len(arrays['observations'])}`",
        f"- action count: `{len(arrays['actions'])}`",
        f"- setup count: `{len(metadata)}`",
        f"- phase counts: `{phase_counts}`",
        "",
        "## Included Setups",
        "",
    ]
    for item in metadata:
        lines.append(
            f"- setup `{int(item['setup_id'])}`: `r0={item['r0_over_target']:.5f}`, `dt={int(item['dt'])}`, `thrust={int(item['thrust_scale'])}`, `steps={int(item['num_steps'])}`"
        )
    lines.extend([
        "",
        "## Purpose",
        "",
        "- This dataset is intended for behavior cloning of the explicit phase controller before PPO fine-tuning.",
        "- Phase labels are included so cloning and phase-conditioned reward shaping can share the same supervision source.",
    ])
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    arrays, metadata = build_dataset()
    np.savez_compressed(NPZ_PATH, **arrays)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(arrays, metadata)
    print(f"Saved dataset to: {NPZ_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
