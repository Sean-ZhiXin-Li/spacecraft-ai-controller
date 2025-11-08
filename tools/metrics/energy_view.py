import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class EnergyConfig:
    G: float
    M: float
    m: float
    dt: float
    target_radius: float
    energy_eps: float = 1e-3
    angmom_eps: float = 1e-3


def load_replay(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Replay file not found: {path}")

    data = np.load(path, allow_pickle=True)

    if isinstance(data, np.lib.npyio.NpzFile):
        if "obs" in data.files:
            obs = data["obs"]
        else:
            obs = data[data.files[0]]
        actions = data["actions"] if "actions" in data.files else None
    else:
        obs = data
        actions = None

    return np.asarray(obs), (None if actions is None else np.asarray(actions))


def extract_state(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if obs.shape[-1] < 4:
        raise ValueError(f"Expected obs with at least 4 dims [x,y,vx,vy], got shape {obs.shape}")
    x = obs[:, 0]
    y = obs[:, 1]
    vx = obs[:, 2]
    vy = obs[:, 3]
    return x, y, vx, vy


def compute_energy_and_L(
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    cfg: EnergyConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    r = np.sqrt(x * x + y * y)
    v = np.sqrt(vx * vx + vy * vy)

    kinetic = 0.5 * cfg.m * v * v
    potential = -cfg.G * cfg.M * cfg.m / r
    E_t = kinetic + potential

    L_t = cfg.m * (x * vy - y * vx)

    r_star = cfg.target_radius
    v_star = np.sqrt(cfg.G * cfg.M / r_star)
    E_star = -cfg.G * cfg.M * cfg.m / (2.0 * r_star)
    L_star = cfg.m * r_star * v_star

    return E_t, L_t, r, v, E_star, L_star


def _final_window_slice(T: int, frac: float = 0.1, min_len: int = 200) -> slice:
    win = max(int(T * frac), min_len)
    start = max(0, T - win)
    return slice(start, T)


def compute_energy_metrics(
    E_t: np.ndarray,
    L_t: np.ndarray,
    r_t: np.ndarray,
    v_t: np.ndarray,
    E_star: float,
    L_star: float,
    actions: Optional[np.ndarray],
    cfg: EnergyConfig
) -> Dict[str, float]:
    T = len(E_t)
    if T < 5:
        raise ValueError("Trajectory too short for energy metrics.")

    energy_error = E_t - E_star
    abs_err = np.abs(energy_error)
    rel_err = abs_err / (abs(E_star) + 1e-12)

    energy_convergence_step = -1
    converged_idx = np.where(rel_err < cfg.energy_eps)[0]
    if converged_idx.size > 0:
        energy_convergence_step = int(converged_idx[0])

    fw = _final_window_slice(T)
    energy_drift_percent = float(
        np.mean(energy_error[fw]) / (abs(E_star) + 1e-12)
    )

    ang_error = L_t - L_star
    angular_momentum_error_mean = float(
        np.mean(ang_error[fw]) / (abs(L_star) + 1e-12)
    )
    angular_momentum_error_final = float(
        ang_error[-1] / (abs(L_star) + 1e-12)
    )

    dE = np.diff(E_t)
    if len(dE) > 2:
        sign_flip_count = int(
            np.sum(np.sign(dE[1:]) * np.sign(dE[:-1]) < 0)
        )
        energy_oscillation_index = float(
            sign_flip_count / (len(dE) - 1 + 1e-9)
        )
    else:
        energy_oscillation_index = 0.0

    thrust_energy_ratio = -1.0
    if actions is not None:
        T_eff = min(len(dE), len(actions))
        if T_eff > 0:
            thrust = np.linalg.norm(actions[:T_eff], axis=-1)
            speed = v_t[:T_eff]
            ideal_dE = thrust * speed * cfg.dt
            denom = np.sum(np.abs(ideal_dE)) + 1e-12
            num = np.sum(np.abs(dE[:T_eff]))
            thrust_energy_ratio = float(num / denom)

    E_required = E_star - E_t[:-1]
    valid_mask = np.abs(E_required) > cfg.energy_eps * abs(E_star)
    if np.any(valid_mask):
        eta = np.abs(dE[valid_mask]) / (np.abs(E_required[valid_mask]) + 1e-12)
        energy_convergence_ratio_median = float(np.median(eta))
    else:
        energy_convergence_ratio_median = 0.0

    return {
        "energy_convergence_step": float(energy_convergence_step),
        "energy_drift_percent": energy_drift_percent,
        "angular_momentum_error_mean": angular_momentum_error_mean,
        "angular_momentum_error_final": angular_momentum_error_final,
        "energy_oscillation_index": energy_oscillation_index,
        "thrust_energy_ratio": thrust_energy_ratio,
        "energy_convergence_ratio_median": energy_convergence_ratio_median,
    }


def analyze_replay(
    replay_path: str,
    cfg: EnergyConfig,
    save_json: Optional[str] = None
) -> Dict[str, float]:
    obs, actions = load_replay(replay_path)
    x, y, vx, vy = extract_state(obs)
    E_t, L_t, r_t, v_t, E_star, L_star = compute_energy_and_L(x, y, vx, vy, cfg)
    metrics = compute_energy_metrics(E_t, L_t, r_t, v_t, E_star, L_star, actions, cfg)

    if save_json is not None:
        os.makedirs(os.path.dirname(save_json), exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    cfg = EnergyConfig(
        G=6.67430e-11,
        M=1.989e30,
        m=722.0,
        dt=3600.0,
        target_radius=7.5e12,
        energy_eps=0.05,
        angmom_eps=0.05,
    )

    targets = [
        (
            "logs/new_week_1/spiral_in/high_thrust/replay.npz",
            "logs/new_week_1/spiral_in/high_thrust/metrics_energy.json",
            "NEW_WEEK_1 spiral_in high_thrust",
        ),
    ]

    for replay_path, save_path, label in targets:
        if not os.path.exists(replay_path):
            print(f"[WARN] {label}: replay not found at {replay_path}, skip.")
            continue

        print(f"[INFO] Analyzing {label} ...")
        metrics = analyze_replay(replay_path, cfg, save_json=save_path)
        print(f"[OK] {label} metrics saved to {save_path}")
        print(json.dumps(metrics, indent=2))
        print("-" * 40)
