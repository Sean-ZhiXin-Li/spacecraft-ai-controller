import numpy as np
from typing import Optional

def radial_distance(xy: np.ndarray) -> np.ndarray:
    """
    xy: shape (T, 2) -> columns [x, y]
    return r: shape (T,)
    """
    return np.linalg.norm(xy, axis=1)

def radial_error(r: np.ndarray, r_target: float) -> np.ndarray:
    """
    r: (T,), r_target: scalar
    e_t = r_t - r_target
    """
    return r - float(r_target)

def norm_error(e: np.ndarray, r_target: float) -> np.ndarray:
    return e / float(r_target)

def vec_norm(arr: np.ndarray, axis: int = 1) -> np.ndarray:
    """Generic L2 norm along axis (e.g., actions)."""
    return np.linalg.norm(arr, axis=axis)

def convergence_step(norm_e: np.ndarray, eps: float = 0.005, window: int = 20) -> Optional[int]:
    """
    First index t where |norm_e[t:t+window]| <= eps for all window steps.
    Return None if not found.
    """
    T = len(norm_e)
    if T < window:
        return None
    mask = np.abs(norm_e) <= eps
    for t in range(0, T - window + 1):
        if np.all(mask[t:t+window]):
            return t
    return None

def steady_state_error(norm_e: np.ndarray, tail: int = 50, reducer: str = "mean") -> float:
    """Mean/median of last `tail` normalized errors by abs value."""
    tail = min(tail, len(norm_e))
    arr = np.abs(norm_e[-tail:])
    if reducer == "median":
        return float(np.median(arr))
    return float(np.mean(arr))

def oscillation_index(e: np.ndarray, tail: int = 50) -> float:
    """
    Sign-flip rate in the last `tail` steps.
    Value in [0,1]. Higher -> more oscillatory around target.
    """
    tail = min(tail, len(e))
    seg = e[-tail:]
    signs = np.sign(seg)
    flips = np.sum(np.diff(signs) != 0)
    return float(flips / max(1, tail - 1))

def saturation_rate(a: np.ndarray, high: float = 0.95) -> float:
    """
    Fraction of steps where ||a_t|| >= high.
    If actions are in [-1,1], set high ~ 0.95 (norm).
    """
    norms = vec_norm(a)
    sat = np.mean(norms >= high)
    return float(sat)

def under_power_ratio(a: np.ndarray, norm_e: np.ndarray, e_thresh: float = 0.01, a_low: float = 0.2) -> float:
    """
    Ratio of steps where error is significant but action magnitude is too low.
    """
    norms = vec_norm(a)
    mask = (np.abs(norm_e) >= e_thresh) & (norms <= a_low)
    return float(np.mean(mask))

def drift_percent(norm_e: np.ndarray) -> float:
    """Global mean |norm_e| as percent (0~1)."""
    return float(np.mean(np.abs(norm_e)))
