"""Compare helpers for replay validation."""
from typing import Dict, Tuple
import math

def l2(a, b) -> float:
    """Compute L2 distance between two same-length vectors (list/tuple)."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def within_thresholds(metrics: Dict[str, float],
                      thresholds: Dict[str, float]) -> Tuple[bool, Dict[str, Tuple[float, float]]]:
    """
    Return (ok, detail) where detail maps name -> (value, threshold).
    """
    detail = {}
    ok = True
    for k, thr in thresholds.items():
        v = float(metrics.get(k, float("inf")))
        detail[k] = (v, thr)
        if v > thr:
            ok = False
    return ok, detail
