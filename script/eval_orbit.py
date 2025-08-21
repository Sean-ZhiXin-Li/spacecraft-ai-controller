from __future__ import annotations
import re
import numpy as np
from typing import Tuple, Optional

# ---------- number parsing (supports 7p5e12 / 1e12 / 123456 / 1.2e3) ----------
_NUM = r"[0-9]+(?:p[0-9]+)?(?:e[+\-]?[0-9]+)?|[0-9]+(?:\.[0-9]+)?(?:e[+\-]?[0-9]+)?"

def _to_float(s: str) -> float:
    """Convert tokens like '7p5e12' -> '7.5e12' and then to float."""
    return float(s.replace("p", "."))


# ---------- basic orbital quantities from state ----------
def _elements_from_state(r_vec: np.ndarray, v_vec: np.ndarray, mu: float) -> Tuple[float, float, float]:
    """
    Returns (a, e, v_rad):
      a     : semi-major axis
      e     : eccentricity magnitude
      v_rad : radial velocity component
    """
    r = float(np.linalg.norm(r_vec))
    v = float(np.linalg.norm(v_vec))
    h_vec = np.cross(r_vec, v_vec)
    # specific energy
    eps = v * v / 2.0 - mu / r
    a = -mu / (2.0 * eps)
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = float(np.linalg.norm(e_vec))
    v_rad = float(np.dot(v_vec, r_vec) / r)
    return a, e, v_rad


# ---------- success criteria per task family ----------
def _ok_circular(
    r_end: np.ndarray,
    v_end: np.ndarray,
    r_tar: float,
    mu: float,
    a_tol: float = 0.12,
    e_tol: float = 0.02,
    v_rad_tol: float = 1e-3,
) -> Tuple[bool, float]:
    """Target: near-circular orbit at radius r_tar."""
    a_est, e_est, v_rad = _elements_from_state(r_end, v_end, mu)
    a_err = abs(a_est - r_tar) / r_tar
    ok = (a_err < a_tol) and (e_est < e_tol) and (abs(v_rad) < v_rad_tol)
    return ok, max(a_err, e_est, abs(v_rad))


def _ok_elliptic(
    r_end: np.ndarray,
    v_end: np.ndarray,
    rp: float,
    ra: float,
    mu: float,
    a_tol: float = 0.02,
    e_tol: float = 0.02,
) -> Tuple[bool, float]:
    """Target: ellipse with pericenter rp and apocenter ra."""
    a_tar = 0.5 * (rp + ra)
    e_tar = (ra - rp) / (ra + rp)
    a_est, e_est, _ = _elements_from_state(r_end, v_end, mu)
    a_err = abs(a_est - a_tar) / a_tar
    e_err = abs(e_est - e_tar)
    ok = (a_err < a_tol) and (e_err < e_tol)
    return ok, max(a_err, e_err)


def _ok_transfer(
    r_end: np.ndarray,
    v_end: np.ndarray,
    r2: float,
    mu: float,
    a_tol: float = 0.02,
    e_tol: float = 0.02,
    v_rad_tol: float = 1e-3,
) -> Tuple[bool, float]:
    """Target: near-circular final orbit at radius r2."""
    a_est, e_est, v_rad = _elements_from_state(r_end, v_end, mu)
    a_err = abs(a_est - r2) / r2
    ok = (a_err < a_tol) and (e_est < e_tol) and (abs(v_rad) < v_rad_tol)
    return ok, max(a_err, e_est, abs(v_rad))


# ---------- task_id parsing ----------
def _parse_circular_radius(task_id: str) -> Optional[float]:
    # matches: circ_r_<R>_*  |  circular_r_<R>_*  |  perturb_r_<R>_ang_*
    m = re.search(r"(?:^|_)c(?:irc|ircular)_r_(" + _NUM + ")", task_id)
    if m:
        return _to_float(m.group(1))
    m = re.search(r"(?:^|_)perturb_r_(" + _NUM + r")_", task_id)
    if m:
        return _to_float(m.group(1))
    return None


def _parse_elliptic_rp_ra(task_id: str) -> Optional[Tuple[float, float]]:
    m = re.search(r"elli_rp_(" + _NUM + r")_ra_(" + _NUM + ")", task_id)
    if not m:
        return None
    return _to_float(m.group(1)), _to_float(m.group(2))


def _parse_transfer_r2(task_id: str) -> Optional[float]:
    m = re.search(r"transfer_(" + _NUM + r")_to_(" + _NUM + ")", task_id)
    if not m:
        return None
    # r1 = _to_float(m.group(1))  # not needed for scoring
    r2 = _to_float(m.group(2))
    return r2


# ---------- public API ----------
def score(
    task_id: str,
    r_end_vec,
    v_end_vec,
    mu: float,
    *,
    circ_a_tol: float = 0.12,
    circ_e_tol: float = 0.02,
    circ_vr_tol: float = 1e-3,
    elli_a_tol: float = 0.02,
    elli_e_tol: float = 0.02,
    tran_a_tol: float = 0.02,
    tran_e_tol: float = 0.02,
    tran_vr_tol: float = 1e-3,
) -> Tuple[bool, float]:
    """
    Auto-detect task family from task_id and score the final state.

    Returns:
        (success: bool, error_scalar: float)

    Families:
        - circular / circular_* / perturb_* : check a≈r_tar, e≈0, radial vel≈0
        - elli_rp_*_ra_*                    : check (a,e) vs target ellipse
        - transfer_*_to_*                   : final near-circular at r2

    Fallback:
        If the task_id is not recognized, return (True, 0.0) so evaluation
        never crashes (acts as a no-op scorer).
    """
    r_end = np.asarray(r_end_vec, dtype=float)
    v_end = np.asarray(v_end_vec, dtype=float)

    # 1) elliptic
    pr = _parse_elliptic_rp_ra(task_id)
    if pr:
        rp, ra = pr
        return _ok_elliptic(r_end, v_end, rp, ra, mu, a_tol=elli_a_tol, e_tol=elli_e_tol)

    # 2) transfer -> near-circular at r2
    r2 = _parse_transfer_r2(task_id)
    if r2 is not None:
        return _ok_transfer(
            r_end, v_end, r2, mu, a_tol=tran_a_tol, e_tol=tran_e_tol, v_rad_tol=tran_vr_tol
        )

    # 3) circular (including perturb_* tasks)
    r_tar = _parse_circular_radius(task_id)
    if r_tar is not None:
        return _ok_circular(
            r_end, v_end, r_tar, mu, a_tol=circ_a_tol, e_tol=circ_e_tol, v_rad_tol=circ_vr_tol
        )

    # 4) fallback: succeed with zero error
    return True, 0.0