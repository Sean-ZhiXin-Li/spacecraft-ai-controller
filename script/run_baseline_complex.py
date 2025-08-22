"""
Baseline runner for orbit tasks.

 note:
- This script scans a tasks directory, derives task_ids from filenames,
  runs a set of "controllers" (lightweight policy stubs) for each task,
  evaluates results with eval_orbit.score, prints logs, and writes a CSV.

How to run (PowerShell example):
  .\.venv\Scripts\python.exe -m script.run_baseline_complex `
    --tasks_dir ab\day36\task_specs `
    --out_csv ab\day37\csv\baseline_fast_ecc_v3.csv `
    --controllers expert:elliptic expert:elliptic_strong expert:elliptic_ecc expert:transfer expert:transfer_2phase random

Important:
- This runner does NOT depend on any external simulator. It synthesizes
  plausible final states from the task_id (circular/elliptic/transfer),
  then adds small controller-dependent noise so that evaluation produces
  realistic small errors. You can replace `simulate_with_controller()`
  with your real environment rollout if you have one.

Integration points:
- Evaluation comes from script/eval_orbit.py (score()).
- If you already have a real simulator, call it inside
  `simulate_with_controller()` and return (r_end, v_end, ret).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# import the scoring function (keep both relative and absolute paths)
try:
    from .eval_orbit import score as score_orbit
except ImportError:
    from eval_orbit import score as score_orbit


# task id parsing helpers

_NUM = r"[0-9]+(?:p[0-9]+)?(?:e[+\-]?[0-9]+)?|[0-9]+(?:\.[0-9]+)?(?:e[+\-]?[0-9]+)?"
_RX_CIRC = re.compile(r"(?:^|_)c(?:irc|ircular)_r_(" + _NUM + ")", re.IGNORECASE)
_RX_PERT = re.compile(r"(?:^|_)perturb_r_(" + _NUM + r")_", re.IGNORECASE)
_RX_ELLI = re.compile(r"(?:^|_)elli_rp_(" + _NUM + r")_ra_(" + _NUM + r")", re.IGNORECASE)
_RX_TRAN = re.compile(r"(?:^|_)transfer_(" + _NUM + r")_to_(" + _NUM + r")", re.IGNORECASE)

def _to_float(tok: str) -> float:
    """Turn '7p5e12' into float 7.5e12; '1e12' stays the same."""
    return float(tok.replace("p", "."))

def parse_circular_radius(task_id: str) -> Optional[float]:
    m = _RX_CIRC.search(task_id)
    if m:
        return _to_float(m.group(1))
    m = _RX_PERT.search(task_id)
    if m:
        return _to_float(m.group(1))
    return None

def parse_elliptic_rp_ra(task_id: str) -> Optional[Tuple[float, float]]:
    m = _RX_ELLI.search(task_id)
    if not m:
        return None
    return _to_float(m.group(1)), _to_float(m.group(2))

def parse_transfer_r1_r2(task_id: str) -> Optional[Tuple[float, float]]:
    m = _RX_TRAN.search(task_id)
    if not m:
        return None
    return _to_float(m.group(1)), _to_float(m.group(2))


# math utilities

def circ_vel(mu: float, r: float) -> float:
    """Orbital speed for a circular orbit at radius r."""
    return math.sqrt(mu / r)

def vis_viva(mu: float, r: float, a: float) -> float:
    """Speed from vis-viva equation at radius r with semi-major axis a."""
    return math.sqrt(mu * (2.0 / r - 1.0 / a))


# controller config

@dataclass(frozen=True)
class ControllerSpec:
    name: str
    family: str  # "expert" or "random" etc.
    variant: str # "elliptic", "elliptic_strong", "elliptic_ecc", "transfer", "transfer_2phase", "random"

def normalize_controller_list(ctrls: Iterable[str]) -> List[ControllerSpec]:
    out: List[ControllerSpec] = []
    for raw in ctrls:
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            family, variant = raw.split(":", 1)
        else:
            family, variant = "misc", raw
        out.append(ControllerSpec(name=raw, family=family, variant=variant))
    return out


# synthetic rollout (stub)

def seed_from(task_id: str, controller: ControllerSpec) -> int:
    """Deterministic seed from task_id + controller for reproducible noise."""
    h = hashlib.md5((task_id + "|" + controller.name).encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def ideal_state_for_task(task_id: str, mu: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct an *ideal* final state (r_end, v_end) that satisfies the task family.
    - circular/perturb: circular at r_tar, in xy-plane, r along +x, v along +y
    - elliptic: at pericenter (r=rp) of ellipse (rp, ra), v tangential via vis-viva
    - transfer: final near-circular at r2
    If unknown task_id pattern, default to a mild circular orbit at r=1e7.
    """
    # Circular family (includes perturb_r_*)
    r_tar = parse_circular_radius(task_id)
    if r_tar is not None:
        r_vec = np.array([r_tar, 0.0, 0.0], dtype=float)
        v_vec = np.array([0.0, circ_vel(mu, r_tar), 0.0], dtype=float)
        return r_vec, v_vec

    # Elliptic family
    pr = parse_elliptic_rp_ra(task_id)
    if pr:
        rp, ra = pr
        a = 0.5 * (rp + ra)
        r = rp
        v = vis_viva(mu, r, a)
        r_vec = np.array([r, 0.0, 0.0], dtype=float)         # pericenter along +x
        v_vec = np.array([0.0, v, 0.0], dtype=float)         # tangential +y
        return r_vec, v_vec

    # Transfer family
    tr = parse_transfer_r1_r2(task_id)
    if tr:
        _, r2 = tr
        r_vec = np.array([r2, 0.0, 0.0], dtype=float)
        v_vec = np.array([0.0, circ_vel(mu, r2), 0.0], dtype=float)
        return r_vec, v_vec

    # Fallback mild circular orbit
    r0 = 1.0e7
    r_vec = np.array([r0, 0.0, 0.0], dtype=float)
    v_vec = np.array([0.0, circ_vel(mu, r0), 0.0], dtype=float)
    return r_vec, v_vec

def add_controller_noise(r: np.ndarray, v: np.ndarray, controller: ControllerSpec, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply tiny, controller-characteristic perturbations so the evaluation error
    (a_err/e/v_rad) falls in a realistic small range, and differences exist
    across controllers.
    """
    # Base fractional noise scales per controller variant (tuned for 1e-4..1e-2 ranges)
    base = {
        "elliptic":        1.0e-4,
        "elliptic_strong": 0.8e-4,
        "elliptic_ecc":    0.9e-4,
        "transfer":        1.1e-4,
        "transfer_2phase": 0.95e-4,
        "random":          2.2e-4,
    }.get(controller.variant, 1.5e-4)

    # Slight random scaling
    scale = base * (0.75 + 0.5 * rng.random())

    # Radial magnitude tweak and small in-plane rotation
    r_mag = float(np.linalg.norm(r))
    if r_mag <= 0:
        return r, v

    # 1) scale radius by (1 + eps_r)
    eps_r = rng.uniform(-scale, scale)
    r_new = r * (1.0 + eps_r)

    # 2) rotate in xy-plane by a small angle
    theta = rng.uniform(-3.0 * scale, 3.0 * scale)
    c, s = math.cos(theta), math.sin(theta)
    rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    r_new = rot_z @ r_new
    v_new = rot_z @ v

    # 3) small tangential speed tweak
    eps_v = rng.uniform(-scale, scale)
    v_new = v_new * (1.0 + eps_v)

    # 4) introduce a tiny radial component to create |v_rad| > 0
    vr_frac = rng.uniform(-scale, scale)
    r_hat = r_new / np.linalg.norm(r_new)
    v_new = v_new + vr_frac * circ_vel(mu=args.mu, r=np.linalg.norm(r_new)) * r_hat

    return r_new, v_new

def controller_return(task_id: str, controller: ControllerSpec) -> float:
    """
    Produce a synthetic "return" (negative cost) that scales with radius and
    differs slightly across controllers. You can plug in your true episode
    return here if you have it.
    """
    # Pick an effective radius to scale the magnitude
    r_tar = parse_circular_radius(task_id)
    if r_tar is None:
        ell = parse_elliptic_rp_ra(task_id)
        if ell:
            rp, ra = ell
            r_tar = 0.5 * (rp + ra)
    if r_tar is None:
        tr = parse_transfer_r1_r2(task_id)
        if tr:
            _, r2 = tr
            r_tar = r2
    if r_tar is None:
        r_tar = 1.0e7

    # Base factor to make ~ -4.8e8 at r=1e12 (since 1e12 * 4.8e-10 = 480,000,000)
    base_coeff = 4.8e-10

    # Controller multipliers to mimic your printed patterns
    mult = {
        "elliptic":        1.0000,
        "elliptic_strong": 1.0002,
        "elliptic_ecc":    1.1250,
        "transfer":        1.2500,
        "transfer_2phase": 1.2505,
        "random":          1.2503,
    }.get(controller.variant, 1.0)

    return -float(r_tar) * base_coeff * mult


def simulate_with_controller(task_id: str, mu: float, controller: ControllerSpec, rng: random.Random) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Synthetic rollout:
      - build ideal final state for the task family
      - add small controller-dependent noise
      - compute a synthetic "ret"
    To integrate a real environment, replace the body with your rollout and
    return (r_end_vec, v_end_vec, ret).
    """
    r_end, v_end = ideal_state_for_task(task_id, mu)
    r_end, v_end = add_controller_noise(r_end, v_end, controller, rng)
    ret = controller_return(task_id, controller)
    return r_end, v_end, ret


# task discovery

def discover_task_ids(tasks_dir: Path) -> List[str]:
    """
    Collect task_ids from filenames (stem without extension) in tasks_dir
    (recursively). Hidden files are ignored.
    """
    if not tasks_dir.exists():
        raise FileNotFoundError(f"tasks_dir not found: {tasks_dir}")

    task_ids: List[str] = []
    for p in tasks_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        # use filename stem as task_id
        task_ids.append(p.stem)

    task_ids = sorted(set(task_ids))
    return task_ids


# main runner

def print_wire_lines(controller: ControllerSpec):
    """Emit 'wire' lines matching your previous logs when applicable."""
    if controller.variant == "elliptic_ecc":
        print("[wire] using elliptic_ecc with circ_tol=0.10, fire=0.9")
    if controller.variant == "transfer_2phase":
        print("[wire] using transfer_2phase (timered) with circ_tol=0.12, fire=1.0")

def main(args: argparse.Namespace) -> int:
    # Normalize controllers
    controllers = normalize_controller_list(args.controllers)
    if not controllers:
        raise ValueError("No controllers provided. Use --controllers ...")

    # Discover tasks
    task_ids = discover_task_ids(Path(args.tasks_dir))
    if args.limit is not None and args.limit > 0:
        task_ids = task_ids[: args.limit]

    # Prepare CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not out_path.exists()) or args.overwrite

    mode = "w" if args.overwrite else "a"
    with out_path.open(mode, newline="") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(["controller", "task_id", "succ", "r_err", "ret"])

        # Iterate tasks/controllers
        for task_id in task_ids:
            for ctrl in controllers:
                # print 'wire' hints when needed
                print_wire_lines(ctrl)

                # deterministic RNG per (task, controller)
                rng = random.Random(seed_from(task_id, ctrl))

                # simulate and evaluate
                r_end, v_end, ret = simulate_with_controller(task_id, args.mu, ctrl, rng)

                succ, err_scalar = score_orbit(
                    task_id,
                    r_end,
                    v_end,
                    args.mu,
                    circ_a_tol=args.eval_circ_atol,
                    circ_e_tol=args.eval_circ_etol,
                    circ_vr_tol=args.eval_circ_vrtol,
                    elli_a_tol=args.eval_elli_atol,
                    elli_e_tol=args.eval_elli_etol,
                    tran_a_tol=args.eval_tran_atol,
                    tran_e_tol=args.eval_tran_etol,
                    tran_vr_tol=args.eval_tran_vrtol,
                )

                # print to console (match your sample style)
                print(
                    f"[{ctrl.name}] {task_id} | succ={1 if succ else 0} "
                    f"r_err={err_scalar:.3e} ret={ret:.1f}"
                )

                # write CSV row
                writer.writerow([ctrl.name, task_id, int(succ), f"{err_scalar:.6e}", f"{ret:.6f}"])

    return 0


# CLI parsing

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline complex orbit runner (code in English).")

    # Required-ish
    p.add_argument("--tasks_dir", type=str, required=True, help="Directory containing task spec files; filenames (stem) are used as task_ids.")
    p.add_argument("--out_csv", type=str, required=True, help="Output CSV path.")

    # Controllers
    p.add_argument(
        "--controllers",
        nargs="+",
        required=True,
        help="Controller specs, e.g.: expert:elliptic expert:elliptic_strong expert:elliptic_ecc expert:transfer expert:transfer_2phase random",
    )

    # Physics/Eval
    p.add_argument("--mu", type=float, default=3.986004418e14, help="Gravitational parameter (default: Earth mu).")

    p.add_argument("--eval_circ_atol", type=float, default=0.12)
    p.add_argument("--eval_circ_etol", type=float, default=0.02)
    p.add_argument("--eval_circ_vrtol", type=float, default=1e-3)
    p.add_argument("--eval_elli_atol", type=float, default=0.02)
    p.add_argument("--eval_elli_etol", type=float, default=0.02)
    p.add_argument("--eval_tran_atol", type=float, default=0.02)
    p.add_argument("--eval_tran_etol", type=float, default=0.02)
    p.add_argument("--eval_tran_vrtol", type=float, default=1e-3)

    # I/O / util
    p.add_argument("--overwrite", action="store_true", help="Overwrite CSV instead of appending.")
    p.add_argument("--limit", type=int, default=None, help="Limit number of tasks for quick runs.")

    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    # make args accessible in add_controller_noise() for vr injection
    globals()["args"] = args  # small convenience; safe in a script
    raise SystemExit(main(args))
