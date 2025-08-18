import sys
import csv
import subprocess
from pathlib import Path
from collections import OrderedDict

# Paths
THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parent
PY = sys.executable

# Locate replay_worst.py
REPLAY_CANDIDATES = [
    PROJECT_ROOT / "script" / "replay_worst.py",
    PROJECT_ROOT / "replay_worst.py",
]
REPLAY = next((p for p in REPLAY_CANDIDATES if p.exists()), None)
if REPLAY is None:
    print("[ERROR] Cannot find replay_worst.py. Tried:")
    for p in REPLAY_CANDIDATES:
        print("  -", p)
    sys.exit(1)

# Locate input folder (prefer task_specs)
IN1 = PROJECT_ROOT / "ab" / "day32" / "worst" / "task_specs"
IN2 = PROJECT_ROOT / "ab" / "day32" / "worst"
IN_DIR = IN1 if IN1.exists() else IN2
if not IN_DIR.exists():
    print("[ERROR] worst-task folder missing:")
    print("  -", IN1)
    print("  -", IN2)
    sys.exit(1)

OUT_ROOT = PROJECT_ROOT / "ab" / "day34"
AGENT = "greedy_energy_rt"

# Parameter sets
PARAM_SETS = [
    # Baseline trio
    dict(label="A",                a_hi=3.5e-4, fire_frac=0.35, band_in=1.25, band_out=1.80),
    dict(label="B",                a_hi=2.8e-4, fire_frac=0.30, band_in=1.20, band_out=1.90),
    dict(label="C",                a_hi=2.2e-4, fire_frac=0.22, band_in=1.15, band_out=2.00),

    # Plus variants
    dict(label="C_plus",           a_hi=2.0e-4, fire_frac=0.20, band_in=1.15, band_out=2.10),
    dict(label="B_plus",           a_hi=2.6e-4, fire_frac=0.28, band_in=1.20, band_out=1.95),

    # Follow-ups（你已验证）
    dict(label="C_plus_wider",     a_hi=2.0e-4, fire_frac=0.20, band_in=1.18, band_out=2.15),
    dict(label="C_plus_lower",     a_hi=1.8e-4, fire_frac=0.18, band_in=1.15, band_out=2.10),
    dict(label="lower_wider",      a_hi=1.8e-4, fire_frac=0.18, band_in=1.18, band_out=2.12),
    dict(label="lower_pulse",      a_hi=1.8e-4, fire_frac=0.16, band_in=1.15, band_out=2.10),

    # Combined / Min（你也已验证）
    dict(label="lower_combo",      a_hi=1.8e-4, fire_frac=0.16, band_in=1.18, band_out=2.12),
    dict(label="lower_min",        a_hi=1.7e-4, fire_frac=0.16, band_in=1.18, band_out=2.12),

    # NEW closing probes（本次新增）
    dict(label="lower_min_tight",  a_hi=1.7e-4, fire_frac=0.16, band_in=1.18, band_out=2.10),
    dict(label="lower_min_pulse15",a_hi=1.7e-4, fire_frac=0.15, band_in=1.18, band_out=2.12),
]

def run_one(ps):
    out_dir = OUT_ROOT / f"day34_{ps['label']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, str(REPLAY),
        "--in_dir", str(IN_DIR),
        "--out_root", str(out_dir),
        "--agent", AGENT,
        "--a_lo", "0.0",
        "--a_hi", f"{ps['a_hi']:.18g}",
        "--fire_frac", str(ps["fire_frac"]),
        "--band_in", str(ps["band_in"]),
        "--band_out", str(ps["band_out"]),
        "--seed", "999",
        "--verbose",
    ]
    print(f"\n==> Running {ps['label']}")
    print("   REPLAY:", REPLAY)
    print("   IN_DIR:", IN_DIR)
    print("   OUT   :", out_dir)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return out_dir

def find_csv(out_dir: Path) -> Path | None:
    for name in ("replay_compare.csv", "replay_worst_ab.csv"):
        p = out_dir / "replay" / name
        if p.exists():
            return p
    return None

def read_dfuel(csv_path: Path):
    """
    Parse replay CSV and return (task->dfuel, total).
    Compatible with columns: 'Δfuel', 'd_fuel', 'delta_fuel', 'dfuel'
    """
    tasks = OrderedDict()
    total = 0.0
    if not csv_path or not csv_path.exists():
        return tasks, total
    with csv_path.open("r", encoding="utf-8") as f:
        cr = csv.DictReader(f)
        name_keys = ("name", "task", "id")
        fuel_keys = ("Δfuel", "d_fuel", "delta_fuel", "dfuel")
        for r in cr:
            nk = next((k for k in name_keys if k in r and r[k]), None)
            if not nk:
                continue
            name = r[nk]
            val = None
            for fk in fuel_keys:
                if fk in r and r[fk]:
                    try:
                        val = float(r[fk]); break
                    except ValueError:
                        pass
            if val is None:
                continue
            tasks[name] = val
            total += val
    return tasks, total

def write_summary(path: Path, labels, all_tasks, per_label_tasks, per_label_total):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task"] + list(labels))
        for t in all_tasks:
            row = [t]
            for lab in labels:
                v = per_label_tasks[lab].get(t, "")
                row.append(f"{v:.6g}" if isinstance(v, (int, float)) else "")
            w.writerow(row)
        w.writerow(["TOTAL"] + [f"{per_label_total[lab]:.6g}" for lab in labels])

def main():
    labels = [ps["label"] for ps in PARAM_SETS]
    per_label_tasks = {lab: OrderedDict() for lab in labels}
    per_label_total  = OrderedDict((lab, 0.0) for lab in labels)

    for ps in PARAM_SETS:
        out_dir = run_one(ps)
        csv_path = find_csv(out_dir)
        print("   CSV:", csv_path if csv_path else "(not found)")
        task2dfuel, total = read_dfuel(csv_path) if csv_path else (OrderedDict(), 0.0)
        per_label_tasks[ps["label"]] = task2dfuel
        per_label_total[ps["label"]] = total
        print(f"   Total Δfuel (Agent − Expert): {total:.3e}")

    # union of tasks
    all_tasks = sorted({t for lab in labels for t in per_label_tasks[lab].keys()})
    summary = OUT_ROOT / "summary_day34_all.csv"
    write_summary(summary, labels, all_tasks, per_label_tasks, per_label_total)
    print("\n==> Summary saved to:", summary)
    print("    (Last row TOTAL gives per-set totals)")

if __name__ == "__main__":
    main()