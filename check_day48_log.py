"""
Day 48 log sanity checker.
- Loads logs/day48/test_log_day48.csv
- Prints a tidy summary
- Runs a few sanity checks
"""

from pathlib import Path
import math
import pandas as pd

CSV_PATH = Path("logs/day48/test_log_day48.csv")

def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"[ERROR] CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"[OK] Loaded CSV: {CSV_PATH}")
    print(f"[INFO] Rows: {len(df)} | Columns: {list(df.columns)}")

    # ---- Pretty print a compact summary ----
    print("\n=== Summary (scenario, steps, dv1, total_reward, final_orbit_error, elapsed_s) ===")
    for _, r in df.iterrows():
        print(f"{r['scenario']:>9} | steps={r['steps']} | dv1={r['dv1']} | "
              f"ret={r['total_reward']} | err={r['final_orbit_error']} | t={r.get('elapsed_s','?')}s")

    # ---- Sanity checks ----
    errors = []

    # 1) must have exactly 3 rows and expected scenarios
    expected = {"circular", "elliptic", "transfer"}
    scenarios = set(df["scenario"])
    if len(df) != 3:
        errors.append(f"Expected 3 rows (one per scenario), got {len(df)}.")
    if scenarios != expected:
        errors.append(f"Expected scenarios {expected}, got {scenarios}.")

    # 2) dv1 should be nonzero only for transfer
    for _, r in df.iterrows():
        dv1 = float(r["dv1"])
        sc = r["scenario"]
        if sc == "transfer":
            if not (abs(dv1) > 0 and math.isfinite(dv1)):
                errors.append("dv1 should be nonzero & finite for transfer scenario.")
        else:
            if abs(dv1) > 1e-9:
                errors.append(f"dv1 should be ~0 for {sc}, got {dv1}")

    # 3) steps >= 1
    for _, r in df.iterrows():
        if r["steps"] < 1:
            errors.append(f"steps < 1 for {r['scenario']}")

    # 4) final_orbit_error finite check
    for _, r in df.iterrows():
        try:
            val = float(r["final_orbit_error"])
            if not math.isfinite(val):
                print(f"[WARN] final_orbit_error non-finite for {r['scenario']}: {r['final_orbit_error']}")
        except Exception:
            print(f"[WARN] final_orbit_error not numeric for {r['scenario']}")

    print("\n=== Checks ===")
    if errors:
        print("[FAIL]")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)
    else:
        print("[PASS] Day 48 CSV looks good!")

if __name__ == "__main__":
    main()
