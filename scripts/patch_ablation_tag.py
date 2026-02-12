# scripts/patch_ablation_tag.py
# Patch difficulty_tag in raw CSV for a specific dedup_key (in-place rewrite).

from pathlib import Path
import csv

CSV_PATH = Path("analysis/results/ablation_thrust_x_difficulty.csv")
TARGET_KEY = "b6c313a8"
NEW_TAG = "Easy"

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames
        rows = list(r)

    if not fieldnames:
        raise RuntimeError("CSV has no header/fieldnames")

    patched = 0
    for rr in rows:
        if (rr.get("dedup_key") or "").strip() == TARGET_KEY:
            rr["difficulty_tag"] = NEW_TAG
            patched += 1

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] patched={patched} key={TARGET_KEY} -> difficulty_tag={NEW_TAG}")

if __name__ == "__main__":
    main()
