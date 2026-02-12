# scripts/make_ablation_clean_csv.py
# Build analysis/results/ablation_thrust_x_difficulty_clean.csv from raw CSV.
# - Enforce schema
# - Drop malformed rows
# - Deduplicate by dedup_key (keep last occurrence)

from pathlib import Path
import csv

RAW_PATH = Path("analysis/results/ablation_thrust_x_difficulty.csv")
CLEAN_PATH = Path("analysis/results/ablation_thrust_x_difficulty_clean.csv")

FIELDS = [
    "thrust_newton",
    "difficulty_tag",
    "r0_over_target",
    "avg_radius_error",
    "final_radius_error",
    "saturation_rate_mean",
    "total_reward",
    "dedup_key",
]

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"raw csv not found: {RAW_PATH}")

    rows_in = []
    with RAW_PATH.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for rr in r:
            rows_in.append(rr)

    # Normalize + filter
    normalized = []
    dropped = 0

    for rr in rows_in:
        key = (rr.get("dedup_key") or "").strip()
        diff = (rr.get("difficulty_tag") or "").strip()
        if not key or not diff:
            dropped += 1
            continue

        thrust = _to_float(rr.get("thrust_newton", ""))
        r0 = _to_float(rr.get("r0_over_target", ""))
        avg_err = _to_float(rr.get("avg_radius_error", ""))
        fin_err = _to_float(rr.get("final_radius_error", ""))
        sat = _to_float(rr.get("saturation_rate_mean", ""))
        rew = _to_float(rr.get("total_reward", ""))

        # Keep only well-formed numeric rows
        if None in [thrust, r0, avg_err, fin_err, sat, rew]:
            dropped += 1
            continue

        normalized.append({
            "thrust_newton": thrust,
            "difficulty_tag": diff,
            "r0_over_target": r0,
            "avg_radius_error": avg_err,
            "final_radius_error": fin_err,
            "saturation_rate_mean": sat,
            "total_reward": rew,
            "dedup_key": key,
        })

    # Deduplicate: keep last row for each dedup_key
    by_key = {}
    for rr in normalized:
        by_key[rr["dedup_key"]] = rr

    deduped = list(by_key.values())

    # Sort for stable plotting (difficulty then thrust)
    deduped.sort(key=lambda x: (x["difficulty_tag"], x["thrust_newton"], x["r0_over_target"], x["dedup_key"]))

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLEAN_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rr in deduped:
            w.writerow(rr)

    print(f"[OK] wrote: {CLEAN_PATH} | in={len(rows_in)} kept={len(normalized)} dropped={dropped} deduped={len(deduped)}")

if __name__ == "__main__":
    main()
