import csv
import math
from pathlib import Path

CSV_IN = Path("analysis/results/ablation_thrust_x_difficulty.csv")
CSV_OUT = Path("analysis/results/ablation_thrust_x_difficulty_clean.csv")

HEADER = [
    "thrust_newton",
    "difficulty_tag",
    "r0_over_target",
    "avg_radius_error",
    "final_radius_error",
    "saturation_rate_mean",
    "total_reward",
    "dedup_key",
]

def is_hex8(s: str) -> bool:
    if not isinstance(s, str) or len(s) != 8:
        return False
    try:
        int(s, 16)
        return True
    except Exception:
        return False

def f(x: str):
    # safe float parse
    try:
        return float(x)
    except Exception:
        return math.nan

def normalize_row(raw: list[str]) -> dict | None:
    """
    Accepts a raw CSV row (list of strings).
    Returns normalized dict in HEADER order, or None if row is junk.
    Handles:
      - correct format: [thrust, tag, ..., total_reward, dedup]
      - shifted format: [dedup, thrust, tag, ..., sat_rate, total_reward?]
      - missing dedup
    """
    if not raw:
        return None
    raw = [x.strip() for x in raw]

    # Case: shifted: first cell is 8-hex key, second is thrust
    if len(raw) >= 2 and is_hex8(raw[0]) and raw[1]:
        dedup = raw[0]
        # shift right by 1
        raw = raw[1:] + [""]  # keep length-ish
        # now raw[0] should be thrust
        out = {
            "thrust_newton": f(raw[0]),
            "difficulty_tag": raw[1] if len(raw) > 1 else "",
            "r0_over_target": f(raw[2]) if len(raw) > 2 else math.nan,
            "avg_radius_error": f(raw[3]) if len(raw) > 3 else math.nan,
            "final_radius_error": f(raw[4]) if len(raw) > 4 else math.nan,
            "saturation_rate_mean": f(raw[5]) if len(raw) > 5 else math.nan,
            "total_reward": f(raw[6]) if len(raw) > 6 else math.nan,
            "dedup_key": dedup,
        }
        return out

    # Case: normal-ish
    if len(raw) < 7:
        return None

    out = {
        "thrust_newton": f(raw[0]),
        "difficulty_tag": raw[1],
        "r0_over_target": f(raw[2]),
        "avg_radius_error": f(raw[3]),
        "final_radius_error": f(raw[4]),
        "saturation_rate_mean": f(raw[5]),
        "total_reward": f(raw[6]),
        "dedup_key": raw[7] if len(raw) > 7 else "",
    }

    # Basic validity: thrust and r0_over_target should be numbers
    if not math.isfinite(out["thrust_newton"]) or not math.isfinite(out["r0_over_target"]):
        return None
    return out

def main():
    if not CSV_IN.exists():
        raise SystemExit(f"missing: {CSV_IN}")

    rows = []
    with CSV_IN.open("r", newline="", encoding="utf-8") as f_in:
        r = csv.reader(f_in)
        header = next(r, None)  # consume header
        for raw in r:
            nr = normalize_row(raw)
            if nr is not None:
                rows.append(nr)

    # Dedup: keep last occurrence of each dedup_key if present,
    # otherwise keep all (but you'll ideally regenerate later)
    seen = {}
    no_key = []
    for rr in rows:
        k = rr.get("dedup_key", "").strip()
        if is_hex8(k):
            seen[k] = rr  # last wins
        else:
            no_key.append(rr)

    cleaned = list(seen.values()) + no_key

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=HEADER)
        w.writeheader()
        for rr in cleaned:
            w.writerow({k: rr.get(k, "") for k in HEADER})

    print(f"wrote: {CSV_OUT}  (rows_in={len(rows)} rows_out={len(cleaned)})")
    print("Next: open the clean file and decide if you want to replace the original.")

if __name__ == "__main__":
    main()
