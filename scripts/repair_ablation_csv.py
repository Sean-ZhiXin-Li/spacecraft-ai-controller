# scripts/repair_ablation_csv.py
# Repair ablation CSV by keeping only well-formed rows (expected 8 columns)
# and writing a normalized raw CSV for downstream clean/validate.

from pathlib import Path

EXPECTED_HEADER = [
    "thrust_newton",
    "difficulty_tag",
    "r0_over_target",
    "avg_radius_error",
    "final_radius_error",
    "saturation_rate_mean",
    "total_reward",
    "dedup_key",
]

def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def main():
    src = Path("analysis/results/ablation_thrust_x_difficulty.csv")
    dst = Path("analysis/results/ablation_thrust_x_difficulty_repaired.csv")

    lines = src.read_text(encoding="utf-8").splitlines()

    kept = []
    dropped = 0

    for ln in lines:
        if not ln.strip():
            continue
        parts = [p.strip() for p in ln.split(",")]

        # Skip header-like lines
        if parts[:2] == ["thrust_newton", "difficulty_tag"] or parts[:1] == ["dedup_key"]:
            continue

        # Case A: Correct 8 columns: thrust,...,dedup_key
        if len(parts) == 8 and is_number(parts[0]) and parts[1] in {"Easy", "Hard", "Medium"}:
            kept.append(parts)
            continue

        # Case B: Some older/broken rows might start with dedup_key then 7 fields
        # Example: 4326679c,800.0,Hard,...,-20795
        if len(parts) == 8 and (not is_number(parts[0])) and is_number(parts[1]):
            # Treat first as dedup_key and shift
            dedup = parts[0]
            shifted = parts[1:] + [dedup]
            # shifted now length 8: thrust,...,total_reward,dedup_key
            if is_number(shifted[0]) and shifted[1] in {"Easy", "Hard", "Medium"} and shifted[-1]:
                kept.append(shifted)
                continue

        dropped += 1

    # Write repaired file with header
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_lines = [",".join(EXPECTED_HEADER)]
    out_lines += [",".join(row) for row in kept]
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {dst} | kept={len(kept)} dropped={dropped}")

if __name__ == "__main__":
    main()
