# scripts/validate_ablation_csv.py
# Validate ablation CSV health: required columns, NaNs, duplicates.
# Exit code: 0 for OK, 1 for FAIL.

import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLS = [
    "thrust_newton",
    "difficulty_tag",
    "r0_over_target",
    "avg_radius_error",
    "final_radius_error",
    "saturation_rate_mean",
    "total_reward",
    "dedup_key",
]


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    csv_path = Path("analysis/results/ablation_thrust_x_difficulty_clean.csv")
    if not csv_path.exists():
        fail(f"missing file: {csv_path}")

    df = pd.read_csv(csv_path)

    rows = len(df)
    if rows == 0:
        fail("empty csv (rows=0)")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        fail(f"missing columns: {missing}")

    nan_counts = df[REQUIRED_COLS].isna().sum().to_dict()
    nan_total_reward = int(nan_counts.get("total_reward", 0))
    if nan_total_reward != 0:
        fail(f"nan_total_reward={nan_total_reward}")

    dedup_non_null_ratio = float(df["dedup_key"].notna().mean())
    if dedup_non_null_ratio < 1.0:
        fail(f"dedup_key_non_null_ratio={dedup_non_null_ratio:.3f} (want 1.000)")

    dup_mask = df["dedup_key"].duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    if dup_count != 0:
        sample = df.loc[dup_mask, "dedup_key"].head(5).tolist()
        fail(f"duplicated dedup_key count={dup_count}, sample={sample}")

    dedup_unique = int(df["dedup_key"].nunique())

    print(
        f"[OK] rows={rows} dedup_unique={dedup_unique} "
        f"nan_total_reward={nan_total_reward} dedup_key_non_null_ratio={dedup_non_null_ratio:.3f}"
    )
    print(f"[INFO] nan_counts={nan_counts}")


if __name__ == "__main__":
    main()
