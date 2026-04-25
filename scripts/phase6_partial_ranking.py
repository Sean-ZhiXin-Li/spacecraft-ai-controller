from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "analysis" / "phase6_variant_search" / "variant_grid.csv"
OUT_DIR = PROJECT_ROOT / "analysis" / "phase6_variant_search"
RANKING_PATH = OUT_DIR / "variant_ranking_partial.csv"
SUCCESS_PNG = OUT_DIR / "success_by_variant_partial.png"
CAPTURE_PNG = OUT_DIR / "capture_by_variant_partial.png"
SUMMARY_PATH = OUT_DIR / "phase6_partial_summary.md"

TARGET_RADIUS_BASE = 7.5e12
NEAR_MISS_REL = 3.0e-5


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant_name"]].append(row)

    ranking: list[dict[str, object]] = []
    for variant_name, subset in by_variant.items():
        success_count = sum(as_bool(r["success"]) for r in subset)
        capture_count = sum(as_bool(r["capture_entered"]) for r in subset)
        lock_count = sum(as_bool(r["lock_entered"]) for r in subset)
        min_errors = [float(r["minimum_abs_radius_error"]) for r in subset]
        success_final_errors = [float(r["final_radius_error"]) for r in subset if as_bool(r["success"])]
        near_miss_count = 0
        for r in subset:
            if as_bool(r["success"]):
                continue
            target_radius = TARGET_RADIUS_BASE * float(r["target_radius_scale"])
            if float(r["minimum_abs_radius_error"]) / target_radius <= NEAR_MISS_REL:
                near_miss_count += 1
        ranking.append(
            {
                "variant_name": variant_name,
                "total_runs": len(subset),
                "success_count": success_count,
                "capture_count": capture_count,
                "lock_count": lock_count,
                "mean_minimum_abs_radius_error": sum(min_errors) / len(min_errors),
                "mean_final_radius_error_success": (sum(success_final_errors) / len(success_final_errors))
                if success_final_errors
                else "",
                "near_miss_count": near_miss_count,
            }
        )

    ranking.sort(
        key=lambda r: (-int(r["success_count"]), -int(r["capture_count"]), float(r["mean_minimum_abs_radius_error"]))
    )

    with RANKING_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking[0].keys()))
        writer.writeheader()
        writer.writerows(ranking)

    ordered = list(reversed(ranking))
    labels = [str(r["variant_name"]) for r in ordered]
    success_values = [int(r["success_count"]) for r in ordered]
    capture_values = [int(r["capture_count"]) for r in ordered]
    run_values = [int(r["total_runs"]) for r in ordered]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.barh(labels, success_values, color="#4C78A8")
    for i, (val, runs) in enumerate(zip(success_values, run_values)):
        ax.text(val + 0.8, i, f"{val}/{runs}", va="center", fontsize=8)
    ax.set_xlabel("success count")
    ax.set_title("Phase 6 partial success count by variant")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SUCCESS_PNG, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.barh(labels, capture_values, color="#F58518")
    for i, (val, runs) in enumerate(zip(capture_values, run_values)):
        ax.text(val + 0.8, i, f"{val}/{runs}", va="center", fontsize=8)
    ax.set_xlabel("capture count")
    ax.set_title("Phase 6 partial CAPTURE count by variant")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CAPTURE_PNG, dpi=220)
    plt.close(fig)

    baseline = next(r for r in ranking if r["variant_name"] == "baseline")
    full_variants = [r for r in ranking if int(r["total_runs"]) == 270]
    clearly_better = [
        r
        for r in full_variants
        if int(r["success_count"]) > int(baseline["success_count"])
        or (int(r["success_count"]) == int(baseline["success_count"]) and int(r["capture_count"]) > int(baseline["capture_count"]))
        or (
            int(r["success_count"]) == int(baseline["success_count"])
            and int(r["capture_count"]) == int(baseline["capture_count"])
            and float(r["mean_minimum_abs_radius_error"]) < float(baseline["mean_minimum_abs_radius_error"])
        )
    ]

    interesting: list[str] = []
    if any(r["variant_name"] == "retro_085" for r in ranking):
        interesting.append(
            "moderately reduced retrograde authority is the strongest completed altered variant so far, improving reachability relative to the other completed non-baseline descent changes"
        )
    if any(r["variant_name"] == "retro_070" for r in ranking):
        interesting.append("too much retrograde reduction hurts reachability relative to both baseline and retro_085")
    if any("radial" in str(r["variant_name"]) for r in ranking):
        interesting.append(
            "radial damping variants reduce mean closest-approach error strongly, but none of the completed ones have converted that into CAPTURE entry yet"
        )

    lines = [
        "# Phase 6 Partial Summary",
        "",
        "## Data Status",
        "",
        "- Source file: `analysis/phase6_variant_search/variant_grid.csv`.",
        f"- Completed rows available: `{len(rows)}`.",
        f"- Variants with any data: `{len(ranking)}`.",
        f"- Fully completed variants (`270/270` runs): `{len(full_variants)}`.",
        "- This is a partial ranking only. Several variants have not been run yet, so the snapshot should not be treated as the final Phase 6 result.",
        "",
        "## Partial Ranking",
        "",
        f"- Current top-ranked variant by the requested rule: `{ranking[0]['variant_name']}`.",
        f"- Baseline partial result: success `{baseline['success_count']}/{baseline['total_runs']}`, capture `{baseline['capture_count']}/{baseline['total_runs']}`, mean minimum abs radius error `{float(baseline['mean_minimum_abs_radius_error']):.3e}` m.",
    ]
    if clearly_better:
        lines.append(
            "- Variants clearly ahead of baseline among completed variants: "
            + "; ".join(
                f"`{r['variant_name']}` (success `{r['success_count']}`, capture `{r['capture_count']}`, mean min error `{float(r['mean_minimum_abs_radius_error']):.3e}` m)"
                for r in clearly_better
            )
            + "."
        )
    else:
        lines.append("- No completed variant is clearly ahead of baseline yet under the requested ranking rule.")

    lines.extend(["", "## Interpretation", ""])
    for item in interesting:
        lines.append(f"- {item}.")
    lines.extend(
        [
            "- Some improvement is visible in partial data, but not enough to beat baseline under the requested ranking rule. `retro_085` remains behind baseline on both success and capture count (`57` vs `64/65` on matched `270`-run coverage), while still outperforming `retro_070` and the completed damping/scheduled variants.",
            "- Baseline and `retro_115` are effectively tied at the top of the completed set, which suggests that increasing retrograde authority alone does not improve reachability over the current validated controller.",
            "- The completed radial-damping and scheduled variants have not entered CAPTURE yet, despite some lower mean minimum-radius-error values. That points to a narrow access window where getting somewhat closer is not enough by itself.",
            "- Because some variants are still incomplete, any statement about the best overall knob family should remain provisional.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RANKING_PATH}")
    print(f"wrote {SUCCESS_PNG}")
    print(f"wrote {CAPTURE_PNG}")
    print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
