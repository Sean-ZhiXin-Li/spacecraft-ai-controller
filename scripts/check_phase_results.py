from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_rows(relative_path: str) -> list[dict[str, str]]:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise AssertionError(f"missing required artifact: {relative_path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def count(rows: Iterable[dict[str, str]], field: str) -> int:
    return sum(truth(row.get(field, "")) for row in rows)


def check_equal(label: str, actual: object, expected: object, failures: list[str]) -> None:
    if actual == expected:
        print(f"PASS {label}: {actual}")
    else:
        message = f"FAIL {label}: expected {expected}, got {actual}"
        print(message)
        failures.append(message)


def check_phase34(failures: list[str]) -> None:
    rows = read_rows("analysis/phase34_post_cross_sync/phase34_results.csv")
    radius_priority = [
        row
        for row in rows
        if row.get("controller_name") == "phase34_post_cross_sync"
        and row.get("post_cross_mode") == "radius_priority"
    ]
    baseline = [
        row
        for row in rows
        if row.get("controller_name") == "phase31_phase22_baseline_reference"
    ]

    check_equal("Phase34 radius_priority cases", len(radius_priority), 24, failures)
    check_equal("Phase34 radius_priority crossings", count(radius_priority, "crossing_occurs"), 8, failures)
    check_equal("Phase34 radius_priority recoverable crossings", count(radius_priority, "recoverable_crossing"), 8, failures)
    check_equal("Phase34 reduced Phase31-style baseline cases", len(baseline), 24, failures)
    check_equal("Phase34 reduced Phase31-style baseline crossings", count(baseline, "crossing_occurs"), 8, failures)
    check_equal("Phase34 reduced Phase31-style baseline recoverable crossings", count(baseline, "recoverable_crossing"), 0, failures)


def check_phase36b(failures: list[str]) -> None:
    rows = read_rows("analysis/phase36b_transfer_family_benchmark/phase36b_results.csv")
    expected_families = {
        "baseline_phase34",
        "spiral_approach",
        "grazing_corridor",
        "redesigned_delayed_crossing",
    }
    families = {row.get("transfer_family") for row in rows}
    check_equal("Phase36B families", families, expected_families, failures)
    for family in sorted(expected_families):
        subset = [row for row in rows if row.get("transfer_family") == family]
        check_equal(f"Phase36B {family} cases", len(subset), 24, failures)
        check_equal(f"Phase36B {family} crossings", count(subset, "crossing_occurs"), 8, failures)
        check_equal(f"Phase36B {family} recoverable crossings", count(subset, "recoverable_crossing"), 8, failures)
        check_equal(f"Phase36B {family} overspeed", count(subset, "overspeed"), 0, failures)
        check_equal(f"Phase36B {family} instability", count(subset, "instability"), 0, failures)


def check_phase36c(failures: list[str]) -> None:
    rows = read_rows("analysis/phase36c_non_crossing_geometry_diagnosis/non_crossing_case_set.csv")
    labels = Counter(row.get("baseline_dominant_failure_label", "") for row in rows)
    check_equal("Phase36C baseline non-crossing cases", len(rows), 16, failures)
    check_equal("Phase36C near_crossing labels", labels.get("near_crossing", 0), 8, failures)
    check_equal("Phase36C over_conservative_transfer labels", labels.get("over_conservative_transfer", 0), 8, failures)


def check_phase37a(failures: list[str]) -> None:
    rows = read_rows("analysis/phase37a_radial_commit_timing/phase37a_results.csv")
    check_equal("Phase37A total rows", len(rows), 144, failures)
    check_equal(
        "Phase37A new crossings on baseline non-crossing cases",
        count(rows, "new_crossing_on_baseline_non_crossing_case"),
        0,
        failures,
    )
    check_equal("Phase37A total overspeed", count(rows, "overspeed"), 0, failures)
    check_equal("Phase37A total instability", count(rows, "instability"), 0, failures)

    for variant in ["delayed_commit_low", "delayed_commit_medium"]:
        subset = [row for row in rows if row.get("variant_name") == variant]
        check_equal(f"Phase37A {variant} cases", len(subset), 24, failures)
        check_equal(f"Phase37A {variant} crossings", count(subset, "crossing_occurs"), 8, failures)
        check_equal(f"Phase37A {variant} recoverable crossings", count(subset, "recoverable_crossing"), 8, failures)


def main() -> int:
    failures: list[str] = []
    checks: list[Callable[[list[str]], None]] = [
        check_phase34,
        check_phase36b,
        check_phase36c,
        check_phase37a,
    ]
    for check in checks:
        print(f"\n== {check.__name__} ==")
        try:
            check(failures)
        except Exception as exc:  # noqa: BLE001 - command-line guard should report any validation failure.
            message = f"FAIL {check.__name__}: {exc}"
            print(message)
            failures.append(message)

    if failures:
        print(f"\nResult regression guard FAILED with {len(failures)} issue(s).")
        return 1

    print("\nResult regression guard PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
