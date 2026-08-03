from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_assurance.recovery_branch_state_extractor import (  # noqa: E402
    BranchStateExtractionError,
    build_frozen_registry_payloads,
    build_source_case_inventory,
    load_registry_config,
    publish_registry_payloads,
    repository_state,
    validate_static_contract,
)
from runtime_assurance.recovery_branch_state_registry import OUTPUT_PATH  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the frozen multi-case recovery branch-state registry. "
            "Default and validation modes execute no transition and write nothing."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute-frozen-registry-generation", action="store_true")
    return parser


def _print_plan() -> None:
    config = load_registry_config(ROOT)
    cases = build_source_case_inventory(ROOT)
    print(f"REPOSITORY_HEAD {repository_state(ROOT)[0]}")
    print(f"SOURCE_INVENTORY {config['source_inventory_path']}")
    print(f"EXPECTED_CASE_COUNT {config['source_case_count']}")
    print("ELIGIBLE_RULE complete frozen initialization, simulator, controller, and source hashes")
    print("SELECTION_MEMBER_A legacy canonical")
    print("SELECTION_MEMBER_B closest predicted ratio <= 1.90")
    print("SELECTION_MEMBER_C closest predicted ratio > 1.90")
    print("SELECTION_MEMBER_D strongest remaining absolute tangential error ratio")
    print(f"BOUNDARY_REGISTRY {config['boundary_registry_path']}")
    for case in cases:
        print(
            f"BOUNDARY {case.case_id} type={case.boundary.boundary_type} "
            f"transition={case.boundary.boundary_transition_count} "
            f"terminal={case.boundary.terminal_transition_count}"
        )
    print("GLOBAL_PREFIX_DEFAULT none")
    print(f"OUTPUT_DIRECTORY {OUTPUT_PATH.as_posix()}")
    print(f"ELIGIBLE_STATIC_COUNT {sum(item.eligible_for_generation for item in cases)}")
    print("RECOVERY_BRANCH_EXECUTION false")
    print("AUTOMATIC_RETRY false")
    print("EXECUTION disabled")


def _print_validation() -> int:
    report = validate_static_contract(ROOT, require_output_absent=True)
    for error in report.errors:
        print(f"FAIL {error}")
    print(f"SOURCE_CASE_COUNT {report.source_case_count}")
    print(f"ELIGIBLE_CASE_COUNT {report.eligible_case_count}")
    print(f"INELIGIBLE_CASE_COUNT {report.ineligible_case_count}")
    print(f"TRACKED_CLEAN {str(report.tracked_clean).lower()}")
    print(f"STAGED_CLEAN {str(report.staged_clean).lower()}")
    print(f"STATIC_VALIDATION {'PASS' if report.valid else 'FAIL'}")
    print("SIMULATION_EXECUTED false")
    print("WRITE_PERFORMED false")
    return 0 if report.valid else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.plan, args.validate_only, args.execute_frozen_registry_generation)):
        parser.print_help()
        return 2
    try:
        if args.plan:
            _print_plan()
            return 0
        if args.validate_only:
            return _print_validation()
        report = validate_static_contract(ROOT, require_output_absent=True)
        if not report.valid:
            raise BranchStateExtractionError(
                "static validation failed: " + "; ".join(report.errors)
            )
        if not report.tracked_clean or not report.staged_clean:
            raise BranchStateExtractionError(
                "frozen generation requires a clean tracked tree and no staged changes"
            )
        payloads, metadata = build_frozen_registry_payloads(
            ROOT, implementation_commit=report.head_commit
        )
        publication = publish_registry_payloads(ROOT, payloads)
        print(f"REGISTRY_DIRECTORY {publication.target_directory}")
        print(f"REGISTRY_MEMBER_COUNT {publication.member_count}")
        print(f"TOTAL_NOMINAL_PREFIX_EXECUTIONS {publication.total_execution_count}")
        print(f"REGISTRY_MANIFEST_HASH {publication.registry_manifest_hash}")
        print(f"REGISTRY_AGGREGATE_HASH {publication.registry_aggregate_hash}")
        print(f"SELECTED_MEMBER_B {metadata['selection'].member_b_case_id}")
        print(f"SELECTED_MEMBER_C {metadata['selection'].member_c_case_id}")
        print(f"SELECTED_MEMBER_D {metadata['selection'].member_d_case_id}")
        print("RECOVERY_BRANCH_EXECUTION false")
        print("AUTOMATIC_RETRY false")
        return 0
    except (BranchStateExtractionError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
