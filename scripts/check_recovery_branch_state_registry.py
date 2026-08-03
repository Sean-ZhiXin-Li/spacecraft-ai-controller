from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_assurance.recovery_branch_state_extractor import (  # noqa: E402
    BranchStateExtractionError,
    source_inventory_document,
    validate_published_registry,
    validate_static_contract,
)
from runtime_assurance.recovery_branch_state_registry import (  # noqa: E402
    OUTPUT_PATH,
    load_branch_state_registry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only validator for Recovery Branch-State Registry v0."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-static", action="store_true")
    mode.add_argument("--validate-published", action="store_true")
    mode.add_argument("--print-source-inventory", action="store_true")
    mode.add_argument("--print-selection", action="store_true")
    mode.add_argument("--print-determinism", action="store_true")
    mode.add_argument("--print-members", action="store_true")
    return parser


def _load_published(name: str) -> dict[str, object]:
    path = ROOT / OUTPUT_PATH / name
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BranchStateExtractionError(f"published artifact is not an object: {name}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any(
        (
            args.validate_static,
            args.validate_published,
            args.print_source_inventory,
            args.print_selection,
            args.print_determinism,
            args.print_members,
        )
    ):
        parser.print_help()
        return 2
    try:
        if args.validate_static:
            report = validate_static_contract(ROOT, require_output_absent=False)
            for error in report.errors:
                print(f"FAIL {error}")
            print(f"SOURCE_CASE_COUNT {report.source_case_count}")
            print(f"ELIGIBLE_CASE_COUNT {report.eligible_case_count}")
            print(f"INELIGIBLE_CASE_COUNT {report.ineligible_case_count}")
            print(f"STATIC_VALIDATION {'PASS' if report.valid else 'FAIL'}")
            return 0 if report.valid else 1
        if args.validate_published:
            result = validate_published_registry(ROOT)
            print("PUBLISHED_REGISTRY PASS")
            print(f"ARTIFACT_COUNT {len(result.artifact_paths)}")
            print(f"REGISTRY_MEMBER_COUNT {result.member_count}")
            print(f"TOTAL_NOMINAL_PREFIX_EXECUTIONS {result.total_execution_count}")
            print(f"REGISTRY_MANIFEST_HASH {result.registry_manifest_hash}")
            print(f"REGISTRY_AGGREGATE_HASH {result.registry_aggregate_hash}")
            return 0
        if args.print_source_inventory:
            document = (
                _load_published("source_case_inventory.json")
                if (ROOT / OUTPUT_PATH).is_dir()
                else source_inventory_document(ROOT)
            )
            for item in document["cases"]:
                print(
                    f"{item['case_id']} eligible={str(item['eligible_for_generation']).lower()} "
                    f"boundary={item.get('boundary_type')} "
                    f"transition={item.get('boundary_transition_count')} "
                    f"terminal={item.get('terminal_transition_count')} "
                    f"predicted={item.get('predicted_speed_ratio_if_available')}"
                )
            print(f"SOURCE_CASE_COUNT {document['source_case_count']}")
            print(f"ELIGIBLE_CASE_COUNT {document['eligible_case_count']}")
            print(f"INELIGIBLE_CASE_COUNT {document['ineligible_case_count']}")
            return 0
        if args.print_selection:
            document = _load_published("selection_report.json")
            for role in ("a", "b", "c", "d"):
                print(f"MEMBER_{role.upper()} {document[f'selected_member_{role}']}")
            return 0
        if args.print_determinism:
            document = _load_published("determinism_report.json")
            for item in document["members"]:
                print(f"{item['case_id']} {item['determinism_status']}")
            print(f"DETERMINISM_FAILURE_COUNT {document['determinism_failure_count']}")
            print(
                f"CANONICAL_REPRODUCTION_FAILURE_COUNT "
                f"{document['canonical_reproduction_failure_count']}"
            )
            return 0
        registry = load_branch_state_registry(ROOT)
        for member in registry.members:
            print(
                f"{member.registry_member_id} case={member.case_id} "
                f"scope={member.artifact_scope} hash={member.canonical_branch_state_hash}"
            )
        print(f"REGISTRY_MEMBER_COUNT {len(registry.members)}")
        return 0
    except (BranchStateExtractionError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
