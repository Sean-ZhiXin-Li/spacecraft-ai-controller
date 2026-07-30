from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_assurance.recovery_experiment_preflight import load_json_object
from runtime_assurance.staged_recovery_instrumentation_validation import (
    BRANCH_STATE_RELATIVE_PATH,
    EXPECTED_EVENT_COUNT,
    MANIFEST_RELATIVE_PATH,
    OUTPUT_RELATIVE_PATH,
    VALIDATION_BRANCH_ID,
    VALIDATION_HORIZON,
    VALIDATION_ID,
    InstrumentationValidationError,
    execute_frozen_validation_pair,
    require_clean_committed_repository,
    validate_static_configuration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, validate, or explicitly execute the frozen Stage 0C paired "
            "instrumentation validation."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute-frozen-validation", action="store_true")
    return parser


def _plan() -> dict[str, object]:
    manifest = load_json_object(ROOT / MANIFEST_RELATIVE_PATH)
    branch_state = load_json_object(ROOT / BRANCH_STATE_RELATIVE_PATH)
    source_case = manifest["source_case"]
    ordering = branch_state["branch_ordering"]
    assert isinstance(source_case, dict) and isinstance(ordering, dict)
    return {
        "validation_id": VALIDATION_ID,
        "case_id": source_case["case_id"],
        "seed": source_case["seed"],
        "branch_id": VALIDATION_BRANCH_ID,
        "branch_state_hash": branch_state["canonical_branch_state_hash"],
        "nominal_prefix_transition_count": ordering[
            "realized_prefix_transition_count"
        ],
        "validation_horizon": VALIDATION_HORIZON,
        "paired_runs": ["logger_disabled_baseline", "logger_enabled_observed"],
        "expected_event_count": EXPECTED_EVENT_COUNT,
        "output_directory": OUTPUT_RELATIVE_PATH.as_posix(),
        "execution_disabled": True,
        "scientific_recovery_result": False,
        "staged_recovery_execution": "not_authorized",
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.plan, args.validate_only, args.execute_frozen_validation)):
        parser.print_help()
        return 2
    try:
        if args.plan:
            print(json.dumps(_plan(), sort_keys=True, indent=2))
            return 0
        if args.validate_only:
            report = validate_static_configuration(ROOT, require_output_absent=True)
            print(f"STATIC_VALIDATION {'PASS' if report.valid else 'FAIL'}")
            print("NO_TRANSITION_EXECUTED true")
            return 0 if report.valid else 1

        implementation_commit = require_clean_committed_repository(ROOT)
        report = validate_static_configuration(ROOT, require_output_absent=True)
        if not report.valid:
            raise InstrumentationValidationError(
                f"static validation failed: {', '.join(report.errors)}"
            )
        result = execute_frozen_validation_pair(
            ROOT,
            implementation_commit=implementation_commit,
        )
        print(f"VALIDATION_ID {VALIDATION_ID}")
        print(f"IMPLEMENTATION_COMMIT {implementation_commit}")
        print(f"BASELINE_TRANSITIONS {result.baseline.recovery_transition_count}")
        print(f"OBSERVED_TRANSITIONS {result.observed.recovery_transition_count}")
        print(f"TRACE_EVENTS {len(result.events)}")
        print(
            "ALL_EQUIVALENCE_CHECKS "
            + ("true" if result.equivalence.all_equivalence_checks else "false")
        )
        print(
            "UNEXPECTEDLY_MISSING_FIELDS "
            f"{result.field_completeness.unexpectedly_missing_fields}"
        )
        print(
            f"INVALID_REQUIRED_FIELDS {result.field_completeness.invalid_required_fields}"
        )
        print(f"TRACE_MANIFEST_HASH {result.publication.trace_manifest_hash}")
        print(f"TRACE_AGGREGATE_HASH {result.publication.trace_aggregate_hash}")
        print(f"PUBLISHED_DIRECTORY {result.publication.target_directory}")
        print("STAGED_RECOVERY_EXECUTION not_authorized")
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("NO_AUTOMATIC_RETRY true", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
