from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_assurance.staged_recovery_instrumentation_validation import (
    EQUIVALENCE_FILENAME,
    FIELD_COMPLETENESS_FILENAME,
    OUTPUT_RELATIVE_PATH,
    InstrumentationValidationError,
    validate_published_bundle,
    validate_static_configuration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the frozen Stage 0C instrumentation-validation contract."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-static", action="store_true")
    mode.add_argument("--validate-published", action="store_true")
    mode.add_argument("--print-equivalence", action="store_true")
    mode.add_argument("--print-field-completeness", action="store_true")
    return parser


def _load_published(name: str) -> dict[str, object]:
    path = ROOT / OUTPUT_RELATIVE_PATH / name
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InstrumentationValidationError(f"published JSON must be an object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any(
        (
            args.validate_static,
            args.validate_published,
            args.print_equivalence,
            args.print_field_completeness,
        )
    ):
        parser.print_help()
        return 2
    try:
        if args.validate_static:
            report = validate_static_configuration(ROOT, require_output_absent=True)
            for check_id, passed in report.checks:
                print(f"{check_id} {'PASS' if passed else 'FAIL'}")
            print(f"MANIFEST_HASH {report.manifest_hash}")
            print(f"BRANCH_STATE_HASH {report.branch_state_hash}")
            print(f"STATIC_VALIDATION {'PASS' if report.valid else 'FAIL'}")
            return 0 if report.valid else 1
        if args.validate_published:
            result = validate_published_bundle(ROOT)
            print(f"PUBLISHED_VALIDATION PASS")
            print(f"ARTIFACT_COUNT {len(result.artifact_paths)}")
            print(f"TRACE_MANIFEST_HASH {result.trace_manifest_hash}")
            print(f"TRACE_AGGREGATE_HASH {result.trace_aggregate_hash}")
            return 0
        if args.print_equivalence:
            document = _load_published(EQUIVALENCE_FILENAME)
            checks = document.get("checks")
            if not isinstance(checks, list):
                raise InstrumentationValidationError("equivalence checks are missing")
            for item in checks:
                if not isinstance(item, dict):
                    raise InstrumentationValidationError("equivalence check is invalid")
                print(f"{item['check_id']} {'PASS' if item['passed'] else 'FAIL'}")
            print(
                "ALL_EQUIVALENCE_CHECKS "
                + ("true" if document.get("all_equivalence_checks") is True else "false")
            )
            return 0
        document = _load_published(FIELD_COMPLETENESS_FILENAME)
        totals = document.get("totals")
        if not isinstance(totals, dict):
            raise InstrumentationValidationError("field completeness totals are missing")
        for key in sorted(totals):
            print(f"{key.upper()} {totals[key]}")
        return 0
    except (InstrumentationValidationError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
