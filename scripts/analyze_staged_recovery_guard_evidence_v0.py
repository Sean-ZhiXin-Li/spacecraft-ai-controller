from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_assurance.staged_recovery_guard_evidence import (
    ANALYSIS_ID,
    OUTPUT_RELATIVE_PATH,
    PUBLISHED_FILENAMES,
    SOURCE_BRANCH_ID,
    SOURCE_EVENT_COUNT,
    SOURCE_SEED,
    SOURCE_STAGE0C_DIRECTORY,
    SOURCE_TRACE_AGGREGATE_HASH,
    SOURCE_TRACE_MANIFEST_CANONICAL_HASH,
    SOURCE_TRANSITION_COUNT,
    GuardEvidenceError,
    build_analysis_payloads,
    publish_analysis_payloads,
    require_clean_committed_repository,
    validate_static_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, validate, or execute the frozen Stage 1A offline guard-evidence analysis."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute-frozen-analysis", action="store_true")
    return parser


def _plan() -> dict[str, object]:
    return {
        "analysis_id": ANALYSIS_ID,
        "source_stage0c_directory": SOURCE_STAGE0C_DIRECTORY.as_posix(),
        "source_trace_manifest_canonical_hash": SOURCE_TRACE_MANIFEST_CANONICAL_HASH,
        "source_trace_aggregate_hash": SOURCE_TRACE_AGGREGATE_HASH,
        "source_event_count": SOURCE_EVENT_COUNT,
        "source_transition_count": SOURCE_TRANSITION_COUNT,
        "source_branch": SOURCE_BRANCH_ID,
        "source_seed": SOURCE_SEED,
        "window_lengths": list(range(1, SOURCE_TRANSITION_COUNT + 1)),
        "output_directory": OUTPUT_RELATIVE_PATH.as_posix(),
        "expected_artifacts": list(PUBLISHED_FILENAMES),
        "new_runtime_execution": False,
        "new_measured_trace": False,
        "staged_recovery_execution": "not_authorized",
        "claim_restrictions": [
            "no recovery-performance claim",
            "no phase-policy validity claim",
            "no threshold or hysteresis selection",
            "no formal safety claim",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.plan, args.validate_only, args.execute_frozen_analysis)):
        parser.print_help()
        return 2
    try:
        if args.plan:
            print(json.dumps(_plan(), ensure_ascii=True, sort_keys=True, indent=2))
            return 0
        if args.validate_only:
            report = validate_static_contract(ROOT, require_output_absent=True)
            print(f"STATIC_VALIDATION {'PASS' if report.valid else 'FAIL'}")
            print(f"SOURCE_EVENTS {report.source_event_count}")
            print(f"SOURCE_TRANSITIONS {report.source_transition_count}")
            print("SIMULATOR_EXECUTION false")
            return 0 if report.valid else 1

        implementation_commit = require_clean_committed_repository(ROOT)
        report = validate_static_contract(ROOT, require_output_absent=True)
        if not report.valid:
            raise GuardEvidenceError(
                "static validation failed: " + ",".join(report.errors)
            )
        payloads = build_analysis_payloads(
            ROOT, implementation_commit=implementation_commit
        )
        publication = publish_analysis_payloads(ROOT, payloads)
        print(f"ANALYSIS_ID {ANALYSIS_ID}")
        print(f"IMPLEMENTATION_COMMIT {implementation_commit}")
        print(f"SOURCE_EVENTS {report.source_event_count}")
        print(f"SOURCE_TRANSITIONS {report.source_transition_count}")
        print(f"ARTIFACT_COUNT {len(publication.artifact_paths)}")
        print(f"ANALYSIS_MANIFEST_HASH {publication.analysis_manifest_hash}")
        print(f"GUARD_TRACE_AGGREGATE_HASH {publication.guard_trace_aggregate_hash}")
        print(f"PUBLISHED_DIRECTORY {publication.target_directory}")
        print("NEW_RUNTIME_EXECUTION false")
        print("NEW_MEASURED_TRACE false")
        print("STAGED_RECOVERY_EXECUTION not_authorized")
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("NO_AUTOMATIC_RETRY true", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
