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
    ANALYSIS_MANIFEST_FILENAME,
    NO_PROGRESS_FILENAME,
    OUTPUT_RELATIVE_PATH,
    PHASE_MATRIX_FILENAME,
    SIGNAL_PROFILE_FILENAME,
    GuardEvidenceError,
    validate_published_analysis,
    validate_static_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the Stage 1A offline guard-evidence contract and artifacts."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-static", action="store_true")
    mode.add_argument("--validate-published", action="store_true")
    mode.add_argument("--print-signal-summary", action="store_true")
    mode.add_argument("--print-phase-observability", action="store_true")
    mode.add_argument("--print-unresolved-parameters", action="store_true")
    return parser


def _load(name: str) -> dict[str, object]:
    path = ROOT / OUTPUT_RELATIVE_PATH / name
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GuardEvidenceError(f"published artifact must be an object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any(
        (
            args.validate_static,
            args.validate_published,
            args.print_signal_summary,
            args.print_phase_observability,
            args.print_unresolved_parameters,
        )
    ):
        parser.print_help()
        return 2
    try:
        if args.validate_static:
            report = validate_static_contract(ROOT, require_output_absent=False)
            for error in report.errors:
                print(f"FAIL {error}")
            print(f"SOURCE_EVENTS {report.source_event_count}")
            print(f"SOURCE_TRANSITIONS {report.source_transition_count}")
            print(f"STATIC_VALIDATION {'PASS' if report.valid else 'FAIL'}")
            print("PHASE_GUARD_POLICY not_frozen")
            print("STAGED_RECOVERY_EXECUTION not_authorized")
            return 0 if report.valid else 1
        if args.validate_published:
            result = validate_published_analysis(ROOT)
            print("PUBLISHED_ANALYSIS PASS")
            print(f"ARTIFACT_COUNT {len(result.artifact_paths)}")
            print(f"ANALYSIS_MANIFEST_HASH {result.analysis_manifest_hash}")
            print(f"GUARD_TRACE_AGGREGATE_HASH {result.guard_trace_aggregate_hash}")
            return 0
        if args.print_signal_summary:
            document = _load(SIGNAL_PROFILE_FILENAME)
            print(f"PROFILE_COUNT {document['profile_count']}")
            for key, value in sorted(document["profile_counts"].items()):
                print(f"{key.upper()} {value}")
            return 0
        if args.print_phase_observability:
            document = _load(PHASE_MATRIX_FILENAME)
            for item in document["phases"]:
                print(
                    f"{item['phase_id']} {item['current_observability_status']} "
                    f"policy={item['policy_authorization']} action_law={str(item['action_law_complete']).lower()}"
                )
            for key, value in sorted(document["totals"].items()):
                print(f"{key.upper()} {value}")
            return 0
        document = _load(NO_PROGRESS_FILENAME)
        parameters = document.get("unresolved_parameters")
        if not isinstance(parameters, list):
            raise GuardEvidenceError("unresolved parameter inventory is missing")
        for item in parameters:
            print(
                f"{item['parameter_id']} {item['current_status']} "
                f"authorization={item['authorization_status']}"
            )
        print(f"UNRESOLVED_PARAMETER_COUNT {len(parameters)}")
        print("SELECTED_PARAMETER_COUNT 0")
        return 0
    except (GuardEvidenceError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
