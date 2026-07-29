from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_assurance.staged_recovery_runtime_logger import (
    REAL_RUNNER_INTEGRATION_STATUS,
    REAL_TRACE_VALIDATION_STATUS,
    SOURCE_ARCHITECTURE_CANONICAL_HASH,
    SOURCE_INSTRUMENTATION_ARTIFACT_HASHES,
    SOURCE_INSTRUMENTATION_CANONICAL_HASH,
    STAGED_EXECUTION_STATUS,
    event_schema_document,
    field_coverage_document,
    integration_contract_document,
    logger_manifest_document,
    protected_trace_paths,
    validate_logger_contract_documents,
)


ROOT = PROJECT_ROOT
ARTIFACT_DIRECTORY = ROOT / "analysis" / "staged_recovery_runtime_logger_v0"
ARTIFACT_PATHS = {
    "manifest": ARTIFACT_DIRECTORY / "logger_manifest.json",
    "event_schema": ARTIFACT_DIRECTORY / "event_schema.json",
    "integration_contract": ARTIFACT_DIRECTORY / "integration_contract.json",
    "field_coverage": ARTIFACT_DIRECTORY / "field_coverage.json",
}
EXPECTED_METADATA_FILES = {
    "logger_manifest.json",
    "event_schema.json",
    "integration_contract.json",
    "field_coverage.json",
    "summary.md",
}


def _read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_repository_contract() -> tuple[str, ...]:
    errors: list[str] = []
    documents: dict[str, Mapping[str, object]] = {}
    for name, path in ARTIFACT_PATHS.items():
        if not path.is_file():
            errors.append(f"missing_artifact:{path.relative_to(ROOT).as_posix()}")
            continue
        try:
            documents[name] = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid_json:{name}:{exc}")
    if len(documents) == len(ARTIFACT_PATHS):
        report = validate_logger_contract_documents(
            documents["manifest"],
            documents["event_schema"],
            documents["integration_contract"],
            documents["field_coverage"],
        )
        errors.extend(report.errors)
    expected_documents = {
        "manifest": logger_manifest_document(),
        "event_schema": event_schema_document(),
        "integration_contract": integration_contract_document(),
        "field_coverage": field_coverage_document(),
    }
    for name, expected in expected_documents.items():
        if name in documents and documents[name] != expected:
            errors.append(f"generated_contract_mismatch:{name}")
    for relative, expected_hash in SOURCE_INSTRUMENTATION_ARTIFACT_HASHES:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing_stage0a_source:{relative}")
        elif _sha256(path) != expected_hash:
            errors.append(f"stage0a_source_hash_mismatch:{relative}")
    instrumentation_manifest = (
        ROOT
        / "analysis"
        / "staged_recovery_instrumentation_v0"
        / "instrumentation_manifest.json"
    )
    architecture_manifest = (
        ROOT
        / "analysis"
        / "staged_recovery_architecture_v0"
        / "architecture_manifest.json"
    )
    if instrumentation_manifest.is_file():
        document = _read_json(instrumentation_manifest)
        if document.get("canonical_payload_hash") != SOURCE_INSTRUMENTATION_CANONICAL_HASH:
            errors.append("source_instrumentation_canonical_hash_mismatch")
    if architecture_manifest.is_file():
        document = _read_json(architecture_manifest)
        if document.get("canonical_payload_hash") != SOURCE_ARCHITECTURE_CANONICAL_HASH:
            errors.append("source_architecture_canonical_hash_mismatch")
    if ARTIFACT_DIRECTORY.is_dir():
        unexpected = sorted(
            path.name
            for path in ARTIFACT_DIRECTORY.iterdir()
            if path.name not in EXPECTED_METADATA_FILES
        )
        if unexpected:
            errors.append(f"real_or_unexpected_trace_artifact_present:{unexpected}")
    manifest = documents.get("manifest", {})
    if manifest.get("real_runner_integration") != REAL_RUNNER_INTEGRATION_STATUS:
        errors.append("real_runner_integration_status_drift")
    if manifest.get("real_trace_validation") != REAL_TRACE_VALIDATION_STATUS:
        errors.append("real_trace_validation_status_drift")
    if manifest.get("staged_recovery_execution") != STAGED_EXECUTION_STATUS:
        errors.append("staged_execution_status_drift")
    protected = protected_trace_paths(ROOT)
    required = {
        (ROOT / "analysis" / "final_veto_ablation_v0").resolve(),
        (ROOT / "analysis" / "recovery_action_branching_nonformal_v0").resolve(),
        (ROOT / "analysis" / "staged_recovery_instrumentation_v0").resolve(),
        (ROOT / "runtime_assurance").resolve(),
    }
    if not required.issubset(set(protected)):
        errors.append("protected_path_policy_incomplete")
    return tuple(sorted(set(errors)))


def _print_coverage() -> None:
    document = field_coverage_document()
    print(f"ARCHITECTURE_SIGNALS {document['architecture_signal_count']}")
    print(f"STAGE0A_FIELDS {document['stage0a_field_count']}")
    for classification, count in document["logger_coverage_counts"]:
        print(f"{str(classification).upper()} {count}")
    print(f"REAL_TRACE_HAS_VALIDATED {str(document['real_trace_has_validated']).lower()}")


def _print_integration_contract() -> None:
    document = integration_contract_document()
    print("EVENT_TYPES initial_snapshot transition terminal")
    print("REQUIRED_CALL_ORDER " + " -> ".join(document["required_call_order"]))
    for event in document["event_contracts"]:
        print(f"EVENT {event['event_type']}")
        print("  CALLER_OWNED " + "; ".join(event["caller_owned_fields"]))
        print("  LOGGER_DERIVED " + "; ".join(event["logger_derived_fields"]))
    print("REAL_RUNNER_CONNECTED false")
    print("STAGED_RECOVERY_EXECUTION not_authorized")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Stage 0B observational runtime logger contract; no runner "
            "or simulator execution is available."
        )
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--print-coverage", action="store_true")
    modes.add_argument("--print-integration-contract", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any(
        (args.validate_only, args.print_coverage, args.print_integration_contract)
    ):
        parser.print_help()
        return 0
    if args.print_coverage:
        _print_coverage()
        return 0
    if args.print_integration_contract:
        _print_integration_contract()
        return 0
    errors = validate_repository_contract()
    if errors:
        print("STAGED_RECOVERY_RUNTIME_LOGGER INVALID")
        for error in errors:
            print(f"ERROR {error}")
        return 1
    manifest_hash = logger_manifest_document()["canonical_payload_hash"]
    print("STAGED_RECOVERY_RUNTIME_LOGGER VALID")
    print(f"LOGGER_MANIFEST_CANONICAL_HASH {manifest_hash}")
    print("REAL_RUNNER_INTEGRATION not_implemented")
    print("REAL_TRACE_VALIDATION not_performed")
    print("STAGED_RECOVERY_EXECUTION not_authorized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
