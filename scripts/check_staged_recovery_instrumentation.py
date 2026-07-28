from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_assurance.staged_recovery_contract import (
    validate_architecture_manifest_document,
)
from runtime_assurance.staged_recovery_instrumentation import (
    SOURCE_ARCHITECTURE_ARTIFACT_HASHES,
    SOURCE_ARCHITECTURE_CANONICAL_HASH,
    architecture_signal_coverage,
    coverage_counts,
    derivation_traceability_document,
    field_catalog_document,
    instrumentation_manifest_document,
    validate_instrumentation_contract,
)


INSTRUMENTATION_DIRECTORY = Path("analysis/staged_recovery_instrumentation_v0")
MANIFEST_PATH = INSTRUMENTATION_DIRECTORY / "instrumentation_manifest.json"
FIELD_CATALOG_PATH = INSTRUMENTATION_DIRECTORY / "field_catalog.json"
TRACEABILITY_PATH = INSTRUMENTATION_DIRECTORY / "derivation_traceability.json"
SOURCE_ARCHITECTURE_MANIFEST_PATH = Path(
    "analysis/staged_recovery_architecture_v0/architecture_manifest.json"
)

_PROTECTED_GROUPS = {
    "historical_phase34_37": (
        "analysis/phase34_post_cross_sync",
        "analysis/phase35_crossing_basin_expansion",
        "analysis/phase36b_transfer_family_benchmark",
        "analysis/phase36c_non_crossing_geometry_diagnosis",
        "analysis/phase37a_radial_commit_timing",
        "analysis/phase37b_weak_tangential_subset",
        "scripts/check_phase_results.py",
    ),
    "final_veto_v0": ("analysis/final_veto_ablation_v0",),
    "frozen_recovery_inputs": (
        "analysis/recovery_action_branching_nonformal_v0/manifest.json",
        "analysis/recovery_action_branching_nonformal_v0/branch_state.json",
    ),
    "published_recovery_results": (
        "analysis/recovery_action_branching_nonformal_v0/results.csv",
        "analysis/recovery_action_branching_nonformal_v0/decision_log.jsonl",
        "analysis/recovery_action_branching_nonformal_v0/summary.md",
        "analysis/recovery_action_branching_nonformal_v0/comparison.png",
    ),
    "recovery_mechanism_diagnosis_v0": (
        "analysis/recovery_branch_mechanism_diagnosis_v0",
    ),
    "staged_recovery_architecture_v0": (
        "runtime_assurance/staged_recovery_contract.py",
        "Tests/test_staged_recovery_contract.py",
        "docs/architecture/staged_recovery_architecture_v0.md",
        "docs/experiments/staged_recovery_minimal_experiment_plan_v0.md",
        "analysis/staged_recovery_architecture_v0",
    ),
}

_EXPECTED_PROTECTED_HASHES = {
    "historical_phase34_37": (
        "5be1ab928ad018c433c97869a3ffb7ad796ba8a49c9eeaedd901a410245a8501"
    ),
    "final_veto_v0": (
        "125a3f064e288eca471553b8335c757501c9c771f92aa4a711d88e6faba8fb95"
    ),
    "frozen_recovery_inputs": (
        "241c780206bab4a1ca892159a9c310852cf538cd5f7efa254e0bf9c83d258622"
    ),
    "published_recovery_results": (
        "1745d0c307547f0285793db6e64019ca03d1cc40f23829d68073b5e01ee49037"
    ),
    "recovery_mechanism_diagnosis_v0": (
        "482c4f75cc85197f4057d6b77cf24617bc51aae8937f174eb6c25b65ecbecd02"
    ),
    "staged_recovery_architecture_v0": (
        "68aa75b26ad1e9d4e9a7f1eba168d13cdf36ae48b44cf6e3885fca95320ad217"
    ),
}


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path.as_posix()}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.as_posix()} must contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate_paths_hash(root: Path, relative_paths: Sequence[str]) -> str:
    files: list[Path] = []
    for relative_path in relative_paths:
        path = (root / PurePosixPath(relative_path)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"protected path escapes repository: {relative_path}") from exc
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(child for child in path.rglob("*") if child.is_file())
        else:
            raise ValueError(f"protected path is missing: {relative_path}")
    lines = []
    for path in sorted(files, key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        lines.append(f"{relative}|{_file_sha256(path)}")
    return hashlib.sha256(os.linesep.join(lines).encode("utf-8")).hexdigest()


def validate_repository_contract(repository_root: Path) -> tuple[str, ...]:
    root = repository_root.resolve()
    errors: list[str] = []
    try:
        manifest = _load_json(root / MANIFEST_PATH)
        catalog = _load_json(root / FIELD_CATALOG_PATH)
        traceability = _load_json(root / TRACEABILITY_PATH)
    except ValueError as exc:
        return (str(exc),)

    report = validate_instrumentation_contract(
        manifest=manifest,
        catalog=catalog,
        traceability=traceability,
    )
    errors.extend(report.errors)
    if manifest != instrumentation_manifest_document():
        errors.append("instrumentation_manifest_does_not_match_implementation")
    if catalog != field_catalog_document():
        errors.append("field_catalog_does_not_match_implementation")
    if traceability != derivation_traceability_document():
        errors.append("derivation_traceability_does_not_match_implementation")

    architecture_path = root / SOURCE_ARCHITECTURE_MANIFEST_PATH
    try:
        architecture_document = _load_json(architecture_path)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        errors.extend(
            f"source_architecture:{error}"
            for error in validate_architecture_manifest_document(architecture_document)
        )
        if architecture_document.get("canonical_payload_hash") != SOURCE_ARCHITECTURE_CANONICAL_HASH:
            errors.append("source_architecture_canonical_hash_mismatch")

    for relative_path, expected_hash in SOURCE_ARCHITECTURE_ARTIFACT_HASHES:
        path = root / PurePosixPath(relative_path)
        if not path.is_file() or _file_sha256(path) != expected_hash:
            errors.append(f"source_architecture_artifact_hash_mismatch:{relative_path}")

    for group_id, relative_paths in _PROTECTED_GROUPS.items():
        try:
            digest = _aggregate_paths_hash(root, relative_paths)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if digest != _EXPECTED_PROTECTED_HASHES[group_id]:
            errors.append(f"protected_evidence_hash_mismatch:{group_id}")
    return tuple(sorted(set(errors)))


def _print_coverage() -> None:
    counts = dict(coverage_counts())
    print(f"ARCHITECTURE_DECLARED_SIGNALS {len(architecture_signal_coverage())}")
    for classification, count in coverage_counts():
        print(f"{classification.upper()} {count}")
    pure_schema_count = sum(counts.values())
    runner_integration_count = (
        counts["requires_runtime_phase_integration"]
        + counts["requires_future_evaluator"]
        + counts["not_yet_supported"]
    )
    print(f"PURE_SCHEMA_REPRESENTED {pure_schema_count}")
    print(f"REQUIRES_RUNNER_OR_FUTURE_LOGIC {runner_integration_count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the import-pure Staged Recovery Instrumentation v0 contract. "
            "No mode executes a transition or writes a file."
        )
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--print-coverage", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.validate_only and not args.print_coverage:
        parser.print_help()
        return 2
    repository_root = Path(__file__).resolve().parents[1]
    errors = validate_repository_contract(repository_root)
    if errors:
        print("STAGED_RECOVERY_INSTRUMENTATION_VALIDATION FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    if args.print_coverage:
        _print_coverage()
        return 0
    print("STAGED_RECOVERY_INSTRUMENTATION_VALIDATION PASS")
    print(f"ARCHITECTURE_DECLARED_SIGNALS {len(architecture_signal_coverage())}")
    print("RUNTIME_LOGGER_INTEGRATION not_implemented")
    print("STAGED_RECOVERY_EXECUTION not_authorized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
