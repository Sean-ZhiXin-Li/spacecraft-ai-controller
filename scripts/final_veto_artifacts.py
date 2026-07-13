from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ARM_SCHEMA_VERSION = "result_schema_v1"
PAIR_SCHEMA_VERSION = "final_veto_pair_schema_v0"
DECISION_SCHEMA_VERSION = "decision_log_schema_v0"

ARM_FIELDNAMES = (
    "schema_version",
    "benchmark_id",
    "benchmark_version",
    "experiment_id",
    "experiment_status",
    "implementation_commit",
    "run_id",
    "paired_run_id",
    "case_config_hash",
    "run_config_hash",
    "subset_id",
    "case_id",
    "arm_id",
    "seed",
    "controller_id",
    "controller_family",
    "artifact_path",
    "source_script",
    "monitor_enabled",
    "monitor_id",
    "hazard_target",
    "hazard_threshold",
    "hazard_comparator",
    "r0_over_target",
    "initial_velocity_angle_deg",
    "thrust_scale",
    "crossed_target_radius",
    "first_crossing_step",
    "recoverable_crossing",
    "final_simulator_success",
    "overspeed",
    "max_speed_ratio",
    "instability",
    "unsafe_state",
    "invalid_simulation",
    "terminal_label",
    "precursor_labels",
    "diagnostic_labels",
    "manual_audit_note",
    "label_taxonomy_version",
    "is_full_benchmark",
    "subset_claim_scope",
    "regression_set_membership",
    "known_phase34_recoverable_case",
    "monitor_evaluation_count",
    "allow_count",
    "veto_count",
    "fallback_count",
    "false_negative_count",
    "fallback_failure_count",
    "invalid_monitor_evaluation_count",
    "nominal_actions_unchanged_count",
    "steps",
    "accepted_as_progress",
    "acceptance_reason",
    "is_formal_experiment",
)

PAIR_FIELDNAMES = (
    "pair_schema_version",
    "experiment_id",
    "paired_run_id",
    "case_id",
    "subset_id",
    "seed",
    "case_config_hash",
    "off_run_id",
    "on_run_id",
    "pair_complete",
    "pair_valid",
    "off_overspeed",
    "on_overspeed",
    "off_crossed_target_radius",
    "on_crossed_target_radius",
    "off_recoverable_crossing",
    "on_recoverable_crossing",
    "off_final_simulator_success",
    "on_final_simulator_success",
    "off_invalid_simulation",
    "on_invalid_simulation",
    "on_monitor_evaluation_count",
    "on_allow_count",
    "on_veto_count",
    "on_fallback_count",
    "on_false_negative_count",
    "on_fallback_failure_count",
    "avoided_failure",
    "blocked_success",
    "unnecessary_veto",
    "performance_cost_summary",
    "claim_eligible",
    "claim_ineligibility_reason",
    "is_formal_experiment",
)

DECISION_FIELDNAMES = (
    "decision_schema_version",
    "decision_id",
    "experiment_id",
    "run_id",
    "paired_run_id",
    "case_id",
    "subset_id",
    "arm_id",
    "step",
    "phase",
    "active_stage",
    "decision_type",
    "decision_reason",
    "decision_scope",
    "decision_authority",
    "monitor_id",
    "state_summary",
    "safety_level",
    "recoverability_level",
    "trust_flags",
    "nominal_proposed_action",
    "executed_action",
    "predicted_nominal_speed_ratio",
    "predicted_fallback_speed_ratio",
    "realized_executed_speed_ratio",
    "hazard_threshold",
    "hazard_comparator",
    "veto_status",
    "veto_reason",
    "fallback_available",
    "fallback_action",
    "fallback_executed",
    "fallback_failure",
    "invalid_evaluation",
    "manual_audit_note",
    "is_formal_experiment",
)

ARM_JSON_LIST_FIELDS = frozenset(
    {"precursor_labels", "diagnostic_labels", "regression_set_membership"}
)


class ArtifactWriteError(ValueError):
    pass


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_directory(
    output_directory: Path,
    *,
    repository_root: Path | None = None,
    protected_paths: Sequence[str] = (),
) -> Path:
    output = output_directory.resolve()
    if repository_root is not None:
        root = repository_root.resolve()
        for relative in protected_paths:
            protected = (root / relative.rstrip("/")).resolve()
            if output == protected or _is_relative_to(output, protected):
                raise ArtifactWriteError(
                    f"output directory overlaps protected path: {relative}"
                )
    return output


def resolve_output_path(
    output_directory: Path,
    relative_path: str | Path,
    *,
    repository_root: Path | None = None,
    protected_paths: Sequence[str] = (),
) -> Path:
    output = validate_output_directory(
        output_directory,
        repository_root=repository_root,
        protected_paths=protected_paths,
    )
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactWriteError("artifact path must be relative and contain no '..'")
    destination = (output / relative).resolve()
    if not _is_relative_to(destination, output):
        raise ArtifactWriteError("artifact path escapes the supplied output directory")
    if repository_root is not None:
        root = repository_root.resolve()
        for protected_text in protected_paths:
            protected = (root / protected_text.rstrip("/")).resolve()
            if destination == protected or _is_relative_to(destination, protected):
                raise ArtifactWriteError(
                    f"artifact path overlaps protected path: {protected_text}"
                )
    return destination


def encode_csv_value(field: str, value: object, list_fields: frozenset[str]) -> object:
    if value is None:
        return ""
    if field in list_fields:
        if not isinstance(value, (list, tuple)):
            raise ArtifactWriteError(f"{field} must be a list or tuple")
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return value


def _open_atomic_text(destination: Path) -> tuple[Path, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    return temporary, os.fdopen(descriptor, "w", encoding="utf-8", newline="")


def write_csv_atomic(
    output_directory: Path,
    filename: str | Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str],
    *,
    list_fields: frozenset[str] = frozenset(),
    repository_root: Path | None = None,
    protected_paths: Sequence[str] = (),
) -> Path:
    destination = resolve_output_path(
        output_directory,
        filename,
        repository_root=repository_root,
        protected_paths=protected_paths,
    )
    if destination.exists():
        raise ArtifactWriteError(f"refusing to overwrite existing artifact: {destination}")

    temporary: Path | None = None
    handle = None
    try:
        temporary, handle = _open_atomic_text(destination)
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: encode_csv_value(field, row.get(field), list_fields)
                    for field in fieldnames
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(temporary, destination)
        temporary = None
    except Exception:
        if handle is not None:
            handle.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


class JsonlEventWriter:
    def __init__(
        self,
        output_directory: Path,
        filename: str | Path,
        *,
        repository_root: Path | None = None,
        protected_paths: Sequence[str] = (),
    ) -> None:
        self.destination = resolve_output_path(
            output_directory,
            filename,
            repository_root=repository_root,
            protected_paths=protected_paths,
        )
        self._temporary: Path | None = None
        self._handle = None

    def __enter__(self) -> "JsonlEventWriter":
        if self.destination.exists():
            raise ArtifactWriteError(
                f"refusing to overwrite existing artifact: {self.destination}"
            )
        self._temporary, self._handle = _open_atomic_text(self.destination)
        return self

    def write_event(self, event: Mapping[str, object]) -> None:
        if self._handle is None:
            raise ArtifactWriteError("JSONL writer is not open")
        self._handle.write(
            json.dumps(event, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        )
        self._handle.write("\n")
        self._handle.flush()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._handle is not None:
            if exc_type is None:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None
        if exc_type is None and self._temporary is not None:
            os.replace(self._temporary, self.destination)
            self._temporary = None
        elif self._temporary is not None:
            self._temporary.unlink(missing_ok=True)
            self._temporary = None
        return False


def write_jsonl_atomic(
    output_directory: Path,
    filename: str | Path,
    events: Iterable[Mapping[str, object]],
    *,
    repository_root: Path | None = None,
    protected_paths: Sequence[str] = (),
) -> Path:
    with JsonlEventWriter(
        output_directory,
        filename,
        repository_root=repository_root,
        protected_paths=protected_paths,
    ) as writer:
        for event in events:
            writer.write_event(event)
        return writer.destination


__all__ = [
    "ARM_FIELDNAMES",
    "ARM_JSON_LIST_FIELDS",
    "ARM_SCHEMA_VERSION",
    "ArtifactWriteError",
    "DECISION_FIELDNAMES",
    "DECISION_SCHEMA_VERSION",
    "JsonlEventWriter",
    "PAIR_FIELDNAMES",
    "PAIR_SCHEMA_VERSION",
    "encode_csv_value",
    "resolve_output_path",
    "validate_output_directory",
    "write_csv_atomic",
    "write_jsonl_atomic",
]
