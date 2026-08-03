from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

from runtime_assurance.recovery_branch_state_registry import (
    LEGACY_ARTIFACT_PATH,
    LEGACY_CASE_ID,
    canonical_sha256,
    file_sha256,
)


BOUNDARY_REGISTRY_ID = "recovery_branch_boundary_registry_v0"
BOUNDARY_SCHEMA_VERSION = "recovery_branch_boundary_registry_v0"
BOUNDARY_CONFIG_PATH = Path("configs/recovery_branch_boundary_registry_v0.json")
COMPLETED_DATE = "2026-08-03"
FINAL_VETO_RESULTS_PATH = Path("analysis/final_veto_ablation_v0/results.csv")

LEGACY_FIXED_PREFIX = "legacy_fixed_prefix"
SOURCE_DECLARED_FIXED_PREFIX = "source_declared_fixed_prefix"
MONITOR_OFF_PRETERMINAL_STATE = "monitor_off_preterminal_state"
INELIGIBLE = "ineligible"
BOUNDARY_TYPES = frozenset(
    {
        LEGACY_FIXED_PREFIX,
        SOURCE_DECLARED_FIXED_PREFIX,
        MONITOR_OFF_PRETERMINAL_STATE,
        INELIGIBLE,
    }
)
BOUNDARY_STATUSES = frozenset({"frozen", "ineligible"})
PRETERMINAL_INDEXING_SEMANTICS = (
    "pre_transition_step_N_state_equals_state_after_N_minus_1_realized_transitions;"
    "transition_N_executes_next;terminal_predicates_evaluate_on_transition_N_next_state"
)
PRETERMINAL_STATE_SEMANTICS = (
    "last_complete_valid_pre_transition_state_before_the_frozen_terminal_transition"
)


class RecoveryBranchBoundaryError(ValueError):
    pass


def boundary_registry_payload(document: Mapping[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(dict(document))
    payload.pop("canonical_payload_hash", None)
    return payload


def _sha(value: object, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise RecoveryBranchBoundaryError(f"{name} must be a SHA-256 digest")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RecoveryBranchBoundaryError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _optional_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, name)


@dataclass(frozen=True, slots=True)
class RecoveryBranchBoundary:
    case_id: str
    seed: int
    boundary_type: str
    boundary_status: str
    boundary_transition_count: int | None
    terminal_transition_count: int | None
    preterminal_state_transition_count: int | None
    terminal_reason: str | None
    source_artifact: str
    source_artifact_hash: str
    source_json_path: str
    source_configuration_hash: str
    simulator_configuration_hash: str
    controller_configuration_hash: str
    indexing_semantics: str
    boundary_state_semantics: str
    terminal_evidence_source: str
    terminal_evidence_hash: str
    terminal_evidence_record_hash: str
    terminal_evidence_path: str
    eligibility: bool
    ineligibility_reason: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RecoveryBranchBoundary":
        try:
            boundary = cls(
                case_id=str(value["case_id"]),
                seed=_integer(value["seed"], "boundary.seed"),
                boundary_type=str(value["boundary_type"]),
                boundary_status=str(value["boundary_status"]),
                boundary_transition_count=_optional_integer(
                    value["boundary_transition_count"],
                    "boundary.boundary_transition_count",
                ),
                terminal_transition_count=_optional_integer(
                    value["terminal_transition_count"],
                    "boundary.terminal_transition_count",
                ),
                preterminal_state_transition_count=_optional_integer(
                    value["preterminal_state_transition_count"],
                    "boundary.preterminal_state_transition_count",
                ),
                terminal_reason=(
                    None if value["terminal_reason"] is None else str(value["terminal_reason"])
                ),
                source_artifact=str(value["source_artifact"]),
                source_artifact_hash=_sha(
                    value["source_artifact_hash"], "boundary.source_artifact_hash"
                ),
                source_json_path=str(value["source_json_path"]),
                source_configuration_hash=_sha(
                    value["source_configuration_hash"],
                    "boundary.source_configuration_hash",
                ),
                simulator_configuration_hash=_sha(
                    value["simulator_configuration_hash"],
                    "boundary.simulator_configuration_hash",
                ),
                controller_configuration_hash=_sha(
                    value["controller_configuration_hash"],
                    "boundary.controller_configuration_hash",
                ),
                indexing_semantics=str(value["indexing_semantics"]),
                boundary_state_semantics=str(value["boundary_state_semantics"]),
                terminal_evidence_source=str(value["terminal_evidence_source"]),
                terminal_evidence_hash=_sha(
                    value["terminal_evidence_hash"], "boundary.terminal_evidence_hash"
                ),
                terminal_evidence_record_hash=_sha(
                    value["terminal_evidence_record_hash"],
                    "boundary.terminal_evidence_record_hash",
                ),
                terminal_evidence_path=str(value["terminal_evidence_path"]),
                eligibility=cast(bool, value["eligibility"]),
                ineligibility_reason=(
                    None
                    if value["ineligibility_reason"] is None
                    else str(value["ineligibility_reason"])
                ),
            )
        except KeyError as exc:
            raise RecoveryBranchBoundaryError(
                f"boundary record missing field {exc.args[0]}"
            ) from exc
        boundary.validate()
        return boundary

    def validate(self) -> None:
        if not self.case_id:
            raise RecoveryBranchBoundaryError("boundary case identity is required")
        if self.boundary_type not in BOUNDARY_TYPES:
            raise RecoveryBranchBoundaryError("unsupported boundary type")
        if self.boundary_status not in BOUNDARY_STATUSES:
            raise RecoveryBranchBoundaryError("unsupported boundary status")
        if type(self.eligibility) is not bool:
            raise RecoveryBranchBoundaryError("boundary eligibility must be boolean")
        if not self.source_artifact or not self.source_json_path:
            raise RecoveryBranchBoundaryError("boundary source provenance is incomplete")
        if not self.terminal_evidence_source or not self.terminal_evidence_path:
            raise RecoveryBranchBoundaryError("terminal evidence provenance is incomplete")
        if self.boundary_type == INELIGIBLE:
            if self.eligibility or self.boundary_status != "ineligible":
                raise RecoveryBranchBoundaryError("ineligible boundary status mismatch")
            if not self.ineligibility_reason:
                raise RecoveryBranchBoundaryError("ineligible boundary requires a reason")
            return
        if not self.eligibility or self.boundary_status != "frozen":
            raise RecoveryBranchBoundaryError("eligible boundary must be frozen")
        if self.ineligibility_reason is not None:
            raise RecoveryBranchBoundaryError("eligible boundary may not have an ineligibility reason")
        if self.boundary_transition_count is None:
            raise RecoveryBranchBoundaryError("eligible boundary transition count is required")
        if self.indexing_semantics != PRETERMINAL_INDEXING_SEMANTICS:
            raise RecoveryBranchBoundaryError("boundary indexing semantics mismatch")
        if self.boundary_state_semantics != PRETERMINAL_STATE_SEMANTICS:
            raise RecoveryBranchBoundaryError("boundary state semantics mismatch")
        if self.boundary_type == LEGACY_FIXED_PREFIX:
            if self.case_id != LEGACY_CASE_ID or self.boundary_transition_count != 27:
                raise RecoveryBranchBoundaryError("legacy fixed-prefix boundary mismatch")
        elif self.boundary_type == MONITOR_OFF_PRETERMINAL_STATE:
            if self.terminal_transition_count is None or self.terminal_reason is None:
                raise RecoveryBranchBoundaryError("preterminal boundary lacks terminal evidence")
            if self.preterminal_state_transition_count != self.boundary_transition_count:
                raise RecoveryBranchBoundaryError("preterminal state count differs from boundary count")
            if self.terminal_transition_count != self.boundary_transition_count + 1:
                raise RecoveryBranchBoundaryError(
                    "off-by-one boundary substitution: terminal transition must follow the preterminal state"
                )
        elif self.boundary_type == SOURCE_DECLARED_FIXED_PREFIX:
            if self.boundary_transition_count < 1:
                raise RecoveryBranchBoundaryError("source fixed prefix must be positive")

    @property
    def branch_step(self) -> int:
        if self.boundary_transition_count is None:
            raise RecoveryBranchBoundaryError("ineligible boundary has no branch step")
        return self.boundary_transition_count + 1

    def as_document(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class RecoveryBranchBoundaryRegistry:
    registry_id: str
    schema_version: str
    boundaries: tuple[RecoveryBranchBoundary, ...]
    canonical_payload_hash: str

    def for_case(self, case_id: str) -> RecoveryBranchBoundary:
        matches = tuple(item for item in self.boundaries if item.case_id == case_id)
        if len(matches) != 1:
            raise RecoveryBranchBoundaryError(f"unknown boundary case: {case_id!r}")
        return matches[0]


def _load_terminal_rows(repository_root: Path) -> dict[str, dict[str, str]]:
    path = repository_root / FINAL_VETO_RESULTS_PATH
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("arm_id") == "monitor_off"
        ]
    result = {row["case_id"]: row for row in rows}
    if len(rows) != 13 or len(result) != 13:
        raise RecoveryBranchBoundaryError("Final Veto monitor-off row inventory is not exactly 13")
    return result


def load_recovery_branch_boundary_registry(
    repository_root: Path,
) -> RecoveryBranchBoundaryRegistry:
    root = repository_root.resolve()
    path = root / BOUNDARY_CONFIG_PATH
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryBranchBoundaryError(f"cannot load boundary registry: {exc}") from exc
    if not isinstance(value, dict):
        raise RecoveryBranchBoundaryError("boundary registry must be an object")
    if value.get("registry_id") != BOUNDARY_REGISTRY_ID:
        raise RecoveryBranchBoundaryError("boundary registry identity mismatch")
    if value.get("schema_version") != BOUNDARY_SCHEMA_VERSION:
        raise RecoveryBranchBoundaryError("boundary registry schema mismatch")
    if value.get("completed_date") != COMPLETED_DATE:
        raise RecoveryBranchBoundaryError("boundary registry completed date mismatch")
    supplied_hash = value.get("canonical_payload_hash")
    if supplied_hash != canonical_sha256(boundary_registry_payload(value)):
        raise RecoveryBranchBoundaryError("boundary registry canonical hash mismatch")
    raw_boundaries = value.get("boundaries")
    if not isinstance(raw_boundaries, list):
        raise RecoveryBranchBoundaryError("boundary registry records must be a list")
    boundaries = tuple(
        RecoveryBranchBoundary.from_mapping(item)
        for item in raw_boundaries
        if isinstance(item, dict)
    )
    case_ids = tuple(item.case_id for item in boundaries)
    if len(boundaries) != 13 or len(case_ids) != len(set(case_ids)):
        raise RecoveryBranchBoundaryError("boundary registry must contain 13 unique cases")
    if case_ids != tuple(sorted(case_ids)):
        raise RecoveryBranchBoundaryError("boundary registry ordering is not deterministic")
    if sum(item.boundary_type == LEGACY_FIXED_PREFIX for item in boundaries) != 1:
        raise RecoveryBranchBoundaryError("boundary registry requires one legacy boundary")

    terminal_rows = _load_terminal_rows(root)
    results_hash = file_sha256(root / FINAL_VETO_RESULTS_PATH)
    legacy_hash = file_sha256(root / LEGACY_ARTIFACT_PATH)
    for boundary in boundaries:
        source_path = root / Path(boundary.source_artifact)
        if not source_path.is_file() or file_sha256(source_path) != boundary.source_artifact_hash:
            raise RecoveryBranchBoundaryError(
                f"boundary source artifact hash mismatch: {boundary.case_id}"
            )
        if (
            boundary.terminal_evidence_source != FINAL_VETO_RESULTS_PATH.as_posix()
            or boundary.terminal_evidence_hash != results_hash
        ):
            raise RecoveryBranchBoundaryError(
                f"terminal evidence artifact mismatch: {boundary.case_id}"
            )
        row = terminal_rows.get(boundary.case_id)
        if row is None:
            raise RecoveryBranchBoundaryError(
                f"terminal evidence row missing: {boundary.case_id}"
            )
        if canonical_sha256(row) != boundary.terminal_evidence_record_hash:
            raise RecoveryBranchBoundaryError(
                f"terminal evidence row hash mismatch: {boundary.case_id}"
            )
        if int(row["steps"]) != boundary.terminal_transition_count:
            raise RecoveryBranchBoundaryError(
                f"terminal transition count mismatch: {boundary.case_id}"
            )
        if row["termination_reason"] != boundary.terminal_reason:
            raise RecoveryBranchBoundaryError(
                f"terminal reason mismatch: {boundary.case_id}"
            )
        if boundary.case_id == LEGACY_CASE_ID and boundary.source_artifact_hash != legacy_hash:
            raise RecoveryBranchBoundaryError("legacy boundary artifact hash mismatch")
    return RecoveryBranchBoundaryRegistry(
        registry_id=BOUNDARY_REGISTRY_ID,
        schema_version=BOUNDARY_SCHEMA_VERSION,
        boundaries=boundaries,
        canonical_payload_hash=cast(str, supplied_hash),
    )


__all__ = [
    "BOUNDARY_CONFIG_PATH",
    "BOUNDARY_REGISTRY_ID",
    "BOUNDARY_SCHEMA_VERSION",
    "BOUNDARY_STATUSES",
    "BOUNDARY_TYPES",
    "COMPLETED_DATE",
    "FINAL_VETO_RESULTS_PATH",
    "INELIGIBLE",
    "LEGACY_FIXED_PREFIX",
    "MONITOR_OFF_PRETERMINAL_STATE",
    "PRETERMINAL_INDEXING_SEMANTICS",
    "PRETERMINAL_STATE_SEMANTICS",
    "SOURCE_DECLARED_FIXED_PREFIX",
    "RecoveryBranchBoundary",
    "RecoveryBranchBoundaryError",
    "RecoveryBranchBoundaryRegistry",
    "boundary_registry_payload",
    "load_recovery_branch_boundary_registry",
]
