from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence, cast

from runtime_assurance.final_veto_monitor import (
    MONITOR_ID,
    OVERSPEED_COMPARATOR,
    OVERSPEED_THRESHOLD,
)
from runtime_assurance.recovery_evaluators import (
    PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
    PHASE34_RECOVERABLE_VR_RATIO_MAX,
    PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
)


REGISTRY_ID = "recovery_branch_state_registry_v0"
REGISTRY_SCHEMA_VERSION = "recovery_branch_state_registry_manifest_v0"
REGISTRY_MEMBER_SCHEMA_VERSION = "recovery_branch_state_registry_member_v0"
CONFIG_SCHEMA_VERSION = "recovery_branch_state_registry_config_v0"
COMPLETED_DATE = "2026-08-02"

LEGACY_MEMBER_ID = "legacy_canonical"
LEGACY_CASE_ID = (
    "phase35_radial_energy_push_overspeed_stress_v0"
    "__r0_0p98__angle_150__thrust_8000"
)
LEGACY_ARTIFACT_PATH = Path(
    "analysis/recovery_action_branching_nonformal_v0/branch_state.json"
)
CONFIG_PATH = Path("configs/recovery_branch_state_registry_v0.json")
OUTPUT_PATH = Path("analysis/recovery_branch_state_registry_v0")
MANIFEST_PATH = OUTPUT_PATH / "registry_manifest.json"

LEGACY_SCHEMA_VERSION = "recovery_branch_state_v0"
PREFIX_TRANSITION_COUNT = 27
BRANCH_STEP = PREFIX_TRANSITION_COUNT + 1

ARTIFACT_FILENAMES = (
    "branch_state_index.json",
    "determinism_report.json",
    "prefix_execution_report.json",
    "registry_manifest.json",
    "selection_report.json",
    "source_case_inventory.json",
    "summary.md",
)

MEMBER_SCOPES = frozenset({"legacy_external_artifact", "registry_local_artifact"})
GENERATION_STATUSES = frozenset({"legacy_validated", "generated_and_validated"})


class BranchStateRegistryError(ValueError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BranchStateRegistryError(
            f"value is not canonical-JSON serializable: {exc}"
        ) from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_file_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BranchStateRegistryError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise BranchStateRegistryError(f"{name} must be a finite number")
    return converted


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BranchStateRegistryError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BranchStateRegistryError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BranchStateRegistryError(f"cannot load JSON object {path}: {exc}") from exc
    return _mapping(value, path.as_posix())


def generated_scientific_payload(document: Mapping[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(dict(document))
    payload.pop("canonical_payload_hash", None)
    payload.pop("raw_artifact_hash", None)
    payload.pop("generated_date", None)
    return payload


def generated_raw_payload(document: Mapping[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(dict(document))
    payload.pop("raw_artifact_hash", None)
    return payload


def attach_generated_hashes(document: Mapping[str, object]) -> dict[str, object]:
    result = copy.deepcopy(dict(document))
    result.pop("canonical_payload_hash", None)
    result.pop("raw_artifact_hash", None)
    result["canonical_payload_hash"] = canonical_sha256(
        generated_scientific_payload(result)
    )
    result["raw_artifact_hash"] = canonical_sha256(generated_raw_payload(result))
    return result


def _state_values(document: Mapping[str, object]) -> tuple[float, float, float, float]:
    state = _mapping(document.get("state"), "state")
    x = _finite(state.get("position_x"), "state.position_x")
    y = _finite(state.get("position_y"), "state.position_y")
    vx = _finite(state.get("velocity_x"), "state.velocity_x")
    vy = _finite(state.get("velocity_y"), "state.velocity_y")
    for name, expected in (
        ("position_x", x),
        ("position_y", y),
        ("velocity_x", vx),
        ("velocity_y", vy),
    ):
        if _finite(document.get(name), name) != expected:
            raise BranchStateRegistryError(f"{name} does not match state.{name}")
    return x, y, vx, vy


def _validate_derived_state(document: Mapping[str, object]) -> None:
    x, y, vx, vy = _state_values(document)
    simulator = _mapping(document.get("simulator_configuration"), "simulator_configuration")
    constants = _mapping(simulator.get("simulator_constants"), "simulator_constants")
    mu = _finite(constants.get("mu"), "simulator_constants.mu")
    target_radius = _finite(constants.get("target_radius"), "target_radius")
    target_speed = _finite(
        constants.get("target_circular_speed"), "target_circular_speed"
    )
    epsilon = _finite(
        constants.get("speed_ratio_denominator_epsilon"),
        "speed_ratio_denominator_epsilon",
    )
    if mu <= 0.0 or target_radius <= 0.0 or target_speed <= 0.0 or epsilon < 0.0:
        raise BranchStateRegistryError("simulator constants are not physically valid")
    radius = math.hypot(x, y)
    speed = math.hypot(vx, vy)
    if radius <= 0.0:
        raise BranchStateRegistryError("position norm must be positive")
    er_x, er_y = x / radius, y / radius
    et_x, et_y = -er_y, er_x
    radial_velocity = vx * er_x + vy * er_y
    tangential_velocity = vx * et_x + vy * et_y
    signed_radius_error = radius - target_radius
    tangential_error = tangential_velocity - target_speed
    expected = {
        "radius": radius,
        "speed": speed,
        "radial_velocity": radial_velocity,
        "tangential_velocity": tangential_velocity,
        "target_radius": target_radius,
        "target_circular_speed": target_speed,
        "signed_radius_error": signed_radius_error,
        "absolute_radius_gap": abs(signed_radius_error),
        "radius_error_ratio": signed_radius_error / target_radius,
        "radial_velocity_ratio": radial_velocity / (target_speed + epsilon),
        "tangential_velocity_error": tangential_error,
        "tangential_velocity_error_ratio": tangential_error / (target_speed + epsilon),
        "realized_speed_ratio": speed / (target_speed + epsilon),
    }
    for field, expected_value in expected.items():
        supplied = _finite(document.get(field), field)
        if supplied != expected_value:
            raise BranchStateRegistryError(f"{field} does not recompute exactly")
    if _finite(document.get("overspeed_headroom"), "overspeed_headroom") != (
        OVERSPEED_THRESHOLD - expected["realized_speed_ratio"]
    ):
        raise BranchStateRegistryError("overspeed_headroom does not recompute exactly")
    component_expected = {
        "radius_component_pass": abs(expected["radius_error_ratio"])
        <= PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
        "radial_velocity_component_pass": abs(expected["radial_velocity_ratio"])
        <= PHASE34_RECOVERABLE_VR_RATIO_MAX,
        "tangential_velocity_component_pass": abs(
            expected["tangential_velocity_error_ratio"]
        )
        <= PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
    }
    for field, expected_value in component_expected.items():
        if document.get(field) is not expected_value:
            raise BranchStateRegistryError(f"{field} does not match inclusive threshold")
    combined = all(component_expected.values())
    if document.get("phase34_compatible_recoverability_pass") is not combined:
        raise BranchStateRegistryError(
            "phase34_compatible_recoverability_pass does not match components"
        )


def validate_generated_branch_state_document(
    document: Mapping[str, object],
) -> tuple[str, ...]:
    if document.get("schema_version") != REGISTRY_MEMBER_SCHEMA_VERSION:
        raise BranchStateRegistryError("unsupported generated branch-state schema")
    if document.get("state_origin") != "deterministic_nominal_prefix_execution":
        raise BranchStateRegistryError("generated state origin is not deterministic prefix execution")
    if document.get("reconstructed_from_log") is not False:
        raise BranchStateRegistryError("generated state may not be reconstructed from a log")
    if document.get("manually_authored_state") is not False:
        raise BranchStateRegistryError("generated state may not be manually authored")
    if document.get("perturbed_from_existing_state") is not False:
        raise BranchStateRegistryError("generated state may not perturb an existing state")
    member_id = document.get("registry_member_id")
    case_id = document.get("case_id")
    if not isinstance(member_id, str) or not member_id:
        raise BranchStateRegistryError("registry_member_id is required")
    if not isinstance(case_id, str) or not case_id:
        raise BranchStateRegistryError("case_id is required")
    _integer(document.get("seed"), "seed")
    prefix_count = _integer(
        document.get("nominal_prefix_transition_count"),
        "nominal_prefix_transition_count",
        minimum=1,
    )
    if prefix_count != PREFIX_TRANSITION_COUNT:
        raise BranchStateRegistryError("generated state does not use the frozen prefix count")
    if _integer(document.get("actual_transition_count"), "actual_transition_count") != prefix_count:
        raise BranchStateRegistryError("actual transition count does not equal prefix count")
    if _integer(document.get("prefix_action_count"), "prefix_action_count") != prefix_count:
        raise BranchStateRegistryError("prefix action count does not equal prefix count")
    if _integer(document.get("branch_step"), "branch_step", minimum=1) != BRANCH_STEP:
        raise BranchStateRegistryError("branch step does not follow the frozen prefix")
    if document.get("terminal_before_branch") is not False:
        raise BranchStateRegistryError("source trajectory terminated before extraction")
    if document.get("terminal_reason_before_branch") is not None:
        raise BranchStateRegistryError("terminal reason must be null before extraction")
    for field in (
        "initial_state_hash",
        "prefix_action_trace_hash",
        "prefix_state_trace_hash",
        "source_case_hash",
        "source_configuration_hash",
        "simulator_configuration_hash",
        "constants_hash",
        "transition_implementation_hash",
        "nominal_controller_hash",
        "canonical_payload_hash",
        "raw_artifact_hash",
    ):
        if not _is_sha256(document.get(field)):
            raise BranchStateRegistryError(f"{field} must be a SHA-256 digest")
    if document.get("canonical_payload_hash") != canonical_sha256(
        generated_scientific_payload(document)
    ):
        raise BranchStateRegistryError("generated canonical payload hash mismatch")
    if document.get("raw_artifact_hash") != canonical_sha256(
        generated_raw_payload(document)
    ):
        raise BranchStateRegistryError("generated raw payload hash mismatch")
    case_configuration = _mapping(document.get("case_configuration"), "case_configuration")
    if case_configuration.get("case_id") != case_id:
        raise BranchStateRegistryError("case configuration identity mismatch")
    if document.get("case_configuration_hash") != canonical_sha256(case_configuration):
        raise BranchStateRegistryError("case configuration hash mismatch")
    simulator = _mapping(document.get("simulator_configuration"), "simulator_configuration")
    if document.get("simulator_configuration_hash") != canonical_sha256(simulator):
        raise BranchStateRegistryError("simulator configuration hash mismatch")
    constants = _mapping(simulator.get("simulator_constants"), "simulator_constants")
    if document.get("constants_hash") != canonical_sha256(constants):
        raise BranchStateRegistryError("simulator constants hash mismatch")
    if document.get("threshold") != OVERSPEED_THRESHOLD:
        raise BranchStateRegistryError("overspeed threshold changed")
    if document.get("comparator") != OVERSPEED_COMPARATOR:
        raise BranchStateRegistryError("overspeed comparator changed")
    monitor = _mapping(document.get("monitor_decision"), "monitor_decision")
    if monitor.get("monitor_id") != MONITOR_ID:
        raise BranchStateRegistryError("monitor identity mismatch")
    predicted_ratio = _finite(document.get("predicted_speed_ratio"), "predicted_speed_ratio")
    expected_decision = "veto" if predicted_ratio > OVERSPEED_THRESHOLD else "allow"
    if monitor.get("decision") != expected_decision:
        raise BranchStateRegistryError("monitor decision does not match predicted ratio")
    if monitor.get("veto_applied") is not (expected_decision == "veto"):
        raise BranchStateRegistryError("monitor veto status does not match decision")
    _validate_derived_state(document)
    return (
        "generated schema and provenance are valid",
        "Cartesian and derived state recompute exactly",
        "monitor evidence preserves strict overspeed semantics",
        "canonical and raw payload hashes recompute",
    )


@dataclass(frozen=True, slots=True)
class RegistryMember:
    registry_member_id: str
    case_id: str
    seed: int
    artifact_path: str
    artifact_scope: str
    state_origin: str
    source_case_artifact: str
    source_case_hash: str
    source_configuration_hash: str
    simulator_configuration_hash: str
    constants_hash: str
    transition_implementation_hash: str
    nominal_controller_hash: str
    source_commit: str
    nominal_prefix_transition_count: int
    branch_step: int
    canonical_branch_state_hash: str
    raw_artifact_hash: str
    legacy_member: bool
    generation_status: str
    determinism_status: str
    executable_status: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RegistryMember":
        try:
            member = cls(
                registry_member_id=str(value["registry_member_id"]),
                case_id=str(value["case_id"]),
                seed=_integer(value["seed"], "member.seed"),
                artifact_path=str(value["artifact_path"]),
                artifact_scope=str(value["artifact_scope"]),
                state_origin=str(value["state_origin"]),
                source_case_artifact=str(value["source_case_artifact"]),
                source_case_hash=str(value["source_case_hash"]),
                source_configuration_hash=str(value["source_configuration_hash"]),
                simulator_configuration_hash=str(value["simulator_configuration_hash"]),
                constants_hash=str(value["constants_hash"]),
                transition_implementation_hash=str(value["transition_implementation_hash"]),
                nominal_controller_hash=str(value["nominal_controller_hash"]),
                source_commit=str(value["source_commit"]),
                nominal_prefix_transition_count=_integer(
                    value["nominal_prefix_transition_count"],
                    "member.nominal_prefix_transition_count",
                    minimum=1,
                ),
                branch_step=_integer(value["branch_step"], "member.branch_step", minimum=1),
                canonical_branch_state_hash=str(value["canonical_branch_state_hash"]),
                raw_artifact_hash=str(value["raw_artifact_hash"]),
                legacy_member=cast(bool, value["legacy_member"]),
                generation_status=str(value["generation_status"]),
                determinism_status=str(value["determinism_status"]),
                executable_status=str(value["executable_status"]),
            )
        except KeyError as exc:
            raise BranchStateRegistryError(f"registry member missing field {exc.args[0]}") from exc
        if not member.registry_member_id or not member.case_id:
            raise BranchStateRegistryError("registry member identity is required")
        if member.artifact_scope not in MEMBER_SCOPES:
            raise BranchStateRegistryError("unsupported registry artifact scope")
        if member.generation_status not in GENERATION_STATUSES:
            raise BranchStateRegistryError("unsupported generation status")
        if member.state_origin != "deterministic_nominal_prefix_execution":
            raise BranchStateRegistryError("registry member state origin is unsupported")
        if member.nominal_prefix_transition_count != PREFIX_TRANSITION_COUNT:
            raise BranchStateRegistryError("registry member prefix count is not frozen")
        if member.branch_step != BRANCH_STEP:
            raise BranchStateRegistryError("registry member branch step is not frozen")
        if member.determinism_status != "passed" or member.executable_status != "validated":
            raise BranchStateRegistryError("published member is not deterministic and executable")
        if not isinstance(value.get("legacy_member"), bool):
            raise BranchStateRegistryError("legacy_member must be boolean")
        if member.legacy_member:
            if (
                member.registry_member_id != LEGACY_MEMBER_ID
                or member.case_id != LEGACY_CASE_ID
                or member.artifact_scope != "legacy_external_artifact"
                or member.generation_status != "legacy_validated"
            ):
                raise BranchStateRegistryError("legacy registry member classification mismatch")
        elif (
            member.artifact_scope != "registry_local_artifact"
            or member.generation_status != "generated_and_validated"
        ):
            raise BranchStateRegistryError("generated registry member classification mismatch")
        for digest in (
            member.source_case_hash,
            member.source_configuration_hash,
            member.simulator_configuration_hash,
            member.constants_hash,
            member.transition_implementation_hash,
            member.nominal_controller_hash,
            member.canonical_branch_state_hash,
            member.raw_artifact_hash,
        ):
            if not _is_sha256(digest):
                raise BranchStateRegistryError("registry member contains an invalid digest")
        return member

    def as_document(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class BranchStateRegistry:
    registry_id: str
    schema_version: str
    members: tuple[RegistryMember, ...]
    canonical_manifest_hash: str
    manifest_path: str


@dataclass(frozen=True, slots=True)
class RegisteredBranchState:
    member: RegistryMember
    _document_json: str

    @property
    def case_id(self) -> str:
        return self.member.case_id

    @property
    def registry_member_id(self) -> str:
        return self.member.registry_member_id

    def as_document(self) -> dict[str, object]:
        value = json.loads(self._document_json)
        return _mapping(value, "registered branch state")


def manifest_scientific_payload(document: Mapping[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(dict(document))
    payload.pop("canonical_manifest_hash", None)
    payload.pop("result_commit_when_available", None)
    return payload


def validate_registry_manifest_document(
    document: Mapping[str, object],
) -> tuple[RegistryMember, ...]:
    if document.get("registry_id") != REGISTRY_ID:
        raise BranchStateRegistryError("registry identity mismatch")
    if document.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise BranchStateRegistryError("registry schema version mismatch")
    supplied_hash = document.get("canonical_manifest_hash")
    if not _is_sha256(supplied_hash) or supplied_hash != canonical_sha256(
        manifest_scientific_payload(document)
    ):
        raise BranchStateRegistryError("registry manifest canonical hash mismatch")
    raw_members = document.get("member_index")
    if not isinstance(raw_members, list):
        raise BranchStateRegistryError("member_index must be a list")
    members = tuple(RegistryMember.from_mapping(_mapping(item, "member")) for item in raw_members)
    ids = tuple(member.registry_member_id for member in members)
    cases = tuple(member.case_id for member in members)
    if ids != tuple(sorted(ids)):
        raise BranchStateRegistryError("registry member ordering is not deterministic")
    if len(ids) != len(set(ids)):
        raise BranchStateRegistryError("duplicate registry member ID")
    if len(cases) != len(set(cases)):
        raise BranchStateRegistryError("duplicate registry case ID")
    if len(members) < 4:
        raise BranchStateRegistryError("registry requires at least four members")
    legacy = tuple(member for member in members if member.legacy_member)
    if len(legacy) != 1 or legacy[0].registry_member_id != LEGACY_MEMBER_ID:
        raise BranchStateRegistryError("registry requires exactly one legacy canonical member")
    if sum(not member.legacy_member for member in members) < 3:
        raise BranchStateRegistryError("registry requires at least three generated members")
    if document.get("Stage_1B_prerequisite_status") != "satisfied":
        raise BranchStateRegistryError("Stage 1B prerequisite is not satisfied")
    return members


def _resolve_member_path(
    repository_root: Path,
    member: RegistryMember,
) -> Path:
    root = repository_root.resolve()
    relative = PurePosixPath(member.artifact_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise BranchStateRegistryError("registry artifact path traversal is forbidden")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BranchStateRegistryError("registry artifact escapes the repository") from exc
    if member.artifact_scope == "legacy_external_artifact":
        expected = (root / LEGACY_ARTIFACT_PATH).resolve()
        if path != expected:
            raise BranchStateRegistryError("legacy member path substitution detected")
    else:
        expected_parent = (root / OUTPUT_PATH / "branch_states").resolve()
        try:
            path.relative_to(expected_parent)
        except ValueError as exc:
            raise BranchStateRegistryError("registry-local member path is outside branch_states") from exc
    if path.is_symlink() or not path.is_file():
        raise BranchStateRegistryError("registered branch-state artifact is missing or symbolic")
    return path


def load_branch_state_registry(
    repository_root: Path,
    manifest_path: Path | None = None,
) -> BranchStateRegistry:
    root = repository_root.resolve()
    path = (manifest_path or (root / MANIFEST_PATH)).resolve()
    if path != (root / MANIFEST_PATH).resolve():
        raise BranchStateRegistryError("arbitrary registry manifest paths are forbidden")
    document = _load_json_object(path)
    members = validate_registry_manifest_document(document)
    return BranchStateRegistry(
        registry_id=REGISTRY_ID,
        schema_version=REGISTRY_SCHEMA_VERSION,
        members=members,
        canonical_manifest_hash=str(document["canonical_manifest_hash"]),
        manifest_path=path.as_posix(),
    )


def list_registered_branch_states(
    repository_root: Path,
) -> tuple[RegistryMember, ...]:
    return load_branch_state_registry(repository_root).members


def load_registered_branch_state(
    repository_root: Path,
    registry_member_id: str,
) -> RegisteredBranchState:
    if not isinstance(registry_member_id, str) or not registry_member_id:
        raise BranchStateRegistryError("registry member ID is required")
    registry = load_branch_state_registry(repository_root)
    matches = tuple(
        member for member in registry.members if member.registry_member_id == registry_member_id
    )
    if len(matches) != 1:
        raise BranchStateRegistryError(f"unknown registry member: {registry_member_id!r}")
    member = matches[0]
    path = _resolve_member_path(repository_root, member)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != member.raw_artifact_hash:
        raise BranchStateRegistryError("registered artifact raw hash mismatch")
    document = _load_json_object(path)
    if document.get("case_id") != member.case_id:
        raise BranchStateRegistryError("registered artifact case identity mismatch")
    common_expected = {
        "seed": member.seed,
        "source_commit": member.source_commit,
        "simulator_configuration_hash": member.simulator_configuration_hash,
    }
    for field, expected in common_expected.items():
        if document.get(field) != expected:
            raise BranchStateRegistryError(
                f"registered artifact {field} does not match registry provenance"
            )
    if member.legacy_member:
        from runtime_assurance.recovery_branch_executor import (
            validate_branch_state_integrity,
        )

        validate_branch_state_integrity(document)
        if document.get("canonical_branch_state_hash") != member.canonical_branch_state_hash:
            raise BranchStateRegistryError("legacy canonical branch-state hash mismatch")
        if document.get("case_configuration_hash") != member.source_configuration_hash:
            raise BranchStateRegistryError("legacy source configuration hash mismatch")
        if document.get("simulator_constants_hash") != member.constants_hash:
            raise BranchStateRegistryError("legacy constants hash mismatch")
        ordering = _mapping(document.get("branch_ordering"), "branch_ordering")
        if (
            document.get("branch_step") != member.branch_step
            or ordering.get("realized_prefix_transition_count")
            != member.nominal_prefix_transition_count
        ):
            raise BranchStateRegistryError("legacy prefix provenance mismatch")
    else:
        validate_generated_branch_state_document(document)
        if document.get("canonical_payload_hash") != member.canonical_branch_state_hash:
            raise BranchStateRegistryError("generated canonical branch-state hash mismatch")
        generated_expected = {
            "registry_member_id": member.registry_member_id,
            "state_origin": member.state_origin,
            "source_case_artifact": member.source_case_artifact,
            "source_case_hash": member.source_case_hash,
            "source_configuration_hash": member.source_configuration_hash,
            "constants_hash": member.constants_hash,
            "transition_implementation_hash": member.transition_implementation_hash,
            "nominal_controller_hash": member.nominal_controller_hash,
            "nominal_prefix_transition_count": member.nominal_prefix_transition_count,
            "branch_step": member.branch_step,
        }
        for field, expected in generated_expected.items():
            if document.get(field) != expected:
                raise BranchStateRegistryError(
                    f"generated artifact {field} does not match registry provenance"
                )
    return RegisteredBranchState(
        member=member,
        _document_json=canonical_json_bytes(document).decode("utf-8"),
    )


def registry_aggregate_hash(members: Sequence[RegistryMember]) -> str:
    return canonical_sha256(
        [
            {
                "registry_member_id": member.registry_member_id,
                "case_id": member.case_id,
                "canonical_branch_state_hash": member.canonical_branch_state_hash,
                "raw_artifact_hash": member.raw_artifact_hash,
            }
            for member in sorted(members, key=lambda item: item.registry_member_id)
        ]
    )


__all__ = [
    "ARTIFACT_FILENAMES",
    "BRANCH_STEP",
    "COMPLETED_DATE",
    "CONFIG_PATH",
    "CONFIG_SCHEMA_VERSION",
    "LEGACY_ARTIFACT_PATH",
    "LEGACY_CASE_ID",
    "LEGACY_MEMBER_ID",
    "MANIFEST_PATH",
    "OUTPUT_PATH",
    "PREFIX_TRANSITION_COUNT",
    "REGISTRY_ID",
    "REGISTRY_MEMBER_SCHEMA_VERSION",
    "REGISTRY_SCHEMA_VERSION",
    "BranchStateRegistry",
    "BranchStateRegistryError",
    "RegisteredBranchState",
    "RegistryMember",
    "attach_generated_hashes",
    "canonical_json_bytes",
    "canonical_json_file_bytes",
    "canonical_sha256",
    "file_sha256",
    "generated_raw_payload",
    "generated_scientific_payload",
    "list_registered_branch_states",
    "load_branch_state_registry",
    "load_registered_branch_state",
    "manifest_scientific_payload",
    "registry_aggregate_hash",
    "validate_generated_branch_state_document",
    "validate_registry_manifest_document",
]
