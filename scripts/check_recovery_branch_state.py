from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "recovery_branch_state_v0"
CANONICALIZATION_ID = "recovery_branch_state_canonical_json_v0"
SOURCE_COMMIT = "8d5592079f0b523a8aec04317bd792673929a35b"
SOURCE_CASE_ID = (
    "phase35_radial_energy_push_overspeed_stress_v0"
    "__r0_0p98__angle_150__thrust_8000"
)
SOURCE_SUBSET_ID = "phase35_radial_energy_push_overspeed_stress_v0"
SOURCE_R0_OVER_TARGET = 0.98
SOURCE_ANGLE_DEG = 150.0
SOURCE_THRUST_SCALE = 8000.0
SOURCE_SEED = 0
HAZARD_THRESHOLD = 1.90
HAZARD_COMPARATOR = ">"
MONITOR_ID = "one_step_overspeed_veto_v0"
ARTIFACT_RELATIVE_PATH = Path(
    "analysis/recovery_action_branching_nonformal_v0/branch_state.json"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEYS = frozenset(
    {
        "avoided_failure",
        "branch_id",
        "branch_outcome",
        "branch_winner",
        "comparison_metrics",
        "comparison_score",
        "experiment_results",
        "recovery_action",
        "recovery_action_executed",
        "recovery_success",
        "result_rows",
        "results_csv",
        "reward",
        "selected_branch",
        "simulator_success",
        "success",
        "success_label",
        "task_recovery",
        "terminal_label",
        "winner",
        "winning_branch",
    }
)


class BranchStateValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def find_repository_root(start: Path | None = None) -> Path:
    candidate = (start or Path(__file__)).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists() and (directory / "scripts").is_dir():
            return directory
    raise FileNotFoundError("could not locate repository root")


def canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise BranchStateValidationError(
            [f"value cannot be serialized as canonical JSON: {exc}"]
        ) from exc
    return text.encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def attach_canonical_branch_state_hash(
    document_without_hash: Mapping[str, object],
) -> dict[str, Any]:
    document = copy.deepcopy(dict(document_without_hash))
    document.pop("canonical_branch_state_hash", None)
    document["canonical_branch_state_hash"] = canonical_sha256(document)
    return document


def write_canonical_branch_state(
    path: Path,
    document: Mapping[str, object],
    *,
    refuse_overwrite: bool = True,
) -> None:
    path = path.resolve()
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"refusing to overwrite branch-state artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(document))
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _mapping(value: object, path: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return {}
    return value


def _finite_number(value: object, path: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{path} must be a finite number")
        return None
    converted = float(value)
    if not math.isfinite(converted):
        errors.append(f"{path} must be a finite number")
        return None
    return converted


def _finite_vector(
    value: object,
    length: int,
    path: str,
    errors: list[str],
) -> list[float] | None:
    if not isinstance(value, list) or len(value) != length:
        errors.append(f"{path} must be a {length}-component JSON array")
        return None
    converted: list[float] = []
    for index, component in enumerate(value):
        number = _finite_number(component, f"{path}[{index}]", errors)
        if number is None:
            return None
        converted.append(number)
    return converted


def _forbidden_key_locations(value: object, path: str = "$") -> list[str]:
    locations: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in _FORBIDDEN_KEYS or key.startswith("measured_result"):
                locations.append(child_path)
            locations.extend(_forbidden_key_locations(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            locations.extend(_forbidden_key_locations(child, f"{path}[{index}]"))
    return locations


def _check_hash(
    value: object,
    expected_payload: object,
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        errors.append(f"{path} must be a lowercase SHA-256 digest")
        return
    expected = canonical_sha256(expected_payload)
    if value != expected:
        errors.append(f"{path} does not match its canonical payload")


def validate_branch_state_data(data: Mapping[str, object]) -> list[str]:
    document = dict(data)
    errors: list[str] = []
    passes: list[str] = []

    forbidden = _forbidden_key_locations(document)
    if forbidden:
        errors.append(f"forbidden experiment or recovery fields found at: {forbidden}")
    else:
        passes.append("no recovery outcome, winner, result, or comparison fields are present")

    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    else:
        passes.append("schema version is frozen")

    if document.get("source_commit") != SOURCE_COMMIT:
        errors.append(f"source_commit must be the frozen commit {SOURCE_COMMIT}")
    elif document.get("implementation_commit") != SOURCE_COMMIT:
        errors.append("implementation_commit must match the frozen source trajectory commit")
    else:
        passes.append("source and trajectory implementation commits are frozen")

    timestamp = document.get("extraction_timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        errors.append("extraction_timestamp must be a nonempty string")
    elif document.get("extraction_timestamp_policy") != (
        "frozen_milestone_timestamp_for_reproducible_hashing"
    ):
        errors.append("extraction_timestamp_policy is missing or changed")
    else:
        passes.append("extraction timestamp has a deterministic provenance policy")

    if document.get("case_id") != SOURCE_CASE_ID:
        errors.append(f"case_id must be {SOURCE_CASE_ID}")
    if document.get("subset_id") != SOURCE_SUBSET_ID:
        errors.append(f"subset_id must be {SOURCE_SUBSET_ID}")
    if document.get("seed") != SOURCE_SEED:
        errors.append(f"seed must be {SOURCE_SEED}")

    case_configuration = _mapping(
        document.get("case_configuration"), "case_configuration", errors
    )
    expected_case_configuration = {
        "case_id": SOURCE_CASE_ID,
        "controller_id": "phase35_crossing_basin_expansion",
        "initial_velocity_angle_deg": SOURCE_ANGLE_DEG,
        "post_cross_mode": "radius_priority",
        "r0_over_target": SOURCE_R0_OVER_TARGET,
        "seed": SOURCE_SEED,
        "subset_id": SOURCE_SUBSET_ID,
        "thrust_scale": SOURCE_THRUST_SCALE,
        "upstream_variant": "radial_energy_push",
    }
    if case_configuration != expected_case_configuration:
        errors.append("case_configuration does not match the frozen source case")
    else:
        passes.append("source case and nominal controller context are exact")
    _check_hash(
        document.get("case_configuration_hash"),
        case_configuration,
        "case_configuration_hash",
        errors,
    )

    simulator_configuration = _mapping(
        document.get("simulator_configuration"),
        "simulator_configuration",
        errors,
    )
    simulator_constants = _mapping(
        simulator_configuration.get("simulator_constants"),
        "simulator_configuration.simulator_constants",
        errors,
    )
    _check_hash(
        document.get("simulator_constants_hash"),
        simulator_constants,
        "simulator_constants_hash",
        errors,
    )
    _check_hash(
        document.get("simulator_configuration_hash"),
        simulator_configuration,
        "simulator_configuration_hash",
        errors,
    )
    if simulator_configuration.get("thrust_scale") != SOURCE_THRUST_SCALE:
        errors.append("simulator_configuration.thrust_scale must match the source case")
    else:
        passes.append("simulator and case configuration hashes are internally consistent")

    state = _mapping(document.get("state"), "state", errors)
    state_names = (
        "position_x",
        "position_y",
        "velocity_x",
        "velocity_y",
    )
    state_values: list[float] = []
    for name in state_names:
        value = _finite_number(state.get(name), f"state.{name}", errors)
        if value is not None:
            state_values.append(value)
    current_phase = state.get("current_phase")
    if not isinstance(current_phase, str) or not current_phase:
        errors.append("state.current_phase must be a nonempty string")
    state_vector = _finite_vector(document.get("state_vector"), 4, "state_vector", errors)
    position = _finite_vector(document.get("position"), 2, "position", errors)
    velocity = _finite_vector(document.get("velocity"), 2, "velocity", errors)
    if len(state_values) == 4:
        if state_vector != state_values:
            errors.append("state_vector must match the named state components")
        if position != state_values[:2]:
            errors.append("position must match position_x and position_y")
        if velocity != state_values[2:]:
            errors.append("velocity must match velocity_x and velocity_y")
    if document.get("phase") != current_phase:
        errors.append("phase must match state.current_phase")
    if not isinstance(document.get("active_stage"), str) or not document.get(
        "active_stage"
    ):
        errors.append("active_stage must be a nonempty string")
    if not any(error.startswith(("state.", "state_vector", "position", "velocity", "phase", "active_stage")) for error in errors):
        passes.append("current state, phase, and active stage are complete and finite")

    step = document.get("step")
    branch_step = document.get("branch_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        errors.append("step must be a positive integer")
    if branch_step != step:
        errors.append("branch_step must equal step")

    nominal_action = _finite_vector(
        document.get("nominal_action"), 2, "nominal_action", errors
    )
    nominal_proposed_action = _finite_vector(
        document.get("nominal_proposed_action"),
        2,
        "nominal_proposed_action",
        errors,
    )
    if nominal_action != nominal_proposed_action:
        errors.append("nominal action representations must match")

    predicted_state = _mapping(
        document.get("predicted_next_state"), "predicted_next_state", errors
    )
    predicted_values: list[float] = []
    for name in state_names:
        value = _finite_number(
            predicted_state.get(name), f"predicted_next_state.{name}", errors
        )
        if value is not None:
            predicted_values.append(value)
    predicted_vector = _finite_vector(
        document.get("predicted_nominal_next_state"),
        4,
        "predicted_nominal_next_state",
        errors,
    )
    if len(predicted_values) == 4 and predicted_vector != predicted_values:
        errors.append(
            "predicted_nominal_next_state must match the named predicted components"
        )

    predicted_ratio = _finite_number(
        document.get("predicted_speed_ratio"), "predicted_speed_ratio", errors
    )
    duplicate_ratio = _finite_number(
        document.get("predicted_nominal_speed_ratio"),
        "predicted_nominal_speed_ratio",
        errors,
    )
    if predicted_ratio != duplicate_ratio:
        errors.append("predicted speed-ratio representations must match")
    if predicted_ratio is not None and not predicted_ratio > HAZARD_THRESHOLD:
        errors.append("branch prediction must be strictly greater than 1.90")

    if document.get("threshold") != HAZARD_THRESHOLD or document.get(
        "hazard_threshold"
    ) != HAZARD_THRESHOLD:
        errors.append("threshold fields must equal 1.90")
    if document.get("comparator") != HAZARD_COMPARATOR or document.get(
        "hazard_comparator"
    ) != HAZARD_COMPARATOR:
        errors.append("comparator fields must equal strict >")
    monitor_decision = _mapping(
        document.get("monitor_decision"), "monitor_decision", errors
    )
    if monitor_decision != {
        "decision": "veto",
        "monitor_id": MONITOR_ID,
        "reason": "predicted_nominal_overspeed",
        "veto_applied": True,
    }:
        errors.append("monitor_decision must record the frozen valid veto decision")
    else:
        passes.append("strict overspeed branch trigger and veto decision are exact")

    ordering = _mapping(document.get("branch_ordering"), "branch_ordering", errors)
    required_ordering = {
        "capture_boundary": (
            "after_valid_monitor_evaluation_before_nominal_or_fallback_execution"
        ),
        "before_final_veto_fallback_execution": True,
        "before_nominal_action_execution": True,
        "final_veto_fallback_executed": False,
        "monitor_evaluation_completed": True,
        "nominal_action_executed": False,
        "prior_monitor_decisions": "all_allow",
        "prior_veto_count": 0,
    }
    for key, expected in required_ordering.items():
        if ordering.get(key) != expected:
            errors.append(f"branch_ordering.{key} must equal {expected!r}")
    if isinstance(step, int) and not isinstance(step, bool):
        expected_prefix_count = step - 1
        if ordering.get("realized_prefix_transition_count") != expected_prefix_count:
            errors.append(
                "branch_ordering.realized_prefix_transition_count must equal step - 1"
            )
        if ordering.get("prior_valid_monitor_evaluation_count") != expected_prefix_count:
            errors.append(
                "branch_ordering.prior_valid_monitor_evaluation_count must equal step - 1"
            )
    if not any(error.startswith("branch_ordering") for error in errors):
        passes.append("branch ordering proves capture occurred before either action")

    canonicalization = _mapping(
        document.get("canonicalization"), "canonicalization", errors
    )
    expected_canonicalization = {
        "allow_nan": False,
        "canonicalization_id": CANONICALIZATION_ID,
        "encoding": "utf-8",
        "hash_algorithm": "sha256",
        "hash_field_excluded_from_input": "canonical_branch_state_hash",
        "json_separators": [",", ":"],
        "json_sort_keys": True,
    }
    if canonicalization != expected_canonicalization:
        errors.append("canonicalization contract is missing or changed")
    else:
        passes.append("canonical UTF-8 JSON contract is explicit")

    supplied_hash = document.get("canonical_branch_state_hash")
    if not isinstance(supplied_hash, str) or not _SHA256_RE.fullmatch(supplied_hash):
        errors.append("canonical_branch_state_hash must be a lowercase SHA-256 digest")
    else:
        hash_input = copy.deepcopy(document)
        hash_input.pop("canonical_branch_state_hash", None)
        expected_hash = canonical_sha256(hash_input)
        if supplied_hash != expected_hash:
            errors.append(
                "canonical_branch_state_hash does not match the complete canonical document"
            )
        else:
            passes.append("canonical branch-state hash matches the complete document")

    if errors:
        raise BranchStateValidationError(errors)
    return passes


def load_branch_state(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"branch-state artifact not found: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BranchStateValidationError(["artifact is not valid UTF-8"]) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BranchStateValidationError([f"artifact is invalid JSON: {exc}"]) from exc
    if not isinstance(data, dict):
        raise BranchStateValidationError(["branch-state artifact must be a JSON object"])
    return data, raw


def validate_branch_state(path: Path) -> list[str]:
    data, raw = load_branch_state(path)
    passes = validate_branch_state_data(data)
    expected_bytes = canonical_json_bytes(data)
    if raw != expected_bytes:
        raise BranchStateValidationError(
            [
                "artifact bytes are not canonical UTF-8 JSON with sorted keys, "
                "stable separators, and no trailing bytes"
            ]
        )
    passes.append("artifact bytes use the frozen deterministic serialization")
    return passes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the extracted Recovery Action Branching v0 state."
    )
    parser.add_argument(
        "--branch-state",
        type=Path,
        default=None,
        help=(
            "Artifact path; defaults to analysis/recovery_action_branching_"
            "nonformal_v0/branch_state.json under the repository root."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        repository_root = find_repository_root()
        artifact_path = (
            args.branch_state.resolve()
            if args.branch_state
            else repository_root / ARTIFACT_RELATIVE_PATH
        )
        passes = validate_branch_state(artifact_path)
    except (FileNotFoundError, BranchStateValidationError) as exc:
        errors = exc.errors if isinstance(exc, BranchStateValidationError) else [str(exc)]
        for error in errors:
            print(f"FAIL {error}")
        print(f"Recovery branch-state validation FAILED with {len(errors)} issue(s).")
        return 1

    for message in passes:
        print(f"PASS {message}")
    print(f"Recovery branch-state validation PASSED with {len(passes)} checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
