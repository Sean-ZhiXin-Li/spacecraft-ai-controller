from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from runtime_assurance.final_veto_monitor import (
    OVERSPEED_COMPARATOR,
    OVERSPEED_THRESHOLD,
    FinalVetoDecision,
    OneStepPrediction,
    evaluate_overspeed_veto,
)
from runtime_assurance.recovery_branch_executor import (
    RecoveryBranchExecutionResult,
    execute_registered_recovery_branch,
    generate_tangential_correction_action,
    generate_velocity_opposed_action,
    generate_zero_action,
)
from runtime_assurance.recovery_branch_state_extractor import (
    protected_evidence_hashes as historical_protected_evidence_hashes,
)
from runtime_assurance.recovery_branch_state_registry import (
    RegisteredBranchState,
    file_sha256,
    load_registered_branch_state,
)
from runtime_assurance.recovery_stop_conditions import (
    INVALID_RECOVERY_EVALUATION,
    RecoveryStopConditionReport,
    evaluate_recovery_stop_conditions,
)
from runtime_assurance.stage2a_hazard_arrest_authority import (
    PROVISIONAL_ACTION_SOURCE,
    AuthorityBlockedReason,
    AuthorityEvidenceStatus,
    HazardArrestAuthorityInput,
    HazardArrestAuthoritySession,
    HazardArrestEvidence,
    HazardEvidenceKind,
    consume_hazard_arrest_proposal,
    request_hazard_arrest_proposal,
)
from runtime_assurance.staged_recovery_instrumentation import (
    CURRENT_GRAVITY_MODEL_ID,
    CartesianState2D as InstrumentationState2D,
    OrbitalConfiguration,
    derive_orbital_state,
    derive_phase34_recoverability,
)
from runtime_assurance.staged_recovery_logger_adapter import runtime_state_hash
from runtime_assurance.staged_recovery_shadow_calibration import (
    TRACE_SET_OUTPUT_PATH,
    atomic_publish_new_directory,
    load_trace_set,
)
from runtime_assurance.staged_recovery_shadow_runtime import (
    EXPECTED_REGISTRY_AGGREGATE_HASH,
    EXPECTED_REGISTRY_MANIFEST_HASH,
    build_registered_runtime_identity,
)
from simulator.phase34_35_transition import (
    CartesianState2D,
    NormalizedAction2D,
    Phase3435DynamicsContext,
    step_phase34_35_transition,
)


STAGE2A_EXPERIMENT_ID = "stage2a_one_intervention_hazard_arrest_v0"
STAGE2A_SCHEMA_VERSION = "stage2a_one_intervention_hazard_arrest_v0"
SOURCE_TRACE_STATE_HASH_SCHEMA = "stage1b_cartesian_xy_vx_vy_v0"
COMPLETED_DATE = "2026-08-21"
QUALIFICATION_OUTPUT_PATH = Path("analysis/stage2a_hazard_arrest_qualification_v0")
EXPERIMENT_OUTPUT_PATH = Path("analysis/stage2a_hazard_arrest_experiment_v0")
MAXIMUM_PHYSICAL_TRANSITIONS_PER_RUN = 32
NORMAL_BRANCH_IDS = (
    "zero_action_reference_v0",
    "tangential_error_correction_v0",
)
HAZARD_BRANCH_ID = "velocity_opposed_thrust_v0"
RELEASE_AUTHORIZED = "authorized_to_predeclared_normal_branch"
RELEASE_NOT_AUTHORIZED = "not_authorized"
QUALIFICATION_ARTIFACTS = (
    "eligible_boundaries.json",
    "qualification_manifest.json",
    "selected_experiment.json",
    "summary.md",
)
EXPERIMENT_ARTIFACTS = (
    "active_summary.json",
    "authority_report.json",
    "baseline_summary.json",
    "boundary_equivalence_report.json",
    "experiment_manifest.json",
    "final_veto_report.json",
    "intervention_effect.json",
    "protected_evidence_report.json",
    "release_report.json",
    "selected_case.json",
    "summary.md",
    "traces/active.jsonl",
    "traces/baseline.jsonl",
)
NEWER_PROTECTED_PATHS = {
    "recovery_branch_state_registry_v0": (
        "analysis/recovery_branch_state_registry_v0",
    ),
    "staged_recovery_shadow_smoke_v0": (
        "analysis/staged_recovery_shadow_smoke_v0",
    ),
    "staged_recovery_shadow_calibration_trace_set_v0": (
        "analysis/staged_recovery_shadow_calibration_trace_set_v0",
    ),
    "staged_recovery_shadow_calibration_v0": (
        "analysis/staged_recovery_shadow_calibration_v0",
    ),
    "stage2a_active_hazard_arrest_preflight_v0": (
        "docs/architecture/stage2a_active_hazard_arrest_preflight_v0.md",
        "analysis/stage2a_active_hazard_arrest_preflight_v0",
    ),
    "stage2a_hazard_arrest_authority_adapter_v0": (
        "runtime_assurance/stage2a_hazard_arrest_authority.py",
        "Tests/test_stage2a_hazard_arrest_authority.py",
        "docs/architecture/stage2a_hazard_arrest_authority_adapter_v0.md",
    ),
}


class Stage2AHazardArrestRunnerError(RuntimeError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_trace_state_hash(state: CartesianState2D) -> str:
    """Recompute the exact Cartesian hash schema frozen in Stage 1B traces."""
    return canonical_sha256(
        {"x": state.x, "y": state.y, "vx": state.vx, "vy": state.vy}
    )


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage2AHazardArrestRunnerError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise Stage2AHazardArrestRunnerError(f"{name} must be finite")
    return result


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Stage2AHazardArrestRunnerError(f"{name} must be a mapping")
    return value


def _pairs(value: object) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, list):
        raise Stage2AHazardArrestRunnerError("instrumentation fields must be pairs")
    result: dict[str, Mapping[str, object]] = {}
    for pair in value:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not isinstance(pair[1], Mapping)
        ):
            raise Stage2AHazardArrestRunnerError(
                "instrumentation field pair is malformed"
            )
        result[pair[0]] = pair[1]
    return result


def _field_number(fields: Mapping[str, Mapping[str, object]], field_id: str) -> float:
    field = _mapping(fields.get(field_id), field_id)
    if field.get("valid") is not True:
        raise Stage2AHazardArrestRunnerError(f"{field_id} is unavailable")
    return _finite(field.get("value"), field_id)


def _state_from_source_event(event: Mapping[str, object]) -> CartesianState2D:
    observation = _mapping(event.get("pre_observation"), "pre_observation")
    fields = _pairs(observation.get("fields"))
    state = CartesianState2D(
        x=_field_number(fields, "position_x"),
        y=_field_number(fields, "position_y"),
        vx=_field_number(fields, "velocity_x"),
        vy=_field_number(fields, "velocity_y"),
    )
    if source_trace_state_hash(state) != event.get("pre_state_hash"):
        raise Stage2AHazardArrestRunnerError("source boundary state hash mismatch")
    return state


def _current_speed_ratio(event: Mapping[str, object]) -> float:
    observation = _mapping(event.get("pre_observation"), "pre_observation")
    return _field_number(_pairs(observation.get("fields")), "realized_speed_ratio")


def _guard_statuses(row: Mapping[str, object]) -> dict[str, str]:
    values = row.get("guard_evaluations")
    if not isinstance(values, list):
        raise Stage2AHazardArrestRunnerError("guard evaluations are missing")
    return {
        str(item["guard_atom_id"]): str(item["status"])
        for item in values
        if isinstance(item, Mapping)
    }


def _registered_dynamics(
    registered: RegisteredBranchState,
) -> tuple[Phase3435DynamicsContext, float, float]:
    document = registered.as_document()
    configuration = _mapping(
        document.get("simulator_configuration"), "simulator_configuration"
    )
    constants = _mapping(
        configuration.get("simulator_constants"), "simulator_constants"
    )
    dynamics = Phase3435DynamicsContext(
        mu=_finite(constants.get("mu"), "mu"),
        dt=_finite(constants.get("dt"), "dt"),
        mass=_finite(constants.get("mass"), "mass"),
        thrust_scale=_finite(configuration.get("thrust_scale"), "thrust_scale"),
    )
    return (
        dynamics,
        _finite(constants.get("target_circular_speed"), "target_circular_speed"),
        _finite(
            constants.get("speed_ratio_denominator_epsilon"),
            "speed_ratio_denominator_epsilon",
        ),
    )


def _branch_action(
    branch_id: str,
    state: CartesianState2D,
    target_circular_speed: float,
) -> tuple[float, float]:
    if branch_id == "zero_action_reference_v0":
        return generate_zero_action()
    if branch_id == "tangential_error_correction_v0":
        return generate_tangential_correction_action(state, target_circular_speed)
    if branch_id == HAZARD_BRANCH_ID:
        return generate_velocity_opposed_action(state)
    raise Stage2AHazardArrestRunnerError(f"unsupported Stage 2A branch: {branch_id}")


@dataclass(frozen=True, slots=True)
class OneStepActionEvaluation:
    branch_id: str
    action: tuple[float, float]
    action_hash: str
    predicted_state: CartesianState2D
    predicted_state_hash: str
    predicted_speed_ratio: float
    predicted_headroom: float
    final_veto_decision: FinalVetoDecision
    fallback_prediction_count: int
    physical_transition_count: int = 0

    def as_document(self) -> dict[str, object]:
        return {
            "branch_id": self.branch_id,
            "action": list(self.action),
            "action_hash": self.action_hash,
            "predicted_state": state_document(self.predicted_state),
            "predicted_state_hash": self.predicted_state_hash,
            "predicted_speed_ratio": self.predicted_speed_ratio,
            "predicted_headroom": self.predicted_headroom,
            "final_veto_decision": self.final_veto_decision.decision,
            "final_veto_reason": self.final_veto_decision.reason,
            "fallback_prediction_count": self.fallback_prediction_count,
            "fallback_execution_count": 0,
            "physical_transition_count": self.physical_transition_count,
        }


def state_document(state: CartesianState2D) -> dict[str, float]:
    return {
        "position_x": state.x,
        "position_y": state.y,
        "velocity_x": state.vx,
        "velocity_y": state.vy,
    }


def evaluate_branch_without_execution(
    registered: RegisteredBranchState,
    state: CartesianState2D,
    branch_id: str,
) -> OneStepActionEvaluation:
    dynamics, target_speed, epsilon = _registered_dynamics(registered)
    action = _branch_action(branch_id, state, target_speed)
    predictions: list[OneStepPrediction[CartesianState2D]] = []

    def predictor(
        current: CartesianState2D,
        proposed_action: tuple[float, float],
    ) -> OneStepPrediction[CartesianState2D]:
        transition = step_phase34_35_transition(
            current,
            NormalizedAction2D(*proposed_action),
            dynamics,
        )
        prediction = OneStepPrediction(
            next_state=transition.next_state,
            speed_ratio=(
                math.hypot(transition.next_state.vx, transition.next_state.vy)
                / (target_speed + epsilon)
            ),
        )
        predictions.append(prediction)
        return prediction

    decision = evaluate_overspeed_veto(
        state,
        action,
        predictor,
        threshold=OVERSPEED_THRESHOLD,
    )
    if not predictions:
        raise Stage2AHazardArrestRunnerError("one-step prediction was not produced")
    nominal = predictions[0]
    return OneStepActionEvaluation(
        branch_id=branch_id,
        action=action,
        action_hash=canonical_sha256({"action": list(action)}),
        predicted_state=nominal.next_state,
        predicted_state_hash=runtime_state_hash(nominal.next_state),
        predicted_speed_ratio=nominal.speed_ratio,
        predicted_headroom=OVERSPEED_THRESHOLD - nominal.speed_ratio,
        final_veto_decision=decision,
        fallback_prediction_count=max(0, len(predictions) - 1),
    )


def _trace_entry_map(index: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    traces = index.get("traces")
    if not isinstance(traces, list):
        raise Stage2AHazardArrestRunnerError("Stage 1B trace index is malformed")
    return {
        str(item["trace_id"]): item
        for item in traces
        if isinstance(item, Mapping)
    }


def qualify_frozen_stage1b_boundaries(
    repository_root: Path,
) -> tuple[dict[str, object], ...]:
    root = repository_root.resolve()
    index, trace_rows = load_trace_set(root)
    entries = _trace_entry_map(index)
    eligible: list[dict[str, object]] = []
    states_inspected = 0
    normal_prediction_evaluations = 0
    hazard_prediction_evaluations = 0
    for rows in trace_rows:
        if not rows:
            continue
        trace_id = str(rows[0].get("trace_id"))
        entry = entries[trace_id]
        if entry.get("explicit_abort") is True:
            continue
        member_id = str(entry["registry_member_id"])
        registered = load_registered_branch_state(root, member_id)
        for row in rows:
            source = _mapping(row.get("source_event"), "source_event")
            if source.get("event_type") != "transition":
                continue
            event_index = int(source["event_index"])
            prefix_count = event_index - 1
            if prefix_count < 0 or prefix_count >= MAXIMUM_PHYSICAL_TRANSITIONS_PER_RUN:
                continue
            states_inspected += 1
            if source.get("event_valid") is not True:
                continue
            statuses = _guard_statuses(row)
            if any(
                statuses.get(atom) != "true"
                for atom in (
                    "state_evidence_valid",
                    "instrumentation_evaluation_valid",
                    "recovery_evaluation_valid",
                )
            ):
                continue
            state = _state_from_source_event(source)
            realized_ratio = _current_speed_ratio(source)
            if realized_ratio > OVERSPEED_THRESHOLD:
                continue
            for normal_branch_id in NORMAL_BRANCH_IDS:
                normal_prediction_evaluations += 1
                normal = evaluate_branch_without_execution(
                    registered, state, normal_branch_id
                )
                if not (
                    normal.predicted_speed_ratio > OVERSPEED_THRESHOLD
                    and normal.final_veto_decision.decision == "veto"
                ):
                    continue
                hazard_prediction_evaluations += 1
                hazard = evaluate_branch_without_execution(
                    registered, state, HAZARD_BRANCH_ID
                )
                if hazard.final_veto_decision.decision != "allow":
                    continue
                candidate = {
                    "registry_member_id": member_id,
                    "case_id": str(entry["case_id"]),
                    "source_trace_id": trace_id,
                    "source_trace_path": str(entry["trace_path"]),
                    "source_trace_sha256": str(entry["trace_sha256"]),
                    "source_trace_record_hash": str(
                        row["canonical_trace_record_hash"]
                    ),
                    "prefix_branch_id": str(entry["branch_id"]),
                    "prefix_transition_count": prefix_count,
                    "boundary_event_index": event_index,
                    "boundary_state": state_document(state),
                    "boundary_state_hash": str(source["pre_state_hash"]),
                    "boundary_state_hash_schema": SOURCE_TRACE_STATE_HASH_SCHEMA,
                    "runtime_boundary_state_hash": runtime_state_hash(state),
                    "current_realized_speed_ratio": realized_ratio,
                    "current_realized_headroom": OVERSPEED_THRESHOLD - realized_ratio,
                    "normal_branch_id": normal_branch_id,
                    "normal_action": list(normal.action),
                    "normal_action_hash": normal.action_hash,
                    "normal_predicted_speed_ratio": normal.predicted_speed_ratio,
                    "normal_predicted_headroom": normal.predicted_headroom,
                    "normal_predicted_state_hash": normal.predicted_state_hash,
                    "normal_final_veto_decision": normal.final_veto_decision.decision,
                    "hazard_action_source": HAZARD_BRANCH_ID,
                    "hazard_action": list(hazard.action),
                    "hazard_action_hash": hazard.action_hash,
                    "hazard_predicted_speed_ratio": hazard.predicted_speed_ratio,
                    "hazard_predicted_headroom": hazard.predicted_headroom,
                    "hazard_predicted_state_hash": hazard.predicted_state_hash,
                    "hazard_final_veto_decision": hazard.final_veto_decision.decision,
                    "selection_key": [
                        member_id,
                        trace_id,
                        prefix_count,
                        normal_branch_id,
                    ],
                    "remaining_transition_capacity": (
                        MAXIMUM_PHYSICAL_TRANSITIONS_PER_RUN - prefix_count
                    ),
                }
                candidate["candidate_identity_hash"] = canonical_sha256(candidate)
                eligible.append(candidate)
    eligible.sort(key=lambda item: tuple(item["selection_key"]))
    metadata = {
        "states_inspected": states_inspected,
        "normal_prediction_evaluations": normal_prediction_evaluations,
        "hazard_prediction_evaluations": hazard_prediction_evaluations,
        "offline_prediction_evaluations": (
            normal_prediction_evaluations + hazard_prediction_evaluations
        ),
    }
    return (metadata, *eligible)


def build_qualification_payloads(
    repository_root: Path,
    *,
    implementation_commit: str,
) -> dict[str, bytes]:
    results = qualify_frozen_stage1b_boundaries(repository_root)
    metadata = dict(results[0])
    eligible = [dict(item) for item in results[1:]]
    selected = dict(eligible[0]) if eligible else None
    if selected is not None:
        selected["selection_status"] = "eligible_boundary_selected"
        selected["selection_rule"] = (
            "lexical_registry_member_source_trace_prefix_count_normal_branch"
        )
        selected["selection_hash"] = canonical_sha256(selected)
    selected_document: dict[str, object] = (
        selected
        if selected is not None
        else {
            "selection_status": "no_eligible_boundary",
            "selection_rule": (
                "lexical_registry_member_source_trace_prefix_count_normal_branch"
            ),
            "selection_hash": canonical_sha256(
                {
                    "selection_status": "no_eligible_boundary",
                    "selection_rule": (
                        "lexical_registry_member_source_trace_prefix_count_normal_branch"
                    ),
                }
            ),
        }
    )
    eligible_document = {
        "schema_version": STAGE2A_SCHEMA_VERSION,
        "eligibility_contract": {
            "realized_speed_ratio": "<=1.90",
            "normal_predicted_speed_ratio": ">1.90",
            "normal_action_final_veto": "veto",
            "hazard_action_source": HAZARD_BRANCH_ID,
            "hazard_action_final_veto": "allow",
            "maximum_physical_transitions_per_run": (
                MAXIMUM_PHYSICAL_TRANSITIONS_PER_RUN
            ),
        },
        "eligible_boundary_count": len(eligible),
        "eligible_boundaries": eligible,
    }
    manifest = {
        "qualification_id": "stage2a_hazard_arrest_qualification_v0",
        "schema_version": STAGE2A_SCHEMA_VERSION,
        "completed_date": COMPLETED_DATE,
        "implementation_commit": implementation_commit,
        "source_trace_set_path": TRACE_SET_OUTPUT_PATH.as_posix(),
        "source_trace_set_aggregate_hash": (
            "ab4fd8a70e2aa446e4996126a53685999f55a24baa2522a688ed72b0c2d5cfa0"
        ),
        "registry_manifest_hash": EXPECTED_REGISTRY_MANIFEST_HASH,
        "registry_aggregate_hash": EXPECTED_REGISTRY_AGGREGATE_HASH,
        **metadata,
        "eligible_boundary_count": len(eligible),
        "selection_status": selected_document["selection_status"],
        "selected_experiment_hash": selected_document["selection_hash"],
        "physical_executions": 0,
        "automatic_retry": False,
        "scientific_thresholds_added": False,
        "active_authority_executed": False,
        "artifact_filenames": list(QUALIFICATION_ARTIFACTS),
    }
    manifest["canonical_manifest_hash"] = canonical_sha256(manifest)
    summary = (
        "# Stage 2A Hazard-Arrest Qualification v0\n\n"
        f"Completed: {COMPLETED_DATE}\n\n"
        "## Status\n\n"
        + (
            "One provenance-bound execution boundary was selected deterministically.\n\n"
            if eligible
            else "No eligible execution boundary exists in the frozen Stage 1B traces.\n\n"
        )
        + "## Boundary\n\n"
        "Qualification used only frozen measured Stage 1B trace states and prediction-only "
        "evaluation. Physical executions: 0. The ordering did not rank intervention quality.\n\n"
        "## Claims\n\n"
        "This audit establishes execution-path eligibility only. It does not demonstrate "
        "recovery improvement, stability, optimality, safety, or threshold validation.\n"
    ).encode("utf-8")
    return {
        "qualification_manifest.json": json.dumps(
            manifest, indent=2, sort_keys=True
        ).encode("utf-8")
        + b"\n",
        "eligible_boundaries.json": json.dumps(
            eligible_document, indent=2, sort_keys=True
        ).encode("utf-8")
        + b"\n",
        "selected_experiment.json": json.dumps(
            selected_document, indent=2, sort_keys=True
        ).encode("utf-8")
        + b"\n",
        "summary.md": summary,
    }


def validate_qualification_payloads(payloads: Mapping[str, bytes]) -> None:
    if set(payloads) != set(QUALIFICATION_ARTIFACTS):
        raise Stage2AHazardArrestRunnerError(
            "qualification artifact set is incomplete"
        )
    manifest = json.loads(payloads["qualification_manifest.json"])
    supplied = manifest.pop("canonical_manifest_hash", None)
    if supplied != canonical_sha256(manifest):
        raise Stage2AHazardArrestRunnerError(
            "qualification manifest hash mismatch"
        )
    if manifest.get("physical_executions") != 0:
        raise Stage2AHazardArrestRunnerError(
            "qualification cannot execute physics"
        )
    eligible = json.loads(payloads["eligible_boundaries.json"])
    if eligible.get("schema_version") != STAGE2A_SCHEMA_VERSION:
        raise Stage2AHazardArrestRunnerError("qualification schema version mismatch")
    boundaries = eligible.get("eligible_boundaries")
    if not isinstance(boundaries, list):
        raise Stage2AHazardArrestRunnerError("eligible boundary list is malformed")
    if eligible.get("eligible_boundary_count") != len(boundaries) or manifest.get(
        "eligible_boundary_count"
    ) != len(boundaries):
        raise Stage2AHazardArrestRunnerError("eligible boundary count mismatch")
    if boundaries != sorted(boundaries, key=lambda item: tuple(item["selection_key"])):
        raise Stage2AHazardArrestRunnerError("eligible boundary ordering changed")
    for boundary in boundaries:
        candidate = dict(boundary)
        candidate_hash = candidate.pop("candidate_identity_hash", None)
        if candidate_hash != canonical_sha256(candidate):
            raise Stage2AHazardArrestRunnerError(
                "eligible boundary identity hash mismatch"
            )
        if (
            candidate.get("current_realized_speed_ratio") > OVERSPEED_THRESHOLD
            or candidate.get("normal_predicted_speed_ratio") <= OVERSPEED_THRESHOLD
            or candidate.get("normal_final_veto_decision") != "veto"
            or candidate.get("hazard_action_source") != HAZARD_BRANCH_ID
            or candidate.get("hazard_final_veto_decision") != "allow"
            or candidate.get("normal_branch_id") not in NORMAL_BRANCH_IDS
            or candidate.get("remaining_transition_capacity", 0) < 1
        ):
            raise Stage2AHazardArrestRunnerError(
                "eligible boundary violates the frozen trigger contract"
            )
        expected_key = [
            candidate.get("registry_member_id"),
            candidate.get("source_trace_id"),
            candidate.get("prefix_transition_count"),
            candidate.get("normal_branch_id"),
        ]
        if candidate.get("selection_key") != expected_key:
            raise Stage2AHazardArrestRunnerError(
                "eligible boundary selection key mismatch"
            )
    selected = json.loads(payloads["selected_experiment.json"])
    selection_hash = selected.pop("selection_hash", None)
    if selection_hash != canonical_sha256(selected):
        raise Stage2AHazardArrestRunnerError("selected experiment hash mismatch")
    if manifest.get("selected_experiment_hash") != selection_hash:
        raise Stage2AHazardArrestRunnerError(
            "qualification manifest selection hash mismatch"
        )
    if boundaries:
        for key, value in boundaries[0].items():
            if selected.get(key) != value:
                raise Stage2AHazardArrestRunnerError(
                    "selected experiment is not the first eligible boundary"
                )
        if selected.get("selection_status") != "eligible_boundary_selected":
            raise Stage2AHazardArrestRunnerError("selected boundary status mismatch")
    elif selected.get("selection_status") != "no_eligible_boundary":
        raise Stage2AHazardArrestRunnerError("no-eligible status mismatch")


def load_qualification_payloads(repository_root: Path) -> dict[str, bytes]:
    source = repository_root / QUALIFICATION_OUTPUT_PATH
    payloads = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    validate_qualification_payloads(payloads)
    return payloads


def load_selected_experiment(repository_root: Path) -> dict[str, object]:
    payloads = load_qualification_payloads(repository_root)
    selected = json.loads(payloads["selected_experiment.json"])
    if selected.get("selection_status") != "eligible_boundary_selected":
        raise Stage2AHazardArrestRunnerError(
            "qualification did not select an eligible boundary"
        )
    return selected


def _source_trace(
    repository_root: Path,
    selected: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    path = repository_root / TRACE_SET_OUTPUT_PATH / str(selected["source_trace_path"])
    if file_sha256(path) != selected.get("source_trace_sha256"):
        raise Stage2AHazardArrestRunnerError("selected source trace hash mismatch")
    rows = tuple(json.loads(line) for line in path.read_text("utf-8").splitlines())
    return rows


def _trace_record(document: Mapping[str, object]) -> dict[str, object]:
    result = dict(document)
    result["canonical_record_hash"] = canonical_sha256(result)
    return result


@dataclass(frozen=True, slots=True)
class PrefixReplay:
    final_state: CartesianState2D
    records: tuple[dict[str, object], ...]
    physical_transition_count: int


RegisteredExecutor = Callable[..., RecoveryBranchExecutionResult]


def reproduce_selected_prefix(
    repository_root: Path,
    selected: Mapping[str, object],
    *,
    implementation_commit: str,
    step_executor: RegisteredExecutor = execute_registered_recovery_branch,
) -> PrefixReplay:
    member_id = str(selected["registry_member_id"])
    registered = load_registered_branch_state(repository_root, member_id)
    _, current = build_registered_runtime_identity(
        registered,
        implementation_commit=implementation_commit,
        branch_id=str(selected["prefix_branch_id"]),
    )
    source_rows = _source_trace(repository_root, selected)
    initial_source = _mapping(source_rows[0].get("source_event"), "initial source")
    if source_trace_state_hash(current) != initial_source.get("pre_state_hash"):
        raise Stage2AHazardArrestRunnerError("prefix initial state mismatch")
    records: list[dict[str, object]] = []
    prefix_count = int(selected["prefix_transition_count"])
    for step in range(1, prefix_count + 1):
        source = _mapping(source_rows[step].get("source_event"), "prefix source")
        execution = step_executor(
            member_id,
            str(selected["prefix_branch_id"]),
            horizon_steps=1,
            current_state=current,
        )
        monitor = execution.monitor_decision
        checks = {
            "pre_state_hash": source_trace_state_hash(execution.previous_state)
            == source.get("pre_state_hash"),
            "proposed_action": list(execution.action or ())
            == source.get("proposed_action"),
            "monitor_decision": monitor is not None
            and monitor.decision == source.get("monitor_decision"),
            "predicted_state_hash": execution.predicted_nominal_state is not None
            and source_trace_state_hash(execution.predicted_nominal_state)
            == source.get("predicted_state_hash"),
            "realized_state_hash": execution.next_state is not None
            and source_trace_state_hash(execution.next_state)
            == source.get("realized_state_hash"),
            "transition_executed": execution.executed is True
            and source.get("transition_executed") is True,
        }
        if not all(checks.values()) or execution.next_state is None:
            raise Stage2AHazardArrestRunnerError(
                f"prefix reproduction mismatch at transition {step}: {checks}"
            )
        records.append(
            _trace_record(
                {
                    "event_type": "prefix_transition",
                    "prefix_transition_index": step,
                    "source_event_hash": source["canonical_event_sha256"],
                    "pre_state_hash": source_trace_state_hash(
                        execution.previous_state
                    ),
                    "proposed_action": list(execution.action),
                    "final_veto_decision": monitor.decision,
                    "predicted_state_hash": source_trace_state_hash(
                        execution.predicted_nominal_state
                    ),
                    "realized_state_hash": source_trace_state_hash(
                        execution.next_state
                    ),
                    "physical_transition_count": 1,
                }
            )
        )
        current = execution.next_state
    if source_trace_state_hash(current) != selected.get("boundary_state_hash"):
        raise Stage2AHazardArrestRunnerError("reproduced boundary state mismatch")
    if runtime_state_hash(current) != selected.get("runtime_boundary_state_hash"):
        raise Stage2AHazardArrestRunnerError(
            "reproduced runtime boundary state mismatch"
        )
    return PrefixReplay(
        final_state=current,
        records=tuple(records),
        physical_transition_count=prefix_count,
    )


def _verify_boundary_evaluation(
    evaluation: OneStepActionEvaluation,
    selected: Mapping[str, object],
    prefix: str,
) -> None:
    checks = {
        "action_hash": evaluation.action_hash == selected[f"{prefix}_action_hash"],
        "predicted_state_hash": evaluation.predicted_state_hash
        == selected[f"{prefix}_predicted_state_hash"],
        "predicted_speed_ratio": evaluation.predicted_speed_ratio
        == selected[f"{prefix}_predicted_speed_ratio"],
        "final_veto_decision": evaluation.final_veto_decision.decision
        == selected[f"{prefix}_final_veto_decision"],
    }
    if not all(checks.values()):
        raise Stage2AHazardArrestRunnerError(
            f"{prefix} boundary identity mismatch: {checks}"
        )


def _speed_ratio_for_registered(
    registered: RegisteredBranchState,
    state: CartesianState2D,
) -> float:
    _, target_speed, epsilon = _registered_dynamics(registered)
    return math.hypot(state.vx, state.vy) / (target_speed + epsilon)


def _stop_report(
    registered: RegisteredBranchState,
    execution: RecoveryBranchExecutionResult,
    *,
    recovery_transition_count: int,
) -> RecoveryStopConditionReport:
    document = registered.as_document()
    configuration = _mapping(document["simulator_configuration"], "configuration")
    constants = _mapping(configuration["simulator_constants"], "constants")
    ratio = (
        _speed_ratio_for_registered(registered, execution.next_state)
        if execution.next_state is not None
        else None
    )
    return evaluate_recovery_stop_conditions(
        execution_terminal_reason=execution.terminal_reason,
        next_state=execution.next_state,
        realized_speed_ratio=ratio,
        overspeed_threshold=OVERSPEED_THRESHOLD,
        recovery_transition_count=recovery_transition_count,
        recovery_horizon_steps=MAXIMUM_PHYSICAL_TRANSITIONS_PER_RUN,
        total_transition_count=(
            registered.member.nominal_prefix_transition_count
            + recovery_transition_count
        ),
        total_horizon_steps=int(constants["max_steps"]),
    )


def _instrumented_metrics(
    registered: RegisteredBranchState,
    state: CartesianState2D,
    *,
    source_step: int,
) -> dict[str, object]:
    document = registered.as_document()
    configuration = _mapping(document["simulator_configuration"], "configuration")
    constants = _mapping(configuration["simulator_constants"], "constants")
    orbital = derive_orbital_state(
        InstrumentationState2D(state.x, state.y, state.vx, state.vy),
        OrbitalConfiguration(
            mu=_finite(constants["mu"], "mu"),
            target_radius=_finite(constants["target_radius"], "target_radius"),
            ratio_denominator_epsilon=_finite(
                constants["speed_ratio_denominator_epsilon"], "ratio epsilon"
            ),
            speed_ratio_denominator_epsilon=_finite(
                constants["speed_ratio_denominator_epsilon"], "speed epsilon"
            ),
            gravity_model_id=CURRENT_GRAVITY_MODEL_ID,
        ),
        source_step=source_step,
    )
    selected_fields = (
        "radius_error_ratio",
        "radial_velocity",
        "radial_velocity_ratio",
        "tangential_velocity_error",
        "tangential_velocity_error_ratio",
        "realized_speed_ratio",
        "overspeed_headroom",
        "specific_energy_error",
    )
    values = {field: orbital.field(field).value for field in selected_fields}
    recovery = derive_phase34_recoverability(
        orbital.field("radius_error_ratio"),
        orbital.field("radial_velocity_ratio"),
        orbital.field("tangential_velocity_error_ratio"),
        source_step=source_step,
    )
    for field in (
        "radius_component_pass",
        "radial_velocity_component_pass",
        "tangential_velocity_component_pass",
        "phase34_compatible_recoverability",
    ):
        values[field] = recovery.field(field).value
    return values


@dataclass(frozen=True, slots=True)
class Stage2AMeasuredExperiment:
    selected: Mapping[str, object]
    baseline_summary: Mapping[str, object]
    active_summary: Mapping[str, object]
    boundary_equivalence: Mapping[str, object]
    authority_report: Mapping[str, object]
    final_veto_report: Mapping[str, object]
    intervention_effect: Mapping[str, object]
    release_report: Mapping[str, object]
    baseline_trace: tuple[Mapping[str, object], ...]
    active_trace: tuple[Mapping[str, object], ...]


def execute_selected_experiment(
    repository_root: Path,
    selected: Mapping[str, object],
    *,
    implementation_commit: str,
    step_executor: RegisteredExecutor = execute_registered_recovery_branch,
) -> Stage2AMeasuredExperiment:
    member_id = str(selected["registry_member_id"])
    registered = load_registered_branch_state(repository_root, member_id)
    baseline_prefix = reproduce_selected_prefix(
        repository_root,
        selected,
        implementation_commit=implementation_commit,
        step_executor=step_executor,
    )
    active_prefix = reproduce_selected_prefix(
        repository_root,
        selected,
        implementation_commit=implementation_commit,
        step_executor=step_executor,
    )
    prefix_checks = {
        "same_boundary_state": baseline_prefix.final_state
        == active_prefix.final_state,
        "same_prefix_records": baseline_prefix.records == active_prefix.records,
        "same_prefix_transition_count": baseline_prefix.physical_transition_count
        == active_prefix.physical_transition_count,
    }
    if not all(prefix_checks.values()):
        raise Stage2AHazardArrestRunnerError("baseline/active prefix mismatch")
    boundary_state = baseline_prefix.final_state
    normal_prediction = evaluate_branch_without_execution(
        registered, boundary_state, str(selected["normal_branch_id"])
    )
    _verify_boundary_evaluation(normal_prediction, selected, "normal")
    if not (
        _speed_ratio_for_registered(registered, boundary_state)
        <= OVERSPEED_THRESHOLD
        and normal_prediction.predicted_speed_ratio > OVERSPEED_THRESHOLD
    ):
        raise Stage2AHazardArrestRunnerError("frozen authority trigger no longer holds")

    baseline_execution = step_executor(
        member_id,
        str(selected["normal_branch_id"]),
        horizon_steps=1,
        current_state=boundary_state,
    )
    if (
        baseline_execution.executed
        or baseline_execution.transition_count != 0
        or baseline_execution.monitor_decision is None
        or baseline_execution.monitor_decision.decision != "veto"
    ):
        raise Stage2AHazardArrestRunnerError(
            "baseline boundary did not preserve normal-action veto"
        )
    if canonical_sha256({"action": list(baseline_execution.action or ())}) != selected[
        "normal_action_hash"
    ]:
        raise Stage2AHazardArrestRunnerError("baseline action identity mismatch")
    if (
        baseline_execution.predicted_nominal_state is None
        or runtime_state_hash(baseline_execution.predicted_nominal_state)
        != selected["normal_predicted_state_hash"]
        or baseline_execution.monitor_decision.decision
        != selected["normal_final_veto_decision"]
    ):
        raise Stage2AHazardArrestRunnerError(
            "baseline prediction or Final Veto identity mismatch"
        )
    baseline_stop = _stop_report(
        registered,
        baseline_execution,
        recovery_transition_count=baseline_prefix.physical_transition_count,
    )

    active_boundary_state = active_prefix.final_state
    session = HazardArrestAuthoritySession(
        session_id=f"{STAGE2A_EXPERIMENT_ID}:{selected['selection_hash']}"
    )
    authority = HazardArrestAuthorityInput(
        authority_enabled=True,
        authority_granted=True,
        requested_phase="hazard_arrest",
        trigger_evidence_authorized=True,
    )
    evidence = HazardArrestEvidence(
        state=active_boundary_state,
        state_status=AuthorityEvidenceStatus.AVAILABLE_VALID,
        instrumentation_status=AuthorityEvidenceStatus.AVAILABLE_VALID,
        recovery_evaluation_status=AuthorityEvidenceStatus.AVAILABLE_VALID,
        hazard_status=AuthorityEvidenceStatus.AVAILABLE_VALID,
        hazard_kind=HazardEvidenceKind.PREDICTED_OVERSPEED,
        hazard_asserted=True,
        realized_speed_ratio=_speed_ratio_for_registered(
            registered, active_boundary_state
        ),
        predicted_speed_ratio=normal_prediction.predicted_speed_ratio,
    )
    authority_result = request_hazard_arrest_proposal(
        session, authority, evidence
    )
    proposal = authority_result.decision.proposal
    if proposal is None or proposal.action_source != PROVISIONAL_ACTION_SOURCE:
        raise Stage2AHazardArrestRunnerError("authority adapter did not produce hazard proposal")
    if canonical_sha256({"action": list(proposal.action)}) != selected[
        "hazard_action_hash"
    ]:
        raise Stage2AHazardArrestRunnerError("hazard proposal identity mismatch")
    hazard_prediction = evaluate_branch_without_execution(
        registered, active_boundary_state, HAZARD_BRANCH_ID
    )
    _verify_boundary_evaluation(hazard_prediction, selected, "hazard")
    active_execution = step_executor(
        member_id,
        HAZARD_BRANCH_ID,
        horizon_steps=1,
        current_state=active_boundary_state,
    )
    consumed_session = consume_hazard_arrest_proposal(
        authority_result.session, proposal
    )
    if active_execution.action != proposal.action:
        raise Stage2AHazardArrestRunnerError("executed hazard action differs from proposal")
    if active_execution.monitor_decision is None:
        raise Stage2AHazardArrestRunnerError("hazard proposal bypassed Final Veto")
    if (
        active_execution.predicted_nominal_state is None
        or runtime_state_hash(active_execution.predicted_nominal_state)
        != selected["hazard_predicted_state_hash"]
        or active_execution.monitor_decision.decision
        != selected["hazard_final_veto_decision"]
    ):
        raise Stage2AHazardArrestRunnerError(
            "hazard prediction or Final Veto identity mismatch"
        )
    if active_execution.executed:
        if (
            active_execution.transition_count != 1
            or active_execution.next_state is None
            or active_execution.predicted_nominal_state != active_execution.next_state
        ):
            raise Stage2AHazardArrestRunnerError(
                "hazard prediction and realization differ"
            )
    elif active_execution.transition_count != 0:
        raise Stage2AHazardArrestRunnerError("rejected hazard action changed counters")
    active_count = active_prefix.physical_transition_count + active_execution.transition_count
    active_stop = _stop_report(
        registered,
        active_execution,
        recovery_transition_count=active_count,
    )

    release_evaluated = active_execution.executed
    resumed_prediction: OneStepActionEvaluation | None = None
    release_status = RELEASE_NOT_AUTHORIZED
    release_reason = "intervention_not_executed"
    if active_execution.executed and active_execution.next_state is not None:
        resumed_prediction = evaluate_branch_without_execution(
            registered,
            active_execution.next_state,
            str(selected["normal_branch_id"]),
        )
        post_ratio = _speed_ratio_for_registered(
            registered, active_execution.next_state
        )
        adverse_stop = active_stop.terminal_reason is not None
        evaluator_valid = (
            active_stop.status_for(INVALID_RECOVERY_EVALUATION) == "clear"
            and False
        )
        if adverse_stop:
            release_reason = f"active_adverse_stop:{active_stop.terminal_reason}"
        elif post_ratio > OVERSPEED_THRESHOLD:
            release_reason = "post_intervention_realized_overspeed"
        elif resumed_prediction.predicted_speed_ratio > OVERSPEED_THRESHOLD:
            release_reason = "resumed_normal_prediction_overspeed"
        elif not evaluator_valid:
            release_reason = "release_evaluator_evidence_not_available"
        else:
            release_status = RELEASE_AUTHORIZED
            release_reason = "fresh_clear_evidence_for_predeclared_normal_branch"

    baseline_summary = {
        "run_id": "baseline",
        "registry_member_id": member_id,
        "prefix_transition_count": baseline_prefix.physical_transition_count,
        "boundary_transition_count": 0,
        "physical_transition_count": baseline_prefix.physical_transition_count,
        "normal_branch_id": selected["normal_branch_id"],
        "normal_action": list(baseline_execution.action or ()),
        "normal_predicted_speed_ratio": normal_prediction.predicted_speed_ratio,
        "final_veto_decision": baseline_execution.monitor_decision.decision,
        "fallback_execution_count": 0,
        "terminal_reason": baseline_stop.terminal_reason,
        "final_state_hash": runtime_state_hash(boundary_state),
    }
    post_state = active_execution.next_state
    active_summary = {
        "run_id": "active",
        "registry_member_id": member_id,
        "prefix_transition_count": active_prefix.physical_transition_count,
        "boundary_transition_count": active_execution.transition_count,
        "physical_transition_count": active_count,
        "proposal_id": proposal.proposal_id,
        "proposal_count": consumed_session.proposal_count,
        "proposal_consumed": consumed_session.proposal_consumed,
        "hazard_action": list(proposal.action),
        "hazard_final_veto_decision": active_execution.monitor_decision.decision,
        "fallback_execution_count": 0,
        "terminal_reason": active_stop.terminal_reason,
        "final_state_hash": runtime_state_hash(post_state or active_boundary_state),
    }
    boundary_equivalence = {
        "checks": prefix_checks,
        "same_boundary_state_hash": source_trace_state_hash(boundary_state)
        == selected["boundary_state_hash"],
        "same_runtime_boundary_state_hash": runtime_state_hash(boundary_state)
        == selected["runtime_boundary_state_hash"],
        "same_normal_action_hash": normal_prediction.action_hash
        == selected["normal_action_hash"],
        "same_normal_prediction_hash": normal_prediction.predicted_state_hash
        == selected["normal_predicted_state_hash"],
    }
    boundary_equivalence["all_required_prefix_checks"] = all(
        value
        for key, value in boundary_equivalence.items()
        if key != "checks"
    ) and all(prefix_checks.values())
    authority_report = {
        "requested_phase": "hazard_arrest",
        "proposal_generated": authority_result.decision.proposal_generated,
        "proposal_count": consumed_session.proposal_count,
        "proposal_consumed": consumed_session.proposal_consumed,
        "second_intervention_count": 0,
        "unauthorized_phase_count": 0,
        "authority_leakage_count": 0,
        "invalid_evidence_consumption_count": 0,
        "shadow_output_consumed_by_physical_runtime": False,
    }
    final_veto_report = {
        "monitor_id": active_execution.monitor_decision.monitor_id,
        "baseline_decision": baseline_execution.monitor_decision.decision,
        "hazard_decision": active_execution.monitor_decision.decision,
        "final_veto_bypass_count": 0,
        "fallback_execution_count": 0,
        "normal_counterfactual_prediction_label": (
            "counterfactual_one_step_prediction"
        ),
    }
    pre_metrics = _instrumented_metrics(
        registered, boundary_state, source_step=int(selected["prefix_transition_count"])
    )
    post_metrics = (
        _instrumented_metrics(registered, post_state, source_step=active_count)
        if post_state is not None
        else None
    )
    intervention_effect = {
        "intervention_executed": active_execution.executed,
        "prediction_realization_equal": (
            active_execution.predicted_nominal_state == post_state
            if active_execution.executed
            else None
        ),
        "pre_intervention": pre_metrics,
        "post_intervention": post_metrics,
        "component_changes": (
            {
                key: post_metrics[key] - pre_metrics[key]
                for key in (
                    "realized_speed_ratio",
                    "overspeed_headroom",
                    "radius_error_ratio",
                    "radial_velocity",
                    "tangential_velocity_error",
                    "specific_energy_error",
                )
            }
            if post_metrics is not None
            else None
        ),
    }
    release_report = {
        "release_evaluated": release_evaluated,
        "release_status": release_status,
        "release_reason": release_reason,
        "return_authority_to": selected["normal_branch_id"],
        "resumed_normal_predicted_speed_ratio": (
            resumed_prediction.predicted_speed_ratio
            if resumed_prediction is not None
            else None
        ),
        "resumed_normal_predicted_headroom": (
            resumed_prediction.predicted_headroom
            if resumed_prediction is not None
            else None
        ),
        "resumed_physical_action_count": 0,
    }
    baseline_boundary_record = _trace_record(
        {
            "event_type": "baseline_boundary_veto",
            "boundary_state_hash": selected["boundary_state_hash"],
            "normal_action": list(baseline_execution.action or ()),
            "predicted_state_hash": normal_prediction.predicted_state_hash,
            "predicted_speed_ratio": normal_prediction.predicted_speed_ratio,
            "final_veto_decision": "veto",
            "physical_transition_count": 0,
            "fallback_execution_count": 0,
        }
    )
    active_records = [
        *active_prefix.records,
        _trace_record(
            {
                "event_type": "authority_request",
                "boundary_state_hash": selected["boundary_state_hash"],
                "normal_predicted_speed_ratio": normal_prediction.predicted_speed_ratio,
                "proposal_id": proposal.proposal_id,
                "proposal_action": list(proposal.action),
                "physical_transition_count": 0,
            }
        ),
        _trace_record(
            {
                "event_type": "hazard_intervention",
                "proposal_id": proposal.proposal_id,
                "proposal_consumed": consumed_session.proposal_consumed,
                "final_veto_decision": active_execution.monitor_decision.decision,
                "predicted_state_hash": hazard_prediction.predicted_state_hash,
                "realized_state_hash": active_execution.next_state_hash,
                "prediction_realization_equal": (
                    active_execution.predicted_nominal_state == post_state
                    if active_execution.executed
                    else None
                ),
                "physical_transition_count": active_execution.transition_count,
            }
        ),
        _trace_record(
            {
                "event_type": "release_evaluation",
                **release_report,
                "physical_transition_count": 0,
            }
        ),
    ]
    return Stage2AMeasuredExperiment(
        selected=selected,
        baseline_summary=baseline_summary,
        active_summary=active_summary,
        boundary_equivalence=boundary_equivalence,
        authority_report=authority_report,
        final_veto_report=final_veto_report,
        intervention_effect=intervention_effect,
        release_report=release_report,
        baseline_trace=(*baseline_prefix.records, baseline_boundary_record),
        active_trace=tuple(active_records),
    )


def _aggregate_paths(repository_root: Path, relative_paths: Sequence[str]) -> str:
    files: list[Path] = []
    for relative in relative_paths:
        path = (repository_root / relative).resolve()
        try:
            path.relative_to(repository_root.resolve())
        except ValueError as exc:
            raise Stage2AHazardArrestRunnerError(
                f"protected path escapes repository: {relative}"
            ) from exc
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
        else:
            raise Stage2AHazardArrestRunnerError(
                f"protected path is missing: {relative}"
            )
    rows = [
        f"{path.relative_to(repository_root).as_posix()}|{file_sha256(path)}"
        for path in sorted(files, key=lambda item: item.as_posix())
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def protected_evidence_hashes(repository_root: Path) -> dict[str, str]:
    values = dict(historical_protected_evidence_hashes(repository_root))
    values.update(
        {
            name: _aggregate_paths(repository_root, paths)
            for name, paths in sorted(NEWER_PROTECTED_PATHS.items())
        }
    )
    return dict(sorted(values.items()))


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"\n".join(canonical_json_bytes(row) for row in rows) + b"\n"


def build_experiment_payloads(
    experiment: Stage2AMeasuredExperiment,
    *,
    implementation_commit: str,
    selection_commit: str,
    protected_before: Mapping[str, str],
    protected_after: Mapping[str, str],
) -> dict[str, bytes]:
    if dict(protected_before) != dict(protected_after):
        raise Stage2AHazardArrestRunnerError("protected evidence changed")
    selected = dict(experiment.selected)
    baseline_trace = _jsonl_bytes(experiment.baseline_trace)
    active_trace = _jsonl_bytes(experiment.active_trace)
    protected_report = {
        "before": dict(protected_before),
        "after": dict(protected_after),
        "all_protected_evidence_unchanged": True,
    }
    payloads = {
        "selected_case.json": _json_bytes(selected),
        "baseline_summary.json": _json_bytes(experiment.baseline_summary),
        "active_summary.json": _json_bytes(experiment.active_summary),
        "boundary_equivalence_report.json": _json_bytes(
            experiment.boundary_equivalence
        ),
        "authority_report.json": _json_bytes(experiment.authority_report),
        "final_veto_report.json": _json_bytes(experiment.final_veto_report),
        "intervention_effect.json": _json_bytes(experiment.intervention_effect),
        "release_report.json": _json_bytes(experiment.release_report),
        "protected_evidence_report.json": _json_bytes(protected_report),
        "traces/baseline.jsonl": baseline_trace,
        "traces/active.jsonl": active_trace,
    }
    manifest = {
        "experiment_id": STAGE2A_EXPERIMENT_ID,
        "schema_version": STAGE2A_SCHEMA_VERSION,
        "completed_date": COMPLETED_DATE,
        "implementation_commit": implementation_commit,
        "selection_commit": selection_commit,
        "selected_experiment_hash": selected["selection_hash"],
        "registry_member_id": selected["registry_member_id"],
        "source_trace_id": selected["source_trace_id"],
        "prefix_transition_count": selected["prefix_transition_count"],
        "normal_branch_id": selected["normal_branch_id"],
        "baseline_bounded_run_count": 1,
        "active_bounded_run_count": 1,
        "formal_invocation_count": 1,
        "automatic_retry_count": 0,
        "maximum_physical_transitions_per_run": (
            MAXIMUM_PHYSICAL_TRANSITIONS_PER_RUN
        ),
        "baseline_trace_sha256": sha256_bytes(baseline_trace),
        "active_trace_sha256": sha256_bytes(active_trace),
        "unauthorized_phase_count": 0,
        "authority_leakage_count": 0,
        "final_veto_bypass_count": 0,
        "fallback_execution_count": 0,
        "second_intervention_count": 0,
        "invalid_evidence_consumption_count": 0,
        "protected_evidence_unchanged": True,
        "scientific_claim": (
            "one provenance-bound one-intervention execution-path result only"
        ),
        "artifact_filenames": list(EXPERIMENT_ARTIFACTS),
    }
    manifest["canonical_manifest_hash"] = canonical_sha256(manifest)
    summary = (
        "# Stage 2A One-Intervention Hazard-Arrest Experiment v0\n\n"
        f"Completed: {COMPLETED_DATE}\n\n"
        "## Status\n\n"
        "One frozen baseline/active pair completed under the one-intervention authority boundary.\n\n"
        "## Interpretation\n\n"
        "The baseline normal proposal was vetoed without a fallback transition. The active "
        "run submitted at most one existing velocity-opposed proposal through unchanged Final "
        "Veto and stopped after release evaluation. A vetoed normal prediction is a "
        "counterfactual one-step prediction, not a measured baseline next state.\n\n"
        "## Claim restrictions\n\n"
        "This result does not establish general recovery improvement, controller superiority, "
        "stability, optimality, formal safety, handoff readiness, retreat capability, hardware "
        "validity, or deployment readiness.\n"
    ).encode("utf-8")
    payloads["summary.md"] = summary
    payloads["experiment_manifest.json"] = _json_bytes(manifest)
    return payloads


def validate_experiment_payloads(payloads: Mapping[str, bytes]) -> None:
    if set(payloads) != set(EXPERIMENT_ARTIFACTS):
        raise Stage2AHazardArrestRunnerError("experiment artifact set is incomplete")
    manifest = json.loads(payloads["experiment_manifest.json"])
    supplied = manifest.pop("canonical_manifest_hash", None)
    if supplied != canonical_sha256(manifest):
        raise Stage2AHazardArrestRunnerError("experiment manifest hash mismatch")
    if sha256_bytes(payloads["traces/baseline.jsonl"]) != manifest[
        "baseline_trace_sha256"
    ]:
        raise Stage2AHazardArrestRunnerError("baseline trace hash mismatch")
    if sha256_bytes(payloads["traces/active.jsonl"]) != manifest[
        "active_trace_sha256"
    ]:
        raise Stage2AHazardArrestRunnerError("active trace hash mismatch")
    for path in ("traces/baseline.jsonl", "traces/active.jsonl"):
        for line in payloads[path].splitlines():
            row = json.loads(line)
            supplied_hash = row.pop("canonical_record_hash", None)
            if supplied_hash != canonical_sha256(row):
                raise Stage2AHazardArrestRunnerError("trace record hash mismatch")
    baseline = json.loads(payloads["baseline_summary.json"])
    active = json.loads(payloads["active_summary.json"])
    selected = json.loads(payloads["selected_case.json"])
    boundary = json.loads(payloads["boundary_equivalence_report.json"])
    authority = json.loads(payloads["authority_report.json"])
    veto = json.loads(payloads["final_veto_report.json"])
    effect = json.loads(payloads["intervention_effect.json"])
    release = json.loads(payloads["release_report.json"])
    protected = json.loads(payloads["protected_evidence_report.json"])
    selected_copy = dict(selected)
    selection_hash = selected_copy.pop("selection_hash", None)
    if selection_hash != canonical_sha256(selected_copy) or manifest.get(
        "selected_experiment_hash"
    ) != selection_hash:
        raise Stage2AHazardArrestRunnerError("frozen selection identity mismatch")
    if manifest.get("registry_member_id") != selected.get(
        "registry_member_id"
    ) or manifest.get("normal_branch_id") != selected.get("normal_branch_id"):
        raise Stage2AHazardArrestRunnerError("experiment provenance mismatch")
    if boundary.get("all_required_prefix_checks") is not True or any(
        value is not True for value in boundary.get("checks", {}).values()
    ):
        raise Stage2AHazardArrestRunnerError("prefix equivalence report failed")
    if any(
        boundary.get(field) is not True
        for field in (
            "same_boundary_state_hash",
            "same_runtime_boundary_state_hash",
            "same_normal_action_hash",
            "same_normal_prediction_hash",
        )
    ):
        raise Stage2AHazardArrestRunnerError("boundary identity report failed")
    if baseline.get("boundary_transition_count") != 0 or baseline.get(
        "final_veto_decision"
    ) != "veto":
        raise Stage2AHazardArrestRunnerError("baseline veto boundary mismatch")
    if (
        baseline.get("normal_predicted_speed_ratio", 0) <= OVERSPEED_THRESHOLD
        or selected.get("current_realized_speed_ratio", math.inf)
        > OVERSPEED_THRESHOLD
    ):
        raise Stage2AHazardArrestRunnerError("measured trigger contract mismatch")
    if active.get("boundary_transition_count") not in (0, 1):
        raise Stage2AHazardArrestRunnerError("active transition count is invalid")
    if (
        authority.get("requested_phase") != "hazard_arrest"
        or authority.get("proposal_generated") is not True
        or authority.get("proposal_count") != 1
        or authority.get("proposal_consumed") is not True
    ):
        raise Stage2AHazardArrestRunnerError("proposal lifecycle mismatch")
    if veto.get("baseline_decision") != "veto" or veto.get(
        "hazard_decision"
    ) != active.get("hazard_final_veto_decision"):
        raise Stage2AHazardArrestRunnerError("Final Veto report mismatch")
    if active.get("boundary_transition_count") == 1 and effect.get(
        "prediction_realization_equal"
    ) is not True:
        raise Stage2AHazardArrestRunnerError(
            "intervention prediction/realization mismatch"
        )
    if release.get("return_authority_to") != selected.get(
        "normal_branch_id"
    ) or release.get("release_status") not in (
        RELEASE_AUTHORIZED,
        RELEASE_NOT_AUTHORIZED,
    ):
        raise Stage2AHazardArrestRunnerError("release contract mismatch")
    if any(
        value != 0
        for value in (
            authority.get("second_intervention_count"),
            authority.get("unauthorized_phase_count"),
            authority.get("authority_leakage_count"),
            authority.get("invalid_evidence_consumption_count"),
            veto.get("final_veto_bypass_count"),
            veto.get("fallback_execution_count"),
            release.get("resumed_physical_action_count"),
        )
    ):
        raise Stage2AHazardArrestRunnerError("Stage 2A isolation contract failed")
    if protected.get("all_protected_evidence_unchanged") is not True:
        raise Stage2AHazardArrestRunnerError("protected evidence report failed")
    if protected.get("before") != protected.get("after"):
        raise Stage2AHazardArrestRunnerError("protected evidence hashes differ")


def load_experiment_payloads(repository_root: Path) -> dict[str, bytes]:
    source = repository_root / EXPERIMENT_OUTPUT_PATH
    payloads = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    validate_experiment_payloads(payloads)
    return payloads


def validate_static_sources(repository_root: Path) -> dict[str, object]:
    index, rows = load_trace_set(repository_root)
    if len(rows) != 13:
        raise Stage2AHazardArrestRunnerError("Stage 1B trace count mismatch")
    source_states_validated = 0
    for trace in rows:
        for row in trace:
            source = _mapping(row.get("source_event"), "source_event")
            if source.get("event_type") == "transition":
                _state_from_source_event(source)
                source_states_validated += 1
    manifest = json.loads(
        (
            repository_root
            / "analysis/stage2a_active_hazard_arrest_preflight_v0/preflight_manifest.json"
        ).read_text("utf-8")
    )
    supplied = manifest.pop("canonical_manifest_hash", None)
    if supplied != canonical_sha256(manifest):
        raise Stage2AHazardArrestRunnerError("Stage 2A preflight hash mismatch")
    if OVERSPEED_THRESHOLD != 1.90 or OVERSPEED_COMPARATOR != ">":
        raise Stage2AHazardArrestRunnerError("overspeed semantics changed")
    if PROVISIONAL_ACTION_SOURCE != HAZARD_BRANCH_ID:
        raise Stage2AHazardArrestRunnerError("hazard action mapping changed")
    return {
        "source_trace_count": len(rows),
        "source_trace_index_count": len(index["traces"]),
        "preflight_manifest_hash": supplied,
        "overspeed_threshold": OVERSPEED_THRESHOLD,
        "overspeed_comparator": OVERSPEED_COMPARATOR,
        "source_state_hash_schema": SOURCE_TRACE_STATE_HASH_SCHEMA,
        "source_states_validated": source_states_validated,
        "physical_executions": 0,
    }


__all__ = [
    "COMPLETED_DATE",
    "EXPERIMENT_ARTIFACTS",
    "EXPERIMENT_OUTPUT_PATH",
    "HAZARD_BRANCH_ID",
    "MAXIMUM_PHYSICAL_TRANSITIONS_PER_RUN",
    "NORMAL_BRANCH_IDS",
    "OneStepActionEvaluation",
    "PrefixReplay",
    "QUALIFICATION_ARTIFACTS",
    "QUALIFICATION_OUTPUT_PATH",
    "RELEASE_AUTHORIZED",
    "RELEASE_NOT_AUTHORIZED",
    "SOURCE_TRACE_STATE_HASH_SCHEMA",
    "STAGE2A_EXPERIMENT_ID",
    "STAGE2A_SCHEMA_VERSION",
    "Stage2AHazardArrestRunnerError",
    "Stage2AMeasuredExperiment",
    "build_experiment_payloads",
    "build_qualification_payloads",
    "canonical_json_bytes",
    "canonical_sha256",
    "evaluate_branch_without_execution",
    "execute_selected_experiment",
    "load_experiment_payloads",
    "load_qualification_payloads",
    "load_selected_experiment",
    "protected_evidence_hashes",
    "qualify_frozen_stage1b_boundaries",
    "reproduce_selected_prefix",
    "sha256_bytes",
    "source_trace_state_hash",
    "state_document",
    "validate_experiment_payloads",
    "validate_qualification_payloads",
    "validate_static_sources",
]
