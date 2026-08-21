from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

from runtime_assurance.final_veto_monitor import (
    OVERSPEED_COMPARATOR,
    OVERSPEED_THRESHOLD,
    FinalVetoDecision,
    OneStepPrediction,
    evaluate_overspeed_veto,
)
from runtime_assurance.recovery_branch_executor import (
    generate_tangential_correction_action,
    generate_velocity_opposed_action,
    generate_zero_action,
)
from runtime_assurance.recovery_branch_state_extractor import (
    LegacyReproductionResult,
    PrefixExecutionResult,
    SourceCaseDefinition,
    _simulator_configuration_for_thrust,
    build_source_case_inventory,
    execute_nominal_prefix,
    protected_evidence_hashes as historical_protected_evidence_hashes,
    reproduce_legacy_canonical,
    source_inventory_document,
)
from runtime_assurance.recovery_branch_boundary_registry import LEGACY_FIXED_PREFIX
from runtime_assurance.recovery_branch_state_registry import (
    LEGACY_CASE_ID,
    LEGACY_MEMBER_ID,
    RegisteredBranchState,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    load_branch_state_registry,
    load_registered_branch_state,
    registry_aggregate_hash,
    validate_generated_branch_state_document,
)
from runtime_assurance.staged_recovery_logger_adapter import runtime_state_hash
from simulator.phase34_35_transition import (
    CartesianState2D,
    NormalizedAction2D,
    Phase3435DynamicsContext,
    Phase3435TransitionResult,
    step_phase34_35_transition,
)


DISCOVERY_ID = "stage2a_prediction_boundary_discovery_v0"
DISCOVERY_SCHEMA_VERSION = "stage2a_prediction_boundary_discovery_v0"
COMPLETED_DATE = "2026-08-21"
PLAN_PATH = Path("configs/stage2a_prediction_boundary_discovery_v0.json")
OUTPUT_PATH = Path("analysis/stage2a_prediction_boundary_discovery_v0")
SOURCE_CASE_COUNT = 13
MAXIMUM_PHYSICAL_TRANSITIONS_PER_TRAJECTORY = 32
DISCOVERY_BRANCH_IDS = (
    "zero_action_reference_v0",
    "tangential_error_correction_v0",
    "velocity_opposed_thrust_v0",
)
PLANNED_TRAJECTORY_COUNT = SOURCE_CASE_COUNT * len(DISCOVERY_BRANCH_IDS)
RESULT_ARTIFACTS = (
    "candidate_boundaries.json",
    "coverage_summary.json",
    "discovery_manifest.json",
    "discovery_plan.json",
    "near_boundary_diagnostics.json",
    "protected_evidence_report.json",
    "summary.md",
    "trajectory_index.json",
)
DIAGNOSTIC_BINS = (
    "predicted_ratio_lt_1p80",
    "predicted_ratio_1p80_to_lt_1p85",
    "predicted_ratio_1p85_to_1p90_inclusive",
    "predicted_ratio_gt_1p90",
)
LEGACY_PREFIX_REPORT_PATH = Path(
    "analysis/recovery_branch_state_registry_v0/prefix_execution_report.json"
)
NEWER_PROTECTED_PATHS: dict[str, tuple[str, ...]] = {
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
    "stage2a_hazard_arrest_runner_v0": (
        "runtime_assurance/stage2a_hazard_arrest_runner.py",
        "scripts/qualify_stage2a_hazard_arrest_case_v0.py",
        "scripts/run_stage2a_hazard_arrest_experiment_v0.py",
        "scripts/check_stage2a_hazard_arrest_experiment_v0.py",
        "Tests/test_stage2a_hazard_arrest_runner.py",
        "docs/architecture/stage2a_hazard_arrest_runner_v0.md",
    ),
    "stage2a_hazard_arrest_qualification_v0": (
        "analysis/stage2a_hazard_arrest_qualification_v0",
    ),
}


class PredictionBoundaryDiscoveryError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PredictionBoundaryDiscoveryError(f"{name} must be an object")
    return dict(value)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionBoundaryDiscoveryError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise PredictionBoundaryDiscoveryError(f"{name} must be finite")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PredictionBoundaryDiscoveryError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def state_document(state: CartesianState2D) -> dict[str, float]:
    return {
        "position_x": state.x,
        "position_y": state.y,
        "velocity_x": state.vx,
        "velocity_y": state.vy,
    }


def state_from_branch_document(document: Mapping[str, object]) -> CartesianState2D:
    source = document
    if not all(
        field in document
        for field in ("position_x", "position_y", "velocity_x", "velocity_y")
    ):
        source = _mapping(document.get("state"), "state")
    state = CartesianState2D(
        x=_finite(source.get("position_x"), "position_x"),
        y=_finite(source.get("position_y"), "position_y"),
        vx=_finite(source.get("velocity_x"), "velocity_x"),
        vy=_finite(source.get("velocity_y"), "velocity_y"),
    )
    return state


def _dynamics_from_branch_document(
    document: Mapping[str, object],
) -> tuple[Phase3435DynamicsContext, float, float]:
    simulator = _mapping(document.get("simulator_configuration"), "simulator_configuration")
    constants = _mapping(simulator.get("simulator_constants"), "simulator_constants")
    dynamics = Phase3435DynamicsContext(
        mu=_finite(constants.get("mu"), "mu"),
        dt=_finite(constants.get("dt"), "dt"),
        mass=_finite(constants.get("mass"), "mass"),
        thrust_scale=_finite(simulator.get("thrust_scale"), "thrust_scale"),
    )
    target_speed = _finite(
        constants.get("target_circular_speed"), "target_circular_speed"
    )
    epsilon = _finite(
        constants.get("speed_ratio_denominator_epsilon"),
        "speed_ratio_denominator_epsilon",
    )
    if dynamics.dt <= 0.0 or dynamics.mass <= 0.0 or target_speed <= 0.0:
        raise PredictionBoundaryDiscoveryError("invalid simulator configuration")
    if epsilon < 0.0:
        raise PredictionBoundaryDiscoveryError("invalid speed-ratio epsilon")
    return dynamics, target_speed, epsilon


def _branch_action(
    branch_id: str,
    state: CartesianState2D,
    target_speed: float,
) -> tuple[float, float]:
    if branch_id == "zero_action_reference_v0":
        return generate_zero_action()
    if branch_id == "tangential_error_correction_v0":
        return generate_tangential_correction_action(state, target_speed)
    if branch_id == "velocity_opposed_thrust_v0":
        return generate_velocity_opposed_action(state)
    raise PredictionBoundaryDiscoveryError(
        f"unsupported discovery branch: {branch_id!r}"
    )


def is_prediction_boundary_candidate(
    realized_speed_ratio: float,
    predicted_speed_ratio: float,
) -> bool:
    realized = _finite(realized_speed_ratio, "realized_speed_ratio")
    predicted = _finite(predicted_speed_ratio, "predicted_speed_ratio")
    return realized <= OVERSPEED_THRESHOLD and predicted > OVERSPEED_THRESHOLD


@dataclass(frozen=True, slots=True)
class NormalActionEvaluation:
    branch_id: str
    current_state: CartesianState2D
    current_state_hash: str
    realized_speed_ratio: float
    realized_headroom: float
    action: tuple[float, float]
    action_hash: str
    predicted_transition: Phase3435TransitionResult
    predicted_state_hash: str
    predicted_speed_ratio: float
    predicted_headroom: float
    final_veto_decision: FinalVetoDecision
    fallback_prediction_count: int
    candidate_boundary: bool
    physical_transition_count: int = 0
    active_authority_granted: bool = False
    hazard_arrest_interventions: int = 0

    def as_document(self) -> dict[str, object]:
        return {
            "branch_id": self.branch_id,
            "current_state": state_document(self.current_state),
            "current_state_hash": self.current_state_hash,
            "realized_speed_ratio": self.realized_speed_ratio,
            "realized_headroom": self.realized_headroom,
            "normal_action": list(self.action),
            "normal_action_hash": self.action_hash,
            "predicted_state": state_document(self.predicted_transition.next_state),
            "predicted_state_hash": self.predicted_state_hash,
            "predicted_speed_ratio": self.predicted_speed_ratio,
            "predicted_headroom": self.predicted_headroom,
            "final_veto_decision": self.final_veto_decision.decision,
            "final_veto_reason": self.final_veto_decision.reason,
            "final_veto_applied": self.final_veto_decision.veto_applied,
            "fallback_prediction_count": self.fallback_prediction_count,
            "fallback_execution_count": 0,
            "candidate_boundary": self.candidate_boundary,
            "physical_transition_count": self.physical_transition_count,
            "active_authority_granted": self.active_authority_granted,
            "hazard_arrest_interventions": self.hazard_arrest_interventions,
        }


@dataclass(frozen=True, slots=True)
class DiscoverySource:
    case: SourceCaseDefinition
    provenance_kind: str
    registry_member_id: str | None
    document_json: str
    boundary_type: str
    boundary_transition_count: int
    actual_transition_count: int
    branch_step: int
    initial_state_hash: str
    prefix_action_trace_hash: str
    prefix_state_trace_hash: str
    canonical_source_hash: str
    source_configuration_hash: str
    source_equivalence_checks: tuple[tuple[str, bool], ...]

    def document(self) -> dict[str, object]:
        value = json.loads(self.document_json)
        return _mapping(value, "discovery source document")


def generated_discovery_source(prefix: PrefixExecutionResult) -> DiscoverySource:
    if prefix.boundary_type == LEGACY_FIXED_PREFIX:
        raise PredictionBoundaryDiscoveryError(
            "legacy fixed prefix cannot use generated discovery-source validation"
        )
    document = prefix.document()
    validate_generated_branch_state_document(document)
    return DiscoverySource(
        case=prefix.case,
        provenance_kind="generated_registry_member",
        registry_member_id=None,
        document_json=canonical_json_bytes(document).decode("utf-8"),
        boundary_type=prefix.boundary_type,
        boundary_transition_count=prefix.boundary_transition_count,
        actual_transition_count=prefix.actual_transition_count,
        branch_step=prefix.branch_step,
        initial_state_hash=prefix.initial_state_hash,
        prefix_action_trace_hash=prefix.prefix_action_trace_hash,
        prefix_state_trace_hash=prefix.prefix_state_trace_hash,
        canonical_source_hash=prefix.canonical_payload_hash,
        source_configuration_hash=prefix.case.source_configuration_hash,
        source_equivalence_checks=(("generated_member_validation", True),),
    )


def _load_frozen_legacy_prefix_evidence(
    repository_root: Path,
) -> dict[str, object]:
    path = repository_root.resolve() / LEGACY_PREFIX_REPORT_PATH
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictionBoundaryDiscoveryError(
            "frozen legacy prefix evidence is unavailable"
        ) from exc
    report = _mapping(document, "legacy prefix report")
    supplied = report.get("canonical_payload_hash")
    payload = dict(report)
    payload.pop("canonical_payload_hash", None)
    if supplied != canonical_sha256(payload):
        raise PredictionBoundaryDiscoveryError(
            "frozen legacy prefix report hash mismatch"
        )
    executions = report.get("executions")
    if not isinstance(executions, list):
        raise PredictionBoundaryDiscoveryError("legacy prefix executions are missing")
    matches = [
        _mapping(item, "legacy prefix execution")
        for item in executions
        if isinstance(item, Mapping)
        and item.get("execution_role") == "canonical_reproduction"
        and item.get("case_id") == LEGACY_CASE_ID
    ]
    if len(matches) != 1:
        raise PredictionBoundaryDiscoveryError(
            "frozen canonical reproduction evidence is not unique"
        )
    return matches[0]


def prepare_legacy_discovery_source(
    repository_root: Path,
    case: SourceCaseDefinition,
    *,
    registered_loader: Callable[[Path, str], RegisteredBranchState] = (
        load_registered_branch_state
    ),
    legacy_reproducer: Callable[[Path], LegacyReproductionResult] = (
        reproduce_legacy_canonical
    ),
    frozen_prefix_evidence_loader: Callable[[Path], Mapping[str, object]] = (
        _load_frozen_legacy_prefix_evidence
    ),
) -> DiscoverySource:
    if case.case_id != LEGACY_CASE_ID or case.boundary.boundary_type != LEGACY_FIXED_PREFIX:
        raise PredictionBoundaryDiscoveryError(
            "legacy discovery source requires the frozen canonical case"
        )
    registered = registered_loader(repository_root, LEGACY_MEMBER_ID)
    if (
        registered.registry_member_id != LEGACY_MEMBER_ID
        or registered.case_id != LEGACY_CASE_ID
        or not registered.member.legacy_member
        or registered.member.artifact_scope != "legacy_external_artifact"
        or registered.member.generation_status != "legacy_validated"
    ):
        raise PredictionBoundaryDiscoveryError(
            "registered legacy discovery source identity mismatch"
        )
    published = registered.as_document()
    reproduction = legacy_reproducer(repository_root)
    reproduced = reproduction.document()
    prefix_evidence = _mapping(
        frozen_prefix_evidence_loader(repository_root), "legacy prefix evidence"
    )
    published_ordering = _mapping(
        published.get("branch_ordering"), "legacy branch ordering"
    )
    reproduced_ordering = _mapping(
        reproduced.get("branch_ordering"), "reproduced branch ordering"
    )
    published_state = _mapping(published.get("state"), "legacy state")
    reproduced_state = _mapping(reproduced.get("state"), "reproduced state")
    published_monitor = _mapping(
        published.get("monitor_decision"), "legacy monitor decision"
    )
    reproduced_monitor = _mapping(
        reproduced.get("monitor_decision"), "reproduced monitor decision"
    )
    checks = {
        "registered_member_id": registered.registry_member_id == LEGACY_MEMBER_ID,
        "case_id": (
            published.get("case_id")
            == reproduced.get("case_id")
            == case.case_id
            == LEGACY_CASE_ID
        ),
        "branch_step": (
            published.get("branch_step")
            == reproduced.get("branch_step")
            == reproduction.branch_step
            == registered.member.branch_step
            == case.boundary.branch_step
        ),
        "realized_prefix_transition_count": (
            published_ordering.get("realized_prefix_transition_count")
            == reproduced_ordering.get("realized_prefix_transition_count")
            == reproduction.actual_transition_count
            == registered.member.nominal_prefix_transition_count
            == case.nominal_prefix_transition_count
        ),
        "Cartesian_boundary_state": published_state == reproduced_state,
        "source_configuration_identity": (
            published.get("case_configuration_hash")
            == reproduced.get("case_configuration_hash")
            == registered.member.source_configuration_hash
        ),
        "simulator_configuration_identity": (
            published.get("simulator_configuration_hash")
            == reproduced.get("simulator_configuration_hash")
            == registered.member.simulator_configuration_hash
        ),
        "simulator_constants_identity": (
            published.get("simulator_constants_hash")
            == reproduced.get("simulator_constants_hash")
            == registered.member.constants_hash
        ),
        "overspeed_threshold": (
            published.get("threshold")
            == reproduced.get("threshold")
            == OVERSPEED_THRESHOLD
        ),
        "overspeed_comparator": (
            published.get("comparator")
            == reproduced.get("comparator")
            == OVERSPEED_COMPARATOR
        ),
        "nominal_action_identity": (
            published.get("nominal_action") == reproduced.get("nominal_action")
        ),
        "predicted_state_identity": (
            published.get("predicted_next_state")
            == reproduced.get("predicted_next_state")
        ),
        "predicted_speed_ratio": (
            published.get("predicted_speed_ratio")
            == reproduced.get("predicted_speed_ratio")
        ),
        "monitor_decision_identity": published_monitor == reproduced_monitor,
        "canonical_legacy_document": published == reproduced,
        "canonical_legacy_hash": (
            published.get("canonical_branch_state_hash")
            == reproduced.get("canonical_branch_state_hash")
            == registered.member.canonical_branch_state_hash
        ),
        "initial_state_hash": (
            reproduction.initial_state_hash
            == prefix_evidence.get("initial_state_hash")
        ),
        "prefix_action_trace_hash": (
            reproduction.prefix_action_trace_hash
            == prefix_evidence.get("prefix_action_trace_hash")
        ),
        "prefix_state_trace_hash": (
            reproduction.prefix_state_trace_hash
            == prefix_evidence.get("prefix_state_trace_hash")
        ),
        "frozen_prefix_actual_transition_count": (
            reproduction.actual_transition_count
            == prefix_evidence.get("actual_transition_count")
        ),
        "frozen_prefix_branch_step": (
            reproduction.branch_step == prefix_evidence.get("branch_step")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise PredictionBoundaryDiscoveryError(
            "legacy discovery source reproduction mismatch: " + ", ".join(failed)
        )
    return DiscoverySource(
        case=case,
        provenance_kind="legacy_canonical_registered_reproduction",
        registry_member_id=LEGACY_MEMBER_ID,
        document_json=canonical_json_bytes(published).decode("utf-8"),
        boundary_type=LEGACY_FIXED_PREFIX,
        boundary_transition_count=case.nominal_prefix_transition_count,
        actual_transition_count=reproduction.actual_transition_count,
        branch_step=reproduction.branch_step,
        initial_state_hash=reproduction.initial_state_hash,
        prefix_action_trace_hash=reproduction.prefix_action_trace_hash,
        prefix_state_trace_hash=reproduction.prefix_state_trace_hash,
        canonical_source_hash=str(published["canonical_branch_state_hash"]),
        source_configuration_hash=registered.member.source_configuration_hash,
        source_equivalence_checks=tuple(sorted(checks.items())),
    )


def evaluate_normal_action(
    branch_document: Mapping[str, object],
    state: CartesianState2D,
    branch_id: str,
) -> NormalActionEvaluation:
    dynamics, target_speed, epsilon = _dynamics_from_branch_document(branch_document)
    action = _branch_action(branch_id, state, target_speed)
    predicted: list[Phase3435TransitionResult] = []

    def predictor(
        current: CartesianState2D,
        proposed_action: tuple[float, float],
    ) -> OneStepPrediction[CartesianState2D]:
        transition = step_phase34_35_transition(
            current,
            NormalizedAction2D(*proposed_action),
            dynamics,
        )
        predicted.append(transition)
        return OneStepPrediction(
            next_state=transition.next_state,
            speed_ratio=(
                math.hypot(transition.next_state.vx, transition.next_state.vy)
                / (target_speed + epsilon)
            ),
        )

    decision = evaluate_overspeed_veto(
        state,
        action,
        predictor,
        threshold=OVERSPEED_THRESHOLD,
    )
    if not predicted:
        raise PredictionBoundaryDiscoveryError("normal prediction was not captured")
    nominal = predicted[0]
    realized_ratio = math.hypot(state.vx, state.vy) / (target_speed + epsilon)
    predicted_ratio = decision.predicted_nominal_speed_ratio
    candidate = is_prediction_boundary_candidate(realized_ratio, predicted_ratio)
    if candidate and decision.decision != "veto":
        raise PredictionBoundaryDiscoveryError(
            "candidate prediction did not receive the unchanged Final Veto"
        )
    return NormalActionEvaluation(
        branch_id=branch_id,
        current_state=state,
        current_state_hash=runtime_state_hash(state),
        realized_speed_ratio=realized_ratio,
        realized_headroom=OVERSPEED_THRESHOLD - realized_ratio,
        action=action,
        action_hash=canonical_sha256({"action": list(action)}),
        predicted_transition=nominal,
        predicted_state_hash=runtime_state_hash(nominal.next_state),
        predicted_speed_ratio=predicted_ratio,
        predicted_headroom=OVERSPEED_THRESHOLD - predicted_ratio,
        final_veto_decision=decision,
        fallback_prediction_count=max(0, len(predicted) - 1),
        candidate_boundary=candidate,
    )


def execute_allowed_normal_transition(
    branch_document: Mapping[str, object],
    evaluation: NormalActionEvaluation,
) -> Phase3435TransitionResult:
    if evaluation.candidate_boundary or evaluation.final_veto_decision.decision != "allow":
        raise PredictionBoundaryDiscoveryError(
            "vetoed discovery proposal cannot execute a physical transition"
        )
    if evaluation.final_veto_decision.executed_action != evaluation.action:
        raise PredictionBoundaryDiscoveryError("Final Veto changed an allowed action")
    dynamics, _, _ = _dynamics_from_branch_document(branch_document)
    realized = step_phase34_35_transition(
        evaluation.current_state,
        NormalizedAction2D(*evaluation.action),
        dynamics,
    )
    if realized != evaluation.predicted_transition:
        raise PredictionBoundaryDiscoveryError(
            "normal prediction diverged from the executed transition"
        )
    return realized


EvaluationFunction = Callable[
    [Mapping[str, object], CartesianState2D, str], NormalActionEvaluation
]
ExecutionFunction = Callable[
    [Mapping[str, object], NormalActionEvaluation], Phase3435TransitionResult
]


@dataclass(frozen=True, slots=True)
class DiscoveryTrajectoryResult:
    trajectory_id: str
    case_id: str
    case_family: str
    branch_id: str
    source_provenance_kind: str
    source_registry_member_id: str | None
    source_canonical_hash: str
    source_equivalence_checks: tuple[tuple[str, bool], ...]
    source_configuration_hash: str
    simulator_configuration_hash: str
    constants_hash: str
    transition_implementation_hash: str
    nominal_controller_hash: str
    boundary_type: str
    prefix_transition_count: int
    prefix_actual_transition_count: int
    prefix_action_trace_hash: str
    prefix_state_trace_hash: str
    source_boundary_state_hash: str
    records: tuple[dict[str, object], ...]
    action_trace_hash: str
    state_trace_hash: str
    source_trajectory_hash: str
    states_evaluated: int
    physical_transition_count: int
    final_veto_rejection_count: int
    candidate_boundary_count: int
    terminal_reason: str


def _case_family(case_id: str) -> str:
    marker = "__r0_"
    return case_id.split(marker, 1)[0]


def trajectory_id(case_id: str, branch_id: str) -> str:
    return f"discovery__{case_id}__{branch_id}"


def build_trajectory_definitions(
    cases: Sequence[SourceCaseDefinition],
) -> tuple[tuple[SourceCaseDefinition, str], ...]:
    ordered_cases = sorted(cases, key=lambda item: item.case_id)
    if len(ordered_cases) != SOURCE_CASE_COUNT:
        raise PredictionBoundaryDiscoveryError("source-case count is not frozen at 13")
    if any(not case.eligible_for_generation for case in ordered_cases):
        raise PredictionBoundaryDiscoveryError("discovery source case is ineligible")
    return tuple(
        (case, branch_id)
        for case in ordered_cases
        for branch_id in DISCOVERY_BRANCH_IDS
    )


def _record(
    evaluation: NormalActionEvaluation,
    *,
    event_index: int,
    physical_transition_count_before: int,
    transition_executed: bool,
    realized_next_state: CartesianState2D | None,
    stop_reason: str | None,
) -> dict[str, object]:
    document = evaluation.as_document()
    document.update(
        {
            "event_index": event_index,
            "physical_transition_count_before": physical_transition_count_before,
            "physical_transition_count_after": physical_transition_count_before
            + int(transition_executed),
            "transition_executed": transition_executed,
            "realized_next_state": (
                None if realized_next_state is None else state_document(realized_next_state)
            ),
            "realized_next_state_hash": (
                None if realized_next_state is None else runtime_state_hash(realized_next_state)
            ),
            "stop_reason": stop_reason,
        }
    )
    document["canonical_observation_hash"] = canonical_sha256(document)
    return document


def run_discovery_trajectory(
    source: DiscoverySource,
    branch_id: str,
    *,
    maximum_physical_transitions: int = MAXIMUM_PHYSICAL_TRANSITIONS_PER_TRAJECTORY,
    evaluator: EvaluationFunction = evaluate_normal_action,
    transition_executor: ExecutionFunction = execute_allowed_normal_transition,
) -> DiscoveryTrajectoryResult:
    if branch_id not in DISCOVERY_BRANCH_IDS:
        raise PredictionBoundaryDiscoveryError("trajectory uses an undeclared branch")
    limit = _integer(
        maximum_physical_transitions,
        "maximum_physical_transitions",
        minimum=1,
    )
    if limit != MAXIMUM_PHYSICAL_TRANSITIONS_PER_TRAJECTORY:
        raise PredictionBoundaryDiscoveryError("discovery horizon must remain exactly 32")
    document = source.document()
    state = state_from_branch_document(document)
    records: list[dict[str, object]] = []
    executed_actions: list[list[float]] = []
    realized_states: list[dict[str, float]] = [state_document(state)]
    physical_count = 0
    veto_count = 0
    candidate_count = 0
    terminal_reason = "discovery_transition_horizon_complete"

    while physical_count < limit:
        evaluation = evaluator(document, state, branch_id)
        event_index = len(records)
        if evaluation.candidate_boundary:
            veto_count += 1
            candidate_count += 1
            terminal_reason = "candidate_boundary_detected"
            records.append(
                _record(
                    evaluation,
                    event_index=event_index,
                    physical_transition_count_before=physical_count,
                    transition_executed=False,
                    realized_next_state=None,
                    stop_reason=terminal_reason,
                )
            )
            break
        if evaluation.realized_speed_ratio > OVERSPEED_THRESHOLD:
            terminal_reason = "realized_overspeed_adverse_terminal"
            if evaluation.final_veto_decision.decision == "veto":
                veto_count += 1
            records.append(
                _record(
                    evaluation,
                    event_index=event_index,
                    physical_transition_count_before=physical_count,
                    transition_executed=False,
                    realized_next_state=None,
                    stop_reason=terminal_reason,
                )
            )
            break
        if evaluation.final_veto_decision.decision == "veto":
            veto_count += 1
            terminal_reason = "normal_action_vetoed_without_candidate"
            records.append(
                _record(
                    evaluation,
                    event_index=event_index,
                    physical_transition_count_before=physical_count,
                    transition_executed=False,
                    realized_next_state=None,
                    stop_reason=terminal_reason,
                )
            )
            break
        realized = transition_executor(document, evaluation)
        records.append(
            _record(
                evaluation,
                event_index=event_index,
                physical_transition_count_before=physical_count,
                transition_executed=True,
                realized_next_state=realized.next_state,
                stop_reason=(
                    "discovery_transition_horizon_complete"
                    if physical_count + 1 == limit
                    else None
                ),
            )
        )
        executed_actions.append(list(evaluation.action))
        state = realized.next_state
        realized_states.append(state_document(state))
        physical_count += 1

    record_tuple = tuple(records)
    action_trace_hash = canonical_sha256(executed_actions)
    state_trace_hash = canonical_sha256(realized_states)
    identity = {
        "trajectory_id": trajectory_id(source.case.case_id, branch_id),
        "case_id": source.case.case_id,
        "branch_id": branch_id,
        "source_provenance_kind": source.provenance_kind,
        "source_registry_member_id": source.registry_member_id,
        "source_canonical_hash": source.canonical_source_hash,
        "source_equivalence_checks": [
            list(item) for item in source.source_equivalence_checks
        ],
        "source_configuration_hash": source.source_configuration_hash,
        "boundary_type": source.boundary_type,
        "prefix_transition_count": source.boundary_transition_count,
        "prefix_actual_transition_count": source.actual_transition_count,
        "prefix_action_trace_hash": source.prefix_action_trace_hash,
        "prefix_state_trace_hash": source.prefix_state_trace_hash,
        "source_boundary_state_hash": runtime_state_hash(
            state_from_branch_document(document)
        ),
        "action_trace_hash": action_trace_hash,
        "state_trace_hash": state_trace_hash,
        "records": record_tuple,
    }
    source_trajectory_hash = canonical_sha256(identity)
    return DiscoveryTrajectoryResult(
        trajectory_id=str(identity["trajectory_id"]),
        case_id=source.case.case_id,
        case_family=_case_family(source.case.case_id),
        branch_id=branch_id,
        source_provenance_kind=source.provenance_kind,
        source_registry_member_id=source.registry_member_id,
        source_canonical_hash=source.canonical_source_hash,
        source_equivalence_checks=source.source_equivalence_checks,
        source_configuration_hash=source.source_configuration_hash,
        simulator_configuration_hash=str(document["simulator_configuration_hash"]),
        constants_hash=str(
            document.get("constants_hash", document.get("simulator_constants_hash"))
        ),
        transition_implementation_hash=source.case.transition_implementation_hash,
        nominal_controller_hash=source.case.nominal_controller_hash,
        boundary_type=source.boundary_type,
        prefix_transition_count=source.boundary_transition_count,
        prefix_actual_transition_count=source.actual_transition_count,
        prefix_action_trace_hash=source.prefix_action_trace_hash,
        prefix_state_trace_hash=source.prefix_state_trace_hash,
        source_boundary_state_hash=str(identity["source_boundary_state_hash"]),
        records=record_tuple,
        action_trace_hash=action_trace_hash,
        state_trace_hash=state_trace_hash,
        source_trajectory_hash=source_trajectory_hash,
        states_evaluated=len(record_tuple),
        physical_transition_count=physical_count,
        final_veto_rejection_count=veto_count,
        candidate_boundary_count=candidate_count,
        terminal_reason=terminal_reason,
    )


def prepare_discovery_sources(
    repository_root: Path,
    cases: Sequence[SourceCaseDefinition],
    *,
    implementation_commit: str,
    prefix_executor: Callable[..., PrefixExecutionResult] = execute_nominal_prefix,
    legacy_source_preparer: Callable[..., DiscoverySource] = (
        prepare_legacy_discovery_source
    ),
) -> tuple[DiscoverySource, ...]:
    sources: list[DiscoverySource] = []
    for index, case in enumerate(sorted(cases, key=lambda item: item.case_id), start=1):
        if case.boundary.boundary_type == LEGACY_FIXED_PREFIX:
            source = legacy_source_preparer(repository_root, case)
        else:
            prefix = prefix_executor(
                repository_root,
                case,
                execution_role="candidate_discovery",
                execution_id=f"discovery_prefix_{index:02d}",
                implementation_commit=implementation_commit,
            )
            source = generated_discovery_source(prefix)
        sources.append(source)
    return tuple(sources)


def execute_frozen_discovery(
    repository_root: Path,
    *,
    implementation_commit: str,
    prefix_executor: Callable[..., PrefixExecutionResult] = execute_nominal_prefix,
    legacy_source_preparer: Callable[..., DiscoverySource] = (
        prepare_legacy_discovery_source
    ),
) -> tuple[tuple[DiscoverySource, ...], tuple[DiscoveryTrajectoryResult, ...]]:
    cases = build_source_case_inventory(repository_root)
    definitions = build_trajectory_definitions(cases)
    sources = prepare_discovery_sources(
        repository_root,
        cases,
        implementation_commit=implementation_commit,
        prefix_executor=prefix_executor,
        legacy_source_preparer=legacy_source_preparer,
    )
    by_case = {source.case.case_id: source for source in sources}
    trajectories = tuple(
        run_discovery_trajectory(by_case[case.case_id], branch_id)
        for case, branch_id in definitions
    )
    return sources, trajectories


def plan_scientific_payload(document: Mapping[str, object]) -> dict[str, object]:
    payload = dict(document)
    payload.pop("canonical_plan_hash", None)
    return payload


def load_discovery_plan(repository_root: Path) -> dict[str, object]:
    path = repository_root.resolve() / PLAN_PATH
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictionBoundaryDiscoveryError("discovery plan is unavailable") from exc
    plan = _mapping(value, "discovery plan")
    validate_discovery_plan(repository_root, plan)
    return plan


def validate_discovery_plan(
    repository_root: Path,
    plan: Mapping[str, object],
) -> None:
    if plan.get("discovery_id") != DISCOVERY_ID:
        raise PredictionBoundaryDiscoveryError("discovery plan identity mismatch")
    if plan.get("schema_version") != DISCOVERY_SCHEMA_VERSION:
        raise PredictionBoundaryDiscoveryError("discovery plan schema mismatch")
    supplied = plan.get("canonical_plan_hash")
    if supplied != canonical_sha256(plan_scientific_payload(plan)):
        raise PredictionBoundaryDiscoveryError("discovery plan hash mismatch")
    if plan.get("overspeed_threshold") != OVERSPEED_THRESHOLD:
        raise PredictionBoundaryDiscoveryError("overspeed threshold changed")
    if plan.get("overspeed_comparator") != OVERSPEED_COMPARATOR:
        raise PredictionBoundaryDiscoveryError("overspeed comparator changed")
    if tuple(plan.get("recovery_branches", ())) != DISCOVERY_BRANCH_IDS:
        raise PredictionBoundaryDiscoveryError("discovery branch matrix changed")
    if plan.get("maximum_physical_transitions_per_trajectory") != (
        MAXIMUM_PHYSICAL_TRANSITIONS_PER_TRAJECTORY
    ):
        raise PredictionBoundaryDiscoveryError("discovery horizon changed")
    if plan.get("total_planned_trajectory_count") != PLANNED_TRAJECTORY_COUNT:
        raise PredictionBoundaryDiscoveryError("planned trajectory count changed")
    if plan.get("candidate_boundary_definition") != {
        "current_realized_speed_ratio": "<=1.90",
        "normal_predicted_speed_ratio": ">1.90",
        "ordering": "before_normal_action_execution",
        "first_candidate_terminates_trajectory": True,
    }:
        raise PredictionBoundaryDiscoveryError("candidate definition changed")
    if (
        plan.get("active_authority_granted") is not False
        or plan.get("hazard_arrest_interventions") != 0
        or plan.get("result_publication_path") != OUTPUT_PATH.as_posix()
    ):
        raise PredictionBoundaryDiscoveryError("discovery authority boundary changed")

    inventory = build_source_case_inventory(repository_root)
    expected_ids = [item.case_id for item in sorted(inventory, key=lambda item: item.case_id)]
    if plan.get("source_case_ids") != expected_ids:
        raise PredictionBoundaryDiscoveryError("source-case inventory changed")
    if len(expected_ids) != SOURCE_CASE_COUNT or any(
        not item.eligible_for_generation for item in inventory
    ):
        raise PredictionBoundaryDiscoveryError("frozen source inventory is incomplete")
    source_inventory = source_inventory_document(repository_root)
    registry = load_branch_state_registry(repository_root)
    simulator_8000 = _simulator_configuration_for_thrust(8000.0)
    simulator_10000 = _simulator_configuration_for_thrust(10000.0)
    simulator_constants = _mapping(
        simulator_8000.get("simulator_constants"), "simulator_constants"
    )
    if simulator_constants != _mapping(
        simulator_10000.get("simulator_constants"), "simulator_constants"
    ):
        raise PredictionBoundaryDiscoveryError("simulator constants differ by thrust")
    expected_hashes = _mapping(plan.get("source_hashes"), "source_hashes")
    current_hashes = {
        "final_veto_manifest_raw_hash": source_inventory["source_artifact_hash"],
        "boundary_registry_raw_hash": source_inventory["boundary_registry_hash"],
        "source_inventory_canonical_hash": source_inventory["canonical_payload_hash"],
        "registry_manifest_canonical_hash": registry.canonical_manifest_hash,
        "registry_aggregate_hash": registry_aggregate_hash(registry.members),
        "simulator_constants_hash": canonical_sha256(simulator_constants),
        "simulator_configuration_hash_thrust_8000": canonical_sha256(simulator_8000),
        "simulator_configuration_hash_thrust_10000": canonical_sha256(simulator_10000),
    }
    if expected_hashes != current_hashes:
        raise PredictionBoundaryDiscoveryError("frozen source hashes changed")


def _trajectory_index_entry(result: DiscoveryTrajectoryResult) -> dict[str, object]:
    return {
        "trajectory_id": result.trajectory_id,
        "case_id": result.case_id,
        "case_family": result.case_family,
        "branch_id": result.branch_id,
        "source_provenance_kind": result.source_provenance_kind,
        "source_registry_member_id": result.source_registry_member_id,
        "source_canonical_hash": result.source_canonical_hash,
        "source_equivalence_checks": [list(item) for item in result.source_equivalence_checks],
        "source_configuration_hash": result.source_configuration_hash,
        "simulator_configuration_hash": result.simulator_configuration_hash,
        "constants_hash": result.constants_hash,
        "transition_implementation_hash": result.transition_implementation_hash,
        "nominal_controller_hash": result.nominal_controller_hash,
        "boundary_type": result.boundary_type,
        "prefix_transition_count": result.prefix_transition_count,
        "prefix_actual_transition_count": result.prefix_actual_transition_count,
        "prefix_action_trace_hash": result.prefix_action_trace_hash,
        "prefix_state_trace_hash": result.prefix_state_trace_hash,
        "source_boundary_state_hash": result.source_boundary_state_hash,
        "action_trace_hash": result.action_trace_hash,
        "state_trace_hash": result.state_trace_hash,
        "source_trajectory_hash": result.source_trajectory_hash,
        "records": list(result.records),
        "states_evaluated": result.states_evaluated,
        "physical_transition_count": result.physical_transition_count,
        "final_veto_rejection_count": result.final_veto_rejection_count,
        "candidate_boundary_count": result.candidate_boundary_count,
        "terminal_reason": result.terminal_reason,
    }


def _candidate_document(
    result: DiscoveryTrajectoryResult,
    record: Mapping[str, object],
    *,
    implementation_commit: str,
    plan_hash: str,
) -> dict[str, object]:
    identity = {
        "source_trajectory_hash": result.source_trajectory_hash,
        "boundary_event_index": record["event_index"],
        "current_state_hash": record["current_state_hash"],
        "normal_action_hash": record["normal_action_hash"],
        "predicted_state_hash": record["predicted_state_hash"],
    }
    candidate_id = f"candidate__{canonical_sha256(identity)[:24]}"
    candidate = {
        "candidate_id": candidate_id,
        "source_case": result.case_id,
        "case_family": result.case_family,
        "parameter_configuration": {
            "source_configuration_hash": result.source_configuration_hash,
            "simulator_configuration_hash": result.simulator_configuration_hash,
            "constants_hash": result.constants_hash,
        },
        "branch_id": result.branch_id,
        "prefix_transition_count": result.prefix_transition_count,
        "prefix_actual_transition_count": result.prefix_actual_transition_count,
        "boundary_event_index": record["event_index"],
        "current_state": record["current_state"],
        "current_state_hash": record["current_state_hash"],
        "realized_speed_ratio": record["realized_speed_ratio"],
        "normal_action": record["normal_action"],
        "action_hash": record["normal_action_hash"],
        "predicted_state": record["predicted_state"],
        "predicted_state_hash": record["predicted_state_hash"],
        "predicted_speed_ratio": record["predicted_speed_ratio"],
        "predicted_overspeed_headroom": record["predicted_headroom"],
        "Final_Veto_decision": record["final_veto_decision"],
        "Final_Veto_reason": record["final_veto_reason"],
        "candidate_action_physically_executed": False,
        "fallback_execution_count": 0,
        "active_authority_granted": False,
        "hazard_arrest_interventions": 0,
        "source_trajectory_hash": result.source_trajectory_hash,
        "source_observation_hash": record["canonical_observation_hash"],
        "prefix_action_trace_hash": result.prefix_action_trace_hash,
        "prefix_state_trace_hash": result.prefix_state_trace_hash,
        "implementation_commit": implementation_commit,
        "discovery_plan_hash": plan_hash,
    }
    candidate["canonical_candidate_hash"] = canonical_sha256(candidate)
    return candidate


def _diagnostic_bin(value: float) -> str:
    if value < 1.80:
        return DIAGNOSTIC_BINS[0]
    if value < 1.85:
        return DIAGNOSTIC_BINS[1]
    if value <= OVERSPEED_THRESHOLD:
        return DIAGNOSTIC_BINS[2]
    return DIAGNOSTIC_BINS[3]


def _trajectory_diagnostic(result: DiscoveryTrajectoryResult) -> dict[str, object]:
    records = result.records
    valid_current = [
        record
        for record in records
        if float(record["realized_speed_ratio"]) <= OVERSPEED_THRESHOLD
    ]
    closest = min(
        valid_current,
        key=lambda item: (
            abs(float(item["predicted_headroom"])),
            int(item["event_index"]),
        ),
    ) if valid_current else None
    return {
        "trajectory_id": result.trajectory_id,
        "case_id": result.case_id,
        "case_family": result.case_family,
        "branch_id": result.branch_id,
        "maximum_realized_speed_ratio_while_clear": (
            None
            if not valid_current
            else max(float(item["realized_speed_ratio"]) for item in valid_current)
        ),
        "maximum_normal_predicted_speed_ratio": (
            None
            if not records
            else max(float(item["predicted_speed_ratio"]) for item in records)
        ),
        "minimum_predicted_overspeed_headroom_while_current_clear": (
            None
            if not valid_current
            else min(float(item["predicted_headroom"]) for item in valid_current)
        ),
        "closest_approach_event_index": (
            None if closest is None else closest["event_index"]
        ),
        "closest_approach_state_hash": (
            None if closest is None else closest["current_state_hash"]
        ),
        "closest_approach_action": (
            None if closest is None else closest["normal_action"]
        ),
        "closest_approach_predicted_state_hash": (
            None if closest is None else closest["predicted_state_hash"]
        ),
        "closest_approach_predicted_speed_ratio": (
            None if closest is None else closest["predicted_speed_ratio"]
        ),
        "closest_approach_Final_Veto_decision": (
            None if closest is None else closest["final_veto_decision"]
        ),
        "candidate_boundary_count": result.candidate_boundary_count,
        "scientific_limitation": (
            "The 1.80 and 1.85 bins are diagnostics only and are not authority thresholds."
        ),
    }


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True).encode(
        "utf-8"
    ) + b"\n"


def build_discovery_payloads(
    repository_root: Path,
    sources: Sequence[DiscoverySource],
    trajectories: Sequence[DiscoveryTrajectoryResult],
    *,
    implementation_commit: str,
    protected_before: Mapping[str, str],
    protected_after: Mapping[str, str],
) -> dict[str, bytes]:
    if dict(protected_before) != dict(protected_after):
        raise PredictionBoundaryDiscoveryError("protected evidence changed")
    plan = load_discovery_plan(repository_root)
    plan_hash = str(plan["canonical_plan_hash"])
    expected_order = [
        trajectory_id(case.case_id, branch)
        for case, branch in build_trajectory_definitions(
            build_source_case_inventory(repository_root)
        )
    ]
    if [item.trajectory_id for item in trajectories] != expected_order:
        raise PredictionBoundaryDiscoveryError("trajectory ordering changed")
    if len(sources) != SOURCE_CASE_COUNT or len(trajectories) != PLANNED_TRAJECTORY_COUNT:
        raise PredictionBoundaryDiscoveryError("discovery execution count mismatch")

    index_entries = [_trajectory_index_entry(item) for item in trajectories]
    candidates = [
        _candidate_document(
            result,
            next(record for record in result.records if record["candidate_boundary"] is True),
            implementation_commit=implementation_commit,
            plan_hash=plan_hash,
        )
        for result in trajectories
        if result.candidate_boundary_count == 1
    ]
    diagnostics = [_trajectory_diagnostic(item) for item in trajectories]
    all_records = [record for item in trajectories for record in item.records]
    bin_counts = {key: 0 for key in DIAGNOSTIC_BINS}
    for record in all_records:
        bin_counts[_diagnostic_bin(float(record["predicted_speed_ratio"]))] += 1
    below = [
        float(record["predicted_speed_ratio"])
        for record in all_records
        if float(record["predicted_speed_ratio"]) <= OVERSPEED_THRESHOLD
    ]
    branch_transitions = {
        branch: sum(
            item.physical_transition_count
            for item in trajectories
            if item.branch_id == branch
        )
        for branch in DISCOVERY_BRANCH_IDS
    }
    candidate_by_branch = {
        branch: sum(item["branch_id"] == branch for item in candidates)
        for branch in DISCOVERY_BRANCH_IDS
    }
    families = sorted({_case_family(source.case.case_id) for source in sources})
    candidate_by_family = {
        family: sum(item["case_family"] == family for item in candidates)
        for family in families
    }
    prefix_physical_transitions = sum(item.actual_transition_count for item in sources)
    branch_physical_transitions = sum(item.physical_transition_count for item in trajectories)
    states_evaluated = len(all_records)
    veto_rejections = sum(item.final_veto_rejection_count for item in trajectories)
    coverage = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "source_case_count": len(sources),
        "planned_trajectory_count": PLANNED_TRAJECTORY_COUNT,
        "started_trajectory_count": len(trajectories),
        "completed_trajectory_count": len(trajectories),
        "states_evaluated": states_evaluated,
        "prefix_physical_transition_count": prefix_physical_transitions,
        "normal_branch_physical_transition_count": branch_physical_transitions,
        "total_physical_transition_count": (
            prefix_physical_transitions + branch_physical_transitions
        ),
        "normal_branch_execution_counts": branch_transitions,
        "Final_Veto_rejection_count": veto_rejections,
        "candidate_boundary_count": len(candidates),
        "candidate_counts_by_branch": candidate_by_branch,
        "candidate_counts_by_case_family": candidate_by_family,
        "maximum_predicted_speed_ratio": (
            None
            if not all_records
            else max(float(item["predicted_speed_ratio"]) for item in all_records)
        ),
        "closest_below_threshold_predicted_speed_ratio": (
            None if not below else max(below)
        ),
        "predicted_speed_ratio_diagnostic_bins": bin_counts,
        "diagnostic_bins_are_authority_thresholds": False,
        "active_authority_granted": False,
        "hazard_arrest_interventions": 0,
        "automatic_retry_count": 0,
    }
    trajectory_index = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "trajectory_ordering": "case_id_lexical_then_declared_branch_order",
        "trajectory_count": len(index_entries),
        "trajectories": index_entries,
    }
    candidate_document = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "candidate_definition": plan["candidate_boundary_definition"],
        "candidate_boundary_count": len(candidates),
        "candidate_boundaries": candidates,
    }
    diagnostic_document = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "trajectory_count": len(diagnostics),
        "diagnostic_bins": {
            "predicted_ratio_lt_1p80": "predicted_speed_ratio < 1.80",
            "predicted_ratio_1p80_to_lt_1p85": "1.80 <= predicted_speed_ratio < 1.85",
            "predicted_ratio_1p85_to_1p90_inclusive": "1.85 <= predicted_speed_ratio <= 1.90",
            "predicted_ratio_gt_1p90": "predicted_speed_ratio > 1.90",
        },
        "diagnostic_bins_are_new_thresholds": False,
        "trajectories": diagnostics,
    }
    protected_report = {
        "before": dict(sorted(protected_before.items())),
        "after": dict(sorted(protected_after.items())),
        "all_protected_evidence_unchanged": True,
    }
    summary_status = (
        "One or more natural prediction-boundary candidates were discovered."
        if candidates
        else "No natural prediction-boundary candidate was discovered in the frozen grid."
    )
    summary = (
        "# Stage 2A Prediction-Boundary Discovery v0\n\n"
        f"Completed: {COMPLETED_DATE}\n\n"
        "## Status\n\n"
        f"{summary_status}\n\n"
        "## Search\n\n"
        "The frozen search used all 13 provenance-complete Final Veto source cases, "
        "their case-specific deterministic branch boundaries, three existing recovery "
        "branches, and at most 32 branch transitions per trajectory. Cases do not share "
        "a synchronized physical time.\n\n"
        "## Authority\n\n"
        "Active authority granted: false. Hazard-arrest interventions: 0. A vetoed normal "
        "proposal executed no transition and no fallback. Velocity-opposed thrust was used "
        "only as an ordinary existing discovery branch.\n\n"
        "## Claim Restrictions\n\n"
        "This bounded discovery result does not demonstrate hazard-arrest effectiveness, "
        "recovery improvement, active-controller safety, stability, optimality, threshold "
        "validity beyond the frozen strict > 1.90 semantics, handoff readiness, multi-step "
        "recovery, hardware validity, or deployment readiness.\n"
    ).encode("utf-8")
    payloads: dict[str, bytes] = {
        "discovery_plan.json": (repository_root / PLAN_PATH).read_bytes(),
        "trajectory_index.json": _json_bytes(trajectory_index),
        "candidate_boundaries.json": _json_bytes(candidate_document),
        "near_boundary_diagnostics.json": _json_bytes(diagnostic_document),
        "coverage_summary.json": _json_bytes(coverage),
        "protected_evidence_report.json": _json_bytes(protected_report),
        "summary.md": summary,
    }
    artifact_hashes = {
        name: sha256_bytes(data) for name, data in sorted(payloads.items())
    }
    discovery_aggregate_hash = canonical_sha256(artifact_hashes)
    manifest = {
        "discovery_id": DISCOVERY_ID,
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "completed_date": COMPLETED_DATE,
        "discovery_implementation_commit": implementation_commit,
        "discovery_plan_hash": plan_hash,
        "source_case_count": len(sources),
        "planned_trajectory_count": PLANNED_TRAJECTORY_COUNT,
        "started_trajectory_count": len(trajectories),
        "completed_trajectory_count": len(trajectories),
        "maximum_physical_transitions_per_trajectory": (
            MAXIMUM_PHYSICAL_TRANSITIONS_PER_TRAJECTORY
        ),
        "prefix_physical_transition_count": prefix_physical_transitions,
        "normal_branch_physical_transition_count": branch_physical_transitions,
        "total_physical_transition_count": (
            prefix_physical_transitions + branch_physical_transitions
        ),
        "states_evaluated": states_evaluated,
        "Final_Veto_rejection_count": veto_rejections,
        "candidate_boundary_count": len(candidates),
        "active_authority_granted": False,
        "hazard_arrest_interventions": 0,
        "automatic_retry_count": 0,
        "normal_action_sources": list(DISCOVERY_BRANCH_IDS),
        "overspeed_threshold": OVERSPEED_THRESHOLD,
        "overspeed_comparator": OVERSPEED_COMPARATOR,
        "discovery_aggregate_hash": discovery_aggregate_hash,
        "artifact_hashes": artifact_hashes,
        "artifact_filenames": list(RESULT_ARTIFACTS),
        "scientific_claim": (
            "bounded natural prediction-boundary coverage under existing physical semantics"
        ),
    }
    manifest["canonical_manifest_hash"] = canonical_sha256(manifest)
    payloads["discovery_manifest.json"] = _json_bytes(manifest)
    validate_discovery_payloads(payloads, source_plan=plan)
    return payloads


def _validate_record(record: Mapping[str, object]) -> None:
    copy = dict(record)
    supplied = copy.pop("canonical_observation_hash", None)
    if supplied != canonical_sha256(copy):
        raise PredictionBoundaryDiscoveryError("observation hash mismatch")
    candidate = record.get("candidate_boundary") is True
    realized = _finite(record.get("realized_speed_ratio"), "realized_speed_ratio")
    predicted = _finite(record.get("predicted_speed_ratio"), "predicted_speed_ratio")
    if candidate != is_prediction_boundary_candidate(realized, predicted):
        raise PredictionBoundaryDiscoveryError("candidate comparison mismatch")
    if candidate and (
        record.get("final_veto_decision") != "veto"
        or record.get("transition_executed") is not False
        or record.get("realized_next_state") is not None
        or record.get("fallback_execution_count") != 0
    ):
        raise PredictionBoundaryDiscoveryError("candidate executed physical action")
    if (
        record.get("active_authority_granted") is not False
        or record.get("hazard_arrest_interventions") != 0
    ):
        raise PredictionBoundaryDiscoveryError("active authority leaked into discovery")


def validate_discovery_payloads(
    payloads: Mapping[str, bytes],
    *,
    source_plan: Mapping[str, object] | None = None,
) -> None:
    if set(payloads) != set(RESULT_ARTIFACTS):
        raise PredictionBoundaryDiscoveryError("discovery artifact set is incomplete")
    plan = _mapping(json.loads(payloads["discovery_plan.json"]), "discovery plan")
    supplied_plan_hash = plan.get("canonical_plan_hash")
    if supplied_plan_hash != canonical_sha256(plan_scientific_payload(plan)):
        raise PredictionBoundaryDiscoveryError("published plan hash mismatch")
    if source_plan is not None and plan != dict(source_plan):
        raise PredictionBoundaryDiscoveryError("published plan differs from source plan")
    manifest = _mapping(
        json.loads(payloads["discovery_manifest.json"]), "discovery manifest"
    )
    manifest_copy = dict(manifest)
    supplied_manifest_hash = manifest_copy.pop("canonical_manifest_hash", None)
    if supplied_manifest_hash != canonical_sha256(manifest_copy):
        raise PredictionBoundaryDiscoveryError("discovery manifest hash mismatch")
    if manifest.get("discovery_plan_hash") != supplied_plan_hash:
        raise PredictionBoundaryDiscoveryError("manifest plan hash mismatch")
    artifact_hashes = _mapping(manifest.get("artifact_hashes"), "artifact_hashes")
    expected_artifact_hashes = {
        name: sha256_bytes(payloads[name])
        for name in sorted(payloads)
        if name != "discovery_manifest.json"
    }
    if artifact_hashes != expected_artifact_hashes:
        raise PredictionBoundaryDiscoveryError("published artifact hash mismatch")
    if manifest.get("discovery_aggregate_hash") != canonical_sha256(
        artifact_hashes
    ):
        raise PredictionBoundaryDiscoveryError("discovery aggregate hash mismatch")
    if (
        manifest.get("planned_trajectory_count") != PLANNED_TRAJECTORY_COUNT
        or manifest.get("started_trajectory_count") != PLANNED_TRAJECTORY_COUNT
        or manifest.get("completed_trajectory_count") != PLANNED_TRAJECTORY_COUNT
        or manifest.get("overspeed_threshold") != OVERSPEED_THRESHOLD
        or manifest.get("overspeed_comparator") != OVERSPEED_COMPARATOR
        or manifest.get("active_authority_granted") is not False
        or manifest.get("hazard_arrest_interventions") != 0
        or manifest.get("automatic_retry_count") != 0
    ):
        raise PredictionBoundaryDiscoveryError("manifest contract changed")

    index = _mapping(json.loads(payloads["trajectory_index.json"]), "trajectory index")
    entries = index.get("trajectories")
    if not isinstance(entries, list) or len(entries) != PLANNED_TRAJECTORY_COUNT:
        raise PredictionBoundaryDiscoveryError("trajectory index count mismatch")
    ids = [str(item["trajectory_id"]) for item in entries if isinstance(item, Mapping)]
    source_ids = cast(list[object], plan.get("source_case_ids"))
    expected_ids = [
        trajectory_id(str(case_id), branch)
        for case_id in source_ids
        for branch in DISCOVERY_BRANCH_IDS
    ]
    if ids != expected_ids or len(ids) != len(entries):
        raise PredictionBoundaryDiscoveryError("trajectory ordering mismatch")
    for item_value in entries:
        item = _mapping(item_value, "trajectory")
        if int(item["physical_transition_count"]) > 32:
            raise PredictionBoundaryDiscoveryError("trajectory exceeded physical horizon")
        records = item.get("records")
        if not isinstance(records, list) or len(records) != item.get("states_evaluated"):
            raise PredictionBoundaryDiscoveryError("trajectory record count mismatch")
        for expected_index, record_value in enumerate(records):
            record = _mapping(record_value, "trajectory record")
            if record.get("event_index") != expected_index:
                raise PredictionBoundaryDiscoveryError("trajectory event ordering mismatch")
            _validate_record(record)
        actions = [
            record["normal_action"]
            for record in records
            if record.get("transition_executed") is True
        ]
        if item.get("action_trace_hash") != canonical_sha256(actions):
            raise PredictionBoundaryDiscoveryError("trajectory action trace hash mismatch")
        if records:
            states = [records[0]["current_state"]] + [
                record["realized_next_state"]
                for record in records
                if record.get("transition_executed") is True
            ]
            if item.get("state_trace_hash") != canonical_sha256(states):
                raise PredictionBoundaryDiscoveryError("trajectory state trace hash mismatch")
        identity = {
            "trajectory_id": item["trajectory_id"],
            "case_id": item["case_id"],
            "branch_id": item["branch_id"],
            "source_provenance_kind": item["source_provenance_kind"],
            "source_registry_member_id": item["source_registry_member_id"],
            "source_canonical_hash": item["source_canonical_hash"],
            "source_equivalence_checks": item["source_equivalence_checks"],
            "source_configuration_hash": item["source_configuration_hash"],
            "boundary_type": item["boundary_type"],
            "prefix_transition_count": item["prefix_transition_count"],
            "prefix_actual_transition_count": item["prefix_actual_transition_count"],
            "prefix_action_trace_hash": item["prefix_action_trace_hash"],
            "prefix_state_trace_hash": item["prefix_state_trace_hash"],
            "source_boundary_state_hash": item["source_boundary_state_hash"],
            "action_trace_hash": item["action_trace_hash"],
            "state_trace_hash": item["state_trace_hash"],
            "records": records,
        }
        if item.get("source_trajectory_hash") != canonical_sha256(identity):
            raise PredictionBoundaryDiscoveryError("source trajectory hash mismatch")
        checks = item.get("source_equivalence_checks")
        if not isinstance(checks, list) or not checks or any(
            not isinstance(check, list)
            or len(check) != 2
            or not isinstance(check[0], str)
            or check[1] is not True
            for check in checks
        ):
            raise PredictionBoundaryDiscoveryError("source equivalence checks failed")
        if item.get("boundary_type") == LEGACY_FIXED_PREFIX:
            if (
                item.get("case_id") != LEGACY_CASE_ID
                or item.get("source_provenance_kind")
                != "legacy_canonical_registered_reproduction"
                or item.get("source_registry_member_id") != LEGACY_MEMBER_ID
            ):
                raise PredictionBoundaryDiscoveryError(
                    "legacy trajectory source provenance mismatch"
                )
        elif (
            item.get("source_provenance_kind") != "generated_registry_member"
            or item.get("source_registry_member_id") is not None
        ):
            raise PredictionBoundaryDiscoveryError(
                "generated trajectory source provenance mismatch"
            )

    candidate_doc = _mapping(
        json.loads(payloads["candidate_boundaries.json"]), "candidate boundaries"
    )
    candidates = candidate_doc.get("candidate_boundaries")
    if not isinstance(candidates, list) or candidate_doc.get(
        "candidate_boundary_count"
    ) != len(candidates):
        raise PredictionBoundaryDiscoveryError("candidate count mismatch")
    entries_by_hash = {str(item["source_trajectory_hash"]): item for item in entries}
    for candidate_value in candidates:
        candidate = _mapping(candidate_value, "candidate")
        copy = dict(candidate)
        supplied = copy.pop("canonical_candidate_hash", None)
        if supplied != canonical_sha256(copy):
            raise PredictionBoundaryDiscoveryError("candidate hash mismatch")
        if (
            _finite(candidate.get("realized_speed_ratio"), "realized")
            > OVERSPEED_THRESHOLD
            or _finite(candidate.get("predicted_speed_ratio"), "predicted")
            <= OVERSPEED_THRESHOLD
            or candidate.get("Final_Veto_decision") != "veto"
            or candidate.get("candidate_action_physically_executed") is not False
            or candidate.get("fallback_execution_count") != 0
            or candidate.get("active_authority_granted") is not False
            or candidate.get("hazard_arrest_interventions") != 0
            or candidate.get("source_trajectory_hash") not in entries_by_hash
            or candidate.get("discovery_plan_hash") != supplied_plan_hash
        ):
            raise PredictionBoundaryDiscoveryError("candidate contract violation")
        source_entry = entries_by_hash[str(candidate["source_trajectory_hash"])]
        source_records = cast(list[object], source_entry["records"])
        source_index = _integer(
            candidate.get("boundary_event_index"), "boundary_event_index"
        )
        if source_index >= len(source_records):
            raise PredictionBoundaryDiscoveryError("candidate source event is missing")
        source_record = _mapping(source_records[source_index], "candidate source record")
        expected_source_fields = {
            "source_observation_hash": source_record["canonical_observation_hash"],
            "current_state_hash": source_record["current_state_hash"],
            "action_hash": source_record["normal_action_hash"],
            "predicted_state_hash": source_record["predicted_state_hash"],
            "realized_speed_ratio": source_record["realized_speed_ratio"],
            "predicted_speed_ratio": source_record["predicted_speed_ratio"],
        }
        if any(candidate.get(key) != value for key, value in expected_source_fields.items()):
            raise PredictionBoundaryDiscoveryError("candidate source provenance mismatch")
    if manifest.get("candidate_boundary_count") != len(candidates):
        raise PredictionBoundaryDiscoveryError("manifest candidate count mismatch")

    coverage = _mapping(json.loads(payloads["coverage_summary.json"]), "coverage")
    bins = _mapping(
        coverage.get("predicted_speed_ratio_diagnostic_bins"), "diagnostic bins"
    )
    if set(bins) != set(DIAGNOSTIC_BINS) or sum(int(value) for value in bins.values()) != coverage.get(
        "states_evaluated"
    ):
        raise PredictionBoundaryDiscoveryError("diagnostic coverage mismatch")
    if coverage.get("candidate_boundary_count") != len(candidates):
        raise PredictionBoundaryDiscoveryError("coverage candidate count mismatch")
    if coverage.get("total_physical_transition_count") != (
        coverage.get("prefix_physical_transition_count")
        + coverage.get("normal_branch_physical_transition_count")
    ):
        raise PredictionBoundaryDiscoveryError("physical transition accounting mismatch")
    protected = _mapping(
        json.loads(payloads["protected_evidence_report.json"]), "protected report"
    )
    if (
        protected.get("before") != protected.get("after")
        or protected.get("all_protected_evidence_unchanged") is not True
    ):
        raise PredictionBoundaryDiscoveryError("protected evidence report failed")
    summary = payloads["summary.md"].decode("utf-8")
    for phrase in (
        "Active authority granted: false",
        "Hazard-arrest interventions: 0",
        "does not demonstrate hazard-arrest effectiveness",
    ):
        if phrase not in summary:
            raise PredictionBoundaryDiscoveryError("summary claim boundary is incomplete")


def _aggregate_paths(repository_root: Path, relative_paths: Sequence[str]) -> str:
    root = repository_root.resolve()
    files: list[Path] = []
    for relative in relative_paths:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PredictionBoundaryDiscoveryError(
                f"protected path escapes repository: {relative}"
            ) from exc
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
        else:
            raise PredictionBoundaryDiscoveryError(
                f"protected path is missing: {relative}"
            )
    rows = [
        f"{path.relative_to(root).as_posix()}|{file_sha256(path)}"
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


def load_published_payloads(repository_root: Path) -> dict[str, bytes]:
    source = repository_root.resolve() / OUTPUT_PATH
    if not source.is_dir():
        raise PredictionBoundaryDiscoveryError("published discovery result is missing")
    payloads = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    plan = load_discovery_plan(repository_root)
    validate_discovery_payloads(payloads, source_plan=plan)
    protected = json.loads(payloads["protected_evidence_report.json"])
    if protected.get("after") != protected_evidence_hashes(repository_root):
        raise PredictionBoundaryDiscoveryError("current protected evidence hash mismatch")
    return payloads


def validate_static_sources(
    repository_root: Path,
    *,
    require_output_absent: bool = False,
) -> dict[str, object]:
    root = repository_root.resolve()
    plan = load_discovery_plan(root)
    definitions = build_trajectory_definitions(build_source_case_inventory(root))
    qualification_manifest = (
        root
        / "analysis"
        / "stage2a_hazard_arrest_qualification_v0"
        / "qualification_manifest.json"
    )
    if not qualification_manifest.is_file():
        raise PredictionBoundaryDiscoveryError("frozen Stage 2A qualification is missing")
    qualification = _mapping(
        json.loads(qualification_manifest.read_text("utf-8")),
        "qualification manifest",
    )
    qualification_copy = dict(qualification)
    supplied_qualification_hash = qualification_copy.pop(
        "canonical_manifest_hash", None
    )
    if supplied_qualification_hash != canonical_sha256(qualification_copy):
        raise PredictionBoundaryDiscoveryError(
            "frozen Stage 2A qualification manifest hash mismatch"
        )
    if (
        qualification.get("eligible_boundary_count") != 0
        or qualification.get("physical_executions") != 0
        or qualification.get("active_authority_executed") is not False
    ):
        raise PredictionBoundaryDiscoveryError(
            "discovery requires the frozen no-eligible qualification result"
        )
    if require_output_absent and (root / OUTPUT_PATH).exists():
        raise PredictionBoundaryDiscoveryError("discovery output already exists")
    hashes = protected_evidence_hashes(root)
    return {
        "valid": True,
        "source_case_count": SOURCE_CASE_COUNT,
        "planned_trajectory_count": len(definitions),
        "maximum_physical_transitions_per_trajectory": (
            MAXIMUM_PHYSICAL_TRANSITIONS_PER_TRAJECTORY
        ),
        "plan_hash": plan["canonical_plan_hash"],
        "protected_group_count": len(hashes),
        "simulation_executed": False,
        "write_performed": False,
        "active_authority_granted": False,
        "hazard_arrest_interventions": 0,
    }


__all__ = [
    "COMPLETED_DATE",
    "DIAGNOSTIC_BINS",
    "DISCOVERY_BRANCH_IDS",
    "DISCOVERY_ID",
    "DISCOVERY_SCHEMA_VERSION",
    "DiscoverySource",
    "DiscoveryTrajectoryResult",
    "MAXIMUM_PHYSICAL_TRANSITIONS_PER_TRAJECTORY",
    "NormalActionEvaluation",
    "OUTPUT_PATH",
    "PLAN_PATH",
    "PLANNED_TRAJECTORY_COUNT",
    "PredictionBoundaryDiscoveryError",
    "RESULT_ARTIFACTS",
    "build_discovery_payloads",
    "build_trajectory_definitions",
    "evaluate_normal_action",
    "execute_allowed_normal_transition",
    "execute_frozen_discovery",
    "generated_discovery_source",
    "is_prediction_boundary_candidate",
    "load_discovery_plan",
    "load_published_payloads",
    "protected_evidence_hashes",
    "prepare_discovery_sources",
    "prepare_legacy_discovery_source",
    "run_discovery_trajectory",
    "sha256_bytes",
    "state_document",
    "state_from_branch_document",
    "trajectory_id",
    "validate_discovery_payloads",
    "validate_discovery_plan",
    "validate_static_sources",
]
