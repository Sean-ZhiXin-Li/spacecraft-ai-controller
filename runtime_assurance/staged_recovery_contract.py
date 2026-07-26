from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Mapping, Sequence


ARCHITECTURE_ID = "staged_recovery_architecture_v0"
ARCHITECTURE_VERSION = "staged_recovery_architecture_v0"
ARCHITECTURE_COMPLETED_DATE = "2026-07-26"
ARCHITECTURE_MANIFEST_SCHEMA_VERSION = "staged_recovery_architecture_manifest_v0"
EVIDENCE_TRACEABILITY_SCHEMA_VERSION = "staged_recovery_evidence_traceability_v0"
NO_PROGRESS_CONTRACT_VERSION = "staged_recovery_no_progress_contract_v0"
HYSTERESIS_CONTRACT_VERSION = "staged_recovery_hysteresis_contract_v0"

EXECUTION_NOT_AUTHORIZED = "not_authorized"
EXECUTION_NOT_AUTHORIZED_REASON = (
    "phase actions, numerical transition guards, no-progress thresholds, "
    "hysteresis parameters, and required trajectory instrumentation are not "
    "yet frozen and validated"
)

PHASE34_RADIUS_ERROR_RATIO_MAX = 0.0025
PHASE34_RADIAL_VELOCITY_RATIO_MAX = 0.02
PHASE34_TANGENTIAL_VELOCITY_ERROR_RATIO_MAX = 0.25

FROZEN_ADVERSE_STOP_PRIORITY = (
    "invalid_simulation",
    "invalid_recovery_evaluation",
    "overspeed",
    "instability",
    "unsafe_state",
    "action_rejected",
    "explicit_abort",
    "recovery_success",
    "recovery_horizon_exhausted",
    "total_horizon_exhausted",
)


class RecoveryPhase(str, Enum):
    HAZARD_ARREST = "hazard_arrest"
    STABILIZATION_ASSESSMENT = "stabilization_assessment"
    RADIAL_RECOMMITMENT = "radial_recommitment"
    TANGENTIAL_ALIGNMENT = "tangential_alignment"
    CROSSING_PREPARATION = "crossing_preparation"
    RECOVERABILITY_VERIFICATION = "recoverability_verification"
    NOMINAL_HANDOFF = "nominal_handoff"
    RETREAT = "retreat"
    EXPLICIT_ABORT = "explicit_abort"


class RecoveryTransitionReason(str, Enum):
    HAZARD_ARRESTED = "hazard_arrested"
    ASSESSMENT_REQUIRES_ABORT = "assessment_requires_abort"
    STABILIZATION_ACCEPTED = "stabilization_accepted"
    RECOVERY_NOT_JUSTIFIED = "recovery_not_justified"
    RADIAL_PROGRESS_READY = "radial_progress_ready"
    REASSESSMENT_REQUIRED = "reassessment_required"
    TANGENTIAL_ALIGNMENT_READY = "tangential_alignment_ready"
    RADIAL_RECOMMITMENT_REQUIRED = "radial_recommitment_required"
    CROSSING_PREPARATION_READY = "crossing_preparation_ready"
    TANGENTIAL_REALIGNMENT_REQUIRED = "tangential_realignment_required"
    RECOVERABILITY_VERIFIED = "recoverability_verified"
    CONTROLLED_RECOVERY_CONTINUES = "controlled_recovery_continues"
    RETREAT_TERMINATED = "retreat_terminated"
    EXPLICIT_ABORT_SELECTED = "explicit_abort_selected"


class RecoveryEvidenceSupport(str, Enum):
    DIRECTLY_SUPPORTED = "directly_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONSISTENT_WITH_EVIDENCE = "consistent_with_evidence"
    DESIGN_HYPOTHESIS = "design_hypothesis"
    NOT_YET_SPECIFIED = "not_yet_specified"


class RecoverySignalStatus(str, Enum):
    MEASURED = "measured"
    DERIVED = "derived"
    ONE_STEP_PREDICTED = "one_step_predicted"
    MULTI_STEP_PREDICTED = "multi_step_predicted"
    HEURISTIC = "heuristic"
    NOT_EVALUATED = "not_evaluated"
    INVALID = "invalid"


class RecoveryNoProgressStatus(str, Enum):
    PROGRESSING = "progressing"
    STALLED = "stalled"
    REGRESSING = "regressing"
    NOT_EVALUATED = "not_evaluated"
    INVALID = "invalid"


class RecoveryContractError(ValueError):
    pass


_SUPPORTED_VALUE_STATUSES = frozenset(
    {
        RecoverySignalStatus.MEASURED,
        RecoverySignalStatus.DERIVED,
        RecoverySignalStatus.ONE_STEP_PREDICTED,
        RecoverySignalStatus.MULTI_STEP_PREDICTED,
        RecoverySignalStatus.HEURISTIC,
    }
)


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, tuple):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_json_value(item)
            for key, item in value.items()
        )
    return False


@dataclass(frozen=True, slots=True)
class RecoverySignalValue:
    status: RecoverySignalStatus
    value: object = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, RecoverySignalStatus):
            raise RecoveryContractError("signal status must be RecoverySignalStatus")
        if self.status in {
            RecoverySignalStatus.NOT_EVALUATED,
            RecoverySignalStatus.INVALID,
        }:
            if self.value is not None:
                raise RecoveryContractError(
                    "not_evaluated and invalid signals must not carry a favorable value"
                )
        elif self.value is None:
            raise RecoveryContractError("evaluated signal must carry an explicit value")
        if not _is_json_value(self.value):
            raise RecoveryContractError("signal value must be finite and JSON-compatible")
        if self.source is not None and not self.source:
            raise RecoveryContractError("signal source must be nonempty when present")

    @property
    def is_available(self) -> bool:
        return self.status in _SUPPORTED_VALUE_STATUSES


SIGNAL_IDS = (
    "simulation_validity",
    "recovery_evaluation_validity",
    "overspeed_status",
    "instability_status",
    "unsafe_state_status",
    "final_veto_decision",
    "action_rejection_status",
    "explicit_abort_requested",
    "position_x",
    "position_y",
    "velocity_x",
    "velocity_y",
    "radius",
    "target_radius",
    "target_radius_error",
    "radial_velocity",
    "tangential_velocity",
    "target_circular_speed",
    "radial_velocity_ratio",
    "tangential_velocity_error_ratio",
    "realized_speed_ratio",
    "predicted_speed_ratio",
    "overspeed_headroom",
    "orbital_energy_or_proxy",
    "target_radius_crossing",
    "first_crossing_step",
    "phase34_compatible_recoverability",
    "recoverability_component_vector",
    "recovery_success_v0",
    "recovery_horizon_remaining",
    "recovery_horizon_exhausted",
    "total_horizon_remaining",
    "total_horizon_exhausted",
    "radial_progress",
    "radial_progress_direction",
    "radial_progress_rate",
    "tangential_progress",
    "crossing_progress",
    "predicted_crossing_direction",
    "crossing_proximity",
    "energy_change_direction",
    "available_correction_authority",
    "action_saturation_margin",
    "no_progress_status",
    "phase_dwell_count",
    "phase_transition_count",
    "recent_phase_history",
    "phase_transition_reason",
    "proposed_action",
    "executed_action",
    "handoff_readiness",
    "simulator_success",
)


@dataclass(frozen=True, slots=True)
class StagedRecoveryObservation:
    case_id: str
    seed: int
    branch_state_hash: str
    simulator_configuration_hash: str
    constants_hash: str
    implementation_commit: str
    recovery_step: int
    total_transition_count: int
    current_phase: RecoveryPhase
    previous_phase: RecoveryPhase | None
    signals: tuple[tuple[str, RecoverySignalValue], ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.case_id, "case_id"),
            (self.branch_state_hash, "branch_state_hash"),
            (self.simulator_configuration_hash, "simulator_configuration_hash"),
            (self.constants_hash, "constants_hash"),
            (self.implementation_commit, "implementation_commit"),
        ):
            if not isinstance(value, str) or not value:
                raise RecoveryContractError(f"{name} must be nonempty")
        for value, name in (
            (self.seed, "seed"),
            (self.recovery_step, "recovery_step"),
            (self.total_transition_count, "total_transition_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RecoveryContractError(f"{name} must be a nonnegative integer")
        if not isinstance(self.current_phase, RecoveryPhase):
            raise RecoveryContractError("current_phase must be a RecoveryPhase")
        if self.previous_phase is not None and not isinstance(
            self.previous_phase, RecoveryPhase
        ):
            raise RecoveryContractError("previous_phase must be a RecoveryPhase or None")
        keys = tuple(key for key, _ in self.signals)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise RecoveryContractError("signals must use unique keys in sorted order")
        if set(keys) != set(SIGNAL_IDS):
            missing = sorted(set(SIGNAL_IDS) - set(keys))
            unknown = sorted(set(keys) - set(SIGNAL_IDS))
            raise RecoveryContractError(
                f"observation signal set mismatch; missing={missing}, unknown={unknown}"
            )
        if any(not isinstance(value, RecoverySignalValue) for _, value in self.signals):
            raise RecoveryContractError("every observation signal must be status-bearing")

    def signal(self, signal_id: str) -> RecoverySignalValue:
        for current_id, value in self.signals:
            if current_id == signal_id:
                return value
        raise KeyError(signal_id)


@dataclass(frozen=True, slots=True)
class RecoveryPhaseContract:
    phase_id: RecoveryPhase
    purpose: str
    terminal: bool
    required_signal_ids: tuple[str, ...]
    conceptual_outputs: tuple[str, ...]
    action_objective: str
    action_required_inputs: tuple[str, ...]
    prohibited_outcomes: tuple[str, ...]
    action_output_contract: str
    saturation_awareness_required: bool
    final_veto_required: bool
    progress_monitoring_required: bool
    action_law_status: RecoveryEvidenceSupport
    evidence_support: RecoveryEvidenceSupport


@dataclass(frozen=True, slots=True)
class RecoveryTransitionContract:
    transition_id: str
    source_phase: RecoveryPhase
    destination_phase: RecoveryPhase
    required_signal_ids: tuple[str, ...]
    prohibited_trigger_signal_ids: tuple[str, ...]
    transition_reason: RecoveryTransitionReason
    adverse_stop_precedence: bool
    minimum_evidence_support: RecoveryEvidenceSupport
    evidence_support: RecoveryEvidenceSupport
    threshold_status: RecoveryEvidenceSupport
    executable: bool
    hysteresis_required: bool
    requires_new_evidence: bool


@dataclass(frozen=True, slots=True)
class RecoveryPhaseDecision:
    source_phase: RecoveryPhase
    destination_phase: RecoveryPhase | None
    status: str
    reason: str
    missing_signal_ids: tuple[str, ...] = ()
    invalid_signal_ids: tuple[str, ...] = ()
    adverse_stop: str | None = None


@dataclass(frozen=True, slots=True)
class RecoveryNoProgressContract:
    contract_version: str = NO_PROGRESS_CONTRACT_VERSION
    monitored_signal_id: str | None = None
    desired_direction: str | None = None
    observation_window: int | None = None
    minimum_meaningful_improvement: float | None = None
    phase_specific_dwell_limit: int | None = None
    missing_evidence_status: RecoveryNoProgressStatus = (
        RecoveryNoProgressStatus.NOT_EVALUATED
    )
    invalid_evidence_status: RecoveryNoProgressStatus = RecoveryNoProgressStatus.INVALID
    executable: bool = False


@dataclass(frozen=True, slots=True)
class RecoveryNoProgressResult:
    status: RecoveryNoProgressStatus
    reason: str
    signal_id: str | None
    sample_count: int
    measured_improvement: float | None


@dataclass(frozen=True, slots=True)
class RecoveryHysteresisContract:
    contract_version: str = HYSTERESIS_CONTRACT_VERSION
    minimum_phase_dwell: int | None = None
    entry_threshold_id: str | None = None
    exit_threshold_id: str | None = None
    transition_cooldown: int | None = None
    consecutive_evidence_count: int | None = None
    phase_transition_budget: int | None = None
    repeated_cycle_window: int | None = None
    requires_new_evidence: bool = True
    preserves_raw_transition_reason: bool = True
    executable: bool = False


@dataclass(frozen=True, slots=True)
class RecoveryArchitectureManifest:
    architecture_id: str
    version: str
    completed_date: str
    source_evidence_commits: tuple[tuple[str, str], ...]
    source_artifact_hashes: tuple[tuple[str, str], ...]
    phase_contracts: tuple[RecoveryPhaseContract, ...]
    transitions: tuple[RecoveryTransitionContract, ...]
    forbidden_transitions: tuple[tuple[str, str], ...]
    adverse_stop_priority: tuple[str, ...]
    required_signal_ids: tuple[str, ...]
    unresolved_threshold_ids: tuple[str, ...]
    no_progress_contract: RecoveryNoProgressContract
    hysteresis_contract: RecoveryHysteresisContract
    action_law_status: RecoveryEvidenceSupport
    execution_authorization_status: str
    execution_authorization_reason: str
    claim_restrictions: tuple[str, ...]
    reserved_future_experiment_ids: tuple[str, ...]
    canonical_payload_hash: str


@dataclass(frozen=True, slots=True)
class RecoveryArchitectureValidationReport:
    structurally_valid: bool
    execution_authorized: bool
    errors: tuple[str, ...]
    unresolved_items: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceTraceabilityItem:
    item_id: str
    description: str
    evidence_support: RecoveryEvidenceSupport
    sources: tuple[tuple[tuple[str, str], ...], ...]
    limitation: str
    unresolved_dependency: str | None


def unavailable_signals() -> tuple[tuple[str, RecoverySignalValue], ...]:
    return tuple(
        (signal_id, RecoverySignalValue(RecoverySignalStatus.NOT_EVALUATED))
        for signal_id in sorted(SIGNAL_IDS)
    )


def replace_signals(
    signals: tuple[tuple[str, RecoverySignalValue], ...],
    **updates: RecoverySignalValue,
) -> tuple[tuple[str, RecoverySignalValue], ...]:
    values = dict(signals)
    unknown = sorted(set(updates) - set(SIGNAL_IDS))
    if unknown:
        raise RecoveryContractError(f"unknown signals: {unknown}")
    values.update(updates)
    return tuple(sorted(values.items()))


_PHASE_PURPOSES = {
    RecoveryPhase.HAZARD_ARREST: (
        "Suppress the immediate declared hazard-producing behavior without "
        "claiming task recovery."
    ),
    RecoveryPhase.STABILIZATION_ASSESSMENT: (
        "Determine whether the post-arrest state is valid, observable, and usable "
        "for recovery planning."
    ),
    RecoveryPhase.RADIAL_RECOMMITMENT: (
        "Restore explicitly monitored task-directed radial progress toward the "
        "target-radius crossing condition."
    ),
    RecoveryPhase.TANGENTIAL_ALIGNMENT: (
        "Reduce tangential error without erasing required radial progress or "
        "mistaking one component for full recovery."
    ),
    RecoveryPhase.CROSSING_PREPARATION: (
        "Coordinate radial and tangential geometry before a target-radius crossing."
    ),
    RecoveryPhase.RECOVERABILITY_VERIFICATION: (
        "Verify crossing, Phase34-compatible recoverability, and Recovery Success "
        "v0 prerequisites as separate evidence."
    ),
    RecoveryPhase.NOMINAL_HANDOFF: (
        "Record a verified, provenance-bearing return of authority to the nominal "
        "controller without implementing that switch."
    ),
    RecoveryPhase.RETREAT: (
        "Represent movement toward a separately declared lower-risk state without "
        "calling retreat task recovery."
    ),
    RecoveryPhase.EXPLICIT_ABORT: (
        "Terminate without a further physical recovery transition and without "
        "claiming recovery, unsafe state, or simulator success."
    ),
}


def default_phase_contracts() -> tuple[RecoveryPhaseContract, ...]:
    specifications = (
        (
            RecoveryPhase.HAZARD_ARREST,
            False,
            (
                "simulation_validity",
                "overspeed_status",
                "final_veto_decision",
                "proposed_action",
                "executed_action",
                "realized_speed_ratio",
                "overspeed_headroom",
            ),
            (
                "declared_hazard_status",
                "arrest_completion_status",
                "proposed_and_executed_action_evidence",
            ),
            "Suppress a declared imminent hazard while preserving valid evidence.",
            ("predicted_speed_ratio", "overspeed_headroom", "final_veto_decision"),
            ("task_recovery_claim", "premature_nominal_handoff"),
            "status-bearing proposed action only; physical action law unresolved",
            True,
            True,
            True,
            RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
        ),
        (
            RecoveryPhase.STABILIZATION_ASSESSMENT,
            False,
            (
                "simulation_validity",
                "recovery_evaluation_validity",
                "instability_status",
                "unsafe_state_status",
                "realized_speed_ratio",
                "recoverability_component_vector",
                "radial_progress",
            ),
            ("assessment_status", "observability_status", "recovery_planning_status"),
            "Assess validity and recovery evidence; do not produce a thrust action.",
            (),
            ("absence_of_overspeed_as_stabilization", "unsupported_phase_advance"),
            "assessment decision only; missing evidence remains not_evaluated",
            False,
            False,
            True,
            RecoveryEvidenceSupport.CONSISTENT_WITH_EVIDENCE,
        ),
        (
            RecoveryPhase.RADIAL_RECOMMITMENT,
            False,
            (
                "radius",
                "target_radius_error",
                "radial_velocity",
                "radial_velocity_ratio",
                "radial_progress",
                "radial_progress_direction",
                "radial_progress_rate",
                "overspeed_headroom",
                "available_correction_authority",
                "orbital_energy_or_proxy",
            ),
            ("radial_progress_status", "authority_status", "energy_change_status"),
            "Restore target-directed radial progress under declared authority.",
            (
                "target_radius_error",
                "radial_velocity",
                "overspeed_headroom",
                "available_correction_authority",
            ),
            ("unmonitored_motion_suppression", "overspeed", "unbounded_energy_change"),
            "bounded action proposal; law, target velocity, gain, and duration unresolved",
            True,
            True,
            True,
            RecoveryEvidenceSupport.PARTIALLY_SUPPORTED,
        ),
        (
            RecoveryPhase.TANGENTIAL_ALIGNMENT,
            False,
            (
                "tangential_velocity",
                "target_circular_speed",
                "tangential_velocity_error_ratio",
                "radial_progress",
                "overspeed_headroom",
                "orbital_energy_or_proxy",
                "action_saturation_margin",
            ),
            ("tangential_progress_status", "radial_progress_preservation_status"),
            "Reduce tangential error while preserving declared radial progress.",
            (
                "tangential_velocity_error_ratio",
                "radial_progress",
                "overspeed_headroom",
            ),
            (
                "tangential_improvement_as_full_recovery",
                "loss_of_required_radial_progress",
            ),
            "bounded action proposal; the previous 0.25 branch is evidence, not policy",
            True,
            True,
            True,
            RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
        ),
        (
            RecoveryPhase.CROSSING_PREPARATION,
            False,
            (
                "target_radius_error",
                "radial_velocity_ratio",
                "tangential_velocity_error_ratio",
                "crossing_progress",
                "predicted_crossing_direction",
                "crossing_proximity",
                "overspeed_headroom",
                "recoverability_component_vector",
            ),
            ("crossing_readiness_status", "recoverability_approach_status"),
            "Coordinate radial and tangential conditions before crossing.",
            (
                "target_radius_error",
                "radial_velocity_ratio",
                "tangential_velocity_error_ratio",
            ),
            ("unsupported_crossing_prediction", "incompatible_crossing_geometry"),
            "bounded action proposal; crossing predictor and guard thresholds unresolved",
            True,
            True,
            True,
            RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
        ),
        (
            RecoveryPhase.RECOVERABILITY_VERIFICATION,
            False,
            (
                "target_radius_crossing",
                "phase34_compatible_recoverability",
                "recoverability_component_vector",
                "recovery_success_v0",
                "simulation_validity",
                "recovery_evaluation_validity",
            ),
            ("crossing_status", "recoverability_status", "recovery_success_status"),
            "Evaluate existing predicates only; do not generate an action.",
            (),
            (
                "crossing_as_recovery",
                "simulator_success_as_recovery",
                "hazard_avoidance_as_recovery",
            ),
            "verification decision using frozen inclusive Phase34 thresholds",
            False,
            False,
            True,
            RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
        ),
        (
            RecoveryPhase.NOMINAL_HANDOFF,
            True,
            (
                "simulation_validity",
                "recovery_evaluation_validity",
                "overspeed_status",
                "instability_status",
                "unsafe_state_status",
                "action_rejection_status",
                "target_radius_crossing",
                "phase34_compatible_recoverability",
                "recovery_success_v0",
                "handoff_readiness",
            ),
            ("handoff_authorization_evidence", "handoff_provenance"),
            "No action is implemented; authority transfer remains a future boundary.",
            (),
            ("premature_handoff", "handoff_during_adverse_stop"),
            "terminal architecture decision only; controller switching not implemented",
            False,
            False,
            False,
            RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
        ),
        (
            RecoveryPhase.RETREAT,
            False,
            (
                "simulation_validity",
                "overspeed_status",
                "available_correction_authority",
                "no_progress_status",
            ),
            ("retreat_status", "task_abandonment_status"),
            "Move toward a separately frozen lower-risk target.",
            ("available_correction_authority", "overspeed_headroom"),
            ("retreat_as_task_recovery", "undeclared_retreat_target"),
            "retreat target, action, success predicate, and thresholds unresolved",
            True,
            True,
            True,
            RecoveryEvidenceSupport.NOT_YET_SPECIFIED,
        ),
        (
            RecoveryPhase.EXPLICIT_ABORT,
            True,
            (
                "explicit_abort_requested",
                "simulation_validity",
                "overspeed_status",
            ),
            ("terminal_abort_reason", "zero_post_decision_transition_evidence"),
            "Execute no physical recovery transition.",
            (),
            ("recovery_success_claim", "nominal_handoff", "physical_transition"),
            "terminal decision event with zero post-decision physical transitions",
            False,
            False,
            False,
            RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
        ),
    )
    phase_contracts = []
    for (
        phase_id,
        terminal,
        required,
        outputs,
        objective,
        action_inputs,
        prohibited,
        output_contract,
        saturation_required,
        veto_required,
        progress_required,
        evidence_support,
    ) in specifications:
        phase_contracts.append(
            RecoveryPhaseContract(
                phase_id=phase_id,
                purpose=_PHASE_PURPOSES[phase_id],
                terminal=terminal,
                required_signal_ids=tuple(required),
                conceptual_outputs=tuple(outputs),
                action_objective=objective,
                action_required_inputs=tuple(action_inputs),
                prohibited_outcomes=tuple(prohibited),
                action_output_contract=output_contract,
                saturation_awareness_required=saturation_required,
                final_veto_required=veto_required,
                progress_monitoring_required=progress_required,
                action_law_status=(
                    RecoveryEvidenceSupport.NOT_YET_SPECIFIED
                    if phase_id
                    in {
                        RecoveryPhase.HAZARD_ARREST,
                        RecoveryPhase.RADIAL_RECOMMITMENT,
                        RecoveryPhase.TANGENTIAL_ALIGNMENT,
                        RecoveryPhase.CROSSING_PREPARATION,
                        RecoveryPhase.RETREAT,
                    }
                    else RecoveryEvidenceSupport.DESIGN_HYPOTHESIS
                ),
                evidence_support=evidence_support,
            )
        )
    return tuple(phase_contracts)


_ALLOWED_GRAPH = (
    (RecoveryPhase.HAZARD_ARREST, RecoveryPhase.STABILIZATION_ASSESSMENT),
    (RecoveryPhase.HAZARD_ARREST, RecoveryPhase.EXPLICIT_ABORT),
    (RecoveryPhase.STABILIZATION_ASSESSMENT, RecoveryPhase.RADIAL_RECOMMITMENT),
    (RecoveryPhase.STABILIZATION_ASSESSMENT, RecoveryPhase.RETREAT),
    (RecoveryPhase.STABILIZATION_ASSESSMENT, RecoveryPhase.EXPLICIT_ABORT),
    (RecoveryPhase.RADIAL_RECOMMITMENT, RecoveryPhase.TANGENTIAL_ALIGNMENT),
    (RecoveryPhase.RADIAL_RECOMMITMENT, RecoveryPhase.STABILIZATION_ASSESSMENT),
    (RecoveryPhase.RADIAL_RECOMMITMENT, RecoveryPhase.RETREAT),
    (RecoveryPhase.RADIAL_RECOMMITMENT, RecoveryPhase.EXPLICIT_ABORT),
    (RecoveryPhase.TANGENTIAL_ALIGNMENT, RecoveryPhase.CROSSING_PREPARATION),
    (RecoveryPhase.TANGENTIAL_ALIGNMENT, RecoveryPhase.RADIAL_RECOMMITMENT),
    (RecoveryPhase.TANGENTIAL_ALIGNMENT, RecoveryPhase.STABILIZATION_ASSESSMENT),
    (RecoveryPhase.TANGENTIAL_ALIGNMENT, RecoveryPhase.RETREAT),
    (RecoveryPhase.TANGENTIAL_ALIGNMENT, RecoveryPhase.EXPLICIT_ABORT),
    (RecoveryPhase.CROSSING_PREPARATION, RecoveryPhase.RECOVERABILITY_VERIFICATION),
    (RecoveryPhase.CROSSING_PREPARATION, RecoveryPhase.RADIAL_RECOMMITMENT),
    (RecoveryPhase.CROSSING_PREPARATION, RecoveryPhase.TANGENTIAL_ALIGNMENT),
    (RecoveryPhase.CROSSING_PREPARATION, RecoveryPhase.STABILIZATION_ASSESSMENT),
    (RecoveryPhase.CROSSING_PREPARATION, RecoveryPhase.RETREAT),
    (RecoveryPhase.CROSSING_PREPARATION, RecoveryPhase.EXPLICIT_ABORT),
    (RecoveryPhase.RECOVERABILITY_VERIFICATION, RecoveryPhase.NOMINAL_HANDOFF),
    (RecoveryPhase.RECOVERABILITY_VERIFICATION, RecoveryPhase.RADIAL_RECOMMITMENT),
    (RecoveryPhase.RECOVERABILITY_VERIFICATION, RecoveryPhase.TANGENTIAL_ALIGNMENT),
    (RecoveryPhase.RECOVERABILITY_VERIFICATION, RecoveryPhase.RETREAT),
    (RecoveryPhase.RECOVERABILITY_VERIFICATION, RecoveryPhase.EXPLICIT_ABORT),
    (RecoveryPhase.RETREAT, RecoveryPhase.EXPLICIT_ABORT),
)


def _transition_metadata(
    source: RecoveryPhase, destination: RecoveryPhase
) -> tuple[tuple[str, ...], RecoveryTransitionReason, RecoveryEvidenceSupport]:
    if destination == RecoveryPhase.EXPLICIT_ABORT:
        return (
            ("explicit_abort_requested",),
            RecoveryTransitionReason.EXPLICIT_ABORT_SELECTED,
            RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
        )
    if destination == RecoveryPhase.RETREAT:
        return (
            ("no_progress_status", "available_correction_authority"),
            RecoveryTransitionReason.RECOVERY_NOT_JUSTIFIED,
            RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
        )
    metadata = {
        (
            RecoveryPhase.HAZARD_ARREST,
            RecoveryPhase.STABILIZATION_ASSESSMENT,
        ): (
            ("overspeed_status", "simulation_validity", "overspeed_headroom"),
            RecoveryTransitionReason.HAZARD_ARRESTED,
            RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
        ),
        (
            RecoveryPhase.STABILIZATION_ASSESSMENT,
            RecoveryPhase.RADIAL_RECOMMITMENT,
        ): (
            (
                "simulation_validity",
                "recovery_evaluation_validity",
                "instability_status",
                "unsafe_state_status",
                "radial_progress",
            ),
            RecoveryTransitionReason.STABILIZATION_ACCEPTED,
            RecoveryEvidenceSupport.CONSISTENT_WITH_EVIDENCE,
        ),
        (
            RecoveryPhase.RADIAL_RECOMMITMENT,
            RecoveryPhase.TANGENTIAL_ALIGNMENT,
        ): (
            (
                "radial_progress",
                "radial_progress_direction",
                "radial_progress_rate",
                "target_radius_error",
                "overspeed_headroom",
            ),
            RecoveryTransitionReason.RADIAL_PROGRESS_READY,
            RecoveryEvidenceSupport.PARTIALLY_SUPPORTED,
        ),
        (
            RecoveryPhase.RADIAL_RECOMMITMENT,
            RecoveryPhase.STABILIZATION_ASSESSMENT,
        ): (
            ("no_progress_status", "simulation_validity"),
            RecoveryTransitionReason.REASSESSMENT_REQUIRED,
            RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
        ),
        (
            RecoveryPhase.TANGENTIAL_ALIGNMENT,
            RecoveryPhase.CROSSING_PREPARATION,
        ): (
            (
                "tangential_velocity_error_ratio",
                "radial_progress",
                "overspeed_headroom",
            ),
            RecoveryTransitionReason.TANGENTIAL_ALIGNMENT_READY,
            RecoveryEvidenceSupport.PARTIALLY_SUPPORTED,
        ),
        (
            RecoveryPhase.TANGENTIAL_ALIGNMENT,
            RecoveryPhase.RADIAL_RECOMMITMENT,
        ): (
            ("radial_progress", "tangential_progress"),
            RecoveryTransitionReason.RADIAL_RECOMMITMENT_REQUIRED,
            RecoveryEvidenceSupport.CONSISTENT_WITH_EVIDENCE,
        ),
        (
            RecoveryPhase.TANGENTIAL_ALIGNMENT,
            RecoveryPhase.STABILIZATION_ASSESSMENT,
        ): (
            ("no_progress_status", "simulation_validity"),
            RecoveryTransitionReason.REASSESSMENT_REQUIRED,
            RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
        ),
        (
            RecoveryPhase.CROSSING_PREPARATION,
            RecoveryPhase.RECOVERABILITY_VERIFICATION,
        ): (
            (
                "target_radius_crossing",
                "crossing_proximity",
                "predicted_crossing_direction",
                "recoverability_component_vector",
                "simulation_validity",
            ),
            RecoveryTransitionReason.CROSSING_PREPARATION_READY,
            RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
        ),
        (
            RecoveryPhase.CROSSING_PREPARATION,
            RecoveryPhase.RADIAL_RECOMMITMENT,
        ): (
            ("radial_progress", "target_radius_error"),
            RecoveryTransitionReason.RADIAL_RECOMMITMENT_REQUIRED,
            RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
        ),
        (
            RecoveryPhase.CROSSING_PREPARATION,
            RecoveryPhase.TANGENTIAL_ALIGNMENT,
        ): (
            ("tangential_progress", "tangential_velocity_error_ratio"),
            RecoveryTransitionReason.TANGENTIAL_REALIGNMENT_REQUIRED,
            RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
        ),
        (
            RecoveryPhase.CROSSING_PREPARATION,
            RecoveryPhase.STABILIZATION_ASSESSMENT,
        ): (
            ("no_progress_status", "simulation_validity"),
            RecoveryTransitionReason.REASSESSMENT_REQUIRED,
            RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
        ),
        (
            RecoveryPhase.RECOVERABILITY_VERIFICATION,
            RecoveryPhase.NOMINAL_HANDOFF,
        ): (
            (
                "recovery_success_v0",
                "phase34_compatible_recoverability",
                "handoff_readiness",
            ),
            RecoveryTransitionReason.RECOVERABILITY_VERIFIED,
            RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
        ),
        (
            RecoveryPhase.RECOVERABILITY_VERIFICATION,
            RecoveryPhase.RADIAL_RECOMMITMENT,
        ): (
            ("target_radius_crossing", "radial_velocity_ratio"),
            RecoveryTransitionReason.CONTROLLED_RECOVERY_CONTINUES,
            RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
        ),
        (
            RecoveryPhase.RECOVERABILITY_VERIFICATION,
            RecoveryPhase.TANGENTIAL_ALIGNMENT,
        ): (
            ("target_radius_crossing", "tangential_velocity_error_ratio"),
            RecoveryTransitionReason.CONTROLLED_RECOVERY_CONTINUES,
            RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
        ),
        (RecoveryPhase.RETREAT, RecoveryPhase.EXPLICIT_ABORT): (
            ("explicit_abort_requested",),
            RecoveryTransitionReason.RETREAT_TERMINATED,
            RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
        ),
    }
    return metadata[(source, destination)]


def default_transition_contracts() -> tuple[RecoveryTransitionContract, ...]:
    contracts = []
    adverse_signals = (
        "simulation_validity",
        "recovery_evaluation_validity",
        "overspeed_status",
        "instability_status",
        "unsafe_state_status",
        "action_rejection_status",
    )
    for source, destination in _ALLOWED_GRAPH:
        required, reason, support = _transition_metadata(source, destination)
        contracts.append(
            RecoveryTransitionContract(
                transition_id=f"{source.value}__to__{destination.value}",
                source_phase=source,
                destination_phase=destination,
                required_signal_ids=tuple(required),
                prohibited_trigger_signal_ids=adverse_signals,
                transition_reason=reason,
                adverse_stop_precedence=True,
                minimum_evidence_support=(
                    RecoveryEvidenceSupport.DIRECTLY_SUPPORTED
                    if destination == RecoveryPhase.NOMINAL_HANDOFF
                    else RecoveryEvidenceSupport.NOT_YET_SPECIFIED
                ),
                evidence_support=support,
                threshold_status=(
                    RecoveryEvidenceSupport.DIRECTLY_SUPPORTED
                    if destination == RecoveryPhase.NOMINAL_HANDOFF
                    else RecoveryEvidenceSupport.NOT_YET_SPECIFIED
                ),
                executable=False,
                hysteresis_required=(
                    destination
                    not in {RecoveryPhase.EXPLICIT_ABORT, RecoveryPhase.NOMINAL_HANDOFF}
                ),
                requires_new_evidence=True,
            )
        )
    return tuple(contracts)


FORBIDDEN_TRANSITIONS = (
    (RecoveryPhase.HAZARD_ARREST.value, RecoveryPhase.NOMINAL_HANDOFF.value),
    (
        RecoveryPhase.STABILIZATION_ASSESSMENT.value,
        RecoveryPhase.NOMINAL_HANDOFF.value,
    ),
    (RecoveryPhase.RADIAL_RECOMMITMENT.value, RecoveryPhase.NOMINAL_HANDOFF.value),
    (RecoveryPhase.TANGENTIAL_ALIGNMENT.value, RecoveryPhase.NOMINAL_HANDOFF.value),
    (RecoveryPhase.EXPLICIT_ABORT.value, "any_active_phase"),
    (RecoveryPhase.NOMINAL_HANDOFF.value, "any_recovery_phase"),
    ("higher_priority_adverse_stop_active", RecoveryPhase.NOMINAL_HANDOFF.value),
)


UNRESOLVED_THRESHOLD_IDS = (
    "hazard_arrest_completion_guard",
    "stabilization_entry_guard",
    "stabilization_exit_guard",
    "radial_progress_target",
    "radial_progress_rate_minimum",
    "radial_phase_dwell_limit",
    "tangential_alignment_entry_guard",
    "tangential_alignment_exit_guard",
    "crossing_preparation_entry_guard",
    "crossing_preparation_exit_guard",
    "crossing_prediction_proximity",
    "retreat_entry_guard",
    "retreat_target_region",
    "retreat_success_predicate",
    "no_progress_observation_window",
    "no_progress_minimum_meaningful_improvement",
    "no_progress_phase_dwell_limit",
    "hysteresis_minimum_phase_dwell",
    "hysteresis_transition_cooldown",
    "hysteresis_consecutive_evidence_count",
    "hysteresis_phase_transition_budget",
    "hysteresis_repeated_cycle_window",
)


SOURCE_EVIDENCE_COMMITS = (
    ("recovery_evaluators_v0", "c2580eaf751a4109c697ac608dcd9cea6d8c2f74"),
    ("recovery_experiment_execution_v0", "2e1fffbb00789c185256d0b13dff65150f21ba50"),
    ("recovery_mechanism_diagnosis_v0", "03c8a355586cc5adeec6e8e3d7e192bb84d5c1d0"),
    ("recovery_result_v0", "5f31c3fd74dbf8e8ea5a60d70d7b88f5a9def7c8"),
)


SOURCE_ARTIFACT_HASHES = (
    (
        "analysis/recovery_action_branching_nonformal_v0/manifest.json",
        "c317f94f937412f4fb5ac826fd97b21002c47cfc9bba6d9ad4eda6c4be1a921b",
    ),
    (
        "analysis/recovery_action_branching_nonformal_v0/branch_state.json",
        "b9fbcdd3544f527c7431d1b3bc5795ea755935a4973a18cbb8d8b710685d64fc",
    ),
    (
        "analysis/recovery_action_branching_nonformal_v0/results.csv",
        "c13abb4e15a6f04a9322c6c7955553464b9815d1b4d5c58374a3100eb4ccc668",
    ),
    (
        "analysis/recovery_action_branching_nonformal_v0/decision_log.jsonl",
        "43cfc05100648b6d0d652a8ac1d9a35f7179ebec78ff0138eaa5e7ab846096b4",
    ),
    (
        "analysis/recovery_action_branching_nonformal_v0/summary.md",
        "3e93152ef22e05a58650561111d4d3d96206391ea324a497993398eab3f8e8c0",
    ),
    (
        "analysis/recovery_action_branching_nonformal_v0/comparison.png",
        "d18310b4e15a9eb26bcc7884e8b56d9d6a90a4b231a999e7b8ed251dc4d902cb",
    ),
    (
        "analysis/recovery_branch_mechanism_diagnosis_v0/field_inventory.json",
        "f4096a9632494c01e2673105d0000df99c8cef8ffb1047130b1660a59cd3d88d",
    ),
    (
        "analysis/recovery_branch_mechanism_diagnosis_v0/branch_metrics.csv",
        "262eeaf7ca8a959695c6e20ab430bb75042c4cf5f7f47c5384059a79f8d913d7",
    ),
    (
        "analysis/recovery_branch_mechanism_diagnosis_v0/checkpoint_comparison.csv",
        "55b668d243bab63b6d63cc72a940f239adeeb2828c871ee795d9d34eb419f097",
    ),
    (
        "analysis/recovery_branch_mechanism_diagnosis_v0/mechanism_evidence.json",
        "e4540824fb0376e70ebeabfac6b95ea9d16ed33eb6bed17c26d65c882a47e881",
    ),
    (
        "analysis/recovery_branch_mechanism_diagnosis_v0/summary.md",
        "3e919721b2c7362aab67f83f518dc6b663deafe2017d77c3b1fdd702bcc2e0a3",
    ),
    (
        "analysis/recovery_branch_mechanism_diagnosis_v0/speed_ratio_trajectory.png",
        "6d0ff8ada82c39215d3ea34a8874d4dbdebe90838d2778fdd9dc5ae354700c9b",
    ),
    (
        "analysis/recovery_branch_mechanism_diagnosis_v0/action_geometry_trajectory.png",
        "df0a039d3442bf03aa826d21d38acded01016ffc3918713866cf6ff4bc88f562",
    ),
)


CLAIM_RESTRICTIONS = (
    "no_formal_safety_claim",
    "no_recovery_controller_implementation_claim",
    "no_staged_recovery_success_claim",
    "no_optimal_phase_logic_claim",
    "no_frozen_state_recoverability_claim",
    "no_benchmark_wide_effectiveness_claim",
    "no_hardware_validity_claim",
    "no_cross_domain_validation_claim",
    "no_deployment_readiness_claim",
)


RESERVED_FUTURE_EXPERIMENT_IDS = (
    "staged_recovery_instrumentation_validation_v0",
    "staged_recovery_single_transition_nonformal_v0",
    "staged_recovery_short_trace_nonformal_v0",
    "staged_recovery_predeclared_experiment_v0",
)


def _to_json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {
            field.name: _to_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    return value


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _to_json_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _manifest_payload(manifest: RecoveryArchitectureManifest) -> dict[str, object]:
    payload = _to_json_value(manifest)
    assert isinstance(payload, dict)
    payload.pop("canonical_payload_hash", None)
    payload["manifest_schema_version"] = ARCHITECTURE_MANIFEST_SCHEMA_VERSION
    payload["phase_order"] = [phase.value for phase in RecoveryPhase]
    payload["phase_ids"] = [phase.value for phase in RecoveryPhase]
    payload["allowed_transitions"] = [
        [
            (
                transition.source_phase.value
                if isinstance(transition.source_phase, RecoveryPhase)
                else str(transition.source_phase)
            ),
            (
                transition.destination_phase.value
                if isinstance(transition.destination_phase, RecoveryPhase)
                else str(transition.destination_phase)
            ),
        ]
        for transition in manifest.transitions
    ]
    payload["no_progress_contract_version"] = NO_PROGRESS_CONTRACT_VERSION
    payload["hysteresis_contract_version"] = HYSTERESIS_CONTRACT_VERSION
    return payload


def build_default_architecture_manifest() -> RecoveryArchitectureManifest:
    manifest = RecoveryArchitectureManifest(
        architecture_id=ARCHITECTURE_ID,
        version=ARCHITECTURE_VERSION,
        completed_date=ARCHITECTURE_COMPLETED_DATE,
        source_evidence_commits=SOURCE_EVIDENCE_COMMITS,
        source_artifact_hashes=SOURCE_ARTIFACT_HASHES,
        phase_contracts=default_phase_contracts(),
        transitions=default_transition_contracts(),
        forbidden_transitions=FORBIDDEN_TRANSITIONS,
        adverse_stop_priority=FROZEN_ADVERSE_STOP_PRIORITY,
        required_signal_ids=SIGNAL_IDS,
        unresolved_threshold_ids=UNRESOLVED_THRESHOLD_IDS,
        no_progress_contract=RecoveryNoProgressContract(),
        hysteresis_contract=RecoveryHysteresisContract(),
        action_law_status=RecoveryEvidenceSupport.NOT_YET_SPECIFIED,
        execution_authorization_status=EXECUTION_NOT_AUTHORIZED,
        execution_authorization_reason=EXECUTION_NOT_AUTHORIZED_REASON,
        claim_restrictions=CLAIM_RESTRICTIONS,
        reserved_future_experiment_ids=RESERVED_FUTURE_EXPERIMENT_IDS,
        canonical_payload_hash="",
    )
    return replace(manifest, canonical_payload_hash=canonical_sha256(_manifest_payload(manifest)))


def architecture_manifest_document(
    manifest: RecoveryArchitectureManifest | None = None,
) -> dict[str, object]:
    current = manifest or build_default_architecture_manifest()
    payload = _manifest_payload(current)
    payload["canonical_payload_hash"] = current.canonical_payload_hash
    return payload


def _phase_graph_has_route(
    phase: RecoveryPhase,
    edges: Mapping[RecoveryPhase, tuple[RecoveryPhase, ...]],
    targets: frozenset[RecoveryPhase],
) -> bool:
    pending = [phase]
    visited: set[RecoveryPhase] = set()
    while pending:
        current = pending.pop()
        if current in targets:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(edges.get(current, ()))
    return False


def _hysteresis_complete(contract: RecoveryHysteresisContract) -> bool:
    numeric_values = (
        contract.minimum_phase_dwell,
        contract.transition_cooldown,
        contract.consecutive_evidence_count,
        contract.phase_transition_budget,
        contract.repeated_cycle_window,
    )
    return (
        all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in numeric_values
        )
        and bool(contract.entry_threshold_id)
        and bool(contract.exit_threshold_id)
        and contract.entry_threshold_id != contract.exit_threshold_id
        and contract.requires_new_evidence
        and contract.preserves_raw_transition_reason
    )


def _no_progress_complete(contract: RecoveryNoProgressContract) -> bool:
    return (
        bool(contract.monitored_signal_id)
        and contract.desired_direction in {"increase", "decrease", "toward_zero"}
        and isinstance(contract.observation_window, int)
        and not isinstance(contract.observation_window, bool)
        and contract.observation_window >= 2
        and _is_finite_number(contract.minimum_meaningful_improvement)
        and float(contract.minimum_meaningful_improvement) > 0.0
        and isinstance(contract.phase_specific_dwell_limit, int)
        and not isinstance(contract.phase_specific_dwell_limit, bool)
        and contract.phase_specific_dwell_limit >= contract.observation_window
    )


def validate_architecture_contract(
    manifest: RecoveryArchitectureManifest,
) -> RecoveryArchitectureValidationReport:
    errors: list[str] = []
    unresolved: list[str] = list(manifest.unresolved_threshold_ids)
    phase_ids = tuple(contract.phase_id for contract in manifest.phase_contracts)
    if len(phase_ids) != len(set(phase_ids)):
        errors.append("duplicate_phase_id")
    if set(phase_ids) != set(RecoveryPhase):
        errors.append("phase_set_mismatch")
    for contract in manifest.phase_contracts:
        if not contract.purpose:
            errors.append(f"phase_missing_purpose:{contract.phase_id.value}")
        if not isinstance(contract.evidence_support, RecoveryEvidenceSupport):
            errors.append(f"phase_evidence_status_invalid:{contract.phase_id.value}")
        if not isinstance(contract.action_law_status, RecoveryEvidenceSupport):
            errors.append(f"phase_action_status_invalid:{contract.phase_id.value}")
        if not contract.terminal and not contract.required_signal_ids:
            errors.append(f"active_phase_missing_required_evidence:{contract.phase_id.value}")
        unknown_signals = set(contract.required_signal_ids) - set(SIGNAL_IDS)
        if unknown_signals:
            errors.append(f"phase_unknown_signal:{contract.phase_id.value}")

    endpoints = []
    transition_ids = []
    edges: dict[RecoveryPhase, list[RecoveryPhase]] = {}
    for transition in manifest.transitions:
        transition_ids.append(transition.transition_id)
        endpoints.append((transition.source_phase, transition.destination_phase))
        if transition.source_phase not in set(phase_ids) or transition.destination_phase not in set(
            phase_ids
        ):
            errors.append(f"unknown_transition_endpoint:{transition.transition_id}")
            continue
        edges.setdefault(transition.source_phase, []).append(transition.destination_phase)
        if not isinstance(transition.evidence_support, RecoveryEvidenceSupport):
            errors.append(f"transition_evidence_status_invalid:{transition.transition_id}")
        if not transition.adverse_stop_precedence:
            errors.append(f"transition_lacks_adverse_precedence:{transition.transition_id}")
        if transition.executable:
            if transition.threshold_status == RecoveryEvidenceSupport.NOT_YET_SPECIFIED:
                errors.append(f"executable_transition_missing_guard:{transition.transition_id}")
            if transition.hysteresis_required and not _hysteresis_complete(
                manifest.hysteresis_contract
            ):
                errors.append(f"executable_transition_missing_hysteresis:{transition.transition_id}")
            if not transition.requires_new_evidence:
                errors.append(f"executable_transition_reuses_evidence:{transition.transition_id}")

    if len(transition_ids) != len(set(transition_ids)) or len(endpoints) != len(
        set(endpoints)
    ):
        errors.append("duplicate_transition")
    forbidden_direct = {
        (RecoveryPhase.HAZARD_ARREST, RecoveryPhase.NOMINAL_HANDOFF),
        (RecoveryPhase.STABILIZATION_ASSESSMENT, RecoveryPhase.NOMINAL_HANDOFF),
        (RecoveryPhase.RADIAL_RECOMMITMENT, RecoveryPhase.NOMINAL_HANDOFF),
        (RecoveryPhase.TANGENTIAL_ALIGNMENT, RecoveryPhase.NOMINAL_HANDOFF),
    }
    if forbidden_direct.intersection(endpoints):
        errors.append("forbidden_direct_nominal_handoff")
    for terminal in (RecoveryPhase.NOMINAL_HANDOFF, RecoveryPhase.EXPLICIT_ABORT):
        if edges.get(terminal):
            errors.append(f"terminal_phase_has_outgoing_transition:{terminal.value}")

    normalized_edges = {key: tuple(value) for key, value in edges.items()}
    reachable = {RecoveryPhase.HAZARD_ARREST}
    pending = [RecoveryPhase.HAZARD_ARREST]
    while pending:
        current = pending.pop()
        for destination in normalized_edges.get(current, ()):
            if destination not in reachable:
                reachable.add(destination)
                pending.append(destination)
    if reachable != set(RecoveryPhase):
        errors.append("unreachable_phase")
    for phase in RecoveryPhase:
        if phase in {RecoveryPhase.NOMINAL_HANDOFF, RecoveryPhase.EXPLICIT_ABORT}:
            continue
        if not _phase_graph_has_route(
            phase,
            normalized_edges,
            frozenset({RecoveryPhase.RETREAT, RecoveryPhase.EXPLICIT_ABORT}),
        ):
            errors.append(f"active_phase_lacks_escape:{phase.value}")

    if manifest.adverse_stop_priority != FROZEN_ADVERSE_STOP_PRIORITY:
        errors.append("adverse_stop_priority_drift")
    if not manifest.required_signal_ids or set(manifest.required_signal_ids) != set(
        SIGNAL_IDS
    ):
        errors.append("required_signal_set_mismatch")
    if manifest.action_law_status != RecoveryEvidenceSupport.NOT_YET_SPECIFIED:
        errors.append("action_laws_must_remain_unresolved")
    if manifest.execution_authorization_status != EXECUTION_NOT_AUTHORIZED:
        errors.append("execution_must_remain_unauthorized")
    if manifest.execution_authorization_reason != EXECUTION_NOT_AUTHORIZED_REASON:
        errors.append("execution_reason_drift")
    if not manifest.unresolved_threshold_ids:
        errors.append("unresolved_thresholds_not_listed")
    if manifest.no_progress_contract.executable and not _no_progress_complete(
        manifest.no_progress_contract
    ):
        errors.append("executable_no_progress_contract_is_incomplete")
    if manifest.hysteresis_contract.executable and not _hysteresis_complete(
        manifest.hysteresis_contract
    ):
        errors.append("executable_hysteresis_contract_is_incomplete")
    if not manifest.hysteresis_contract.requires_new_evidence:
        errors.append("repeated_transitions_may_not_reuse_evidence")
    if not manifest.hysteresis_contract.preserves_raw_transition_reason:
        errors.append("raw_transition_reason_must_be_preserved")
    expected_hash = canonical_sha256(_manifest_payload(manifest))
    if manifest.canonical_payload_hash != expected_hash:
        errors.append("canonical_manifest_hash_mismatch")

    execution_authorized = (
        not errors
        and manifest.execution_authorization_status != EXECUTION_NOT_AUTHORIZED
        and not unresolved
        and manifest.action_law_status != RecoveryEvidenceSupport.NOT_YET_SPECIFIED
        and _no_progress_complete(manifest.no_progress_contract)
        and _hysteresis_complete(manifest.hysteresis_contract)
        and all(transition.executable for transition in manifest.transitions)
    )
    return RecoveryArchitectureValidationReport(
        structurally_valid=not errors,
        execution_authorized=execution_authorized,
        errors=tuple(sorted(set(errors))),
        unresolved_items=tuple(sorted(set(unresolved))),
    )


def validate_architecture_manifest_document(
    document: Mapping[str, object],
) -> tuple[str, ...]:
    errors: list[str] = []
    required = {
        "manifest_schema_version",
        "architecture_id",
        "version",
        "completed_date",
        "phase_ids",
        "phase_order",
        "allowed_transitions",
        "forbidden_transitions",
        "adverse_stop_priority",
        "required_signal_ids",
        "unresolved_threshold_ids",
        "action_law_status",
        "execution_authorization_status",
        "execution_authorization_reason",
        "claim_restrictions",
        "reserved_future_experiment_ids",
        "canonical_payload_hash",
    }
    missing = sorted(required - set(document))
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")
    if document.get("manifest_schema_version") != ARCHITECTURE_MANIFEST_SCHEMA_VERSION:
        errors.append("manifest_schema_version_mismatch")
    if document.get("architecture_id") != ARCHITECTURE_ID:
        errors.append("architecture_id_mismatch")
    if document.get("execution_authorization_status") != EXECUTION_NOT_AUTHORIZED:
        errors.append("execution_status_mismatch")
    if document.get("adverse_stop_priority") != list(FROZEN_ADVERSE_STOP_PRIORITY):
        errors.append("adverse_stop_priority_drift")
    payload = dict(document)
    stored_hash = payload.pop("canonical_payload_hash", None)
    if not isinstance(stored_hash, str) or canonical_sha256(payload) != stored_hash:
        errors.append("canonical_manifest_hash_mismatch")
    return tuple(sorted(set(errors)))


def evaluate_no_progress(
    samples: Sequence[RecoverySignalValue],
    contract: RecoveryNoProgressContract,
    *,
    timed_out: bool = False,
) -> RecoveryNoProgressResult:
    del timed_out  # Timeout is deliberately not evidence of progress or stalling.
    if not _no_progress_complete(contract):
        return RecoveryNoProgressResult(
            status=RecoveryNoProgressStatus.NOT_EVALUATED,
            reason="no_progress_contract_is_not_fully_frozen",
            signal_id=contract.monitored_signal_id,
            sample_count=len(samples),
            measured_improvement=None,
        )
    window = contract.observation_window
    assert window is not None
    selected = tuple(samples[-window:])
    if len(selected) < window:
        return RecoveryNoProgressResult(
            status=RecoveryNoProgressStatus.NOT_EVALUATED,
            reason="insufficient_observation_window",
            signal_id=contract.monitored_signal_id,
            sample_count=len(selected),
            measured_improvement=None,
        )
    if any(sample.status == RecoverySignalStatus.INVALID for sample in selected):
        return RecoveryNoProgressResult(
            status=RecoveryNoProgressStatus.INVALID,
            reason="progress_signal_contains_invalid_evidence",
            signal_id=contract.monitored_signal_id,
            sample_count=len(selected),
            measured_improvement=None,
        )
    if any(not sample.is_available or not _is_finite_number(sample.value) for sample in selected):
        return RecoveryNoProgressResult(
            status=RecoveryNoProgressStatus.NOT_EVALUATED,
            reason="progress_signal_is_missing_or_unsupported",
            signal_id=contract.monitored_signal_id,
            sample_count=len(selected),
            measured_improvement=None,
        )
    first = float(selected[0].value)
    last = float(selected[-1].value)
    if contract.desired_direction == "increase":
        improvement = last - first
    elif contract.desired_direction == "decrease":
        improvement = first - last
    else:
        improvement = abs(first) - abs(last)
    minimum = float(contract.minimum_meaningful_improvement)
    if improvement >= minimum:
        status = RecoveryNoProgressStatus.PROGRESSING
        reason = "minimum_meaningful_improvement_reached"
    elif improvement <= -minimum:
        status = RecoveryNoProgressStatus.REGRESSING
        reason = "progress_signal_moved_opposite_the_declared_direction"
    else:
        status = RecoveryNoProgressStatus.STALLED
        reason = "improvement_below_frozen_minimum"
    return RecoveryNoProgressResult(
        status=status,
        reason=reason,
        signal_id=contract.monitored_signal_id,
        sample_count=len(selected),
        measured_improvement=improvement,
    )


def _triggered_boolean(signal: RecoverySignalValue) -> bool | None:
    if signal.status == RecoverySignalStatus.INVALID:
        return None
    if not signal.is_available or type(signal.value) is not bool:
        return None
    return signal.value


def select_adverse_stop(observation: StagedRecoveryObservation) -> str | None:
    signal_by_stop = {
        "invalid_simulation": ("simulation_validity", False),
        "invalid_recovery_evaluation": ("recovery_evaluation_validity", False),
        "overspeed": ("overspeed_status", True),
        "instability": ("instability_status", True),
        "unsafe_state": ("unsafe_state_status", True),
        "action_rejected": ("action_rejection_status", True),
        "explicit_abort": ("explicit_abort_requested", True),
        "recovery_success": ("recovery_success_v0", True),
        "recovery_horizon_exhausted": ("recovery_horizon_exhausted", True),
        "total_horizon_exhausted": ("total_horizon_exhausted", True),
    }
    for stop in FROZEN_ADVERSE_STOP_PRIORITY:
        if stop not in signal_by_stop:
            continue
        signal_id, trigger_value = signal_by_stop[stop]
        signal = observation.signal(signal_id)
        if signal.status == RecoverySignalStatus.INVALID:
            if stop == "invalid_simulation":
                return "invalid_simulation"
            return "invalid_recovery_evaluation"
        value = _triggered_boolean(signal)
        if value is trigger_value:
            if stop == "recovery_success":
                handoff = evaluate_nominal_handoff(observation)
                return "recovery_success" if handoff.status == "allowed" else None
            return stop
    return None


def evaluate_nominal_handoff(
    observation: StagedRecoveryObservation,
) -> RecoveryPhaseDecision:
    required_true = (
        "simulation_validity",
        "recovery_evaluation_validity",
        "target_radius_crossing",
        "phase34_compatible_recoverability",
        "recovery_success_v0",
        "handoff_readiness",
    )
    required_false = (
        "overspeed_status",
        "instability_status",
        "unsafe_state_status",
        "action_rejection_status",
        "explicit_abort_requested",
    )
    missing: list[str] = []
    invalid: list[str] = []
    unfavorable: list[str] = []
    for signal_id, desired in tuple((item, True) for item in required_true) + tuple(
        (item, False) for item in required_false
    ):
        signal = observation.signal(signal_id)
        if signal.status == RecoverySignalStatus.INVALID:
            invalid.append(signal_id)
        elif not signal.is_available or type(signal.value) is not bool:
            missing.append(signal_id)
        elif signal.value is not desired:
            unfavorable.append(signal_id)
    if invalid:
        return RecoveryPhaseDecision(
            source_phase=RecoveryPhase.RECOVERABILITY_VERIFICATION,
            destination_phase=None,
            status="adverse",
            reason="invalid_handoff_evidence",
            invalid_signal_ids=tuple(sorted(invalid)),
            adverse_stop="invalid_recovery_evaluation",
        )
    adverse = select_adverse_stop_without_success(observation)
    if adverse is not None:
        return RecoveryPhaseDecision(
            source_phase=RecoveryPhase.RECOVERABILITY_VERIFICATION,
            destination_phase=None,
            status="adverse",
            reason="higher_priority_adverse_stop_is_active",
            adverse_stop=adverse,
        )
    if missing:
        return RecoveryPhaseDecision(
            source_phase=RecoveryPhase.RECOVERABILITY_VERIFICATION,
            destination_phase=None,
            status="blocked",
            reason="required_handoff_evidence_is_missing",
            missing_signal_ids=tuple(sorted(missing)),
        )
    if unfavorable:
        return RecoveryPhaseDecision(
            source_phase=RecoveryPhase.RECOVERABILITY_VERIFICATION,
            destination_phase=None,
            status="blocked",
            reason="recovery_predicate_or_handoff_readiness_is_not_satisfied",
            missing_signal_ids=tuple(sorted(unfavorable)),
        )
    return RecoveryPhaseDecision(
        source_phase=RecoveryPhase.RECOVERABILITY_VERIFICATION,
        destination_phase=RecoveryPhase.NOMINAL_HANDOFF,
        status="allowed",
        reason="complete_valid_recovery_and_handoff_evidence",
    )


def select_adverse_stop_without_success(
    observation: StagedRecoveryObservation,
) -> str | None:
    checks = (
        ("invalid_simulation", "simulation_validity", False),
        ("invalid_recovery_evaluation", "recovery_evaluation_validity", False),
        ("overspeed", "overspeed_status", True),
        ("instability", "instability_status", True),
        ("unsafe_state", "unsafe_state_status", True),
        ("action_rejected", "action_rejection_status", True),
        ("explicit_abort", "explicit_abort_requested", True),
    )
    for stop, signal_id, trigger in checks:
        signal = observation.signal(signal_id)
        if signal.status == RecoverySignalStatus.INVALID:
            return (
                "invalid_simulation"
                if signal_id == "simulation_validity"
                else "invalid_recovery_evaluation"
            )
        value = _triggered_boolean(signal)
        if value is trigger:
            return stop
    return None


def evaluate_transition_evidence(
    transition: RecoveryTransitionContract,
    observation: StagedRecoveryObservation,
) -> RecoveryPhaseDecision:
    if observation.current_phase != transition.source_phase:
        raise RecoveryContractError("observation phase does not match transition source")
    adverse = select_adverse_stop_without_success(observation)
    if adverse is not None:
        return RecoveryPhaseDecision(
            source_phase=transition.source_phase,
            destination_phase=None,
            status="adverse",
            reason="adverse_stop_precedes_phase_progression",
            adverse_stop=adverse,
        )
    missing = []
    invalid = []
    for signal_id in transition.required_signal_ids:
        signal = observation.signal(signal_id)
        if signal.status == RecoverySignalStatus.INVALID:
            invalid.append(signal_id)
        elif not signal.is_available:
            missing.append(signal_id)
    if invalid:
        return RecoveryPhaseDecision(
            source_phase=transition.source_phase,
            destination_phase=None,
            status="adverse",
            reason="transition_evidence_is_invalid",
            invalid_signal_ids=tuple(sorted(invalid)),
            adverse_stop="invalid_recovery_evaluation",
        )
    if missing:
        return RecoveryPhaseDecision(
            source_phase=transition.source_phase,
            destination_phase=None,
            status="blocked",
            reason="transition_evidence_is_missing",
            missing_signal_ids=tuple(sorted(missing)),
        )
    if transition.destination_phase == RecoveryPhase.NOMINAL_HANDOFF:
        return evaluate_nominal_handoff(observation)
    if not transition.executable:
        return RecoveryPhaseDecision(
            source_phase=transition.source_phase,
            destination_phase=transition.destination_phase,
            status="architecture_only",
            reason="transition_guard_or_action_contract_is_not_frozen_for_execution",
        )
    return RecoveryPhaseDecision(
        source_phase=transition.source_phase,
        destination_phase=transition.destination_phase,
        status="allowed",
        reason="complete_valid_transition_evidence",
    )


def _source(
    artifact: str, field_or_finding: str, measured_support: str
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "artifact": artifact,
                "field_or_finding": field_or_finding,
                "measured_support": measured_support,
            }.items()
        )
    )


def build_evidence_traceability_items() -> tuple[EvidenceTraceabilityItem, ...]:
    diagnosis_summary = "analysis/recovery_branch_mechanism_diagnosis_v0/summary.md"
    mechanism_evidence = (
        "analysis/recovery_branch_mechanism_diagnosis_v0/mechanism_evidence.json"
    )
    result_summary = "analysis/recovery_action_branching_nonformal_v0/summary.md"
    evaluator_source = "runtime_assurance/recovery_evaluators.py"
    stop_source = "runtime_assurance/recovery_stop_conditions.py"
    items: list[EvidenceTraceabilityItem] = []

    phase_support = {
        RecoveryPhase.HAZARD_ARREST: RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
        RecoveryPhase.STABILIZATION_ASSESSMENT: RecoveryEvidenceSupport.CONSISTENT_WITH_EVIDENCE,
        RecoveryPhase.RADIAL_RECOMMITMENT: RecoveryEvidenceSupport.PARTIALLY_SUPPORTED,
        RecoveryPhase.TANGENTIAL_ALIGNMENT: RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
        RecoveryPhase.CROSSING_PREPARATION: RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
        RecoveryPhase.RECOVERABILITY_VERIFICATION: RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
        RecoveryPhase.NOMINAL_HANDOFF: RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
        RecoveryPhase.RETREAT: RecoveryEvidenceSupport.NOT_YET_SPECIFIED,
        RecoveryPhase.EXPLICIT_ABORT: RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
    }
    phase_findings = {
        RecoveryPhase.HAZARD_ARREST: "all three physical branches avoided overspeed but did not recover",
        RecoveryPhase.STABILIZATION_ASSESSMENT: "absence of overspeed did not establish a usable recovery state",
        RecoveryPhase.RADIAL_RECOMMITMENT: "all tested physical branches failed to restore crossing",
        RecoveryPhase.TANGENTIAL_ALIGNMENT: "tangential correction improved one component without geometric recovery",
        RecoveryPhase.CROSSING_PREPARATION: "no tested physical branch reached target-radius crossing",
        RecoveryPhase.RECOVERABILITY_VERIFICATION: "crossing, recoverability, and Recovery Success v0 remained separate outcomes",
        RecoveryPhase.NOMINAL_HANDOFF: "none of the physical branches satisfied the recovery predicate",
        RecoveryPhase.RETREAT: "retreat target and predicate were not tested or instrumented",
        RecoveryPhase.EXPLICIT_ABORT: "explicit abort executed zero post-branch transitions and was not task recovery",
    }
    for phase in RecoveryPhase:
        items.append(
            EvidenceTraceabilityItem(
                item_id=f"phase.{phase.value}",
                description=_PHASE_PURPOSES[phase],
                evidence_support=phase_support[phase],
                sources=(
                    _source(
                        diagnosis_summary if phase != RecoveryPhase.EXPLICIT_ABORT else result_summary,
                        "branch-by-branch and cross-branch findings",
                        phase_findings[phase],
                    ),
                ),
                limitation=(
                    "The one-case evidence does not determine a numerical phase guard or action law."
                ),
                unresolved_dependency=(
                    None
                    if phase in {RecoveryPhase.RECOVERABILITY_VERIFICATION, RecoveryPhase.EXPLICIT_ABORT}
                    else "freeze phase-specific evidence guards before execution"
                ),
            )
        )

    unavailable_per_step = {
        "radius",
        "position_x",
        "position_y",
        "velocity_x",
        "velocity_y",
        "target_radius_error",
        "radial_velocity",
        "tangential_velocity",
        "radial_velocity_ratio",
        "tangential_velocity_error_ratio",
        "orbital_energy_or_proxy",
        "recoverability_component_vector",
        "radial_progress",
        "radial_progress_direction",
        "radial_progress_rate",
        "tangential_progress",
        "crossing_progress",
        "predicted_crossing_direction",
        "crossing_proximity",
        "energy_change_direction",
    }
    existing_runtime = {
        "simulation_validity",
        "recovery_evaluation_validity",
        "overspeed_status",
        "instability_status",
        "unsafe_state_status",
        "target_radius_crossing",
        "phase34_compatible_recoverability",
        "recovery_success_v0",
        "final_veto_decision",
        "predicted_speed_ratio",
        "realized_speed_ratio",
        "proposed_action",
        "executed_action",
    }
    for signal_id in SIGNAL_IDS:
        if signal_id in unavailable_per_step:
            support = RecoveryEvidenceSupport.DIRECTLY_SUPPORTED
            finding = "per-step physical quantity was not available in the published decision log"
            unresolved_dependency = f"instrument and validate per-step {signal_id}"
        elif signal_id in existing_runtime:
            support = RecoveryEvidenceSupport.DIRECTLY_SUPPORTED
            finding = "existing evaluator or result semantics define this status-bearing field"
            unresolved_dependency = None
        else:
            support = RecoveryEvidenceSupport.DESIGN_HYPOTHESIS
            finding = "required by the proposed staged decision boundary, not measured as a phase signal"
            unresolved_dependency = f"freeze semantics and instrumentation for {signal_id}"
        items.append(
            EvidenceTraceabilityItem(
                item_id=f"signal.{signal_id}",
                description=f"Status-bearing staged recovery signal: {signal_id}.",
                evidence_support=support,
                sources=(
                    _source(
                        diagnosis_summary,
                        "missing instrumentation and measured field inventory",
                        finding,
                    ),
                ),
                limitation="Unknown values must remain not_evaluated or invalid rather than zero or false.",
                unresolved_dependency=unresolved_dependency,
            )
        )

    for transition in default_transition_contracts():
        items.append(
            EvidenceTraceabilityItem(
                item_id=f"transition.{transition.transition_id}",
                description=(
                    f"Architecture-only transition from {transition.source_phase.value} "
                    f"to {transition.destination_phase.value}."
                ),
                evidence_support=transition.evidence_support,
                sources=(
                    _source(
                        mechanism_evidence,
                        "H_missing_phase_switching",
                        "single-mode responses did not recover; staged switching remains a hypothesis",
                    ),
                ),
                limitation="The measured run did not execute this phase transition.",
                unresolved_dependency=(
                    "freeze complete numerical guards and hysteresis before execution"
                ),
            )
        )

    items.extend(
        (
            EvidenceTraceabilityItem(
                item_id="guard.adverse_stop_priority",
                description="Adverse terminal evidence overrides phase progression.",
                evidence_support=RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
                sources=(
                    _source(
                        stop_source,
                        "FROZEN_TERMINATION_PRIORITY",
                        "existing runtime contract defines the exact adverse priority",
                    ),
                ),
                limitation="Priority preservation does not validate phase actions.",
                unresolved_dependency=None,
            ),
            EvidenceTraceabilityItem(
                item_id="guard.phase34_inclusive_recoverability",
                description="Nominal handoff requires the existing inclusive Phase34 predicate.",
                evidence_support=RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
                sources=(
                    _source(
                        evaluator_source,
                        "evaluate_phase34_compatible_recoverability",
                        "inclusive component thresholds are implemented and tested",
                    ),
                ),
                limitation="This predicate is simulator-defined and does not establish formal safety.",
                unresolved_dependency=None,
            ),
            EvidenceTraceabilityItem(
                item_id="no_progress.explicit_configuration",
                description="No-progress evaluation requires one named signal and frozen window, improvement, and dwell settings.",
                evidence_support=RecoveryEvidenceSupport.PARTIALLY_SUPPORTED,
                sources=(
                    _source(
                        diagnosis_summary,
                        "common horizon exhaustion and missing per-step state",
                        "all physical branches exhausted the horizon, but timeout alone did not diagnose progress",
                    ),
                ),
                limitation="Published artifacts cannot calibrate a numerical no-progress threshold.",
                unresolved_dependency="freeze phase-specific no-progress configurations",
            ),
            EvidenceTraceabilityItem(
                item_id="hysteresis.phase_chatter_control",
                description="Executable cycles require dwell, separate entry/exit guards, cooldown, budget, and cycle detection.",
                evidence_support=RecoveryEvidenceSupport.DESIGN_HYPOTHESIS,
                sources=(
                    _source(
                        mechanism_evidence,
                        "H_missing_phase_switching",
                        "phase switching is motivated but was not measured",
                    ),
                ),
                limitation="No measured phase transitions exist from which to calibrate hysteresis.",
                unresolved_dependency="freeze all hysteresis parameters before execution",
            ),
            EvidenceTraceabilityItem(
                item_id="claim.hazard_arrest_not_task_recovery",
                description="Hazard arrest must remain separate from crossing, recoverability, and task recovery.",
                evidence_support=RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
                sources=(
                    _source(
                        result_summary,
                        "hazard and recovery outcomes",
                        "all physical branches avoided overspeed and none recovered",
                    ),
                ),
                limitation="Evidence is one-case and simulator-level.",
                unresolved_dependency=None,
            ),
            EvidenceTraceabilityItem(
                item_id="claim.no_architecture_effectiveness",
                description="The architecture contract cannot be described as a validated recovery policy.",
                evidence_support=RecoveryEvidenceSupport.DIRECTLY_SUPPORTED,
                sources=(
                    _source(
                        diagnosis_summary,
                        "claim restrictions",
                        "the evidence does not determine whether a proposed future policy will work",
                    ),
                ),
                limitation="No staged trajectory has been executed.",
                unresolved_dependency="complete the staged evidence sequence before any effectiveness claim",
            ),
        )
    )
    return tuple(sorted(items, key=lambda item: item.item_id))


def evidence_traceability_document() -> dict[str, object]:
    items = build_evidence_traceability_items()
    payload: dict[str, object] = {
        "architecture_id": ARCHITECTURE_ID,
        "completed_date": ARCHITECTURE_COMPLETED_DATE,
        "evidence_traceability_schema_version": EVIDENCE_TRACEABILITY_SCHEMA_VERSION,
        "items": [_to_json_value(item) for item in items],
        "source_artifact_hashes": [list(item) for item in SOURCE_ARTIFACT_HASHES],
    }
    payload["canonical_payload_hash"] = canonical_sha256(payload)
    return payload


DEFAULT_ARCHITECTURE_MANIFEST = build_default_architecture_manifest()


__all__ = [
    "ARCHITECTURE_COMPLETED_DATE",
    "ARCHITECTURE_ID",
    "ARCHITECTURE_MANIFEST_SCHEMA_VERSION",
    "ARCHITECTURE_VERSION",
    "CLAIM_RESTRICTIONS",
    "DEFAULT_ARCHITECTURE_MANIFEST",
    "EXECUTION_NOT_AUTHORIZED",
    "EXECUTION_NOT_AUTHORIZED_REASON",
    "FROZEN_ADVERSE_STOP_PRIORITY",
    "FORBIDDEN_TRANSITIONS",
    "HYSTERESIS_CONTRACT_VERSION",
    "NO_PROGRESS_CONTRACT_VERSION",
    "PHASE34_RADIAL_VELOCITY_RATIO_MAX",
    "PHASE34_RADIUS_ERROR_RATIO_MAX",
    "PHASE34_TANGENTIAL_VELOCITY_ERROR_RATIO_MAX",
    "RecoveryArchitectureManifest",
    "RecoveryArchitectureValidationReport",
    "RecoveryContractError",
    "RecoveryEvidenceSupport",
    "RecoveryHysteresisContract",
    "RecoveryNoProgressContract",
    "RecoveryNoProgressResult",
    "RecoveryNoProgressStatus",
    "RecoveryPhase",
    "RecoveryPhaseContract",
    "RecoveryPhaseDecision",
    "RecoverySignalStatus",
    "RecoverySignalValue",
    "RecoveryTransitionContract",
    "RecoveryTransitionReason",
    "SIGNAL_IDS",
    "SOURCE_ARTIFACT_HASHES",
    "SOURCE_EVIDENCE_COMMITS",
    "StagedRecoveryObservation",
    "UNRESOLVED_THRESHOLD_IDS",
    "architecture_manifest_document",
    "build_default_architecture_manifest",
    "build_evidence_traceability_items",
    "canonical_json_bytes",
    "canonical_sha256",
    "default_phase_contracts",
    "default_transition_contracts",
    "evaluate_no_progress",
    "evaluate_nominal_handoff",
    "evaluate_transition_evidence",
    "evidence_traceability_document",
    "replace_signals",
    "select_adverse_stop",
    "unavailable_signals",
    "validate_architecture_contract",
    "validate_architecture_manifest_document",
]
