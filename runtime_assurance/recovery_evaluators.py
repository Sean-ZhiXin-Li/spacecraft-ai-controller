from __future__ import annotations

import math
from dataclasses import dataclass


EVALUATION_TRIGGERED = "triggered"
EVALUATION_CLEAR = "clear"
EVALUATION_NOT_EVALUATED = "not_evaluated"
EVALUATION_INVALID = "invalid"
EVALUATION_STATUSES = frozenset(
    {
        EVALUATION_TRIGGERED,
        EVALUATION_CLEAR,
        EVALUATION_NOT_EVALUATED,
        EVALUATION_INVALID,
    }
)

RECOVERY_SUCCESS_EVALUATOR_ID = "recovery_success_v0"
PHASE34_RECOVERABILITY_EVALUATOR_ID = (
    "phase34_compatible_recoverability_predicate_v0"
)
INSTABILITY_EVALUATOR_ID = "repository_supported_instability_v0"
UNSAFE_STATE_EVALUATOR_ID = "repository_supported_unsafe_state_v0"

FROZEN_RECOVERY_HORIZON = 10_000

# Source: Phase21 constants used by Phase34 recoverable_state(). Phase34 applies
# inclusive component comparisons; this adapter deliberately preserves them.
PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX = 2.5e-3
PHASE34_RECOVERABLE_VR_RATIO_MAX = 2.0e-2
PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX = 2.5e-1

# Source: Final Veto result normalization in run_final_veto_ablation.py.
SUPPORTED_INSTABILITY_TERMINATION_REASONS = frozenset(
    {"out_range", "too_close", "radial_stall"}
)

_DetailValue = str | int | float | bool | None
_Details = tuple[tuple[str, _DetailValue], ...]


@dataclass(frozen=True, slots=True)
class RecoveryEvaluationResult:
    evaluator_id: str
    status: str
    triggered: bool | None
    reason: str
    evidence_level: str
    evaluated_step: int | None
    required_inputs_present: bool
    details: _Details

    def __post_init__(self) -> None:
        if not self.evaluator_id:
            raise ValueError("evaluator_id must be nonempty")
        if self.status not in EVALUATION_STATUSES:
            raise ValueError(f"unsupported evaluation status: {self.status!r}")
        expected_triggered = {
            EVALUATION_TRIGGERED: True,
            EVALUATION_CLEAR: False,
            EVALUATION_NOT_EVALUATED: None,
            EVALUATION_INVALID: None,
        }[self.status]
        if self.triggered is not expected_triggered:
            raise ValueError("triggered value does not match evaluation status")
        if self.status in {EVALUATION_TRIGGERED, EVALUATION_CLEAR} and not (
            self.required_inputs_present
        ):
            raise ValueError("triggered and clear results require complete inputs")
        if not self.reason:
            raise ValueError("reason must be nonempty")
        if not self.evidence_level:
            raise ValueError("evidence_level must be nonempty")
        if self.evaluated_step is not None and (
            isinstance(self.evaluated_step, bool)
            or not isinstance(self.evaluated_step, int)
            or self.evaluated_step < 0
        ):
            raise ValueError("evaluated_step must be a nonnegative integer or None")
        detail_keys = tuple(key for key, _ in self.details)
        if detail_keys != tuple(sorted(detail_keys)) or len(detail_keys) != len(
            set(detail_keys)
        ):
            raise ValueError("details must use unique keys in sorted order")
        for key, value in self.details:
            if not isinstance(key, str) or not key:
                raise ValueError("detail keys must be nonempty strings")
            if value is not None and not isinstance(
                value, (str, int, float, bool)
            ):
                raise TypeError("detail values must be JSON scalar values")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("detail values must be finite")

    def detail(self, key: str) -> _DetailValue:
        for current_key, value in self.details:
            if current_key == key:
                return value
        raise KeyError(key)


def _details(**values: _DetailValue) -> _Details:
    return tuple(sorted(values.items()))


def _result(
    evaluator_id: str,
    status: str,
    reason: str,
    *,
    evidence_level: str,
    evaluated_step: int | None,
    required_inputs_present: bool,
    details: _Details = (),
) -> RecoveryEvaluationResult:
    triggered = {
        EVALUATION_TRIGGERED: True,
        EVALUATION_CLEAR: False,
        EVALUATION_NOT_EVALUATED: None,
        EVALUATION_INVALID: None,
    }[status]
    return RecoveryEvaluationResult(
        evaluator_id=evaluator_id,
        status=status,
        triggered=triggered,
        reason=reason,
        evidence_level=evidence_level,
        evaluated_step=evaluated_step,
        required_inputs_present=required_inputs_present,
        details=details,
    )


def _step_is_valid(value: object) -> bool:
    return value is None or (
        not isinstance(value, bool) and isinstance(value, int) and value >= 0
    )


def _integer_is_valid(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _boolean_is_valid(value: object) -> bool:
    return type(value) is bool


def _finite_number_is_valid(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _invalid_step_result(
    evaluator_id: str, *, evidence_level: str
) -> RecoveryEvaluationResult:
    return _result(
        evaluator_id,
        EVALUATION_INVALID,
        "evaluated_step_is_malformed",
        evidence_level=evidence_level,
        evaluated_step=None,
        required_inputs_present=False,
        details=_details(invalid_field="evaluated_step"),
    )


def evaluate_phase34_compatible_recoverability(
    *,
    r_error_ratio: float | None,
    vr_ratio: float | None,
    vt_error_ratio: float | None,
    evaluated_step: int | None = None,
    evidence_level: str = "measured",
) -> RecoveryEvaluationResult:
    if not _step_is_valid(evaluated_step):
        return _invalid_step_result(
            PHASE34_RECOVERABILITY_EVALUATOR_ID,
            evidence_level=evidence_level,
        )

    inputs = {
        "r_error_ratio": r_error_ratio,
        "vr_ratio": vr_ratio,
        "vt_error_ratio": vt_error_ratio,
    }
    malformed = tuple(
        name
        for name, value in inputs.items()
        if value is not None and not _finite_number_is_valid(value)
    )
    if malformed:
        return _result(
            PHASE34_RECOVERABILITY_EVALUATOR_ID,
            EVALUATION_INVALID,
            "recoverability_component_is_malformed_or_nonfinite",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
            details=_details(invalid_fields=",".join(sorted(malformed))),
        )

    missing = tuple(name for name, value in inputs.items() if value is None)
    if missing:
        return _result(
            PHASE34_RECOVERABILITY_EVALUATOR_ID,
            EVALUATION_NOT_EVALUATED,
            "recoverability_components_are_missing",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=False,
            details=_details(missing_fields=",".join(sorted(missing))),
        )

    r_value = float(r_error_ratio)
    vr_value = float(vr_ratio)
    vt_value = float(vt_error_ratio)
    is_recoverable = (
        abs(r_value) <= PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX
        and abs(vr_value) <= PHASE34_RECOVERABLE_VR_RATIO_MAX
        and abs(vt_value) <= PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX
    )
    return _result(
        PHASE34_RECOVERABILITY_EVALUATOR_ID,
        EVALUATION_TRIGGERED if is_recoverable else EVALUATION_CLEAR,
        (
            "all_phase34_component_bounds_are_satisfied"
            if is_recoverable
            else "at_least_one_phase34_component_bound_is_not_satisfied"
        ),
        evidence_level=evidence_level,
        evaluated_step=evaluated_step,
        required_inputs_present=True,
        details=_details(
            comparator="inclusive_absolute_component_bounds",
            r_error_ratio=r_value,
            r_error_ratio_max=PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
            vr_ratio=vr_value,
            vr_ratio_max=PHASE34_RECOVERABLE_VR_RATIO_MAX,
            vt_error_ratio=vt_value,
            vt_error_ratio_max=PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
        ),
    )


def evaluate_recovery_success_v0(
    *,
    declared_overspeed_occurred: bool | None,
    simulation_valid: bool | None,
    recovery_evaluation_valid: bool | None,
    target_radius_crossing: bool | None,
    phase34_compatible_recoverable_crossing: bool | None,
    branch_transition_count: int | None,
    recovery_horizon: int | None,
    branch_step: int | None,
    crossing_step: int | None,
    explicit_abort: bool | None,
    action_rejected: bool | None,
    evaluated_step: int | None = None,
    evidence_level: str = "measured",
) -> RecoveryEvaluationResult:
    if not _step_is_valid(evaluated_step):
        return _invalid_step_result(
            RECOVERY_SUCCESS_EVALUATOR_ID,
            evidence_level=evidence_level,
        )

    boolean_inputs = {
        "action_rejected": action_rejected,
        "declared_overspeed_occurred": declared_overspeed_occurred,
        "explicit_abort": explicit_abort,
        "phase34_compatible_recoverable_crossing": (
            phase34_compatible_recoverable_crossing
        ),
        "recovery_evaluation_valid": recovery_evaluation_valid,
        "simulation_valid": simulation_valid,
        "target_radius_crossing": target_radius_crossing,
    }
    integer_inputs = {
        "branch_step": branch_step,
        "branch_transition_count": branch_transition_count,
        "recovery_horizon": recovery_horizon,
    }
    malformed = [
        name
        for name, value in boolean_inputs.items()
        if value is not None and not _boolean_is_valid(value)
    ]
    malformed.extend(
        name
        for name, value in integer_inputs.items()
        if value is not None and not _integer_is_valid(value)
    )
    if crossing_step is not None and not _integer_is_valid(crossing_step):
        malformed.append("crossing_step")
    if malformed:
        return _result(
            RECOVERY_SUCCESS_EVALUATOR_ID,
            EVALUATION_INVALID,
            "recovery_success_input_is_malformed_or_nonfinite",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
            details=_details(invalid_fields=",".join(sorted(malformed))),
        )

    missing = [name for name, value in boolean_inputs.items() if value is None]
    missing.extend(name for name, value in integer_inputs.items() if value is None)
    if target_radius_crossing is True and crossing_step is None:
        missing.append("crossing_step")
    if missing:
        return _result(
            RECOVERY_SUCCESS_EVALUATOR_ID,
            EVALUATION_NOT_EVALUATED,
            "recovery_success_evidence_is_missing",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=False,
            details=_details(missing_fields=",".join(sorted(missing))),
        )

    if recovery_horizon != FROZEN_RECOVERY_HORIZON:
        return _result(
            RECOVERY_SUCCESS_EVALUATOR_ID,
            EVALUATION_INVALID,
            "recovery_horizon_does_not_match_frozen_contract",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
            details=_details(
                expected_recovery_horizon=FROZEN_RECOVERY_HORIZON,
                recovery_horizon=recovery_horizon,
            ),
        )

    if explicit_abort and action_rejected:
        return _result(
            RECOVERY_SUCCESS_EVALUATOR_ID,
            EVALUATION_INVALID,
            "explicit_abort_and_action_rejection_are_contradictory",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
        )
    if target_radius_crossing is False and crossing_step is not None:
        return _result(
            RECOVERY_SUCCESS_EVALUATOR_ID,
            EVALUATION_INVALID,
            "crossing_step_present_without_target_radius_crossing",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
            details=_details(crossing_step=crossing_step),
        )
    if (
        phase34_compatible_recoverable_crossing is True
        and target_radius_crossing is False
    ):
        return _result(
            RECOVERY_SUCCESS_EVALUATOR_ID,
            EVALUATION_INVALID,
            "recoverable_crossing_present_without_target_radius_crossing",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
        )

    crossing_at_or_after_branch = bool(
        target_radius_crossing
        and crossing_step is not None
        and crossing_step >= branch_step
    )
    within_horizon = branch_transition_count <= recovery_horizon
    details = _details(
        action_rejected=action_rejected,
        branch_step=branch_step,
        branch_transition_count=branch_transition_count,
        crossing_at_or_after_branch=crossing_at_or_after_branch,
        crossing_step=crossing_step,
        declared_overspeed_occurred=declared_overspeed_occurred,
        explicit_abort=explicit_abort,
        phase34_compatible_recoverable_crossing=(
            phase34_compatible_recoverable_crossing
        ),
        recovery_evaluation_valid=recovery_evaluation_valid,
        recovery_horizon=recovery_horizon,
        simulation_valid=simulation_valid,
        target_radius_crossing=target_radius_crossing,
        within_recovery_horizon=within_horizon,
    )

    negative_reason = None
    if explicit_abort:
        negative_reason = "explicit_abort_is_not_recovery_success"
    elif action_rejected:
        negative_reason = "action_rejection_is_not_recovery_success"
    elif not simulation_valid:
        negative_reason = "simulation_is_invalid"
    elif not recovery_evaluation_valid:
        negative_reason = "recovery_evaluation_is_invalid"
    elif declared_overspeed_occurred:
        negative_reason = "declared_overspeed_hazard_occurred"
    elif not target_radius_crossing:
        negative_reason = "target_radius_crossing_did_not_occur"
    elif not crossing_at_or_after_branch:
        negative_reason = "crossing_occurred_before_branch_point"
    elif not phase34_compatible_recoverable_crossing:
        negative_reason = "crossing_did_not_meet_phase34_recoverability"
    elif not within_horizon:
        negative_reason = "recovery_target_was_reached_after_frozen_horizon"

    if negative_reason is not None:
        return _result(
            RECOVERY_SUCCESS_EVALUATOR_ID,
            EVALUATION_CLEAR,
            negative_reason,
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
            details=details,
        )

    return _result(
        RECOVERY_SUCCESS_EVALUATOR_ID,
        EVALUATION_TRIGGERED,
        "all_frozen_recovery_success_predicates_are_satisfied",
        evidence_level=evidence_level,
        evaluated_step=evaluated_step,
        required_inputs_present=True,
        details=details,
    )


def _validate_optional_text(value: object) -> bool:
    return value is None or isinstance(value, str)


def evaluate_instability(
    *,
    instability_flag: bool | None,
    terminal_label: str | None = None,
    termination_reason: str | None = None,
    evaluated_step: int | None = None,
    evidence_level: str = "measured",
) -> RecoveryEvaluationResult:
    if not _step_is_valid(evaluated_step):
        return _invalid_step_result(
            INSTABILITY_EVALUATOR_ID,
            evidence_level=evidence_level,
        )
    malformed = []
    if instability_flag is not None and not _boolean_is_valid(instability_flag):
        malformed.append("instability_flag")
    if not _validate_optional_text(terminal_label):
        malformed.append("terminal_label")
    if not _validate_optional_text(termination_reason):
        malformed.append("termination_reason")
    if malformed:
        return _result(
            INSTABILITY_EVALUATOR_ID,
            EVALUATION_INVALID,
            "instability_evidence_is_malformed",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
            details=_details(invalid_fields=",".join(sorted(malformed))),
        )

    label_trigger = terminal_label == "instability"
    reason_trigger = termination_reason in SUPPORTED_INSTABILITY_TERMINATION_REASONS
    if instability_flag is False and (label_trigger or reason_trigger):
        return _result(
            INSTABILITY_EVALUATOR_ID,
            EVALUATION_INVALID,
            "explicit_instability_flag_conflicts_with_supported_trigger",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
            details=_details(
                instability_flag=False,
                terminal_label=terminal_label,
                termination_reason=termination_reason,
            ),
        )

    if instability_flag is True or label_trigger or reason_trigger:
        sources = []
        if instability_flag is True:
            sources.append("instability_flag")
        if label_trigger:
            sources.append("terminal_label")
        if reason_trigger:
            sources.append("termination_reason")
        return _result(
            INSTABILITY_EVALUATOR_ID,
            EVALUATION_TRIGGERED,
            "repository_supported_instability_evidence_is_present",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
            details=_details(
                evidence_sources=",".join(sources),
                terminal_label=terminal_label,
                termination_reason=termination_reason,
            ),
        )

    if instability_flag is False:
        return _result(
            INSTABILITY_EVALUATOR_ID,
            EVALUATION_CLEAR,
            "explicit_instability_instrumentation_is_clear",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
            details=_details(
                instability_flag=False,
                terminal_label=terminal_label,
                termination_reason=termination_reason,
            ),
        )

    return _result(
        INSTABILITY_EVALUATOR_ID,
        EVALUATION_NOT_EVALUATED,
        "no_supported_instability_instrumentation_is_present",
        evidence_level=evidence_level,
        evaluated_step=evaluated_step,
        required_inputs_present=False,
        details=_details(
            terminal_label=terminal_label,
            termination_reason=termination_reason,
        ),
    )


def evaluate_unsafe_state(
    *,
    unsafe_state_flag: bool | None,
    terminal_label: str | None = None,
    overspeed: bool | None = None,
    invalid_simulation: bool | None = None,
    action_rejected: bool | None = None,
    explicit_abort: bool | None = None,
    simulator_success: bool | None = None,
    evaluated_step: int | None = None,
    evidence_level: str = "measured",
) -> RecoveryEvaluationResult:
    if not _step_is_valid(evaluated_step):
        return _invalid_step_result(
            UNSAFE_STATE_EVALUATOR_ID,
            evidence_level=evidence_level,
        )
    context_flags = {
        "action_rejected": action_rejected,
        "explicit_abort": explicit_abort,
        "invalid_simulation": invalid_simulation,
        "overspeed": overspeed,
        "simulator_success": simulator_success,
        "unsafe_state_flag": unsafe_state_flag,
    }
    malformed = [
        name
        for name, value in context_flags.items()
        if value is not None and not _boolean_is_valid(value)
    ]
    if not _validate_optional_text(terminal_label):
        malformed.append("terminal_label")
    if malformed:
        return _result(
            UNSAFE_STATE_EVALUATOR_ID,
            EVALUATION_INVALID,
            "unsafe_state_evidence_is_malformed",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
            details=_details(invalid_fields=",".join(sorted(malformed))),
        )

    label_trigger = terminal_label == "unsafe_state"
    if unsafe_state_flag is False and label_trigger:
        return _result(
            UNSAFE_STATE_EVALUATOR_ID,
            EVALUATION_INVALID,
            "explicit_unsafe_state_flag_conflicts_with_terminal_label",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
            details=_details(
                terminal_label=terminal_label,
                unsafe_state_flag=False,
            ),
        )

    context_details = _details(
        action_rejected=action_rejected,
        explicit_abort=explicit_abort,
        invalid_simulation=invalid_simulation,
        overspeed=overspeed,
        simulator_success=simulator_success,
        terminal_label=terminal_label,
        unsafe_state_flag=unsafe_state_flag,
    )
    if unsafe_state_flag is True or label_trigger:
        return _result(
            UNSAFE_STATE_EVALUATOR_ID,
            EVALUATION_TRIGGERED,
            "explicit_repository_unsafe_state_evidence_is_present",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
            details=context_details,
        )
    if unsafe_state_flag is False:
        return _result(
            UNSAFE_STATE_EVALUATOR_ID,
            EVALUATION_CLEAR,
            "explicit_unsafe_state_instrumentation_is_clear",
            evidence_level=evidence_level,
            evaluated_step=evaluated_step,
            required_inputs_present=True,
            details=context_details,
        )
    return _result(
        UNSAFE_STATE_EVALUATOR_ID,
        EVALUATION_NOT_EVALUATED,
        "no_explicit_unsafe_state_instrumentation_is_present",
        evidence_level=evidence_level,
        evaluated_step=evaluated_step,
        required_inputs_present=False,
        details=context_details,
    )


__all__ = [
    "EVALUATION_CLEAR",
    "EVALUATION_INVALID",
    "EVALUATION_NOT_EVALUATED",
    "EVALUATION_STATUSES",
    "EVALUATION_TRIGGERED",
    "FROZEN_RECOVERY_HORIZON",
    "INSTABILITY_EVALUATOR_ID",
    "PHASE34_RECOVERABILITY_EVALUATOR_ID",
    "PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX",
    "PHASE34_RECOVERABLE_VR_RATIO_MAX",
    "PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX",
    "RECOVERY_SUCCESS_EVALUATOR_ID",
    "SUPPORTED_INSTABILITY_TERMINATION_REASONS",
    "UNSAFE_STATE_EVALUATOR_ID",
    "RecoveryEvaluationResult",
    "evaluate_instability",
    "evaluate_phase34_compatible_recoverability",
    "evaluate_recovery_success_v0",
    "evaluate_unsafe_state",
]
