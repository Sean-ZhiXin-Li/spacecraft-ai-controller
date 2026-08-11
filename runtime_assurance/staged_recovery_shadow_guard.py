from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from runtime_assurance.staged_recovery_contract import (
    EXECUTION_NOT_AUTHORIZED,
    RecoveryPhase,
    default_transition_contracts,
)
from runtime_assurance.staged_recovery_guard_evidence import (
    GuardAtomEvaluation,
    GuardEvidenceStatus,
)


SHADOW_POLICY_ID = "engineering_shadow_policy_v0"
SHADOW_SCHEMA_VERSION = "staged_recovery_shadow_guard_v0"
UNASSIGNED = "unassigned"
MINIMUM_PHASE_DWELL_STEPS = 2
TRANSITION_COOLDOWN_STEPS = 1
MAXIMUM_SHADOW_TRANSITIONS_PER_TRACE = 8
CONTROL_AUTHORITY = False
SHADOW_OUTPUT_CONSUMED_BY_PHYSICAL_RUNTIME = False


class ShadowGuardError(ValueError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def architecture_phase_ids() -> tuple[str, ...]:
    return tuple(phase.value for phase in RecoveryPhase)


def allowed_shadow_edges() -> frozenset[tuple[str, str]]:
    return frozenset(
        (contract.source_phase.value, contract.destination_phase.value)
        for contract in default_transition_contracts()
    )


@dataclass(frozen=True, slots=True)
class ShadowGuardParameters:
    minimum_phase_dwell_steps: int = MINIMUM_PHASE_DWELL_STEPS
    transition_cooldown_steps: int = TRANSITION_COOLDOWN_STEPS
    maximum_shadow_transitions_per_trace: int = MAXIMUM_SHADOW_TRANSITIONS_PER_TRACE

    def __post_init__(self) -> None:
        for value, name, minimum in (
            (self.minimum_phase_dwell_steps, "minimum_phase_dwell_steps", 1),
            (self.transition_cooldown_steps, "transition_cooldown_steps", 0),
            (
                self.maximum_shadow_transitions_per_trace,
                "maximum_shadow_transitions_per_trace",
                1,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ShadowGuardError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True, slots=True)
class ShadowGuardResolution:
    desired_phase: str
    transition_reason: str
    true_guard_atoms: tuple[str, ...]
    false_guard_atoms: tuple[str, ...]
    unavailable_guard_atoms: tuple[str, ...]
    invalid_guard_atoms: tuple[str, ...]
    priority_reason: str
    handoff_ready_supplied: bool | None
    policy_id: str = SHADOW_POLICY_ID
    nominal_handoff_authorized: bool = False
    control_authority: bool = CONTROL_AUTHORITY

    def __post_init__(self) -> None:
        if self.desired_phase not in architecture_phase_ids():
            raise ShadowGuardError("shadow resolver selected an unknown architecture phase")
        if self.nominal_handoff_authorized:
            raise ShadowGuardError("nominal handoff is not authorized in Stage 1B-A")
        if self.control_authority:
            raise ShadowGuardError("shadow guard cannot have control authority")


@dataclass(frozen=True, slots=True)
class ShadowRuntimeState:
    current_phase: str = UNASSIGNED
    phase_dwell_steps: int = 0
    cooldown_steps_remaining: int = 0
    shadow_transition_count: int = 0
    phase_history: tuple[str, ...] = ()
    transition_reason_history: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.current_phase != UNASSIGNED and self.current_phase not in architecture_phase_ids():
            raise ShadowGuardError("shadow state contains an unknown phase")
        if UNASSIGNED in self.phase_history:
            raise ShadowGuardError("unassigned cannot be serialized in phase history")
        for value in (
            self.phase_dwell_steps,
            self.cooldown_steps_remaining,
            self.shadow_transition_count,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ShadowGuardError("shadow counters must be nonnegative integers")


@dataclass(frozen=True, slots=True)
class ShadowTransitionRecord:
    event_index: int
    recovery_step: int
    source_event_hash: str
    current_shadow_phase: str
    desired_shadow_phase: str
    resulting_shadow_phase: str
    transition_reason: str
    shadow_transition_executed: bool
    transition_blocked: bool
    block_reason: str | None
    dwell_before: int
    dwell_after: int
    cooldown_before: int
    cooldown_after: int
    transition_budget_used: int
    transition_budget_remaining: int
    two_cycle_detected: bool
    three_cycle_detected: bool
    rapid_reversal_detected: bool
    repeated_transition_reason: bool
    true_guard_atoms: tuple[str, ...]
    false_guard_atoms: tuple[str, ...]
    unavailable_guard_atoms: tuple[str, ...]
    invalid_guard_atoms: tuple[str, ...]
    priority_reason: str
    nominal_handoff_recommended: bool
    control_authority: bool
    shadow_output_consumed_by_physical_runtime: bool
    staged_recovery_execution: str
    canonical_record_hash: str = ""

    def scientific_document(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field != "canonical_record_hash"
        }

    def as_document(self) -> dict[str, object]:
        document = self.scientific_document()
        document["canonical_record_hash"] = self.canonical_record_hash
        return document


def _atom_map(evaluations: Sequence[GuardAtomEvaluation]) -> dict[str, GuardAtomEvaluation]:
    result = {item.guard_atom_id: item for item in evaluations}
    if len(result) != len(evaluations):
        raise ShadowGuardError("duplicate guard atom evaluation")
    return result


def resolve_shadow_phase(
    evaluations: Sequence[GuardAtomEvaluation],
    *,
    handoff_ready: bool | None = None,
    unsafe_state: bool | None = None,
) -> ShadowGuardResolution:
    atoms = _atom_map(evaluations)
    true_ids = tuple(sorted(key for key, item in atoms.items() if item.status == GuardEvidenceStatus.TRUE))
    false_ids = tuple(sorted(key for key, item in atoms.items() if item.status == GuardEvidenceStatus.FALSE))
    invalid_ids = tuple(sorted(key for key, item in atoms.items() if item.status == GuardEvidenceStatus.INVALID))
    unavailable_ids = tuple(sorted(
        key for key, item in atoms.items()
        if item.status in {
            GuardEvidenceStatus.NOT_EVALUATED,
            GuardEvidenceStatus.UNSUPPORTED,
            GuardEvidenceStatus.POLICY_UNRESOLVED,
        }
    ))

    def is_true(atom_id: str) -> bool:
        item = atoms.get(atom_id)
        return item is not None and item.status == GuardEvidenceStatus.TRUE

    def is_false(atom_id: str) -> bool:
        item = atoms.get(atom_id)
        return item is not None and item.status == GuardEvidenceStatus.FALSE

    critical_invalid = bool(invalid_ids) or any(
        is_true(atom_id) is False and atoms.get(atom_id) is not None
        and atoms[atom_id].status == GuardEvidenceStatus.FALSE
        for atom_id in ("state_evidence_valid", "instrumentation_evaluation_valid")
    )
    if is_true("explicit_abort_requested"):
        phase, reason = RecoveryPhase.EXPLICIT_ABORT.value, "externally_supplied_explicit_abort"
    elif unsafe_state is True or critical_invalid:
        phase, reason = RecoveryPhase.RETREAT.value, "invalid_or_externally_unsafe_evidence"
    elif is_true("realized_overspeed") or is_true("predicted_overspeed"):
        phase, reason = RecoveryPhase.HAZARD_ARREST.value, "realized_or_predicted_overspeed"
    elif is_true("phase34_compatible_recoverability_pass"):
        # Handoff is deliberately not selected: Stage 1B-A has no externally
        # validated handoff evaluator or active authorization.
        phase, reason = RecoveryPhase.RECOVERABILITY_VERIFICATION.value, "phase34_compatible_recoverability"
    elif is_false("recoverability_radius_component_pass") or is_false(
        "recoverability_radial_velocity_component_pass"
    ):
        phase, reason = RecoveryPhase.RADIAL_RECOMMITMENT.value, "radius_or_radial_component_not_satisfied"
    elif is_false("recoverability_tangential_velocity_component_pass"):
        phase, reason = RecoveryPhase.TANGENTIAL_ALIGNMENT.value, "tangential_component_not_satisfied"
    elif is_true("no_eligible_crossing"):
        phase, reason = RecoveryPhase.CROSSING_PREPARATION.value, "eligible_crossing_not_observed"
    else:
        phase, reason = RecoveryPhase.STABILIZATION_ASSESSMENT.value, "stabilization_fallback_or_incomplete_evidence"

    return ShadowGuardResolution(
        desired_phase=phase,
        transition_reason=reason,
        true_guard_atoms=true_ids,
        false_guard_atoms=false_ids,
        unavailable_guard_atoms=unavailable_ids,
        invalid_guard_atoms=invalid_ids,
        priority_reason=reason,
        handoff_ready_supplied=handoff_ready,
    )


class ShadowPhaseMachine:
    def __init__(self, parameters: ShadowGuardParameters | None = None) -> None:
        self._parameters = parameters or ShadowGuardParameters()
        self._state = ShadowRuntimeState()

    @property
    def state(self) -> ShadowRuntimeState:
        return self._state

    def step(
        self,
        resolution: ShadowGuardResolution,
        *,
        event_index: int,
        recovery_step: int,
        source_event_hash: str,
        terminal_event: bool = False,
    ) -> ShadowTransitionRecord:
        before = self._state
        desired = resolution.desired_phase
        blocked = False
        block_reason: str | None = None
        transitioned = False
        resulting = before.current_phase
        cooldown_after = max(0, before.cooldown_steps_remaining - 1)

        if terminal_event and before.current_phase != UNASSIGNED:
            block_reason = "terminal_event_holds_shadow_state"
        elif before.current_phase == UNASSIGNED:
            resulting = desired
            transitioned = True
        elif desired == before.current_phase:
            pass
        elif (before.current_phase, desired) not in allowed_shadow_edges():
            blocked, block_reason = True, "graph_edge_not_allowed"
        elif before.phase_dwell_steps < self._parameters.minimum_phase_dwell_steps:
            blocked, block_reason = True, "minimum_phase_dwell_not_satisfied"
        elif before.cooldown_steps_remaining > 0:
            blocked, block_reason = True, "transition_cooldown_active"
        elif before.shadow_transition_count >= self._parameters.maximum_shadow_transitions_per_trace:
            blocked, block_reason = True, "transition_budget_exhausted"
        else:
            resulting = desired
            transitioned = True

        if transitioned:
            history = before.phase_history + (resulting,)
            reasons = before.transition_reason_history + (resolution.transition_reason,)
            if before.current_phase == UNASSIGNED:
                transition_count = before.shadow_transition_count
            else:
                transition_count = before.shadow_transition_count + 1
            state = ShadowRuntimeState(
                current_phase=resulting,
                phase_dwell_steps=1,
                cooldown_steps_remaining=(
                    0 if before.current_phase == UNASSIGNED else self._parameters.transition_cooldown_steps
                ),
                shadow_transition_count=transition_count,
                phase_history=history,
                transition_reason_history=reasons,
            )
        else:
            current = desired if before.current_phase == UNASSIGNED else before.current_phase
            state = replace(
                before,
                current_phase=current,
                phase_dwell_steps=before.phase_dwell_steps + 1,
                cooldown_steps_remaining=cooldown_after,
            )

        two_cycle = len(state.phase_history) >= 3 and state.phase_history[-1] == state.phase_history[-3]
        three_cycle = len(state.phase_history) >= 4 and state.phase_history[-1] == state.phase_history[-4]
        repeated = (
            transitioned
            and len(state.transition_reason_history) >= 2
            and state.transition_reason_history[-1] == state.transition_reason_history[-2]
        )
        remaining = max(
            0,
            self._parameters.maximum_shadow_transitions_per_trace - state.shadow_transition_count,
        )
        record = ShadowTransitionRecord(
            event_index=event_index,
            recovery_step=recovery_step,
            source_event_hash=source_event_hash,
            current_shadow_phase=before.current_phase,
            desired_shadow_phase=desired,
            resulting_shadow_phase=state.current_phase,
            transition_reason=resolution.transition_reason,
            shadow_transition_executed=transitioned,
            transition_blocked=blocked,
            block_reason=block_reason,
            dwell_before=before.phase_dwell_steps,
            dwell_after=state.phase_dwell_steps,
            cooldown_before=before.cooldown_steps_remaining,
            cooldown_after=state.cooldown_steps_remaining,
            transition_budget_used=state.shadow_transition_count,
            transition_budget_remaining=remaining,
            two_cycle_detected=two_cycle,
            three_cycle_detected=three_cycle,
            rapid_reversal_detected=two_cycle,
            repeated_transition_reason=repeated,
            true_guard_atoms=resolution.true_guard_atoms,
            false_guard_atoms=resolution.false_guard_atoms,
            unavailable_guard_atoms=resolution.unavailable_guard_atoms,
            invalid_guard_atoms=resolution.invalid_guard_atoms,
            priority_reason=resolution.priority_reason,
            nominal_handoff_recommended=(desired == RecoveryPhase.NOMINAL_HANDOFF.value),
            control_authority=False,
            shadow_output_consumed_by_physical_runtime=False,
            staged_recovery_execution=EXECUTION_NOT_AUTHORIZED,
        )
        record = replace(record, canonical_record_hash=canonical_sha256(record.scientific_document()))
        self._state = state
        return record


__all__ = [
    "CONTROL_AUTHORITY",
    "MAXIMUM_SHADOW_TRANSITIONS_PER_TRACE",
    "MINIMUM_PHASE_DWELL_STEPS",
    "SHADOW_OUTPUT_CONSUMED_BY_PHYSICAL_RUNTIME",
    "SHADOW_POLICY_ID",
    "TRANSITION_COOLDOWN_STEPS",
    "UNASSIGNED",
    "ShadowGuardError",
    "ShadowGuardParameters",
    "ShadowGuardResolution",
    "ShadowPhaseMachine",
    "ShadowRuntimeState",
    "ShadowTransitionRecord",
    "allowed_shadow_edges",
    "architecture_phase_ids",
    "canonical_sha256",
    "resolve_shadow_phase",
]
