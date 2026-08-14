from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import Enum

from runtime_assurance.recovery_branch_executor import (
    Action2D,
    generate_velocity_opposed_action,
)
from runtime_assurance.staged_recovery_contract import RecoveryPhase
from simulator.phase34_35_transition import CartesianState2D


AUTHORITY_ADAPTER_ID = "stage2a_one_intervention_hazard_arrest_authority_v0"
PROVISIONAL_ACTION_SOURCE = "velocity_opposed_thrust_v0"
MAXIMUM_PROVISIONAL_PROPOSALS_PER_SESSION = 1
FINAL_VETO_EVALUATION_REQUIRED = True
FINAL_VETO_BYPASS_ALLOWED = False
ADAPTER_EXECUTES_ACTION = False
SHADOW_OUTPUT_CONSUMED_BY_PHYSICAL_RUNTIME = False
STAGED_RECOVERY_EXECUTION = "not_authorized"
RELEASE_STATUS = "not_implemented"


class HazardArrestAuthorityError(ValueError):
    pass


class AuthorityEvidenceStatus(str, Enum):
    AVAILABLE_VALID = "available_valid"
    MISSING = "missing"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"


class HazardEvidenceKind(str, Enum):
    PREDICTED_OVERSPEED = "predicted_overspeed"
    REALIZED_OVERSPEED = "realized_overspeed"


class AuthorityDecisionStatus(str, Enum):
    BLOCKED = "blocked"
    PROVISIONAL_PROPOSAL_GENERATED = "provisional_proposal_generated"


class AuthorityBlockedReason(str, Enum):
    AUTHORITY_DISABLED = "authority_disabled"
    AUTHORITY_NOT_GRANTED = "authority_not_granted"
    REQUESTED_PHASE_NOT_AUTHORIZED = "requested_phase_not_authorized"
    TRIGGER_EVIDENCE_NOT_AUTHORIZED = "trigger_evidence_not_authorized"
    MISSING_EVIDENCE = "missing_evidence"
    INVALID_EVIDENCE = "invalid_evidence"
    UNSUPPORTED_EVIDENCE = "unsupported_evidence"
    HAZARD_NOT_ASSERTED = "hazard_not_asserted"
    REALIZED_OVERSPEED_IS_ADVERSE_TERMINAL = (
        "realized_overspeed_is_adverse_terminal"
    )
    EXPLICIT_ABORT_IS_TERMINAL = "explicit_abort_is_terminal"
    PROPOSAL_ALREADY_GENERATED = "proposal_already_generated"
    PROPOSAL_ALREADY_CONSUMED = "proposal_already_consumed"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_finite_optional(value: float | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HazardArrestAuthorityError(f"{name} must be finite or None")
    if not math.isfinite(float(value)):
        raise HazardArrestAuthorityError(f"{name} must be finite or None")


@dataclass(frozen=True, slots=True)
class HazardArrestAuthorityInput:
    authority_enabled: bool = False
    authority_granted: bool = False
    requested_phase: str | None = None
    trigger_evidence_authorized: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.authority_enabled, "authority_enabled"),
            (self.authority_granted, "authority_granted"),
            (self.trigger_evidence_authorized, "trigger_evidence_authorized"),
        ):
            if not isinstance(value, bool):
                raise HazardArrestAuthorityError(f"{name} must be boolean")
        if self.requested_phase is not None and not self.requested_phase:
            raise HazardArrestAuthorityError(
                "requested_phase must be nonempty or None"
            )


@dataclass(frozen=True, slots=True)
class HazardArrestEvidence:
    state: CartesianState2D | None
    state_status: AuthorityEvidenceStatus
    instrumentation_status: AuthorityEvidenceStatus
    recovery_evaluation_status: AuthorityEvidenceStatus
    hazard_status: AuthorityEvidenceStatus
    hazard_kind: HazardEvidenceKind | None = None
    hazard_asserted: bool | None = None
    realized_speed_ratio: float | None = None
    predicted_speed_ratio: float | None = None
    explicit_abort_requested: bool = False

    def __post_init__(self) -> None:
        for status in (
            self.state_status,
            self.instrumentation_status,
            self.recovery_evaluation_status,
            self.hazard_status,
        ):
            if not isinstance(status, AuthorityEvidenceStatus):
                raise HazardArrestAuthorityError("unsupported evidence status")
        if not isinstance(self.explicit_abort_requested, bool):
            raise HazardArrestAuthorityError(
                "explicit_abort_requested must be boolean"
            )
        if self.hazard_asserted is not None and not isinstance(
            self.hazard_asserted, bool
        ):
            raise HazardArrestAuthorityError(
                "hazard_asserted must be boolean or None"
            )
        _require_finite_optional(
            self.realized_speed_ratio, "realized_speed_ratio"
        )
        _require_finite_optional(
            self.predicted_speed_ratio, "predicted_speed_ratio"
        )
        if self.state_status == AuthorityEvidenceStatus.AVAILABLE_VALID:
            if not isinstance(self.state, CartesianState2D):
                raise HazardArrestAuthorityError(
                    "available state evidence requires a CartesianState2D"
                )
        elif self.state is not None:
            raise HazardArrestAuthorityError(
                "unavailable state evidence must not carry a state"
            )
        if self.hazard_status == AuthorityEvidenceStatus.AVAILABLE_VALID:
            if self.hazard_kind is None or self.hazard_asserted is None:
                raise HazardArrestAuthorityError(
                    "available hazard evidence requires kind and assertion"
                )
            if (
                self.hazard_kind == HazardEvidenceKind.PREDICTED_OVERSPEED
                and self.predicted_speed_ratio is None
            ):
                raise HazardArrestAuthorityError(
                    "predicted overspeed evidence requires predicted_speed_ratio"
                )
            if (
                self.hazard_kind == HazardEvidenceKind.REALIZED_OVERSPEED
                and self.realized_speed_ratio is None
            ):
                raise HazardArrestAuthorityError(
                    "realized overspeed evidence requires realized_speed_ratio"
                )
        elif self.hazard_kind is not None or self.hazard_asserted is not None:
            raise HazardArrestAuthorityError(
                "unavailable hazard evidence must not carry kind or assertion"
            )


@dataclass(frozen=True, slots=True)
class HazardArrestAuthoritySession:
    session_id: str
    proposal_count: int = 0
    proposal_consumed: bool = False
    proposal_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise HazardArrestAuthorityError("session_id must be nonempty")
        if (
            isinstance(self.proposal_count, bool)
            or not isinstance(self.proposal_count, int)
            or self.proposal_count not in (0, 1)
        ):
            raise HazardArrestAuthorityError("proposal_count must be zero or one")
        if self.proposal_consumed and self.proposal_count != 1:
            raise HazardArrestAuthorityError(
                "a consumed proposal requires proposal_count=1"
            )
        if (self.proposal_count == 0) != (self.proposal_id is None):
            raise HazardArrestAuthorityError(
                "proposal_id presence must match proposal_count"
            )


@dataclass(frozen=True, slots=True)
class ProvisionalHazardArrestProposal:
    proposal_id: str
    session_id: str
    requested_phase: str
    action_source: str
    action: Action2D
    final_veto_evaluation_required: bool = FINAL_VETO_EVALUATION_REQUIRED
    final_veto_bypass_allowed: bool = FINAL_VETO_BYPASS_ALLOWED
    adapter_executes_action: bool = ADAPTER_EXECUTES_ACTION
    shadow_output_consumed_by_physical_runtime: bool = (
        SHADOW_OUTPUT_CONSUMED_BY_PHYSICAL_RUNTIME
    )

    def __post_init__(self) -> None:
        if self.requested_phase != RecoveryPhase.HAZARD_ARREST.value:
            raise HazardArrestAuthorityError(
                "a provisional proposal may target only hazard_arrest"
            )
        if self.action_source != PROVISIONAL_ACTION_SOURCE:
            raise HazardArrestAuthorityError(
                "provisional proposal must reuse velocity_opposed_thrust_v0"
            )
        if not self.final_veto_evaluation_required or self.final_veto_bypass_allowed:
            raise HazardArrestAuthorityError("Final Veto must remain mandatory")
        if self.adapter_executes_action:
            raise HazardArrestAuthorityError("authority adapter cannot execute actions")
        if self.shadow_output_consumed_by_physical_runtime:
            raise HazardArrestAuthorityError(
                "shadow output cannot be consumed by physical runtime"
            )


@dataclass(frozen=True, slots=True)
class HazardArrestAuthorityDecision:
    status: AuthorityDecisionStatus
    authority_enabled: bool
    authority_granted: bool
    evidence_available_and_valid: bool
    proposal_generated: bool
    proposal_consumed: bool
    final_veto_evaluation_required: bool
    blocked_reason: AuthorityBlockedReason | None
    proposal: ProvisionalHazardArrestProposal | None
    release_status: str = RELEASE_STATUS
    staged_recovery_execution: str = STAGED_RECOVERY_EXECUTION

    def __post_init__(self) -> None:
        generated = self.status == AuthorityDecisionStatus.PROVISIONAL_PROPOSAL_GENERATED
        if generated != self.proposal_generated:
            raise HazardArrestAuthorityError("decision status and proposal flag differ")
        if generated != (self.proposal is not None):
            raise HazardArrestAuthorityError("decision proposal presence is inconsistent")
        if generated == (self.blocked_reason is not None):
            raise HazardArrestAuthorityError("decision blocked reason is inconsistent")
        if self.final_veto_evaluation_required != generated:
            raise HazardArrestAuthorityError(
                "Final Veto is required exactly when a proposal is generated"
            )
        if self.release_status != RELEASE_STATUS:
            raise HazardArrestAuthorityError("release behavior is not implemented")
        if self.staged_recovery_execution != STAGED_RECOVERY_EXECUTION:
            raise HazardArrestAuthorityError("staged recovery execution is unauthorized")


@dataclass(frozen=True, slots=True)
class HazardArrestAuthorityResult:
    session: HazardArrestAuthoritySession
    decision: HazardArrestAuthorityDecision


def _blocked(
    session: HazardArrestAuthoritySession,
    authority: HazardArrestAuthorityInput,
    reason: AuthorityBlockedReason,
    *,
    evidence_valid: bool = False,
) -> HazardArrestAuthorityResult:
    return HazardArrestAuthorityResult(
        session=session,
        decision=HazardArrestAuthorityDecision(
            status=AuthorityDecisionStatus.BLOCKED,
            authority_enabled=authority.authority_enabled,
            authority_granted=authority.authority_granted,
            evidence_available_and_valid=evidence_valid,
            proposal_generated=False,
            proposal_consumed=session.proposal_consumed,
            final_veto_evaluation_required=False,
            blocked_reason=reason,
            proposal=None,
        ),
    )


def request_hazard_arrest_proposal(
    session: HazardArrestAuthoritySession,
    authority: HazardArrestAuthorityInput | None,
    evidence: HazardArrestEvidence,
) -> HazardArrestAuthorityResult:
    supplied_authority = authority or HazardArrestAuthorityInput()

    if evidence.explicit_abort_requested:
        return _blocked(
            session,
            supplied_authority,
            AuthorityBlockedReason.EXPLICIT_ABORT_IS_TERMINAL,
        )
    if session.proposal_consumed:
        return _blocked(
            session,
            supplied_authority,
            AuthorityBlockedReason.PROPOSAL_ALREADY_CONSUMED,
        )
    if session.proposal_count:
        return _blocked(
            session,
            supplied_authority,
            AuthorityBlockedReason.PROPOSAL_ALREADY_GENERATED,
        )
    if not supplied_authority.authority_enabled:
        return _blocked(
            session,
            supplied_authority,
            AuthorityBlockedReason.AUTHORITY_DISABLED,
        )
    if not supplied_authority.authority_granted:
        return _blocked(
            session,
            supplied_authority,
            AuthorityBlockedReason.AUTHORITY_NOT_GRANTED,
        )
    if supplied_authority.requested_phase != RecoveryPhase.HAZARD_ARREST.value:
        return _blocked(
            session,
            supplied_authority,
            AuthorityBlockedReason.REQUESTED_PHASE_NOT_AUTHORIZED,
        )
    if not supplied_authority.trigger_evidence_authorized:
        return _blocked(
            session,
            supplied_authority,
            AuthorityBlockedReason.TRIGGER_EVIDENCE_NOT_AUTHORIZED,
        )

    statuses = (
        evidence.state_status,
        evidence.instrumentation_status,
        evidence.recovery_evaluation_status,
        evidence.hazard_status,
    )
    if AuthorityEvidenceStatus.INVALID in statuses:
        return _blocked(
            session,
            supplied_authority,
            AuthorityBlockedReason.INVALID_EVIDENCE,
        )
    if AuthorityEvidenceStatus.UNSUPPORTED in statuses:
        return _blocked(
            session,
            supplied_authority,
            AuthorityBlockedReason.UNSUPPORTED_EVIDENCE,
        )
    if AuthorityEvidenceStatus.MISSING in statuses:
        return _blocked(
            session,
            supplied_authority,
            AuthorityBlockedReason.MISSING_EVIDENCE,
        )
    if not evidence.hazard_asserted:
        return _blocked(
            session,
            supplied_authority,
            AuthorityBlockedReason.HAZARD_NOT_ASSERTED,
            evidence_valid=True,
        )
    if evidence.hazard_kind == HazardEvidenceKind.REALIZED_OVERSPEED:
        return _blocked(
            session,
            supplied_authority,
            AuthorityBlockedReason.REALIZED_OVERSPEED_IS_ADVERSE_TERMINAL,
            evidence_valid=True,
        )

    assert evidence.state is not None
    action = generate_velocity_opposed_action(evidence.state)
    proposal_document = {
        "adapter_id": AUTHORITY_ADAPTER_ID,
        "session_id": session.session_id,
        "requested_phase": RecoveryPhase.HAZARD_ARREST.value,
        "action_source": PROVISIONAL_ACTION_SOURCE,
        "action": action,
        "hazard_kind": evidence.hazard_kind.value,
        "realized_speed_ratio": evidence.realized_speed_ratio,
        "predicted_speed_ratio": evidence.predicted_speed_ratio,
    }
    proposal_id = _canonical_hash(proposal_document)
    proposal = ProvisionalHazardArrestProposal(
        proposal_id=proposal_id,
        session_id=session.session_id,
        requested_phase=RecoveryPhase.HAZARD_ARREST.value,
        action_source=PROVISIONAL_ACTION_SOURCE,
        action=action,
    )
    updated_session = replace(
        session,
        proposal_count=1,
        proposal_id=proposal_id,
    )
    return HazardArrestAuthorityResult(
        session=updated_session,
        decision=HazardArrestAuthorityDecision(
            status=AuthorityDecisionStatus.PROVISIONAL_PROPOSAL_GENERATED,
            authority_enabled=True,
            authority_granted=True,
            evidence_available_and_valid=True,
            proposal_generated=True,
            proposal_consumed=False,
            final_veto_evaluation_required=True,
            blocked_reason=None,
            proposal=proposal,
        ),
    )


def consume_hazard_arrest_proposal(
    session: HazardArrestAuthoritySession,
    proposal: ProvisionalHazardArrestProposal,
) -> HazardArrestAuthoritySession:
    if session.proposal_count != 1 or session.proposal_id is None:
        raise HazardArrestAuthorityError("session has no proposal to consume")
    if session.proposal_consumed:
        raise HazardArrestAuthorityError("proposal has already been consumed")
    if proposal.session_id != session.session_id or proposal.proposal_id != session.proposal_id:
        raise HazardArrestAuthorityError("proposal does not belong to authority session")
    if not proposal.final_veto_evaluation_required or proposal.final_veto_bypass_allowed:
        raise HazardArrestAuthorityError("proposal cannot bypass Final Veto")
    return replace(session, proposal_consumed=True)


__all__ = [
    "ADAPTER_EXECUTES_ACTION",
    "AUTHORITY_ADAPTER_ID",
    "FINAL_VETO_BYPASS_ALLOWED",
    "FINAL_VETO_EVALUATION_REQUIRED",
    "MAXIMUM_PROVISIONAL_PROPOSALS_PER_SESSION",
    "PROVISIONAL_ACTION_SOURCE",
    "RELEASE_STATUS",
    "SHADOW_OUTPUT_CONSUMED_BY_PHYSICAL_RUNTIME",
    "STAGED_RECOVERY_EXECUTION",
    "AuthorityBlockedReason",
    "AuthorityDecisionStatus",
    "AuthorityEvidenceStatus",
    "HazardArrestAuthorityDecision",
    "HazardArrestAuthorityError",
    "HazardArrestAuthorityInput",
    "HazardArrestAuthorityResult",
    "HazardArrestAuthoritySession",
    "HazardArrestEvidence",
    "HazardEvidenceKind",
    "ProvisionalHazardArrestProposal",
    "consume_hazard_arrest_proposal",
    "request_hazard_arrest_proposal",
]
