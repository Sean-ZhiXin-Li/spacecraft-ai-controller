from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace
from unittest import mock

from runtime_assurance.recovery_branch_executor import (
    generate_velocity_opposed_action,
)
from runtime_assurance.stage2a_hazard_arrest_authority import (
    ADAPTER_EXECUTES_ACTION,
    FINAL_VETO_BYPASS_ALLOWED,
    MAXIMUM_PROVISIONAL_PROPOSALS_PER_SESSION,
    RELEASE_STATUS,
    SHADOW_OUTPUT_CONSUMED_BY_PHYSICAL_RUNTIME,
    AuthorityBlockedReason,
    AuthorityDecisionStatus,
    AuthorityEvidenceStatus,
    HazardArrestAuthorityInput,
    HazardArrestAuthoritySession,
    HazardArrestEvidence,
    HazardEvidenceKind,
    consume_hazard_arrest_proposal,
    request_hazard_arrest_proposal,
)
from runtime_assurance.staged_recovery_contract import RecoveryPhase
from simulator.phase34_35_transition import CartesianState2D


class Stage2AHazardArrestAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = CartesianState2D(x=3.0, y=4.0, vx=3.0, vy=4.0)
        self.session = HazardArrestAuthoritySession(session_id="case-a")
        self.authority = HazardArrestAuthorityInput(
            authority_enabled=True,
            authority_granted=True,
            requested_phase=RecoveryPhase.HAZARD_ARREST.value,
            trigger_evidence_authorized=True,
        )
        self.evidence = HazardArrestEvidence(
            state=self.state,
            state_status=AuthorityEvidenceStatus.AVAILABLE_VALID,
            instrumentation_status=AuthorityEvidenceStatus.AVAILABLE_VALID,
            recovery_evaluation_status=AuthorityEvidenceStatus.AVAILABLE_VALID,
            hazard_status=AuthorityEvidenceStatus.AVAILABLE_VALID,
            hazard_kind=HazardEvidenceKind.PREDICTED_OVERSPEED,
            hazard_asserted=True,
            realized_speed_ratio=1.8,
            predicted_speed_ratio=1.91,
        )

    def test_default_authority_is_disabled_and_produces_no_proposal(self) -> None:
        result = request_hazard_arrest_proposal(
            self.session, None, self.evidence
        )
        self.assertEqual(result.decision.status, AuthorityDecisionStatus.BLOCKED)
        self.assertEqual(
            result.decision.blocked_reason,
            AuthorityBlockedReason.AUTHORITY_DISABLED,
        )
        self.assertIsNone(result.decision.proposal)

    def test_external_grant_is_required(self) -> None:
        authority = replace(self.authority, authority_granted=False)
        result = request_hazard_arrest_proposal(
            self.session, authority, self.evidence
        )
        self.assertEqual(
            result.decision.blocked_reason,
            AuthorityBlockedReason.AUTHORITY_NOT_GRANTED,
        )

    def test_missing_evidence_blocks_proposal(self) -> None:
        evidence = replace(
            self.evidence,
            state=None,
            state_status=AuthorityEvidenceStatus.MISSING,
        )
        result = request_hazard_arrest_proposal(
            self.session, self.authority, evidence
        )
        self.assertEqual(
            result.decision.blocked_reason,
            AuthorityBlockedReason.MISSING_EVIDENCE,
        )

    def test_invalid_evidence_blocks_proposal(self) -> None:
        evidence = replace(
            self.evidence,
            instrumentation_status=AuthorityEvidenceStatus.INVALID,
        )
        result = request_hazard_arrest_proposal(
            self.session, self.authority, evidence
        )
        self.assertEqual(
            result.decision.blocked_reason,
            AuthorityBlockedReason.INVALID_EVIDENCE,
        )

    def test_unsupported_evidence_remains_unsupported(self) -> None:
        evidence = replace(
            self.evidence,
            recovery_evaluation_status=AuthorityEvidenceStatus.UNSUPPORTED,
        )
        result = request_hazard_arrest_proposal(
            self.session, self.authority, evidence
        )
        self.assertEqual(
            result.decision.blocked_reason,
            AuthorityBlockedReason.UNSUPPORTED_EVIDENCE,
        )

    def test_external_trigger_authorization_is_separate_from_raw_ratio(self) -> None:
        authority = replace(self.authority, trigger_evidence_authorized=False)
        result = request_hazard_arrest_proposal(
            self.session, authority, self.evidence
        )
        self.assertEqual(
            result.decision.blocked_reason,
            AuthorityBlockedReason.TRIGGER_EVIDENCE_NOT_AUTHORIZED,
        )

    def test_valid_authority_and_evidence_generate_one_provisional_proposal(self) -> None:
        result = request_hazard_arrest_proposal(
            self.session, self.authority, self.evidence
        )
        proposal = result.decision.proposal
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(result.session.proposal_count, 1)
        self.assertFalse(result.session.proposal_consumed)
        self.assertEqual(MAXIMUM_PROVISIONAL_PROPOSALS_PER_SESSION, 1)
        self.assertEqual(proposal.action, generate_velocity_opposed_action(self.state))
        self.assertTrue(proposal.final_veto_evaluation_required)
        self.assertFalse(proposal.final_veto_bypass_allowed)
        self.assertFalse(proposal.adapter_executes_action)

    def test_second_proposal_is_blocked_before_consumption(self) -> None:
        first = request_hazard_arrest_proposal(
            self.session, self.authority, self.evidence
        )
        second = request_hazard_arrest_proposal(
            first.session, self.authority, self.evidence
        )
        self.assertEqual(
            second.decision.blocked_reason,
            AuthorityBlockedReason.PROPOSAL_ALREADY_GENERATED,
        )

    def test_consumed_proposal_blocks_second_proposal(self) -> None:
        first = request_hazard_arrest_proposal(
            self.session, self.authority, self.evidence
        )
        assert first.decision.proposal is not None
        consumed = consume_hazard_arrest_proposal(
            first.session, first.decision.proposal
        )
        self.assertTrue(consumed.proposal_consumed)
        second = request_hazard_arrest_proposal(
            consumed, self.authority, self.evidence
        )
        self.assertEqual(
            second.decision.blocked_reason,
            AuthorityBlockedReason.PROPOSAL_ALREADY_CONSUMED,
        )

    def test_action_identity_is_deterministic(self) -> None:
        first = request_hazard_arrest_proposal(
            self.session, self.authority, self.evidence
        )
        second = request_hazard_arrest_proposal(
            self.session, self.authority, self.evidence
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first.decision.proposal.proposal_id,
            second.decision.proposal.proposal_id,
        )

    def test_state_is_not_mutated_and_no_simulator_transition_is_called(self) -> None:
        before = self.state
        with mock.patch(
            "runtime_assurance.recovery_branch_executor.step_phase34_35_transition",
            side_effect=AssertionError("simulator transition called"),
        ) as transition:
            result = request_hazard_arrest_proposal(
                self.session, self.authority, self.evidence
            )
        transition.assert_not_called()
        self.assertEqual(self.state, before)
        self.assertFalse(result.decision.proposal.adapter_executes_action)

    def test_final_veto_cannot_be_bypassed(self) -> None:
        result = request_hazard_arrest_proposal(
            self.session, self.authority, self.evidence
        )
        assert result.decision.proposal is not None
        self.assertTrue(result.decision.final_veto_evaluation_required)
        self.assertFalse(FINAL_VETO_BYPASS_ALLOWED)
        self.assertFalse(ADAPTER_EXECUTES_ACTION)

    def test_all_other_staged_phases_remain_unauthorized(self) -> None:
        for phase in RecoveryPhase:
            if phase == RecoveryPhase.HAZARD_ARREST:
                continue
            with self.subTest(phase=phase.value):
                authority = replace(self.authority, requested_phase=phase.value)
                result = request_hazard_arrest_proposal(
                    self.session, authority, self.evidence
                )
                self.assertEqual(
                    result.decision.blocked_reason,
                    AuthorityBlockedReason.REQUESTED_PHASE_NOT_AUTHORIZED,
                )
                self.assertIsNone(result.decision.proposal)

    def test_nominal_handoff_remains_unauthorized(self) -> None:
        authority = replace(
            self.authority,
            requested_phase=RecoveryPhase.NOMINAL_HANDOFF.value,
        )
        result = request_hazard_arrest_proposal(
            self.session, authority, self.evidence
        )
        self.assertEqual(
            result.decision.blocked_reason,
            AuthorityBlockedReason.REQUESTED_PHASE_NOT_AUTHORIZED,
        )

    def test_explicit_abort_is_terminal_and_never_an_action(self) -> None:
        evidence = replace(self.evidence, explicit_abort_requested=True)
        result = request_hazard_arrest_proposal(
            self.session, self.authority, evidence
        )
        self.assertEqual(
            result.decision.blocked_reason,
            AuthorityBlockedReason.EXPLICIT_ABORT_IS_TERMINAL,
        )
        self.assertIsNone(result.decision.proposal)

    def test_realized_overspeed_is_preserved_as_adverse_terminal_evidence(self) -> None:
        evidence = replace(
            self.evidence,
            hazard_kind=HazardEvidenceKind.REALIZED_OVERSPEED,
            realized_speed_ratio=1.91,
        )
        result = request_hazard_arrest_proposal(
            self.session, self.authority, evidence
        )
        self.assertEqual(
            result.decision.blocked_reason,
            AuthorityBlockedReason.REALIZED_OVERSPEED_IS_ADVERSE_TERMINAL,
        )

    def test_release_behavior_remains_not_implemented(self) -> None:
        result = request_hazard_arrest_proposal(
            self.session, self.authority, self.evidence
        )
        self.assertEqual(result.decision.release_status, RELEASE_STATUS)
        self.assertEqual(RELEASE_STATUS, "not_implemented")

    def test_authority_outputs_are_immutable(self) -> None:
        result = request_hazard_arrest_proposal(
            self.session, self.authority, self.evidence
        )
        with self.assertRaises(FrozenInstanceError):
            result.session.proposal_count = 2

    def test_shadow_output_has_no_physical_runtime_consumer(self) -> None:
        self.assertFalse(SHADOW_OUTPUT_CONSUMED_BY_PHYSICAL_RUNTIME)


if __name__ == "__main__":
    unittest.main()
