from __future__ import annotations

import unittest

from runtime_assurance.staged_recovery_guard_evidence import (
    GuardAtomEvaluation,
    GuardEvidenceLevel,
    GuardEvidenceStatus,
)
from runtime_assurance.staged_recovery_shadow_guard import (
    ShadowPhaseMachine,
    allowed_shadow_edges,
    architecture_phase_ids,
    resolve_shadow_phase,
)


def atom(atom_id: str, status: GuardEvidenceStatus) -> GuardAtomEvaluation:
    value = True if status == GuardEvidenceStatus.TRUE else False if status == GuardEvidenceStatus.FALSE else None
    return GuardAtomEvaluation(
        guard_atom_id=atom_id,
        status=status,
        value=value,
        evidence_level=GuardEvidenceLevel.DERIVED,
        raw_source_values=(),
        comparator="fixture",
        threshold_or_parameter_reference=None,
        reason="fixture",
    )


class ShadowGuardResolutionTests(unittest.TestCase):
    def test_architecture_identity_and_graph_are_reused(self) -> None:
        self.assertEqual(len(architecture_phase_ids()), 9)
        self.assertEqual(len(allowed_shadow_edges()), 26)

    def test_deterministic_initial_phase_and_priority(self) -> None:
        values = (
            atom("explicit_abort_requested", GuardEvidenceStatus.TRUE),
            atom("realized_overspeed", GuardEvidenceStatus.TRUE),
        )
        first = resolve_shadow_phase(values)
        second = resolve_shadow_phase(values)
        self.assertEqual(first, second)
        self.assertEqual(first.desired_phase, "explicit_abort")

    def test_invalid_evidence_recommends_retreat(self) -> None:
        result = resolve_shadow_phase((atom("state_evidence_valid", GuardEvidenceStatus.INVALID),))
        self.assertEqual(result.desired_phase, "retreat")
        self.assertIn("state_evidence_valid", result.invalid_guard_atoms)

    def test_realized_and_predicted_overspeed_recommend_hazard_arrest(self) -> None:
        for atom_id in ("realized_overspeed", "predicted_overspeed"):
            with self.subTest(atom_id=atom_id):
                result = resolve_shadow_phase((atom(atom_id, GuardEvidenceStatus.TRUE),))
                self.assertEqual(result.desired_phase, "hazard_arrest")

    def test_recoverability_and_component_priorities(self) -> None:
        result = resolve_shadow_phase((atom("phase34_compatible_recoverability_pass", GuardEvidenceStatus.TRUE),))
        self.assertEqual(result.desired_phase, "recoverability_verification")
        radial = resolve_shadow_phase((atom("recoverability_radius_component_pass", GuardEvidenceStatus.FALSE),))
        self.assertEqual(radial.desired_phase, "radial_recommitment")
        tangential = resolve_shadow_phase((
            atom("recoverability_radius_component_pass", GuardEvidenceStatus.TRUE),
            atom("recoverability_radial_velocity_component_pass", GuardEvidenceStatus.TRUE),
            atom("recoverability_tangential_velocity_component_pass", GuardEvidenceStatus.FALSE),
        ))
        self.assertEqual(tangential.desired_phase, "tangential_alignment")

    def test_unavailable_guard_is_not_consumed_as_false(self) -> None:
        result = resolve_shadow_phase((
            atom("recoverability_radius_component_pass", GuardEvidenceStatus.NOT_EVALUATED),
        ))
        self.assertEqual(result.desired_phase, "stabilization_assessment")
        self.assertIn("recoverability_radius_component_pass", result.unavailable_guard_atoms)

    def test_handoff_remains_blocked_even_when_readiness_argument_is_true(self) -> None:
        result = resolve_shadow_phase(
            (atom("phase34_compatible_recoverability_pass", GuardEvidenceStatus.TRUE),),
            handoff_ready=True,
        )
        self.assertFalse(result.nominal_handoff_authorized)
        self.assertNotEqual(result.desired_phase, "nominal_handoff")


class ShadowPhaseMachineTests(unittest.TestCase):
    def resolution(self, phase: str, reason: str = "fixture"):
        base = resolve_shadow_phase(())
        return type(base)(
            desired_phase=phase,
            transition_reason=reason,
            true_guard_atoms=(), false_guard_atoms=(), unavailable_guard_atoms=(),
            invalid_guard_atoms=(), priority_reason=reason, handoff_ready_supplied=None,
        )

    def step(self, machine: ShadowPhaseMachine, phase: str, index: int, reason: str = "fixture"):
        return machine.step(
            self.resolution(phase, reason), event_index=index, recovery_step=index,
            source_event_hash=f"{index:064x}",
        )

    def test_initial_assignment_is_internal_and_deterministic(self) -> None:
        machine = ShadowPhaseMachine()
        record = self.step(machine, "hazard_arrest", 0)
        self.assertTrue(record.shadow_transition_executed)
        self.assertEqual(machine.state.shadow_transition_count, 0)
        self.assertNotIn("unassigned", machine.state.phase_history)

    def test_graph_legality_blocks_forbidden_direct_edge(self) -> None:
        machine = ShadowPhaseMachine()
        self.step(machine, "hazard_arrest", 0)
        self.step(machine, "hazard_arrest", 1)
        record = self.step(machine, "recoverability_verification", 2)
        self.assertTrue(record.transition_blocked)
        self.assertEqual(record.block_reason, "graph_edge_not_allowed")

    def test_dwell_and_cooldown(self) -> None:
        machine = ShadowPhaseMachine()
        self.step(machine, "hazard_arrest", 0)
        blocked = self.step(machine, "stabilization_assessment", 1)
        self.assertEqual(blocked.block_reason, "minimum_phase_dwell_not_satisfied")
        moved = self.step(machine, "stabilization_assessment", 2)
        self.assertTrue(moved.shadow_transition_executed)
        cooldown = self.step(machine, "radial_recommitment", 3)
        self.assertEqual(cooldown.block_reason, "minimum_phase_dwell_not_satisfied")
        # A hold satisfies dwell and expires the one-event cooldown.
        self.step(machine, "stabilization_assessment", 4)
        moved_again = self.step(machine, "radial_recommitment", 5)
        self.assertTrue(moved_again.shadow_transition_executed)

    def test_two_cycle_and_repeated_reason_detection(self) -> None:
        machine = ShadowPhaseMachine()
        self.step(machine, "stabilization_assessment", 0, "same")
        self.step(machine, "stabilization_assessment", 1, "hold")
        self.step(machine, "radial_recommitment", 2, "same")
        self.step(machine, "radial_recommitment", 3, "hold")
        record = self.step(machine, "stabilization_assessment", 4, "same")
        self.assertTrue(record.two_cycle_detected)
        self.assertTrue(record.repeated_transition_reason)

    def test_transition_budget_never_goes_negative(self) -> None:
        machine = ShadowPhaseMachine()
        machine._state = type(machine.state)(
            current_phase="stabilization_assessment", phase_dwell_steps=2,
            cooldown_steps_remaining=0, shadow_transition_count=8,
            phase_history=("stabilization_assessment",), transition_reason_history=(),
        )
        record = self.step(machine, "radial_recommitment", 9)
        self.assertEqual(record.block_reason, "transition_budget_exhausted")
        self.assertEqual(record.transition_budget_remaining, 0)
        self.assertFalse(record.control_authority)
        self.assertFalse(record.shadow_output_consumed_by_physical_runtime)


if __name__ == "__main__":
    unittest.main()
