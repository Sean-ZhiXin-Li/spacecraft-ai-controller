from __future__ import annotations

import copy
import inspect
import json
import unittest
from pathlib import Path
from unittest import mock

from runtime_assurance.final_veto_monitor import FinalVetoDecision
from runtime_assurance.final_veto_runner_types import PreTransitionActionContext, RolloutCaseContext
from runtime_assurance.recovery_branch_state_registry import canonical_json_bytes, canonical_sha256
from runtime_assurance.stage2a_prediction_boundary_discovery import (
    NormalActionEvaluation,
    is_prediction_boundary_candidate,
)
from runtime_assurance.stage2a_prediction_boundary_discovery_d2 import (
    ANGLE_GRID,
    D1_ANCHOR_PREDICTED_SPEED_RATIO,
    D1_MANIFEST_HASH,
    D1_PLAN_HASH,
    MAXIMUM_RECOVERY_TRANSITIONS,
    OUTPUT_PATH,
    RECOVERY_BRANCH_ID,
    R0_OVER_TARGET,
    SEED,
    THRUST_SCALE,
    D2DiscoveryError,
    D2SourceBoundary,
    _NaturalFirstVetoHook,
    _SourceBoundaryCaptured,
    _source_execution_identity,
    build_d2_payloads,
    build_source_cases,
    execute_natural_source_case,
    load_d2_plan,
    protected_evidence_hashes,
    run_zero_action_recovery,
    source_case_id,
    validate_d2_payloads,
    validate_d2_plan,
    validate_static_sources,
)
from runtime_assurance.staged_recovery_logger_adapter import runtime_state_hash
from simulator.phase34_35_transition import (
    CartesianState2D,
    NormalizedAction2D,
    Phase3435TransitionResult,
)


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CHECKS = tuple(
    (name, True)
    for name in (
        "D1_event_index", "angle", "boundary_state_hash", "case_id",
        "legacy_boundary_type", "legacy_registry_member", "legacy_reproduction_checks",
        "r0", "seed", "thrust", "zero_action",
        "zero_action_predicted_speed_ratio", "zero_action_predicted_state_hash",
    )
)


def _decision(predicted: float, allowed: bool) -> FinalVetoDecision:
    action = (0.0, 0.0)
    return FinalVetoDecision(
        monitor_id="one_step_overspeed_veto_v0",
        decision="allow" if allowed else "veto",
        reason="predicted_nominal_within_threshold" if allowed else "predicted_nominal_overspeed",
        threshold=1.90,
        comparator=">",
        nominal_action=action,
        executed_action=action,
        fallback_action=action,
        predicted_nominal_speed_ratio=predicted,
        predicted_fallback_speed_ratio=None if allowed else 1.0,
        fallback_predicted_to_exceed_threshold=None if allowed else False,
        veto_applied=not allowed,
    )


def _evaluation(
    state: CartesianState2D, realized: float, predicted: float, allowed: bool
) -> NormalActionEvaluation:
    action = (0.0, 0.0)
    next_state = CartesianState2D(state.x + 1.0, state.y, predicted, 0.0)
    transition = Phase3435TransitionResult(next_state, NormalizedAction2D(*action))
    return NormalActionEvaluation(
        branch_id=RECOVERY_BRANCH_ID,
        current_state=state,
        current_state_hash=runtime_state_hash(state),
        realized_speed_ratio=realized,
        realized_headroom=1.90 - realized,
        action=action,
        action_hash=canonical_sha256({"action": list(action)}),
        predicted_transition=transition,
        predicted_state_hash=runtime_state_hash(next_state),
        predicted_speed_ratio=predicted,
        predicted_headroom=1.90 - predicted,
        final_veto_decision=_decision(predicted, allowed),
        fallback_prediction_count=0 if allowed else 1,
        candidate_boundary=is_prediction_boundary_candidate(realized, predicted),
    )


def _boundary(index: int) -> D2SourceBoundary:
    case = build_source_cases()[index]
    state = CartesianState2D(1.0 + index, 0.0, 1.0, 0.0)
    predicted = CartesianState2D(2.0 + index, 0.0, 2.0, 0.0)
    constants = {"dt": 1.0, "mass": 1.0, "mu": 1.0, "target_circular_speed": 1.0,
                 "speed_ratio_denominator_epsilon": 0.0}
    simulator = {"simulator_constants": constants, "thrust_scale": 1.0}
    document = {
        "case_id": case.case_id,
        "case_configuration": case.configuration(),
        "seed": case.seed,
        "position_x": state.x, "position_y": state.y,
        "velocity_x": state.vx, "velocity_y": state.vy,
        "state_origin": "legacy_first_veto_branch_point" if case.anchor else "natural_phase35_nominal_first_veto_execution",
        "manually_authored_state": False,
        "perturbed_from_existing_state": False,
        "predicted_next_state": {"position_x": predicted.x, "position_y": predicted.y,
                                 "velocity_x": predicted.vx, "velocity_y": predicted.vy},
        "predicted_speed_ratio": 2.0,
        "nominal_action": [1.0, 0.0],
        "monitor_decision": {"decision": "veto"},
        "simulator_configuration": simulator,
        "simulator_configuration_hash": canonical_sha256(simulator),
        "constants_hash": canonical_sha256(constants),
    }
    action_hash = canonical_sha256([[1.0, 0.0]])
    states_hash = canonical_sha256([{"position_x": state.x, "position_y": state.y,
                                     "velocity_x": state.vx, "velocity_y": state.vy}])
    identity = _source_execution_identity(
        case, status="available", prefix_count=1, branch_step=2,
        boundary_state_hash=runtime_state_hash(state), predicted_speed_ratio=2.0,
        action_trace_hash=action_hash, state_trace_hash=states_hash,
    )
    return D2SourceBoundary(
        case=case, status="available", unavailability_reason=None,
        document_json=canonical_json_bytes(document).decode("utf-8"),
        source_execution_hash=canonical_sha256(identity), source_prefix_transition_count=1,
        branch_step=2, boundary_state_hash=runtime_state_hash(state), realized_speed_ratio=1.0,
        nominal_action=(1.0, 0.0), predicted_state_hash=runtime_state_hash(predicted),
        predicted_speed_ratio=2.0, final_veto_decision="veto",
        prefix_action_trace_hash=action_hash, prefix_state_trace_hash=states_hash,
        source_final_veto_rejection_count=1, source_vetoed_proposal_transition_count=0,
        fallback_execution_count=0, anchor_equivalence_checks=ANCHOR_CHECKS if case.anchor else (),
    )


def _state(boundary: D2SourceBoundary) -> CartesianState2D:
    value = boundary.document()
    return CartesianState2D(value["position_x"], value["position_y"], value["velocity_x"], value["velocity_y"])


def _context(index: int, step: int, predicted: float) -> PreTransitionActionContext:
    case = build_source_cases()[index]
    state = CartesianState2D(1.0, 0.0, 1.0, 0.0)
    nominal = (1.0, 0.0)
    def predictor(current, action):
        ratio = predicted if action == nominal else 1.0
        return Phase3435TransitionResult(
            CartesianState2D(current.x + 1.0, current.y, ratio, 0.0), NormalizedAction2D(*action)
        )
    return PreTransitionActionContext(
        step=step, phase="PRE_CROSS", active_stage="radial_energy_push",
        current_state=state, nominal_action=nominal, predict_transition=predictor,
        compute_speed_ratio=lambda value: abs(value.vx),
        case=RolloutCaseContext(
            case_id=case.case_id, controller_id=case.controller_id,
            controller_family="phase35_crossing_basin_expansion",
            r0_over_target=case.r0_over_target, initial_velocity_angle_deg=case.angle,
            thrust_scale=case.thrust_scale, target_radius=1.0, target_circular_speed=1.0,
            post_cross_mode=case.post_cross_mode, upstream_variant=case.upstream_variant,
        ),
    )


class Stage2ATargetedDiscoveryD2Tests(unittest.TestCase):
    def test_frozen_grid_dimensions_ids_and_ordering(self) -> None:
        self.assertEqual(
            ANGLE_GRID,
            (150.0, 155.0, 160.0, 162.5, 165.0, 167.5, 170.0, 172.5, 175.0),
        )
        cases = build_source_cases()
        self.assertEqual(len(cases), 9)
        self.assertTrue(all(item.r0_over_target == R0_OVER_TARGET for item in cases))
        self.assertTrue(all(item.thrust_scale == THRUST_SCALE for item in cases))
        self.assertTrue(all(item.seed == SEED for item in cases))
        self.assertEqual(cases[3].case_id, source_case_id(162.5))
        self.assertEqual(cases, build_source_cases())

    def test_no_manual_cartesian_override_api_exists(self) -> None:
        parameters = inspect.signature(execute_natural_source_case).parameters
        for forbidden in ("state", "position", "velocity", "x", "y", "vx", "vy"):
            self.assertNotIn(forbidden, parameters)

    def test_first_veto_capture_precedes_action_and_fallback(self) -> None:
        hook = _NaturalFirstVetoHook(
            build_source_cases()[1], {"simulator_constants": {}}, "a" * 40
        )
        with self.assertRaises(_SourceBoundaryCaptured) as captured:
            hook(_context(1, 1, 2.0))
        boundary = captured.exception.boundary
        self.assertEqual(boundary.source_prefix_transition_count, 0)
        self.assertEqual(boundary.branch_step, 1)
        self.assertEqual(boundary.source_vetoed_proposal_transition_count, 0)
        self.assertEqual(boundary.fallback_execution_count, 0)
        self.assertEqual(boundary.final_veto_decision, "veto")

    def test_allowed_prefix_then_first_veto_counts_exactly(self) -> None:
        hook = _NaturalFirstVetoHook(
            build_source_cases()[1], {"simulator_constants": {}}, "a" * 40
        )
        allowed = hook(_context(1, 1, 1.8))
        self.assertEqual(allowed.executed_action, allowed.nominal_action)
        with self.assertRaises(_SourceBoundaryCaptured) as captured:
            hook(_context(1, 2, 2.0))
        self.assertEqual(captured.exception.boundary.source_prefix_transition_count, 1)
        self.assertEqual(captured.exception.boundary.branch_step, 2)

    def test_source_without_veto_is_explicitly_unavailable(self) -> None:
        with mock.patch(
            "scripts.explicit_controller_phase35_crossing_basin_expansion.rollout_phase35_case",
            return_value={"termination_reason": "terminal_without_veto"},
        ):
            result = execute_natural_source_case(
                ROOT, build_source_cases()[1], implementation_commit="a" * 40
            )
        self.assertEqual(result.status, "unavailable")
        self.assertIsNone(result.document_json)
        self.assertEqual(result.fallback_execution_count, 0)

    def test_anchor_contract_is_frozen_to_legacy_and_d1(self) -> None:
        plan = load_d2_plan(ROOT)
        self.assertEqual(plan["D1_dependency"]["manifest_hash"], D1_MANIFEST_HASH)
        self.assertEqual(plan["D1_dependency"]["scientific_plan_hash"], D1_PLAN_HASH)
        self.assertEqual(
            plan["D1_dependency"]["anchor_predicted_speed_ratio"],
            D1_ANCHOR_PREDICTED_SPEED_RATIO,
        )
        source = inspect.getsource(
            __import__(
                "runtime_assurance.stage2a_prediction_boundary_discovery_d2",
                fromlist=["reproduce_anchor_source"],
            ).reproduce_anchor_source
        )
        self.assertIn("prepare_legacy_discovery_source", source)
        self.assertIn("_load_d1_anchor_evidence", source)

    def test_strict_candidate_boundary_and_realized_adverse(self) -> None:
        self.assertTrue(is_prediction_boundary_candidate(1.90, 1.9000000001))
        self.assertFalse(is_prediction_boundary_candidate(1.90, 1.90))
        self.assertFalse(is_prediction_boundary_candidate(1.9000000001, 2.0))
        adverse = _evaluation(CartesianState2D(1.0, 0.0, 2.0, 0.0), 1.91, 1.8, True)
        result = run_zero_action_recovery(_boundary(0), evaluator=mock.Mock(return_value=adverse))
        self.assertEqual(result.terminal_reason, "realized_overspeed_adverse_terminal")
        self.assertEqual(result.physical_transition_count, 0)

    def test_candidate_veto_executes_no_transition_or_fallback(self) -> None:
        item = _evaluation(CartesianState2D(1.0, 0.0, 1.0, 0.0), 1.0, 2.0, False)
        executor = mock.Mock()
        result = run_zero_action_recovery(
            _boundary(0), evaluator=mock.Mock(return_value=item), transition_executor=executor
        )
        executor.assert_not_called()
        self.assertEqual(result.candidate_count, 1)
        self.assertEqual(result.physical_transition_count, 0)
        self.assertEqual(result.fallback_execution_count, 0)
        self.assertFalse(result.records[0]["transition_executed"])

    def test_allowed_zero_action_prediction_equals_realization(self) -> None:
        calls = 0
        def evaluator(document, state, branch):
            nonlocal calls
            calls += 1
            return _evaluation(state, 1.0, 1.8 if calls == 1 else 2.0, calls == 1)
        result = run_zero_action_recovery(
            _boundary(0), evaluator=evaluator,
            transition_executor=lambda document, item: item.predicted_transition,
        )
        self.assertEqual(result.physical_transition_count, 1)
        self.assertEqual(result.records[0]["predicted_state_hash"], result.records[0]["realized_next_state_hash"])

    def test_zero_action_is_sole_branch_and_horizon_is_eight(self) -> None:
        self.assertEqual(RECOVERY_BRANCH_ID, "zero_action_reference_v0")
        result = run_zero_action_recovery(
            _boundary(0),
            evaluator=lambda document, state, branch: _evaluation(state, 1.0, 1.8, True),
            transition_executor=lambda document, item: item.predicted_transition,
        )
        self.assertEqual(MAXIMUM_RECOVERY_TRANSITIONS, 8)
        self.assertEqual(result.physical_transition_count, 8)

    def test_stage2a_authority_is_absent_and_counts_are_zero(self) -> None:
        source = (ROOT / "runtime_assurance/stage2a_prediction_boundary_discovery_d2.py").read_text("utf-8")
        self.assertNotIn("from runtime_assurance.stage2a_hazard_arrest_authority", source)
        self.assertNotIn("velocity_opposed_thrust_v0", source)
        candidate = _evaluation(CartesianState2D(1.0, 0.0, 1.0, 0.0), 1.0, 2.0, False)
        result = run_zero_action_recovery(_boundary(0), evaluator=mock.Mock(return_value=candidate))
        self.assertEqual(result.records[0]["hazard_arrest_interventions"], 0)
        self.assertFalse(result.records[0]["active_authority_granted"])

    def test_plan_hash_and_protected_d1_validate(self) -> None:
        plan = load_d2_plan(ROOT)
        self.assertEqual(plan["canonical_plan_hash"], "48635494001a616fa4cbcc0aeeccbaea14521d7531ddda16c42af12dceaf23a3")
        self.assertEqual(plan["protected_evidence_hashes"], protected_evidence_hashes(ROOT))
        changed = copy.deepcopy(plan)
        changed["thrust_scale"] = 9000.0
        with self.assertRaises(D2DiscoveryError):
            validate_d2_plan(ROOT, changed)

    def test_static_validation_is_read_only(self) -> None:
        report = validate_static_sources(ROOT)
        self.assertFalse(report["simulation_executed"])
        self.assertFalse(report["write_performed"])
        self.assertFalse(report["active_authority_granted"])
        self.assertEqual(report["hazard_arrest_interventions"], 0)

    def test_payloads_are_deterministic_and_exact(self) -> None:
        boundaries = tuple(_boundary(index) for index in range(9))
        recoveries = tuple(
            run_zero_action_recovery(
                boundary,
                evaluator=mock.Mock(return_value=_evaluation(_state(boundary), 1.0, 2.0, False)),
            )
            for boundary in boundaries
        )
        protected = protected_evidence_hashes(ROOT)
        first = build_d2_payloads(
            ROOT, boundaries, recoveries, implementation_commit="a" * 40,
            protected_before=protected, protected_after=protected,
        )
        second = build_d2_payloads(
            ROOT, boundaries, recoveries, implementation_commit="a" * 40,
            protected_before=protected, protected_after=protected,
        )
        self.assertEqual(first, second)
        validate_d2_payloads(first, source_plan=load_d2_plan(ROOT))
        manifest = json.loads(first["discovery_manifest.json"])
        self.assertEqual(manifest["candidate_boundary_count"], 9)
        self.assertEqual(manifest["hazard_arrest_interventions"], 0)
        with self.assertRaises(D2DiscoveryError):
            validate_d2_payloads({})

    def test_default_clis_do_not_execute_or_read_results(self) -> None:
        from scripts import check_stage2a_prediction_boundary_discovery_d2_v0 as checker
        from scripts import run_stage2a_prediction_boundary_discovery_d2_v0 as runner
        output_existed_before = (ROOT / OUTPUT_PATH).exists()
        with mock.patch.object(runner, "_execute") as execute:
            self.assertEqual(runner.main([]), 0)
        execute.assert_not_called()
        with mock.patch.object(checker, "load_published_payloads") as load:
            self.assertEqual(checker.main([]), 0)
        load.assert_not_called()
        self.assertEqual((ROOT / OUTPUT_PATH).exists(), output_existed_before)


if __name__ == "__main__":
    unittest.main()
