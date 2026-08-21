from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from runtime_assurance.final_veto_monitor import FinalVetoDecision
from runtime_assurance.recovery_branch_state_extractor import (
    PrefixExecutionResult,
    build_source_case_inventory,
)
from runtime_assurance.recovery_branch_state_registry import canonical_sha256
from runtime_assurance.stage2a_prediction_boundary_discovery import (
    DISCOVERY_BRANCH_IDS,
    MAXIMUM_PHYSICAL_TRANSITIONS_PER_TRAJECTORY,
    OUTPUT_PATH,
    PLAN_PATH,
    PLANNED_TRAJECTORY_COUNT,
    DiscoveryTrajectoryResult,
    NormalActionEvaluation,
    PredictionBoundaryDiscoveryError,
    build_discovery_payloads,
    build_trajectory_definitions,
    evaluate_normal_action,
    is_prediction_boundary_candidate,
    load_discovery_plan,
    run_discovery_trajectory,
    trajectory_id,
    validate_discovery_payloads,
    validate_discovery_plan,
)
from runtime_assurance.staged_recovery_logger_adapter import runtime_state_hash
from runtime_assurance.staged_recovery_shadow_calibration import (
    ShadowCalibrationError,
    atomic_publish_new_directory,
)
from simulator.phase34_35_transition import (
    CartesianState2D,
    NormalizedAction2D,
    Phase3435TransitionResult,
)


ROOT = Path(__file__).resolve().parents[1]


def branch_document(state: CartesianState2D | None = None) -> dict[str, object]:
    current = state or CartesianState2D(1.0, 0.0, 0.5, 0.0)
    constants = {
        "dt": 1.0,
        "mass": 1.0,
        "mu": 1.0,
        "target_circular_speed": 1.0,
        "speed_ratio_denominator_epsilon": 0.0,
    }
    simulator = {"simulator_constants": constants, "thrust_scale": 1.0}
    return {
        "position_x": current.x,
        "position_y": current.y,
        "velocity_x": current.vx,
        "velocity_y": current.vy,
        "simulator_configuration": simulator,
        "simulator_configuration_hash": canonical_sha256(simulator),
        "constants_hash": canonical_sha256(constants),
    }


def decision(
    action: tuple[float, float],
    *,
    predicted_ratio: float,
    allowed: bool,
) -> FinalVetoDecision:
    return FinalVetoDecision(
        monitor_id="one_step_overspeed_veto_v0",
        decision="allow" if allowed else "veto",
        reason=(
            "predicted_nominal_within_threshold"
            if allowed
            else "predicted_nominal_overspeed"
        ),
        threshold=1.90,
        comparator=">",
        nominal_action=action,
        executed_action=action if allowed else (0.0, 0.0),
        fallback_action=(0.0, 0.0),
        predicted_nominal_speed_ratio=predicted_ratio,
        predicted_fallback_speed_ratio=None if allowed else 1.0,
        fallback_predicted_to_exceed_threshold=None if allowed else False,
        veto_applied=not allowed,
    )


def evaluation(
    state: CartesianState2D,
    *,
    predicted_ratio: float,
    candidate: bool,
    allowed: bool,
) -> NormalActionEvaluation:
    action = (0.0, 0.0)
    next_state = CartesianState2D(
        state.x + 1.0,
        state.y,
        predicted_ratio,
        0.0,
    )
    transition = Phase3435TransitionResult(
        next_state=next_state,
        executed_action=NormalizedAction2D(*action),
    )
    realized_ratio = 1.0 if candidate else abs(state.vx)
    return NormalActionEvaluation(
        branch_id="zero_action_reference_v0",
        current_state=state,
        current_state_hash=runtime_state_hash(state),
        realized_speed_ratio=realized_ratio,
        realized_headroom=1.90 - realized_ratio,
        action=action,
        action_hash=canonical_sha256({"action": list(action)}),
        predicted_transition=transition,
        predicted_state_hash=runtime_state_hash(next_state),
        predicted_speed_ratio=predicted_ratio,
        predicted_headroom=1.90 - predicted_ratio,
        final_veto_decision=decision(
            action, predicted_ratio=predicted_ratio, allowed=allowed
        ),
        fallback_prediction_count=0 if allowed else 1,
        candidate_boundary=candidate,
    )


def prefix_for_case(case: object, state: CartesianState2D | None = None) -> PrefixExecutionResult:
    document = branch_document(state)
    document.update(
        {
            "canonical_payload_hash": "a" * 64,
            "branch_step": int(case.nominal_prefix_transition_count) + 1,
        }
    )
    return PrefixExecutionResult(
        execution_id=f"prefix__{case.case_id}",
        case=case,
        execution_role="candidate_discovery",
        document_json=json.dumps(document, sort_keys=True, separators=(",", ":")),
        actual_transition_count=int(case.nominal_prefix_transition_count),
        branch_step=int(case.nominal_prefix_transition_count) + 1,
        initial_state_hash="b" * 64,
        prefix_action_trace_hash="c" * 64,
        prefix_state_trace_hash="d" * 64,
        canonical_payload_hash="a" * 64,
        predicted_speed_ratio=1.0,
        tangential_velocity_error_ratio=0.0,
        boundary_type=str(case.boundary.boundary_type),
        boundary_transition_count=int(case.nominal_prefix_transition_count),
        terminal_transition_count=int(case.boundary.terminal_transition_count),
        terminal_reason=str(case.boundary.terminal_reason),
    )


class PredictionBoundaryDiscoveryTests(unittest.TestCase):
    def test_strict_candidate_comparison(self) -> None:
        self.assertTrue(is_prediction_boundary_candidate(1.90, 1.9000000001))
        self.assertFalse(is_prediction_boundary_candidate(1.90, 1.90))
        self.assertFalse(is_prediction_boundary_candidate(1.9000000001, 2.0))

    def test_exactly_1p90_is_clear(self) -> None:
        self.assertFalse(is_prediction_boundary_candidate(1.0, 1.90))

    def test_realized_overspeed_cannot_be_candidate(self) -> None:
        self.assertFalse(is_prediction_boundary_candidate(1.91, 1.91))

    def test_predicted_overspeed_is_detected_through_unchanged_veto(self) -> None:
        state = CartesianState2D(1.0, 0.0, 1.0, 0.0)

        def predicted_step(current, action, dynamics):
            return Phase3435TransitionResult(
                next_state=CartesianState2D(2.0, 0.0, 2.0, 0.0),
                executed_action=NormalizedAction2D(action.action_x, action.action_y),
            )

        with mock.patch(
            "runtime_assurance.stage2a_prediction_boundary_discovery.step_phase34_35_transition",
            side_effect=predicted_step,
        ):
            result = evaluate_normal_action(
                branch_document(state), state, "zero_action_reference_v0"
            )
        self.assertTrue(result.candidate_boundary)
        self.assertEqual(result.final_veto_decision.decision, "veto")
        self.assertEqual(result.physical_transition_count, 0)
        self.assertGreaterEqual(result.fallback_prediction_count, 1)

    def test_candidate_stops_before_physical_execution(self) -> None:
        case = SimpleNamespace(
            case_id="case",
            source_configuration_hash="1" * 64,
            transition_implementation_hash="2" * 64,
            nominal_controller_hash="3" * 64,
            nominal_prefix_transition_count=1,
            boundary=SimpleNamespace(
                boundary_type="source_declared_fixed_prefix",
                terminal_transition_count=0,
                terminal_reason="not_applicable",
            ),
        )
        prefix = prefix_for_case(case)
        candidate = evaluation(
            CartesianState2D(1.0, 0.0, 1.0, 0.0),
            predicted_ratio=2.0,
            candidate=True,
            allowed=False,
        )
        executor = mock.Mock()
        result = run_discovery_trajectory(
            prefix,
            "zero_action_reference_v0",
            evaluator=mock.Mock(return_value=candidate),
            transition_executor=executor,
        )
        executor.assert_not_called()
        self.assertEqual(result.physical_transition_count, 0)
        self.assertEqual(result.candidate_boundary_count, 1)
        self.assertEqual(result.final_veto_rejection_count, 1)
        self.assertEqual(len(result.records), 1)
        self.assertFalse(result.records[0]["transition_executed"])
        self.assertEqual(result.records[0]["fallback_execution_count"], 0)

    def test_first_candidate_terminates_after_prior_allowed_transition(self) -> None:
        case = SimpleNamespace(
            case_id="case",
            source_configuration_hash="1" * 64,
            transition_implementation_hash="2" * 64,
            nominal_controller_hash="3" * 64,
            nominal_prefix_transition_count=1,
            boundary=SimpleNamespace(
                boundary_type="source_declared_fixed_prefix",
                terminal_transition_count=0,
                terminal_reason="not_applicable",
            ),
        )
        prefix = prefix_for_case(case)
        calls = 0

        def evaluator(document, state, branch_id):
            nonlocal calls
            calls += 1
            return evaluation(
                state,
                predicted_ratio=1.5 if calls == 1 else 2.0,
                candidate=calls == 2,
                allowed=calls == 1,
            )

        def executor(document, item):
            return item.predicted_transition

        result = run_discovery_trajectory(
            prefix,
            "zero_action_reference_v0",
            evaluator=evaluator,
            transition_executor=executor,
        )
        self.assertEqual(result.physical_transition_count, 1)
        self.assertEqual(result.states_evaluated, 2)
        self.assertEqual(result.terminal_reason, "candidate_boundary_detected")

    def test_vetoed_non_candidate_executes_no_fallback(self) -> None:
        case = SimpleNamespace(
            case_id="case",
            source_configuration_hash="1" * 64,
            transition_implementation_hash="2" * 64,
            nominal_controller_hash="3" * 64,
            nominal_prefix_transition_count=1,
            boundary=SimpleNamespace(
                boundary_type="source_declared_fixed_prefix",
                terminal_transition_count=0,
                terminal_reason="not_applicable",
            ),
        )
        prefix = prefix_for_case(case, CartesianState2D(1.0, 0.0, 2.0, 0.0))
        item = evaluation(
            CartesianState2D(1.0, 0.0, 2.0, 0.0),
            predicted_ratio=2.0,
            candidate=False,
            allowed=False,
        )
        executor = mock.Mock()
        result = run_discovery_trajectory(
            prefix,
            "zero_action_reference_v0",
            evaluator=mock.Mock(return_value=item),
            transition_executor=executor,
        )
        executor.assert_not_called()
        self.assertEqual(result.records[0]["fallback_execution_count"], 0)
        self.assertEqual(result.physical_transition_count, 0)

    def test_discovery_authority_fields_are_frozen_false(self) -> None:
        item = evaluation(
            CartesianState2D(1.0, 0.0, 1.0, 0.0),
            predicted_ratio=2.0,
            candidate=True,
            allowed=False,
        )
        self.assertFalse(item.active_authority_granted)
        self.assertEqual(item.hazard_arrest_interventions, 0)
        source = (
            ROOT
            / "runtime_assurance"
            / "stage2a_prediction_boundary_discovery.py"
        ).read_text("utf-8")
        self.assertNotIn("stage2a_hazard_arrest_authority import", source)

    def test_frozen_grid_has_thirteen_cases_and_thirty_nine_trajectories(self) -> None:
        cases = build_source_case_inventory(ROOT)
        definitions = build_trajectory_definitions(cases)
        self.assertEqual(len(cases), 13)
        self.assertEqual(len(definitions), 39)
        self.assertEqual(PLANNED_TRAJECTORY_COUNT, 39)
        self.assertEqual(MAXIMUM_PHYSICAL_TRANSITIONS_PER_TRAJECTORY, 32)
        self.assertEqual(
            tuple(branch for _, branch in definitions[:3]), DISCOVERY_BRANCH_IDS
        )

    def test_trajectory_order_is_deterministic(self) -> None:
        cases = build_source_case_inventory(ROOT)
        first = build_trajectory_definitions(cases)
        second = build_trajectory_definitions(tuple(reversed(cases)))
        self.assertEqual(
            tuple(trajectory_id(case.case_id, branch) for case, branch in first),
            tuple(trajectory_id(case.case_id, branch) for case, branch in second),
        )

    def test_plan_hash_and_sources_validate(self) -> None:
        plan = load_discovery_plan(ROOT)
        validate_discovery_plan(ROOT, plan)
        self.assertEqual(plan["overspeed_threshold"], 1.90)
        self.assertEqual(plan["overspeed_comparator"], ">")
        self.assertFalse(plan["active_authority_granted"])
        self.assertEqual(plan["hazard_arrest_interventions"], 0)

    def test_plan_mutation_invalidates_hash(self) -> None:
        plan = load_discovery_plan(ROOT)
        changed = copy.deepcopy(plan)
        changed["overspeed_threshold"] = 1.89
        with self.assertRaises(PredictionBoundaryDiscoveryError):
            validate_discovery_plan(ROOT, changed)

    def test_state_and_action_hashes_are_deterministic(self) -> None:
        state = CartesianState2D(1.0, 2.0, 3.0, 4.0)
        first = evaluation(
            state, predicted_ratio=1.5, candidate=False, allowed=True
        )
        second = evaluation(
            state, predicted_ratio=1.5, candidate=False, allowed=True
        )
        self.assertEqual(first.current_state_hash, second.current_state_hash)
        self.assertEqual(first.action_hash, second.action_hash)
        self.assertEqual(first.predicted_state_hash, second.predicted_state_hash)

    def test_payloads_are_deterministic_and_exact(self) -> None:
        cases = build_source_case_inventory(ROOT)
        prefixes = tuple(prefix_for_case(case) for case in sorted(cases, key=lambda item: item.case_id))
        trajectories = tuple(
            self._empty_trajectory(prefix, branch)
            for prefix in prefixes
            for branch in DISCOVERY_BRANCH_IDS
        )
        first = build_discovery_payloads(
            ROOT,
            prefixes,
            trajectories,
            implementation_commit="1" * 40,
            protected_before={"protected": "2" * 64},
            protected_after={"protected": "2" * 64},
        )
        second = build_discovery_payloads(
            ROOT,
            prefixes,
            trajectories,
            implementation_commit="1" * 40,
            protected_before={"protected": "2" * 64},
            protected_after={"protected": "2" * 64},
        )
        self.assertEqual(first, second)
        validate_discovery_payloads(first, source_plan=load_discovery_plan(ROOT))
        self.assertEqual(set(first), {
            "candidate_boundaries.json",
            "coverage_summary.json",
            "discovery_manifest.json",
            "discovery_plan.json",
            "near_boundary_diagnostics.json",
            "protected_evidence_report.json",
            "summary.md",
            "trajectory_index.json",
        })

    def test_payload_mutation_is_rejected(self) -> None:
        cases = build_source_case_inventory(ROOT)
        prefixes = tuple(prefix_for_case(case) for case in sorted(cases, key=lambda item: item.case_id))
        trajectories = tuple(
            self._empty_trajectory(prefix, branch)
            for prefix in prefixes
            for branch in DISCOVERY_BRANCH_IDS
        )
        payloads = build_discovery_payloads(
            ROOT,
            prefixes,
            trajectories,
            implementation_commit="1" * 40,
            protected_before={"protected": "2" * 64},
            protected_after={"protected": "2" * 64},
        )
        changed = dict(payloads)
        changed["coverage_summary.json"] += b" "
        with self.assertRaises(PredictionBoundaryDiscoveryError):
            validate_discovery_payloads(changed)

    def test_atomic_publication_rejects_existing_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            target = root / "analysis" / "result"
            target.mkdir(parents=True)
            with self.assertRaises(ShadowCalibrationError):
                atomic_publish_new_directory(
                    root,
                    Path("analysis/result"),
                    {"result.json": b"{}\n"},
                    lambda payloads: None,
                )

    def test_atomic_publication_validator_failure_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            target = root / "analysis" / "result"

            def fail(payloads):
                raise PredictionBoundaryDiscoveryError("staged failure")

            with self.assertRaises(PredictionBoundaryDiscoveryError):
                atomic_publish_new_directory(
                    root,
                    Path("analysis/result"),
                    {"result.json": b"{}\n"},
                    fail,
                )
            self.assertFalse(target.exists())

    def test_default_and_plan_cli_do_not_execute_discovery(self) -> None:
        from scripts import run_stage2a_prediction_boundary_discovery_v0 as cli

        with mock.patch.object(cli, "execute_frozen_discovery") as execute:
            self.assertEqual(cli.main([]), 0)
            self.assertEqual(cli.main(["--plan"]), 0)
        execute.assert_not_called()

    def test_cli_has_no_scientific_override_flags(self) -> None:
        from scripts import run_stage2a_prediction_boundary_discovery_v0 as cli

        parser = cli.build_parser()
        option_strings = {
            value
            for action in parser._actions
            for value in action.option_strings
        }
        for forbidden in (
            "--case",
            "--branch",
            "--threshold",
            "--horizon",
            "--output",
            "--retry",
            "--state",
        ):
            self.assertNotIn(forbidden, option_strings)

    def test_no_result_directory_is_created_by_tests(self) -> None:
        # Lifecycle-aware: this assertion applies before the formal publication only.
        if (ROOT / OUTPUT_PATH).exists():
            self.skipTest("frozen discovery result has already been published")
        self.assertFalse((ROOT / OUTPUT_PATH).exists())

    def test_protected_artifact_paths_are_not_write_targets(self) -> None:
        source = (
            ROOT
            / "runtime_assurance"
            / "stage2a_prediction_boundary_discovery.py"
        ).read_text("utf-8")
        self.assertEqual(source.count("atomic_publish_new_directory"), 0)
        self.assertIn("OUTPUT_PATH = Path(\"analysis/stage2a_prediction_boundary_discovery_v0\")", source)

    @staticmethod
    def _empty_trajectory(
        prefix: PrefixExecutionResult,
        branch: str,
    ) -> DiscoveryTrajectoryResult:
        document = prefix.document()
        boundary_hash = runtime_state_hash(
            CartesianState2D(
                float(document["position_x"]),
                float(document["position_y"]),
                float(document["velocity_x"]),
                float(document["velocity_y"]),
            )
        )
        action_trace_hash = canonical_sha256([])
        state_trace_hash = canonical_sha256([branch_document()])
        identity = {
            "trajectory_id": trajectory_id(prefix.case.case_id, branch),
            "case_id": prefix.case.case_id,
            "branch_id": branch,
            "source_configuration_hash": prefix.case.source_configuration_hash,
            "boundary_type": prefix.boundary_type,
            "prefix_transition_count": prefix.boundary_transition_count,
            "prefix_actual_transition_count": prefix.actual_transition_count,
            "prefix_action_trace_hash": prefix.prefix_action_trace_hash,
            "prefix_state_trace_hash": prefix.prefix_state_trace_hash,
            "source_boundary_state_hash": boundary_hash,
            "action_trace_hash": action_trace_hash,
            "state_trace_hash": state_trace_hash,
            "records": (),
        }
        return DiscoveryTrajectoryResult(
            trajectory_id=trajectory_id(prefix.case.case_id, branch),
            case_id=prefix.case.case_id,
            case_family=prefix.case.case_id.split("__r0_", 1)[0],
            branch_id=branch,
            source_configuration_hash=prefix.case.source_configuration_hash,
            simulator_configuration_hash=str(document["simulator_configuration_hash"]),
            constants_hash=str(document["constants_hash"]),
            transition_implementation_hash=prefix.case.transition_implementation_hash,
            nominal_controller_hash=prefix.case.nominal_controller_hash,
            boundary_type=prefix.boundary_type,
            prefix_transition_count=prefix.boundary_transition_count,
            prefix_actual_transition_count=prefix.actual_transition_count,
            prefix_action_trace_hash=prefix.prefix_action_trace_hash,
            prefix_state_trace_hash=prefix.prefix_state_trace_hash,
            source_boundary_state_hash=boundary_hash,
            records=(),
            action_trace_hash=action_trace_hash,
            state_trace_hash=state_trace_hash,
            source_trajectory_hash=canonical_sha256(identity),
            states_evaluated=0,
            physical_transition_count=0,
            final_veto_rejection_count=0,
            candidate_boundary_count=0,
            terminal_reason="discovery_transition_horizon_complete",
        )


if __name__ == "__main__":
    unittest.main()
