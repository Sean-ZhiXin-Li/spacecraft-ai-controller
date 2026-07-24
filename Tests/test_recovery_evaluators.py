from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import unittest
from pathlib import Path

from runtime_assurance.recovery_evaluators import (
    EVALUATION_CLEAR,
    EVALUATION_INVALID,
    EVALUATION_NOT_EVALUATED,
    EVALUATION_TRIGGERED,
    FROZEN_RECOVERY_HORIZON,
    PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
    PHASE34_RECOVERABLE_VR_RATIO_MAX,
    PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
    RecoveryEvaluationResult,
    evaluate_instability,
    evaluate_phase34_compatible_recoverability,
    evaluate_recovery_success_v0,
    evaluate_unsafe_state,
)
from runtime_assurance.recovery_stop_conditions import (
    ACTION_REJECTED,
    CLEAR,
    EXPLICIT_ABORT,
    INSTABILITY,
    INVALID_RECOVERY_EVALUATION,
    INVALID_SIMULATION,
    NOT_EVALUATED,
    OVERSPEED,
    RECOVERY_SUCCESS,
    TRIGGERED,
    UNSAFE_STATE,
    evaluate_recovery_stop_conditions,
)
from scripts.run_recovery_branch_runner import MAX_RUNNER_HORIZON_STEPS
from simulator.phase34_35_transition import CartesianState2D


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ARTIFACT_DIRECTORY = (
    REPOSITORY_ROOT / "analysis" / "recovery_action_branching_nonformal_v0"
)


def _success_evaluation(**overrides: object) -> RecoveryEvaluationResult:
    values: dict[str, object] = {
        "declared_overspeed_occurred": False,
        "simulation_valid": True,
        "recovery_evaluation_valid": True,
        "target_radius_crossing": True,
        "phase34_compatible_recoverable_crossing": True,
        "branch_transition_count": 100,
        "recovery_horizon": FROZEN_RECOVERY_HORIZON,
        "branch_step": 500,
        "crossing_step": 600,
        "explicit_abort": False,
        "action_rejected": False,
        "evaluated_step": 600,
    }
    values.update(overrides)
    return evaluate_recovery_success_v0(**values)  # type: ignore[arg-type]


def _phase34_source_reference():
    phase21_path = (
        REPOSITORY_ROOT
        / "scripts"
        / "explicit_controller_phase21_orbital_transfer_planner.py"
    )
    phase34_path = (
        REPOSITORY_ROOT
        / "scripts"
        / "explicit_controller_phase34_post_cross_sync.py"
    )
    phase21_tree = ast.parse(phase21_path.read_text(encoding="utf-8"))
    constant_names = {
        "RECOVERABLE_R_RATIO",
        "RECOVERABLE_VR_RATIO",
        "RECOVERABLE_VT_RATIO",
    }
    constants: dict[str, float] = {}
    for node in phase21_tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in constant_names:
            constants[target.id] = float(ast.literal_eval(node.value))
    if set(constants) != constant_names:
        raise AssertionError("could not extract Phase34 recoverability constants")

    phase34_tree = ast.parse(phase34_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in phase34_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "recoverable_state"
    )
    namespace: dict[str, object] = dict(constants)
    isolated_module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    exec(compile(isolated_module, str(phase34_path), "exec"), namespace)
    return namespace["recoverable_state"], constants


class RecoveryEvaluationResultTests(unittest.TestCase):
    def test_complete_positive_evidence_returns_recovery_success(self) -> None:
        result = _success_evaluation()
        self.assertEqual(result.status, EVALUATION_TRIGGERED)
        self.assertIs(result.triggered, True)
        self.assertIs(result.required_inputs_present, True)

    def test_result_is_immutable_and_details_are_canonical(self) -> None:
        result = _success_evaluation()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.status = EVALUATION_CLEAR  # type: ignore[misc]
        keys = [key for key, _ in result.details]
        self.assertEqual(keys, sorted(keys))
        json.dumps(dict(result.details), sort_keys=True, allow_nan=False)

    def test_status_and_triggered_value_cannot_disagree(self) -> None:
        with self.assertRaises(ValueError):
            RecoveryEvaluationResult(
                evaluator_id="invalid_test",
                status=EVALUATION_TRIGGERED,
                triggered=False,
                reason="intentional mismatch",
                evidence_level="measured",
                evaluated_step=1,
                required_inputs_present=True,
                details=(),
            )


class Phase34RecoverabilityTests(unittest.TestCase):
    def test_known_positive_boundary_passes(self) -> None:
        result = evaluate_phase34_compatible_recoverability(
            r_error_ratio=PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
            vr_ratio=-PHASE34_RECOVERABLE_VR_RATIO_MAX,
            vt_error_ratio=PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
        )
        self.assertEqual(result.status, EVALUATION_TRIGGERED)

    def test_known_negative_boundary_fails(self) -> None:
        result = evaluate_phase34_compatible_recoverability(
            r_error_ratio=math.nextafter(
                PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX, math.inf
            ),
            vr_ratio=0.0,
            vt_error_ratio=0.0,
        )
        self.assertEqual(result.status, EVALUATION_CLEAR)

    def test_comparisons_are_inclusive_not_strict(self) -> None:
        for values in (
            (
                PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
                PHASE34_RECOVERABLE_VR_RATIO_MAX,
                PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
            ),
            (
                -PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
                -PHASE34_RECOVERABLE_VR_RATIO_MAX,
                -PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
            ),
        ):
            with self.subTest(values=values):
                result = evaluate_phase34_compatible_recoverability(
                    r_error_ratio=values[0],
                    vr_ratio=values[1],
                    vt_error_ratio=values[2],
                )
                self.assertEqual(result.status, EVALUATION_TRIGGERED)

    def test_adapter_matches_existing_phase34_source_on_representative_cases(self) -> None:
        reference, constants = _phase34_source_reference()
        self.assertEqual(
            constants["RECOVERABLE_R_RATIO"],
            PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
        )
        self.assertEqual(
            constants["RECOVERABLE_VR_RATIO"],
            PHASE34_RECOVERABLE_VR_RATIO_MAX,
        )
        self.assertEqual(
            constants["RECOVERABLE_VT_RATIO"],
            PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
        )
        cases = (
            (0.0, 0.0, 0.0),
            (0.0025, -0.02, 0.25),
            (-0.0025, 0.02, -0.25),
            (0.0025000001, 0.0, 0.0),
            (0.0, 0.020000001, 0.0),
            (0.0, 0.0, -0.250000001),
        )
        for values in cases:
            with self.subTest(values=values):
                expected = reference(*values)
                actual = evaluate_phase34_compatible_recoverability(
                    r_error_ratio=values[0],
                    vr_ratio=values[1],
                    vt_error_ratio=values[2],
                )
                self.assertEqual(actual.triggered, expected)

    def test_missing_component_is_not_evaluated(self) -> None:
        result = evaluate_phase34_compatible_recoverability(
            r_error_ratio=0.0,
            vr_ratio=None,
            vt_error_ratio=0.0,
        )
        self.assertEqual(result.status, EVALUATION_NOT_EVALUATED)
        self.assertIsNone(result.triggered)

    def test_nonfinite_component_is_invalid(self) -> None:
        result = evaluate_phase34_compatible_recoverability(
            r_error_ratio=0.0,
            vr_ratio=math.inf,
            vt_error_ratio=0.0,
        )
        self.assertEqual(result.status, EVALUATION_INVALID)


class RecoverySuccessTests(unittest.TestCase):
    def test_no_overspeed_without_crossing_is_not_success(self) -> None:
        result = _success_evaluation(
            target_radius_crossing=False,
            phase34_compatible_recoverable_crossing=False,
            crossing_step=None,
        )
        self.assertEqual(result.status, EVALUATION_CLEAR)
        self.assertEqual(result.reason, "target_radius_crossing_did_not_occur")

    def test_crossing_without_recoverability_is_not_success(self) -> None:
        result = _success_evaluation(
            phase34_compatible_recoverable_crossing=False
        )
        self.assertEqual(result.status, EVALUATION_CLEAR)
        self.assertEqual(
            result.reason, "crossing_did_not_meet_phase34_recoverability"
        )

    def test_crossing_before_branch_point_is_rejected_as_recovery(self) -> None:
        result = _success_evaluation(branch_step=500, crossing_step=499)
        self.assertEqual(result.status, EVALUATION_CLEAR)
        self.assertEqual(result.reason, "crossing_occurred_before_branch_point")

    def test_crossing_at_branch_point_is_allowed(self) -> None:
        result = _success_evaluation(branch_step=500, crossing_step=500)
        self.assertEqual(result.status, EVALUATION_TRIGGERED)

    def test_exactly_frozen_horizon_is_within_horizon(self) -> None:
        result = _success_evaluation(
            branch_transition_count=FROZEN_RECOVERY_HORIZON
        )
        self.assertEqual(result.status, EVALUATION_TRIGGERED)

    def test_after_frozen_horizon_is_not_success(self) -> None:
        result = _success_evaluation(
            branch_transition_count=FROZEN_RECOVERY_HORIZON + 1
        )
        self.assertEqual(result.status, EVALUATION_CLEAR)
        self.assertEqual(
            result.reason, "recovery_target_was_reached_after_frozen_horizon"
        )

    def test_horizon_drift_is_invalid(self) -> None:
        result = _success_evaluation(recovery_horizon=9_999)
        self.assertEqual(result.status, EVALUATION_INVALID)

    def test_invalid_simulation_is_not_success(self) -> None:
        result = _success_evaluation(simulation_valid=False)
        self.assertEqual(result.status, EVALUATION_CLEAR)
        self.assertEqual(result.reason, "simulation_is_invalid")

    def test_invalid_recovery_evaluation_is_not_success(self) -> None:
        result = _success_evaluation(recovery_evaluation_valid=False)
        self.assertEqual(result.status, EVALUATION_CLEAR)
        self.assertEqual(result.reason, "recovery_evaluation_is_invalid")

    def test_explicit_abort_is_not_success(self) -> None:
        result = _success_evaluation(explicit_abort=True)
        self.assertEqual(result.status, EVALUATION_CLEAR)
        self.assertEqual(result.reason, "explicit_abort_is_not_recovery_success")

    def test_action_rejection_is_not_success(self) -> None:
        result = _success_evaluation(action_rejected=True)
        self.assertEqual(result.status, EVALUATION_CLEAR)
        self.assertEqual(result.reason, "action_rejection_is_not_recovery_success")

    def test_horizon_exhaustion_without_crossing_is_clear_not_invalid(self) -> None:
        result = _success_evaluation(
            target_radius_crossing=False,
            phase34_compatible_recoverable_crossing=False,
            crossing_step=None,
            branch_transition_count=FROZEN_RECOVERY_HORIZON,
        )
        self.assertEqual(result.status, EVALUATION_CLEAR)

    def test_missing_recoverability_evidence_is_not_evaluated(self) -> None:
        result = _success_evaluation(
            phase34_compatible_recoverable_crossing=None
        )
        self.assertEqual(result.status, EVALUATION_NOT_EVALUATED)
        self.assertIsNone(result.triggered)

    def test_nonfinite_or_noninteger_count_is_invalid(self) -> None:
        result = _success_evaluation(branch_transition_count=math.nan)
        self.assertEqual(result.status, EVALUATION_INVALID)

    def test_recoverable_crossing_without_crossing_is_invalid(self) -> None:
        result = _success_evaluation(
            target_radius_crossing=False,
            phase34_compatible_recoverable_crossing=True,
            crossing_step=None,
        )
        self.assertEqual(result.status, EVALUATION_INVALID)


class InstabilityEvaluatorTests(unittest.TestCase):
    def test_supported_flag_triggers(self) -> None:
        result = evaluate_instability(instability_flag=True)
        self.assertEqual(result.status, EVALUATION_TRIGGERED)

    def test_supported_termination_reason_triggers(self) -> None:
        for reason in ("out_range", "too_close", "radial_stall"):
            with self.subTest(reason=reason):
                result = evaluate_instability(
                    instability_flag=None,
                    termination_reason=reason,
                )
                self.assertEqual(result.status, EVALUATION_TRIGGERED)

    def test_supported_clear_flag_is_clear(self) -> None:
        result = evaluate_instability(instability_flag=False)
        self.assertEqual(result.status, EVALUATION_CLEAR)

    def test_missing_evidence_is_not_evaluated(self) -> None:
        result = evaluate_instability(instability_flag=None)
        self.assertEqual(result.status, EVALUATION_NOT_EVALUATED)

    def test_overspeed_alone_does_not_imply_instability(self) -> None:
        result = evaluate_instability(
            instability_flag=None,
            terminal_label="overspeed",
            termination_reason="overspeed",
        )
        self.assertEqual(result.status, EVALUATION_NOT_EVALUATED)

    def test_contradictory_evidence_is_invalid(self) -> None:
        result = evaluate_instability(
            instability_flag=False,
            terminal_label="instability",
        )
        self.assertEqual(result.status, EVALUATION_INVALID)


class UnsafeStateEvaluatorTests(unittest.TestCase):
    def test_supported_flag_triggers(self) -> None:
        result = evaluate_unsafe_state(unsafe_state_flag=True)
        self.assertEqual(result.status, EVALUATION_TRIGGERED)

    def test_supported_terminal_label_triggers(self) -> None:
        result = evaluate_unsafe_state(
            unsafe_state_flag=None,
            terminal_label="unsafe_state",
        )
        self.assertEqual(result.status, EVALUATION_TRIGGERED)

    def test_supported_clear_flag_is_clear(self) -> None:
        result = evaluate_unsafe_state(unsafe_state_flag=False)
        self.assertEqual(result.status, EVALUATION_CLEAR)

    def test_missing_instrumentation_is_not_evaluated(self) -> None:
        result = evaluate_unsafe_state(unsafe_state_flag=None)
        self.assertEqual(result.status, EVALUATION_NOT_EVALUATED)

    def test_overspeed_does_not_imply_unsafe_state(self) -> None:
        result = evaluate_unsafe_state(
            unsafe_state_flag=None,
            overspeed=True,
            terminal_label="overspeed",
        )
        self.assertEqual(result.status, EVALUATION_NOT_EVALUATED)

    def test_simulator_failure_does_not_imply_unsafe_state(self) -> None:
        result = evaluate_unsafe_state(
            unsafe_state_flag=None,
            simulator_success=False,
        )
        self.assertEqual(result.status, EVALUATION_NOT_EVALUATED)

    def test_explicit_abort_does_not_imply_unsafe_state(self) -> None:
        result = evaluate_unsafe_state(
            unsafe_state_flag=None,
            explicit_abort=True,
        )
        self.assertEqual(result.status, EVALUATION_NOT_EVALUATED)

    def test_contradictory_evidence_is_invalid(self) -> None:
        result = evaluate_unsafe_state(
            unsafe_state_flag=False,
            terminal_label="unsafe_state",
        )
        self.assertEqual(result.status, EVALUATION_INVALID)


class StopConditionIntegrationTests(unittest.TestCase):
    def _evaluate(self, **overrides: object):
        values: dict[str, object] = {
            "execution_terminal_reason": "one_step_horizon_complete",
            "next_state": CartesianState2D(1.0, 0.0, 0.0, 1.0),
            "realized_speed_ratio": 1.0,
            "overspeed_threshold": 1.90,
            "recovery_transition_count": 1,
            "recovery_horizon_steps": 32,
            "total_transition_count": 501,
            "total_horizon_steps": 100_000,
        }
        values.update(overrides)
        return evaluate_recovery_stop_conditions(**values)  # type: ignore[arg-type]

    def test_overspeed_beats_simultaneous_recovery_success(self) -> None:
        report = self._evaluate(
            realized_speed_ratio=1.900001,
            recovery_success_evaluation=_success_evaluation(),
        )
        self.assertEqual(report.terminal_reason, OVERSPEED)
        self.assertEqual(report.status_for(RECOVERY_SUCCESS), CLEAR)

    def test_invalid_simulation_beats_recovery_success(self) -> None:
        report = self._evaluate(
            next_state=CartesianState2D(math.nan, 0.0, 0.0, 1.0),
            recovery_success_evaluation=_success_evaluation(),
        )
        self.assertEqual(report.terminal_reason, INVALID_SIMULATION)
        self.assertEqual(report.status_for(RECOVERY_SUCCESS), CLEAR)

    def test_action_rejection_terminates_without_recovery_success(self) -> None:
        report = self._evaluate(
            execution_terminal_reason="recovery_action_rejected",
            next_state=None,
            realized_speed_ratio=None,
            recovery_transition_count=0,
            total_transition_count=500,
            recovery_success_evaluation=_success_evaluation(),
        )
        self.assertEqual(report.terminal_reason, ACTION_REJECTED)
        self.assertEqual(report.status_for(RECOVERY_SUCCESS), CLEAR)

    def test_explicit_abort_terminates_without_recovery_success(self) -> None:
        report = self._evaluate(
            execution_terminal_reason="explicit_recovery_abort",
            next_state=None,
            realized_speed_ratio=None,
            recovery_transition_count=0,
            total_transition_count=500,
            recovery_success_evaluation=_success_evaluation(),
        )
        self.assertEqual(report.terminal_reason, EXPLICIT_ABORT)
        self.assertEqual(report.status_for(RECOVERY_SUCCESS), CLEAR)

    def test_not_evaluated_does_not_trigger_stop(self) -> None:
        report = self._evaluate(
            recovery_success_evaluation=_success_evaluation(
                phase34_compatible_recoverable_crossing=None
            )
        )
        self.assertIsNone(report.terminal_reason)
        self.assertEqual(report.status_for(RECOVERY_SUCCESS), NOT_EVALUATED)

    def test_complete_recovery_success_triggers_stop(self) -> None:
        report = self._evaluate(
            recovery_success_evaluation=_success_evaluation()
        )
        self.assertEqual(report.terminal_reason, RECOVERY_SUCCESS)
        self.assertEqual(report.status_for(RECOVERY_SUCCESS), TRIGGERED)

    def test_invalid_evaluator_evidence_triggers_invalid_recovery_stop(self) -> None:
        report = self._evaluate(
            recovery_success_evaluation=_success_evaluation(recovery_horizon=9_999)
        )
        self.assertEqual(report.terminal_reason, INVALID_RECOVERY_EVALUATION)
        self.assertEqual(
            report.status_for(INVALID_RECOVERY_EVALUATION), TRIGGERED
        )

    def test_instability_and_unsafe_state_use_frozen_priority(self) -> None:
        report = self._evaluate(
            instability_evaluation=evaluate_instability(instability_flag=True),
            unsafe_state_evaluation=evaluate_unsafe_state(unsafe_state_flag=True),
        )
        self.assertEqual(report.terminal_reason, INSTABILITY)
        self.assertEqual(report.status_for(INSTABILITY), TRIGGERED)
        self.assertEqual(report.status_for(UNSAFE_STATE), TRIGGERED)


class RepositorySafetyTests(unittest.TestCase):
    def test_evaluator_module_imports_no_rollout_or_physics_modules(self) -> None:
        path = REPOSITORY_ROOT / "runtime_assurance" / "recovery_evaluators.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertTrue(
            imported_roots.isdisjoint(
                {"controller", "envs", "scripts", "simulator"}
            )
        )

    def test_evaluator_calls_do_not_modify_protected_inputs(self) -> None:
        protected = (
            RECOVERY_ARTIFACT_DIRECTORY / "manifest.json",
            RECOVERY_ARTIFACT_DIRECTORY / "branch_state.json",
            REPOSITORY_ROOT
            / "analysis"
            / "final_veto_ablation_v0"
            / "manifest.json",
        )
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected
        }
        _success_evaluation()
        evaluate_instability(instability_flag=False)
        evaluate_unsafe_state(unsafe_state_flag=None)
        after = {
            path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected
        }
        self.assertEqual(after, before)

    def test_runner_horizon_remains_capped_at_32(self) -> None:
        self.assertEqual(MAX_RUNNER_HORIZON_STEPS, 32)

    def test_recovery_output_paths_are_absent_or_complete(self) -> None:
        names = (
            "results.csv",
            "decision_log.jsonl",
            "summary.md",
            "comparison.png",
        )
        existing = {
            name
            for name in names
            if (RECOVERY_ARTIFACT_DIRECTORY / name).is_file()
        }
        self.assertIn(existing, (set(), set(names)))
        for name in existing:
            with self.subTest(name=name):
                self.assertGreater((RECOVERY_ARTIFACT_DIRECTORY / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
