from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from runtime_assurance.final_veto_monitor import FinalVetoDecision
from runtime_assurance.recovery_branch_executor import RecoveryBranchExecutionResult
from runtime_assurance.recovery_experiment_artifacts import (
    RecoveryExperimentBundle,
    canonical_sha256,
    validate_recovery_experiment_bundle,
)
from runtime_assurance.recovery_experiment_preflight import (
    EXPECTED_RUNTIME_STOP_PRIORITY,
    build_artifact_contract,
    load_json_object,
)
from runtime_assurance.recovery_experiment_runner import (
    RecoveryExperimentExecutionError,
    _build_frozen_recovery_bundle,
    _run_branch,
)
from scripts.run_recovery_branch_runner import MAX_RUNNER_HORIZON_STEPS
from simulator.phase34_35_transition import CartesianState2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "analysis"
    / "recovery_action_branching_nonformal_v0"
    / "manifest.json"
)
BRANCH_STATE_PATH = MANIFEST_PATH.with_name("branch_state.json")


def state_hash(state: CartesianState2D) -> str:
    return canonical_sha256(
        {
            "position_x": state.x,
            "position_y": state.y,
            "velocity_x": state.vx,
            "velocity_y": state.vy,
        }
    )


def decision(action, ratio: float, *, veto: bool = False) -> FinalVetoDecision:
    return FinalVetoDecision(
        monitor_id="one_step_overspeed_veto_v0",
        decision="veto" if veto else "allow",
        reason=(
            "predicted_nominal_overspeed"
            if veto
            else "predicted_nominal_within_threshold"
        ),
        threshold=1.90,
        comparator=">",
        nominal_action=action,
        executed_action=(0.0, 0.0) if veto else action,
        fallback_action=(0.0, 0.0),
        predicted_nominal_speed_ratio=ratio,
        predicted_fallback_speed_ratio=1.5 if veto else None,
        fallback_predicted_to_exceed_threshold=False if veto else None,
        veto_applied=veto,
    )


class SyntheticStepExecutor:
    def __init__(self, *, hold_until_horizon: bool = False) -> None:
        self.calls: list[tuple[str, CartesianState2D, int]] = []
        self.branch_document_ids: list[int] = []
        self.hold_until_horizon = hold_until_horizon

    def __call__(
        self,
        branch_state,
        branch_id,
        horizon_steps=1,
        *,
        current_state=None,
    ):
        assert current_state is not None
        self.calls.append((branch_id, current_state, horizon_steps))
        self.branch_document_ids.append(id(branch_state))
        previous_hash = state_hash(current_state)
        if branch_id == "explicit_abort_v0":
            return RecoveryBranchExecutionResult(
                branch_id=branch_id,
                executed=False,
                action=None,
                previous_state_hash=previous_hash,
                next_state_hash=None,
                terminal_reason="explicit_recovery_abort",
                transition_count=0,
                valid=True,
                previous_state=current_state,
                next_state=None,
                monitor_decision=None,
            )
        if branch_id == "velocity_opposed_thrust_v0" and not self.hold_until_horizon:
            action = (-0.25, 0.0)
            predicted = CartesianState2D(
                current_state.x,
                current_state.y,
                current_state.vx,
                current_state.vy + 1000.0,
            )
            return RecoveryBranchExecutionResult(
                branch_id=branch_id,
                executed=False,
                action=action,
                previous_state_hash=previous_hash,
                next_state_hash=None,
                terminal_reason="recovery_action_rejected",
                transition_count=0,
                valid=True,
                previous_state=current_state,
                next_state=None,
                monitor_decision=decision(action, 1.91, veto=True),
                predicted_nominal_state=predicted,
                predicted_fallback_state=current_state,
            )

        action = (0.0, 0.0)
        if self.hold_until_horizon:
            next_state = current_state
            ratio = math.hypot(next_state.vx, next_state.vy) / 4207.165744298648
        elif branch_id == "tangential_error_correction_v0":
            next_state = CartesianState2D(
                x=7.5e12,
                y=0.0,
                vx=0.0,
                vy=4207.165744298648,
            )
            action = (0.0, 0.25)
            ratio = 1.0
        else:
            next_state = CartesianState2D(
                x=3.0 * 7.5e12,
                y=0.0,
                vx=0.0,
                vy=4207.165744298648,
            )
            ratio = 1.0
        return RecoveryBranchExecutionResult(
            branch_id=branch_id,
            executed=True,
            action=action,
            previous_state_hash=previous_hash,
            next_state_hash=state_hash(next_state),
            terminal_reason="one_step_horizon_complete",
            transition_count=1,
            valid=True,
            previous_state=current_state,
            next_state=next_state,
            monitor_decision=decision(action, ratio),
            predicted_nominal_state=next_state,
        )


class RecoveryExperimentRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_json_object(MANIFEST_PATH)
        self.branch_state = load_json_object(BRANCH_STATE_PATH)
        self.contract = build_artifact_contract(
            self.manifest,
            self.branch_state,
            implementation_commit="a" * 40,
        )

    def loader(self):
        return copy.deepcopy(self.branch_state)

    def test_frozen_configuration_is_loaded_from_contract(self) -> None:
        self.assertEqual(self.contract.recovery_horizon, 10_000)
        self.assertEqual(self.contract.total_horizon, 100_000)
        self.assertEqual(self.contract.hazard_threshold, 1.90)
        self.assertEqual(self.contract.hazard_comparator, ">")
        self.assertEqual(self.contract.stop_priority, EXPECTED_RUNTIME_STOP_PRIORITY)

    def test_complete_synthetic_bundle_uses_frozen_order(self) -> None:
        executor = SyntheticStepExecutor()
        bundle = _build_frozen_recovery_bundle(
            contract=self.contract,
            branch_state_loader=self.loader,
            step_executor=executor,
            is_synthetic=True,
        )
        self.assertEqual(
            tuple(record.branch_id for record in bundle.records),
            self.contract.branch_ids,
        )
        report = validate_recovery_experiment_bundle(bundle, self.contract)
        self.assertTrue(report.valid, report.errors)

    def test_every_branch_receives_fresh_canonical_state(self) -> None:
        loaded = []

        def loader():
            value = copy.deepcopy(self.branch_state)
            loaded.append(value)
            return value

        executor = SyntheticStepExecutor()
        _build_frozen_recovery_bundle(
            contract=self.contract,
            branch_state_loader=loader,
            step_executor=executor,
            is_synthetic=True,
        )
        self.assertEqual(len(loaded), 4)
        self.assertEqual(len({id(value) for value in loaded}), 4)
        self.assertTrue(
            all(
                value["canonical_branch_state_hash"]
                == self.contract.branch_state_hash
                for value in loaded
            )
        )

    def test_branch_a_mutation_cannot_affect_branch_b(self) -> None:
        calls = 0

        def loader():
            nonlocal calls
            calls += 1
            value = copy.deepcopy(self.branch_state)
            value["test_instance"] = calls
            value.pop("test_instance")
            return value

        bundle = _build_frozen_recovery_bundle(
            contract=self.contract,
            branch_state_loader=loader,
            step_executor=SyntheticStepExecutor(),
            is_synthetic=True,
        )
        self.assertEqual(calls, 4)
        self.assertEqual(
            {record.branch_state_hash for record in bundle.records},
            {self.contract.branch_state_hash},
        )

    def test_action_rejection_executes_no_transition_or_fallback(self) -> None:
        run = _run_branch(
            branch_state=self.loader(),
            branch_id="velocity_opposed_thrust_v0",
            contract=self.contract,
            step_executor=SyntheticStepExecutor(),
        )
        self.assertEqual(run.record.terminal_reason, "action_rejected")
        self.assertEqual(run.record.recovery_transition_count, 0)
        self.assertFalse(run.events[0].transition_occurred)
        self.assertIsNone(run.events[0].executed_action)

    def test_explicit_abort_executes_zero_transitions(self) -> None:
        run = _run_branch(
            branch_state=self.loader(),
            branch_id="explicit_abort_v0",
            contract=self.contract,
            step_executor=SyntheticStepExecutor(),
        )
        self.assertEqual(run.record.terminal_reason, "explicit_abort")
        self.assertEqual(run.record.recovery_transition_count, 0)
        self.assertEqual(run.events[0].final_veto_decision, "not_applicable")

    def test_scientific_failure_is_a_valid_branch_record(self) -> None:
        run = _run_branch(
            branch_state=self.loader(),
            branch_id="zero_action_reference_v0",
            contract=self.contract,
            step_executor=SyntheticStepExecutor(),
        )
        self.assertTrue(run.record.valid)
        self.assertEqual(run.record.terminal_reason, "instability")
        self.assertEqual(run.record.instability_status, "triggered")

    def test_recovery_success_uses_crossing_and_recoverability(self) -> None:
        run = _run_branch(
            branch_state=self.loader(),
            branch_id="tangential_error_correction_v0",
            contract=self.contract,
            step_executor=SyntheticStepExecutor(),
        )
        self.assertTrue(run.record.crossed_target_radius)
        self.assertTrue(run.record.phase34_compatible_recoverable_crossing)
        self.assertTrue(run.record.recovery_success)
        self.assertEqual(run.record.terminal_reason, "recovery_success")

    def test_exactly_ten_thousand_transitions_are_within_horizon(self) -> None:
        executor = SyntheticStepExecutor(hold_until_horizon=True)
        run = _run_branch(
            branch_state=self.loader(),
            branch_id="zero_action_reference_v0",
            contract=self.contract,
            step_executor=executor,
        )
        self.assertEqual(run.record.recovery_transition_count, 10_000)
        self.assertEqual(run.record.terminal_reason, "recovery_horizon_exhausted")
        self.assertEqual(len(executor.calls), 10_000)

    def test_total_horizon_is_enforced(self) -> None:
        contract = replace(
            self.contract,
            total_horizon=self.contract.nominal_prefix_transition_count + 2,
        )
        run = _run_branch(
            branch_state=self.loader(),
            branch_id="zero_action_reference_v0",
            contract=contract,
            step_executor=SyntheticStepExecutor(hold_until_horizon=True),
        )
        self.assertEqual(run.record.recovery_transition_count, 2)
        self.assertEqual(run.record.terminal_reason, "total_horizon_exhausted")

    def test_executor_state_mismatch_is_infrastructure_failure(self) -> None:
        executor = SyntheticStepExecutor()

        def bad_executor(*args, **kwargs):
            result = executor(*args, **kwargs)
            return replace(result, previous_state_hash="0" * 64)

        with self.assertRaises(RecoveryExperimentExecutionError):
            _run_branch(
                branch_state=self.loader(),
                branch_id="zero_action_reference_v0",
                contract=self.contract,
                step_executor=bad_executor,
            )

    def test_prediction_realization_mismatch_is_infrastructure_failure(self) -> None:
        executor = SyntheticStepExecutor()

        def bad_executor(*args, **kwargs):
            result = executor(*args, **kwargs)
            if result.executed:
                return replace(result, predicted_nominal_state=result.previous_state)
            return result

        with self.assertRaises(RecoveryExperimentExecutionError):
            _run_branch(
                branch_state=self.loader(),
                branch_id="zero_action_reference_v0",
                contract=self.contract,
                step_executor=bad_executor,
            )

    def test_branch_state_hash_mismatch_fails_before_transition(self) -> None:
        branch_state = self.loader()
        branch_state["canonical_branch_state_hash"] = "0" * 64
        executor = SyntheticStepExecutor()
        with self.assertRaises(Exception):
            _run_branch(
                branch_state=branch_state,
                branch_id="zero_action_reference_v0",
                contract=self.contract,
                step_executor=executor,
            )
        self.assertEqual(executor.calls, [])

    def test_bundle_rejects_inconsistent_implementation_commit(self) -> None:
        bundle = _build_frozen_recovery_bundle(
            contract=self.contract,
            branch_state_loader=self.loader,
            step_executor=SyntheticStepExecutor(),
            is_synthetic=True,
        )
        changed = replace(bundle.records[0], implementation_commit="b" * 40)
        invalid = RecoveryExperimentBundle(
            records=(changed,) + bundle.records[1:],
            decision_events=bundle.decision_events,
            is_synthetic=True,
        )
        self.assertFalse(validate_recovery_experiment_bundle(invalid, self.contract).valid)

    def test_bounded_runner_cap_remains_32(self) -> None:
        self.assertEqual(MAX_RUNNER_HORIZON_STEPS, 32)

    def test_real_runner_module_does_not_import_bounded_cap(self) -> None:
        source = (
            PROJECT_ROOT
            / "runtime_assurance"
            / "recovery_experiment_runner.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("MAX_RUNNER_HORIZON_STEPS", source)
        self.assertNotIn("run_recovery_branch_runner", source)

    def test_frozen_input_files_are_not_rewritten(self) -> None:
        before_manifest = MANIFEST_PATH.read_bytes()
        before_state = BRANCH_STATE_PATH.read_bytes()
        _build_frozen_recovery_bundle(
            contract=self.contract,
            branch_state_loader=self.loader,
            step_executor=SyntheticStepExecutor(),
            is_synthetic=True,
        )
        self.assertEqual(MANIFEST_PATH.read_bytes(), before_manifest)
        self.assertEqual(BRANCH_STATE_PATH.read_bytes(), before_state)


if __name__ == "__main__":
    unittest.main()
