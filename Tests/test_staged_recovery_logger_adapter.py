from __future__ import annotations

import json
import math
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from runtime_assurance.final_veto_monitor import FinalVetoDecision
from runtime_assurance.recovery_branch_executor import RecoveryBranchExecutionResult
from runtime_assurance.staged_recovery_logger_adapter import (
    MAX_BOUNDED_VALIDATION_HORIZON,
    ObserverIntegrationError,
    RuntimeSnapshotType,
    Stage0BValidationLoggerAdapter,
    StagedRecoveryLoggerAdapterError,
    build_bounded_runtime_identity,
    run_bounded_recovery_validation_path,
    runtime_state_hash,
)
from runtime_assurance.staged_recovery_runtime_logger import ActionDisposition
from simulator.phase34_35_transition import CartesianState2D


ROOT = Path(__file__).resolve().parents[1]
BRANCH_STATE_PATH = (
    ROOT / "analysis/recovery_action_branching_nonformal_v0/branch_state.json"
)
IMPLEMENTATION = "1" * 40


def branch_state() -> dict[str, object]:
    return json.loads(BRANCH_STATE_PATH.read_text(encoding="utf-8"))


def fake_step_executor(
    document,
    branch_id,
    horizon_steps=1,
    *,
    current_state=None,
):
    state_document = document["state"]
    state = current_state or CartesianState2D(
        state_document["position_x"],
        state_document["position_y"],
        state_document["velocity_x"],
        state_document["velocity_y"],
    )
    speed = math.hypot(state.vx, state.vy)
    action = (-0.25 * state.vx / speed, -0.25 * state.vy / speed)
    next_state = CartesianState2D(
        state.x + state.vx * 0.001,
        state.y + state.vy * 0.001,
        state.vx * 0.9999,
        state.vy * 0.9999,
    )
    constants = document["simulator_configuration"]["simulator_constants"]
    ratio = math.hypot(next_state.vx, next_state.vy) / (
        constants["target_circular_speed"]
        + constants["speed_ratio_denominator_epsilon"]
    )
    decision = FinalVetoDecision(
        monitor_id="one_step_overspeed_veto_v0",
        decision="allow",
        reason="predicted_nominal_within_threshold",
        threshold=1.9,
        comparator=">",
        nominal_action=action,
        executed_action=action,
        fallback_action=(0.0, 0.0),
        predicted_nominal_speed_ratio=ratio,
        predicted_fallback_speed_ratio=None,
        fallback_predicted_to_exceed_threshold=None,
        veto_applied=False,
    )
    return RecoveryBranchExecutionResult(
        branch_id=branch_id,
        executed=True,
        action=action,
        previous_state_hash=runtime_state_hash(state),
        next_state_hash=runtime_state_hash(next_state),
        terminal_reason="one_step_horizon_complete",
        transition_count=1,
        valid=True,
        previous_state=state,
        next_state=next_state,
        monitor_decision=decision,
        predicted_nominal_state=next_state,
    )


class ObserverBoundaryTests(unittest.TestCase):
    def run_path(self, observer=None, horizon=8):
        return run_bounded_recovery_validation_path(
            branch_state(),
            branch_id="velocity_opposed_thrust_v0",
            horizon_steps=horizon,
            implementation_commit=IMPLEMENTATION,
            observer=observer,
            step_executor=fake_step_executor,
        )

    def test_observer_default_is_disabled(self) -> None:
        run = self.run_path()
        self.assertEqual(run.recovery_transition_count, 8)

    def test_disabled_and_enabled_paths_are_identical(self) -> None:
        received = []
        baseline = self.run_path()
        observed = self.run_path(received.append)
        self.assertEqual(baseline, observed)
        self.assertEqual(len(received), 10)

    def test_observer_receives_immutable_snapshot(self) -> None:
        received = []
        self.run_path(received.append)
        with self.assertRaises(FrozenInstanceError):
            received[0].recovery_step = 4

    def test_observer_return_value_is_rejected(self) -> None:
        with self.assertRaises(ObserverIntegrationError):
            self.run_path(lambda _snapshot: "replacement")

    def test_observer_exception_is_infrastructure_failure(self) -> None:
        def broken(_snapshot):
            raise RuntimeError("observer failure")

        with self.assertRaises(ObserverIntegrationError) as context:
            self.run_path(broken)
        self.assertEqual(context.exception.physical_transitions, 0)

    def test_snapshot_sequence_is_initial_transitions_terminal(self) -> None:
        run = self.run_path()
        self.assertEqual(run.snapshots[0].snapshot_type, RuntimeSnapshotType.INITIAL)
        self.assertTrue(
            all(
                item.snapshot_type == RuntimeSnapshotType.TRANSITION
                for item in run.snapshots[1:-1]
            )
        )
        self.assertEqual(run.snapshots[-1].snapshot_type, RuntimeSnapshotType.TERMINAL)

    def test_transition_counters_follow_realized_transitions(self) -> None:
        run = self.run_path()
        self.assertEqual(
            [item.recovery_step for item in run.transition_snapshots],
            list(range(1, 9)),
        )
        prefix = run.identity.nominal_prefix_transition_count
        self.assertEqual(
            [item.total_transition_count for item in run.transition_snapshots],
            list(range(prefix + 1, prefix + 9)),
        )

    def test_terminal_does_not_increment_counters(self) -> None:
        run = self.run_path()
        self.assertEqual(run.snapshots[-1].recovery_step, 8)
        self.assertEqual(
            run.snapshots[-1].total_transition_count,
            run.transition_snapshots[-1].total_transition_count,
        )

    def test_action_monitor_prediction_and_state_are_preserved(self) -> None:
        item = self.run_path().transition_snapshots[0]
        self.assertIsNotNone(item.proposed_action)
        self.assertEqual(item.executed_action, item.proposed_action)
        self.assertEqual(item.monitor_decision, "allow")
        self.assertEqual(item.predicted_next_state, item.realized_next_state)
        self.assertEqual(item.predicted_speed_ratio, item.realized_speed_ratio)
        self.assertEqual(item.action_disposition, ActionDisposition.EXECUTED_UNCHANGED)

    def test_bounded_cap_remains_32(self) -> None:
        self.assertEqual(MAX_BOUNDED_VALIDATION_HORIZON, 32)
        with self.assertRaises(StagedRecoveryLoggerAdapterError):
            self.run_path(horizon=33)

    def test_boolean_horizon_is_rejected(self) -> None:
        with self.assertRaises(StagedRecoveryLoggerAdapterError):
            self.run_path(horizon=True)

    def test_identity_uses_frozen_hashes(self) -> None:
        identity, _ = build_bounded_runtime_identity(
            branch_state(),
            branch_id="velocity_opposed_thrust_v0",
            implementation_commit=IMPLEMENTATION,
        )
        self.assertEqual(
            identity.branch_state_hash,
            "8b017254a8db2584a6732bcd086447ba405cf949d9e932cf03e71543b2cdb898",
        )
        self.assertEqual(identity.nominal_prefix_transition_count, 27)


class Stage0BAdapterTests(unittest.TestCase):
    def test_adapter_constructs_ten_events(self) -> None:
        document = branch_state()
        identity, _ = build_bounded_runtime_identity(
            document,
            branch_id="velocity_opposed_thrust_v0",
            implementation_commit=IMPLEMENTATION,
        )
        adapter = Stage0BValidationLoggerAdapter(
            identity,
            session_id="synthetic_stage0c_fixture",
            max_events=10,
        )
        run = run_bounded_recovery_validation_path(
            document,
            branch_id=identity.branch_id,
            horizon_steps=8,
            implementation_commit=IMPLEMENTATION,
            observer=adapter.observe,
            step_executor=fake_step_executor,
        )
        bundle = adapter.finalize()
        self.assertEqual(len(bundle.events), 10)
        self.assertEqual(bundle.events[-1].terminal_reason, run.validation_terminal_reason)
        self.assertEqual(adapter.runtime_terminal_reason, "recovery_horizon_exhausted")

    def test_stage_phase_and_no_progress_are_not_invented(self) -> None:
        document = branch_state()
        identity, _ = build_bounded_runtime_identity(
            document,
            branch_id="velocity_opposed_thrust_v0",
            implementation_commit=IMPLEMENTATION,
        )
        adapter = Stage0BValidationLoggerAdapter(identity, session_id="fixture", max_events=10)
        run_bounded_recovery_validation_path(
            document,
            branch_id=identity.branch_id,
            horizon_steps=8,
            implementation_commit=IMPLEMENTATION,
            observer=adapter.observe,
            step_executor=fake_step_executor,
        )
        event = adapter.finalize().events[1]
        self.assertEqual(event.phase_evidence, ())
        self.assertEqual(event.pre_observation.field("current_phase").value, None)
        self.assertEqual(event.pre_observation.field("no_progress_status").value, None)

    def test_source_branch_state_is_not_rewritten(self) -> None:
        before = BRANCH_STATE_PATH.read_bytes()
        self.run_adapter_once()
        self.assertEqual(BRANCH_STATE_PATH.read_bytes(), before)

    def run_adapter_once(self):
        document = branch_state()
        identity, _ = build_bounded_runtime_identity(
            document,
            branch_id="velocity_opposed_thrust_v0",
            implementation_commit=IMPLEMENTATION,
        )
        adapter = Stage0BValidationLoggerAdapter(identity, session_id="fixture", max_events=10)
        run_bounded_recovery_validation_path(
            document,
            branch_id=identity.branch_id,
            horizon_steps=8,
            implementation_commit=IMPLEMENTATION,
            observer=adapter.observe,
            step_executor=fake_step_executor,
        )
        return adapter.finalize()


if __name__ == "__main__":
    unittest.main()
