from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

from runtime_assurance.recovery_branch_executor import (
    ACTION_MAGNITUDE,
    SUPPORTED_BRANCH_IDS,
    BranchStateIntegrityError,
    RecoveryBranchExecutorError,
    execute_recovery_branch,
    generate_explicit_abort,
    generate_tangential_correction_action,
    generate_velocity_opposed_action,
    generate_zero_action,
    load_frozen_branch_state,
    validate_branch_state_integrity,
)
from simulator.phase34_35_transition import (
    CartesianState2D,
    NormalizedAction2D,
    Phase3435DynamicsContext,
    step_phase34_35_transition,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RecoveryBranchExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.branch_state = load_frozen_branch_state()

    def test_correct_frozen_branch_state_is_accepted(self) -> None:
        validate_branch_state_integrity(self.branch_state)

    def test_modified_branch_state_without_matching_hash_is_rejected(self) -> None:
        modified = copy.deepcopy(self.branch_state)
        modified["state"]["velocity_x"] += 1.0
        with self.assertRaises(BranchStateIntegrityError):
            execute_recovery_branch(modified, "explicit_abort_v0")

    def test_supported_branch_ids_are_exact(self) -> None:
        self.assertEqual(
            SUPPORTED_BRANCH_IDS,
            {
                "zero_action_reference_v0",
                "velocity_opposed_thrust_v0",
                "tangential_error_correction_v0",
                "explicit_abort_v0",
            },
        )
        with self.assertRaises(RecoveryBranchExecutorError):
            execute_recovery_branch(self.branch_state, "unknown_branch")

    def test_zero_action_is_exact(self) -> None:
        self.assertEqual(generate_zero_action(), (0.0, 0.0))

    def test_velocity_opposed_action_has_exact_direction_and_magnitude(self) -> None:
        state = CartesianState2D(x=1.0, y=0.0, vx=3.0, vy=4.0)
        action = generate_velocity_opposed_action(state)
        self.assertEqual(action, (-0.15, -0.2))
        self.assertEqual(math.hypot(*action), ACTION_MAGNITUDE)
        self.assertLess(action[0] * state.vx + action[1] * state.vy, 0.0)
        self.assertEqual(
            generate_velocity_opposed_action(
                CartesianState2D(x=1.0, y=0.0, vx=0.0, vy=0.0)
            ),
            (0.0, 0.0),
        )

    def test_tangential_action_uses_frozen_sign_convention(self) -> None:
        above_target = CartesianState2D(x=1.0, y=0.0, vx=0.0, vy=2.0)
        below_target = CartesianState2D(x=1.0, y=0.0, vx=0.0, vy=0.0)
        on_target = CartesianState2D(x=1.0, y=0.0, vx=0.0, vy=1.0)
        self.assertEqual(
            generate_tangential_correction_action(above_target, 1.0),
            (0.0, -0.25),
        )
        self.assertEqual(
            generate_tangential_correction_action(below_target, 1.0),
            (0.0, 0.25),
        )
        self.assertEqual(
            generate_tangential_correction_action(on_target, 1.0),
            (0.0, 0.0),
        )

    def test_tangential_action_rejects_zero_position_norm(self) -> None:
        with self.assertRaises(RecoveryBranchExecutorError):
            generate_tangential_correction_action(
                CartesianState2D(x=0.0, y=0.0, vx=1.0, vy=1.0),
                1.0,
            )

    def test_explicit_abort_generates_no_action_or_transition(self) -> None:
        self.assertIsNone(generate_explicit_abort())
        with mock.patch(
            "runtime_assurance.recovery_branch_executor.step_phase34_35_transition"
        ) as transition:
            result = execute_recovery_branch(
                self.branch_state,
                "explicit_abort_v0",
                horizon_steps=1,
            )
        transition.assert_not_called()
        self.assertIs(result.executed, False)
        self.assertIsNone(result.action)
        self.assertIsNone(result.next_state)
        self.assertIsNone(result.next_state_hash)
        self.assertEqual(result.transition_count, 0)
        self.assertEqual(result.terminal_reason, "explicit_recovery_abort")
        self.assertIs(result.valid, True)

    def test_zero_action_execution_matches_existing_transition_exactly(self) -> None:
        state_data = self.branch_state["state"]
        previous_state = CartesianState2D(
            x=state_data["position_x"],
            y=state_data["position_y"],
            vx=state_data["velocity_x"],
            vy=state_data["velocity_y"],
        )
        simulator_configuration = self.branch_state["simulator_configuration"]
        constants = simulator_configuration["simulator_constants"]
        dynamics = Phase3435DynamicsContext(
            mu=constants["mu"],
            dt=constants["dt"],
            mass=constants["mass"],
            thrust_scale=simulator_configuration["thrust_scale"],
        )
        expected = step_phase34_35_transition(
            previous_state,
            NormalizedAction2D(0.0, 0.0),
            dynamics,
        )
        result = execute_recovery_branch(
            self.branch_state,
            "zero_action_reference_v0",
            horizon_steps=1,
        )
        self.assertIs(result.executed, True)
        self.assertEqual(result.action, (0.0, 0.0))
        self.assertEqual(result.next_state, expected.next_state)
        self.assertEqual(result.transition_count, 1)
        self.assertEqual(result.terminal_reason, "one_step_horizon_complete")
        self.assertIsNotNone(result.next_state_hash)
        self.assertEqual(result.monitor_decision.decision, "allow")

    def test_executor_calls_existing_transition_for_prediction_and_execution(self) -> None:
        with mock.patch(
            "runtime_assurance.recovery_branch_executor.step_phase34_35_transition",
            wraps=step_phase34_35_transition,
        ) as transition:
            result = execute_recovery_branch(
                self.branch_state,
                "zero_action_reference_v0",
            )
        self.assertIs(result.executed, True)
        self.assertEqual(transition.call_count, 2)
        self.assertEqual(
            transition.call_args_list[0].args,
            transition.call_args_list[1].args,
        )

    def test_v0_rejects_any_horizon_other_than_one(self) -> None:
        for horizon in (0, 1.0, 2, 10000):
            with self.subTest(horizon=horizon):
                with self.assertRaises(RecoveryBranchExecutorError):
                    execute_recovery_branch(
                        self.branch_state,
                        "zero_action_reference_v0",
                        horizon_steps=horizon,
                    )

    def test_execution_result_is_immutable(self) -> None:
        result = execute_recovery_branch(self.branch_state, "explicit_abort_v0")
        with self.assertRaises(FrozenInstanceError):
            result.valid = False

    def test_executor_import_is_pure_and_does_not_load_phase_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            environment = os.environ.copy()
            existing_pythonpath = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = os.pathsep.join(
                part
                for part in (str(REPOSITORY_ROOT), existing_pythonpath)
                if part
            )
            code = (
                "import json,sys; "
                "import runtime_assurance.recovery_branch_executor; "
                "print(json.dumps({'phase_modules': sorted(name for name in "
                "sys.modules if name.startswith('scripts.explicit_controller_'))}))"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=temporary_directory,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            self.assertEqual(json.loads(completed.stdout)["phase_modules"], [])
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
