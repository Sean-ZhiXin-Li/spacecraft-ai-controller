from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from runtime_assurance.recovery_branch_executor import (
    DEFAULT_BRANCH_STATE_PATH,
    BranchStateIntegrityError,
    execute_recovery_branch,
    load_frozen_branch_state,
)
from runtime_assurance.recovery_stop_conditions import (
    FROZEN_STOP_LABELS,
    NOT_EVALUATED,
    RECOVERY_SUCCESS,
)
from scripts.run_recovery_branch_runner import (
    DIAGNOSTIC_LOG_NAME,
    MAX_RUNNER_HORIZON_STEPS,
    RecoveryBranchRunnerError,
    build_recovery_branch_plan,
    main,
    run_recovery_branch,
    validate_output_directory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ZERO_BRANCH = "zero_action_reference_v0"


class RecoveryBranchRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.branch_state = load_frozen_branch_state()

    def test_valid_frozen_branch_state_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            plan = build_recovery_branch_plan(
                ZERO_BRANCH,
                branch_state_path=DEFAULT_BRANCH_STATE_PATH,
                horizon_steps=1,
                output_dir=temporary_directory,
            )
        self.assertEqual(plan["branch_id"], ZERO_BRANCH)
        self.assertIs(plan["execution_authorized"], False)
        self.assertEqual(plan["recovery_success_evaluation"], NOT_EVALUATED)

    def test_modified_branch_state_hash_is_rejected(self) -> None:
        modified = copy.deepcopy(self.branch_state)
        modified["state"]["velocity_y"] += 1.0
        with tempfile.TemporaryDirectory() as temporary_directory:
            branch_state_path = Path(temporary_directory) / "branch_state.json"
            branch_state_path.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaises(BranchStateIntegrityError):
                build_recovery_branch_plan(
                    ZERO_BRANCH,
                    branch_state_path=branch_state_path,
                    horizon_steps=1,
                    output_dir=Path(temporary_directory) / "output",
                )

    def test_plan_mode_does_not_execute_or_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "planned_output"
            stdout = io.StringIO()
            with mock.patch(
                "scripts.run_recovery_branch_runner.execute_recovery_branch"
            ) as execute, redirect_stdout(stdout):
                return_code = main(
                    [
                        "--plan",
                        "--branch-id",
                        ZERO_BRANCH,
                        "--branch-state",
                        str(DEFAULT_BRANCH_STATE_PATH),
                        "--horizon-steps",
                        "2",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(return_code, 0)
            execute.assert_not_called()
            self.assertFalse(output_dir.exists())
            self.assertIn("No rollout executed.", stdout.getvalue())

    def test_validate_only_does_not_execute_or_create_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "validated_output"
            stdout = io.StringIO()
            with mock.patch(
                "scripts.run_recovery_branch_runner.execute_recovery_branch"
            ) as execute, redirect_stdout(stdout):
                return_code = main(
                    [
                        "--validate-only",
                        "--branch-state",
                        str(DEFAULT_BRANCH_STATE_PATH),
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(return_code, 0)
            execute.assert_not_called()
            self.assertFalse(output_dir.exists())
            self.assertIn("VALIDATION PASS", stdout.getvalue())

    def test_default_cli_invocation_does_not_execute(self) -> None:
        stdout = io.StringIO()
        with mock.patch(
            "scripts.run_recovery_branch_runner.execute_recovery_branch"
        ) as execute, redirect_stdout(stdout):
            return_code = main([])
        self.assertEqual(return_code, 0)
        execute.assert_not_called()
        self.assertIn("No rollout executed.", stdout.getvalue())

    def test_one_step_runner_matches_executor(self) -> None:
        expected = execute_recovery_branch(self.branch_state, ZERO_BRANCH)
        with tempfile.TemporaryDirectory() as temporary_directory:
            actual = run_recovery_branch(
                ZERO_BRANCH,
                branch_state_path=DEFAULT_BRANCH_STATE_PATH,
                horizon_steps=1,
                output_dir=temporary_directory,
            )
        self.assertEqual(actual.transition_count, expected.transition_count)
        self.assertEqual(len(actual.records), 1)
        self.assertEqual(actual.records[0].action, expected.action)
        self.assertEqual(actual.records[0].state_hash, expected.next_state_hash)
        self.assertEqual(actual.terminal_reason, "recovery_horizon_exhausted")
        self.assertEqual(actual.recovery_success_status, NOT_EVALUATED)

    def test_two_step_runner_is_bounded_and_uses_one_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_recovery_branch(
                ZERO_BRANCH,
                branch_state_path=DEFAULT_BRANCH_STATE_PATH,
                horizon_steps=2,
                output_dir=temporary_directory,
            )
        self.assertEqual(result.branch_id, ZERO_BRANCH)
        self.assertEqual(result.transition_count, 2)
        self.assertEqual(len(result.records), 2)
        self.assertTrue(all(record.branch_id == ZERO_BRANCH for record in result.records))
        self.assertEqual(result.terminal_reason, "recovery_horizon_exhausted")

    def test_invalid_and_long_horizons_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            for horizon in (0, 1.0, MAX_RUNNER_HORIZON_STEPS + 1, 10000):
                with self.subTest(horizon=horizon):
                    with self.assertRaises(RecoveryBranchRunnerError):
                        run_recovery_branch(
                            ZERO_BRANCH,
                            branch_state_path=DEFAULT_BRANCH_STATE_PATH,
                            horizon_steps=horizon,
                            output_dir=temporary_directory,
                        )

    def test_explicit_abort_has_no_transition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_recovery_branch(
                "explicit_abort_v0",
                branch_state_path=DEFAULT_BRANCH_STATE_PATH,
                horizon_steps=1,
                output_dir=temporary_directory,
            )
        self.assertEqual(result.transition_count, 0)
        self.assertEqual(result.terminal_reason, "explicit_abort")
        self.assertEqual(len(result.records), 1)
        self.assertIs(result.records[0].transition_executed, False)
        self.assertIsNone(result.records[0].action)

    def test_stop_labels_are_exact_and_recovery_success_is_not_evaluated(self) -> None:
        self.assertEqual(
            FROZEN_STOP_LABELS,
            (
                "recovery_success",
                "overspeed",
                "instability",
                "unsafe_state",
                "invalid_simulation",
                "invalid_recovery_evaluation",
                "action_rejected",
                "explicit_abort",
                "recovery_horizon_exhausted",
                "total_horizon_exhausted",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_recovery_branch(
                ZERO_BRANCH,
                branch_state_path=DEFAULT_BRANCH_STATE_PATH,
                horizon_steps=1,
                output_dir=temporary_directory,
            )
        self.assertEqual(result.recovery_success_status, NOT_EVALUATED)
        self.assertEqual(RECOVERY_SUCCESS, "recovery_success")

    def test_diagnostic_jsonl_is_explicit_bounded_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_recovery_branch(
                ZERO_BRANCH,
                branch_state_path=DEFAULT_BRANCH_STATE_PATH,
                horizon_steps=1,
                output_dir=temporary_directory,
                write_diagnostics=True,
            )
            expected_path = Path(temporary_directory) / DIAGNOSTIC_LOG_NAME
            self.assertEqual(result.diagnostic_log_path, expected_path)
            lines = expected_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertIs(payload["is_formal_experiment"], False)
            self.assertEqual(payload["branch_id"], ZERO_BRANCH)
            self.assertNotIn("recovery_success", payload)
            self.assertNotIn("comparison", payload)
            self.assertEqual(
                lines[0],
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )

    def test_protected_and_formal_output_paths_are_rejected(self) -> None:
        protected_paths = (
            REPOSITORY_ROOT / "analysis" / "final_veto_ablation_v0",
            REPOSITORY_ROOT
            / "analysis"
            / "recovery_action_branching_nonformal_v0",
            REPOSITORY_ROOT / "analysis" / "phase34_post_cross_sync",
        )
        for path in protected_paths:
            with self.subTest(path=path):
                with self.assertRaises(RecoveryBranchRunnerError):
                    validate_output_directory(path)

    def test_runner_owns_no_controller_or_physics_implementation(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts" / "run_recovery_branch_runner.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("explicit_controller_phase", source)
        self.assertNotIn("step_phase34_35_transition", source)
        self.assertNotIn("gravitational", source.lower())
        self.assertNotIn("thrust acceleration", source.lower())


if __name__ == "__main__":
    unittest.main()
