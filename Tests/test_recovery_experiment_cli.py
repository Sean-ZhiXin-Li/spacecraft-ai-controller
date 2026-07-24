from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

from runtime_assurance.recovery_experiment_runner import (
    RecoveryExperimentExecutionError,
)
from scripts import run_recovery_experiment_v0 as cli


def fake_contract():
    return SimpleNamespace(
        experiment_id="recovery_action_branching_nonformal_v0",
        case_id="frozen_case",
        branch_state_hash="a" * 64,
        branch_ids=(
            "zero_action_reference_v0",
            "velocity_opposed_thrust_v0",
            "tangential_error_correction_v0",
            "explicit_abort_v0",
        ),
        recovery_horizon=10_000,
        total_horizon=100_000,
        hazard_comparator=">",
        hazard_threshold=1.90,
        stop_priority=("invalid_simulation", "recovery_success"),
        output_filenames=(
            "results.csv",
            "decision_log.jsonl",
            "summary.md",
            "comparison.png",
        ),
    )


class RecoveryExperimentCliTests(unittest.TestCase):
    def invoke(self, argv):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    @mock.patch.object(cli, "run_frozen_recovery_experiment")
    def test_default_invocation_executes_nothing(self, execute) -> None:
        code, output, _ = self.invoke([])
        self.assertEqual(code, 2)
        self.assertIn("No recovery transition executed", output)
        execute.assert_not_called()

    @mock.patch.object(cli, "run_frozen_recovery_experiment")
    @mock.patch.object(cli, "run_recovery_experiment_preflight")
    def test_plan_executes_nothing(self, preflight, execute) -> None:
        preflight.return_value = (SimpleNamespace(ready=True), fake_contract())
        code, output, _ = self.invoke(["--plan"])
        self.assertEqual(code, 0)
        self.assertIn("EXECUTION_DISABLED true", output)
        execute.assert_not_called()

    @mock.patch.object(cli, "preflight_report_lines", return_value=("READY true",))
    @mock.patch.object(cli, "run_frozen_recovery_experiment")
    @mock.patch.object(cli, "run_recovery_experiment_preflight")
    def test_validate_only_executes_nothing(self, preflight, execute, _lines) -> None:
        preflight.return_value = (SimpleNamespace(ready=True), fake_contract())
        code, output, _ = self.invoke(["--validate-only"])
        self.assertEqual(code, 0)
        self.assertIn("No recovery transition executed", output)
        execute.assert_not_called()

    @mock.patch.object(cli, "preflight_report_lines", return_value=("READY true",))
    @mock.patch.object(cli, "run_frozen_recovery_experiment")
    @mock.patch.object(cli, "run_recovery_experiment_preflight")
    def test_preflight_executes_nothing(self, preflight, execute, _lines) -> None:
        preflight.return_value = (SimpleNamespace(ready=True), fake_contract())
        code, output, _ = self.invoke(["--experiment-preflight"])
        self.assertEqual(code, 0)
        self.assertIn("No recovery transition executed", output)
        execute.assert_not_called()
        self.assertTrue(preflight.call_args.kwargs["require_clean_repository"])

    def test_parser_exposes_no_scientific_override(self) -> None:
        options = {action.dest for action in cli.build_parser()._actions}
        for forbidden in (
            "branch",
            "seed",
            "threshold",
            "action_magnitude",
            "recovery_horizon",
            "total_horizon",
            "retry",
            "resume",
        ):
            self.assertNotIn(forbidden, options)

    @mock.patch.object(cli, "run_frozen_recovery_experiment")
    def test_execute_mode_invokes_real_api_exactly_once(self, execute) -> None:
        record = SimpleNamespace(
            branch_id="explicit_abort_v0",
            terminal_reason="explicit_abort",
            recovery_transition_count=0,
            overspeed_status="clear",
            crossed_target_radius=False,
            phase34_compatible_recoverable_crossing=False,
            recovery_success=False,
        )
        execute.return_value = SimpleNamespace(
            preflight_report=SimpleNamespace(implementation_commit="a" * 40),
            bundle=SimpleNamespace(records=(record,), decision_events=(object(),)),
            publication=SimpleNamespace(artifact_hashes=(("results.csv", "b" * 64),)),
        )
        code, output, _ = self.invoke(["--execute-frozen-experiment"])
        self.assertEqual(code, 0)
        self.assertIn("EXECUTION_COMPLETE true", output)
        execute.assert_called_once()

    @mock.patch.object(cli, "run_frozen_recovery_experiment")
    def test_execution_failure_is_not_retried(self, execute) -> None:
        execute.side_effect = RecoveryExperimentExecutionError(
            "synthetic failure",
            branch_id="zero_action_reference_v0",
            attempted_recovery_step=3,
            realized_transitions=2,
        )
        code, _, error = self.invoke(["--execute-frozen-experiment"])
        self.assertEqual(code, 1)
        self.assertIn("No automatic retry performed", error)
        execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
