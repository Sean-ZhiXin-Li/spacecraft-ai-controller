from __future__ import annotations

import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime_assurance.recovery_evaluators import (
    INSTABILITY_EVALUATOR_ID,
    PHASE34_RECOVERABILITY_EVALUATOR_ID,
    RECOVERY_SUCCESS_EVALUATOR_ID,
)
from runtime_assurance.recovery_experiment_preflight import (
    EXPECTED_RUNTIME_STOP_PRIORITY,
    FROZEN_BRANCH_STATE_CANONICAL_HASH,
    FROZEN_MANIFEST_CANONICAL_HASH,
    RecoveryRepositoryState,
    preflight_report_lines,
    run_recovery_experiment_preflight,
)
from scripts.check_recovery_experiment_preflight import main as preflight_main
from scripts.check_recovery_branch_state import attach_canonical_branch_state_hash
from scripts.run_recovery_branch_runner import MAX_RUNNER_HORIZON_STEPS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "analysis"
    / "recovery_action_branching_nonformal_v0"
    / "manifest.json"
)
BRANCH_STATE_PATH = MANIFEST_PATH.with_name("branch_state.json")


def load_json(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


class RecoveryExperimentPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_json(MANIFEST_PATH)
        self.branch_state = load_json(BRANCH_STATE_PATH)
        self.repository_state = RecoveryRepositoryState(
            is_git_worktree=True,
            head_commit="a" * 40,
            tracked_clean=True,
            staged_clean=True,
            untracked_paths=(),
        )
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_report(
        self,
        *,
        manifest=None,
        branch_state=None,
        repository_state=None,
        require_clean=False,
        **kwargs,
    ):
        report, contract = run_recovery_experiment_preflight(
            repository_root=self.root,
            manifest_data=manifest or self.manifest,
            branch_state_data=branch_state or self.branch_state,
            repository_state=repository_state or self.repository_state,
            require_clean_repository=require_clean,
            **kwargs,
        )
        return report, contract

    @staticmethod
    def check(report, check_id):
        return next(check for check in report.checks if check.check_id == check_id)

    def test_valid_frozen_static_configuration_passes_validate_only(self) -> None:
        report, contract = self.run_report()
        self.assertTrue(report.ready, report.failed_check_ids)
        self.assertIsNotNone(contract)
        self.assertEqual(report.manifest_hash, FROZEN_MANIFEST_CANONICAL_HASH)
        self.assertEqual(report.branch_state_hash, FROZEN_BRANCH_STATE_CANONICAL_HASH)

    def test_clean_committed_synthetic_repository_state_passes_full_preflight(self) -> None:
        report, _ = self.run_report(require_clean=True)
        self.assertTrue(report.ready, report.failed_check_ids)
        self.assertTrue(report.working_tree_clean)

    def test_branch_state_hash_mismatch_fails(self) -> None:
        branch_state = copy.deepcopy(self.branch_state)
        branch_state["state"]["position_x"] += 1.0
        report, _ = self.run_report(branch_state=branch_state)
        self.assertEqual(self.check(report, "branch_state_hash_recomputation").status, "failed")

    def test_recomputed_but_changed_branch_state_still_fails_frozen_hash(self) -> None:
        branch_state = copy.deepcopy(self.branch_state)
        branch_state["state"]["position_x"] += 1.0
        branch_state = attach_canonical_branch_state_hash(branch_state)
        report, _ = self.run_report(branch_state=branch_state)
        self.assertEqual(self.check(report, "branch_state_hash").status, "failed")

    def test_manifest_hash_mismatch_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["notes"].append("mutation")
        report, _ = self.run_report(manifest=manifest)
        self.assertEqual(self.check(report, "manifest_hash").status, "failed")

    def test_missing_branch_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["branches"].pop()
        report, _ = self.run_report(manifest=manifest)
        self.assertEqual(self.check(report, "four_branch_manifest").status, "failed")

    def test_duplicate_branch_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["branches"][-1] = copy.deepcopy(manifest["branches"][0])
        report, _ = self.run_report(manifest=manifest)
        self.assertEqual(self.check(report, "four_branch_manifest").status, "failed")

    def test_undeclared_fifth_branch_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        extra = copy.deepcopy(manifest["branches"][0])
        extra["branch_id"] = "undeclared_branch_v0"
        manifest["branches"].append(extra)
        report, _ = self.run_report(manifest=manifest)
        self.assertEqual(self.check(report, "four_branch_manifest").status, "failed")

    def test_action_magnitude_drift_fails_manifest_contract(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["branches"][1]["action_magnitude_before_clipping"] = 0.3
        report, _ = self.run_report(manifest=manifest)
        self.assertEqual(self.check(report, "manifest_contract").status, "failed")

    def test_recovery_horizon_drift_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["horizons"]["recovery_horizon_realized_transitions"] = 9999
        report, _ = self.run_report(manifest=manifest)
        self.assertEqual(self.check(report, "scientific_horizons").status, "failed")

    def test_total_horizon_drift_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["horizons"]["total_episode_horizon_realized_transitions"] = 99999
        report, _ = self.run_report(manifest=manifest)
        self.assertEqual(self.check(report, "scientific_horizons").status, "failed")

    def test_threshold_drift_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["hazard"]["threshold"] = 1.91
        report, _ = self.run_report(manifest=manifest)
        self.assertEqual(self.check(report, "hazard_contract").status, "failed")

    def test_comparator_drift_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["hazard"]["comparator"] = ">="
        report, _ = self.run_report(manifest=manifest)
        self.assertEqual(self.check(report, "hazard_contract").status, "failed")

    def test_stop_priority_drift_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        priority = manifest["stop_conditions"]["priority_order"]
        priority[2], priority[3] = priority[3], priority[2]
        report, _ = self.run_report(manifest=manifest)
        self.assertEqual(self.check(report, "manifest_stop_priority").status, "failed")

    def test_runtime_stop_priority_override_drift_fails(self) -> None:
        drift = EXPECTED_RUNTIME_STOP_PRIORITY[:-1]
        report, _ = self.run_report(expected_runtime_stop_priority=drift)
        self.assertEqual(self.check(report, "runtime_stop_priority").status, "failed")

    def test_unavailable_evaluator_fails(self) -> None:
        available = frozenset(
            {
                RECOVERY_SUCCESS_EVALUATOR_ID,
                PHASE34_RECOVERABILITY_EVALUATOR_ID,
                INSTABILITY_EVALUATOR_ID,
            }
        )
        report, _ = self.run_report(available_evaluator_ids=available)
        self.assertEqual(self.check(report, "evaluator_imports").status, "failed")

    def test_dirty_tracked_worktree_fails_experiment_preflight(self) -> None:
        state = replace_repository_state(self.repository_state, tracked_clean=False)
        report, _ = self.run_report(
            repository_state=state, require_clean=True
        )
        self.assertEqual(self.check(report, "tracked_tree_clean").status, "failed")

    def test_staged_change_fails_experiment_preflight(self) -> None:
        state = replace_repository_state(self.repository_state, staged_clean=False)
        report, _ = self.run_report(
            repository_state=state, require_clean=True
        )
        self.assertEqual(self.check(report, "staging_area_clean").status, "failed")

    def test_unrelated_untracked_file_does_not_fail(self) -> None:
        state = replace_repository_state(
            self.repository_state,
            untracked_paths=("paper/submission.pdf",),
        )
        report, _ = self.run_report(repository_state=state, require_clean=True)
        self.assertTrue(report.ready, report.failed_check_ids)

    def test_untracked_reserved_output_collision_fails(self) -> None:
        state = replace_repository_state(
            self.repository_state,
            untracked_paths=(
                "analysis/recovery_action_branching_nonformal_v0/results.csv",
            ),
        )
        report, _ = self.run_report(repository_state=state)
        self.assertEqual(
            self.check(report, "untracked_output_collisions").status, "failed"
        )

    def test_existing_reserved_artifact_fails(self) -> None:
        path = (
            self.root
            / "analysis"
            / "recovery_action_branching_nonformal_v0"
            / "results.csv"
        )
        path.parent.mkdir(parents=True)
        path.write_text("collision", encoding="utf-8")
        report, _ = self.run_report()
        self.assertEqual(self.check(report, "output_collisions").status, "failed")

    def test_protected_output_directory_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["output_contract"]["base_directory"] = (
            "analysis/phase34_post_cross_sync/"
        )
        report, _ = self.run_report(manifest=manifest)
        self.assertEqual(self.check(report, "output_location").status, "failed")

    def test_preflight_report_order_is_deterministic(self) -> None:
        first, _ = self.run_report()
        second, _ = self.run_report()
        self.assertEqual(
            tuple(check.check_id for check in first.checks),
            tuple(check.check_id for check in second.checks),
        )
        self.assertEqual(preflight_report_lines(first), preflight_report_lines(second))

    def test_existing_bounded_runner_cap_remains_32(self) -> None:
        self.assertEqual(MAX_RUNNER_HORIZON_STEPS, 32)

    def test_preflight_confirms_no_branch_selection_api(self) -> None:
        report, _ = self.run_report()
        self.assertEqual(self.check(report, "branch_selection_absent").status, "passed")

    def test_plan_mode_calls_no_executor_transition(self) -> None:
        output = io.StringIO()
        with mock.patch(
            "runtime_assurance.recovery_branch_executor.execute_recovery_branch"
        ) as execute, contextlib.redirect_stdout(output):
            status = preflight_main(["--plan"])
        self.assertEqual(status, 0)
        execute.assert_not_called()
        self.assertIn("EXECUTION_DISABLED true", output.getvalue())

    def test_validate_only_calls_no_executor_transition(self) -> None:
        with mock.patch(
            "runtime_assurance.recovery_branch_executor.execute_recovery_branch"
        ) as execute, contextlib.redirect_stdout(io.StringIO()):
            status = preflight_main(["--validate-only"])
        self.assertEqual(status, 0)
        execute.assert_not_called()

    def test_experiment_preflight_core_calls_no_executor_transition(self) -> None:
        with mock.patch(
            "runtime_assurance.recovery_branch_executor.execute_recovery_branch"
        ) as execute:
            report, _ = self.run_report(require_clean=True)
        self.assertTrue(report.ready, report.failed_check_ids)
        execute.assert_not_called()

    def test_default_invocation_is_documented_no_op(self) -> None:
        with mock.patch(
            "runtime_assurance.recovery_branch_executor.execute_recovery_branch"
        ) as execute, contextlib.redirect_stdout(io.StringIO()):
            status = preflight_main([])
        self.assertEqual(status, 2)
        execute.assert_not_called()


def replace_repository_state(state, **changes):
    values = {
        "is_git_worktree": state.is_git_worktree,
        "head_commit": state.head_commit,
        "tracked_clean": state.tracked_clean,
        "staged_clean": state.staged_clean,
        "untracked_paths": state.untracked_paths,
    }
    values.update(changes)
    return RecoveryRepositoryState(**values)


if __name__ == "__main__":
    unittest.main()
