from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from unittest import mock

from runtime_assurance.recovery_experiment_preflight import RecoveryRepositoryState
from runtime_assurance.staged_recovery_instrumentation_validation import (
    EXPECTED_EVENT_COUNT,
    OUTPUT_RELATIVE_PATH,
    PUBLISHED_FILENAMES,
    TRACE_CLASSIFICATION,
    VALIDATION_BRANCH_ID,
    VALIDATION_HORIZON,
    InstrumentationValidationError,
    build_field_completeness,
    build_validation_payloads,
    compare_bounded_runs,
    publish_validation_payloads,
    require_clean_committed_repository,
    validate_payload_bytes,
    validate_static_configuration,
)
from runtime_assurance.staged_recovery_logger_adapter import (
    Stage0BValidationLoggerAdapter,
    build_bounded_runtime_identity,
    run_bounded_recovery_validation_path,
)
from runtime_assurance.staged_recovery_runtime_logger import (
    aggregate_trace_sha256,
    event_document,
)
from Tests.test_staged_recovery_logger_adapter import (
    IMPLEMENTATION,
    branch_state,
    fake_step_executor,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_CLI = ROOT / "scripts/run_staged_recovery_instrumentation_validation_v0.py"
CHECK_CLI = ROOT / "scripts/check_staged_recovery_instrumentation_validation.py"


@lru_cache(maxsize=1)
def make_pair():
    document_a = branch_state()
    baseline = run_bounded_recovery_validation_path(
        document_a,
        branch_id=VALIDATION_BRANCH_ID,
        horizon_steps=VALIDATION_HORIZON,
        implementation_commit=IMPLEMENTATION,
        step_executor=fake_step_executor,
    )
    document_b = branch_state()
    identity, _ = build_bounded_runtime_identity(
        document_b,
        branch_id=VALIDATION_BRANCH_ID,
        implementation_commit=IMPLEMENTATION,
    )
    adapter = Stage0BValidationLoggerAdapter(
        identity, session_id="synthetic_stage0c_fixture", max_events=10
    )
    observed = run_bounded_recovery_validation_path(
        document_b,
        branch_id=VALIDATION_BRANCH_ID,
        horizon_steps=VALIDATION_HORIZON,
        implementation_commit=IMPLEMENTATION,
        observer=adapter.observe,
        step_executor=fake_step_executor,
    )
    events = adapter.finalize().events
    equivalence = compare_bounded_runs(
        baseline,
        observed,
        manifest_hash="e9cb96eae714bc0d8ed66d1a85f29baed2819d0d425a3ce9742b7e77ac236bad",
    )
    completeness = build_field_completeness(events)
    return baseline, observed, events, equivalence, completeness


class FrozenConfigurationTests(unittest.TestCase):
    def test_static_configuration_passes_without_execution(self) -> None:
        report = validate_static_configuration(ROOT, require_output_absent=True)
        self.assertTrue(report.valid, report.errors)

    def test_validation_branch_and_horizon_are_frozen(self) -> None:
        self.assertEqual(VALIDATION_BRANCH_ID, "velocity_opposed_thrust_v0")
        self.assertEqual(VALIDATION_HORIZON, 8)
        self.assertEqual(EXPECTED_EVENT_COUNT, 10)


class EquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline, cls.observed, cls.events, cls.report, cls.completeness = make_pair()

    def test_all_required_equivalence_checks_pass(self) -> None:
        self.assertTrue(self.report.all_equivalence_checks)
        self.assertTrue(all(check.passed for check in self.report.checks))

    def test_baseline_trace_is_not_published(self) -> None:
        self.assertFalse(self.report.baseline_trace_published)
        self.assertTrue(self.report.observed_trace_published)

    def test_initial_and_final_state_equality(self) -> None:
        checks = {item.check_id: item for item in self.report.checks}
        self.assertTrue(checks["same_initial_state_hash"].passed)
        self.assertTrue(checks["same_final_state_hash"].passed)

    def test_action_monitor_state_counter_and_terminal_checks_exist(self) -> None:
        ids = {item.check_id for item in self.report.checks}
        required = {
            "same_proposed_action_sequence",
            "same_monitor_prediction_sequence",
            "same_monitor_decision_sequence",
            "same_executed_action_sequence",
            "same_predicted_state_hash_sequence",
            "same_realized_state_hash_sequence",
            "same_recovery_step_sequence",
            "same_total_transition_count_sequence",
            "same_runtime_terminal_reason",
        }
        self.assertTrue(required.issubset(ids))

    def test_proposed_action_mismatch_fails(self) -> None:
        changed = replace(
            self.observed.transition_snapshots[0],
            proposed_action=(0.0, 0.0),
        )
        observed = replace(
            self.observed,
            transition_snapshots=(changed,) + self.observed.transition_snapshots[1:],
        )
        report = compare_bounded_runs(
            self.baseline, observed, manifest_hash="a" * 64
        )
        failed = {check.check_id for check in report.checks if not check.passed}
        self.assertIn("same_proposed_action_sequence", failed)

    def test_monitor_prediction_mismatch_fails(self) -> None:
        changed = replace(
            self.observed.transition_snapshots[0],
            predicted_speed_ratio=0.0,
        )
        observed = replace(
            self.observed,
            transition_snapshots=(changed,) + self.observed.transition_snapshots[1:],
        )
        report = compare_bounded_runs(self.baseline, observed, manifest_hash="a" * 64)
        self.assertFalse(report.all_equivalence_checks)

    def test_counter_mismatch_fails(self) -> None:
        changed = replace(
            self.observed.transition_snapshots[-1],
            total_transition_count=999,
        )
        observed = replace(
            self.observed,
            transition_snapshots=self.observed.transition_snapshots[:-1] + (changed,),
        )
        report = compare_bounded_runs(self.baseline, observed, manifest_hash="a" * 64)
        self.assertFalse(report.all_equivalence_checks)

    def test_terminal_reason_mismatch_fails(self) -> None:
        observed = replace(self.observed, runtime_terminal_reason="different")
        report = compare_bounded_runs(self.baseline, observed, manifest_hash="a" * 64)
        self.assertFalse(report.all_equivalence_checks)


class EventAndCompletenessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline, cls.observed, cls.events, cls.equivalence, cls.report = make_pair()

    def test_trace_structure_is_one_eight_one(self) -> None:
        self.assertEqual(len(self.events), 10)
        self.assertEqual(self.events[0].event_type.value, "initial_snapshot")
        self.assertEqual(
            [event.event_type.value for event in self.events[1:9]],
            ["transition"] * 8,
        )
        self.assertEqual(self.events[-1].event_type.value, "terminal")

    def test_terminal_retains_recovery_step(self) -> None:
        self.assertEqual(self.events[-2].recovery_step, 8)
        self.assertEqual(self.events[-1].recovery_step, 8)

    def test_measured_predicted_and_realized_states_remain_separate(self) -> None:
        event = self.events[1]
        self.assertIsNotNone(event.pre_observation)
        self.assertIsNotNone(event.predicted_observation)
        self.assertIsNotNone(event.post_observation)
        self.assertEqual(event.predicted_state_hash, event.realized_state_hash)

    def test_crossing_and_progress_use_measured_pre_post_states(self) -> None:
        event = self.events[1]
        self.assertTrue(event.post_observation.field("target_radius_crossing").available)
        self.assertTrue(dict(event.progress_sample)["delta_radial_velocity"].available)

    def test_action_geometry_is_derived(self) -> None:
        geometry = dict(self.events[1].action_geometry)
        self.assertTrue(geometry["proposed_action_magnitude"].available)
        self.assertTrue(geometry["proposed_action_radial_component"].available)

    def test_phase_and_no_progress_are_correctly_unavailable(self) -> None:
        rows = {
            (entry.schema_source, entry.field_id): entry
            for entry in self.report.entries
        }
        self.assertEqual(
            rows[("stage0a", "current_phase")].completeness_status,
            "correctly_not_evaluated",
        )
        self.assertEqual(
            rows[("stage0a", "no_progress_status")].completeness_status,
            "correctly_not_evaluated",
        )

    def test_no_required_field_is_invalid_or_missing(self) -> None:
        self.assertEqual(self.report.unexpectedly_missing_fields, 0)
        self.assertEqual(self.report.invalid_required_fields, 0)

    def test_unsupported_fields_remain_unsupported(self) -> None:
        rows = {
            (entry.schema_source, entry.field_id): entry
            for entry in self.report.entries
        }
        self.assertEqual(
            rows[("stage0a", "available_correction_authority")].completeness_status,
            "unsupported",
        )


class PayloadAndPublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline, self.observed, self.events, self.equivalence, self.completeness = make_pair()
        manifest = json.loads(
            (ROOT / "analysis/recovery_action_branching_nonformal_v0/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.payloads = build_validation_payloads(
            repository_root=ROOT,
            implementation_commit=IMPLEMENTATION,
            manifest=manifest,
            branch_state=branch_state(),
            run=self.observed,
            events=self.events,
            equivalence=self.equivalence,
            completeness=self.completeness,
        )

    def test_payload_set_is_exact_and_valid(self) -> None:
        self.assertEqual(set(self.payloads), set(PUBLISHED_FILENAMES))
        validate_payload_bytes(self.payloads)

    def test_trace_classification_is_measured_validation(self) -> None:
        trace_manifest = json.loads(self.payloads["trace_manifest.json"])
        self.assertEqual(trace_manifest["trace_classification"], TRACE_CLASSIFICATION)
        self.assertFalse(trace_manifest["scientific_result"])
        self.assertEqual(trace_manifest["staged_recovery_execution"], "not_authorized")

    def test_event_and_trace_hashes_recompute(self) -> None:
        documents = [event_document(event) for event in self.events]
        self.assertEqual(
            aggregate_trace_sha256(self.events),
            json.loads(self.payloads["trace_manifest.json"])["aggregate_trace_hash"],
        )
        self.assertEqual(len(documents), 10)

    def test_event_mutation_invalidates_payload(self) -> None:
        payloads = dict(self.payloads)
        lines = payloads["staged_recovery_trace.jsonl"].decode().splitlines()
        event = json.loads(lines[1])
        event["recovery_step"] = 99
        lines[1] = json.dumps(event, sort_keys=True, separators=(",", ":"))
        payloads["staged_recovery_trace.jsonl"] = ("\n".join(lines) + "\n").encode()
        with self.assertRaises(InstrumentationValidationError):
            validate_payload_bytes(payloads)

    def test_valid_bundle_publishes_atomically(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            target = Path(temporary) / "trace"
            result = publish_validation_payloads(
                self.payloads, target_directory=target, repository_root=ROOT
            )
            self.assertTrue(result.published)
            self.assertEqual(
                {path.name for path in target.iterdir()}, set(PUBLISHED_FILENAMES)
            )

    def test_existing_target_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            target = Path(temporary) / "trace"
            target.mkdir()
            sentinel = target / "user.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(Exception):
                publish_validation_payloads(
                    self.payloads, target_directory=target, repository_root=ROOT
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_writer_failure_publishes_nothing_and_cleans_staging(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            parent = Path(temporary)
            target = parent / "trace"

            def fail(_stage):
                raise RuntimeError("injected")

            with self.assertRaises(RuntimeError):
                publish_validation_payloads(
                    self.payloads,
                    target_directory=target,
                    repository_root=ROOT,
                    failure_injector=fail,
                )
            self.assertFalse(target.exists())
            self.assertFalse(any(".trace.staging-" in item.name for item in parent.iterdir()))

    def test_protected_target_is_rejected(self) -> None:
        target = ROOT / "analysis/recovery_action_branching_nonformal_v0/new_trace"
        with self.assertRaises(Exception):
            publish_validation_payloads(
                self.payloads, target_directory=target, repository_root=ROOT
            )

    def test_repeated_payload_generation_is_deterministic(self) -> None:
        manifest = json.loads(
            (ROOT / "analysis/recovery_action_branching_nonformal_v0/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        again = build_validation_payloads(
            repository_root=ROOT,
            implementation_commit=IMPLEMENTATION,
            manifest=manifest,
            branch_state=branch_state(),
            run=self.observed,
            events=self.events,
            equivalence=self.equivalence,
            completeness=self.completeness,
        )
        self.assertEqual(self.payloads, again)


class CliAndRepositorySafetyTests(unittest.TestCase):
    def run_cli(self, script: Path, *args: str):
        return subprocess.run(
            [sys.executable, str(script), *args],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

    def test_default_cli_prints_help_without_execution(self) -> None:
        output = ROOT / OUTPUT_RELATIVE_PATH
        before = output.exists()
        result = self.run_cli(RUN_CLI)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout)
        self.assertEqual(output.exists(), before)

    def test_plan_and_validate_only_do_not_execute(self) -> None:
        plan = self.run_cli(RUN_CLI, "--plan")
        validate = self.run_cli(RUN_CLI, "--validate-only")
        self.assertEqual(plan.returncode, 0)
        self.assertEqual(validate.returncode, 0)
        self.assertIn('"execution_disabled": true', plan.stdout)
        self.assertIn("NO_TRANSITION_EXECUTED true", validate.stdout)

    def test_static_checker_does_not_execute(self) -> None:
        result = self.run_cli(CHECK_CLI, "--validate-static")
        self.assertEqual(result.returncode, 0)
        self.assertIn("STATIC_VALIDATION PASS", result.stdout)

    def test_scientific_override_flags_do_not_exist(self) -> None:
        for flag in ("--branch", "--horizon", "--retry", "--threshold"):
            result = self.run_cli(RUN_CLI, flag, "x")
            self.assertNotEqual(result.returncode, 0)

    def test_dirty_or_staged_repository_blocks_execution_gate(self) -> None:
        dirty = RecoveryRepositoryState(True, "a" * 40, False, True, ())
        staged = RecoveryRepositoryState(True, "a" * 40, True, False, ())
        with mock.patch(
            "runtime_assurance.staged_recovery_instrumentation_validation.inspect_repository_state",
            return_value=dirty,
        ):
            with self.assertRaises(InstrumentationValidationError):
                require_clean_committed_repository(ROOT)
        with mock.patch(
            "runtime_assurance.staged_recovery_instrumentation_validation.inspect_repository_state",
            return_value=staged,
        ):
            with self.assertRaises(InstrumentationValidationError):
                require_clean_committed_repository(ROOT)

    def test_unrelated_untracked_file_does_not_block_gate(self) -> None:
        clean = RecoveryRepositoryState(
            True, "a" * 40, True, True, ("paper/unrelated.pdf",)
        )
        with mock.patch(
            "runtime_assurance.staged_recovery_instrumentation_validation.inspect_repository_state",
            return_value=clean,
        ):
            self.assertEqual(require_clean_committed_repository(ROOT), "a" * 40)

    def test_frozen_files_are_unchanged_by_synthetic_validation(self) -> None:
        protected = [
            ROOT / "analysis/recovery_action_branching_nonformal_v0/manifest.json",
            ROOT / "analysis/recovery_action_branching_nonformal_v0/branch_state.json",
            ROOT / "analysis/staged_recovery_instrumentation_v0/instrumentation_manifest.json",
            ROOT / "analysis/staged_recovery_runtime_logger_v0/logger_manifest.json",
        ]
        before = {path: path.read_bytes() for path in protected}
        make_pair()
        self.assertEqual({path: path.read_bytes() for path in protected}, before)


if __name__ == "__main__":
    unittest.main()
