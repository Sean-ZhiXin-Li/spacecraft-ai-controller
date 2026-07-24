from __future__ import annotations

import copy
import csv
import inspect
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from runtime_assurance.recovery_experiment_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    COMPARISON_FILENAME,
    DECISION_EVENT_SCHEMA_VERSION,
    DECISION_LOG_FILENAME,
    PUBLISHED_ARTIFACT_FILENAMES,
    RESULTS_FILENAME,
    SUMMARY_FILENAME,
    RecoveryArtifactError,
    RecoveryBranchExperimentRecord,
    RecoveryDecisionEvent,
    RecoveryExperimentBundle,
    decision_log_jsonl_bytes,
    publish_recovery_experiment_bundle,
    recompute_record_payload_hash,
    results_csv_bytes,
    summary_markdown_bytes,
    validate_recovery_experiment_bundle,
    with_record_payload_hash,
    write_comparison_png,
    write_results_csv,
)
from runtime_assurance.recovery_experiment_preflight import (
    EXPECTED_RUNTIME_STOP_PRIORITY,
    build_artifact_contract,
    load_json_object,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "analysis"
    / "recovery_action_branching_nonformal_v0"
    / "manifest.json"
)
BRANCH_STATE_PATH = MANIFEST_PATH.with_name("branch_state.json")
IMPLEMENTATION_COMMIT = "a" * 40


def build_contract():
    return build_artifact_contract(
        load_json_object(MANIFEST_PATH),
        load_json_object(BRANCH_STATE_PATH),
        implementation_commit=IMPLEMENTATION_COMMIT,
    )


def build_record(contract, branch_id: str):
    success = branch_id == "velocity_opposed_thrust_v0"
    rejected = branch_id == "tangential_error_correction_v0"
    aborted = branch_id == "explicit_abort_v0"
    transitions = 0 if rejected or aborted else (8 if success else 10)
    terminal_reason = (
        "recovery_success"
        if success
        else "action_rejected"
        if rejected
        else "explicit_abort"
        if aborted
        else "recovery_horizon_exhausted"
    )
    record = RecoveryBranchExperimentRecord(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        experiment_id=contract.experiment_id,
        case_id=contract.case_id,
        seed=contract.seed,
        branch_id=branch_id,
        branch_state_hash=contract.branch_state_hash,
        manifest_hash=contract.manifest_hash,
        implementation_commit=contract.implementation_commit,
        case_configuration_hash=contract.case_configuration_hash,
        simulator_configuration_hash=contract.simulator_configuration_hash,
        simulator_constants_hash=contract.simulator_constants_hash,
        branch_step=contract.branch_step,
        nominal_prefix_transition_count=contract.nominal_prefix_transition_count,
        recovery_transition_count=transitions,
        total_transition_count=contract.nominal_prefix_transition_count + transitions,
        recovery_horizon=contract.recovery_horizon,
        total_horizon=contract.total_horizon,
        hazard_threshold=contract.hazard_threshold,
        hazard_comparator=contract.hazard_comparator,
        stop_priority=contract.stop_priority,
        stop_priority_version=contract.stop_priority_version,
        terminal_reason=terminal_reason,
        branch_terminal_label=(
            "recovery_action_rejected"
            if rejected
            else "explicit_recovery_abort"
            if aborted
            else terminal_reason
        ),
        controlled_terminal_label=(
            "success"
            if success
            else "timeout"
            if terminal_reason == "recovery_horizon_exhausted"
            else "unknown_with_manual_audit"
        ),
        recovery_outcome=(
            "hazard_avoided_and_task_recovered"
            if success
            else "recovery_action_rejected"
            if rejected
            else "hazard_avoided_through_termination"
            if aborted
            else "hazard_avoided_task_stalled"
        ),
        valid=True,
        overspeed_status="clear",
        instability_status="clear",
        unsafe_state_status="clear",
        invalid_simulation_status="clear",
        invalid_recovery_evaluation_status="clear",
        crossed_target_radius=success,
        first_crossing_step=contract.branch_step + 5 if success else None,
        phase34_compatible_recoverable_crossing=success,
        first_recoverable_crossing_step=contract.branch_step + 5 if success else None,
        recovery_success_status="triggered" if success else "clear",
        recovery_success=success,
        recovery_success_step=contract.branch_step + 5 if success else None,
        final_simulator_success=success,
        overspeed_headroom=0.4,
        action_saturation_margin=None if aborted else 0.75,
        available_correction_authority=None,
        required_to_available_correction_ratio=None,
        normalized_control_effort=None if aborted else float(transitions) * 0.25,
        delta_v_proxy=None,
        recovery_steps=transitions,
        crossing_delay=5 if success else None,
        final_radius_error=0.0,
        final_radial_velocity_error=0.0,
        final_tangential_velocity_error=0.0,
        task_abandonment_status=not success,
        monitor_evaluation_count=0 if aborted else transitions + int(rejected),
        intervention_count=1 if rejected else 0,
        allow_count=transitions,
        veto_count=1 if rejected else 0,
        intervention_rate=(1.0 if rejected else 0.0),
        first_intervention_step=1 if rejected else None,
        last_intervention_step=1 if rejected else None,
        longest_intervention_streak=1 if rejected else 0,
        veto_segment_count=1 if rejected else 0,
        action_suppression_duration=1 if rejected else 0,
        recovery_action_rejection_count=1 if rejected else 0,
        action_rejected=rejected,
        rejected_action_executed=False,
        physical_transition_executed=transitions > 0,
        evaluator_versions=(
            "phase34_compatible_recoverability_predicate_v0",
            "recovery_success_v0",
            "repository_supported_instability_v0",
            "repository_supported_unsafe_state_v0",
        ),
        result_payload_hash="",
    )
    return with_record_payload_hash(record)


def build_event(contract, record, branch_index: int):
    aborted = record.branch_id == "explicit_abort_v0"
    rejected = record.action_rejected
    return RecoveryDecisionEvent(
        decision_event_schema_version=DECISION_EVENT_SCHEMA_VERSION,
        experiment_id=contract.experiment_id,
        case_id=contract.case_id,
        seed=contract.seed,
        branch_id=record.branch_id,
        branch_state_hash=contract.branch_state_hash,
        event_index=0,
        post_branch_step=0 if aborted or rejected else 1,
        total_transition_count=(
            contract.nominal_prefix_transition_count
            + (1 if record.physical_transition_executed else 0)
        ),
        proposed_action=None if aborted else (0.0, 0.0),
        final_veto_decision=(
            "not_applicable" if aborted else "veto" if rejected else "allow"
        ),
        executed_action=None if aborted or rejected else (0.0, 0.0),
        transition_occurred=record.physical_transition_executed,
        current_state_hash="c" * 64,
        predicted_next_state_hash=None if aborted else "e" * 64,
        next_state_hash="d" * 64 if record.physical_transition_executed else None,
        predicted_speed_ratio=None if aborted else 1.5,
        realized_speed_ratio=(
            1.4 if record.physical_transition_executed else None
        ),
        hazard_threshold=contract.hazard_threshold,
        hazard_comparator=contract.hazard_comparator,
        triggered_stop_condition=record.terminal_reason,
        evaluator_statuses=(
            ("instability", "clear"),
            ("recovery_success", record.recovery_success_status),
            ("unsafe_state", "clear"),
        ),
        terminal_reason=record.terminal_reason,
        evidence_level="synthetic",
    )


def build_bundle(contract=None):
    contract = contract or build_contract()
    records = tuple(build_record(contract, branch_id) for branch_id in contract.branch_ids)
    events = tuple(
        build_event(contract, record, index)
        for index, record in enumerate(records)
    )
    return contract, RecoveryExperimentBundle(records, events, is_synthetic=True)


class RecoveryExperimentArtifactSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract, self.bundle = build_bundle()

    def report(self, bundle=None):
        return validate_recovery_experiment_bundle(
            bundle or self.bundle, self.contract
        )

    def test_valid_four_branch_synthetic_bundle_passes(self) -> None:
        report = self.report()
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.record_count, 4)

    def test_missing_branch_fails(self) -> None:
        bundle = replace(self.bundle, records=self.bundle.records[:-1])
        self.assertFalse(self.report(bundle).valid)

    def test_duplicate_branch_fails(self) -> None:
        records = self.bundle.records[:-1] + (self.bundle.records[0],)
        self.assertFalse(self.report(replace(self.bundle, records=records)).valid)

    def test_wrong_branch_order_fails(self) -> None:
        records = tuple(reversed(self.bundle.records))
        self.assertFalse(self.report(replace(self.bundle, records=records)).valid)

    def test_inconsistent_branch_state_hash_fails(self) -> None:
        changed = replace(self.bundle.records[0], branch_state_hash="b" * 64)
        changed = with_record_payload_hash(changed)
        bundle = replace(self.bundle, records=(changed,) + self.bundle.records[1:])
        self.assertFalse(self.report(bundle).valid)

    def test_inconsistent_manifest_hash_fails(self) -> None:
        changed = replace(self.bundle.records[0], manifest_hash="b" * 64)
        changed = with_record_payload_hash(changed)
        self.assertFalse(
            self.report(replace(self.bundle, records=(changed,) + self.bundle.records[1:])).valid
        )

    def test_inconsistent_seed_fails(self) -> None:
        changed = replace(self.bundle.records[0], seed=1)
        changed = with_record_payload_hash(changed)
        self.assertFalse(
            self.report(replace(self.bundle, records=(changed,) + self.bundle.records[1:])).valid
        )

    def test_inconsistent_horizon_fails(self) -> None:
        changed = replace(self.bundle.records[0], recovery_horizon=9999)
        changed = with_record_payload_hash(changed)
        self.assertFalse(
            self.report(replace(self.bundle, records=(changed,) + self.bundle.records[1:])).valid
        )

    def test_inconsistent_total_horizon_fails(self) -> None:
        changed = replace(self.bundle.records[0], total_horizon=99_999)
        changed = with_record_payload_hash(changed)
        self.assertFalse(
            self.report(replace(self.bundle, records=(changed,) + self.bundle.records[1:])).valid
        )

    def test_threshold_drift_fails(self) -> None:
        changed = replace(self.bundle.records[0], hazard_threshold=1.91)
        changed = with_record_payload_hash(changed)
        self.assertFalse(
            self.report(replace(self.bundle, records=(changed,) + self.bundle.records[1:])).valid
        )

    def test_comparator_drift_fails(self) -> None:
        changed = replace(self.bundle.records[0], hazard_comparator=">=")
        changed = with_record_payload_hash(changed)
        self.assertFalse(
            self.report(replace(self.bundle, records=(changed,) + self.bundle.records[1:])).valid
        )

    def test_stop_priority_drift_fails(self) -> None:
        changed = replace(
            self.bundle.records[0], stop_priority=tuple(reversed(EXPECTED_RUNTIME_STOP_PRIORITY))
        )
        changed = with_record_payload_hash(changed)
        self.assertFalse(
            self.report(replace(self.bundle, records=(changed,) + self.bundle.records[1:])).valid
        )

    def test_artifact_schema_drift_fails(self) -> None:
        changed = replace(self.bundle.records[0], artifact_schema_version="drift")
        changed = with_record_payload_hash(changed)
        self.assertFalse(
            self.report(replace(self.bundle, records=(changed,) + self.bundle.records[1:])).valid
        )

    def test_implementation_commit_drift_fails(self) -> None:
        changed = replace(self.bundle.records[0], implementation_commit="b" * 40)
        changed = with_record_payload_hash(changed)
        self.assertFalse(
            self.report(replace(self.bundle, records=(changed,) + self.bundle.records[1:])).valid
        )

    def test_inconsistent_simulator_hash_fails(self) -> None:
        changed = replace(self.bundle.records[0], simulator_configuration_hash="f" * 64)
        changed = with_record_payload_hash(changed)
        self.assertFalse(
            self.report(replace(self.bundle, records=(changed,) + self.bundle.records[1:])).valid
        )

    def test_explicit_abort_with_transition_fails(self) -> None:
        records = list(self.bundle.records)
        changed = replace(
            records[-1],
            recovery_transition_count=1,
            total_transition_count=self.contract.nominal_prefix_transition_count + 1,
            physical_transition_executed=True,
        )
        records[-1] = with_record_payload_hash(changed)
        self.assertFalse(self.report(replace(self.bundle, records=tuple(records))).valid)

    def test_rejected_action_marked_executed_fails(self) -> None:
        records = list(self.bundle.records)
        changed = replace(records[2], rejected_action_executed=True)
        records[2] = with_record_payload_hash(changed)
        self.assertFalse(self.report(replace(self.bundle, records=tuple(records))).valid)

    def test_excessive_transition_count_fails(self) -> None:
        changed = replace(
            self.bundle.records[0],
            recovery_transition_count=10_001,
            total_transition_count=self.contract.nominal_prefix_transition_count + 10_001,
        )
        changed = with_record_payload_hash(changed)
        self.assertFalse(
            self.report(replace(self.bundle, records=(changed,) + self.bundle.records[1:])).valid
        )

    def test_missing_required_evaluator_evidence_fails(self) -> None:
        changed = replace(self.bundle.records[0], instability_status="not_evaluated")
        changed = with_record_payload_hash(changed)
        report = self.report(
            replace(self.bundle, records=(changed,) + self.bundle.records[1:])
        )
        self.assertFalse(report.valid)
        self.assertTrue(any("unavailable" in error for error in report.errors))

    def test_unsupported_metrics_remain_null(self) -> None:
        record = self.bundle.records[-1]
        self.assertIsNone(record.delta_v_proxy)
        self.assertIsNone(record.crossing_delay)
        self.assertIs(record.crossed_target_radius, False)

    def test_payload_hash_recomputes(self) -> None:
        for record in self.bundle.records:
            self.assertEqual(record.result_payload_hash, recompute_record_payload_hash(record))

    def test_payload_mutation_invalidates_hash(self) -> None:
        changed = replace(self.bundle.records[0], terminal_reason="mutated")
        bundle = replace(self.bundle, records=(changed,) + self.bundle.records[1:])
        self.assertFalse(self.report(bundle).valid)


class RecoveryExperimentWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract, self.bundle = build_bundle()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_repeated_csv_serialization_is_byte_identical(self) -> None:
        self.assertEqual(
            results_csv_bytes(self.bundle, self.contract),
            results_csv_bytes(self.bundle, self.contract),
        )

    def test_csv_has_frozen_order_and_one_row_per_branch(self) -> None:
        text = results_csv_bytes(self.bundle, self.contract).decode("utf-8")
        rows = list(csv.DictReader(text.splitlines()))
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            tuple(row["branch_id"] for row in rows), self.contract.branch_ids
        )

    def test_null_and_false_remain_distinct_in_csv(self) -> None:
        text = results_csv_bytes(self.bundle, self.contract).decode("utf-8")
        rows = list(csv.DictReader(text.splitlines()))
        abort = rows[-1]
        self.assertEqual(abort["crossing_delay"], "")
        self.assertEqual(abort["crossed_target_radius"], "false")

    def test_repeated_jsonl_serialization_is_byte_identical(self) -> None:
        first = decision_log_jsonl_bytes(self.bundle, self.contract)
        second = decision_log_jsonl_bytes(self.bundle, self.contract)
        self.assertEqual(first, second)
        self.assertEqual(len(first.decode("utf-8").splitlines()), 4)

    def test_jsonl_objects_use_deterministic_branch_order(self) -> None:
        lines = decision_log_jsonl_bytes(self.bundle, self.contract).decode().splitlines()
        branch_ids = tuple(json.loads(line)["branch_id"] for line in lines)
        self.assertEqual(branch_ids, self.contract.branch_ids)

    def test_repeated_summary_is_byte_identical_and_scoped(self) -> None:
        first = summary_markdown_bytes(self.bundle, self.contract)
        second = summary_markdown_bytes(self.bundle, self.contract)
        self.assertEqual(first, second)
        text = first.decode("utf-8")
        self.assertIn("## Non-Claims", text)
        self.assertNotIn("best branch", text.lower())
        self.assertNotIn("winner", text.lower())

    def test_repeated_plot_is_byte_identical(self) -> None:
        first = self.root / "first.png"
        second = self.root / "second.png"
        write_comparison_png(first, self.bundle, self.contract)
        write_comparison_png(second, self.bundle, self.contract)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertTrue(first.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_plot_writer_uses_no_pyplot_or_gui(self) -> None:
        source = inspect.getsource(write_comparison_png)
        self.assertNotIn("pyplot", source)
        self.assertIn("FigureCanvasAgg", source)

    def test_writer_refuses_overwrite(self) -> None:
        path = self.root / RESULTS_FILENAME
        write_results_csv(path, self.bundle, self.contract)
        original = path.read_bytes()
        with self.assertRaises(RecoveryArtifactError):
            write_results_csv(path, self.bundle, self.contract)
        self.assertEqual(path.read_bytes(), original)

    def test_valid_synthetic_bundle_publishes_complete_artifacts(self) -> None:
        target = self.root / "synthetic-bundle"
        result = publish_recovery_experiment_bundle(
            self.bundle,
            self.contract,
            target,
            repository_root=self.root,
        )
        self.assertTrue(result.published)
        self.assertEqual(
            {path.name for path in target.iterdir()},
            set(PUBLISHED_ARTIFACT_FILENAMES),
        )
        self.assertEqual(tuple(name for name, _ in result.artifact_hashes), PUBLISHED_ARTIFACT_FILENAMES)

    def test_invalid_bundle_publishes_nothing(self) -> None:
        target = self.root / "invalid"
        invalid = replace(self.bundle, records=self.bundle.records[:-1])
        with self.assertRaises(RecoveryArtifactError):
            publish_recovery_experiment_bundle(
                invalid, self.contract, target, repository_root=self.root
            )
        self.assertFalse(target.exists())

    def test_simulated_writer_failure_publishes_nothing_and_cleans_staging(self) -> None:
        target = self.root / "failed"

        def fail(stage: str) -> None:
            if stage == "after_summary":
                raise RuntimeError("synthetic failure")

        with self.assertRaises(RuntimeError):
            publish_recovery_experiment_bundle(
                self.bundle,
                self.contract,
                target,
                repository_root=self.root,
                failure_injector=fail,
            )
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob(".recovery-experiment-stage-*")), [])

    def test_publish_failure_after_first_move_rolls_back(self) -> None:
        target = self.root / "rollback"

        def fail(stage: str) -> None:
            if stage == f"before_publish:{DECISION_LOG_FILENAME}":
                raise RuntimeError("synthetic publish failure")

        with self.assertRaises(RuntimeError):
            publish_recovery_experiment_bundle(
                self.bundle,
                self.contract,
                target,
                repository_root=self.root,
                failure_injector=fail,
            )
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob(".recovery-experiment-stage-*")), [])

    def test_staged_validation_failure_publishes_nothing(self) -> None:
        target = self.root / "staged-invalid"
        with mock.patch(
            "runtime_assurance.recovery_experiment_artifacts.validate_staged_artifacts",
            side_effect=RecoveryArtifactError("synthetic validation failure"),
        ):
            with self.assertRaises(RecoveryArtifactError):
                publish_recovery_experiment_bundle(
                    self.bundle,
                    self.contract,
                    target,
                    repository_root=self.root,
                )
        self.assertFalse(target.exists())

    def test_existing_target_artifact_is_never_deleted(self) -> None:
        target = self.root / "existing"
        target.mkdir()
        existing = target / RESULTS_FILENAME
        existing.write_bytes(b"user data")
        with self.assertRaises(RecoveryArtifactError):
            publish_recovery_experiment_bundle(
                self.bundle,
                self.contract,
                target,
                repository_root=self.root,
            )
        self.assertEqual(existing.read_bytes(), b"user data")

    def test_partial_target_bundle_is_rejected(self) -> None:
        target = self.root / "partial"
        target.mkdir()
        (target / SUMMARY_FILENAME).write_text("partial", encoding="utf-8")
        with self.assertRaises(RecoveryArtifactError):
            publish_recovery_experiment_bundle(
                self.bundle,
                self.contract,
                target,
                repository_root=self.root,
            )

    def test_protected_path_is_rejected(self) -> None:
        protected = self.root / "analysis" / "phase34_post_cross_sync"
        protected.mkdir(parents=True)
        with self.assertRaises(RecoveryArtifactError):
            publish_recovery_experiment_bundle(
                self.bundle,
                self.contract,
                protected,
                repository_root=self.root,
            )

    def test_synthetic_bundle_cannot_use_frozen_output_directory(self) -> None:
        target = self.root / self.contract.output_directory
        target.mkdir(parents=True)
        with self.assertRaises(RecoveryArtifactError):
            publish_recovery_experiment_bundle(
                self.bundle,
                self.contract,
                target,
                repository_root=self.root,
            )

    def test_real_recovery_result_artifacts_are_absent_or_complete(self) -> None:
        real_output = MANIFEST_PATH.parent
        existing = {
            filename
            for filename in PUBLISHED_ARTIFACT_FILENAMES
            if (real_output / filename).is_file()
        }
        self.assertIn(existing, (set(), set(PUBLISHED_ARTIFACT_FILENAMES)))
        for filename in existing:
            self.assertGreater((real_output / filename).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
