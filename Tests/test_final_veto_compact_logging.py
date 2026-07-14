from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.check_final_veto_results import read_jsonl, validate_decision_events
from scripts.final_veto_compact_log import (
    FORMAL_PUBLIC_LOG_BYTE_BUDGET,
    LOG_MODE_COMPACT,
    LOG_MODE_FULL_TRACE,
    MAX_COMPACT_RECORDS_PER_RUN,
    CompactDecisionLogError,
    DecisionLogStream,
    DecisionStreamStatistics,
    canonical_json_bytes,
    compact_logging_preflight_errors,
    convert_full_trace_to_compact,
    dry_run_expected_compact_bytes_per_monitor_arm,
    estimate_compact_logging_plan,
    logging_configuration_errors,
)
from scripts.run_final_veto_ablation import (
    PROJECT_ROOT,
    build_diagnostic_summary,
    build_pair_record,
    build_planned_jobs,
    load_frozen_manifest,
)


STRESS_DIRECTORY = PROJECT_ROOT / "analysis" / "final_veto_ablation_smoke_stress_v0"
STRESS_LOG = STRESS_DIRECTORY / "smoke_decision_log.jsonl"
STRESS_ARMS = STRESS_DIRECTORY / "smoke_results.csv"


def logical_event(
    step: int,
    *,
    veto: bool = False,
    phase: str = "DESCENT",
    active_stage: str = "radial_energy_push",
    invalid: bool = False,
    false_negative: bool = False,
    fallback_failure: bool = False,
    predicted_fallback: float | None = None,
    realized: float | None = None,
    formal: bool = False,
) -> dict[str, object]:
    if invalid:
        decision_type = "unknown"
        reason = "invalid_simulation"
        veto_status = "unknown"
        executed_action = None
        predicted_nominal = None
        realized_ratio = None
    else:
        decision_type = "veto_action" if veto else "continue"
        reason = (
            "predicted_nominal_exceeds_overspeed_threshold"
            if veto
            else "predicted_nominal_within_threshold"
        )
        veto_status = "veto" if veto else "allow"
        executed_action = [0.0, 0.0] if veto else [0.25, -0.5]
        predicted_nominal = 2.0 if veto else 1.5
        realized_ratio = realized if realized is not None else (1.5 if veto else 1.5)
    if veto and predicted_fallback is None:
        predicted_fallback = 1.5
    return {
        "decision_schema_version": "decision_log_schema_v0",
        "decision_id": f"run-on__step_{step}",
        "experiment_id": "final_veto_overspeed_ablation_v0",
        "run_id": "run-on",
        "paired_run_id": "pair-1",
        "case_id": "case-1",
        "subset_id": "phase35_radial_energy_push_overspeed_stress_v0",
        "arm_id": "monitor_on",
        "step": step,
        "phase": phase,
        "active_stage": active_stage,
        "decision_type": decision_type,
        "decision_reason": reason,
        "decision_scope": "veto",
        "decision_authority": "runtime_assurance",
        "monitor_id": "one_step_overspeed_veto_v0",
        "state_summary": "{}",
        "safety_level": "nominal",
        "recoverability_level": "unknown",
        "trust_flags": ["none"],
        "nominal_proposed_action": [0.25, -0.5],
        "executed_action": executed_action,
        "predicted_nominal_speed_ratio": predicted_nominal,
        "predicted_fallback_speed_ratio": predicted_fallback if veto else None,
        "realized_executed_speed_ratio": realized_ratio,
        "hazard_threshold": 1.90,
        "hazard_comparator": ">",
        "veto_status": veto_status,
        "veto_reason": reason,
        "fallback_available": True,
        "fallback_action": [0.0, 0.0],
        "fallback_executed": veto,
        "fallback_failure": fallback_failure,
        "invalid_evaluation": invalid,
        "manual_audit_note": "synthetic compact-log fixture",
        "is_formal_experiment": formal,
        "false_negative": false_negative,
    }


def terminal_record(*, formal: bool = False) -> dict[str, object]:
    return {
        "decision_schema_version": "decision_log_schema_v0",
        "experiment_id": "final_veto_overspeed_ablation_v0",
        "run_id": "run-on",
        "paired_run_id": "pair-1",
        "case_id": "case-1",
        "subset_id": "phase35_radial_energy_push_overspeed_stress_v0",
        "arm_id": "monitor_on",
        "monitor_id": "one_step_overspeed_veto_v0",
        "step": 100,
        "terminal_state": True,
        "terminal_label": "no_crossing",
        "termination_reason": "max_steps",
        "is_formal_experiment": formal,
    }


class CompactSegmentTests(unittest.TestCase):
    def stream(self, output: list[dict[str, object]]) -> DecisionLogStream:
        return DecisionLogStream(
            output.append,
            mode=LOG_MODE_COMPACT,
            is_formal_experiment=False,
        )

    def test_repeated_allows_collapse_into_one_segment(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        for step in range(1, 4):
            stream.consume(logical_event(step))
        stream.close()
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["event_kind"], "decision_segment")
        self.assertEqual((output[0]["start_step"], output[0]["end_step"]), (1, 3))
        self.assertEqual(output[0]["step_count"], 3)

    def test_repeated_vetoes_collapse_into_one_segment(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        for step in range(1, 5):
            stream.consume(logical_event(step, veto=True))
        stream.close()
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["decision_type"], "veto_action")
        self.assertEqual(output[0]["step_count"], 4)
        self.assertEqual(output[0]["minimum_predicted_fallback_speed_ratio"], 1.5)

    def test_allow_to_veto_transition_creates_two_segments(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1))
        stream.consume(logical_event(2, veto=True))
        stream.close()
        self.assertEqual([row["decision_type"] for row in output], ["continue", "veto_action"])

    def test_run_id_change_splits_a_segment(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1))
        changed_run = logical_event(2)
        changed_run["run_id"] = "run-on-replacement"
        stream.consume(changed_run)
        stream.close()
        self.assertEqual(len(output), 2)
        self.assertEqual(
            [row["run_id"] for row in output],
            ["run-on", "run-on-replacement"],
        )

    def test_phase_change_splits_a_segment(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1, phase="DESCENT"))
        stream.consume(logical_event(2, phase="POST_CROSS"))
        stream.close()
        self.assertEqual(len(output), 2)

    def test_active_stage_change_splits_a_segment(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1, active_stage="radial_energy_push"))
        stream.consume(logical_event(2, active_stage="radius_priority"))
        stream.close()
        self.assertEqual(len(output), 2)

    def test_invalid_evaluation_flushes_immediately(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1))
        stream.consume(logical_event(2, invalid=True))
        self.assertEqual(len(output), 2)
        self.assertEqual(output[1]["event_kind"], "decision_event")
        self.assertIn("invalid_monitor_evaluation", output[1]["exception_reasons"])

    def test_false_negative_flushes_immediately(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1, false_negative=True, realized=1.91))
        self.assertEqual(output[0]["event_kind"], "decision_event")
        self.assertIn("false_negative", output[0]["exception_reasons"])

    def test_fallback_failure_flushes_immediately(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(
            logical_event(
                1,
                veto=True,
                fallback_failure=True,
                predicted_fallback=1.95,
                realized=1.95,
            )
        )
        self.assertEqual(output[0]["event_kind"], "decision_event")
        self.assertIn("fallback_failure", output[0]["exception_reasons"])

    def test_fallback_classification_mismatch_is_dedicated(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(
            logical_event(1, veto=True, predicted_fallback=1.89, realized=1.91)
        )
        self.assertIn("fallback_classification_mismatch", output[0]["exception_reasons"])

    def test_final_segment_flushes_on_close(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1))
        self.assertEqual(output, [])
        stream.close()
        self.assertEqual(len(output), 1)

    def test_terminal_transition_is_a_dedicated_record(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1))
        stream.finish_run(terminal_record())
        self.assertEqual([row["event_kind"] for row in output], [
            "decision_segment",
            "terminal_transition",
        ])

    def test_compact_records_validate_against_distinct_contracts(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1))
        stream.finish_run(terminal_record())
        self.assertEqual(validate_decision_events(output), [])

    def test_valid_compact_decision_cannot_use_unknown_reason(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1))
        stream.close()
        output[0]["decision_reason"] = "unknown"
        self.assertTrue(
            any(
                "invalid explicit decision reason" in error
                for error in validate_decision_events(output)
            )
        )

    def test_compact_validation_requires_one_terminal_transition(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1))
        stream.close()
        self.assertTrue(
            any("exactly one terminal transition" in error for error in validate_decision_events(output))
        )

    def test_compact_validation_rejects_hidden_false_negative(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1, realized=1.91))
        stream.finish_run(terminal_record())
        self.assertTrue(
            any("hides a false negative" in error for error in validate_decision_events(output))
        )

    def test_compact_validation_rejects_hidden_fallback_failure(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(
            logical_event(1, veto=True, predicted_fallback=1.95, realized=1.95)
        )
        stream.finish_run(terminal_record())
        self.assertTrue(
            any("hides a fallback failure" in error for error in validate_decision_events(output))
        )

    def test_compact_validation_checks_exception_reason_flags(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1, false_negative=True, realized=1.91))
        stream.finish_run(terminal_record())
        output[0]["exception_reasons"] = ["fallback_failure"]
        self.assertTrue(
            any(
                "exceptional reasons do not match" in error
                for error in validate_decision_events(output)
            )
        )

    def test_compact_validation_checks_aggregate_ranges(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1))
        stream.finish_run(terminal_record())
        output[0]["minimum_predicted_nominal_speed_ratio"] = 1.6
        self.assertTrue(
            any(
                "inconsistent predicted_nominal_speed_ratio aggregates" in error
                for error in validate_decision_events(output)
            )
        )

    def test_compact_validation_rejects_hidden_fallback_classification_mismatch(self) -> None:
        output: list[dict[str, object]] = []
        stream = self.stream(output)
        stream.consume(logical_event(1, veto=True, predicted_fallback=1.89, realized=1.89))
        stream.finish_run(terminal_record())
        output[0]["minimum_predicted_fallback_speed_ratio"] = 1.95
        output[0]["maximum_predicted_fallback_speed_ratio"] = 1.95
        self.assertTrue(
            any(
                "hides a fallback classification mismatch" in error
                for error in validate_decision_events(output)
            )
        )


class StreamIntegrityTests(unittest.TestCase):
    def test_counters_equal_the_logical_event_stream(self) -> None:
        output: list[dict[str, object]] = []
        stream = DecisionLogStream(
            output.append,
            mode=LOG_MODE_COMPACT,
            is_formal_experiment=False,
        )
        events = [logical_event(1), logical_event(2, veto=True), logical_event(3, veto=True)]
        for event in events:
            stream.consume(event)
        stream.close()
        stats = stream.statistics
        self.assertEqual(stats.event_count, 3)
        self.assertEqual((stats.allow_count, stats.veto_count, stats.fallback_count), (1, 2, 2))
        self.assertEqual(sum(int(row["step_count"]) for row in output), 3)

    def test_sha256_digest_is_deterministic(self) -> None:
        first = DecisionStreamStatistics()
        second = DecisionStreamStatistics()
        event = logical_event(1)
        first.observe(event)
        second.observe(dict(reversed(list(event.items()))))
        self.assertEqual(first.sha256, second.sha256)

    def test_changing_one_logical_event_changes_the_digest(self) -> None:
        first = DecisionStreamStatistics()
        second = DecisionStreamStatistics()
        first.observe(logical_event(1))
        changed = logical_event(1)
        changed["realized_executed_speed_ratio"] = 1.5000001
        second.observe(changed)
        self.assertNotEqual(first.sha256, second.sha256)

    def test_canonical_serialization_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            canonical_json_bytes({"ratio": float("nan")})

    def test_compact_mode_remains_streaming(self) -> None:
        record_count = 0

        def count_record(_record) -> None:
            nonlocal record_count
            record_count += 1

        stream = DecisionLogStream(
            count_record,
            mode=LOG_MODE_COMPACT,
            is_formal_experiment=False,
        )
        for step in range(1, 10_001):
            stream.consume(logical_event(step))
            self.assertLessEqual(stream.buffered_logical_event_count, 1)
        stream.close()
        self.assertEqual(record_count, 1)
        self.assertEqual(stream.max_buffered_logical_events, 1)

    def test_compact_record_limit_fails_closed(self) -> None:
        output: list[dict[str, object]] = []
        stream = DecisionLogStream(
            output.append,
            mode=LOG_MODE_COMPACT,
            is_formal_experiment=False,
            maximum_records=1,
        )
        stream.consume(logical_event(1))
        stream.consume(logical_event(2, veto=True))
        with self.assertRaises(CompactDecisionLogError):
            stream.close()


class LoggingModeAndEstimateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.jobs = build_planned_jobs(load_frozen_manifest())

    def test_formal_mode_rejects_full_per_step_logging(self) -> None:
        with self.assertRaises(CompactDecisionLogError):
            DecisionLogStream(
                lambda _record: None,
                mode=LOG_MODE_FULL_TRACE,
                is_formal_experiment=True,
            )
        errors = compact_logging_preflight_errors(
            self.jobs,
            mode=LOG_MODE_FULL_TRACE,
        )
        self.assertTrue(any("unbounded per-step" in error for error in errors))

    def test_formal_preflight_rejects_disabled_stream_digest(self) -> None:
        errors = compact_logging_preflight_errors(
            self.jobs,
            mode=LOG_MODE_COMPACT,
            digest_enabled=False,
        )
        self.assertTrue(any("deterministic stream digest" in error for error in errors))

    def test_full_trace_requires_explicit_nonformal_external_path(self) -> None:
        errors = logging_configuration_errors(
            mode=LOG_MODE_FULL_TRACE,
            is_formal_experiment=False,
            full_trace_path=None,
            repository_root=PROJECT_ROOT,
        )
        self.assertTrue(any("user-supplied" in error for error in errors))
        self.assertEqual(
            logging_configuration_errors(
                mode=LOG_MODE_FULL_TRACE,
                is_formal_experiment=False,
                full_trace_path=Path("E:/tmp/final_veto_full_trace.jsonl"),
                repository_root=PROJECT_ROOT,
            ),
            [],
        )

    def test_full_trace_inside_repository_is_rejected(self) -> None:
        errors = logging_configuration_errors(
            mode=LOG_MODE_FULL_TRACE,
            is_formal_experiment=False,
            full_trace_path=PROJECT_ROOT / "trace.jsonl",
            repository_root=PROJECT_ROOT,
        )
        self.assertTrue(any("outside the repository" in error for error in errors))

    def test_formal_compact_estimate_is_finite_and_within_budget(self) -> None:
        estimate = estimate_compact_logging_plan(self.jobs)
        self.assertEqual(estimate.monitor_on_jobs, 13)
        self.assertEqual(estimate.maximum_logical_events, 1_300_000)
        self.assertEqual(
            estimate.semantic_record_upper_bound_without_policy_cap,
            1_300_013,
        )
        self.assertEqual(estimate.enforced_record_limit_per_run, 1_024)
        self.assertEqual(
            estimate.maximum_public_records,
            13 * MAX_COMPACT_RECORDS_PER_RUN,
        )
        expected_per_arm = max(
            dry_run_expected_compact_bytes_per_monitor_arm(job)
            for job in self.jobs
            if job.monitor_enabled
        )
        self.assertEqual(estimate.expected_serialized_bytes, 13 * expected_per_arm)
        self.assertLessEqual(estimate.maximum_public_bytes, FORMAL_PUBLIC_LOG_BYTE_BUDGET)
        self.assertEqual(compact_logging_preflight_errors(self.jobs, mode=LOG_MODE_COMPACT), [])

    def test_observed_full_trace_volume_estimate_is_recorded(self) -> None:
        observed_average_bytes = 1733.30338
        estimated = round(13 * 100_000 * observed_average_bytes)
        self.assertEqual(estimated, 2_253_294_394)


class ConverterInputAuditTests(unittest.TestCase):
    def write_fixture(
        self,
        root: Path,
        event: dict[str, object],
        *,
        arm_overrides: dict[str, str] | None = None,
    ) -> tuple[Path, Path, Path]:
        input_path = root / "full.jsonl"
        input_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
        arm_path = root / "arms.csv"
        arm = {
            "experiment_id": "final_veto_overspeed_ablation_v0",
            "run_id": "run-on",
            "paired_run_id": "pair-1",
            "case_id": "case-1",
            "subset_id": "phase35_radial_energy_push_overspeed_stress_v0",
            "arm_id": "monitor_on",
            "monitor_id": "one_step_overspeed_veto_v0",
            "is_formal_experiment": "false",
            "monitor_evaluation_count": "1",
            "invalid_monitor_evaluation_count": "0",
            "allow_count": "1",
            "veto_count": "0",
            "fallback_count": "0",
            "false_negative_count": "0",
            "fallback_failure_count": "0",
            "steps": "1",
            "terminal_label": "no_crossing",
        }
        arm.update(arm_overrides or {})
        with arm_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(arm))
            writer.writeheader()
            writer.writerow(arm)
        return input_path, arm_path, root / "compact.jsonl"

    def test_converter_requires_arm_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "full.jsonl"
            input_path.write_text(json.dumps(logical_event(1)) + "\n", encoding="utf-8")
            with self.assertRaises(CompactDecisionLogError):
                convert_full_trace_to_compact(input_path, root / "compact.jsonl")

    def test_converter_requires_explicit_nonformal_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = logical_event(1)
            event.pop("is_formal_experiment")
            input_path, arm_path, output_path = self.write_fixture(root, event)
            with self.assertRaises(CompactDecisionLogError):
                convert_full_trace_to_compact(
                    input_path,
                    output_path,
                    arm_results_path=arm_path,
                )
            self.assertFalse(output_path.exists())

    def test_converter_rejects_arm_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = logical_event(1)
            event["case_id"] = "wrong-case"
            input_path, arm_path, output_path = self.write_fixture(root, event)
            with self.assertRaises(CompactDecisionLogError):
                convert_full_trace_to_compact(
                    input_path,
                    output_path,
                    arm_results_path=arm_path,
                )
            self.assertFalse(output_path.exists())

    def test_converter_counter_drift_does_not_publish_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path, arm_path, output_path = self.write_fixture(
                root,
                logical_event(1),
                arm_overrides={"allow_count": "0"},
            )
            with self.assertRaises(CompactDecisionLogError):
                convert_full_trace_to_compact(
                    input_path,
                    output_path,
                    arm_results_path=arm_path,
                )
            self.assertFalse(output_path.exists())

    def test_explicit_nonformal_full_trace_round_trips_to_compact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = logical_event(1)
            input_path, arm_path, output_path = self.write_fixture(root, event)
            full_records: list[dict[str, object]] = []
            stream = DecisionLogStream(
                full_records.append,
                mode=LOG_MODE_FULL_TRACE,
                is_formal_experiment=False,
            )
            stream.consume(event)
            stream.finish_run(terminal_record())
            self.assertEqual(full_records, [event])
            input_path.write_text(
                "".join(json.dumps(record) + "\n" for record in full_records),
                encoding="utf-8",
            )
            report = convert_full_trace_to_compact(
                input_path,
                output_path,
                arm_results_path=arm_path,
            )
            self.assertTrue(report["counter_equality"])
            self.assertEqual(report["compact_record_count"], 2)
            self.assertEqual(validate_decision_events(read_jsonl(output_path)), [])


class InterventionDiagnosticTests(unittest.TestCase):
    def test_intervention_rates_and_veto_burden(self) -> None:
        stats = DecisionStreamStatistics()
        for step in range(1, 28):
            stats.observe(logical_event(step))
        for step in range(28, 101):
            stats.observe(logical_event(step, veto=True))
        self.assertAlmostEqual(stats.intervention_rate, 73 / 100)
        self.assertAlmostEqual(stats.allow_rate, 27 / 100)
        self.assertAlmostEqual(stats.fallback_rate, 73 / 100)
        self.assertEqual((stats.first_veto_step, stats.last_veto_step), (28, 100))
        self.assertEqual(stats.longest_consecutive_veto_steps, 73)
        self.assertEqual(stats.longest_consecutive_allow_steps, 27)
        self.assertEqual((stats.allow_segment_count, stats.veto_segment_count), (1, 1))

    def test_phase_split_counts_two_allow_segments_but_one_allow_streak(self) -> None:
        stats = DecisionStreamStatistics()
        stats.observe(logical_event(1, phase="DESCENT"))
        stats.observe(logical_event(2, phase="POST_CROSS"))
        self.assertEqual(stats.allow_segment_count, 2)
        self.assertEqual(stats.longest_consecutive_allow_steps, 2)

    def pair(self) -> dict[str, object]:
        shared = {
            "experiment_id": "final_veto_overspeed_ablation_v0",
            "paired_run_id": "pair-1",
            "case_id": "case-1",
            "subset_id": "phase35_radial_energy_push_overspeed_stress_v0",
            "seed": 0,
            "case_config_hash": "hash",
            "controller_id": "phase35_crossing_basin_expansion",
            "r0_over_target": 0.98,
            "initial_velocity_angle_deg": 150.0,
            "thrust_scale": 8000.0,
            "invalid_simulation": False,
            "crossed_target_radius": False,
            "recoverable_crossing": False,
            "final_simulator_success": False,
            "is_formal_experiment": False,
        }
        off = {
            **shared,
            "run_id": "pair-1__monitor_off",
            "arm_id": "monitor_off",
            "overspeed": True,
            "steps": 28,
            "terminal_label": "overspeed",
            "termination_reason": "overspeed",
        }
        on = {
            **shared,
            "run_id": "pair-1__monitor_on",
            "arm_id": "monitor_on",
            "overspeed": False,
            "steps": 100_000,
            "terminal_label": "no_crossing",
            "termination_reason": "max_steps",
            "monitor_evaluation_count": 100_000,
            "allow_count": 27,
            "veto_count": 99_973,
            "fallback_count": 99_973,
            "false_negative_count": 0,
            "fallback_failure_count": 0,
            "intervention_rate": 0.99973,
        }
        return build_pair_record([off, on])

    def test_terminal_transition_and_horizon_extension_are_explicit(self) -> None:
        pair = self.pair()
        self.assertEqual(pair["terminal_outcome_transition"], "overspeed -> max_steps")
        self.assertEqual(pair["step_count_delta"], 99_972)
        self.assertEqual(pair["monitor_induced_horizon_extension"], 99_972)

    def test_hazard_avoidance_does_not_imply_task_recovery(self) -> None:
        pair = self.pair()
        self.assertTrue(pair["avoided_failure"])
        self.assertTrue(pair["declared_hazard_avoided"])
        self.assertTrue(pair["task_outcome_preserved"])
        self.assertFalse(pair["task_recovered_after_hazard_avoidance"])

    def test_summary_separates_hazard_task_burden_cost_and_transition(self) -> None:
        pair = self.pair()
        summary = build_diagnostic_summary([], [pair])
        for heading in (
            "## Declared hazard reduction",
            "## Task outcome",
            "## Intervention burden",
            "## Performance cost",
            "## Terminal failure-mode transition",
        ):
            self.assertIn(heading, summary)
        self.assertIn("does not by itself mean the task recovered", summary)


@unittest.skipUnless(
    STRESS_LOG.exists() and STRESS_ARMS.exists(),
    "local nonformal stress smoke is not present",
)
class ExistingStressConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.input_stat_before = STRESS_LOG.stat()
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / "compact_preview.jsonl"
        with mock.patch(
            "scripts.run_final_veto_ablation.execute_jobs_to_directory"
        ) as execute, mock.patch(
            "scripts.explicit_controller_phase34_post_cross_sync.rollout_phase34_case"
        ) as phase34_rollout, mock.patch(
            "scripts.explicit_controller_phase35_crossing_basin_expansion.rollout_phase35_case"
        ) as phase35_rollout:
            cls.report = convert_full_trace_to_compact(
                STRESS_LOG,
                cls.output,
                arm_results_path=STRESS_ARMS,
            )
            execute.assert_not_called()
            phase34_rollout.assert_not_called()
            phase35_rollout.assert_not_called()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_existing_stress_counters_convert_without_drift(self) -> None:
        self.assertTrue(self.report["counter_equality"])
        self.assertEqual(self.report["original_event_count"], 100_000)
        self.assertEqual(self.report["allow_event_count"], 27)
        self.assertEqual(self.report["veto_event_count"], 99_973)
        self.assertEqual(self.report["fallback_event_count"], 99_973)
        self.assertEqual(self.report["false_negative_count"], 0)
        self.assertEqual(self.report["fallback_failure_count"], 0)

    def test_existing_stress_digest_is_deterministic(self) -> None:
        self.assertEqual(
            self.report["full_stream_sha256"],
            "bf8f28dd617389bc25c5979f404a8cd24e9f9a118bcc114f4771daf75e165b96",
        )

    def test_existing_stress_compact_output_is_substantially_smaller(self) -> None:
        self.assertEqual(self.report["compact_record_count"], 3)
        self.assertEqual(self.report["compact_segment_count"], 2)
        self.assertEqual(self.report["compact_dedicated_event_count"], 0)
        self.assertEqual(self.report["compact_terminal_record_count"], 1)
        self.assertGreater(self.report["compression_ratio"], 1_000)
        self.assertLess(self.report["compact_byte_size"], self.report["original_byte_size"] / 100)

    def test_existing_stress_burden_and_terminal_transition(self) -> None:
        self.assertAlmostEqual(self.report["intervention_rate"], 0.99973)
        self.assertEqual(self.report["first_veto_step"], 28)
        self.assertEqual(self.report["last_veto_step"], 100_000)
        self.assertEqual(self.report["longest_consecutive_veto_steps"], 99_973)
        self.assertEqual(self.report["terminal_outcome_transitions"], ["overspeed -> max_steps"])

    def test_existing_volume_projects_the_frozen_formal_arm_groups(self) -> None:
        self.assertEqual(
            self.report["estimated_unbounded_preservation_byte_size"],
            1_386_642_704,
        )
        self.assertEqual(
            self.report["estimated_unbounded_stress_byte_size"],
            866_651_690,
        )
        self.assertEqual(
            self.report["estimated_unbounded_formal_byte_size"],
            2_253_294_394,
        )

    def test_conversion_does_not_modify_the_source_or_protected_artifacts(self) -> None:
        after = STRESS_LOG.stat()
        self.assertEqual(after.st_size, self.input_stat_before.st_size)
        self.assertEqual(after.st_mtime_ns, self.input_stat_before.st_mtime_ns)
        self.assertFalse(str(self.output.resolve()).startswith(str(PROJECT_ROOT.resolve())))

    def test_converted_records_are_structurally_valid(self) -> None:
        self.assertEqual(validate_decision_events(read_jsonl(self.output)), [])


if __name__ == "__main__":
    unittest.main()
