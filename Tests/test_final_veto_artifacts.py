from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.final_veto_artifacts import (
    ARM_DIAGNOSTIC_FIELDNAMES,
    ARM_FIELDNAMES,
    ARM_JSON_LIST_FIELDS,
    COMPACT_DECISION_EVENT_FIELDNAMES,
    DECISION_FIELDNAMES,
    DECISION_SEGMENT_FIELDNAMES,
    PAIR_DIAGNOSTIC_FIELDNAMES,
    PAIR_FIELDNAMES,
    TERMINAL_TRANSITION_FIELDNAMES,
    ArtifactWriteError,
    JsonlEventWriter,
    resolve_output_path,
    write_csv_atomic,
    write_jsonl_atomic,
)


def complete_row(fieldnames, **overrides):
    row = {field: "" for field in fieldnames}
    row.update(overrides)
    return row


def arm_row(*, formal: bool = False):
    return complete_row(
        ARM_FIELDNAMES,
        schema_version="result_schema_v1",
        run_id="run-1",
        precursor_labels=["target_radius_crossing"],
        diagnostic_labels=["synthetic_fixture"],
        regression_set_membership=["test"],
        monitor_enabled=True,
        is_formal_experiment=formal,
    )


def pair_row(*, formal: bool = False):
    return complete_row(
        PAIR_FIELDNAMES,
        pair_schema_version="final_veto_pair_schema_v0",
        paired_run_id="pair-1",
        performance_cost_summary={"step_delta": 0},
        is_formal_experiment=formal,
    )


class FinalVetoArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"

    def test_arm_records_have_stable_field_order(self) -> None:
        path = write_csv_atomic(
            self.output,
            "results.csv",
            [arm_row()],
            ARM_FIELDNAMES,
            list_fields=ARM_JSON_LIST_FIELDS,
        )
        header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertEqual(header, list(ARM_FIELDNAMES))

    def test_pair_records_have_stable_field_order(self) -> None:
        path = write_csv_atomic(
            self.output,
            "paired_results.csv",
            [pair_row()],
            PAIR_FIELDNAMES,
        )
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        self.assertEqual(header, list(PAIR_FIELDNAMES))

    def test_list_fields_are_valid_compact_json(self) -> None:
        path = write_csv_atomic(
            self.output,
            "results.csv",
            [arm_row()],
            ARM_FIELDNAMES,
            list_fields=ARM_JSON_LIST_FIELDS,
        )
        with path.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(json.loads(row["precursor_labels"]), ["target_radius_crossing"])
        self.assertEqual(row["precursor_labels"], '["target_radius_crossing"]')
        self.assertEqual(json.loads(row["diagnostic_labels"]), ["synthetic_fixture"])

    def test_jsonl_contains_one_object_per_line(self) -> None:
        events = [
            {"decision_id": "one", "decision_type": "continue"},
            {"decision_id": "two", "decision_type": "veto_action"},
        ]
        path = write_jsonl_atomic(self.output, "decision_log.jsonl", events)
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual([json.loads(line) for line in lines], events)

    def test_jsonl_writer_streams_events(self) -> None:
        with JsonlEventWriter(self.output, "events.jsonl") as writer:
            writer.write_event({"index": 1})
            self.assertTrue(writer._temporary.exists())
            writer.write_event({"index": 2})
        self.assertEqual(
            [json.loads(line) for line in writer.destination.read_text(encoding="utf-8").splitlines()],
            [{"index": 1}, {"index": 2}],
        )

    def test_jsonl_writer_rejects_noncanonical_nan_without_publishing(self) -> None:
        with self.assertRaises(ValueError):
            write_jsonl_atomic(
                self.output,
                "decision_log.jsonl",
                [{"predicted_nominal_speed_ratio": float("nan")}],
            )
        self.assertFalse((self.output / "decision_log.jsonl").exists())
        self.assertEqual(list(self.output.glob("*.tmp")), [])

    def test_writer_refuses_protected_path(self) -> None:
        protected = self.root / "analysis" / "phase34_post_cross_sync"
        with self.assertRaises(ArtifactWriteError):
            write_csv_atomic(
                protected,
                "results.csv",
                [arm_row()],
                ARM_FIELDNAMES,
                list_fields=ARM_JSON_LIST_FIELDS,
                repository_root=self.root,
                protected_paths=["analysis/phase34_post_cross_sync/"],
            )

    def test_writer_refuses_path_traversal(self) -> None:
        with self.assertRaises(ArtifactWriteError):
            resolve_output_path(self.output, "../escaped.csv")

    def test_writer_refuses_existing_formal_file(self) -> None:
        self.output.mkdir(parents=True)
        destination = self.output / "results.csv"
        destination.write_text("existing", encoding="utf-8")
        with self.assertRaises(ArtifactWriteError):
            write_csv_atomic(
                self.output,
                "results.csv",
                [arm_row(formal=True)],
                ARM_FIELDNAMES,
                list_fields=ARM_JSON_LIST_FIELDS,
            )
        self.assertEqual(destination.read_text(encoding="utf-8"), "existing")

    def test_writer_writes_only_inside_supplied_directory(self) -> None:
        destination = write_csv_atomic(
            self.output,
            "nested/results.csv",
            [arm_row()],
            ARM_FIELDNAMES,
            list_fields=ARM_JSON_LIST_FIELDS,
        )
        self.assertEqual(destination.parent, (self.output / "nested").resolve())
        self.assertFalse((self.root / "results.csv").exists())

    def test_temporary_failure_leaves_no_partial_formal_file(self) -> None:
        def failing_rows():
            yield arm_row(formal=True)
            raise RuntimeError("synthetic writer failure")

        with self.assertRaises(RuntimeError):
            write_csv_atomic(
                self.output,
                "results.csv",
                failing_rows(),
                ARM_FIELDNAMES,
                list_fields=ARM_JSON_LIST_FIELDS,
            )
        self.assertFalse((self.output / "results.csv").exists())
        self.assertEqual(list(self.output.glob("*.tmp")), [])

    def test_formal_and_smoke_rows_remain_distinguishable(self) -> None:
        smoke = write_csv_atomic(
            self.output,
            "smoke.csv",
            [arm_row(formal=False)],
            ARM_FIELDNAMES,
            list_fields=ARM_JSON_LIST_FIELDS,
        )
        formal = write_csv_atomic(
            self.output,
            "formal.csv",
            [arm_row(formal=True)],
            ARM_FIELDNAMES,
            list_fields=ARM_JSON_LIST_FIELDS,
        )
        with smoke.open("r", encoding="utf-8", newline="") as handle:
            smoke_value = next(csv.DictReader(handle))["is_formal_experiment"]
        with formal.open("r", encoding="utf-8", newline="") as handle:
            formal_value = next(csv.DictReader(handle))["is_formal_experiment"]
        self.assertEqual(smoke_value, "false")
        self.assertEqual(formal_value, "true")

    def test_decision_field_contract_has_required_runtime_fields(self) -> None:
        for field in (
            "decision_type",
            "decision_reason",
            "nominal_proposed_action",
            "executed_action",
            "realized_executed_speed_ratio",
            "fallback_failure",
        ):
            self.assertIn(field, DECISION_FIELDNAMES)

    def test_compact_contracts_have_a_distinct_event_kind(self) -> None:
        for fields in (
            DECISION_SEGMENT_FIELDNAMES,
            COMPACT_DECISION_EVENT_FIELDNAMES,
            TERMINAL_TRANSITION_FIELDNAMES,
        ):
            self.assertIn("event_kind", fields)
        self.assertNotIn("event_kind", DECISION_FIELDNAMES)

    def test_result_contracts_include_stream_and_burden_diagnostics(self) -> None:
        for field in (
            "decision_stream_event_count",
            "decision_stream_sha256",
            "compact_decision_record_count",
            "decision_log_mode",
            "intervention_rate",
            "longest_consecutive_veto_steps",
        ):
            self.assertIn(field, ARM_DIAGNOSTIC_FIELDNAMES)
        for field in (
            "terminal_outcome_transition",
            "declared_hazard_avoided",
            "task_recovered_after_hazard_avoidance",
        ):
            self.assertIn(field, PAIR_DIAGNOSTIC_FIELDNAMES)


if __name__ == "__main__":
    unittest.main()
