from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from scripts import analyze_recovery_branch_mechanisms_v0 as analysis


class RecoveryBranchMechanismAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self._write_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_fixture(self) -> None:
        manifest = {
            "experiment_id": analysis.EXPERIMENT_ID,
            "branches": [
                {"branch_id": branch} for branch in analysis.BRANCH_ORDER
            ],
        }
        branch_state = {
            "schema_version": "recovery_branch_state_v0",
            "case_id": "synthetic_case",
            "state": {
                "position_x": 0.0,
                "position_y": 9.0,
                "velocity_x": -2.0,
                "velocity_y": 1.0,
                "current_phase": "DESCENT",
            },
            "simulator_configuration": {
                "thrust_scale": 8.0,
                "simulator_constants": {
                    "target_radius": 10.0,
                    "target_circular_speed": 2.0,
                    "mu": 100.0,
                },
            },
        }
        branch_state["canonical_branch_state_hash"] = analysis.canonical_sha256(
            branch_state
        )
        (self.source / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        (self.source / "branch_state.json").write_text(
            json.dumps(branch_state, sort_keys=True), encoding="utf-8"
        )
        manifest_hash = analysis.canonical_sha256(manifest)
        branch_hash = branch_state["canonical_branch_state_hash"]
        rows = []
        for branch in analysis.BRANCH_ORDER:
            physical = branch != "explicit_abort_v0"
            action_effort = (
                "0.0"
                if branch == "zero_action_reference_v0"
                else "0.5"
                if physical
                else ""
            )
            rows.append(
                {
                    "experiment_id": analysis.EXPERIMENT_ID,
                    "branch_id": branch,
                    "branch_state_hash": branch_hash,
                    "manifest_hash": manifest_hash,
                    "implementation_commit": analysis.IMPLEMENTATION_COMMIT,
                    "seed": "0",
                    "branch_step": "28",
                    "nominal_prefix_transition_count": "27",
                    "recovery_transition_count": "2" if physical else "0",
                    "total_transition_count": "29" if physical else "27",
                    "terminal_reason": (
                        "recovery_horizon_exhausted" if physical else "explicit_abort"
                    ),
                    "overspeed_status": "clear",
                    "instability_status": "clear",
                    "unsafe_state_status": "clear",
                    "invalid_simulation_status": "clear",
                    "crossed_target_radius": "false",
                    "phase34_compatible_recoverable_crossing": "false",
                    "recovery_success": "false",
                    "final_simulator_success": "false",
                    "monitor_evaluation_count": "2" if physical else "0",
                    "allow_count": "2" if physical else "0",
                    "veto_count": "0",
                    "normalized_control_effort": action_effort,
                    "delta_v_proxy": action_effort,
                    "final_radius_error": "-0.5" if physical else "-1.0",
                    "final_radial_velocity_error": "0.5" if physical else "1.0",
                    "final_tangential_velocity_error": "-0.25" if physical else "0.0",
                    "first_intervention_step": "",
                    "longest_intervention_streak": "0",
                    "first_crossing_step": "",
                    "first_recoverable_crossing_step": "",
                }
            )
        self._write_rows(rows)

        initial_hash = analysis._branch_state_vector_hash(branch_state)
        events = []
        actions = {
            "zero_action_reference_v0": [0.0, 0.0],
            "velocity_opposed_thrust_v0": [0.15, -0.2],
            "tangential_error_correction_v0": [0.25, 0.0],
        }
        for branch_index, branch in enumerate(analysis.PHYSICAL_BRANCHES):
            current = initial_hash
            for index in range(2):
                next_hash = hashlib.sha256(
                    f"{branch}-{index}".encode("ascii")
                ).hexdigest()
                events.append(
                    self._event(
                        branch,
                        branch_hash,
                        index,
                        current,
                        next_hash,
                        actions[branch],
                        1.8 - branch_index * 0.1 - index * 0.01,
                        terminal=(index == 1),
                    )
                )
                current = next_hash
        events.append(
            {
                **self._event(
                    "explicit_abort_v0",
                    branch_hash,
                    0,
                    initial_hash,
                    None,
                    None,
                    None,
                    terminal=True,
                ),
                "post_branch_step": 0,
                "total_transition_count": 27,
                "final_veto_decision": "not_applicable",
                "transition_occurred": False,
                "terminal_reason": "explicit_abort",
                "triggered_stop_condition": "explicit_abort",
            }
        )
        self._write_events(events)
        (self.source / "summary.md").write_text("synthetic\n", encoding="utf-8")
        (self.source / "comparison.png").write_bytes(b"synthetic-png")

    @staticmethod
    def _event(
        branch: str,
        branch_hash: str,
        index: int,
        current_hash: str,
        next_hash: str | None,
        action: list[float] | None,
        ratio: float | None,
        *,
        terminal: bool,
    ) -> dict[str, object]:
        return {
            "branch_id": branch,
            "branch_state_hash": branch_hash,
            "case_id": "synthetic_case",
            "current_state_hash": current_hash,
            "decision_event_schema_version": "recovery_decision_event_v0",
            "event_index": index,
            "post_branch_step": index + 1,
            "total_transition_count": 28 + index,
            "proposed_action": action,
            "executed_action": action,
            "final_veto_decision": "allow" if action is not None else "not_applicable",
            "transition_occurred": action is not None,
            "predicted_next_state_hash": next_hash,
            "next_state_hash": next_hash,
            "predicted_speed_ratio": ratio,
            "realized_speed_ratio": ratio,
            "hazard_threshold": 1.9,
            "hazard_comparator": ">",
            "evaluator_statuses": [
                ["overspeed", "clear"],
                ["recovery_success", "clear"],
            ],
            "terminal_reason": "recovery_horizon_exhausted" if terminal else None,
            "triggered_stop_condition": (
                "recovery_horizon_exhausted" if terminal else None
            ),
            "experiment_id": analysis.EXPERIMENT_ID,
            "evidence_level": "measured",
            "seed": 0,
        }

    def _read_rows(self) -> list[dict[str, str]]:
        with (self.source / "results.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            return list(csv.DictReader(handle))

    def _write_rows(self, rows: list[dict[str, str]]) -> None:
        fields = list(rows[0])
        with (self.source / "results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def _read_events(self) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in (self.source / "decision_log.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    def _write_events(self, events: list[dict[str, object]]) -> None:
        payload = "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in events
        )
        (self.source / "decision_log.jsonl").write_text(payload, encoding="utf-8")

    def _bundle(self) -> analysis.SourceBundle:
        return analysis.load_source_bundle(
            self.source, enforce_frozen_hashes=False
        )

    def _small_counts(self):
        return mock.patch.object(
            analysis,
            "EXPECTED_EVENT_COUNTS",
            {
                "zero_action_reference_v0": 2,
                "velocity_opposed_thrust_v0": 2,
                "tangential_error_correction_v0": 2,
                "explicit_abort_v0": 1,
            },
        )

    # Artifact parsing

    def test_valid_results_csv_is_accepted(self) -> None:
        self.assertEqual(len(analysis.load_results_csv(self.source / "results.csv")), 4)

    def test_valid_decision_jsonl_is_accepted(self) -> None:
        grouped, *_ = analysis.load_decision_events(self.source / "decision_log.jsonl")
        self.assertEqual(len(grouped["zero_action_reference_v0"]), 2)

    def test_malformed_jsonl_is_rejected(self) -> None:
        (self.source / "decision_log.jsonl").write_text("{bad\n", encoding="utf-8")
        with self.assertRaisesRegex(analysis.MechanismAnalysisError, "malformed"):
            analysis.load_decision_events(self.source / "decision_log.jsonl")

    def test_duplicate_branch_record_is_rejected(self) -> None:
        rows = self._read_rows()
        rows[-1]["branch_id"] = rows[-2]["branch_id"]
        self._write_rows(rows)
        with self.assertRaisesRegex(analysis.MechanismAnalysisError, "duplicate"):
            analysis.load_results_csv(self.source / "results.csv")

    def test_unknown_branch_is_rejected(self) -> None:
        rows = self._read_rows()
        rows[-1]["branch_id"] = "unknown"
        self._write_rows(rows)
        with self.assertRaisesRegex(analysis.MechanismAnalysisError, "unknown"):
            analysis.load_results_csv(self.source / "results.csv")

    def test_wrong_branch_order_is_rejected(self) -> None:
        rows = self._read_rows()
        rows[0], rows[1] = rows[1], rows[0]
        self._write_rows(rows)
        with self.assertRaisesRegex(analysis.MechanismAnalysisError, "branch order"):
            analysis.load_results_csv(self.source / "results.csv")

    def test_inconsistent_branch_state_hash_is_reported(self) -> None:
        rows = self._read_rows()
        rows[0]["branch_state_hash"] = "0" * 64
        self._write_rows(rows)
        with self._small_counts():
            issues = analysis.structural_issues(self._bundle())
        self.assertTrue(any("branch-state hash" in issue for issue in issues))

    def test_inconsistent_manifest_hash_is_reported(self) -> None:
        rows = self._read_rows()
        rows[0]["manifest_hash"] = "0" * 64
        self._write_rows(rows)
        with self._small_counts():
            issues = analysis.structural_issues(self._bundle())
        self.assertTrue(any("manifest hash" in issue for issue in issues))

    def test_event_count_mismatch_is_reported(self) -> None:
        issues = analysis.structural_issues(self._bundle())
        self.assertTrue(any("event-count mismatch" in issue for issue in issues))

    def test_explicit_abort_transition_evidence_is_rejected(self) -> None:
        events = self._read_events()
        events[-1]["transition_occurred"] = True
        events[-1]["next_state_hash"] = "f" * 64
        self._write_events(events)
        with self._small_counts():
            issues = analysis.structural_issues(self._bundle())
        self.assertTrue(any("explicit abort" in issue for issue in issues))

    # Field inventory

    def test_field_inventory_classifies_measured_and_derived_fields(self) -> None:
        inventory = analysis.build_field_inventory(self._bundle())
        classes = inventory["diagnostic_quantity_classification"]
        self.assertEqual(classes["per_step_action_vectors"], "directly_measured")
        self.assertEqual(classes["action_magnitude"], "derivable_from_measured_fields")

    def test_unavailable_physical_quantities_remain_unavailable(self) -> None:
        classes = analysis.build_field_inventory(self._bundle())[
            "diagnostic_quantity_classification"
        ]
        self.assertEqual(classes["per_step_radius"], analysis.UNAVAILABLE)
        self.assertEqual(classes["per_step_orbital_energy"], analysis.UNAVAILABLE)

    def test_state_hashes_are_not_classified_as_state_vectors(self) -> None:
        classes = analysis.build_field_inventory(self._bundle())[
            "diagnostic_quantity_classification"
        ]
        self.assertEqual(classes["state_hashes"], "directly_measured")
        self.assertEqual(classes["per_step_state_vectors"], analysis.UNAVAILABLE)

    def test_missing_checkpoint_fields_are_not_converted_to_zero(self) -> None:
        row = analysis.extract_checkpoints(self._bundle())[0]
        self.assertIsNone(row["radius"])
        payload = analysis.csv_bytes((row,), analysis.CHECKPOINT_FIELDS).decode()
        self.assertIn(analysis.UNAVAILABLE, payload)

    # Metrics

    def test_action_effort_recomputes_correctly(self) -> None:
        events = self._bundle().events_by_branch["velocity_opposed_thrust_v0"]
        self.assertAlmostEqual(analysis.summarize_actions(events)["effort"], 0.5)

    def test_fixed_quarter_action_for_10000_steps_has_effort_2500(self) -> None:
        events = tuple({"executed_action": [0.15, 0.2]} for _ in range(10000))
        self.assertAlmostEqual(analysis.summarize_actions(events)["effort"], 2500.0)

    def test_allow_and_veto_counts_recompute_from_events(self) -> None:
        events = [
            {"post_branch_step": 1, "final_veto_decision": "allow"},
            {"post_branch_step": 2, "final_veto_decision": "veto"},
        ]
        result = analysis.summarize_monitor_decisions(events)
        self.assertEqual(result["evaluation_count"], 2)
        self.assertEqual(result["allow_count"], 1)
        self.assertEqual(result["veto_count"], 1)

    def test_speed_ratio_extrema_recompute_correctly(self) -> None:
        events = [
            {
                "post_branch_step": 1,
                "realized_speed_ratio": 1.8,
                "predicted_speed_ratio": 1.8,
                "hazard_threshold": 1.9,
            },
            {
                "post_branch_step": 2,
                "realized_speed_ratio": 1.7,
                "predicted_speed_ratio": 1.7,
                "hazard_threshold": 1.9,
            },
        ]
        result = analysis.summarize_speed_ratios(events)
        self.assertEqual(result["maximum"], 1.8)
        self.assertEqual(result["minimum"], 1.7)
        self.assertEqual(result["closest_threshold_step"], 1)

    def test_checkpoint_extraction_uses_exact_events_without_interpolation(self) -> None:
        checkpoints = analysis.extract_checkpoints(self._bundle())
        steps = [
            row["recovery_step"]
            for row in checkpoints
            if row["branch_id"] == "zero_action_reference_v0"
        ]
        self.assertEqual(steps, [1])

    # Divergence

    def test_common_initial_state_and_first_divergence_are_detected(self) -> None:
        result = analysis.analyze_trajectory_divergence(self._bundle())
        self.assertTrue(result["common_pre_transition_state_at_step_1"])
        self.assertEqual(result["first_next_state_hash_divergence_step"], 1)

    def test_exact_later_hash_convergence_is_detected(self) -> None:
        events = self._read_events()
        common = "a" * 64
        events[1]["next_state_hash"] = common
        events[3]["next_state_hash"] = common
        self._write_events(events)
        result = analysis.analyze_trajectory_divergence(self._bundle())
        pair = result["pairwise"][
            "zero_action_reference_v0__vs__velocity_opposed_thrust_v0"
        ]
        self.assertTrue(pair["exact_later_state_hash_match_observed"])

    def test_different_hashes_are_not_converted_to_physical_distance(self) -> None:
        result = analysis.analyze_trajectory_divergence(self._bundle())
        self.assertIn("not derivable", result["different_hash_interpretation"])

    # Mechanism classification

    def test_timeout_alone_does_not_prove_horizon_limitation(self) -> None:
        with self._small_counts():
            metrics = analysis.compute_branch_metrics(self._bundle())
        mechanisms = analysis.classify_mechanisms(
            metrics, analysis.analyze_trajectory_divergence(self._bundle())
        )
        horizon = next(item for item in mechanisms if item["mechanism_id"].startswith("F_"))
        self.assertEqual(horizon["status"], "not_evaluable")

    def test_no_crossing_alone_is_only_partial_radial_evidence(self) -> None:
        with self._small_counts():
            metrics = analysis.compute_branch_metrics(self._bundle())
        mechanisms = analysis.classify_mechanisms(
            metrics, analysis.analyze_trajectory_divergence(self._bundle())
        )
        radial = next(item for item in mechanisms if item["mechanism_id"].startswith("B_"))
        self.assertEqual(radial["status"], "partially_supported")

    def test_missing_physical_fields_support_observability_diagnosis(self) -> None:
        with self._small_counts():
            metrics = analysis.compute_branch_metrics(self._bundle())
        mechanisms = analysis.classify_mechanisms(
            metrics, analysis.analyze_trajectory_divergence(self._bundle())
        )
        item = next(item for item in mechanisms if item["mechanism_id"].startswith("I_"))
        self.assertEqual(item["status"], "supported")

    def test_mechanism_claim_scope_is_preserved(self) -> None:
        with self._small_counts():
            metrics = analysis.compute_branch_metrics(self._bundle())
        mechanisms = analysis.classify_mechanisms(
            metrics, analysis.analyze_trajectory_divergence(self._bundle())
        )
        state_region = next(item for item in mechanisms if item["mechanism_id"].startswith("G_"))
        self.assertIn("one frozen branch state", state_region["scope"])

    # Repository and output safety

    def test_script_imports_no_rollout_controller_or_simulator_module(self) -> None:
        tree = ast.parse(Path(analysis.__file__).read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        forbidden = ("recovery_experiment_runner", "recovery_branch_runner", "controller", "simulator")
        self.assertFalse(any(any(token in name for token in forbidden) for name in imports))

    def test_output_inside_source_directory_is_rejected(self) -> None:
        with self.assertRaisesRegex(analysis.MechanismAnalysisError, "overlaps"):
            analysis.validate_output_directory(
                self.source / "diagnosis",
                repository_root=self.root,
                source_directory=self.source,
            )

    def test_repeated_text_analysis_is_byte_deterministic(self) -> None:
        bundle = self._bundle()
        with self._small_counts():
            first = analysis.analysis_text_artifacts(bundle)
            second = analysis.analysis_text_artifacts(bundle)
        self.assertEqual(first, second)

    def test_synthetic_publish_writes_valid_pngs_outside_source(self) -> None:
        bundle = self._bundle()
        target = self.root / "diagnosis"
        with self._small_counts():
            hashes = analysis.publish_analysis(
                bundle, target, repository_root=self.root
            )
        self.assertIn("speed_ratio_trajectory.png", hashes)
        data = (target / "speed_ratio_trajectory.png").read_bytes()
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        width, height = struct.unpack(">II", data[16:24])
        self.assertEqual((width, height), (1200, 660))
        self.assertFalse((self.source / "field_inventory.json").exists())

    def test_publish_refuses_existing_output(self) -> None:
        target = self.root / "diagnosis"
        target.mkdir()
        with self.assertRaisesRegex(analysis.MechanismAnalysisError, "overwrite"):
            analysis.validate_output_directory(
                target,
                repository_root=self.root,
                source_directory=self.source,
            )

    def test_frozen_measured_artifact_hashes_match_declared_sources(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        source = repository / "analysis" / "recovery_action_branching_nonformal_v0"
        actual = {
            name: analysis.sha256_file(source / name)
            for name in analysis.SOURCE_FILENAMES
        }
        self.assertEqual(actual, analysis.FROZEN_SOURCE_HASHES)


if __name__ == "__main__":
    unittest.main()
