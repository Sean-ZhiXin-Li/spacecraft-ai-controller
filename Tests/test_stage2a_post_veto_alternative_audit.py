from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts import analyze_stage2a_post_veto_alternative_audit_v0 as audit


ROOT = Path(__file__).resolve().parents[1]


class PostVetoFrozenEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = audit.validate_sources(ROOT)
        cls.inventory = audit.build_veto_event_inventory(ROOT)
        cls.comparisons = audit.build_exact_state_comparisons(ROOT)
        cls.coverage = audit.build_alternative_coverage(cls.inventory, cls.comparisons)
        cls.payloads = audit.build_payloads(ROOT)

    def test_compact_veto_event_count(self) -> None:
        self.assertEqual(self.inventory["compact_veto_segment_count"], 5)
        self.assertEqual(self.inventory["compact_logical_veto_event_count"], 499877)

    def test_duplicate_aware_veto_event_count(self) -> None:
        self.assertEqual(self.inventory["D2_first_veto_event_count"], 2)
        self.assertEqual(self.inventory["cross_artifact_reproduction_count"], 1)
        self.assertEqual(self.inventory["duplicate_aware_veto_event_count"], 499878)

    def test_compact_state_identity_remains_not_evaluated(self) -> None:
        for segment in self.inventory["segments"]:
            self.assertIsNone(segment["state_identity"])
            self.assertEqual(segment["state_identity_status"], "not_evaluated_compact_log")
            self.assertEqual(segment["nominal_action_per_step"], "not_evaluated_compact_log")

    def test_every_compact_segment_has_safe_zero_fallback(self) -> None:
        for segment in self.inventory["segments"]:
            fallback = segment["zero_action_fallback"]
            self.assertEqual(fallback["action_identity"], "zero_action_reference_v0")
            self.assertLessEqual(fallback["maximum_predicted_speed_ratio"], 1.90)
            self.assertFalse(fallback["fallback_failure"])

    def test_four_exact_veto_states(self) -> None:
        self.assertEqual(self.comparisons["exact_state_comparison_count"], 4)
        self.assertEqual(
            len({item["state_identity"] for item in self.comparisons["comparisons"]}), 4
        )

    def test_exact_nominal_proposals_are_vetoed(self) -> None:
        for comparison in self.comparisons["comparisons"]:
            nominal = comparison["nominal_proposal"]
            self.assertGreater(nominal["predicted_speed_ratio"], 1.90)
            self.assertEqual(nominal["Final_Veto_decision"], "veto")

    def test_each_exact_state_has_safe_zero_action(self) -> None:
        for comparison in self.comparisons["comparisons"]:
            zero = next(
                item
                for item in comparison["alternatives"]
                if item["action_identity"] == "zero_action_reference_v0"
            )
            self.assertTrue(zero["safe_under_frozen_threshold"])

    def test_velocity_opposed_coverage_is_sparse_and_safe(self) -> None:
        result = self.coverage["alternative_results"]["velocity_opposed_thrust_v0"]
        self.assertEqual(result["logical_events_with_available_prediction"], 3)
        self.assertEqual(result["logical_events_safe"], 3)
        self.assertEqual(result["logical_events_not_evaluated"], 499875)
        self.assertEqual(result["maximum_evaluated_predicted_speed_ratio"], 1.824760375803826)

    def test_tangential_coverage_is_sparse_and_safe(self) -> None:
        result = self.coverage["alternative_results"]["tangential_error_correction_v0"]
        self.assertEqual(result["logical_events_with_available_prediction"], 3)
        self.assertEqual(result["logical_events_safe"], 3)
        self.assertEqual(result["logical_events_not_evaluated"], 499875)
        self.assertEqual(result["maximum_evaluated_predicted_speed_ratio"], 1.8494516264933416)

    def test_zero_action_is_most_frequently_safe(self) -> None:
        zero = self.coverage["alternative_results"]["zero_action_reference_v0"]
        self.assertEqual(zero["logical_events_safe"], 499878)
        self.assertEqual(zero["logical_events_not_evaluated"], 0)
        self.assertEqual(self.coverage["most_frequently_safe_alternative"], "zero_action_reference_v0")

    def test_all_veto_events_have_at_least_one_safe_alternative(self) -> None:
        self.assertEqual(
            self.coverage["veto_events_with_at_least_one_safe_alternative"], 499878
        )
        self.assertEqual(self.coverage["veto_events_without_safe_alternative_evidence"], 0)

    def test_evaluated_physical_alternatives_are_below_threshold(self) -> None:
        self.assertTrue(
            self.coverage["evaluated_physical_alternatives_consistently_at_or_below_1p90"]
        )
        self.assertFalse(self.coverage["general_consistency_claim_authorized"])

    def test_explicit_abort_is_terminal_not_action(self) -> None:
        result = self.coverage["alternative_results"]["explicit_abort_v0"]
        self.assertFalse(result["physical_action_alternative"])
        self.assertEqual(result["observed_terminal_semantics_count"], 1)
        self.assertEqual(result["logical_events_with_available_prediction"], 0)
        self.assertIsNone(result["maximum_evaluated_predicted_speed_ratio"])

    def test_missing_alternatives_are_not_converted_to_false(self) -> None:
        angle_155 = next(
            item for item in self.comparisons["comparisons"] if "angle_155" in item["case_id"]
        )
        unavailable = [
            item
            for item in angle_155["alternatives"]
            if item["action_identity"] in {
                "velocity_opposed_thrust_v0",
                "tangential_error_correction_v0",
            }
        ]
        self.assertTrue(all(item["allowed_or_rejected_status"] == "not_evaluated" for item in unavailable))
        self.assertTrue(all(item["safe_under_frozen_threshold"] is None for item in unavailable))

    def test_payload_contract(self) -> None:
        manifest = audit.validate_payloads(self.payloads)
        self.assertEqual(set(self.payloads), set(audit.ALL_FILENAMES))
        self.assertEqual(manifest["physical_executions"], 0)
        self.assertFalse(manifest["Stage_2A_authority_granted"])
        self.assertFalse(manifest["D1_D2_rerun"])

    def test_interpretation_is_replacement_opportunity(self) -> None:
        interpretation = json.loads(self.payloads["final_veto_interpretation.json"])
        self.assertEqual(interpretation["forced_choice"], "action_replacement_opportunity")
        self.assertFalse(interpretation["terminal_safety_barrier"])
        self.assertTrue(interpretation["action_replacement_observed"])
        self.assertEqual(interpretation["authority_consequence"], "none")

    def test_evidence_matrix_preserves_unknowns(self) -> None:
        evidence = json.loads(self.payloads["evidence_matrix.json"])
        self.assertEqual(evidence["unknown_value_policy"], "not_evaluated")
        self.assertFalse(evidence["new_physics_inference"])

    def test_manifest_mutation_is_rejected(self) -> None:
        payloads = dict(self.payloads)
        manifest = json.loads(payloads["audit_manifest.json"])
        manifest["Stage_2A_authority_granted"] = True
        payloads["audit_manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
        with self.assertRaises(audit.PostVetoAuditError):
            audit.validate_payloads(payloads)

    def test_report_mutation_is_rejected(self) -> None:
        payloads = dict(self.payloads)
        report = json.loads(payloads["alternative_coverage.json"])
        report["veto_events_with_at_least_one_safe_alternative"] = 1
        payloads["alternative_coverage.json"] = json.dumps(report, sort_keys=True).encode()
        with self.assertRaises(audit.PostVetoAuditError):
            audit.validate_payloads(payloads)

    def test_source_snapshot_is_stable(self) -> None:
        self.assertEqual(self.snapshot, audit.source_snapshot(ROOT))

    def test_summary_preserves_claim_limits(self) -> None:
        summary = self.payloads["summary.md"].decode("ascii")
        self.assertIn("sparse coverage does not support", summary)
        self.assertIn("authorizes active Stage 2A replacement", summary)
        self.assertIn("Physical executions: 0", summary)


class PostVetoAuditSafetyTests(unittest.TestCase):
    def test_default_cli_writes_nothing(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(audit.main([]), 0)
        self.assertIn("usage:", output.getvalue())

    def test_plan_writes_nothing(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(audit.main(["--plan"]), 0)
        self.assertIn("execution_enabled=false", output.getvalue())

    def test_existing_output_is_not_overwritten(self) -> None:
        payloads = audit.build_payloads(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / audit.OUTPUT_PATH).mkdir(parents=True)
            with self.assertRaises(audit.PostVetoAuditError):
                audit.publish_payloads(root, payloads)

    def test_analyzer_imports_no_execution_modules(self) -> None:
        source = (ROOT / "scripts/analyze_stage2a_post_veto_alternative_audit_v0.py").read_text()
        for module in (
            "simulator.phase34_35_transition",
            "recovery_branch_executor",
            "stage2a_hazard_arrest_runner",
            "run_bounded_recovery",
        ):
            self.assertNotIn(f"import {module}", source)
            self.assertNotIn(f"from {module}", source)

    def test_build_is_read_only_for_frozen_sources(self) -> None:
        before = audit.source_snapshot(ROOT)
        audit.build_payloads(ROOT)
        self.assertEqual(before, audit.source_snapshot(ROOT))


if __name__ == "__main__":
    unittest.main()
