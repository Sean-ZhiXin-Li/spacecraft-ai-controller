from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts import analyze_stage2a_hazard_trigger_relevance_v0 as audit


ROOT = Path(__file__).resolve().parents[1]


class HazardTriggerPredicateTests(unittest.TestCase):
    def _record(self, realized: float, predicted: float) -> dict[str, object]:
        return {
            "realized_speed_ratio": realized,
            "predicted_speed_ratio": predicted,
        }

    def test_trigger_a_strict_boundary(self) -> None:
        self.assertFalse(audit.trigger_a_matches(self._record(1.90, 1.90)))
        self.assertTrue(audit.trigger_a_matches(self._record(1.90, 1.9000001)))
        self.assertFalse(audit.trigger_a_matches(self._record(1.9000001, 2.0)))

    def test_recovery_identity_ignores_source_replication(self) -> None:
        base = {
            "case_id": "case",
            "action_identity": "zero_action_reference_v0",
            "action": [0.0, 0.0],
            "state_values": [1.0, 2.0, 3.0, 4.0],
            "realized_speed_ratio": 1.8,
            "predicted_speed_ratio": 1.81,
            "final_veto_decision": "allow",
            "source_class": "one",
        }
        duplicate = {**base, "source_class": "two"}
        self.assertEqual(audit.recovery_identity(base), audit.recovery_identity(duplicate))

    def test_recovery_identity_changes_with_action(self) -> None:
        base = {
            "case_id": "case",
            "action_identity": "branch_a",
            "action": [0.0, 0.0],
            "state_values": [1.0, 2.0, 3.0, 4.0],
            "realized_speed_ratio": 1.8,
            "predicted_speed_ratio": 1.81,
            "final_veto_decision": "allow",
        }
        changed = {**base, "action_identity": "branch_b"}
        self.assertNotEqual(audit.recovery_identity(base), audit.recovery_identity(changed))


class FrozenHazardEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = audit.validate_sources(ROOT)
        cls.records = audit.collect_recovery_proposals(ROOT)
        cls.trigger_a = audit.build_trigger_a_report(cls.records)
        cls.trigger_b = audit.build_trigger_b_report(ROOT)
        cls.comparison = audit.build_same_boundary_comparison(ROOT, cls.records)
        cls.payloads = audit.build_payloads(ROOT)

    def test_source_identities_validate(self) -> None:
        self.assertEqual(self.snapshot["d2_manifest_hash"], audit.D2_MANIFEST_HASH)
        self.assertEqual(self.snapshot["stage1b_trace_set_hash"], audit.STAGE1B_TRACE_SET_HASH)
        self.assertEqual(self.snapshot["registry_manifest_hash"], audit.REGISTRY_MANIFEST_HASH)

    def test_recovery_proposal_universe(self) -> None:
        self.assertEqual(len(self.records), 400)
        self.assertEqual(self.trigger_a["duplicate_aware_recovery_proposal_count"], 392)
        self.assertEqual(self.trigger_a["cross_artifact_reproduced_record_count"], 8)

    def test_trigger_a_not_observed(self) -> None:
        self.assertEqual(self.trigger_a["trigger_observation_count_raw"], 0)
        self.assertEqual(self.trigger_a["trigger_observation_count_duplicate_aware"], 0)
        self.assertFalse(self.trigger_a["empirically_supported"])

    def test_trigger_a_maximum_is_below_threshold(self) -> None:
        self.assertEqual(
            self.trigger_a["maximum_recovery_action_predicted_speed_ratio"],
            1.8906024003603095,
        )

    def test_provisional_velocity_opposed_action_is_separate(self) -> None:
        item = self.trigger_a["per_action"]["velocity_opposed_thrust_v0"]
        self.assertEqual(item["proposal_count"], 128)
        self.assertEqual(item["trigger_count"], 0)
        self.assertEqual(item["maximum_predicted_speed_ratio"], 1.824760375803826)

    def test_trigger_b_compact_counts(self) -> None:
        self.assertEqual(self.trigger_b["final_veto_compact_segment_count"], 5)
        self.assertEqual(self.trigger_b["final_veto_logical_observation_count"], 499877)
        self.assertEqual(self.trigger_b["d2_first_boundary_observation_count"], 2)
        self.assertEqual(self.trigger_b["cross_artifact_reproduction_count"], 1)
        self.assertEqual(self.trigger_b["duplicate_aware_logical_observation_count"], 499878)

    def test_trigger_b_cases_and_maximum(self) -> None:
        self.assertEqual(self.trigger_b["trigger_case_count"], 6)
        self.assertEqual(
            self.trigger_b["maximum_nominal_action_predicted_speed_ratio"],
            1.9183887199363643,
        )
        self.assertTrue(self.trigger_b["prevented_unsafe_nominal_proposal_execution"])

    def test_compact_veto_segments_are_uniformly_above_threshold(self) -> None:
        for segment in self.trigger_b["compact_segments"]:
            self.assertGreater(segment["minimum_predicted_speed_ratio"], 1.90)
            self.assertTrue(segment["fallback_executed"])

    def test_two_exact_same_boundary_comparisons(self) -> None:
        self.assertEqual(self.comparison["comparable_boundary_count"], 2)
        self.assertTrue(
            all(item["exact_cartesian_state_match"] for item in self.comparison["comparisons"])
        )

    def test_nominal_and_recovery_predictions_remain_action_conditional(self) -> None:
        for item in self.comparison["comparisons"]:
            self.assertGreater(item["nominal_action"]["predicted_speed_ratio"], 1.90)
            self.assertEqual(item["nominal_action"]["Final_Veto_decision"], "veto")
            for recovery in item["recovery_actions"]:
                self.assertLessEqual(recovery["predicted_speed_ratio"], 1.90)

    def test_canonical_boundary_contains_three_recovery_actions(self) -> None:
        canonical = next(
            item for item in self.comparison["comparisons"] if "angle_150" in item["case_id"]
        )
        self.assertEqual(
            {item["action_identity"] for item in canonical["recovery_actions"]},
            {
                "tangential_error_correction_v0",
                "velocity_opposed_thrust_v0",
                "zero_action_reference_v0",
            },
        )

    def test_payload_artifact_set_and_hashes(self) -> None:
        manifest = audit.validate_payloads(self.payloads)
        self.assertEqual(set(self.payloads), set(audit.ALL_FILENAMES))
        self.assertEqual(manifest["physical_executions"], 0)
        self.assertFalse(manifest["Stage_2A_authority_granted"])

    def test_manifest_mutation_is_rejected(self) -> None:
        payloads = dict(self.payloads)
        manifest = json.loads(payloads["audit_manifest.json"])
        manifest["trigger_a_observation_count"] = 1
        payloads["audit_manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
        with self.assertRaises(audit.HazardTriggerAuditError):
            audit.validate_payloads(payloads)

    def test_report_mutation_is_rejected(self) -> None:
        payloads = dict(self.payloads)
        report = json.loads(payloads["trigger_a_report.json"])
        report["empirically_supported"] = True
        payloads["trigger_a_report.json"] = json.dumps(report, sort_keys=True).encode()
        with self.assertRaises(audit.HazardTriggerAuditError):
            audit.validate_payloads(payloads)

    def test_source_snapshot_is_stable(self) -> None:
        self.assertEqual(self.snapshot, audit.source_snapshot(ROOT))

    def test_summary_preserves_non_claims(self) -> None:
        summary = self.payloads["summary.md"].decode("ascii")
        self.assertIn("do not establish that one controller is better", summary)
        self.assertIn("not a general safety proof", summary)
        self.assertIn("No simulation, controller, trajectory", summary)

    def test_evidence_matrix_preserves_not_evaluated(self) -> None:
        matrix = json.loads(self.payloads["evidence_matrix.json"])
        self.assertEqual(matrix["unknown_handling"], "not_evaluated")
        self.assertFalse(matrix["missing_physics_inference"])


class AuditSafetyTests(unittest.TestCase):
    def test_default_cli_writes_nothing(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(audit.main([]), 0)
        self.assertIn("usage:", output.getvalue())

    def test_plan_cli_writes_nothing(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(audit.main(["--plan"]), 0)
        self.assertIn("execution_enabled=false", output.getvalue())

    def test_atomic_publication_rejects_existing_target(self) -> None:
        payloads = audit.build_payloads(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / audit.OUTPUT_PATH).mkdir(parents=True)
            with self.assertRaises(audit.HazardTriggerAuditError):
                audit.publish_payloads(root, payloads)

    def test_analysis_imports_no_runtime_execution_modules(self) -> None:
        source = (ROOT / "scripts/analyze_stage2a_hazard_trigger_relevance_v0.py").read_text()
        prohibited = (
            "simulator.phase34_35_transition",
            "recovery_branch_executor",
            "stage2a_hazard_arrest_runner",
            "run_bounded_recovery",
        )
        for name in prohibited:
            self.assertNotIn(f"import {name}", source)
            self.assertNotIn(f"from {name}", source)

    def test_frozen_sources_are_not_mutated_by_analysis(self) -> None:
        before = audit.source_snapshot(ROOT)
        audit.build_payloads(ROOT)
        self.assertEqual(before, audit.source_snapshot(ROOT))


if __name__ == "__main__":
    unittest.main()
