from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts import analyze_stage2a_safe_alternative_usefulness_v0 as audit


ROOT = Path(__file__).resolve().parents[1]


class SafeAlternativeUsefulnessFrozenEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = audit.validate_sources(ROOT)
        cls.comparison = audit.build_alternative_comparison(ROOT)
        cls.progress = audit.build_progress_metrics(cls.comparison)
        cls.recoverability = audit.build_recoverability_comparison(cls.comparison)
        cls.decision = audit.build_decision_matrix(
            cls.comparison, cls.progress, cls.recoverability
        )
        cls.payloads = audit.build_payloads(ROOT)

    def test_four_exact_veto_states_preserved(self) -> None:
        self.assertEqual(self.comparison["exact_veto_state_count"], 4)

    def test_three_full_action_comparison_states(self) -> None:
        self.assertEqual(self.comparison["full_three_action_comparison_state_count"], 3)
        self.assertEqual(self.comparison["full_usefulness_alternative_observation_count"], 9)

    def test_angle_155_progress_remains_not_evaluated(self) -> None:
        row = next(item for item in self.comparison["comparisons"] if "angle_155" in item["case_id"])
        self.assertTrue(all(item["usefulness_evidence_status"] == "not_evaluated" for item in row["alternatives"]))
        self.assertTrue(all(item["future_recovery_success"] is None for item in row["alternatives"]))

    def test_all_full_observations_are_safe(self) -> None:
        rows = audit._available_rows(self.comparison)
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row["safe_under_frozen_threshold"] for row in rows))
        self.assertTrue(all(row["predicted_speed_ratio"] <= 1.90 for row in rows))

    def test_zero_action_has_mixed_component_progress(self) -> None:
        zero = self.progress["per_action"]["zero_action_reference_v0"]
        self.assertEqual(zero["component_results"]["radius_gap_improvement"]["improved_count"], 3)
        self.assertEqual(zero["component_results"]["radial_component_improvement"]["worsened_count"], 3)
        self.assertEqual(zero["component_results"]["absolute_tangential_error_improvement"]["improved_count"], 3)

    def test_velocity_opposed_improves_radial_component(self) -> None:
        velocity = self.progress["per_action"]["velocity_opposed_thrust_v0"]
        self.assertEqual(velocity["component_results"]["radial_component_improvement"]["improved_count"], 3)

    def test_tangential_correction_improves_tangential_error(self) -> None:
        tangential = self.progress["per_action"]["tangential_error_correction_v0"]
        self.assertEqual(tangential["component_results"]["absolute_tangential_error_improvement"]["improved_count"], 3)

    def test_no_combined_progress_score(self) -> None:
        self.assertFalse(self.progress["combined_scalar_progress_score_created"])
        self.assertTrue(all(row["combined_progress_score"] is None for row in audit._available_rows(self.comparison)))

    def test_higher_ratio_can_have_better_components(self) -> None:
        q4 = self.decision["Q4_higher_risk_better_orbital_progress"]
        self.assertTrue(q4["answer"])
        self.assertEqual(q4["qualified_evidence_count"], 3)

    def test_correlations_are_descriptive_only(self) -> None:
        correlation = self.progress["descriptive_correlations"]
        self.assertEqual(correlation["sample_count"], 9)
        self.assertEqual(correlation["independent_state_count"], 3)
        self.assertFalse(correlation["statistical_inference_authorized"])
        self.assertFalse(correlation["general_correlation_claim_authorized"])

    def test_recoverability_is_not_observed(self) -> None:
        self.assertEqual(self.recoverability["combined_predicate_true_observation_count"], 0)
        self.assertEqual(self.recoverability["eligible_crossing_observation_count"], 0)
        self.assertFalse(self.recoverability["recovery_success_claim_authorized"])

    def test_energy_remains_diagnostic_proxy(self) -> None:
        for row in audit._available_rows(self.comparison):
            self.assertEqual(row["energy_diagnostic_proxy"]["evidence_level"], "diagnostic_proxy")
            self.assertTrue(row["energy_diagnostic_proxy"]["not_a_conserved_invariant_claim"])

    def test_safety_then_usefulness_is_not_authority(self) -> None:
        q5 = self.decision["Q5_future_replacement_objective"]
        self.assertEqual(q5["answer"], "safety_gate_then_explicit_usefulness_evidence")
        self.assertFalse(q5["policy_frozen"])
        self.assertFalse(q5["active_authority_granted"])

    def test_payload_contract(self) -> None:
        manifest = audit.validate_payloads(self.payloads)
        self.assertEqual(set(self.payloads), set(audit.ALL_FILENAMES))
        self.assertEqual(manifest["physical_executions"], 0)
        self.assertEqual(manifest["controller_executions"], 0)
        self.assertFalse(manifest["Stage_2A_authority_granted"])

    def test_manifest_mutation_rejected(self) -> None:
        payloads = dict(self.payloads)
        manifest = json.loads(payloads["audit_manifest.json"])
        manifest["Stage_2A_authority_granted"] = True
        payloads["audit_manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
        with self.assertRaises(audit.SafeAlternativeUsefulnessAuditError):
            audit.validate_payloads(payloads)

    def test_report_mutation_rejected(self) -> None:
        payloads = dict(self.payloads)
        report = json.loads(payloads["progress_metrics.json"])
        report["combined_scalar_progress_score_created"] = True
        payloads["progress_metrics.json"] = json.dumps(report, sort_keys=True).encode()
        with self.assertRaises(audit.SafeAlternativeUsefulnessAuditError):
            audit.validate_payloads(payloads)

    def test_source_snapshot_stable(self) -> None:
        self.assertEqual(self.snapshot, audit.source_snapshot(ROOT))

    def test_summary_preserves_nonclaims(self) -> None:
        summary = self.payloads["summary.md"].decode("ascii")
        self.assertIn("causal recovery usefulness", summary)
        self.assertIn("No scalar utility score", summary)
        self.assertIn("Stage 2A authority remains unauthorized", summary)


class SafeAlternativeUsefulnessUnitTests(unittest.TestCase):
    def test_direction_boundaries(self) -> None:
        self.assertEqual(audit._direction(1.0), "improved")
        self.assertEqual(audit._direction(0.0), "unchanged")
        self.assertEqual(audit._direction(-1.0), "worsened")

    def test_pearson_missing_for_constant_series(self) -> None:
        self.assertIsNone(audit._pearson([1.0, 1.0], [2.0, 3.0]))

    def test_spearman_is_deterministic_with_ties(self) -> None:
        self.assertAlmostEqual(
            audit._spearman([1.0, 2.0, 2.0], [1.0, 2.0, 2.0]), 1.0
        )

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

    def test_existing_output_not_overwritten(self) -> None:
        payloads = audit.build_payloads(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / audit.OUTPUT_PATH).mkdir(parents=True)
            with self.assertRaises(audit.SafeAlternativeUsefulnessAuditError):
                audit.publish_payloads(root, payloads)

    def test_analyzer_imports_no_execution_modules(self) -> None:
        source = (ROOT / "scripts/analyze_stage2a_safe_alternative_usefulness_v0.py").read_text()
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
