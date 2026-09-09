from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts import analyze_stage2a_recovery_objective_definition_v0 as audit


ROOT = Path(__file__).resolve().parents[1]


class RecoveryObjectiveDefinitionFrozenEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = audit.source_snapshot(ROOT)
        cls.payloads = audit.build_payloads(ROOT)
        cls.inventory = json.loads(cls.payloads["objective_inventory.json"])
        cls.metrics = json.loads(cls.payloads["metric_definition.json"])
        cls.conflicts = json.loads(cls.payloads["conflict_analysis.json"])
        cls.unknowns = json.loads(cls.payloads["unknown_evidence.json"])

    def metric(self, metric_id: str) -> dict[str, object]:
        return next(row for row in self.metrics["metrics"] if row["metric_id"] == metric_id)

    def test_source_identities_validate(self) -> None:
        validated = audit.validate_sources(ROOT)
        self.assertEqual(validated["snapshot"], self.snapshot)

    def test_safety_metrics_are_constraints(self) -> None:
        self.assertEqual(
            self.metric("predicted_action_speed_feasibility")["objective_role"],
            "safety_constraint",
        )
        self.assertEqual(
            self.metric("realized_overspeed_state")["objective_role"], "safety_constraint"
        )

    def test_strict_overspeed_threshold_preserved(self) -> None:
        predicted = self.metric("predicted_action_speed_feasibility")["threshold"]
        realized = self.metric("realized_overspeed_state")["threshold"]
        self.assertEqual(predicted, {"value": 1.9, "comparator": "<=", "source": "Final_Veto_v0"})
        self.assertEqual(realized["value"], 1.9)
        self.assertEqual(realized["comparator"], ">")

    def test_phase34_thresholds_preserved(self) -> None:
        self.assertEqual(self.metric("absolute_radius_gap")["threshold"]["value"], 0.0025)
        self.assertEqual(self.metric("radial_velocity_component")["threshold"]["value"], 0.02)
        self.assertEqual(
            self.metric("absolute_tangential_velocity_error")["threshold"]["value"], 0.25
        )

    def test_progress_metrics_are_component_wise(self) -> None:
        roles = self.inventory["metrics_by_role"]
        self.assertIn("absolute_radius_gap", roles["recovery_progress"])
        self.assertIn("absolute_tangential_velocity_error", roles["recovery_progress"])
        self.assertIn("radial_velocity_component", roles["recovery_progress_and_arrival_condition"])

    def test_headroom_is_not_mission_score(self) -> None:
        headroom = self.metric("overspeed_headroom")
        self.assertEqual(headroom["objective_role"], "safety_margin_progress")
        self.assertIn("not_a_mission_score", headroom["use_in_future_semantics"])

    def test_energy_is_diagnostic_only(self) -> None:
        energy = self.metric("specific_energy_error_proxy")
        self.assertEqual(energy["objective_role"], "diagnostic_only")
        self.assertEqual(energy["evidence_level"], "diagnostic_proxy")

    def test_recoverability_and_crossing_are_milestones(self) -> None:
        self.assertEqual(
            self.metric("phase34_compatible_recoverability_components")["objective_role"],
            "recovery_milestone",
        )
        self.assertEqual(
            self.metric("eligible_target_radius_crossing")["objective_role"],
            "recovery_milestone",
        )

    def test_future_evaluators_are_explicit(self) -> None:
        self.assertEqual(self.metric("recovery_success_v0")["objective_role"], "future_evaluator")
        self.assertEqual(self.metric("handoff_readiness")["objective_role"], "future_evaluator")

    def test_conflicts_are_preserved(self) -> None:
        self.assertEqual(self.conflicts["conflict_count"], 7)
        ids = {row["conflict_id"] for row in self.conflicts["conflicts"]}
        self.assertIn("radius_closure_vs_radial_arrival_condition", ids)
        self.assertIn("lower_speed_ratio_vs_tangential_progress", ids)
        self.assertIn("directional_progress_vs_recoverability_milestone", ids)

    def test_observed_conflict_counts_match_stage2a_i(self) -> None:
        radial = next(
            row
            for row in self.conflicts["conflicts"]
            if row["conflict_id"] == "radius_closure_vs_radial_arrival_condition"
        )
        self.assertEqual(radial["stage2a_i_evidence"]["zero_radius_gap_improved_count"], 3)
        self.assertEqual(radial["stage2a_i_evidence"]["zero_radial_component_worsened_count"], 3)

    def test_unknowns_remain_not_evaluated(self) -> None:
        self.assertEqual(self.unknowns["unknown_count"], 9)
        self.assertTrue(all(row["status"] == "not_evaluated" for row in self.unknowns["unknowns"]))
        self.assertTrue(all(row["may_be_converted_to_false"] is False for row in self.unknowns["unknowns"]))

    def test_correction_authority_remains_unknown(self) -> None:
        row = next(
            item
            for item in self.unknowns["unknowns"]
            if item["evidence_id"] == "available_correction_authority"
        )
        self.assertEqual(row["status"], "not_evaluated")
        self.assertFalse(row["may_be_used_as_positive_action_evidence"])

    def test_lexicographic_safety_first_recommendation(self) -> None:
        self.assertEqual(
            self.inventory["future_action_selection_semantics_recommendation"],
            "lexicographic_safety_first",
        )
        self.assertEqual(self.inventory["recommendation_status"], "conceptual_not_a_policy")

    def test_no_combined_score_or_weighting(self) -> None:
        self.assertIsNone(self.metrics["combined_score"])
        self.assertIsNone(self.metrics["weights"])
        self.assertIsNone(self.metrics["optimizer"])
        self.assertIsNone(self.conflicts["conflict_resolution_policy"])
        self.assertIsNone(self.inventory["policy"])
        self.assertIsNone(self.inventory["action_selection"])

    def test_no_authority_or_execution(self) -> None:
        self.assertEqual(self.inventory["physical_executions"], 0)
        self.assertEqual(self.inventory["controller_executions"], 0)
        self.assertFalse(self.inventory["Stage_2A_authority_granted"])
        self.assertEqual(self.inventory["staged_recovery_execution"], "not_authorized")

    def test_payload_contract(self) -> None:
        validated = audit.validate_payloads(self.payloads)
        self.assertEqual(set(self.payloads), set(audit.ALL_FILENAMES))
        self.assertEqual(validated["canonical_payload_hash"], self.inventory["canonical_payload_hash"])

    def test_payload_hash_mutation_rejected(self) -> None:
        payloads = dict(self.payloads)
        metric = json.loads(payloads["metric_definition.json"])
        metric["weights"] = {"radius": 1.0}
        payloads["metric_definition.json"] = json.dumps(metric, sort_keys=True).encode()
        with self.assertRaises(audit.RecoveryObjectiveDefinitionAuditError):
            audit.validate_payloads(payloads)

    def test_source_snapshot_unchanged_by_build(self) -> None:
        self.assertEqual(self.snapshot, audit.source_snapshot(ROOT))

    def test_summary_preserves_nonclaims(self) -> None:
        summary = self.payloads["summary.md"].decode("ascii")
        self.assertIn("not a policy", summary)
        self.assertIn("no combined", summary)
        self.assertIn("Stage 2A authority remains unauthorized", summary)


class RecoveryObjectiveDefinitionSafetyTests(unittest.TestCase):
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
            with self.assertRaises(audit.RecoveryObjectiveDefinitionAuditError):
                audit.publish_payloads(root, payloads)

    def test_analyzer_imports_no_execution_modules(self) -> None:
        source = (ROOT / "scripts/analyze_stage2a_recovery_objective_definition_v0.py").read_text()
        for module in (
            "simulator.phase34_35_transition",
            "recovery_branch_executor",
            "stage2a_hazard_arrest_runner",
            "run_bounded_recovery",
        ):
            self.assertNotIn(f"import {module}", source)
            self.assertNotIn(f"from {module}", source)

    def test_no_threshold_tuning_interface(self) -> None:
        options = audit.build_parser()._option_string_actions
        self.assertNotIn("--threshold", options)
        self.assertNotIn("--weight", options)
        self.assertNotIn("--optimize", options)


if __name__ == "__main__":
    unittest.main()
