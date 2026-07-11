from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.check_final_veto_manifest import (
    EXPECTED_PRESERVATION_CASES,
    EXPECTED_STRESS_CASES,
    FORBIDDEN_MEASURED_OUTCOME_KEYS,
    ManifestValidationError,
    _forbidden_key_locations,
    _is_within_output_directory,
    _overlaps_protected_path,
    case_parameters,
    load_manifest,
    validate_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "analysis" / "final_veto_ablation_v0" / "manifest.json"
VALIDATOR_PATH = REPOSITORY_ROOT / "scripts" / "check_final_veto_manifest.py"


class FinalVetoManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest(MANIFEST_PATH)

    def write_mutated_manifest(self, manifest: dict[str, object]) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

    def test_manifest_loads_successfully(self) -> None:
        self.assertIsInstance(self.manifest, dict)
        self.assertEqual(self.manifest["experiment_id"], "final_veto_overspeed_ablation_v0")

    def test_manifest_validator_returns_success(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(VALIDATOR_PATH), "--manifest", str(MANIFEST_PATH)],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Final Veto manifest validation PASSED", completed.stdout)

    def test_preservation_set_has_eight_unique_expected_cases(self) -> None:
        section = self.manifest["preservation_set"]
        cases = section["cases"]
        parameters = {case_parameters(case) for case in cases}
        case_ids = {case["case_id"] for case in cases}
        self.assertEqual(len(cases), 8)
        self.assertEqual(len(case_ids), 8)
        self.assertEqual(parameters, EXPECTED_PRESERVATION_CASES)

    def test_stress_set_has_five_unique_expected_cases(self) -> None:
        section = self.manifest["diagnostic_stress_set"]
        cases = section["cases"]
        parameters = {case_parameters(case) for case in cases}
        case_ids = {case["case_id"] for case in cases}
        self.assertEqual(len(cases), 5)
        self.assertEqual(len(case_ids), 5)
        self.assertEqual(parameters, EXPECTED_STRESS_CASES)

    def test_preservation_and_stress_sets_are_disjoint(self) -> None:
        preservation = {case_parameters(case) for case in self.manifest["preservation_set"]["cases"]}
        stress = {case_parameters(case) for case in self.manifest["diagnostic_stress_set"]["cases"]}
        self.assertTrue(preservation.isdisjoint(stress))

    def test_threshold_and_comparator_are_frozen(self) -> None:
        hazard = self.manifest["hazard"]
        trigger = self.manifest["monitor"]["veto_trigger"]
        self.assertEqual(hazard["threshold"], 1.90)
        self.assertEqual(trigger["threshold"], 1.90)
        self.assertEqual(hazard["comparator"], ">")
        self.assertEqual(trigger["comparator"], ">")

    def test_prediction_horizon_is_one(self) -> None:
        self.assertEqual(self.manifest["monitor"]["prediction_horizon_steps"], 1)

    def test_fallback_is_zero_action_and_not_proven_safe(self) -> None:
        fallback = self.manifest["fallback"]
        self.assertEqual(fallback["action"], [0.0, 0.0])
        self.assertEqual(fallback["duration_steps"], 1)
        self.assertIs(fallback["proven_safe"], False)

    def test_output_paths_remain_isolated(self) -> None:
        output_contract = self.manifest["output_contract"]
        paths = [output_contract["manifest_path"]]
        paths.extend(item["path"] for item in output_contract["future_artifacts"])
        self.assertTrue(all(_is_within_output_directory(path) for path in paths))

    def test_protected_paths_cannot_be_output_paths(self) -> None:
        protected_paths = self.manifest["protected_paths"]
        output_paths = [item["path"] for item in self.manifest["output_contract"]["future_artifacts"]]
        self.assertFalse(any(_overlaps_protected_path(path, protected_paths) for path in output_paths))

    def test_required_acceptance_criteria_exist(self) -> None:
        criterion_ids = {
            item["criterion_id"] for item in self.manifest["acceptance_criteria"]["rules"]
        }
        required = {
            "protected_historical_guard_passes",
            "no_protected_path_modified",
            "preservation_monitor_on_crossing",
            "preservation_monitor_on_recoverable_crossing",
            "preservation_blocked_successes_zero",
            "stress_monitor_off_hazard_exercised",
            "stress_hazard_reduction",
            "blocked_successes_reported",
            "unnecessary_vetoes_reported",
            "false_negatives_reported",
            "fallback_failures_reported",
            "no_formal_safety_claim",
        }
        self.assertTrue(required <= criterion_ids)

    def test_manifest_contains_no_measured_ablation_outcomes(self) -> None:
        self.assertEqual(_forbidden_key_locations(self.manifest), [])
        serialized = json.dumps(self.manifest)
        for forbidden_key in FORBIDDEN_MEASURED_OUTCOME_KEYS:
            self.assertNotIn(f'"{forbidden_key}"', serialized)

    def test_experiment_status_remains_design_frozen_not_run(self) -> None:
        self.assertEqual(self.manifest["experiment_status"], "design_frozen_not_run")
        self.assertIs(self.manifest["monitor"]["implemented"], False)

    def test_duplicate_case_mutation_fails_validation(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["preservation_set"]["cases"].append(
            copy.deepcopy(mutated["preservation_set"]["cases"][0])
        )
        path = self.write_mutated_manifest(mutated)
        with self.assertRaises(ManifestValidationError):
            validate_manifest(path)

    def test_protected_output_path_mutation_fails_validation(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["output_contract"]["future_artifacts"][0]["path"] = (
            "analysis/phase34_post_cross_sync/results.csv"
        )
        path = self.write_mutated_manifest(mutated)
        with self.assertRaises(ManifestValidationError):
            validate_manifest(path)

    def test_threshold_drift_mutation_fails_validation(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["hazard"]["threshold"] = 1.91
        path = self.write_mutated_manifest(mutated)
        with self.assertRaises(ManifestValidationError):
            validate_manifest(path)


if __name__ == "__main__":
    unittest.main()
