from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.check_recovery_action_branching_manifest import (
    EXPECTED_BRANCH_IDS,
    EXPECTED_OUTPUT_PATHS,
    EXPECTED_SOURCE_CASE,
    ManifestValidationError,
    load_manifest,
    validate_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "analysis"
    / "recovery_action_branching_nonformal_v0"
    / "manifest.json"
)
VALIDATOR_PATH = (
    REPOSITORY_ROOT / "scripts" / "check_recovery_action_branching_manifest.py"
)


class RecoveryActionBranchingManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest(MANIFEST_PATH)

    def write_mutated_manifest(self, manifest: dict[str, object]) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

    def assert_mutation_fails(self, manifest: dict[str, object]) -> None:
        path = self.write_mutated_manifest(manifest)
        with self.assertRaises(ManifestValidationError):
            validate_manifest(path)

    @staticmethod
    def branch(manifest: dict[str, object], branch_id: str) -> dict[str, object]:
        return next(
            branch
            for branch in manifest["branches"]
            if branch["branch_id"] == branch_id
        )

    def test_valid_manifest_passes(self) -> None:
        passes = validate_manifest(
            MANIFEST_PATH,
            repository_root=REPOSITORY_ROOT,
            require_future_outputs_absent=True,
        )
        self.assertGreaterEqual(len(passes), 1)
        self.assertEqual(
            self.manifest["experiment_id"],
            "recovery_action_branching_nonformal_v0",
        )

    def test_validator_cli_returns_success(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(VALIDATOR_PATH), "--manifest", str(MANIFEST_PATH)],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn(
            "Recovery Action Branching manifest validation PASSED",
            completed.stdout,
        )

    def test_exact_source_case_is_frozen(self) -> None:
        source_case = self.manifest["source_case"]
        for key, value in EXPECTED_SOURCE_CASE.items():
            self.assertEqual(source_case[key], value)

    def test_wrong_source_case_fails(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["source_case"]["initial_velocity_angle_deg"] = 165
        self.assert_mutation_fails(mutated)

    def test_missing_branch_fails(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["branches"] = mutated["branches"][:-1]
        self.assert_mutation_fails(mutated)

    def test_duplicate_branch_fails(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["branches"].append(copy.deepcopy(mutated["branches"][0]))
        self.assert_mutation_fails(mutated)

    def test_action_magnitude_drift_fails(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        self.branch(mutated, "velocity_opposed_thrust_v0")[
            "action_magnitude_before_clipping"
        ] = 0.26
        self.assert_mutation_fails(mutated)

    def test_threshold_drift_fails(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["hazard"]["threshold"] = 1.91
        self.assert_mutation_fails(mutated)

    def test_comparator_drift_fails(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["hazard"]["comparator"] = ">="
        self.assert_mutation_fails(mutated)

    def test_recovery_horizon_drift_fails(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["horizons"]["recovery_horizon_realized_transitions"] = 9999
        self.assert_mutation_fails(mutated)

    def test_total_horizon_drift_fails(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["horizons"]["total_episode_horizon_realized_transitions"] = 99999
        self.assert_mutation_fails(mutated)

    def test_missing_coordinate_convention_fails(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        del mutated["coordinate_convention"]["radial_unit_vector"]
        self.assert_mutation_fails(mutated)

    def test_tangential_sign_ambiguity_fails(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["coordinate_convention"]["positive_tangential_unit_vector"] = (
            "e_t = (e_r_y, -e_r_x)"
        )
        self.assert_mutation_fails(mutated)

    def test_recursive_fallback_is_prohibited(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["recovery_action_rejection"]["recursive_branch_selection"] = True
        self.assert_mutation_fails(mutated)

    def test_action_rejection_policy_must_be_complete(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["recovery_action_rejection"]["substitute_zero_action"] = True
        mutated["recovery_action_rejection"]["preserve_evidence_fields"].remove(
            "monitor_reason"
        )
        self.assert_mutation_fails(mutated)

    def test_branch_state_hash_contract_must_be_complete(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["branch_point"]["canonical_hash_contract"][
            "all_branches_require_byte_equivalent_canonical_input"
        ] = False
        self.assert_mutation_fails(mutated)

    def test_measured_outcome_fields_are_prohibited(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["measured_results"] = {"recovery_success_count": 1}
        self.assert_mutation_fails(mutated)

    def test_branch_winner_fields_are_prohibited(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["branch_winner"] = "velocity_opposed_thrust_v0"
        self.assert_mutation_fails(mutated)

    def test_formal_status_is_prohibited(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["is_formal_experiment"] = True
        self.assert_mutation_fails(mutated)

    def test_protected_phase_path_overlap_fails(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["output_contract"]["future_artifacts"][0]["path"] = (
            "analysis/phase34_post_cross_sync/branch_state.json"
        )
        self.assert_mutation_fails(mutated)

    def test_frozen_final_veto_directory_overlap_fails(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["output_contract"]["future_artifacts"][0]["path"] = (
            "analysis/final_veto_ablation_v0/branch_state.json"
        )
        self.assert_mutation_fails(mutated)

    def test_exact_four_branch_ids_are_unique(self) -> None:
        branch_ids = [branch["branch_id"] for branch in self.manifest["branches"]]
        self.assertEqual(len(branch_ids), 4)
        self.assertEqual(len(set(branch_ids)), 4)
        self.assertEqual(set(branch_ids), EXPECTED_BRANCH_IDS)

    def test_future_output_paths_remain_uncreated(self) -> None:
        output_paths = {
            item["path"]
            for item in self.manifest["output_contract"]["future_artifacts"]
        }
        self.assertEqual(output_paths, EXPECTED_OUTPUT_PATHS)
        for relative_path in output_paths:
            self.assertFalse(
                (REPOSITORY_ROOT / relative_path).exists(),
                f"future design-freeze artifact unexpectedly exists: {relative_path}",
            )


if __name__ == "__main__":
    unittest.main()
