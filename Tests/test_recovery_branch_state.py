from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_recovery_branch_state import (
    CANONICALIZATION_ID,
    SCHEMA_VERSION,
    SOURCE_CASE_ID,
    SOURCE_COMMIT,
    SOURCE_SUBSET_ID,
    BranchStateValidationError,
    attach_canonical_branch_state_hash,
    canonical_sha256,
    validate_branch_state,
    validate_branch_state_data,
    write_canonical_branch_state,
)


def _valid_document() -> dict[str, object]:
    case_configuration = {
        "case_id": SOURCE_CASE_ID,
        "controller_id": "phase35_crossing_basin_expansion",
        "initial_velocity_angle_deg": 150.0,
        "post_cross_mode": "radius_priority",
        "r0_over_target": 0.98,
        "seed": 0,
        "subset_id": SOURCE_SUBSET_ID,
        "thrust_scale": 8000.0,
        "upstream_variant": "radial_energy_push",
    }
    simulator_constants = {
        "dt": 100.0,
        "mass": 722.0,
        "mu": 1.0,
        "target_circular_speed": 2.0,
    }
    simulator_configuration = {
        "simulator_constants": simulator_constants,
        "thrust_scale": 8000.0,
    }
    document: dict[str, object] = {
        "active_stage": "radial_energy_push",
        "branch_ordering": {
            "before_final_veto_fallback_execution": True,
            "before_nominal_action_execution": True,
            "capture_boundary": (
                "after_valid_monitor_evaluation_before_nominal_or_fallback_execution"
            ),
            "final_veto_fallback_executed": False,
            "monitor_evaluation_completed": True,
            "nominal_action_executed": False,
            "prior_monitor_decisions": "all_allow",
            "prior_valid_monitor_evaluation_count": 9,
            "prior_veto_count": 0,
            "realized_prefix_transition_count": 9,
        },
        "branch_step": 10,
        "canonicalization": {
            "allow_nan": False,
            "canonicalization_id": CANONICALIZATION_ID,
            "encoding": "utf-8",
            "hash_algorithm": "sha256",
            "hash_field_excluded_from_input": "canonical_branch_state_hash",
            "json_separators": [",", ":"],
            "json_sort_keys": True,
        },
        "case_configuration": case_configuration,
        "case_configuration_hash": canonical_sha256(case_configuration),
        "case_id": SOURCE_CASE_ID,
        "comparator": ">",
        "extraction_timestamp": "2026-07-19T00:00:00+08:00",
        "extraction_timestamp_policy": (
            "frozen_milestone_timestamp_for_reproducible_hashing"
        ),
        "hazard_comparator": ">",
        "hazard_threshold": 1.90,
        "implementation_commit": SOURCE_COMMIT,
        "monitor_decision": {
            "decision": "veto",
            "monitor_id": "one_step_overspeed_veto_v0",
            "reason": "predicted_nominal_overspeed",
            "veto_applied": True,
        },
        "nominal_action": [0.1, -0.2],
        "nominal_proposed_action": [0.1, -0.2],
        "phase": "DESCENT",
        "position": [1.0, 2.0],
        "predicted_next_state": {
            "position_x": 1.1,
            "position_y": 2.1,
            "velocity_x": 3.1,
            "velocity_y": 4.1,
        },
        "predicted_nominal_next_state": [1.1, 2.1, 3.1, 4.1],
        "predicted_nominal_speed_ratio": 1.9000000001,
        "predicted_speed_ratio": 1.9000000001,
        "schema_version": SCHEMA_VERSION,
        "seed": 0,
        "simulator_configuration": simulator_configuration,
        "simulator_configuration_hash": canonical_sha256(simulator_configuration),
        "simulator_constants_hash": canonical_sha256(simulator_constants),
        "source_commit": SOURCE_COMMIT,
        "state": {
            "current_phase": "DESCENT",
            "position_x": 1.0,
            "position_y": 2.0,
            "velocity_x": 3.0,
            "velocity_y": 4.0,
        },
        "state_vector": [1.0, 2.0, 3.0, 4.0],
        "step": 10,
        "subset_id": SOURCE_SUBSET_ID,
        "threshold": 1.90,
        "velocity": [3.0, 4.0],
    }
    return attach_canonical_branch_state_hash(document)


def _rehash(document: dict[str, object]) -> dict[str, object]:
    return attach_canonical_branch_state_hash(document)


class RecoveryBranchStateTests(unittest.TestCase):
    def assertInvalid(self, document: dict[str, object]) -> None:
        with self.assertRaises(BranchStateValidationError):
            validate_branch_state_data(document)

    def test_valid_branch_state_passes(self) -> None:
        document = _valid_document()
        self.assertGreaterEqual(len(validate_branch_state_data(document)), 10)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "branch_state.json"
            write_canonical_branch_state(path, document)
            self.assertGreaterEqual(len(validate_branch_state(path)), 11)

    def test_wrong_case_fails(self) -> None:
        document = _valid_document()
        document["case_id"] = "wrong_case"
        self.assertInvalid(_rehash(document))

    def test_wrong_threshold_fails(self) -> None:
        document = _valid_document()
        document["threshold"] = 1.91
        document["hazard_threshold"] = 1.91
        self.assertInvalid(_rehash(document))

    def test_wrong_comparator_fails(self) -> None:
        document = _valid_document()
        document["comparator"] = ">="
        document["hazard_comparator"] = ">="
        self.assertInvalid(_rehash(document))

    def test_missing_hash_fails(self) -> None:
        document = _valid_document()
        del document["canonical_branch_state_hash"]
        self.assertInvalid(document)

    def test_missing_state_field_fails(self) -> None:
        document = _valid_document()
        del document["state"]["velocity_y"]
        self.assertInvalid(_rehash(document))

    def test_recovery_result_field_fails(self) -> None:
        document = _valid_document()
        document["recovery_success"] = False
        self.assertInvalid(_rehash(document))

    def test_branch_winner_field_fails(self) -> None:
        document = _valid_document()
        document["winner"] = "zero_action_reference_v0"
        self.assertInvalid(_rehash(document))

    def test_experiment_artifact_field_fails(self) -> None:
        document = _valid_document()
        document["results_csv"] = "results.csv"
        self.assertInvalid(_rehash(document))

    def test_non_deterministic_serialization_fails(self) -> None:
        document = _valid_document()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "branch_state.json"
            path.write_text(json.dumps(document, indent=2), encoding="utf-8")
            with self.assertRaises(BranchStateValidationError):
                validate_branch_state(path)

    def test_invalid_branch_ordering_fails(self) -> None:
        document = _valid_document()
        document["branch_ordering"]["before_nominal_action_execution"] = False
        document["branch_ordering"]["nominal_action_executed"] = True
        self.assertInvalid(_rehash(document))

    def test_branch_state_after_fallback_fails(self) -> None:
        document = _valid_document()
        document["branch_ordering"][
            "before_final_veto_fallback_execution"
        ] = False
        document["branch_ordering"]["final_veto_fallback_executed"] = True
        self.assertInvalid(_rehash(document))


if __name__ == "__main__":
    unittest.main()
