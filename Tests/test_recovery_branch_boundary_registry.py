from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from runtime_assurance.recovery_branch_boundary_registry import (
    BOUNDARY_CONFIG_PATH,
    BOUNDARY_REGISTRY_ID,
    INELIGIBLE,
    LEGACY_FIXED_PREFIX,
    MONITOR_OFF_PRETERMINAL_STATE,
    PRETERMINAL_INDEXING_SEMANTICS,
    PRETERMINAL_STATE_SEMANTICS,
    SOURCE_DECLARED_FIXED_PREFIX,
    RecoveryBranchBoundary,
    RecoveryBranchBoundaryError,
    boundary_registry_payload,
    load_recovery_branch_boundary_registry,
)
from runtime_assurance.recovery_branch_state_registry import (
    LEGACY_CASE_ID,
    canonical_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
TRANSITION_22_CASE = (
    "phase35_radial_energy_push_overspeed_stress_v0"
    "__r0_0p98__angle_150__thrust_10000"
)


def boundary_document(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "case_id": "synthetic_case",
        "seed": 0,
        "boundary_type": MONITOR_OFF_PRETERMINAL_STATE,
        "boundary_status": "frozen",
        "boundary_transition_count": 21,
        "terminal_transition_count": 22,
        "preterminal_state_transition_count": 21,
        "terminal_reason": "overspeed",
        "source_artifact": "analysis/final_veto_ablation_v0/results.csv",
        "source_artifact_hash": "1" * 64,
        "source_json_path": "csv_rows[synthetic]",
        "source_configuration_hash": "2" * 64,
        "simulator_configuration_hash": "3" * 64,
        "controller_configuration_hash": "4" * 64,
        "indexing_semantics": PRETERMINAL_INDEXING_SEMANTICS,
        "boundary_state_semantics": PRETERMINAL_STATE_SEMANTICS,
        "terminal_evidence_source": "analysis/final_veto_ablation_v0/results.csv",
        "terminal_evidence_hash": "5" * 64,
        "terminal_evidence_record_hash": "6" * 64,
        "terminal_evidence_path": "csv_rows[synthetic]",
        "eligibility": True,
        "ineligibility_reason": None,
    }
    value.update(overrides)
    return value


class RecoveryBranchBoundaryRegistryTests(unittest.TestCase):
    def test_checked_in_registry_has_all_thirteen_source_backed_cases(self) -> None:
        registry = load_recovery_branch_boundary_registry(ROOT)
        self.assertEqual(registry.registry_id, BOUNDARY_REGISTRY_ID)
        self.assertEqual(len(registry.boundaries), 13)
        self.assertEqual(len({item.case_id for item in registry.boundaries}), 13)
        self.assertTrue(all(item.eligibility for item in registry.boundaries))
        self.assertEqual(
            sum(item.boundary_type == LEGACY_FIXED_PREFIX for item in registry.boundaries),
            1,
        )

    def test_transition_22_mapping_is_pretransition_step_22_after_21_transitions(self) -> None:
        boundary = load_recovery_branch_boundary_registry(ROOT).for_case(
            TRANSITION_22_CASE
        )
        self.assertEqual(boundary.boundary_type, MONITOR_OFF_PRETERMINAL_STATE)
        self.assertEqual(boundary.terminal_transition_count, 22)
        self.assertEqual(boundary.preterminal_state_transition_count, 21)
        self.assertEqual(boundary.boundary_transition_count, 21)
        self.assertEqual(boundary.branch_step, 22)
        self.assertEqual(boundary.terminal_reason, "overspeed")

    def test_off_by_one_terminal_substitutions_are_rejected(self) -> None:
        for field, value in (
            ("boundary_transition_count", 22),
            ("preterminal_state_transition_count", 20),
            ("terminal_transition_count", 21),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    RecoveryBranchBoundaryError,
                    "off-by-one|preterminal state count",
                ):
                    RecoveryBranchBoundary.from_mapping(
                        boundary_document(**{field: value})
                    )

    def test_boundary_definition_is_immutable(self) -> None:
        boundary = RecoveryBranchBoundary.from_mapping(boundary_document())
        with self.assertRaises(FrozenInstanceError):
            boundary.boundary_transition_count = 99

    def test_missing_terminal_evidence_is_rejected(self) -> None:
        with self.assertRaises(RecoveryBranchBoundaryError):
            RecoveryBranchBoundary.from_mapping(
                boundary_document(terminal_transition_count=None)
            )
        with self.assertRaises(RecoveryBranchBoundaryError):
            RecoveryBranchBoundary.from_mapping(boundary_document(terminal_reason=None))

    def test_source_declared_fixed_prefix_is_explicit_not_defaulted(self) -> None:
        boundary = RecoveryBranchBoundary.from_mapping(
            boundary_document(
                boundary_type=SOURCE_DECLARED_FIXED_PREFIX,
                boundary_transition_count=12,
                preterminal_state_transition_count=12,
                terminal_transition_count=13,
            )
        )
        self.assertEqual(boundary.boundary_transition_count, 12)
        config = json.loads((ROOT / BOUNDARY_CONFIG_PATH).read_text(encoding="utf-8"))
        self.assertNotIn("default_boundary_transition_count", config)

    def test_ineligible_boundary_requires_reason_and_cannot_be_eligible(self) -> None:
        boundary = RecoveryBranchBoundary.from_mapping(
            boundary_document(
                boundary_type=INELIGIBLE,
                boundary_status="ineligible",
                boundary_transition_count=None,
                terminal_transition_count=None,
                preterminal_state_transition_count=None,
                terminal_reason=None,
                eligibility=False,
                ineligibility_reason="terminal indexing unavailable",
            )
        )
        self.assertFalse(boundary.eligibility)
        with self.assertRaises(RecoveryBranchBoundaryError):
            RecoveryBranchBoundary.from_mapping(
                boundary_document(
                    boundary_type=INELIGIBLE,
                    boundary_status="ineligible",
                    eligibility=True,
                    ineligibility_reason=None,
                )
            )

    def test_phase34_boundaries_are_frozen_pre_success_states(self) -> None:
        boundaries = load_recovery_branch_boundary_registry(ROOT).boundaries
        phase34 = [item for item in boundaries if item.case_id.startswith("phase34_")]
        self.assertEqual(len(phase34), 8)
        self.assertTrue(
            all(item.boundary_type == MONITOR_OFF_PRETERMINAL_STATE for item in phase34)
        )
        self.assertTrue(all(item.terminal_reason == "success" for item in phase34))
        self.assertTrue(
            all(
                item.terminal_transition_count == item.boundary_transition_count + 1
                for item in phase34
            )
        )

    def test_phase35_boundaries_preserve_legacy_and_pre_overspeed_cases(self) -> None:
        registry = load_recovery_branch_boundary_registry(ROOT)
        phase35 = [item for item in registry.boundaries if item.case_id.startswith("phase35_")]
        self.assertEqual(len(phase35), 5)
        self.assertEqual(registry.for_case(LEGACY_CASE_ID).boundary_type, LEGACY_FIXED_PREFIX)
        generated = [item for item in phase35 if item.case_id != LEGACY_CASE_ID]
        self.assertTrue(all(item.terminal_reason == "overspeed" for item in generated))
        self.assertEqual(
            sorted(item.boundary_transition_count for item in generated),
            [21, 24, 25, 26],
        )

    def test_registry_hash_recomputes_and_payload_mutation_invalidates(self) -> None:
        document = json.loads((ROOT / BOUNDARY_CONFIG_PATH).read_text(encoding="utf-8"))
        supplied = document["canonical_payload_hash"]
        self.assertEqual(supplied, canonical_sha256(boundary_registry_payload(document)))
        changed = copy.deepcopy(document)
        changed["boundaries"][0]["terminal_reason"] = "overspeed"
        self.assertNotEqual(supplied, canonical_sha256(boundary_registry_payload(changed)))

    def test_source_artifact_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_source = ROOT / BOUNDARY_CONFIG_PATH
            config_target = root / BOUNDARY_CONFIG_PATH
            config_target.parent.mkdir(parents=True)
            config_target.write_bytes(config_source.read_bytes())
            results_target = root / "analysis/final_veto_ablation_v0/results.csv"
            results_target.parent.mkdir(parents=True)
            results_target.write_bytes(
                (ROOT / "analysis/final_veto_ablation_v0/results.csv").read_bytes()
                + b"\n"
            )
            legacy_target = root / "analysis/recovery_action_branching_nonformal_v0/branch_state.json"
            legacy_target.parent.mkdir(parents=True)
            legacy_target.write_bytes(
                (ROOT / "analysis/recovery_action_branching_nonformal_v0/branch_state.json").read_bytes()
            )
            with self.assertRaisesRegex(
                RecoveryBranchBoundaryError, "source artifact hash mismatch"
            ):
                load_recovery_branch_boundary_registry(root)


if __name__ == "__main__":
    unittest.main()
