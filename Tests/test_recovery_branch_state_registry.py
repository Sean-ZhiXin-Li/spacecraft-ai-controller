from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

from runtime_assurance.recovery_branch_executor import (
    execute_recovery_branch,
    execute_registered_recovery_branch,
    load_frozen_branch_state,
)
from runtime_assurance.recovery_branch_state_extractor import (
    BranchStateExtractionError,
    publish_registry_payloads,
    validate_registry_payloads,
)
from runtime_assurance.recovery_branch_state_registry import (
    BRANCH_STEP,
    LEGACY_ARTIFACT_PATH,
    LEGACY_CASE_ID,
    LEGACY_MEMBER_ID,
    OUTPUT_PATH,
    PREFIX_TRANSITION_COUNT,
    REGISTRY_ID,
    REGISTRY_MEMBER_SCHEMA_VERSION,
    REGISTRY_SCHEMA_VERSION,
    BranchStateRegistryError,
    RegisteredBranchState,
    RegistryMember,
    attach_generated_hashes,
    canonical_json_bytes,
    canonical_json_file_bytes,
    canonical_sha256,
    load_branch_state_registry,
    load_registered_branch_state,
    manifest_scientific_payload,
    registry_aggregate_hash,
    validate_generated_branch_state_document,
    validate_registry_manifest_document,
)


ROOT = Path(__file__).resolve().parents[1]
LEGACY_BYTES = (ROOT / LEGACY_ARTIFACT_PATH).read_bytes()
LEGACY_DOCUMENT = json.loads(LEGACY_BYTES.decode("utf-8"))


def generated_document(case_id: str = "synthetic_case", member_id: str = "member__synthetic_case") -> dict[str, object]:
    target_radius = 10.0
    mu = 1000.0
    target_speed = math.sqrt(mu / target_radius)
    state = {"position_x": 10.0, "position_y": 0.0, "velocity_x": 0.0, "velocity_y": target_speed}
    simulator_constants = {
        "action_component_max": 1.0,
        "action_component_min": -1.0,
        "dt": 1.0,
        "gravity_denominator_epsilon": 1.0e-12,
        "integration_order": "velocity_then_position_using_updated_velocity",
        "mass": 1.0,
        "max_steps": 100,
        "mu": mu,
        "rollout_overspeed_comparator": ">",
        "rollout_overspeed_threshold": 1.9,
        "speed_ratio_denominator_epsilon": 1.0e-12,
        "target_circular_speed": target_speed,
        "target_radius": target_radius,
        "target_radius_scale": 1.0,
        "transition_function": "simulator.phase34_35_transition.step_phase34_35_transition",
    }
    simulator = {"simulator_constants": simulator_constants, "thrust_scale": 1.0}
    case_configuration = {
        "case_id": case_id,
        "controller_id": "phase34_post_cross_sync",
        "initial_velocity_angle_deg": 0.0,
        "post_cross_mode": "radius_priority",
        "r0_over_target": 1.0,
        "seed": 0,
        "subset_id": "synthetic",
        "thrust_scale": 1.0,
        "upstream_variant": None,
    }
    predicted_speed_ratio = target_speed / (target_speed + 1.0e-12)
    document: dict[str, object] = {
        "schema_version": REGISTRY_MEMBER_SCHEMA_VERSION,
        "registry_member_id": member_id,
        "case_id": case_id,
        "seed": 0,
        "state_origin": "deterministic_nominal_prefix_execution",
        "reconstructed_from_log": False,
        "manually_authored_state": False,
        "perturbed_from_existing_state": False,
        "source_commit": "a" * 40,
        "generation_implementation_commit": "b" * 40,
        "generated_date": "2026-08-03",
        "nominal_prefix_transition_count": PREFIX_TRANSITION_COUNT,
        "actual_transition_count": BRANCH_STEP,
        "branch_step": BRANCH_STEP,
        "boundary_type": "monitor_off_preterminal_state",
        "boundary_status": "frozen",
        "boundary_transition_count": PREFIX_TRANSITION_COUNT,
        "preterminal_state_transition_count": PREFIX_TRANSITION_COUNT,
        "boundary_state_semantics": "last_complete_valid_pre_transition_state_before_the_frozen_terminal_transition",
        "boundary_indexing_semantics": "pre_transition_step_N_state_equals_state_after_N_minus_1_realized_transitions;transition_N_executes_next;terminal_predicates_evaluate_on_transition_N_next_state",
        "boundary_registry_hash": "8" * 64,
        "boundary_source_artifact": "analysis/final_veto_ablation_v0/results.csv",
        "boundary_source_artifact_hash": "9" * 64,
        "boundary_source_json_path": "csv_rows[synthetic]",
        "terminal_transition_count": BRANCH_STEP,
        "terminal_reason": "success",
        "terminal_evidence_source": "analysis/final_veto_ablation_v0/results.csv",
        "terminal_evidence_hash": "a" * 64,
        "terminal_evidence_record_hash": "b" * 64,
        "terminal_evidence_path": "csv_rows[synthetic]",
        "terminal_transition_executed_for_validation": True,
        "terminal_validation_status": "passed",
        "terminal_realized_state_hash": "c" * 64,
        "initial_state_hash": "1" * 64,
        "prefix_action_count": PREFIX_TRANSITION_COUNT,
        "prefix_action_trace_hash": "2" * 64,
        "prefix_state_trace_hash": "3" * 64,
        "terminal_before_branch": False,
        "terminal_reason_before_branch": None,
        **state,
        "state": {"current_phase": "DESCENT", **state},
        "state_vector": list(state.values()),
        "phase": "DESCENT",
        "active_stage": "burn_a",
        "proposed_action": [0.0, 0.0],
        "nominal_action": [0.0, 0.0],
        "predicted_position_x": state["position_x"],
        "predicted_position_y": state["position_y"],
        "predicted_velocity_x": state["velocity_x"],
        "predicted_velocity_y": state["velocity_y"],
        "predicted_next_state": state,
        "predicted_speed": target_speed,
        "predicted_speed_ratio": predicted_speed_ratio,
        "monitor_decision": {
            "decision": "allow",
            "monitor_id": "one_step_overspeed_veto_v0",
            "reason": "predicted_nominal_speed_within_limit",
            "veto_applied": False,
        },
        "threshold": 1.9,
        "comparator": ">",
        "case_configuration": case_configuration,
        "case_configuration_hash": canonical_sha256(case_configuration),
        "source_case_artifact": "analysis/final_veto_ablation_v0/manifest.json",
        "source_case_hash": "4" * 64,
        "source_configuration_hash": "5" * 64,
        "simulator_configuration": simulator,
        "simulator_configuration_hash": canonical_sha256(simulator),
        "constants_hash": canonical_sha256(simulator_constants),
        "transition_implementation_hash": "6" * 64,
        "nominal_controller_hash": "7" * 64,
        "branch_ordering": {
            "capture_boundary": "after_27_realized_nominal_transitions_before_step_28_action_execution",
            "before_nominal_action_execution": True,
            "monitor_evaluation_completed": True,
            "nominal_action_executed": False,
            "realized_prefix_transition_count": PREFIX_TRANSITION_COUNT,
        },
        "radius": target_radius,
        "speed": target_speed,
        "radial_velocity": 0.0,
        "tangential_velocity": target_speed,
        "target_radius": target_radius,
        "target_circular_speed": target_speed,
        "signed_radius_error": 0.0,
        "absolute_radius_gap": 0.0,
        "radius_error_ratio": 0.0,
        "radial_velocity_ratio": 0.0,
        "tangential_velocity_error": 0.0,
        "tangential_velocity_error_ratio": 0.0,
        "realized_speed_ratio": predicted_speed_ratio,
        "overspeed_headroom": 1.9 - predicted_speed_ratio,
        "radius_component_pass": True,
        "radial_velocity_component_pass": True,
        "tangential_velocity_component_pass": True,
        "phase34_compatible_recoverability_pass": True,
    }
    return attach_generated_hashes(document)


def registry_member(
    document: dict[str, object],
    *,
    index: int,
    artifact_filename: str | None = None,
) -> RegistryMember:
    raw = canonical_json_file_bytes(document)
    filename = artifact_filename or f"case_{index}.json"
    return RegistryMember(
        registry_member_id=str(document["registry_member_id"]),
        case_id=str(document["case_id"]),
        seed=0,
        artifact_path=(
            "analysis/recovery_branch_state_registry_v0/branch_states/"
            f"{filename}"
        ),
        artifact_scope="registry_local_artifact",
        state_origin="deterministic_nominal_prefix_execution",
        source_case_artifact="analysis/final_veto_ablation_v0/manifest.json",
        source_case_hash=str(document["source_case_hash"]),
        source_configuration_hash=str(document["source_configuration_hash"]),
        simulator_configuration_hash=str(document["simulator_configuration_hash"]),
        constants_hash=str(document["constants_hash"]),
        transition_implementation_hash=str(document["transition_implementation_hash"]),
        nominal_controller_hash=str(document["nominal_controller_hash"]),
        source_commit=str(document["source_commit"]),
        nominal_prefix_transition_count=PREFIX_TRANSITION_COUNT,
        branch_step=BRANCH_STEP,
        canonical_branch_state_hash=str(document["canonical_payload_hash"]),
        raw_artifact_hash=hashlib.sha256(raw).hexdigest(),
        legacy_member=False,
        generation_status="generated_and_validated",
        determinism_status="passed",
        executable_status="validated",
    )


def legacy_member() -> RegistryMember:
    return RegistryMember(
        registry_member_id=LEGACY_MEMBER_ID,
        case_id=LEGACY_CASE_ID,
        seed=0,
        artifact_path=LEGACY_ARTIFACT_PATH.as_posix(),
        artifact_scope="legacy_external_artifact",
        state_origin="deterministic_nominal_prefix_execution",
        source_case_artifact="analysis/final_veto_ablation_v0/manifest.json",
        source_case_hash="8" * 64,
        source_configuration_hash=str(LEGACY_DOCUMENT["case_configuration_hash"]),
        simulator_configuration_hash=str(LEGACY_DOCUMENT["simulator_configuration_hash"]),
        constants_hash=str(LEGACY_DOCUMENT["simulator_constants_hash"]),
        transition_implementation_hash="a" * 64,
        nominal_controller_hash="b" * 64,
        source_commit=str(LEGACY_DOCUMENT["source_commit"]),
        nominal_prefix_transition_count=PREFIX_TRANSITION_COUNT,
        branch_step=BRANCH_STEP,
        canonical_branch_state_hash=str(LEGACY_DOCUMENT["canonical_branch_state_hash"]),
        raw_artifact_hash=hashlib.sha256(LEGACY_BYTES).hexdigest(),
        legacy_member=True,
        generation_status="legacy_validated",
        determinism_status="passed",
        executable_status="validated",
    )


def manifest_document(members: list[RegistryMember]) -> dict[str, object]:
    document: dict[str, object] = {
        "registry_id": REGISTRY_ID,
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "member_index": [member.as_document() for member in sorted(members, key=lambda item: item.registry_member_id)],
        "Stage_1B_prerequisite_status": "satisfied",
    }
    document["canonical_manifest_hash"] = canonical_sha256(manifest_scientific_payload(document))
    return document


def hashed_document(document: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(document)
    result["canonical_payload_hash"] = canonical_sha256(result)
    return result


def valid_in_memory_payloads() -> tuple[dict[str, bytes], list[RegistryMember]]:
    regression_filename = (
        "phase34_known_recoverable_preservation_v1__r0_1p00__angle_175__thrust_8000.json"
    )
    documents = [generated_document(f"case_{i}", f"member_{i}") for i in range(3)]
    generated_members = [
        registry_member(
            documents[0],
            index=0,
            artifact_filename=regression_filename,
        ),
        registry_member(documents[1], index=1),
        registry_member(documents[2], index=2),
    ]
    members = [legacy_member(), *generated_members]
    manifest = manifest_document(members)
    manifest["registry_aggregate_hash"] = registry_aggregate_hash(members)
    manifest["canonical_manifest_hash"] = canonical_sha256(
        manifest_scientific_payload(manifest)
    )
    payloads = {
        "registry_manifest.json": canonical_json_file_bytes(manifest),
        "source_case_inventory.json": canonical_json_file_bytes(
            hashed_document({"source_case_count": 13})
        ),
        "selection_report.json": canonical_json_file_bytes(
            hashed_document({"selection_rules_frozen_before_execution": True})
        ),
        "determinism_report.json": canonical_json_file_bytes(
            hashed_document({"determinism_failure_count": 0})
        ),
        "prefix_execution_report.json": canonical_json_file_bytes(
            hashed_document(
                {
                    "automatic_retry_count": 0,
                    "recovery_branch_execution_count": 0,
                    "total_nominal_prefix_execution_count": 0,
                }
            )
        ),
        "branch_state_index.json": canonical_json_file_bytes(
            hashed_document({"member_count": 4})
        ),
        "summary.md": b"This registry does not demonstrate recovery performance.\n",
    }
    for member, document in zip(generated_members, documents):
        payloads[member.artifact_path.rsplit(
            "analysis/recovery_branch_state_registry_v0/", 1
        )[1]] = canonical_json_file_bytes(document)
    return payloads, members


class RecoveryBranchStateRegistryTests(unittest.TestCase):
    def test_generated_state_schema_and_hashes_validate(self) -> None:
        self.assertGreaterEqual(len(validate_generated_branch_state_document(generated_document())), 4)

    def test_in_memory_payload_validation_accepts_generated_posix_member_paths(self) -> None:
        payloads, members = valid_in_memory_payloads()
        original_paths = tuple(member.artifact_path for member in members)
        manifest_hash, aggregate_hash, member_count = validate_registry_payloads(
            ROOT, payloads
        )
        regression_key = (
            "branch_states/"
            "phase34_known_recoverable_preservation_v1__r0_1p00__angle_175__thrust_8000.json"
        )
        self.assertIn(regression_key, payloads)
        self.assertEqual(member_count, 4)
        self.assertEqual(aggregate_hash, registry_aggregate_hash(members))
        self.assertEqual(len(manifest_hash), 64)
        self.assertEqual(original_paths, tuple(member.artifact_path for member in members))

    def test_in_memory_payload_artifact_set_remains_exact(self) -> None:
        payloads, _ = valid_in_memory_payloads()
        payloads["branch_states/unexpected.json"] = b"{}\n"
        with self.assertRaisesRegex(
            BranchStateExtractionError, "artifact set is not exact"
        ):
            validate_registry_payloads(ROOT, payloads)

    def test_all_cartesian_fields_are_required_and_nonfinite_rejected(self) -> None:
        for field in ("position_x", "position_y", "velocity_x", "velocity_y"):
            with self.subTest(field=field):
                document = generated_document()
                document["state"].pop(field)
                document = attach_generated_hashes(document)
                with self.assertRaises(BranchStateRegistryError):
                    validate_generated_branch_state_document(document)
        document = generated_document()
        document["state"]["velocity_x"] = float("inf")
        with self.assertRaises(BranchStateRegistryError):
            validate_generated_branch_state_document(attach_generated_hashes(document))

    def test_derived_state_and_inclusive_recoverability_recompute(self) -> None:
        document = generated_document()
        for field in (
            "radius",
            "speed",
            "radial_velocity",
            "tangential_velocity",
            "radius_error_ratio",
            "radial_velocity_ratio",
            "tangential_velocity_error_ratio",
            "overspeed_headroom",
        ):
            changed = copy.deepcopy(document)
            changed[field] = float(changed[field]) + 1.0
            with self.assertRaises(BranchStateRegistryError):
                validate_generated_branch_state_document(attach_generated_hashes(changed))
        self.assertIs(document["radius_component_pass"], True)
        self.assertIs(document["radial_velocity_component_pass"], True)
        self.assertIs(document["tangential_velocity_component_pass"], True)

    def test_canonical_hash_excludes_fixed_date_but_mutation_invalidates(self) -> None:
        first = generated_document()
        changed_date = copy.deepcopy(first)
        changed_date["generated_date"] = "2099-01-01"
        changed_date = attach_generated_hashes(changed_date)
        self.assertEqual(first["canonical_payload_hash"], changed_date["canonical_payload_hash"])
        changed_state = copy.deepcopy(first)
        changed_state["position_x"] = 11.0
        changed_state["state"]["position_x"] = 11.0
        changed_state = attach_generated_hashes(changed_state)
        self.assertNotEqual(first["canonical_payload_hash"], changed_state["canonical_payload_hash"])

    def test_registry_rejects_duplicate_member_and_case_ids(self) -> None:
        documents = [generated_document(f"case_{i}", f"member_{i}") for i in range(3)]
        members = [legacy_member(), *(registry_member(doc, index=i) for i, doc in enumerate(documents))]
        validate_registry_manifest_document(manifest_document(members))
        duplicate_id = copy.deepcopy(members[1].as_document())
        duplicate_id["case_id"] = "other_case"
        with self.assertRaises(BranchStateRegistryError):
            validate_registry_manifest_document(
                manifest_document([members[0], members[1], RegistryMember.from_mapping(duplicate_id), members[2]])
            )
        duplicate_case = copy.deepcopy(members[1].as_document())
        duplicate_case["registry_member_id"] = "other_member"
        with self.assertRaises(BranchStateRegistryError):
            validate_registry_manifest_document(
                manifest_document([members[0], members[1], RegistryMember.from_mapping(duplicate_case), members[2]])
            )

    def _write_registry(self, root: Path) -> tuple[list[RegistryMember], list[dict[str, object]]]:
        (root / LEGACY_ARTIFACT_PATH).parent.mkdir(parents=True)
        (root / LEGACY_ARTIFACT_PATH).write_bytes(LEGACY_BYTES)
        documents = [generated_document(f"case_{i}", f"member_{i}") for i in range(3)]
        members = [legacy_member()]
        for i, document in enumerate(documents):
            path = root / OUTPUT_PATH / "branch_states" / f"case_{i}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(canonical_json_file_bytes(document))
            members.append(registry_member(document, index=i))
        (root / OUTPUT_PATH / "registry_manifest.json").write_text(
            json.dumps(manifest_document(members), sort_keys=True), encoding="utf-8"
        )
        return members, documents

    def test_registry_loads_legacy_and_local_members_immutably(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            members, _ = self._write_registry(root)
            registry = load_branch_state_registry(root)
            self.assertEqual(len(registry.members), 4)
            legacy = load_registered_branch_state(root, LEGACY_MEMBER_ID)
            local = load_registered_branch_state(root, members[1].registry_member_id)
            self.assertEqual(legacy.case_id, LEGACY_CASE_ID)
            self.assertEqual(local.case_id, members[1].case_id)
            with self.assertRaises(FrozenInstanceError):
                local._document_json = "{}"

    def test_unknown_member_path_traversal_and_substitution_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            members, _ = self._write_registry(root)
            with self.assertRaises(BranchStateRegistryError):
                load_registered_branch_state(root, "unknown")
            manifest = manifest_document(members)
            manifest["member_index"][1]["artifact_path"] = "../outside.json"
            manifest["canonical_manifest_hash"] = canonical_sha256(manifest_scientific_payload(manifest))
            (root / OUTPUT_PATH / "registry_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(BranchStateRegistryError):
                load_registered_branch_state(root, members[1].registry_member_id)

    def test_raw_and_canonical_hash_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            members, documents = self._write_registry(root)
            path = root / OUTPUT_PATH / "branch_states" / "case_0.json"
            path.write_bytes(canonical_json_file_bytes({**documents[0], "extra": True}))
            with self.assertRaises(BranchStateRegistryError):
                load_registered_branch_state(root, members[1].registry_member_id)

    def test_member_and_artifact_provenance_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            members, _ = self._write_registry(root)
            manifest = manifest_document(members)
            manifest["member_index"][1]["source_case_hash"] = "f" * 64
            manifest["canonical_manifest_hash"] = canonical_sha256(
                manifest_scientific_payload(manifest)
            )
            (root / OUTPUT_PATH / "registry_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                BranchStateRegistryError, "source_case_hash.*registry provenance"
            ):
                load_registered_branch_state(root, members[1].registry_member_id)

    def test_member_order_and_registry_aggregate_are_deterministic(self) -> None:
        documents = [generated_document(f"case_{i}", f"member_{i}") for i in range(3)]
        members = [legacy_member(), *(registry_member(doc, index=i) for i, doc in enumerate(documents))]
        self.assertEqual(registry_aggregate_hash(members), registry_aggregate_hash(tuple(reversed(members))))
        manifest = manifest_document(list(reversed(members)))
        parsed = validate_registry_manifest_document(manifest)
        self.assertEqual(tuple(item.registry_member_id for item in parsed), tuple(sorted(item.registry_member_id for item in members)))

    def test_legacy_validator_and_executor_behavior_remain_unchanged(self) -> None:
        state = load_frozen_branch_state()
        before = execute_recovery_branch(state, "explicit_abort_v0")
        self.assertEqual(before.terminal_reason, "explicit_recovery_abort")
        self.assertEqual(before.transition_count, 0)
        self.assertEqual((ROOT / LEGACY_ARTIFACT_PATH).read_bytes(), LEGACY_BYTES)

    def test_registered_executor_uses_member_loader_and_preserves_action_core(self) -> None:
        document = generated_document()
        member = registry_member(document, index=0)
        registered = RegisteredBranchState(member=member, _document_json=canonical_json_bytes(document).decode("utf-8"))
        with mock.patch(
            "runtime_assurance.recovery_branch_state_registry.load_registered_branch_state",
            return_value=registered,
        ):
            result = execute_registered_recovery_branch(member.registry_member_id, "explicit_abort_v0")
        self.assertEqual(result.transition_count, 0)
        self.assertEqual(result.terminal_reason, "explicit_recovery_abort")

    def test_atomic_publication_refuses_existing_target_and_cleans_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis").mkdir()
            payloads = {"prefix_execution_report.json": b'{"total_nominal_prefix_execution_count":0}\n'}

            def validator(_root: Path, _payloads: object) -> tuple[str, str, int]:
                return "a" * 64, "b" * 64, 0

            publication = publish_registry_payloads(root, payloads, validator=validator)
            self.assertTrue((Path(publication.target_directory) / "prefix_execution_report.json").is_file())
            with self.assertRaises(BranchStateExtractionError):
                publish_registry_payloads(root, payloads, validator=validator)

    def test_atomic_publication_writer_failure_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis").mkdir()
            payloads = {"prefix_execution_report.json": b'{"total_nominal_prefix_execution_count":0}\n'}

            def validator(_root: Path, _payloads: object) -> tuple[str, str, int]:
                return "a" * 64, "b" * 64, 0

            def broken_writer(_path: Path, _payload: bytes) -> None:
                raise OSError("injected failure")

            with self.assertRaises(OSError):
                publish_registry_payloads(root, payloads, validator=validator, writer=broken_writer)
            self.assertFalse((root / OUTPUT_PATH).exists())
            self.assertEqual(list((root / "analysis").glob(".*staging-*")), [])


if __name__ == "__main__":
    unittest.main()
