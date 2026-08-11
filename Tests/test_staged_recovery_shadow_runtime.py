from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from runtime_assurance.final_veto_monitor import FinalVetoDecision
from runtime_assurance.recovery_branch_executor import RecoveryBranchExecutionResult
from runtime_assurance.recovery_branch_state_registry import load_registered_branch_state
from runtime_assurance.staged_recovery_logger_adapter import ObserverIntegrationError, runtime_state_hash
from runtime_assurance.staged_recovery_shadow_runtime import (
    OUTPUT_PATH,
    PHYSICAL_BRANCH_ID,
    ShadowObservationAdapter,
    build_registered_runtime_identity,
    compare_physical_runs,
    publish_smoke_payloads,
    run_registered_bounded_shadow_path,
    validate_registry_source,
    validate_smoke_payloads,
)


ROOT = Path(__file__).resolve().parents[1]


class ShadowRuntimeBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.member_id = validate_registry_source(ROOT)[0]
        cls.registered = load_registered_branch_state(ROOT, cls.member_id)

    def fake_executor(self, member_id, branch_id, horizon_steps=1, *, current_state=None):
        self.assertEqual(member_id, self.member_id)
        self.assertEqual(branch_id, PHYSICAL_BRANCH_ID)
        self.assertIsNotNone(current_state)
        state = current_state
        action = (0.0, 0.0)
        monitor = FinalVetoDecision(
            monitor_id="one_step_overspeed_veto_v0", decision="allow", reason="nominal_within_threshold",
            threshold=1.9, comparator=">", nominal_action=action, executed_action=action,
            fallback_action=action, predicted_nominal_speed_ratio=1.0,
            predicted_fallback_speed_ratio=None, fallback_predicted_to_exceed_threshold=None,
            veto_applied=False,
        )
        return RecoveryBranchExecutionResult(
            branch_id=branch_id, executed=True, action=action,
            previous_state_hash=runtime_state_hash(state), next_state_hash=runtime_state_hash(state),
            terminal_reason="one_step_horizon_complete", transition_count=1, valid=True,
            previous_state=state, next_state=state, monitor_decision=monitor,
            predicted_nominal_state=state,
        )

    def test_registry_member_loading_and_identity(self) -> None:
        identity, state = build_registered_runtime_identity(
            self.registered, implementation_commit="a" * 40
        )
        self.assertEqual(identity.branch_state_hash, self.registered.member.canonical_branch_state_hash)
        self.assertEqual(runtime_state_hash(state), runtime_state_hash(state))

    def test_disabled_observer_and_shadow_observer_are_physically_equivalent(self) -> None:
        baseline = run_registered_bounded_shadow_path(
            self.registered, implementation_commit="a" * 40, step_executor=self.fake_executor
        )
        observed_snapshots = []
        observed = run_registered_bounded_shadow_path(
            self.registered, implementation_commit="a" * 40,
            observer=lambda snapshot: observed_snapshots.append(snapshot),
            step_executor=self.fake_executor,
        )
        report = compare_physical_runs(baseline, observed)
        self.assertTrue(report["all_equivalence_checks"])
        self.assertEqual(len(observed_snapshots), 34)
        self.assertEqual(baseline.recovery_transition_count, 32)

    def test_stage0b_stage1a_shadow_adapter_is_observational(self) -> None:
        identity, _ = build_registered_runtime_identity(
            self.registered, implementation_commit="a" * 40
        )
        adapter = ShadowObservationAdapter(identity, trace_id="fixture")
        observed = run_registered_bounded_shadow_path(
            self.registered, implementation_commit="a" * 40,
            observer=adapter, step_executor=self.fake_executor,
        )
        self.assertEqual(len(adapter.records), len(observed.snapshots))
        self.assertTrue(all(not record.control_authority for record in adapter.records))
        self.assertTrue(all(
            not record.shadow_output_consumed_by_physical_runtime for record in adapter.records
        ))
        self.assertTrue(all(not record.nominal_handoff_recommended for record in adapter.records))

    def test_shadow_return_cannot_be_consumed(self) -> None:
        with self.assertRaises(ObserverIntegrationError):
            run_registered_bounded_shadow_path(
                self.registered, implementation_commit="a" * 40,
                observer=lambda snapshot: {"replacement_action": (1.0, 1.0)},
                step_executor=self.fake_executor,
            )

    def test_shadow_exception_is_infrastructure_failure(self) -> None:
        def failing(_snapshot):
            raise RuntimeError("fixture failure")
        with self.assertRaises(ObserverIntegrationError):
            run_registered_bounded_shadow_path(
                self.registered, implementation_commit="a" * 40,
                observer=failing, step_executor=self.fake_executor,
            )

    def test_horizon_override_is_forbidden_without_executing(self) -> None:
        called = False
        def executor(*args, **kwargs):
            nonlocal called
            called = True
        with self.assertRaisesRegex(Exception, "frozen at 32"):
            run_registered_bounded_shadow_path(
                self.registered, implementation_commit="a" * 40,
                horizon_steps=1, step_executor=executor,
            )
        self.assertFalse(called)

    def test_frozen_existing_branch_can_be_selected_without_shadow_control(self) -> None:
        calls = []
        def executor(member_id, branch_id, horizon_steps=1, *, current_state=None):
            calls.append(branch_id)
            result = self.fake_executor(member_id, PHYSICAL_BRANCH_ID, horizon_steps, current_state=current_state)
            return replace(result, branch_id=branch_id)
        run_registered_bounded_shadow_path(
            self.registered, implementation_commit="a" * 40,
            branch_id="zero_action_reference_v0", step_executor=executor,
        )
        self.assertEqual(set(calls), {"zero_action_reference_v0"})


class ShadowPublicationTests(unittest.TestCase):
    def payloads(self):
        manifest = {
            "smoke_id": "fixture",
            "canonical_manifest_hash": "",
        }
        from runtime_assurance.staged_recovery_shadow_guard import canonical_sha256
        manifest["canonical_manifest_hash"] = canonical_sha256({"smoke_id": "fixture"})
        equivalence = {"all_pairs_equivalent": True, "physical_equivalence_failures": 0}
        summary = {"unauthorized_physical_effects": 0, "nominal_handoff_recommendations": 0}
        values = {
            "manifest.json": json.dumps(manifest).encode(),
            "trace_index.json": b"{}", "equivalence_report.json": json.dumps(equivalence).encode(),
            "phase_transition_matrix.json": b"{}", "shadow_guard_summary.json": json.dumps(summary).encode(),
            "summary.md": b"fixture",
        }
        for index in range(4):
            values[f"traces/case{index}.jsonl"] = b"{}\n"
        return values

    def test_complete_payload_validates_and_publishes_atomically(self) -> None:
        payloads = self.payloads()
        validate_smoke_payloads(payloads)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = publish_smoke_payloads(root, payloads)
            self.assertTrue(target.is_dir())
            self.assertEqual(len([path for path in target.rglob("*") if path.is_file()]), 10)
            with self.assertRaisesRegex(Exception, "already exists"):
                publish_smoke_payloads(root, payloads)

    def test_partial_payload_and_writer_failure_publish_nothing(self) -> None:
        payloads = self.payloads()
        partial = dict(payloads)
        partial.pop("summary.md")
        with self.assertRaises(Exception):
            validate_smoke_payloads(partial)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def fail(_path, _data):
                raise OSError("fixture writer failure")
            with self.assertRaises(OSError):
                publish_smoke_payloads(root, payloads, writer=fail)
            self.assertFalse((root / OUTPUT_PATH).exists())
            self.assertFalse(any((root / "analysis").glob(".staged_recovery_shadow_smoke_v0.*")))

    def test_payload_mutation_fails_manifest_validation(self) -> None:
        payloads = self.payloads()
        mutated = copy.deepcopy(payloads)
        manifest = json.loads(mutated["manifest.json"])
        manifest["new"] = True
        mutated["manifest.json"] = json.dumps(manifest).encode()
        with self.assertRaisesRegex(Exception, "manifest hash"):
            validate_smoke_payloads(mutated)


if __name__ == "__main__":
    unittest.main()
