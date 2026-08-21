from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime_assurance.final_veto_monitor import FinalVetoDecision
from runtime_assurance.recovery_branch_executor import RecoveryBranchExecutionResult
from runtime_assurance.recovery_branch_state_registry import (
    load_registered_branch_state,
)
from runtime_assurance.stage2a_hazard_arrest_runner import (
    EXPERIMENT_ARTIFACTS,
    HAZARD_BRANCH_ID,
    MAXIMUM_PHYSICAL_TRANSITIONS_PER_RUN,
    NORMAL_BRANCH_IDS,
    QUALIFICATION_ARTIFACTS,
    OneStepActionEvaluation,
    PrefixReplay,
    Stage2AHazardArrestRunnerError,
    Stage2AMeasuredExperiment,
    build_experiment_payloads,
    canonical_sha256,
    evaluate_branch_without_execution,
    execute_selected_experiment,
    load_selected_experiment,
    reproduce_selected_prefix,
    sha256_bytes,
    state_document,
    validate_experiment_payloads,
    validate_qualification_payloads,
)
from runtime_assurance.staged_recovery_logger_adapter import runtime_state_hash
from runtime_assurance.staged_recovery_shadow_calibration import (
    CALIBRATION_OUTPUT_PATH,
    TRACE_SET_OUTPUT_PATH,
    atomic_publish_new_directory,
)
from simulator.phase34_35_transition import (
    CartesianState2D,
    NormalizedAction2D,
    Phase3435TransitionResult,
)


ROOT = Path(__file__).resolve().parents[1]


def monitor(
    action: tuple[float, float],
    *,
    decision: str,
    ratio: float,
) -> FinalVetoDecision:
    return FinalVetoDecision(
        monitor_id="one_step_overspeed_veto_v0",
        decision=decision,
        reason=(
            "nominal_within_threshold"
            if decision == "allow"
            else "predicted_nominal_overspeed"
        ),
        threshold=1.9,
        comparator=">",
        nominal_action=action,
        executed_action=action if decision == "allow" else None,
        fallback_action=(0.0, 0.0),
        predicted_nominal_speed_ratio=ratio,
        predicted_fallback_speed_ratio=None,
        fallback_predicted_to_exceed_threshold=None,
        veto_applied=decision == "veto",
    )


class Stage2ARunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registered = load_registered_branch_state(ROOT, "legacy_canonical")
        cls.state = CartesianState2D(x=1.0, y=0.0, vx=3.0, vy=4.0)

    def test_normal_and_hazard_branch_sets_are_frozen(self) -> None:
        self.assertEqual(
            NORMAL_BRANCH_IDS,
            ("zero_action_reference_v0", "tangential_error_correction_v0"),
        )
        self.assertEqual(HAZARD_BRANCH_ID, "velocity_opposed_thrust_v0")
        self.assertEqual(MAXIMUM_PHYSICAL_TRANSITIONS_PER_RUN, 32)

    def test_prediction_evaluation_executes_no_physical_transition(self) -> None:
        next_state = CartesianState2D(x=2.0, y=0.0, vx=1.0, vy=0.0)

        def predicted_step(state, action, context):
            return Phase3435TransitionResult(
                next_state=next_state,
                executed_action=NormalizedAction2D(action.action_x, action.action_y),
            )

        with mock.patch(
            "runtime_assurance.stage2a_hazard_arrest_runner.step_phase34_35_transition",
            side_effect=predicted_step,
        ) as transition:
            result = evaluate_branch_without_execution(
                self.registered, self.state, HAZARD_BRANCH_ID
            )
        self.assertEqual(result.physical_transition_count, 0)
        self.assertEqual(transition.call_count, 1)
        self.assertEqual(result.action, (-0.15, -0.2))

    def test_prediction_uses_unchanged_final_veto(self) -> None:
        _, target_speed, _ = __import__(
            "runtime_assurance.stage2a_hazard_arrest_runner",
            fromlist=["_registered_dynamics"],
        )._registered_dynamics(self.registered)
        overspeed = CartesianState2D(x=1.0, y=0.0, vx=target_speed * 2.0, vy=0.0)

        def predicted_step(state, action, context):
            return Phase3435TransitionResult(
                next_state=overspeed,
                executed_action=NormalizedAction2D(action.action_x, action.action_y),
            )

        with mock.patch(
            "runtime_assurance.stage2a_hazard_arrest_runner.step_phase34_35_transition",
            side_effect=predicted_step,
        ):
            result = evaluate_branch_without_execution(
                self.registered, self.state, "zero_action_reference_v0"
            )
        self.assertEqual(result.final_veto_decision.decision, "veto")
        self.assertGreaterEqual(result.fallback_prediction_count, 1)
        self.assertEqual(result.physical_transition_count, 0)

    def test_action_hash_is_deterministic(self) -> None:
        next_state = CartesianState2D(x=2.0, y=0.0, vx=1.0, vy=0.0)
        transition = Phase3435TransitionResult(
            next_state=next_state,
            executed_action=NormalizedAction2D(-0.15, -0.2),
        )
        with mock.patch(
            "runtime_assurance.stage2a_hazard_arrest_runner.step_phase34_35_transition",
            return_value=transition,
        ):
            first = evaluate_branch_without_execution(
                self.registered, self.state, HAZARD_BRANCH_ID
            )
            second = evaluate_branch_without_execution(
                self.registered, self.state, HAZARD_BRANCH_ID
            )
        self.assertEqual(first, second)
        self.assertEqual(first.action_hash, second.action_hash)

    def test_unsupported_branch_is_rejected(self) -> None:
        with self.assertRaises(Stage2AHazardArrestRunnerError):
            evaluate_branch_without_execution(
                self.registered, self.state, "velocity_opposed_clone"
            )

    def test_prefix_replay_detects_source_mismatch_without_retry(self) -> None:
        selected = self._source_selected(prefix_count=1)
        initial = {
            "source_event": {"pre_state_hash": runtime_state_hash(self.state)}
        }
        source = {
            "source_event": {
                "pre_state_hash": runtime_state_hash(self.state),
                "proposed_action": [0.0, 0.0],
                "monitor_decision": "allow",
                "predicted_state_hash": "1" * 64,
                "realized_state_hash": "2" * 64,
                "transition_executed": True,
                "canonical_event_sha256": "3" * 64,
            }
        }

        def bad_executor(*args, **kwargs):
            state = kwargs["current_state"]
            return RecoveryBranchExecutionResult(
                branch_id=selected["prefix_branch_id"],
                executed=False,
                action=(0.0, 0.0),
                previous_state_hash=runtime_state_hash(state),
                next_state_hash=None,
                terminal_reason="recovery_action_rejected",
                transition_count=0,
                valid=True,
                previous_state=state,
                next_state=None,
                monitor_decision=monitor((0.0, 0.0), decision="veto", ratio=2.0),
            )

        with (
            mock.patch(
                "runtime_assurance.stage2a_hazard_arrest_runner.load_registered_branch_state",
                return_value=self.registered,
            ),
            mock.patch(
                "runtime_assurance.stage2a_hazard_arrest_runner.build_registered_runtime_identity",
                return_value=(mock.Mock(), self.state),
            ),
            mock.patch(
                "runtime_assurance.stage2a_hazard_arrest_runner._source_trace",
                return_value=(initial, source),
            ),
            self.assertRaisesRegex(
                Stage2AHazardArrestRunnerError, "prefix reproduction mismatch"
            ),
        ):
            reproduce_selected_prefix(
                ROOT,
                selected,
                implementation_commit="a" * 40,
                step_executor=bad_executor,
            )

    def test_selected_source_trace_hash_is_required(self) -> None:
        selected = self._source_selected(prefix_count=0)
        selected["source_trace_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            repository_root = Path(directory)
            source_path = (
                repository_root
                / TRACE_SET_OUTPUT_PATH
                / str(selected["source_trace_path"])
            )
            source_path.parent.mkdir(parents=True)
            source_path.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch(
                    "runtime_assurance.stage2a_hazard_arrest_runner.load_registered_branch_state",
                    return_value=self.registered,
                ),
                mock.patch(
                    "runtime_assurance.stage2a_hazard_arrest_runner.build_registered_runtime_identity",
                    return_value=(mock.Mock(), self.state),
                ),
                self.assertRaisesRegex(
                    Stage2AHazardArrestRunnerError, "source trace hash"
                ),
            ):
                reproduce_selected_prefix(
                    repository_root,
                    selected,
                    implementation_commit="a" * 40,
                    step_executor=mock.Mock(),
                )

    def _source_selected(self, *, prefix_count: int) -> dict[str, object]:
        return {
            "registry_member_id": "legacy_canonical",
            "source_trace_path": "traces/synthetic.jsonl",
            "source_trace_sha256": "f" * 64,
            "prefix_branch_id": "zero_action_reference_v0",
            "prefix_transition_count": prefix_count,
            "boundary_state_hash": runtime_state_hash(self.state),
        }


class Stage2AMeasuredBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registered = load_registered_branch_state(ROOT, "legacy_canonical")
        cls.state = CartesianState2D(x=1.0, y=0.0, vx=3.0, vy=4.0)

    def evaluation(self, branch: str, action, ratio: float, decision: str) -> OneStepActionEvaluation:
        predicted = CartesianState2D(x=2.0, y=0.0, vx=1.0, vy=0.0)
        return OneStepActionEvaluation(
            branch_id=branch,
            action=action,
            action_hash=canonical_sha256({"action": list(action)}),
            predicted_state=predicted,
            predicted_state_hash=runtime_state_hash(predicted),
            predicted_speed_ratio=ratio,
            predicted_headroom=1.9 - ratio,
            final_veto_decision=monitor(action, decision=decision, ratio=ratio),
            fallback_prediction_count=1 if decision == "veto" else 0,
        )

    def selected(self) -> dict[str, object]:
        normal = self.evaluation("zero_action_reference_v0", (0.0, 0.0), 1.91, "veto")
        hazard_action = (-0.15, -0.2)
        hazard = self.evaluation(HAZARD_BRANCH_ID, hazard_action, 1.8, "allow")
        return {
            "registry_member_id": "legacy_canonical",
            "selection_hash": "a" * 64,
            "prefix_branch_id": "zero_action_reference_v0",
            "prefix_transition_count": 0,
            "normal_branch_id": "zero_action_reference_v0",
            "boundary_state_hash": runtime_state_hash(self.state),
            "normal_action_hash": normal.action_hash,
            "normal_predicted_state_hash": normal.predicted_state_hash,
            "normal_predicted_speed_ratio": normal.predicted_speed_ratio,
            "normal_final_veto_decision": "veto",
            "hazard_action_hash": hazard.action_hash,
            "hazard_predicted_state_hash": hazard.predicted_state_hash,
            "hazard_predicted_speed_ratio": hazard.predicted_speed_ratio,
            "hazard_final_veto_decision": "allow",
        }

    def run_once(self) -> Stage2AMeasuredExperiment:
        selected = self.selected()
        normal = self.evaluation("zero_action_reference_v0", (0.0, 0.0), 1.91, "veto")
        hazard = self.evaluation(HAZARD_BRANCH_ID, (-0.15, -0.2), 1.8, "allow")
        resumed = self.evaluation("zero_action_reference_v0", (0.0, 0.0), 1.8, "allow")
        next_state = hazard.predicted_state

        def executor(member_id, branch_id, horizon_steps=1, *, current_state=None):
            if branch_id == "zero_action_reference_v0":
                return RecoveryBranchExecutionResult(
                    branch_id=branch_id,
                    executed=False,
                    action=(0.0, 0.0),
                    previous_state_hash=runtime_state_hash(current_state),
                    next_state_hash=None,
                    terminal_reason="recovery_action_rejected",
                    transition_count=0,
                    valid=True,
                    previous_state=current_state,
                    next_state=None,
                    monitor_decision=normal.final_veto_decision,
                    predicted_nominal_state=normal.predicted_state,
                )
            return RecoveryBranchExecutionResult(
                branch_id=branch_id,
                executed=True,
                action=(-0.15, -0.2),
                previous_state_hash=runtime_state_hash(current_state),
                next_state_hash=runtime_state_hash(next_state),
                terminal_reason="one_step_horizon_complete",
                transition_count=1,
                valid=True,
                previous_state=current_state,
                next_state=next_state,
                monitor_decision=hazard.final_veto_decision,
                predicted_nominal_state=next_state,
            )

        with (
            mock.patch(
                "runtime_assurance.stage2a_hazard_arrest_runner.load_registered_branch_state",
                return_value=self.registered,
            ),
            mock.patch(
                "runtime_assurance.stage2a_hazard_arrest_runner.reproduce_selected_prefix",
                return_value=PrefixReplay(self.state, (), 0),
            ),
            mock.patch(
                "runtime_assurance.stage2a_hazard_arrest_runner.evaluate_branch_without_execution",
                side_effect=[normal, hazard, resumed],
            ),
            mock.patch(
                "runtime_assurance.stage2a_hazard_arrest_runner._speed_ratio_for_registered",
                side_effect=[1.8, 1.8, 1.8, 1.8],
            ),
        ):
            return execute_selected_experiment(
                ROOT,
                selected,
                implementation_commit="b" * 40,
                step_executor=executor,
            )

    def test_valid_trigger_creates_and_consumes_one_proposal(self) -> None:
        result = self.run_once()
        self.assertTrue(result.authority_report["proposal_generated"])
        self.assertEqual(result.authority_report["proposal_count"], 1)
        self.assertTrue(result.authority_report["proposal_consumed"])
        self.assertEqual(result.authority_report["second_intervention_count"], 0)

    def test_final_veto_is_mandatory_and_fallback_is_not_executed(self) -> None:
        result = self.run_once()
        self.assertEqual(result.final_veto_report["baseline_decision"], "veto")
        self.assertEqual(result.final_veto_report["hazard_decision"], "allow")
        self.assertEqual(result.final_veto_report["final_veto_bypass_count"], 0)
        self.assertEqual(result.final_veto_report["fallback_execution_count"], 0)

    def test_allowed_proposal_executes_one_predicted_transition(self) -> None:
        result = self.run_once()
        self.assertEqual(result.active_summary["boundary_transition_count"], 1)
        self.assertTrue(result.intervention_effect["prediction_realization_equal"])

    def test_release_executes_no_resumed_action(self) -> None:
        result = self.run_once()
        self.assertEqual(result.release_report["resumed_physical_action_count"], 0)
        self.assertEqual(result.release_report["return_authority_to"], "zero_action_reference_v0")
        self.assertEqual(result.release_report["release_status"], "not_authorized")

    def test_no_other_staged_phase_receives_authority(self) -> None:
        result = self.run_once()
        self.assertEqual(result.authority_report["unauthorized_phase_count"], 0)
        self.assertEqual(result.authority_report["authority_leakage_count"], 0)
        self.assertFalse(result.authority_report["shadow_output_consumed_by_physical_runtime"])

    def test_mocked_execution_is_deterministic(self) -> None:
        self.assertEqual(self.run_once(), self.run_once())


class Stage2AArtifactTests(unittest.TestCase):
    def fixture(self) -> Stage2AMeasuredExperiment:
        selected = {
            "registry_member_id": "member",
            "source_trace_id": "trace",
            "prefix_transition_count": 1,
            "normal_branch_id": "zero_action_reference_v0",
            "current_realized_speed_ratio": 1.8,
        }
        selected["selection_hash"] = canonical_sha256(selected)
        record = {"event_type": "fixture"}
        record["canonical_record_hash"] = canonical_sha256(record)
        return Stage2AMeasuredExperiment(
            selected=selected,
            baseline_summary={"boundary_transition_count": 0, "final_veto_decision": "veto", "normal_predicted_speed_ratio": 1.91},
            active_summary={"boundary_transition_count": 1, "hazard_final_veto_decision": "allow"},
            boundary_equivalence={"checks": {"same_prefix": True}, "same_boundary_state_hash": True, "same_normal_action_hash": True, "same_normal_prediction_hash": True, "all_required_prefix_checks": True},
            authority_report={"requested_phase": "hazard_arrest", "proposal_generated": True, "proposal_count": 1, "proposal_consumed": True, "second_intervention_count": 0, "unauthorized_phase_count": 0, "authority_leakage_count": 0, "invalid_evidence_consumption_count": 0},
            final_veto_report={"baseline_decision": "veto", "hazard_decision": "allow", "final_veto_bypass_count": 0, "fallback_execution_count": 0},
            intervention_effect={"prediction_realization_equal": True},
            release_report={"return_authority_to": "zero_action_reference_v0", "release_status": "not_authorized", "resumed_physical_action_count": 0},
            baseline_trace=(record,),
            active_trace=(record,),
        )

    def test_experiment_payload_hashes_recompute(self) -> None:
        hashes = {"protected": "1" * 64}
        payloads = build_experiment_payloads(
            self.fixture(),
            implementation_commit="b" * 40,
            selection_commit="c" * 40,
            protected_before=hashes,
            protected_after=hashes,
        )
        self.assertEqual(set(payloads), set(EXPERIMENT_ARTIFACTS))
        validate_experiment_payloads(payloads)

    def test_protected_hash_change_blocks_publication(self) -> None:
        with self.assertRaisesRegex(Stage2AHazardArrestRunnerError, "protected evidence"):
            build_experiment_payloads(
                self.fixture(),
                implementation_commit="b" * 40,
                selection_commit="c" * 40,
                protected_before={"x": "1" * 64},
                protected_after={"x": "2" * 64},
            )

    def test_event_mutation_invalidates_trace_hash(self) -> None:
        hashes = {"protected": "1" * 64}
        payloads = build_experiment_payloads(
            self.fixture(),
            implementation_commit="b" * 40,
            selection_commit="c" * 40,
            protected_before=hashes,
            protected_after=hashes,
        )
        changed = dict(payloads)
        row = json.loads(changed["traces/active.jsonl"])
        row["event_type"] = "mutated"
        changed["traces/active.jsonl"] = json.dumps(row).encode() + b"\n"
        with self.assertRaises(Stage2AHazardArrestRunnerError):
            validate_experiment_payloads(changed)

    def test_boundary_equivalence_mutation_is_rejected(self) -> None:
        hashes = {"protected": "1" * 64}
        payloads = build_experiment_payloads(
            self.fixture(),
            implementation_commit="b" * 40,
            selection_commit="c" * 40,
            protected_before=hashes,
            protected_after=hashes,
        )
        changed = dict(payloads)
        boundary = json.loads(changed["boundary_equivalence_report.json"])
        boundary["same_boundary_state_hash"] = False
        changed["boundary_equivalence_report.json"] = json.dumps(boundary).encode()
        with self.assertRaisesRegex(Stage2AHazardArrestRunnerError, "boundary identity"):
            validate_experiment_payloads(changed)

    def test_selection_mutation_is_rejected(self) -> None:
        hashes = {"protected": "1" * 64}
        payloads = build_experiment_payloads(
            self.fixture(),
            implementation_commit="b" * 40,
            selection_commit="c" * 40,
            protected_before=hashes,
            protected_after=hashes,
        )
        changed = dict(payloads)
        selected = json.loads(changed["selected_case.json"])
        selected["normal_branch_id"] = "tangential_error_correction_v0"
        changed["selected_case.json"] = json.dumps(selected).encode()
        with self.assertRaisesRegex(Stage2AHazardArrestRunnerError, "selection identity"):
            validate_experiment_payloads(changed)

    def test_atomic_publication_rejects_existing_target(self) -> None:
        hashes = {"protected": "1" * 64}
        payloads = build_experiment_payloads(
            self.fixture(),
            implementation_commit="b" * 40,
            selection_commit="c" * 40,
            protected_before=hashes,
            protected_after=hashes,
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            relative = Path(temp).relative_to(ROOT) / "analysis" / "result"
            target = ROOT / relative
            target.mkdir(parents=True)
            with self.assertRaises(Exception):
                atomic_publish_new_directory(
                    ROOT, relative, payloads, validate_experiment_payloads
                )

    def test_qualification_payload_set_is_exact(self) -> None:
        selected = {"selection_status": "no_eligible_boundary", "selection_rule": "lexical"}
        selected["selection_hash"] = canonical_sha256(selected)
        manifest = {
            "physical_executions": 0,
            "eligible_boundary_count": 0,
            "selected_experiment_hash": selected["selection_hash"],
        }
        manifest["canonical_manifest_hash"] = canonical_sha256(manifest)
        payloads = {
            "qualification_manifest.json": json.dumps(manifest).encode(),
            "eligible_boundaries.json": json.dumps({"schema_version": "stage2a_one_intervention_hazard_arrest_v0", "eligible_boundary_count": 0, "eligible_boundaries": []}).encode(),
            "selected_experiment.json": json.dumps(selected).encode(),
            "summary.md": b"summary",
        }
        self.assertEqual(set(payloads), set(QUALIFICATION_ARTIFACTS))
        validate_qualification_payloads(payloads)


class Stage2ACliSafetyTests(unittest.TestCase):
    def run_cli(self, script: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_default_clis_execute_nothing(self) -> None:
        for script in (
            "qualify_stage2a_hazard_arrest_case_v0.py",
            "run_stage2a_hazard_arrest_experiment_v0.py",
            "check_stage2a_hazard_arrest_experiment_v0.py",
        ):
            with self.subTest(script=script):
                result = self.run_cli(script)
                self.assertEqual(result.returncode, 0)
                self.assertIn("usage:", result.stdout)

    def test_cli_has_no_scientific_override_flags(self) -> None:
        for script in (
            "qualify_stage2a_hazard_arrest_case_v0.py",
            "run_stage2a_hazard_arrest_experiment_v0.py",
        ):
            help_text = self.run_cli(script, "--help").stdout
            for forbidden in ("--case", "--threshold", "--branch", "--retry", "--output"):
                self.assertNotIn(forbidden, help_text)

    def test_no_result_is_published_by_unit_tests(self) -> None:
        self.assertFalse((ROOT / "analysis/stage2a_hazard_arrest_experiment_v0").exists())


if __name__ == "__main__":
    unittest.main()
