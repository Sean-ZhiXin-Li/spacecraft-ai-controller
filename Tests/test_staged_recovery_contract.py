from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import runtime_assurance.staged_recovery_contract as contract
from runtime_assurance.staged_recovery_contract import (
    DEFAULT_ARCHITECTURE_MANIFEST,
    EXECUTION_NOT_AUTHORIZED,
    FROZEN_ADVERSE_STOP_PRIORITY,
    RecoveryEvidenceSupport,
    RecoveryHysteresisContract,
    RecoveryNoProgressContract,
    RecoveryNoProgressStatus,
    RecoveryPhase,
    RecoverySignalStatus,
    RecoverySignalValue,
    StagedRecoveryObservation,
    architecture_manifest_document,
    canonical_json_bytes,
    canonical_sha256,
    evaluate_no_progress,
    evaluate_nominal_handoff,
    evaluate_transition_evidence,
    evidence_traceability_document,
    replace_signals,
    select_adverse_stop,
    unavailable_signals,
    validate_architecture_contract,
    validate_architecture_manifest_document,
)


ROOT = Path(__file__).resolve().parents[1]


def measured(value: object) -> RecoverySignalValue:
    return RecoverySignalValue(RecoverySignalStatus.MEASURED, value, "synthetic_test")


def observation(
    *,
    phase: RecoveryPhase = RecoveryPhase.RECOVERABILITY_VERIFICATION,
    **updates: RecoverySignalValue,
) -> StagedRecoveryObservation:
    return StagedRecoveryObservation(
        case_id="synthetic_case",
        seed=0,
        branch_state_hash="a" * 64,
        simulator_configuration_hash="b" * 64,
        constants_hash="c" * 64,
        implementation_commit="d" * 40,
        recovery_step=1,
        total_transition_count=2,
        current_phase=phase,
        previous_phase=None,
        signals=replace_signals(unavailable_signals(), **updates),
    )


def complete_handoff_observation(**updates: RecoverySignalValue) -> StagedRecoveryObservation:
    values = {
        "simulation_validity": measured(True),
        "recovery_evaluation_validity": measured(True),
        "overspeed_status": measured(False),
        "instability_status": measured(False),
        "unsafe_state_status": measured(False),
        "action_rejection_status": measured(False),
        "explicit_abort_requested": measured(False),
        "target_radius_crossing": measured(True),
        "phase34_compatible_recoverability": measured(True),
        "recovery_success_v0": measured(True),
        "handoff_readiness": measured(True),
    }
    values.update(updates)
    return observation(**values)


def rehash(manifest):
    unhashed = replace(manifest, canonical_payload_hash="")
    return replace(
        unhashed,
        canonical_payload_hash=canonical_sha256(contract._manifest_payload(unhashed)),
    )


class PhaseDefinitionTests(unittest.TestCase):
    def test_all_required_phase_ids_exist_and_are_unique(self) -> None:
        phase_ids = tuple(item.phase_id for item in DEFAULT_ARCHITECTURE_MANIFEST.phase_contracts)
        self.assertEqual(phase_ids, tuple(RecoveryPhase))
        self.assertEqual(len(phase_ids), len(set(phase_ids)))

    def test_terminal_phases_are_explicit(self) -> None:
        terminal = {
            item.phase_id
            for item in DEFAULT_ARCHITECTURE_MANIFEST.phase_contracts
            if item.terminal
        }
        self.assertEqual(
            terminal,
            {RecoveryPhase.NOMINAL_HANDOFF, RecoveryPhase.EXPLICIT_ABORT},
        )

    def test_every_active_phase_has_purpose_and_required_evidence(self) -> None:
        for item in DEFAULT_ARCHITECTURE_MANIFEST.phase_contracts:
            self.assertTrue(item.purpose)
            if not item.terminal:
                self.assertTrue(item.required_signal_ids)

    def test_active_action_laws_remain_unresolved(self) -> None:
        active_action_phases = {
            RecoveryPhase.HAZARD_ARREST,
            RecoveryPhase.RADIAL_RECOMMITMENT,
            RecoveryPhase.TANGENTIAL_ALIGNMENT,
            RecoveryPhase.CROSSING_PREPARATION,
            RecoveryPhase.RETREAT,
        }
        for item in DEFAULT_ARCHITECTURE_MANIFEST.phase_contracts:
            if item.phase_id in active_action_phases:
                self.assertEqual(
                    item.action_law_status,
                    RecoveryEvidenceSupport.NOT_YET_SPECIFIED,
                )


class TransitionGraphTests(unittest.TestCase):
    def test_default_graph_validates_structurally(self) -> None:
        report = validate_architecture_contract(DEFAULT_ARCHITECTURE_MANIFEST)
        self.assertTrue(report.structurally_valid, report.errors)
        self.assertFalse(report.execution_authorized)

    def test_unknown_transition_endpoint_fails(self) -> None:
        changed = replace(
            DEFAULT_ARCHITECTURE_MANIFEST.transitions[0],
            destination_phase="unknown_phase",  # type: ignore[arg-type]
        )
        manifest = rehash(
            replace(
                DEFAULT_ARCHITECTURE_MANIFEST,
                transitions=(changed,) + DEFAULT_ARCHITECTURE_MANIFEST.transitions[1:],
            )
        )
        self.assertIn(
            "unknown_transition_endpoint:hazard_arrest__to__stabilization_assessment",
            validate_architecture_contract(manifest).errors,
        )

    def test_duplicate_transition_fails(self) -> None:
        manifest = rehash(
            replace(
                DEFAULT_ARCHITECTURE_MANIFEST,
                transitions=DEFAULT_ARCHITECTURE_MANIFEST.transitions
                + (DEFAULT_ARCHITECTURE_MANIFEST.transitions[0],),
            )
        )
        self.assertIn("duplicate_transition", validate_architecture_contract(manifest).errors)

    def test_forbidden_direct_handoffs_are_absent(self) -> None:
        endpoints = {
            (item.source_phase, item.destination_phase)
            for item in DEFAULT_ARCHITECTURE_MANIFEST.transitions
        }
        for source in (
            RecoveryPhase.HAZARD_ARREST,
            RecoveryPhase.STABILIZATION_ASSESSMENT,
            RecoveryPhase.RADIAL_RECOMMITMENT,
            RecoveryPhase.TANGENTIAL_ALIGNMENT,
        ):
            self.assertNotIn((source, RecoveryPhase.NOMINAL_HANDOFF), endpoints)

    def test_forbidden_direct_handoff_fails_validation(self) -> None:
        changed = replace(
            DEFAULT_ARCHITECTURE_MANIFEST.transitions[0],
            transition_id="hazard_arrest__to__nominal_handoff",
            destination_phase=RecoveryPhase.NOMINAL_HANDOFF,
        )
        manifest = rehash(
            replace(
                DEFAULT_ARCHITECTURE_MANIFEST,
                transitions=(changed,) + DEFAULT_ARCHITECTURE_MANIFEST.transitions[1:],
            )
        )
        self.assertIn(
            "forbidden_direct_nominal_handoff",
            validate_architecture_contract(manifest).errors,
        )

    def test_explicit_abort_has_no_active_outgoing_transition(self) -> None:
        self.assertFalse(
            any(
                item.source_phase == RecoveryPhase.EXPLICIT_ABORT
                for item in DEFAULT_ARCHITECTURE_MANIFEST.transitions
            )
        )

    def test_nominal_handoff_has_no_recovery_outgoing_transition(self) -> None:
        self.assertFalse(
            any(
                item.source_phase == RecoveryPhase.NOMINAL_HANDOFF
                for item in DEFAULT_ARCHITECTURE_MANIFEST.transitions
            )
        )

    def test_terminal_outgoing_transition_fails_validation(self) -> None:
        changed = replace(
            DEFAULT_ARCHITECTURE_MANIFEST.transitions[0],
            transition_id="explicit_abort__to__hazard_arrest",
            source_phase=RecoveryPhase.EXPLICIT_ABORT,
            destination_phase=RecoveryPhase.HAZARD_ARREST,
        )
        manifest = rehash(
            replace(
                DEFAULT_ARCHITECTURE_MANIFEST,
                transitions=DEFAULT_ARCHITECTURE_MANIFEST.transitions + (changed,),
            )
        )
        self.assertIn(
            "terminal_phase_has_outgoing_transition:explicit_abort",
            validate_architecture_contract(manifest).errors,
        )

    def test_unreachable_phase_fails_validation(self) -> None:
        transitions = tuple(
            item
            for item in DEFAULT_ARCHITECTURE_MANIFEST.transitions
            if item.destination_phase != RecoveryPhase.CROSSING_PREPARATION
        )
        manifest = rehash(replace(DEFAULT_ARCHITECTURE_MANIFEST, transitions=transitions))
        self.assertIn("unreachable_phase", validate_architecture_contract(manifest).errors)

    def test_every_active_phase_has_retreat_or_abort_route(self) -> None:
        report = validate_architecture_contract(DEFAULT_ARCHITECTURE_MANIFEST)
        self.assertFalse(any(error.startswith("active_phase_lacks_escape") for error in report.errors))


class RecoverySemanticsTests(unittest.TestCase):
    def test_complete_valid_recovery_allows_handoff_evidence(self) -> None:
        decision = evaluate_nominal_handoff(complete_handoff_observation())
        self.assertEqual(decision.status, "allowed")
        self.assertEqual(decision.destination_phase, RecoveryPhase.NOMINAL_HANDOFF)

    def test_crossing_alone_cannot_authorize_handoff(self) -> None:
        decision = evaluate_nominal_handoff(
            observation(target_radius_crossing=measured(True))
        )
        self.assertEqual(decision.status, "blocked")

    def test_no_overspeed_alone_cannot_authorize_handoff(self) -> None:
        decision = evaluate_nominal_handoff(observation(overspeed_status=measured(False)))
        self.assertEqual(decision.status, "blocked")

    def test_simulator_success_alone_cannot_authorize_handoff(self) -> None:
        decision = evaluate_nominal_handoff(observation(simulator_success=measured(True)))
        self.assertEqual(decision.status, "blocked")

    def test_tangential_improvement_alone_cannot_authorize_handoff(self) -> None:
        decision = evaluate_nominal_handoff(
            observation(tangential_progress=measured(True))
        )
        self.assertEqual(decision.status, "blocked")

    def test_retreat_is_not_recovery_success(self) -> None:
        phase = next(
            item
            for item in DEFAULT_ARCHITECTURE_MANIFEST.phase_contracts
            if item.phase_id == RecoveryPhase.RETREAT
        )
        self.assertIn("retreat_as_task_recovery", phase.prohibited_outcomes)
        self.assertTrue(phase.action_law_status == RecoveryEvidenceSupport.NOT_YET_SPECIFIED)

    def test_explicit_abort_is_not_recovery_success(self) -> None:
        phase = next(
            item
            for item in DEFAULT_ARCHITECTURE_MANIFEST.phase_contracts
            if item.phase_id == RecoveryPhase.EXPLICIT_ABORT
        )
        self.assertIn("recovery_success_claim", phase.prohibited_outcomes)
        self.assertIn("physical_transition", phase.prohibited_outcomes)


class EvidenceHandlingTests(unittest.TestCase):
    def test_missing_required_evidence_blocks_transition(self) -> None:
        transition = DEFAULT_ARCHITECTURE_MANIFEST.transitions[0]
        decision = evaluate_transition_evidence(
            transition,
            observation(phase=transition.source_phase),
        )
        self.assertEqual(decision.status, "blocked")
        self.assertTrue(decision.missing_signal_ids)

    def test_not_evaluated_blocks_positive_transition(self) -> None:
        decision = evaluate_nominal_handoff(complete_handoff_observation(
            handoff_readiness=RecoverySignalValue(RecoverySignalStatus.NOT_EVALUATED)
        ))
        self.assertEqual(decision.status, "blocked")

    def test_invalid_evidence_triggers_adverse_handling(self) -> None:
        decision = evaluate_nominal_handoff(complete_handoff_observation(
            handoff_readiness=RecoverySignalValue(RecoverySignalStatus.INVALID)
        ))
        self.assertEqual(decision.status, "adverse")
        self.assertEqual(decision.adverse_stop, "invalid_recovery_evaluation")

    def test_unknown_boolean_is_not_converted_to_false(self) -> None:
        decision = evaluate_nominal_handoff(complete_handoff_observation(
            overspeed_status=measured("unknown")
        ))
        self.assertEqual(decision.status, "blocked")
        self.assertIn("overspeed_status", decision.missing_signal_ids)

    def test_unknown_numeric_value_is_not_converted_to_zero(self) -> None:
        signal = RecoverySignalValue(RecoverySignalStatus.NOT_EVALUATED)
        self.assertIsNone(signal.value)
        with self.assertRaises(ValueError):
            RecoverySignalValue(RecoverySignalStatus.MEASURED, None)

    def test_invalid_evidence_classification_fails(self) -> None:
        phase = replace(
            DEFAULT_ARCHITECTURE_MANIFEST.phase_contracts[0],
            evidence_support="unsupported",  # type: ignore[arg-type]
        )
        manifest = rehash(
            replace(
                DEFAULT_ARCHITECTURE_MANIFEST,
                phase_contracts=(phase,) + DEFAULT_ARCHITECTURE_MANIFEST.phase_contracts[1:],
            )
        )
        self.assertIn(
            "phase_evidence_status_invalid:hazard_arrest",
            validate_architecture_contract(manifest).errors,
        )


class StopPriorityTests(unittest.TestCase):
    def test_frozen_priority_matches_runtime_source_without_importing_it(self) -> None:
        source = (ROOT / "runtime_assurance/recovery_stop_conditions.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        string_constants: dict[str, str] = {}
        priority_names: tuple[str, ...] | None = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, str):
                        string_constants[target.id] = node.value.value
                if (
                    isinstance(target, ast.Name)
                    and target.id == "_TERMINATION_PRIORITY"
                    and isinstance(node.value, ast.Tuple)
                ):
                    priority_names = tuple(
                        item.id for item in node.value.elts if isinstance(item, ast.Name)
                    )
        self.assertIsNotNone(priority_names)
        runtime_priority = tuple(string_constants[name] for name in priority_names or ())
        self.assertEqual(FROZEN_ADVERSE_STOP_PRIORITY, runtime_priority)

    def test_overspeed_beats_simultaneous_recovery_success(self) -> None:
        current = complete_handoff_observation(overspeed_status=measured(True))
        self.assertEqual(select_adverse_stop(current), "overspeed")

    def test_invalid_simulation_beats_phase_progression(self) -> None:
        current = complete_handoff_observation(simulation_validity=measured(False))
        self.assertEqual(select_adverse_stop(current), "invalid_simulation")
        self.assertEqual(evaluate_nominal_handoff(current).status, "adverse")

    def test_action_rejection_terminates(self) -> None:
        current = complete_handoff_observation(action_rejection_status=measured(True))
        self.assertEqual(select_adverse_stop(current), "action_rejected")

    def test_explicit_abort_cannot_be_overridden_by_success(self) -> None:
        current = complete_handoff_observation(explicit_abort_requested=measured(True))
        self.assertEqual(select_adverse_stop(current), "explicit_abort")

    def test_not_evaluated_does_not_trigger_stop_or_handoff(self) -> None:
        current = observation()
        self.assertIsNone(select_adverse_stop(current))
        self.assertEqual(evaluate_nominal_handoff(current).status, "blocked")

    def test_recovery_success_precedes_simultaneous_horizon_exhaustion(self) -> None:
        current = complete_handoff_observation(
            recovery_horizon_exhausted=measured(True),
            total_horizon_exhausted=measured(True),
        )
        self.assertEqual(select_adverse_stop(current), "recovery_success")

    def test_horizon_exhaustion_is_retained_when_recovery_is_clear(self) -> None:
        current = complete_handoff_observation(
            recovery_success_v0=measured(False),
            recovery_horizon_exhausted=measured(True),
        )
        self.assertEqual(select_adverse_stop(current), "recovery_horizon_exhausted")


class NoProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.complete = RecoveryNoProgressContract(
            monitored_signal_id="target_radius_error",
            desired_direction="toward_zero",
            observation_window=3,
            minimum_meaningful_improvement=0.5,
            phase_specific_dwell_limit=5,
            executable=True,
        )

    def test_missing_window_configuration_is_not_evaluated(self) -> None:
        result = evaluate_no_progress((measured(3.0), measured(2.0)), RecoveryNoProgressContract())
        self.assertEqual(result.status, RecoveryNoProgressStatus.NOT_EVALUATED)

    def test_missing_improvement_threshold_is_not_evaluated(self) -> None:
        incomplete = replace(self.complete, minimum_meaningful_improvement=None)
        result = evaluate_no_progress((measured(3.0), measured(2.0)), incomplete)
        self.assertEqual(result.status, RecoveryNoProgressStatus.NOT_EVALUATED)

    def test_timeout_alone_does_not_prove_stall(self) -> None:
        result = evaluate_no_progress((measured(3.0),), self.complete, timed_out=True)
        self.assertEqual(result.status, RecoveryNoProgressStatus.NOT_EVALUATED)

    def test_missing_signal_gives_not_evaluated(self) -> None:
        samples = (
            measured(3.0),
            RecoverySignalValue(RecoverySignalStatus.NOT_EVALUATED),
            measured(2.0),
        )
        result = evaluate_no_progress(samples, self.complete)
        self.assertEqual(result.status, RecoveryNoProgressStatus.NOT_EVALUATED)

    def test_invalid_samples_give_invalid(self) -> None:
        samples = (
            measured(3.0),
            RecoverySignalValue(RecoverySignalStatus.INVALID),
            measured(2.0),
        )
        result = evaluate_no_progress(samples, self.complete)
        self.assertEqual(result.status, RecoveryNoProgressStatus.INVALID)

    def test_progressing_stalled_and_regressing_are_distinct(self) -> None:
        progressing = evaluate_no_progress(
            (measured(3.0), measured(2.5), measured(2.0)), self.complete
        )
        stalled = evaluate_no_progress(
            (measured(3.0), measured(2.9), measured(2.8)), self.complete
        )
        regressing = evaluate_no_progress(
            (measured(2.0), measured(2.5), measured(3.0)), self.complete
        )
        self.assertEqual(progressing.status, RecoveryNoProgressStatus.PROGRESSING)
        self.assertEqual(stalled.status, RecoveryNoProgressStatus.STALLED)
        self.assertEqual(regressing.status, RecoveryNoProgressStatus.REGRESSING)

    def test_no_progress_status_cannot_generate_recovery_success(self) -> None:
        current = complete_handoff_observation(
            recovery_success_v0=measured(False),
            no_progress_status=measured("stalled"),
        )
        self.assertEqual(evaluate_nominal_handoff(current).status, "blocked")


class HysteresisTests(unittest.TestCase):
    def test_executable_transition_without_hysteresis_fails(self) -> None:
        transition = replace(DEFAULT_ARCHITECTURE_MANIFEST.transitions[0], executable=True)
        manifest = rehash(
            replace(
                DEFAULT_ARCHITECTURE_MANIFEST,
                transitions=(transition,) + DEFAULT_ARCHITECTURE_MANIFEST.transitions[1:],
            )
        )
        self.assertTrue(
            any(
                error.startswith("executable_transition_missing_hysteresis")
                for error in validate_architecture_contract(manifest).errors
            )
        )

    def test_unlimited_phase_switching_budget_fails_executable_hysteresis(self) -> None:
        hysteresis = RecoveryHysteresisContract(
            minimum_phase_dwell=2,
            entry_threshold_id="entry",
            exit_threshold_id="exit",
            transition_cooldown=1,
            consecutive_evidence_count=2,
            phase_transition_budget=None,
            repeated_cycle_window=4,
            executable=True,
        )
        manifest = rehash(
            replace(DEFAULT_ARCHITECTURE_MANIFEST, hysteresis_contract=hysteresis)
        )
        self.assertIn(
            "executable_hysteresis_contract_is_incomplete",
            validate_architecture_contract(manifest).errors,
        )

    def test_repeated_transition_without_new_evidence_fails(self) -> None:
        hysteresis = replace(
            DEFAULT_ARCHITECTURE_MANIFEST.hysteresis_contract,
            requires_new_evidence=False,
        )
        manifest = rehash(
            replace(DEFAULT_ARCHITECTURE_MANIFEST, hysteresis_contract=hysteresis)
        )
        self.assertIn(
            "repeated_transitions_may_not_reuse_evidence",
            validate_architecture_contract(manifest).errors,
        )

    def test_cycle_detection_requirement_is_preserved(self) -> None:
        self.assertIn(
            "hysteresis_repeated_cycle_window",
            DEFAULT_ARCHITECTURE_MANIFEST.unresolved_threshold_ids,
        )
        self.assertTrue(
            DEFAULT_ARCHITECTURE_MANIFEST.hysteresis_contract.requires_new_evidence
        )

    def test_unresolved_hysteresis_keeps_execution_unauthorized(self) -> None:
        report = validate_architecture_contract(DEFAULT_ARCHITECTURE_MANIFEST)
        self.assertFalse(report.execution_authorized)


class ManifestTests(unittest.TestCase):
    def test_canonical_manifest_hash_recomputes(self) -> None:
        document = architecture_manifest_document()
        self.assertEqual(validate_architecture_manifest_document(document), ())

    def test_payload_mutation_invalidates_hash(self) -> None:
        document = architecture_manifest_document()
        document["execution_authorization_status"] = "authorized"
        errors = validate_architecture_manifest_document(document)
        self.assertIn("canonical_manifest_hash_mismatch", errors)
        self.assertIn("execution_status_mismatch", errors)

    def test_volatile_metadata_is_excluded(self) -> None:
        document = architecture_manifest_document()
        self.assertNotIn("generated_at", document)
        self.assertNotIn("execution_timestamp", document)
        self.assertNotIn("computer_name", document)

    def test_execution_status_remains_not_authorized(self) -> None:
        self.assertEqual(
            DEFAULT_ARCHITECTURE_MANIFEST.execution_authorization_status,
            EXECUTION_NOT_AUTHORIZED,
        )

    def test_unresolved_thresholds_and_action_laws_are_listed(self) -> None:
        self.assertTrue(DEFAULT_ARCHITECTURE_MANIFEST.unresolved_threshold_ids)
        self.assertEqual(
            DEFAULT_ARCHITECTURE_MANIFEST.action_law_status,
            RecoveryEvidenceSupport.NOT_YET_SPECIFIED,
        )

    def test_future_experiment_cannot_be_marked_ready(self) -> None:
        report = validate_architecture_contract(DEFAULT_ARCHITECTURE_MANIFEST)
        self.assertTrue(report.structurally_valid)
        self.assertFalse(report.execution_authorized)

    def test_evidence_traceability_is_deterministic(self) -> None:
        first = canonical_json_bytes(evidence_traceability_document())
        second = canonical_json_bytes(evidence_traceability_document())
        self.assertEqual(first, second)
        document = evidence_traceability_document()
        self.assertGreaterEqual(len(document["items"]), 70)

    def test_checked_in_json_artifacts_match_generators(self) -> None:
        architecture = json.loads(
            (ROOT / "analysis/staged_recovery_architecture_v0/architecture_manifest.json")
            .read_text(encoding="utf-8")
        )
        traceability = json.loads(
            (ROOT / "analysis/staged_recovery_architecture_v0/evidence_traceability.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(architecture, architecture_manifest_document())
        self.assertEqual(traceability, evidence_traceability_document())

    def test_required_document_status_and_scope_are_present(self) -> None:
        architecture = (
            ROOT / "docs/architecture/staged_recovery_architecture_v0.md"
        ).read_text(encoding="utf-8")
        summary = (
            ROOT / "analysis/staged_recovery_architecture_v0/summary.md"
        ).read_text(encoding="utf-8")
        plan = (
            ROOT / "docs/experiments/staged_recovery_minimal_experiment_plan_v0.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "Architecture and evidence contract frozen; staged recovery execution not implemented.",
            architecture,
        )
        self.assertIn("This document defines a staged recovery architecture", architecture)
        self.assertIn("no new trajectory executed", summary)
        self.assertIn("Experiment design proposal only; execution not authorized.", plan)

    def test_markdown_artifacts_are_stable_read_only_bytes(self) -> None:
        paths = (
            ROOT / "docs/architecture/staged_recovery_architecture_v0.md",
            ROOT / "docs/experiments/staged_recovery_minimal_experiment_plan_v0.md",
            ROOT / "analysis/staged_recovery_architecture_v0/summary.md",
        )
        first = tuple(path.read_bytes() for path in paths)
        second = tuple(path.read_bytes() for path in paths)
        self.assertEqual(first, second)


class RepositorySafetyTests(unittest.TestCase):
    def test_contract_imports_no_rollout_controller_or_simulator(self) -> None:
        source = (ROOT / "runtime_assurance/staged_recovery_contract.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertFalse(
            any(
                name.startswith(("simulator", "controller", "scripts", "envs"))
                or "runner" in name
                for name in imports
            ),
            imports,
        )

    def test_no_rollout_module_loaded_by_contract_test(self) -> None:
        code = (
            "import sys; "
            "import runtime_assurance.staged_recovery_contract; "
            "forbidden=('runtime_assurance.recovery_experiment_runner',"
            "'runtime_assurance.recovery_branch_runner',"
            "'simulator.phase34_35_transition'); "
            "raise SystemExit(1 if any(name in sys.modules for name in forbidden) else 0)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_source_recovery_artifact_hashes_match_contract(self) -> None:
        for relative, expected in contract.SOURCE_ARTIFACT_HASHES:
            path = ROOT / relative
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_generation_is_deterministic_and_outside_frozen_directory(self) -> None:
        manifest_a = canonical_json_bytes(architecture_manifest_document())
        manifest_b = canonical_json_bytes(architecture_manifest_document())
        trace_a = canonical_json_bytes(evidence_traceability_document())
        trace_b = canonical_json_bytes(evidence_traceability_document())
        self.assertEqual(manifest_a, manifest_b)
        self.assertEqual(trace_a, trace_b)
        self.assertNotIn(b'"output_directory"', manifest_a)
        self.assertNotIn(b'"execution_authorization_status":"authorized"', manifest_a)


if __name__ == "__main__":
    unittest.main()
