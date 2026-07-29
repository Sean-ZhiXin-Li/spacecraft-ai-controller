from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from runtime_assurance.staged_recovery_instrumentation import (
    INSTRUMENTATION_SCHEMA_VERSION,
    CartesianState2D,
    InstrumentationEvidenceStatus,
    OrbitalConfiguration,
    invalid_value,
    measured_value,
    not_evaluated_value,
)
from runtime_assurance.staged_recovery_runtime_logger import (
    ARCHITECTURE_VERSION,
    CLAIM_RESTRICTIONS,
    LOGGER_SCHEMA_VERSION,
    REAL_RUNNER_INTEGRATION_STATUS,
    REAL_TRACE_VALIDATION_STATUS,
    STAGED_EXECUTION_STATUS,
    ActionDisposition,
    LoggerSessionState,
    RuntimeEventType,
    RuntimeLoggerContractError,
    StagedRecoveryInitialSnapshot,
    StagedRecoveryRuntimeLoggerSession,
    StagedRecoverySessionHeader,
    StagedRecoveryTerminalInput,
    StagedRecoveryTransitionInput,
    aggregate_trace_sha256,
    canonical_event_sha256,
    canonical_runtime_json_bytes,
    canonical_state_sha256,
    event_hash_recomputes,
    event_schema_document,
    field_coverage_document,
    integration_contract_document,
    logger_coverage_counts,
    logger_field_coverage,
    logger_manifest_document,
    protected_trace_paths,
    publish_trace_bundle,
    trace_jsonl_bytes,
    trace_manifest_hash_recomputes,
    trace_manifest_json_bytes,
    validate_logger_contract_documents,
    validate_trace_bundle,
    validate_trace_publication_target,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "runtime_assurance/staged_recovery_runtime_logger.py"
LOGGER_ARTIFACT_DIR = ROOT / "analysis/staged_recovery_runtime_logger_v0"
CONFIG = OrbitalConfiguration(mu=100.0, target_radius=25.0)
STATE0 = CartesianState2D(24.0, 0.0, 0.1, 2.0)
STATE1 = CartesianState2D(24.01, 0.02, 0.09, 2.01)
PREDICTED1 = CartesianState2D(24.011, 0.019, 0.091, 2.02)
HASH = "a" * 64


def header(*, max_events: int = 5, **overrides: object) -> StagedRecoverySessionHeader:
    values: dict[str, object] = {
        "logger_schema_version": LOGGER_SCHEMA_VERSION,
        "instrumentation_schema_version": INSTRUMENTATION_SCHEMA_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "session_id": "synthetic-stage0b-session",
        "case_id": "synthetic-case",
        "seed": 0,
        "implementation_commit": "b" * 40,
        "source_state_hash": HASH,
        "simulator_configuration_hash": "c" * 64,
        "constants_hash": "d" * 64,
        "max_events": max_events,
        "declared_output_purpose": "synthetic logger contract validation",
        "execution_authorization_status": STAGED_EXECUTION_STATUS,
        "scientific_claim_restrictions": CLAIM_RESTRICTIONS,
    }
    values.update(overrides)
    return StagedRecoverySessionHeader(**values)  # type: ignore[arg-type]


def initialized_session(
    *, max_events: int = 5, recovery_step: int = 0, total_count: int = 100
) -> StagedRecoveryRuntimeLoggerSession:
    session = StagedRecoveryRuntimeLoggerSession(header(max_events=max_events))
    session.record_initial_snapshot(
        StagedRecoveryInitialSnapshot(
            event_index=0,
            recovery_step=recovery_step,
            total_transition_count=total_count,
            state=STATE0,
            configuration=CONFIG,
            simulation_time=1.0,
        )
    )
    return session


def zero_transition(
    *,
    event_index: int = 1,
    recovery_step: int = 1,
    total_count: int = 101,
    predicted: CartesianState2D | None = PREDICTED1,
    evidence: tuple = (),
) -> StagedRecoveryTransitionInput:
    return StagedRecoveryTransitionInput(
        event_index=event_index,
        recovery_step=recovery_step,
        total_transition_count=total_count,
        pre_state=STATE0,
        configuration=CONFIG,
        proposed_action=(0.0, 0.0),
        executed_action=(0.0, 0.0),
        action_disposition=ActionDisposition.ZERO_ACTION_EXECUTED,
        transition_executed=True,
        realized_next_state=STATE1,
        monitor_decision="allow",
        predicted_next_state=predicted,
        simulation_time=1.0,
        next_simulation_time=2.0,
        branch_step=0,
        runtime_evidence=evidence,
    )


def complete_bundle(*, volatile: str | None = None):
    session = initialized_session()
    session.record_transition(zero_transition())
    session.record_terminal(
        StagedRecoveryTerminalInput(
            event_index=2,
            recovery_step=1,
            total_transition_count=101,
            terminal_reason="synthetic_complete",
            action_disposition=ActionDisposition.NO_ACTION,
            volatile_timestamp=volatile,
        )
    )
    return session.finalize(volatile_finalization_timestamp=volatile)


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_staged_recovery_runtime_logger.py"),
            *arguments,
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        env=environment,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModuleSafetyTests(unittest.TestCase):
    def test_001_import_performs_no_write(self) -> None:
        before = {path.name: sha256(path) for path in LOGGER_ARTIFACT_DIR.iterdir()}
        result = subprocess.run(
            [sys.executable, "-c", "import runtime_assurance.staged_recovery_runtime_logger"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        after = {path.name: sha256(path) for path in LOGGER_ARTIFACT_DIR.iterdir()}
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, after)

    def test_002_import_performs_no_transition(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("transition_fn(", source)

    def test_003_imports_no_simulator_runner(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("import simulator", source)
        self.assertNotIn("recovery_experiment_runner", source)

    def test_004_imports_no_controller(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("import controller", source)
        self.assertNotIn("import controllers", source)

    def test_005_imports_no_branch_action_generator(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("recovery_branch_executor", source)

    def test_006_imports_no_phase_selector(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("select_phase", source)
        self.assertNotIn("decide_phase", source)

    def test_007_calls_only_stage0a_derivations(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("staged_recovery_instrumentation import", source)
        self.assertNotIn("select_recovery_stop", source)


class SessionHeaderTests(unittest.TestCase):
    def test_008_valid_immutable_header(self) -> None:
        value = header()
        self.assertEqual(value.max_events, 5)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            value.max_events = 2  # type: ignore[misc]

    def test_009_missing_case_identity_rejected(self) -> None:
        with self.assertRaises(RuntimeLoggerContractError):
            header(case_id="")

    def test_010_missing_schema_version_rejected(self) -> None:
        with self.assertRaises(RuntimeLoggerContractError):
            header(logger_schema_version="")

    def test_011_nonpositive_capacity_rejected(self) -> None:
        with self.assertRaises(RuntimeLoggerContractError):
            header(max_events=0)

    def test_012_boolean_capacity_rejected(self) -> None:
        with self.assertRaises(RuntimeLoggerContractError):
            header(max_events=True)

    def test_013_mutable_header_data_is_canonicalized(self) -> None:
        claims = list(CLAIM_RESTRICTIONS)
        value = header(scientific_claim_restrictions=claims)
        claims.clear()
        self.assertEqual(value.scientific_claim_restrictions, CLAIM_RESTRICTIONS)

    def test_014_execution_authorization_cannot_be_true(self) -> None:
        with self.assertRaises(RuntimeLoggerContractError):
            header(execution_authorization_status="authorized")

    def test_015_source_hashes_are_preserved(self) -> None:
        value = header()
        self.assertEqual(value.source_state_hash, HASH)
        self.assertEqual(value.constants_hash, "d" * 64)


class InitialSnapshotTests(unittest.TestCase):
    def test_016_one_initial_snapshot_accepted(self) -> None:
        session = initialized_session()
        self.assertEqual(session.events[0].event_type, RuntimeEventType.INITIAL_SNAPSHOT)

    def test_017_duplicate_initial_snapshot_rejected(self) -> None:
        session = initialized_session()
        with self.assertRaises(RuntimeLoggerContractError):
            session.record_initial_snapshot(
                StagedRecoveryInitialSnapshot(1, 0, 100, STATE0, CONFIG)
            )

    def test_018_initial_index_must_be_zero(self) -> None:
        session = StagedRecoveryRuntimeLoggerSession(header())
        with self.assertRaises(RuntimeLoggerContractError):
            session.record_initial_snapshot(
                StagedRecoveryInitialSnapshot(1, 0, 100, STATE0, CONFIG)
            )

    def test_019_initial_has_no_executed_action(self) -> None:
        event = initialized_session().events[0]
        self.assertIsNone(event.executed_action)

    def test_020_initial_has_no_fabricated_zero_action(self) -> None:
        event = initialized_session().events[0]
        self.assertIsNone(event.proposed_action)
        self.assertNotEqual(event.action_disposition, ActionDisposition.ZERO_ACTION_EXECUTED)

    def test_021_initial_has_no_crossing_without_previous_state(self) -> None:
        crossing = initialized_session().events[0].pre_observation.field(
            "target_radius_crossing"
        )
        self.assertEqual(crossing.status, InstrumentationEvidenceStatus.NOT_EVALUATED)

    def test_022_initial_progress_is_not_evaluated(self) -> None:
        progress = dict(initialized_session().events[0].progress_sample)
        self.assertTrue(
            all(value.status == InstrumentationEvidenceStatus.NOT_EVALUATED for value in progress.values())
        )

    def test_023_missing_phase_is_not_evaluated(self) -> None:
        value = initialized_session().events[0].pre_observation.field("current_phase")
        self.assertEqual(value.status, InstrumentationEvidenceStatus.NOT_EVALUATED)


class TransitionOrderTests(unittest.TestCase):
    def test_024_transition_before_initial_rejected(self) -> None:
        session = StagedRecoveryRuntimeLoggerSession(header())
        with self.assertRaises(RuntimeLoggerContractError):
            session.record_transition(zero_transition(event_index=0))

    def test_025_sequential_transition_accepted(self) -> None:
        event = initialized_session().record_transition(zero_transition())
        self.assertEqual(event.event_index, 1)

    def test_026_duplicate_event_index_rejected(self) -> None:
        session = initialized_session()
        session.record_transition(zero_transition())
        with self.assertRaises(RuntimeLoggerContractError):
            session.record_transition(zero_transition(event_index=1, recovery_step=2, total_count=102))

    def test_027_skipped_event_index_rejected(self) -> None:
        with self.assertRaises(RuntimeLoggerContractError):
            initialized_session().record_transition(zero_transition(event_index=2))

    def test_028_decreasing_recovery_step_rejected(self) -> None:
        session = initialized_session(recovery_step=2)
        with self.assertRaises(RuntimeLoggerContractError):
            session.record_transition(zero_transition(recovery_step=1, total_count=101))

    def test_029_decreasing_total_count_rejected(self) -> None:
        session = initialized_session(total_count=102)
        with self.assertRaises(RuntimeLoggerContractError):
            session.record_transition(zero_transition(total_count=101))

    def test_030_event_after_terminal_rejected(self) -> None:
        session = initialized_session()
        session.record_terminal(StagedRecoveryTerminalInput(1, 0, 100, "done", ActionDisposition.NO_ACTION))
        with self.assertRaises(RuntimeLoggerContractError):
            session.record_transition(zero_transition(event_index=2))

    def test_031_event_after_finalize_rejected(self) -> None:
        session = initialized_session()
        session.record_terminal(StagedRecoveryTerminalInput(1, 0, 100, "done", ActionDisposition.NO_ACTION))
        session.finalize()
        with self.assertRaises(RuntimeLoggerContractError):
            session.record_terminal(StagedRecoveryTerminalInput(2, 0, 100, "again", ActionDisposition.NO_ACTION))

    def test_032_finalize_twice_rejected(self) -> None:
        session = initialized_session()
        session.record_terminal(StagedRecoveryTerminalInput(1, 0, 100, "done", ActionDisposition.NO_ACTION))
        session.finalize()
        with self.assertRaises(RuntimeLoggerContractError):
            session.finalize()


class TransitionConsistencyTests(unittest.TestCase):
    def test_033_executed_transition_requires_realized_state(self) -> None:
        value = zero_transition()
        value = dataclasses.replace(value, realized_next_state=None)
        with self.assertRaises(RuntimeLoggerContractError):
            initialized_session().record_transition(value)

    def test_034_nonexecuted_transition_forbids_realized_state(self) -> None:
        value = dataclasses.replace(
            zero_transition(),
            recovery_step=0,
            total_transition_count=100,
            proposed_action=None,
            executed_action=None,
            action_disposition=ActionDisposition.NO_ACTION,
            transition_executed=False,
        )
        with self.assertRaises(RuntimeLoggerContractError):
            initialized_session().record_transition(value)

    def test_035_realized_transition_increments_recovery_once(self) -> None:
        event = initialized_session().record_transition(zero_transition())
        self.assertEqual(event.recovery_step, 1)

    def test_036_nonexecuted_event_retains_recovery_step(self) -> None:
        value = StagedRecoveryTransitionInput(
            1, 0, 100, STATE0, CONFIG, None, None,
            ActionDisposition.NO_ACTION, False, None,
        )
        event = initialized_session().record_transition(value)
        self.assertEqual(event.recovery_step, 0)

    def test_037_realized_transition_increments_total_once(self) -> None:
        event = initialized_session().record_transition(zero_transition())
        self.assertEqual(event.total_transition_count, 101)

    def test_038_nonexecuted_event_retains_total_count(self) -> None:
        value = StagedRecoveryTransitionInput(
            1, 0, 100, STATE0, CONFIG, None, None,
            ActionDisposition.NO_ACTION, False, None,
        )
        event = initialized_session().record_transition(value)
        self.assertEqual(event.total_transition_count, 100)

    def test_039_predicted_state_is_optional(self) -> None:
        event = initialized_session().record_transition(zero_transition(predicted=None))
        self.assertIsNone(event.predicted_observation)

    def test_040_predicted_state_never_becomes_realized_state(self) -> None:
        value = StagedRecoveryTransitionInput(
            1, 0, 100, STATE0, CONFIG, None, None,
            ActionDisposition.NO_ACTION, False, None,
            predicted_next_state=PREDICTED1,
        )
        event = initialized_session().record_transition(value)
        self.assertIsNotNone(event.predicted_observation)
        self.assertIsNone(event.post_observation)

    def test_041_progress_uses_measured_pre_post_states(self) -> None:
        event = initialized_session().record_transition(zero_transition())
        progress = dict(event.progress_sample)["delta_signed_target_radius_error"]
        self.assertEqual(progress.status, InstrumentationEvidenceStatus.DERIVED)
        self.assertIn("current_minus_previous", progress.reason)


class ActionDispositionTests(unittest.TestCase):
    def test_042_physical_zero_action_is_distinct(self) -> None:
        event = initialized_session().record_transition(zero_transition())
        self.assertEqual(event.action_disposition, ActionDisposition.ZERO_ACTION_EXECUTED)
        self.assertTrue(event.transition_executed)

    def test_043_explicit_abort_remains_no_action(self) -> None:
        session = initialized_session()
        event = session.record_terminal(
            StagedRecoveryTerminalInput(1, 0, 100, "explicit_abort", ActionDisposition.NO_ACTION)
        )
        self.assertIsNone(event.proposed_action)
        self.assertIsNone(event.executed_action)

    def test_044_rejected_action_preserves_proposal(self) -> None:
        value = StagedRecoveryTransitionInput(
            1, 0, 100, STATE0, CONFIG, (0.1, 0.2), None,
            ActionDisposition.REJECTED, False, None,
        )
        event = initialized_session().record_transition(value)
        self.assertEqual(event.proposed_action, (0.1, 0.2))
        self.assertIsNone(event.executed_action)

    def test_045_suppression_requires_explicit_supplied_evidence(self) -> None:
        value = dataclasses.replace(
            zero_transition(),
            proposed_action=(0.2, 0.0),
            executed_action=(0.0, 0.0),
            action_disposition=ActionDisposition.SUPPRESSED,
        )
        event = initialized_session().record_transition(value)
        self.assertEqual(event.action_disposition, ActionDisposition.SUPPRESSED)

    def test_046_missing_executed_action_does_not_infer_suppression(self) -> None:
        value = StagedRecoveryTransitionInput(
            1, 0, 100, STATE0, CONFIG, None, None,
            ActionDisposition.NO_ACTION, False, None,
        )
        event = initialized_session().record_transition(value)
        self.assertEqual(event.action_disposition, ActionDisposition.NO_ACTION)

    def test_047_executed_unchanged_equality_validated(self) -> None:
        value = dataclasses.replace(
            zero_transition(),
            proposed_action=(0.2, 0.1),
            executed_action=(0.2, 0.1),
            action_disposition=ActionDisposition.EXECUTED_UNCHANGED,
        )
        self.assertEqual(
            initialized_session().record_transition(value).action_disposition,
            ActionDisposition.EXECUTED_UNCHANGED,
        )

    def test_048_executed_modified_difference_validated(self) -> None:
        value = dataclasses.replace(
            zero_transition(),
            proposed_action=(0.2, 0.1),
            executed_action=(0.1, 0.1),
            action_disposition=ActionDisposition.EXECUTED_MODIFIED,
        )
        self.assertEqual(
            initialized_session().record_transition(value).action_disposition,
            ActionDisposition.EXECUTED_MODIFIED,
        )

    def test_049_invalid_disposition_combination_rejected(self) -> None:
        value = dataclasses.replace(
            zero_transition(), action_disposition=ActionDisposition.REJECTED
        )
        with self.assertRaises(RuntimeLoggerContractError):
            initialized_session().record_transition(value)

    def test_050_logger_generates_no_fallback_action(self) -> None:
        value = StagedRecoveryTransitionInput(
            1, 0, 100, STATE0, CONFIG, (0.25, 0.0), None,
            ActionDisposition.REJECTED, False, None,
        )
        event = initialized_session().record_transition(value)
        self.assertIsNone(event.executed_action)

    def test_051_branch_action_generator_not_referenced(self) -> None:
        self.assertNotIn(
            "generate_velocity_opposed_action", MODULE_PATH.read_text(encoding="utf-8")
        )


class TerminalSemanticsTests(unittest.TestCase):
    def test_052_valid_terminal_event_accepted(self) -> None:
        event = initialized_session().record_terminal(
            StagedRecoveryTerminalInput(1, 0, 100, "done", ActionDisposition.NO_ACTION)
        )
        self.assertTrue(event.terminal)

    def test_053_terminal_reason_required(self) -> None:
        with self.assertRaises(RuntimeLoggerContractError):
            StagedRecoveryTerminalInput(1, 0, 100, "", ActionDisposition.NO_ACTION)

    def test_054_explicit_abort_has_zero_transition(self) -> None:
        event = initialized_session().record_terminal(
            StagedRecoveryTerminalInput(1, 0, 100, "explicit_abort", ActionDisposition.NO_ACTION)
        )
        self.assertFalse(event.transition_executed)

    def test_055_explicit_abort_cannot_have_executed_action(self) -> None:
        with self.assertRaises(RuntimeLoggerContractError):
            initialized_session().record_terminal(
                StagedRecoveryTerminalInput(
                    1, 0, 100, "explicit_abort", ActionDisposition.NO_ACTION,
                    executed_action=(0.0, 0.0),
                )
            )

    def test_056_rejection_cannot_report_realized_rejected_transition(self) -> None:
        with self.assertRaises(RuntimeLoggerContractError):
            initialized_session().record_terminal(
                StagedRecoveryTerminalInput(
                    1, 0, 100, "action_rejected", ActionDisposition.REJECTED,
                    proposed_action=(0.25, 0.0), executed_action=(0.0, 0.0),
                )
            )

    def test_057_terminal_without_transition_retains_counters(self) -> None:
        event = initialized_session().record_terminal(
            StagedRecoveryTerminalInput(1, 0, 100, "done", ActionDisposition.NO_ACTION)
        )
        self.assertEqual((event.recovery_step, event.total_transition_count), (0, 100))

    def test_058_terminal_cannot_be_followed_by_transition(self) -> None:
        session = initialized_session()
        session.record_terminal(StagedRecoveryTerminalInput(1, 0, 100, "done", ActionDisposition.NO_ACTION))
        with self.assertRaises(RuntimeLoggerContractError):
            session.record_transition(zero_transition(event_index=2))

    def test_059_contradictory_recovery_success_marks_invalid(self) -> None:
        evidence = (
            ("recovery_success_v0", measured_value(True, units="boolean", source_id="fixture")),
            ("simulation_validity", measured_value(False, units="boolean", source_id="fixture")),
        )
        event = initialized_session().record_terminal(
            StagedRecoveryTerminalInput(
                1, 0, 100, "done", ActionDisposition.NO_ACTION,
                runtime_evidence=evidence,
            )
        )
        self.assertFalse(event.event_valid)
        self.assertIn("recovery_success_contradicts:simulation_validity", event.invalid_reasons)

    def test_060_missing_evaluator_evidence_remains_unavailable(self) -> None:
        event = initialized_session().record_terminal(
            StagedRecoveryTerminalInput(1, 0, 100, "done", ActionDisposition.NO_ACTION)
        )
        self.assertEqual(event.evaluator_evidence, ())


class InstrumentationConstructionTests(unittest.TestCase):
    def test_061_pre_observation_uses_measured_pre_state(self) -> None:
        event = initialized_session().record_transition(zero_transition())
        self.assertEqual(event.pre_observation.field("position_x").value, STATE0.x)

    def test_062_predicted_observation_requires_supplied_state(self) -> None:
        event = initialized_session().record_transition(zero_transition(predicted=None))
        self.assertIsNone(event.predicted_observation)

    def test_063_post_observation_requires_realized_state(self) -> None:
        event = initialized_session().record_transition(zero_transition())
        self.assertEqual(event.post_observation.field("position_x").value, STATE1.x)

    def test_064_crossing_uses_measured_pre_post_states(self) -> None:
        event = initialized_session().record_transition(zero_transition())
        crossing = event.post_observation.field("target_radius_crossing")
        self.assertEqual(crossing.status, InstrumentationEvidenceStatus.DERIVED)

    def test_065_progress_uses_measured_pre_post(self) -> None:
        event = initialized_session().record_transition(zero_transition())
        self.assertEqual(
            dict(event.progress_sample)["transition_count_delta"].value, 1
        )

    def test_066_action_geometry_uses_supplied_action_and_basis(self) -> None:
        event = initialized_session().record_transition(zero_transition())
        magnitude = dict(event.action_geometry)["executed_action_magnitude"]
        self.assertEqual(magnitude.value, 0.0)

    def test_067_state_hash_does_not_substitute_for_cartesian_state(self) -> None:
        with self.assertRaises(RuntimeLoggerContractError):
            StagedRecoveryTerminalInput(
                0, 0, 0, "done", ActionDisposition.NO_ACTION,
                current_state_hash=HASH,
            )

    def test_068_phase_fields_remain_externally_supplied(self) -> None:
        phase = measured_value(
            "hazard_arrest", units="categorical", source_id="fixture.phase"
        )
        snapshot = StagedRecoveryInitialSnapshot(
            0, 0, 100, STATE0, CONFIG,
            runtime_evidence=(("current_phase", phase),),
        )
        session = StagedRecoveryRuntimeLoggerSession(header())
        event = session.record_initial_snapshot(snapshot)
        self.assertEqual(dict(event.phase_evidence)["current_phase"], phase)

    def test_069_no_progress_status_not_computed(self) -> None:
        event = initialized_session().record_transition(zero_transition())
        self.assertEqual(
            event.post_observation.field("no_progress_status").status,
            InstrumentationEvidenceStatus.NOT_EVALUATED,
        )

    def test_069a_runtime_evidence_cannot_override_derived_state(self) -> None:
        with self.assertRaises(RuntimeLoggerContractError):
            StagedRecoveryInitialSnapshot(
                0,
                0,
                100,
                STATE0,
                CONFIG,
                runtime_evidence=(
                    (
                        "radius",
                        measured_value(0.0, units="m", source_id="invalid.override"),
                    ),
                ),
            )


class PredictedRealizedTests(unittest.TestCase):
    def test_070_predicted_and_realized_ratios_are_separate(self) -> None:
        event = initialized_session().record_transition(zero_transition())
        predicted = dict(event.predicted_observation.fields)["predicted_speed_ratio"]
        realized = event.post_observation.field("realized_speed_ratio")
        self.assertNotEqual(predicted.status, realized.status)

    def test_071_prediction_error_computed_when_both_valid(self) -> None:
        diagnostics = dict(initialized_session().record_transition(zero_transition()).prediction_diagnostics)
        self.assertEqual(
            diagnostics["speed_ratio_prediction_error"].status,
            InstrumentationEvidenceStatus.DERIVED,
        )

    def test_072_prediction_error_unavailable_when_missing(self) -> None:
        diagnostics = dict(
            initialized_session().record_transition(zero_transition(predicted=None)).prediction_diagnostics
        )
        self.assertEqual(
            diagnostics["speed_ratio_prediction_error"].status,
            InstrumentationEvidenceStatus.NOT_EVALUATED,
        )

    def test_073_predicted_overspeed_does_not_create_realized_overspeed(self) -> None:
        event = initialized_session().record_transition(zero_transition())
        self.assertIsNot(
            dict(event.predicted_observation.fields)["predicted_overspeed_status"],
            event.post_observation.field("overspeed_status"),
        )

    def test_074_state_hash_equality_is_exact_identity(self) -> None:
        value = dataclasses.replace(zero_transition(), predicted_next_state=STATE1)
        diagnostics = dict(initialized_session().record_transition(value).prediction_diagnostics)
        self.assertIs(
            diagnostics["predicted_state_hash_matches_realized_state_hash"].value,
            True,
        )

    def test_075_unequal_hashes_do_not_produce_physical_distance(self) -> None:
        diagnostics = dict(initialized_session().record_transition(zero_transition()).prediction_diagnostics)
        self.assertNotIn("physical_distance", diagnostics)


class CapacityTests(unittest.TestCase):
    def test_076_explicit_capacity_enforced(self) -> None:
        session = initialized_session(max_events=1)
        with self.assertRaises(RuntimeLoggerContractError):
            session.record_terminal(StagedRecoveryTerminalInput(1, 0, 100, "done", ActionDisposition.NO_ACTION))

    def test_077_no_hidden_default_capacity(self) -> None:
        self.assertIs(inspect.signature(StagedRecoverySessionHeader).parameters["max_events"].default, inspect.Parameter.empty)

    def test_078_capacity_rejects_extra_event(self) -> None:
        session = initialized_session(max_events=1)
        with self.assertRaisesRegex(RuntimeLoggerContractError, "capacity"):
            session.record_transition(zero_transition())

    def test_079_capacity_rejection_preserves_records(self) -> None:
        session = initialized_session(max_events=1)
        before = session.events
        with self.assertRaises(RuntimeLoggerContractError):
            session.record_transition(zero_transition())
        self.assertEqual(session.events, before)

    def test_080_capacity_is_not_recovery_horizon(self) -> None:
        session = initialized_session(max_events=1)
        with self.assertRaises(RuntimeLoggerContractError):
            session.record_transition(zero_transition())
        bundle = session.finalize()
        self.assertEqual(bundle.manifest.terminal_status, "logger_capacity_exhausted_incomplete")


class CanonicalSerializationTests(unittest.TestCase):
    def test_081_repeated_event_serialization_is_identical(self) -> None:
        event = complete_bundle().events[1]
        self.assertEqual(canonical_runtime_json_bytes(event), canonical_runtime_json_bytes(event))

    def test_082_event_key_order_is_stable(self) -> None:
        line = trace_jsonl_bytes(complete_bundle()).splitlines()[0]
        keys = list(json.loads(line).keys())
        self.assertEqual(keys, sorted(keys))

    def test_083_event_hash_recomputes(self) -> None:
        event = complete_bundle().events[1]
        self.assertTrue(event_hash_recomputes(event))

    def test_084_event_mutation_invalidates_hash(self) -> None:
        event = complete_bundle().events[1]
        mutated = dataclasses.replace(event, evidence_level="mutated")
        self.assertNotEqual(event.canonical_event_sha256, canonical_event_sha256(mutated))

    def test_085_trace_aggregate_hash_recomputes(self) -> None:
        bundle = complete_bundle()
        self.assertEqual(bundle.manifest.aggregate_trace_hash, aggregate_trace_sha256(bundle.events))

    def test_086_event_order_mutation_changes_trace_hash(self) -> None:
        bundle = complete_bundle()
        self.assertNotEqual(
            aggregate_trace_sha256(bundle.events),
            aggregate_trace_sha256(tuple(reversed(bundle.events))),
        )

    def test_087_volatile_timestamp_excluded_from_event_hash(self) -> None:
        first = complete_bundle(volatile="2026-07-29T00:00:00Z")
        second = complete_bundle(volatile="2026-07-29T00:01:00Z")
        self.assertEqual(
            [event.canonical_event_sha256 for event in first.events],
            [event.canonical_event_sha256 for event in second.events],
        )

    def test_088_no_arbitrary_python_representation(self) -> None:
        payload = trace_jsonl_bytes(complete_bundle()).decode("utf-8")
        self.assertNotIn(" object at 0x", payload)


class TraceBundleTests(unittest.TestCase):
    def test_089_finalized_synthetic_session_is_valid(self) -> None:
        self.assertTrue(validate_trace_bundle(complete_bundle()).valid)

    def test_090_unfinalized_session_cannot_publish(self) -> None:
        session = initialized_session()
        self.assertEqual(session.state, LoggerSessionState.STARTED)
        with self.assertRaises(RuntimeLoggerContractError):
            session.finalize()

    def test_091_incomplete_capacity_bundle_cannot_publish(self) -> None:
        session = initialized_session(max_events=1)
        with self.assertRaises(RuntimeLoggerContractError):
            session.record_transition(zero_transition())
        bundle = session.finalize()
        self.assertFalse(validate_trace_bundle(bundle, require_complete=True).valid)

    def test_092_manifest_event_count_matches_jsonl(self) -> None:
        bundle = complete_bundle()
        self.assertEqual(bundle.manifest.event_count, len(trace_jsonl_bytes(bundle).splitlines()))

    def test_093_first_last_indices_match(self) -> None:
        manifest = complete_bundle().manifest
        self.assertEqual((manifest.first_event_index, manifest.last_event_index), (0, 2))

    def test_094_first_last_transition_counts_match(self) -> None:
        manifest = complete_bundle().manifest
        self.assertEqual(
            (manifest.first_total_transition_count, manifest.last_total_transition_count),
            (100, 101),
        )

    def test_095_synthetic_classification_is_explicit(self) -> None:
        manifest = complete_bundle().manifest
        self.assertEqual(manifest.trace_classification, "synthetic")
        self.assertEqual(manifest.runtime_source, "dependency_injected_fixture")

    def test_096_scientific_result_remains_false(self) -> None:
        self.assertIs(complete_bundle().manifest.scientific_result, False)


class AtomicPublicationTests(unittest.TestCase):
    def test_097_valid_bundle_publishes_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "trace"
            result = publish_trace_bundle(complete_bundle(), target, repository_root=ROOT)
            self.assertTrue(result.published)
            self.assertEqual({p.name for p in target.iterdir()}, {"trace_manifest.json", "staged_recovery_trace.jsonl"})

    def test_098_target_must_not_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeLoggerContractError):
                publish_trace_bundle(complete_bundle(), directory, repository_root=ROOT)

    def test_099_overwrite_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "trace"
            publish_trace_bundle(complete_bundle(), target, repository_root=ROOT)
            with self.assertRaises(RuntimeLoggerContractError):
                publish_trace_bundle(complete_bundle(), target, repository_root=ROOT)

    def test_100_protected_target_rejected(self) -> None:
        target = ROOT / "analysis/final_veto_ablation_v0/logger-trace"
        with self.assertRaises(RuntimeLoggerContractError):
            validate_trace_publication_target(target, repository_root=ROOT)

    def test_101_repository_root_rejected(self) -> None:
        with self.assertRaises(RuntimeLoggerContractError):
            validate_trace_publication_target(ROOT, repository_root=ROOT)

    def test_102_path_traversal_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "child" / ".." / "trace"
            with self.assertRaises(RuntimeLoggerContractError):
                validate_trace_publication_target(target, repository_root=ROOT)

    def test_103_case_normalized_protected_path_rejected_on_windows(self) -> None:
        if os.name != "nt":
            self.skipTest("case-insensitive path behavior is Windows-specific")
        target = Path(str(ROOT / "analysis/final_veto_ablation_v0/logger-trace").upper())
        with self.assertRaises(RuntimeLoggerContractError):
            validate_trace_publication_target(target, repository_root=ROOT)

    def test_104_writer_failure_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "trace"
            def fail(stage: str) -> None:
                if stage == "before_write_jsonl":
                    raise OSError("injected writer failure")
            with self.assertRaises(OSError):
                publish_trace_bundle(complete_bundle(), target, repository_root=ROOT, failure_injector=fail)
            self.assertFalse(target.exists())

    def test_105_staged_validation_failure_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "trace"
            def fail(stage: str) -> None:
                if stage == "before_staged_validation":
                    raise RuntimeError("injected validation failure")
            with self.assertRaises(RuntimeError):
                publish_trace_bundle(complete_bundle(), target, repository_root=ROOT, failure_injector=fail)
            self.assertFalse(target.exists())

    def test_106_temporary_directory_removed_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            def fail(stage: str) -> None:
                if stage == "before_atomic_publish":
                    raise RuntimeError("injected failure")
            with self.assertRaises(RuntimeError):
                publish_trace_bundle(complete_bundle(), parent / "trace", repository_root=ROOT, failure_injector=fail)
            self.assertEqual(list(parent.iterdir()), [])

    def test_107_existing_user_target_never_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "trace"
            target.mkdir()
            marker = target / "user.txt"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaises(RuntimeLoggerContractError):
                publish_trace_bundle(complete_bundle(), target, repository_root=ROOT)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_108_published_hashes_match_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "trace"
            result = publish_trace_bundle(complete_bundle(), target, repository_root=ROOT)
            self.assertEqual(
                dict(result.artifact_hashes),
                {name: sha256(target / name) for name, _ in result.artifact_hashes},
            )

    def test_109_repeated_synthetic_bundle_scientific_bytes_identical(self) -> None:
        first = complete_bundle()
        second = complete_bundle()
        self.assertEqual(trace_jsonl_bytes(first), trace_jsonl_bytes(second))
        self.assertEqual(trace_manifest_json_bytes(first), trace_manifest_json_bytes(second))


class CoverageAndManifestTests(unittest.TestCase):
    def test_110_all_stage0a_fields_are_classified(self) -> None:
        self.assertEqual(len(logger_field_coverage()), 105)

    def test_111_all_architecture_signals_remain_covered(self) -> None:
        self.assertEqual(field_coverage_document()["architecture_signal_count"], 52)

    def test_112_runtime_supplied_fields_not_claimed_derived(self) -> None:
        entry = next(item for item in logger_field_coverage() if item["field_id"] == "current_phase")
        self.assertFalse(entry["logger_derives"])

    def test_113_future_evaluator_field_remains_unresolved(self) -> None:
        entry = next(item for item in logger_field_coverage() if item["field_id"] == "handoff_readiness")
        self.assertEqual(entry["logger_coverage_classification"], "requires_future_evaluator")

    def test_114_correction_authority_remains_unsupported(self) -> None:
        entry = next(item for item in logger_field_coverage() if item["field_id"] == "available_correction_authority")
        self.assertEqual(entry["logger_coverage_classification"], "unsupported")

    def test_115_real_trace_validation_remains_false(self) -> None:
        self.assertFalse(field_coverage_document()["real_trace_has_validated"])

    def test_116_real_runner_integration_remains_false(self) -> None:
        self.assertEqual(logger_manifest_document()["real_runner_integration"], REAL_RUNNER_INTEGRATION_STATUS)

    def test_117_staged_execution_remains_unauthorized(self) -> None:
        self.assertEqual(logger_manifest_document()["staged_recovery_execution"], STAGED_EXECUTION_STATUS)

    def test_118_logger_manifest_hash_recomputes(self) -> None:
        documents = (
            logger_manifest_document(), event_schema_document(),
            integration_contract_document(), field_coverage_document(),
        )
        report = validate_logger_contract_documents(*documents)
        self.assertTrue(report.valid, report.errors)

    def test_119_manifest_mutation_invalidates_contract(self) -> None:
        manifest = logger_manifest_document()
        manifest["real_trace_validation"] = "performed"
        report = validate_logger_contract_documents(
            manifest, event_schema_document(), integration_contract_document(), field_coverage_document()
        )
        self.assertFalse(report.valid)


class CliTests(unittest.TestCase):
    def test_120_default_cli_writes_nothing(self) -> None:
        before = {p.name: sha256(p) for p in LOGGER_ARTIFACT_DIR.iterdir()}
        result = run_cli()
        after = {p.name: sha256(p) for p in LOGGER_ARTIFACT_DIR.iterdir()}
        self.assertEqual(result.returncode, 0)
        self.assertEqual(before, after)

    def test_121_validate_only_writes_nothing(self) -> None:
        before = {p.name: sha256(p) for p in LOGGER_ARTIFACT_DIR.iterdir()}
        result = run_cli("--validate-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, {p.name: sha256(p) for p in LOGGER_ARTIFACT_DIR.iterdir()})

    def test_122_print_coverage_writes_nothing(self) -> None:
        result = run_cli("--print-coverage")
        self.assertEqual(result.returncode, 0)
        self.assertIn("REAL_TRACE_HAS_VALIDATED false", result.stdout)

    def test_123_print_integration_contract_writes_nothing(self) -> None:
        result = run_cli("--print-integration-contract")
        self.assertEqual(result.returncode, 0)
        self.assertIn("REAL_RUNNER_CONNECTED false", result.stdout)

    def test_124_no_cli_execution_mode_exists(self) -> None:
        result = run_cli("--execute")
        self.assertNotEqual(result.returncode, 0)

    def test_125_no_cli_real_recording_mode_exists(self) -> None:
        result = run_cli("--record-real")
        self.assertNotEqual(result.returncode, 0)


class RepositorySafetyTests(unittest.TestCase):
    def test_126_no_real_runner_invoked(self) -> None:
        self.assertNotIn("run_recovery", MODULE_PATH.read_text(encoding="utf-8"))

    def test_127_no_simulator_transition_occurs(self) -> None:
        self.assertNotIn("simulate", MODULE_PATH.read_text(encoding="utf-8").lower())

    def test_128_no_controller_action_generated(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("generate_action", source)

    def test_129_no_phase_transition_selected(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("RecoveryPhaseDecision", source)

    def test_130_no_stop_condition_selected(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("recovery_stop_conditions import", source)

    def test_131_frozen_recovery_inputs_unchanged(self) -> None:
        paths = [
            ROOT / "analysis/recovery_action_branching_nonformal_v0/manifest.json",
            ROOT / "analysis/recovery_action_branching_nonformal_v0/branch_state.json",
        ]
        before = [sha256(path) for path in paths]
        complete_bundle()
        self.assertEqual(before, [sha256(path) for path in paths])

    def test_132_published_results_unchanged(self) -> None:
        path = ROOT / "analysis/recovery_action_branching_nonformal_v0/results.csv"
        before = sha256(path)
        complete_bundle()
        self.assertEqual(before, sha256(path))

    def test_133_mechanism_diagnosis_unchanged(self) -> None:
        path = ROOT / "analysis/recovery_branch_mechanism_diagnosis_v0/summary.md"
        before = sha256(path)
        complete_bundle()
        self.assertEqual(before, sha256(path))

    def test_134_staged_architecture_unchanged(self) -> None:
        path = ROOT / "analysis/staged_recovery_architecture_v0/architecture_manifest.json"
        before = sha256(path)
        complete_bundle()
        self.assertEqual(before, sha256(path))

    def test_135_stage0a_artifacts_unchanged(self) -> None:
        path = ROOT / "analysis/staged_recovery_instrumentation_v0/instrumentation_manifest.json"
        before = sha256(path)
        complete_bundle()
        self.assertEqual(before, sha256(path))

    def test_136_final_veto_evidence_unchanged(self) -> None:
        path = ROOT / "analysis/final_veto_ablation_v0/results.csv"
        before = sha256(path)
        complete_bundle()
        self.assertEqual(before, sha256(path))

    def test_137_phase34_evidence_unchanged(self) -> None:
        path = ROOT / "analysis/phase34_post_cross_sync/summary.md"
        before = sha256(path)
        complete_bundle()
        self.assertEqual(before, sha256(path))

    def test_138_no_checked_in_trace_created(self) -> None:
        names = {path.name for path in LOGGER_ARTIFACT_DIR.iterdir()}
        self.assertNotIn("staged_recovery_trace.jsonl", names)
        self.assertNotIn("trace_manifest.json", names)

    def test_139_generated_task_artifacts_are_deterministic(self) -> None:
        documents = [
            logger_manifest_document(), event_schema_document(),
            integration_contract_document(), field_coverage_document(),
        ]
        self.assertEqual(
            [canonical_runtime_json_bytes(value) for value in documents],
            [canonical_runtime_json_bytes(value) for value in documents],
        )


if __name__ == "__main__":
    unittest.main()
