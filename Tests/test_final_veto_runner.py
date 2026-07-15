from __future__ import annotations

import contextlib
import io
import json
import math
import subprocess
import sys
import unittest
from dataclasses import replace
from typing import Mapping
from unittest import mock

from runtime_assurance.final_veto_runner_types import (
    ActionInterceptionResult,
    PostTransitionObservation,
    PreTransitionActionContext,
    RolloutCaseContext,
)
from scripts import explicit_controller_phase34_post_cross_sync as phase34
from scripts import explicit_controller_phase35_crossing_basin_expansion as phase35
from scripts.run_final_veto_ablation import (
    ArmExecutionError,
    ArmHookRecorder,
    FORMAL_ARTIFACT_NAMES,
    FORMAL_OUTPUT_DIRECTORY,
    FORMAL_PREFLIGHT_TEST_MODULES,
    RunnerContractError,
    PROJECT_ROOT,
    build_pair_record,
    build_planned_jobs,
    canonical_sha256,
    load_frozen_manifest,
    main,
    publication_readiness_errors,
    select_smoke_jobs,
    validate_planned_jobs,
)
from simulator.phase34_35_transition import (
    CartesianState2D,
    NormalizedAction2D,
    Phase3435TransitionResult,
)
from Tests.test_final_veto_transition import (
    PHASE34_REFERENCE_JSON,
    PHASE35_REFERENCE_JSON,
)


class StopAfterObservation(RuntimeError):
    pass


def identity_hook(context: PreTransitionActionContext) -> ActionInterceptionResult:
    return ActionInterceptionResult(
        nominal_action=context.nominal_action,
        executed_action=context.nominal_action,
        intervention_applied=False,
    )


def sample_arm_record(job, arm_id: str, **overrides):
    monitor_on = arm_id == "monitor_on"
    record = {
        "experiment_id": job.experiment_id,
        "run_id": f"{job.paired_run_id}__{arm_id}",
        "paired_run_id": job.paired_run_id,
        "case_id": job.case_id,
        "subset_id": job.subset_id,
        "seed": job.seed,
        "case_config_hash": job.case_config_hash,
        "controller_id": job.controller_id,
        "r0_over_target": job.r0_over_target,
        "initial_velocity_angle_deg": job.initial_velocity_angle_deg,
        "thrust_scale": job.thrust_scale,
        "arm_id": arm_id,
        "invalid_simulation": False,
        "overspeed": False,
        "crossed_target_radius": True,
        "recoverable_crossing": True,
        "final_simulator_success": True,
        "monitor_evaluation_count": 10 if monitor_on else 0,
        "allow_count": 9 if monitor_on else 0,
        "veto_count": 1 if monitor_on else 0,
        "fallback_count": 1 if monitor_on else 0,
        "false_negative_count": 0,
        "fallback_failure_count": 0,
        "steps": 100,
        "is_formal_experiment": False,
    }
    record.update(overrides)
    return record


class PairedPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_frozen_manifest()
        cls.jobs = build_planned_jobs(cls.manifest)

    def test_manifest_expands_to_exactly_26_jobs(self) -> None:
        self.assertEqual(len(self.jobs), 26)

    def test_exactly_13_paired_ids(self) -> None:
        self.assertEqual(len({job.paired_run_id for job in self.jobs}), 13)

    def test_each_pair_has_one_off_and_one_on_arm(self) -> None:
        for pair_id in {job.paired_run_id for job in self.jobs}:
            arms = {job.arm_id for job in self.jobs if job.paired_run_id == pair_id}
            self.assertEqual(arms, {"monitor_off", "monitor_on"})

    def test_preservation_contributes_16_jobs(self) -> None:
        self.assertEqual(sum(job.known_phase34_recoverable_case for job in self.jobs), 16)

    def test_stress_contributes_10_jobs(self) -> None:
        self.assertEqual(sum(not job.known_phase34_recoverable_case for job in self.jobs), 10)

    def test_job_ordering_is_deterministic(self) -> None:
        second = build_planned_jobs(self.manifest)
        self.assertEqual(self.jobs, second)

    def test_stable_hashes_reproduce(self) -> None:
        payload = {"b": 2, "a": [1, 3]}
        self.assertEqual(canonical_sha256(payload), canonical_sha256(payload))
        self.assertEqual(
            [job.run_config_hash for job in self.jobs],
            [job.run_config_hash for job in build_planned_jobs(self.manifest)],
        )

    def test_arm_id_does_not_change_shared_case_hash(self) -> None:
        first_pair = self.jobs[:2]
        self.assertNotEqual(first_pair[0].arm_id, first_pair[1].arm_id)
        self.assertEqual(first_pair[0].case_config_hash, first_pair[1].case_config_hash)
        self.assertNotEqual(first_pair[0].run_config_hash, first_pair[1].run_config_hash)

    def test_different_case_parameters_change_case_hash(self) -> None:
        hashes = {
            job.case_config_hash
            for job in self.jobs
            if job.arm_id == "monitor_off"
        }
        self.assertEqual(len(hashes), 13)

    def test_plan_mode_does_not_execute_simulation(self) -> None:
        with mock.patch(
            "scripts.run_final_veto_ablation.execute_jobs_to_directory"
        ) as execute:
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(["--plan"])
        self.assertEqual(result, 0)
        execute.assert_not_called()

    def test_default_invocation_does_not_start_formal_execution(self) -> None:
        with mock.patch(
            "scripts.run_final_veto_ablation.execute_jobs_to_directory"
        ) as execute:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main([])
        self.assertEqual(result, 0)
        self.assertIn("simulation_started=false", output.getvalue())
        execute.assert_not_called()

    def git_path_is_ignored(self, relative_path) -> bool:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={PROJECT_ROOT.as_posix()}",
                "check-ignore",
                "-q",
                "--no-index",
                relative_path.as_posix(),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            return True
        if completed.returncode == 1:
            return False
        self.fail(
            f"git check-ignore failed for {relative_path.as_posix()} "
            f"with exit {completed.returncode}: {completed.stderr.strip()}"
        )

    def test_published_formal_paths_are_publishable_and_validation_is_read_only(self) -> None:
        formal_paths = {
            FORMAL_OUTPUT_DIRECTORY / filename
            for filename in FORMAL_ARTIFACT_NAMES.values()
        }
        self.assertEqual(
            {path.name for path in formal_paths},
            {
                "results.csv",
                "paired_results.csv",
                "comparison.png",
                "decision_log.jsonl",
                "summary.md",
            },
        )
        for relative_path in formal_paths:
            with self.subTest(path=relative_path.as_posix()):
                self.assertFalse(self.git_path_is_ignored(relative_path))

        errors = publication_readiness_errors(self.manifest, PROJECT_ROOT)
        ignored = [error for error in errors if "ignored by .gitignore" in error]
        self.assertEqual(len(ignored), 0)
        existing_names = {
            path.name
            for path in (PROJECT_ROOT / FORMAL_OUTPUT_DIRECTORY).iterdir()
            if path.is_file() and path.name in FORMAL_ARTIFACT_NAMES.values()
        }
        existing_errors = [error for error in errors if "already exists" in error]
        self.assertEqual(len(existing_errors), len(existing_names))
        for filename in existing_names:
            self.assertTrue(any(filename in error for error in existing_errors))

        output_directory = PROJECT_ROOT / FORMAL_OUTPUT_DIRECTORY
        before = {
            path.relative_to(output_directory).as_posix(): path.read_bytes()
            for path in output_directory.rglob("*")
            if path.is_file()
        }
        with mock.patch(
            "scripts.run_final_veto_ablation.execute_jobs_to_directory"
        ) as execute:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(["--validate-only"])
        after = {
            path.relative_to(output_directory).as_posix(): path.read_bytes()
            for path in output_directory.rglob("*")
            if path.is_file()
        }
        self.assertEqual(result, 0)
        self.assertIn("simulation_started=false", output.getvalue())
        self.assertIn("artifacts_written=false", output.getvalue())
        self.assertIn("formal publication is not ready", output.getvalue())
        self.assertEqual(after, before)
        execute.assert_not_called()

    def test_formal_preflight_test_gate_excludes_rollout_modules(self) -> None:
        self.assertIn("Tests.test_final_veto_compact_logging", FORMAL_PREFLIGHT_TEST_MODULES)
        self.assertIn("Tests.test_final_veto_comparison_renderer", FORMAL_PREFLIGHT_TEST_MODULES)
        self.assertNotIn("Tests.test_final_veto_transition", FORMAL_PREFLIGHT_TEST_MODULES)
        self.assertNotIn("Tests.test_final_veto_runner", FORMAL_PREFLIGHT_TEST_MODULES)

    def test_ordinary_csv_outside_exact_publication_exceptions_remains_ignored(self) -> None:
        ordinary_csv = FORMAL_OUTPUT_DIRECTORY / "ordinary_diagnostic.csv"
        self.assertTrue(self.git_path_is_ignored(ordinary_csv))

    def test_publication_readiness_treats_unexpected_git_exit_as_error(self) -> None:
        failed = subprocess.CompletedProcess(
            args=["git", "check-ignore"], returncode=2, stdout="", stderr="fatal"
        )
        with mock.patch(
            "scripts.run_final_veto_ablation.subprocess.run", return_value=failed
        ):
            errors = publication_readiness_errors(self.manifest, PROJECT_ROOT)
        git_errors = [error for error in errors if "git check-ignore exited 2" in error]
        self.assertEqual(len(git_errors), len(FORMAL_ARTIFACT_NAMES))

    def test_smoke_case_selector_selects_one_frozen_stress_pair(self) -> None:
        case_id = (
            "phase35_radial_energy_push_overspeed_stress_v0"
            "__r0_0p98__angle_150__thrust_8000"
        )
        selected = select_smoke_jobs(self.jobs, case_id)
        self.assertEqual(len(selected), 2)
        self.assertEqual({job.arm_id for job in selected}, {"monitor_off", "monitor_on"})
        self.assertEqual({job.case_id for job in selected}, {case_id})
        self.assertEqual({job.upstream_variant for job in selected}, {"radial_energy_push"})

    def test_smoke_case_selector_rejects_unknown_case(self) -> None:
        with self.assertRaises(RunnerContractError):
            select_smoke_jobs(self.jobs, "not_a_frozen_case")

    def test_planner_rejects_missing_arm(self) -> None:
        with self.assertRaises(RunnerContractError):
            validate_planned_jobs(self.jobs[:-1])


class RolloutHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase34_mode = next(mode for mode in phase34.MODES if mode.name == "radius_priority")
        cls.phase35_variant = next(
            variant for variant in phase35.VARIANTS if variant.name == "radial_energy_push"
        )

    def test_no_hook_phase34_matches_previous_reference(self) -> None:
        actual = phase34.rollout_phase34_case(
            self.phase34_mode, 1.0, 150.0, 8000.0
        )
        self.assertEqual(actual, json.loads(PHASE34_REFERENCE_JSON))

    def test_phase34_import_does_not_load_final_veto_monitor(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import scripts.explicit_controller_phase34_post_cross_sync; "
                    "print('runtime_assurance.final_veto_monitor' in sys.modules)"
                ),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout.strip(), "False")

    def test_identity_hook_phase34_equals_no_hook(self) -> None:
        no_hook = phase34.rollout_phase34_case(
            self.phase34_mode, 1.0, 150.0, 8000.0
        )
        identity = phase34.rollout_phase34_case(
            self.phase34_mode,
            1.0,
            150.0,
            8000.0,
            pre_transition_action_hook=identity_hook,
        )
        self.assertEqual(identity, no_hook)

    def test_no_hook_phase35_matches_previous_reference(self) -> None:
        actual = phase35.rollout_phase35_case(
            self.phase35_variant,
            phase35.PHASE34_TERMINAL_MODE,
            0.98,
            150.0,
            8000.0,
        )
        self.assertEqual(actual, json.loads(PHASE35_REFERENCE_JSON))

    def test_identity_hook_phase35_equals_no_hook(self) -> None:
        no_hook = phase35.rollout_phase35_case(
            self.phase35_variant,
            phase35.PHASE34_TERMINAL_MODE,
            0.98,
            150.0,
            8000.0,
        )
        identity = phase35.rollout_phase35_case(
            self.phase35_variant,
            phase35.PHASE34_TERMINAL_MODE,
            0.98,
            150.0,
            8000.0,
            pre_transition_action_hook=identity_hook,
        )
        self.assertEqual(identity, no_hook)

    def test_hook_receives_nominal_state_and_selected_action(self) -> None:
        captured: dict[str, object] = {}

        def intercept(context: PreTransitionActionContext) -> ActionInterceptionResult:
            captured["context"] = context
            return identity_hook(context)

        def observe(observation: PostTransitionObservation) -> None:
            captured["observation"] = observation
            raise StopAfterObservation

        with self.assertRaises(StopAfterObservation):
            phase34.rollout_phase34_case(
                self.phase34_mode,
                1.0,
                150.0,
                8000.0,
                case_id="hook_case",
                pre_transition_action_hook=intercept,
                post_transition_observation_hook=observe,
            )
        context = captured["context"]
        observation = captured["observation"]
        self.assertEqual(context.current_state, observation.previous_state)
        self.assertEqual(context.nominal_action, observation.nominal_action)
        self.assertEqual(observation.nominal_action, observation.executed_action)
        self.assertEqual(context.case.case_id, "hook_case")

    def test_hook_returned_action_is_used_for_exact_realized_transition(self) -> None:
        captured: dict[str, object] = {}

        def intercept(context: PreTransitionActionContext) -> ActionInterceptionResult:
            captured["expected"] = context.predict_transition(
                context.current_state, (0.0, 0.0)
            ).next_state
            captured["nominal"] = context.nominal_action
            return ActionInterceptionResult(
                nominal_action=context.nominal_action,
                executed_action=(0.0, 0.0),
                intervention_applied=True,
                decision_metadata="test_intervention",
            )

        def observe(observation: PostTransitionObservation) -> None:
            captured["observation"] = observation
            raise StopAfterObservation

        with self.assertRaises(StopAfterObservation):
            phase35.rollout_phase35_case(
                self.phase35_variant,
                phase35.PHASE34_TERMINAL_MODE,
                0.98,
                150.0,
                8000.0,
                pre_transition_action_hook=intercept,
                post_transition_observation_hook=observe,
            )
        observation = captured["observation"]
        self.assertEqual(observation.executed_action, (0.0, 0.0))
        self.assertEqual(observation.realized_next_state, captured["expected"])
        self.assertEqual(observation.nominal_action, captured["nominal"])
        self.assertEqual(observation.decision_metadata, "test_intervention")


class ArmHookRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        jobs = build_planned_jobs(load_frozen_manifest())
        self.off_job = jobs[0]
        self.on_job = jobs[1]
        self.state = CartesianState2D(1.0, 0.0, 1.0, 0.0)
        self.nominal_action = (0.25, -0.5)

    def context_for_ratios(
        self,
        nominal_ratio: float,
        fallback_ratio: float = 1.0,
        *,
        calls: list[tuple[CartesianState2D, tuple[float, float]]] | None = None,
    ) -> PreTransitionActionContext:
        def transition(state, action):
            if calls is not None:
                calls.append((state, action))
            ratio = nominal_ratio if action == self.nominal_action else fallback_ratio
            return Phase3435TransitionResult(
                next_state=CartesianState2D(2.0, 0.0, ratio, 0.0),
                executed_action=NormalizedAction2D(action[0], action[1]),
            )

        case = RolloutCaseContext(
            case_id=self.on_job.case_id,
            controller_id=self.on_job.controller_id,
            controller_family=self.on_job.controller_family,
            r0_over_target=self.on_job.r0_over_target,
            initial_velocity_angle_deg=self.on_job.initial_velocity_angle_deg,
            thrust_scale=self.on_job.thrust_scale,
            target_radius=1.0,
            target_circular_speed=1.0,
            post_cross_mode=self.on_job.post_cross_mode,
        )
        return PreTransitionActionContext(
            step=1,
            phase="DESCENT",
            active_stage="test",
            current_state=self.state,
            nominal_action=self.nominal_action,
            predict_transition=transition,
            compute_speed_ratio=lambda state: state.vx,
            case=case,
        )

    def observation_for(self, context, interception, ratio):
        return PostTransitionObservation(
            step=context.step,
            phase=context.phase,
            active_stage=context.active_stage,
            previous_state=context.current_state,
            nominal_action=context.nominal_action,
            executed_action=interception.executed_action,
            realized_next_state=CartesianState2D(2.0, 0.0, ratio, 0.0),
            realized_next_speed_ratio=ratio,
            intervention_applied=interception.intervention_applied,
            decision_metadata=interception.decision_metadata,
            case=context.case,
        )

    def test_monitor_off_executes_nominal_unchanged_and_reports_no_veto(self) -> None:
        recorder = ArmHookRecorder(self.off_job, is_formal_experiment=False)
        context = replace(self.context_for_ratios(3.0), case=replace(
            self.context_for_ratios(3.0).case, case_id=self.off_job.case_id
        ))
        result = recorder.pre_transition(context)
        recorder.post_transition(self.observation_for(context, result, 3.0))
        self.assertEqual(result.executed_action, self.nominal_action)
        self.assertEqual(recorder.monitor_evaluation_count, 0)
        self.assertEqual(recorder.veto_count, 0)

    def test_monitor_on_below_threshold_allows(self) -> None:
        recorder = ArmHookRecorder(self.on_job, is_formal_experiment=False)
        result = recorder.pre_transition(self.context_for_ratios(1.89))
        self.assertEqual(result.executed_action, self.nominal_action)
        self.assertFalse(result.intervention_applied)

    def test_monitor_on_exact_threshold_allows(self) -> None:
        recorder = ArmHookRecorder(self.on_job, is_formal_experiment=False)
        result = recorder.pre_transition(self.context_for_ratios(1.90))
        self.assertEqual(result.decision_metadata.decision, "allow")

    def test_monitor_on_above_threshold_vetoes_to_zero_action(self) -> None:
        recorder = ArmHookRecorder(self.on_job, is_formal_experiment=False)
        result = recorder.pre_transition(self.context_for_ratios(1.9000000001, 1.5))
        self.assertEqual(result.executed_action, (0.0, 0.0))
        self.assertTrue(result.intervention_applied)

    def test_spy_proves_same_predictor_receives_nominal_then_fallback(self) -> None:
        calls: list[tuple[CartesianState2D, tuple[float, float]]] = []
        context = self.context_for_ratios(2.0, 1.2, calls=calls)
        recorder = ArmHookRecorder(self.on_job, is_formal_experiment=False)
        recorder.pre_transition(context)
        self.assertEqual(calls, [(self.state, self.nominal_action), (self.state, (0.0, 0.0))])

    def test_fallback_failure_is_recorded(self) -> None:
        events: list[Mapping[str, object]] = []
        context = self.context_for_ratios(2.0, 1.95)
        recorder = ArmHookRecorder(
            self.on_job, is_formal_experiment=False, event_sink=events.append
        )
        result = recorder.pre_transition(context)
        recorder.post_transition(self.observation_for(context, result, 1.95))
        self.assertEqual(recorder.fallback_failure_count, 1)
        self.assertTrue(events[0]["fallback_failure"])
        self.assertEqual(events[0]["predicted_fallback_speed_ratio"], 1.95)

    def test_allowed_realized_overspeed_is_false_negative(self) -> None:
        events: list[Mapping[str, object]] = []
        context = self.context_for_ratios(1.89)
        recorder = ArmHookRecorder(
            self.on_job, is_formal_experiment=False, event_sink=events.append
        )
        result = recorder.pre_transition(context)
        recorder.post_transition(self.observation_for(context, result, 1.91))
        self.assertEqual(recorder.false_negative_count, 1)
        self.assertTrue(events[0]["false_negative"])
        self.assertEqual(events[0]["predicted_nominal_speed_ratio"], 1.89)
        self.assertEqual(events[0]["realized_executed_speed_ratio"], 1.91)

    def test_invalid_monitor_evaluation_is_not_allow_or_veto(self) -> None:
        recorder = ArmHookRecorder(self.on_job, is_formal_experiment=False)
        context = self.context_for_ratios(math.nan)
        with self.assertRaises(ArmExecutionError):
            recorder.pre_transition(context)
        self.assertEqual(recorder.monitor_evaluation_count, 0)
        self.assertEqual(recorder.allow_count, 0)
        self.assertEqual(recorder.veto_count, 0)
        self.assertEqual(recorder.invalid_monitor_evaluation_count, 1)

    def test_missing_post_transition_event_fails_stream_completeness(self) -> None:
        recorder = ArmHookRecorder(
            self.on_job,
            is_formal_experiment=False,
            event_sink=lambda _event: None,
        )
        recorder.pre_transition(self.context_for_ratios(1.5))
        with self.assertRaises(RunnerContractError):
            recorder.assert_complete_decision_stream()

    def test_valid_allow_and_veto_events_have_explicit_decision_reasons(self) -> None:
        events: list[Mapping[str, object]] = []
        allow_context = self.context_for_ratios(1.89)
        allow_recorder = ArmHookRecorder(
            self.on_job, is_formal_experiment=False, event_sink=events.append
        )
        allow_result = allow_recorder.pre_transition(allow_context)
        allow_recorder.post_transition(
            self.observation_for(allow_context, allow_result, 1.89)
        )

        veto_context = self.context_for_ratios(2.0, 1.5)
        veto_recorder = ArmHookRecorder(
            self.on_job, is_formal_experiment=False, event_sink=events.append
        )
        veto_result = veto_recorder.pre_transition(veto_context)
        veto_recorder.post_transition(
            self.observation_for(veto_context, veto_result, 1.5)
        )

        self.assertNotIn("unknown", {event["decision_reason"] for event in events})
        self.assertEqual(
            [
                (
                    event["decision_type"],
                    event["decision_reason"],
                    event["veto_status"],
                    event["veto_reason"],
                )
                for event in events
            ],
            [
                (
                    "continue",
                    "predicted_nominal_within_threshold",
                    "allow",
                    "predicted_nominal_within_threshold",
                ),
                (
                    "veto_action",
                    "predicted_nominal_exceeds_overspeed_threshold",
                    "veto",
                    "predicted_nominal_exceeds_overspeed_threshold",
                ),
            ],
        )


class PairMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.job = build_planned_jobs(load_frozen_manifest())[0]

    def pair(self, off_overrides=None, on_overrides=None):
        off = sample_arm_record(self.job, "monitor_off", **(off_overrides or {}))
        on = sample_arm_record(self.job, "monitor_on", **(on_overrides or {}))
        return build_pair_record([off, on])

    def test_avoided_failure_true_only_for_valid_off_hazard(self) -> None:
        pair = self.pair({"overspeed": True}, {"overspeed": False})
        self.assertTrue(pair["avoided_failure"])

    def test_avoided_failure_false_without_off_hazard(self) -> None:
        self.assertFalse(self.pair()["avoided_failure"])

    def test_avoided_failure_false_when_on_invalid(self) -> None:
        pair = self.pair({"overspeed": True}, {"invalid_simulation": True})
        self.assertFalse(pair["avoided_failure"])
        self.assertFalse(pair["pair_valid"])

    def test_blocked_success_true_when_recoverability_is_lost(self) -> None:
        pair = self.pair({}, {"recoverable_crossing": False})
        self.assertTrue(pair["blocked_success"])

    def test_blocked_success_false_when_success_is_preserved(self) -> None:
        self.assertFalse(self.pair()["blocked_success"])

    def test_unnecessary_veto_true_for_safe_off_arm_with_veto_activity(self) -> None:
        self.assertTrue(self.pair()["unnecessary_veto"])

    def test_unnecessary_veto_false_without_veto_activity(self) -> None:
        pair = self.pair({}, {"veto_count": 0, "fallback_count": 0, "allow_count": 10})
        self.assertFalse(pair["unnecessary_veto"])

    def test_false_negative_and_fallback_failure_are_aggregated(self) -> None:
        pair = self.pair({}, {"false_negative_count": 2, "fallback_failure_count": 1})
        self.assertEqual(pair["on_false_negative_count"], 2)
        self.assertEqual(pair["on_fallback_failure_count"], 1)

    def test_incomplete_pair_is_rejected(self) -> None:
        with self.assertRaises(RunnerContractError):
            build_pair_record([sample_arm_record(self.job, "monitor_off")])

    def test_mismatched_hashes_are_rejected(self) -> None:
        off = sample_arm_record(self.job, "monitor_off")
        on = sample_arm_record(self.job, "monitor_on", case_config_hash="different")
        with self.assertRaises(RunnerContractError):
            build_pair_record([off, on])

    def test_duplicate_arm_is_rejected(self) -> None:
        off = sample_arm_record(self.job, "monitor_off")
        with self.assertRaises(RunnerContractError):
            build_pair_record([off, dict(off)])


if __name__ == "__main__":
    unittest.main()
