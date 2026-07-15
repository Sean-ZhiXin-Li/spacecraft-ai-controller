from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_assurance.final_veto_monitor import (  # noqa: E402
    FALLBACK_ACTION,
    MONITOR_ID,
    MonitorEvaluationError,
    OneStepPrediction,
    evaluate_overspeed_veto,
)
from runtime_assurance.final_veto_runner_types import (  # noqa: E402
    ActionInterceptionResult,
    PostTransitionObservation,
    PreTransitionActionContext,
)
from scripts.check_final_veto_manifest import (  # noqa: E402
    MANIFEST_RELATIVE_PATH,
    ManifestValidationError,
    load_manifest,
    validate_manifest_data,
)
from scripts.final_veto_artifacts import (  # noqa: E402
    ARM_FIELDNAMES,
    ARM_JSON_LIST_FIELDS,
    ARM_SCHEMA_VERSION,
    DECISION_SCHEMA_VERSION,
    JsonlEventWriter,
    PAIR_FIELDNAMES,
    PAIR_SCHEMA_VERSION,
    ArtifactWriteError,
    validate_output_directory,
    write_csv_atomic,
    write_text_atomic,
)
from scripts.final_veto_compact_log import (  # noqa: E402
    FORMAL_DEFAULT_DECISION_LOG_MODE,
    LOG_MODE_COMPACT,
    LOG_MODE_FULL_TRACE,
    VALID_DECISION_LOG_MODES,
    DecisionLogStream,
    DecisionStreamStatistics,
    compact_logging_preflight_errors,
    estimate_compact_logging_plan,
    infer_terminal_outcome,
    logging_configuration_errors,
    terminal_transition_record,
)
from scripts.render_final_veto_comparison import (  # noqa: E402
    ComparisonRenderError,
    inspect_png,
    load_comparison_data,
    render_comparison_plot,
)


FORMAL_OUTPUT_DIRECTORY = Path("analysis/final_veto_ablation_v0")
FORMAL_ARTIFACT_NAMES = {
    "results": "results.csv",
    "paired_results": "paired_results.csv",
    "decision_log": "decision_log.jsonl",
    "summary": "summary.md",
    "comparison_plot": "comparison.png",
}
FORMAL_PREFLIGHT_TEST_MODULES = (
    "Tests.test_final_veto_manifest",
    "Tests.test_final_veto_monitor",
    "Tests.test_final_veto_artifacts",
    "Tests.test_final_veto_result_validator",
    "Tests.test_final_veto_compact_logging",
    "Tests.test_final_veto_comparison_renderer",
)
VALID_ARM_IDS = ("monitor_off", "monitor_on")


class RunnerContractError(ValueError):
    pass


class FormalPreflightError(RunnerContractError):
    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


class ArmExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PlannedJob:
    experiment_id: str
    subset_id: str
    case_id: str
    paired_run_id: str
    run_id: str
    arm_id: str
    seed: int
    controller_id: str
    controller_family: str
    monitor_enabled: bool
    case_config_hash: str
    run_config_hash: str
    r0_over_target: float
    initial_velocity_angle_deg: float
    thrust_scale: float
    post_cross_mode: str
    upstream_variant: str | None
    known_phase34_recoverable_case: bool


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_frozen_manifest(
    repository_root: Path = PROJECT_ROOT,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    path = manifest_path or repository_root / MANIFEST_RELATIVE_PATH
    manifest = load_manifest(path)
    validate_manifest_data(manifest)
    return manifest


def _controller_contract(section: Mapping[str, Any]) -> tuple[str, str, str, str | None]:
    context = section["nominal_controller_context"]
    if not isinstance(context, dict):
        raise RunnerContractError("nominal_controller_context must be an object")
    phase = str(context.get("phase", ""))
    controller_id = str(context.get("controller_id", ""))
    if phase == "phase34":
        return controller_id, "explicit_controller", str(context["post_cross_mode"]), None
    if phase == "phase35":
        post_context = str(context.get("post_cross_context", ""))
        mode = "radius_priority" if post_context == "phase34_radius_priority" else post_context
        return controller_id, "explicit_controller", mode, str(context["upstream_variant"])
    raise RunnerContractError(f"unsupported controller phase in manifest: {phase!r}")


def build_planned_jobs(manifest: Mapping[str, Any]) -> list[PlannedJob]:
    validate_manifest_data(dict(manifest))
    jobs: list[PlannedJob] = []
    monitor_id = str(manifest["monitor"]["monitor_id"])
    arm_map = {
        str(arm["arm_id"]): bool(arm["monitor_enabled"])
        for arm in manifest["arms"]
    }

    for section_name in ("preservation_set", "diagnostic_stress_set"):
        section = manifest[section_name]
        subset_id = str(section["subset_id"])
        controller_id, controller_family, mode, upstream_variant = _controller_contract(section)
        known_case = bool(section["known_phase34_recoverable_case"])
        for case in section["cases"]:
            case_id = str(case["case_id"])
            seed = int(case["seed"])
            pair_template = str(manifest["pairing"]["paired_run_id_template"])
            paired_run_id = pair_template.format(case_id=case_id, seed=seed)
            case_configuration = {
                "benchmark_id": manifest["benchmark_id"],
                "benchmark_version": manifest["benchmark_version"],
                "experiment_id": manifest["experiment_id"],
                "subset_id": subset_id,
                "case_id": case_id,
                "seed": seed,
                "parameters": {
                    "r0_over_target": float(case["r0_over_target"]),
                    "initial_velocity_angle_deg": float(case["initial_velocity_angle_deg"]),
                    "thrust_scale": float(case["thrust_scale"]),
                    "target_radius_scale": 1.0,
                },
                "nominal_controller": {
                    "controller_id": controller_id,
                    "controller_family": controller_family,
                    "post_cross_mode": mode,
                    "upstream_variant": upstream_variant,
                },
                "simulator_contract": {
                    "transition": "simulator.phase34_35_transition.step_phase34_35_transition",
                    "constants_source": "scripts.explicit_controller_phase21_orbital_transfer_planner",
                    "horizon_source": "scripts.explicit_controller_phase21_orbital_transfer_planner.MAX_STEPS",
                },
                "output_schema": ARM_SCHEMA_VERSION,
            }
            case_hash = canonical_sha256(case_configuration)
            for arm_id in VALID_ARM_IDS:
                monitor_enabled = arm_map[arm_id]
                run_configuration = {
                    "case_config_hash": case_hash,
                    "arm_id": arm_id,
                    "monitor_enabled": monitor_enabled,
                    "monitor_id": monitor_id if monitor_enabled else None,
                }
                jobs.append(
                    PlannedJob(
                        experiment_id=str(manifest["experiment_id"]),
                        subset_id=subset_id,
                        case_id=case_id,
                        paired_run_id=paired_run_id,
                        run_id=f"{paired_run_id}__{arm_id}",
                        arm_id=arm_id,
                        seed=seed,
                        controller_id=controller_id,
                        controller_family=controller_family,
                        monitor_enabled=monitor_enabled,
                        case_config_hash=case_hash,
                        run_config_hash=canonical_sha256(run_configuration),
                        r0_over_target=float(case["r0_over_target"]),
                        initial_velocity_angle_deg=float(case["initial_velocity_angle_deg"]),
                        thrust_scale=float(case["thrust_scale"]),
                        post_cross_mode=mode,
                        upstream_variant=upstream_variant,
                        known_phase34_recoverable_case=known_case,
                    )
                )
    validate_planned_jobs(jobs)
    return jobs


def validate_planned_jobs(jobs: Iterable[PlannedJob]) -> list[str]:
    planned = list(jobs)
    errors: list[str] = []
    if len(planned) != 26:
        errors.append(f"expected 26 jobs, found {len(planned)}")
    pairs: dict[str, list[PlannedJob]] = {}
    for job in planned:
        pairs.setdefault(job.paired_run_id, []).append(job)
    if len(pairs) != 13:
        errors.append(f"expected 13 paired IDs, found {len(pairs)}")
    for pair_id, pair in pairs.items():
        if len(pair) != 2 or {job.arm_id for job in pair} != set(VALID_ARM_IDS):
            errors.append(f"{pair_id} does not contain exactly one off and one on arm")
            continue
        if len({job.case_config_hash for job in pair}) != 1:
            errors.append(f"{pair_id} has mismatched shared case hashes")
        shared = {
            (
                job.case_id,
                job.subset_id,
                job.seed,
                job.controller_id,
                job.r0_over_target,
                job.initial_velocity_angle_deg,
                job.thrust_scale,
            )
            for job in pair
        }
        if len(shared) != 1:
            errors.append(f"{pair_id} has mismatched shared configuration")
    preservation_count = sum(job.known_phase34_recoverable_case for job in planned)
    if preservation_count != 16:
        errors.append(f"expected 16 preservation jobs, found {preservation_count}")
    if len(planned) - preservation_count != 10:
        errors.append(f"expected 10 stress jobs, found {len(planned) - preservation_count}")
    if errors:
        raise RunnerContractError("; ".join(errors))
    return [
        "26 deterministic jobs",
        "13 complete off/on pairs",
        "16 preservation jobs",
        "10 diagnostic stress jobs",
    ]


class ArmHookRecorder:
    def __init__(
        self,
        job: PlannedJob,
        *,
        is_formal_experiment: bool,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        self.job = job
        self.is_formal_experiment = is_formal_experiment
        self.event_sink = event_sink
        self.monitor_evaluation_count = 0
        self.allow_count = 0
        self.veto_count = 0
        self.fallback_count = 0
        self.false_negative_count = 0
        self.fallback_failure_count = 0
        self.invalid_monitor_evaluation_count = 0
        self.nominal_actions_unchanged_count = 0
        self.decision_statistics = DecisionStreamStatistics()

    def _emit_event(self, event: Mapping[str, object]) -> None:
        self.decision_statistics.observe(event)
        if self.event_sink is not None:
            self.event_sink(event)

    def assert_complete_decision_stream(self) -> None:
        expected = self.monitor_evaluation_count + self.invalid_monitor_evaluation_count
        if self.decision_statistics.event_count != expected:
            raise RunnerContractError(
                "a monitor evaluation completed without a logical decision-stream event"
            )

    def pre_transition(self, context: PreTransitionActionContext) -> ActionInterceptionResult:
        if context.case.case_id != self.job.case_id:
            raise ArmExecutionError("rollout hook case identity does not match planned job")
        if not self.job.monitor_enabled:
            return ActionInterceptionResult(
                nominal_action=context.nominal_action,
                executed_action=context.nominal_action,
                intervention_applied=False,
                decision_metadata=None,
            )

        def predictor(state, action):
            transition = context.predict_transition(state, action)
            return OneStepPrediction(
                next_state=transition.next_state,
                speed_ratio=context.compute_speed_ratio(transition.next_state),
            )

        try:
            decision = evaluate_overspeed_veto(
                context.current_state,
                context.nominal_action,
                predictor,
            )
        except MonitorEvaluationError as exc:
            self.invalid_monitor_evaluation_count += 1
            self._emit_event(self._invalid_event(context, exc))
            raise ArmExecutionError("Final Veto monitor evaluation failed") from exc

        self.monitor_evaluation_count += 1
        if decision.decision == "allow":
            self.allow_count += 1
        elif decision.decision == "veto":
            self.veto_count += 1
            self.fallback_count += 1
        else:
            raise ArmExecutionError(f"unexpected monitor decision: {decision.decision!r}")
        return ActionInterceptionResult(
            nominal_action=context.nominal_action,
            executed_action=decision.executed_action,
            intervention_applied=decision.veto_applied,
            decision_metadata=decision,
        )

    def post_transition(self, observation: PostTransitionObservation) -> None:
        realized_values = (
            observation.realized_next_state.x,
            observation.realized_next_state.y,
            observation.realized_next_state.vx,
            observation.realized_next_state.vy,
            observation.realized_next_speed_ratio,
        )
        if not all(math.isfinite(value) for value in realized_values):
            raise ArmExecutionError("executed transition produced a non-finite result")
        if observation.nominal_action == observation.executed_action:
            self.nominal_actions_unchanged_count += 1
        if not self.job.monitor_enabled:
            return
        decision = observation.decision_metadata
        if decision is None or not hasattr(decision, "decision"):
            raise ArmExecutionError("monitor-on transition is missing decision metadata")
        realized_overspeed = observation.realized_next_speed_ratio > 1.90
        fallback_executed = bool(decision.veto_applied)
        false_negative = decision.decision == "allow" and realized_overspeed
        fallback_failure = fallback_executed and realized_overspeed
        if false_negative:
            self.false_negative_count += 1
        if fallback_failure:
            self.fallback_failure_count += 1
        self._emit_event(
            self._decision_event(
                observation,
                false_negative=false_negative,
                fallback_failure=fallback_failure,
            )
        )

    def _decision_event(
        self,
        observation: PostTransitionObservation,
        *,
        false_negative: bool,
        fallback_failure: bool,
    ) -> dict[str, object]:
        decision = observation.decision_metadata
        fallback_executed = bool(decision.veto_applied)
        predicted_executed = (
            decision.predicted_fallback_speed_ratio
            if decision.veto_applied
            else decision.predicted_nominal_speed_ratio
        )
        prediction_matches = predicted_executed == observation.realized_next_speed_ratio
        state_summary = json.dumps(
            {
                "x": observation.previous_state.x,
                "y": observation.previous_state.y,
                "vx": observation.previous_state.vx,
                "vy": observation.previous_state.vy,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        decision_reason = (
            "predicted_nominal_exceeds_overspeed_threshold"
            if decision.veto_applied
            else "predicted_nominal_within_threshold"
        )
        return {
            "decision_schema_version": DECISION_SCHEMA_VERSION,
            "decision_id": f"{self.job.run_id}__step_{observation.step}",
            "experiment_id": self.job.experiment_id,
            "run_id": self.job.run_id,
            "paired_run_id": self.job.paired_run_id,
            "case_id": self.job.case_id,
            "subset_id": self.job.subset_id,
            "arm_id": self.job.arm_id,
            "step": observation.step,
            "phase": observation.phase,
            "active_stage": observation.active_stage,
            "decision_type": "veto_action" if decision.veto_applied else "continue",
            "decision_reason": decision_reason,
            "decision_scope": "veto",
            "decision_authority": "runtime_assurance",
            "monitor_id": decision.monitor_id,
            "state_summary": state_summary,
            "safety_level": "critical" if observation.realized_next_speed_ratio > 1.90 else "nominal",
            "recoverability_level": "unknown",
            "trust_flags": ["none"] if prediction_matches else ["prediction_mismatch"],
            "nominal_proposed_action": list(observation.nominal_action),
            "executed_action": list(observation.executed_action),
            "predicted_nominal_speed_ratio": decision.predicted_nominal_speed_ratio,
            "predicted_fallback_speed_ratio": decision.predicted_fallback_speed_ratio,
            "realized_executed_speed_ratio": observation.realized_next_speed_ratio,
            "hazard_threshold": decision.threshold,
            "hazard_comparator": decision.comparator,
            "veto_status": "veto" if decision.veto_applied else "allow",
            "veto_reason": decision_reason,
            "fallback_available": True,
            "fallback_action": list(decision.fallback_action),
            "fallback_executed": fallback_executed,
            "fallback_failure": fallback_failure,
            "invalid_evaluation": False,
            "manual_audit_note": (
                "Nonformal run; no pair-level counterfactual claim."
                if not self.is_formal_experiment
                else "Pair-level counterfactual interpretation pending."
            ),
            "is_formal_experiment": self.is_formal_experiment,
            "false_negative": false_negative,
        }

    def _invalid_event(
        self,
        context: PreTransitionActionContext,
        error: Exception,
    ) -> dict[str, object]:
        return {
            "decision_schema_version": DECISION_SCHEMA_VERSION,
            "decision_id": f"{self.job.run_id}__step_{context.step}__invalid",
            "experiment_id": self.job.experiment_id,
            "run_id": self.job.run_id,
            "paired_run_id": self.job.paired_run_id,
            "case_id": self.job.case_id,
            "subset_id": self.job.subset_id,
            "arm_id": self.job.arm_id,
            "step": context.step,
            "phase": context.phase,
            "active_stage": context.active_stage,
            "decision_type": "unknown",
            "decision_reason": "invalid_simulation",
            "decision_scope": "veto",
            "decision_authority": "runtime_assurance",
            "monitor_id": MONITOR_ID,
            "state_summary": "monitor evaluation did not produce a valid allow/veto decision",
            "safety_level": "unknown",
            "recoverability_level": "unknown",
            "trust_flags": ["prediction_mismatch"],
            "nominal_proposed_action": list(context.nominal_action),
            "executed_action": None,
            "predicted_nominal_speed_ratio": None,
            "predicted_fallback_speed_ratio": None,
            "realized_executed_speed_ratio": None,
            "hazard_threshold": 1.90,
            "hazard_comparator": ">",
            "veto_status": "unknown",
            "veto_reason": type(error).__name__,
            "fallback_available": True,
            "fallback_action": list(FALLBACK_ACTION),
            "fallback_executed": False,
            "fallback_failure": False,
            "invalid_evaluation": True,
            "manual_audit_note": "Invalid evaluation requires audit; it is not counted as allow or veto.",
            "is_formal_experiment": self.is_formal_experiment,
            "false_negative": False,
        }


def _terminal_label(legacy: Mapping[str, object], invalid: bool) -> str:
    if invalid:
        return "invalid_simulation"
    if bool(legacy.get("overspeed")):
        return "overspeed"
    if bool(legacy.get("instability")) or str(legacy.get("termination_reason")) in {
        "out_range",
        "too_close",
        "radial_stall",
    }:
        return "instability"
    if bool(legacy.get("success")):
        return "success"
    crossing = bool(legacy.get("crossing_occurs"))
    recoverable = bool(legacy.get("recoverable_crossing"))
    if crossing and recoverable:
        return "recoverable_crossing_failed_late"
    if crossing:
        return "crossing_unrecoverable"
    if not crossing:
        return "no_crossing"
    if bool(legacy.get("truncated")):
        return "timeout"
    return "unknown"


def build_arm_record(
    job: PlannedJob,
    legacy: Mapping[str, object],
    recorder: ArmHookRecorder,
    *,
    implementation_commit: str,
    is_formal_experiment: bool,
    invalid_simulation: bool = False,
    manual_audit_note: str = "",
    decision_log_mode: str = FORMAL_DEFAULT_DECISION_LOG_MODE,
    compact_decision_record_count: int = 0,
) -> dict[str, object]:
    termination_reason = str(legacy.get("termination_reason", ""))
    instability = bool(legacy.get("instability")) or termination_reason in {
        "out_range",
        "too_close",
        "radial_stall",
    }
    scope = (
        "protected_preservation_set"
        if job.known_phase34_recoverable_case
        else "diagnostic_stress_set"
    )
    regression_membership = (
        ["known_phase34_recoverable_preservation"]
        if job.known_phase34_recoverable_case
        else ["diagnostic_stress"]
    )
    note = manual_audit_note or (
        "Raw formal arm row; pair-level acceptance has not been evaluated."
        if is_formal_experiment
        else "Nonformal smoke row; not scientific evidence."
    )
    crossing = bool(legacy.get("crossing_occurs"))
    statistics = recorder.decision_statistics
    executed_steps = max(
        int(legacy.get("steps", 0) or 0),
        statistics.last_event_step or 0,
    )
    valid_evaluations = recorder.monitor_evaluation_count
    if valid_evaluations:
        intervention_rate = recorder.veto_count / valid_evaluations
        allow_rate = recorder.allow_count / valid_evaluations
        fallback_rate = recorder.fallback_count / valid_evaluations
    elif not job.monitor_enabled:
        intervention_rate = 0.0
        allow_rate = 0.0
        fallback_rate = 0.0
    else:
        intervention_rate = None
        allow_rate = None
        fallback_rate = None
    terminal_label = _terminal_label(legacy, invalid_simulation)
    termination_reason = (
        "invalid_simulation"
        if invalid_simulation
        else str(legacy.get("termination_reason", "")).strip() or terminal_label
    )
    return {
        "schema_version": ARM_SCHEMA_VERSION,
        "benchmark_id": "recoverability_benchmark",
        "benchmark_version": "v1",
        "experiment_id": job.experiment_id,
        "experiment_status": (
            "formal_executed_pending_validation"
            if is_formal_experiment
            else "nonformal_smoke"
        ),
        "implementation_commit": implementation_commit,
        "run_id": job.run_id,
        "paired_run_id": job.paired_run_id,
        "case_config_hash": job.case_config_hash,
        "run_config_hash": job.run_config_hash,
        "subset_id": job.subset_id,
        "case_id": job.case_id,
        "arm_id": job.arm_id,
        "seed": job.seed,
        "controller_id": job.controller_id,
        "controller_family": job.controller_family,
        "artifact_path": "analysis/final_veto_ablation_v0/results.csv" if is_formal_experiment else "",
        "source_script": "scripts/run_final_veto_ablation.py",
        "monitor_enabled": job.monitor_enabled,
        "monitor_id": MONITOR_ID if job.monitor_enabled else "",
        "hazard_target": "overspeed",
        "hazard_threshold": 1.90,
        "hazard_comparator": ">",
        "r0_over_target": job.r0_over_target,
        "initial_velocity_angle_deg": job.initial_velocity_angle_deg,
        "thrust_scale": job.thrust_scale,
        "crossed_target_radius": crossing,
        "first_crossing_step": legacy.get("crossing_step") if crossing else None,
        "recoverable_crossing": bool(legacy.get("recoverable_crossing")),
        "final_simulator_success": bool(legacy.get("success")),
        "overspeed": bool(legacy.get("overspeed")),
        "max_speed_ratio": legacy.get("max_speed_ratio"),
        "instability": instability,
        "unsafe_state": instability,
        "invalid_simulation": invalid_simulation,
        "terminal_label": terminal_label,
        "precursor_labels": ["target_radius_crossing"] if crossing else [],
        "diagnostic_labels": [],
        "manual_audit_note": note,
        "label_taxonomy_version": "failure_label_taxonomy_v0",
        "is_full_benchmark": False,
        "subset_claim_scope": scope,
        "regression_set_membership": regression_membership,
        "known_phase34_recoverable_case": job.known_phase34_recoverable_case,
        "monitor_evaluation_count": recorder.monitor_evaluation_count,
        "allow_count": recorder.allow_count,
        "veto_count": recorder.veto_count,
        "fallback_count": recorder.fallback_count,
        "false_negative_count": recorder.false_negative_count,
        "fallback_failure_count": recorder.fallback_failure_count,
        "invalid_monitor_evaluation_count": recorder.invalid_monitor_evaluation_count,
        "nominal_actions_unchanged_count": recorder.nominal_actions_unchanged_count,
        "steps": executed_steps,
        "accepted_as_progress": False,
        "acceptance_reason": (
            "raw formal arm requires paired and aggregate validation"
            if is_formal_experiment
            else "nonformal smoke run"
        ),
        "is_formal_experiment": is_formal_experiment,
        "termination_reason": termination_reason,
        "decision_stream_event_count": statistics.event_count,
        "decision_stream_sha256": statistics.sha256,
        "compact_decision_record_count": compact_decision_record_count,
        "decision_log_mode": decision_log_mode,
        "intervention_rate": intervention_rate,
        "allow_rate": allow_rate,
        "fallback_rate": fallback_rate,
        "first_veto_step": statistics.first_veto_step,
        "last_veto_step": statistics.last_veto_step,
        "longest_consecutive_veto_steps": statistics.longest_consecutive_veto_steps,
        "longest_consecutive_allow_steps": statistics.longest_consecutive_allow_steps,
        "veto_segment_count": statistics.veto_segment_count,
        "allow_segment_count": statistics.allow_segment_count,
    }


def execute_job(
    job: PlannedJob,
    *,
    implementation_commit: str,
    is_formal_experiment: bool,
    event_sink: Callable[[Mapping[str, object]], None] | None = None,
    decision_stream: DecisionLogStream | None = None,
    decision_log_mode: str = FORMAL_DEFAULT_DECISION_LOG_MODE,
) -> dict[str, object]:
    sink = decision_stream.consume if decision_stream is not None else event_sink
    recorder = ArmHookRecorder(
        job,
        is_formal_experiment=is_formal_experiment,
        event_sink=sink,
    )
    legacy: Mapping[str, object] = {}
    invalid_simulation = False
    manual_audit_note = ""
    try:
        if job.upstream_variant is None:
            from scripts import explicit_controller_phase34_post_cross_sync as phase34

            mode = next(mode for mode in phase34.MODES if mode.name == job.post_cross_mode)
            legacy = phase34.rollout_phase34_case(
                mode,
                job.r0_over_target,
                job.initial_velocity_angle_deg,
                job.thrust_scale,
                record_trajectory=False,
                case_id=job.case_id,
                pre_transition_action_hook=recorder.pre_transition,
                post_transition_observation_hook=recorder.post_transition,
            )
        else:
            from scripts import explicit_controller_phase35_crossing_basin_expansion as phase35

            variant = next(
                variant for variant in phase35.VARIANTS if variant.name == job.upstream_variant
            )
            mode = next(mode for mode in phase35.PHASE34_MODES if mode.name == job.post_cross_mode)
            legacy = phase35.rollout_phase35_case(
                variant,
                mode,
                job.r0_over_target,
                job.initial_velocity_angle_deg,
                job.thrust_scale,
                record_trajectory=False,
                case_id=job.case_id,
                pre_transition_action_hook=recorder.pre_transition,
                post_transition_observation_hook=recorder.post_transition,
            )
    except (ArmExecutionError, ArithmeticError, ValueError) as exc:
        legacy = {}
        invalid_simulation = True
        manual_audit_note = (
            f"execution failed with {type(exc).__name__}; manual audit required"
        )

    recorder.assert_complete_decision_stream()

    record = build_arm_record(
        job,
        legacy,
        recorder,
        implementation_commit=implementation_commit,
        is_formal_experiment=is_formal_experiment,
        invalid_simulation=invalid_simulation,
        manual_audit_note=manual_audit_note,
        decision_log_mode=decision_log_mode,
    )
    if decision_stream is not None and job.monitor_enabled:
        decision_stream.finish_run(terminal_transition_record(record))
        if (
            decision_stream.decision_stream_event_count
            != recorder.decision_statistics.event_count
            or decision_stream.decision_stream_sha256
            != recorder.decision_statistics.sha256
        ):
            raise RunnerContractError("decision stream digest or count drifted during logging")
        record["compact_decision_record_count"] = (
            decision_stream.compact_decision_record_count
        )
    return record


def _require_boolean(record: Mapping[str, object], field: str) -> bool:
    value = record.get(field)
    if not isinstance(value, bool):
        raise RunnerContractError(f"{field} must be boolean")
    return value


def build_pair_record(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    rows = list(records)
    if len(rows) != 2:
        raise RunnerContractError("a pair requires exactly two arm records")
    by_arm: dict[str, Mapping[str, object]] = {}
    for row in rows:
        arm_id = str(row.get("arm_id", ""))
        if arm_id in by_arm:
            raise RunnerContractError(f"duplicate arm in pair: {arm_id}")
        by_arm[arm_id] = row
    if set(by_arm) != set(VALID_ARM_IDS):
        raise RunnerContractError("pair requires one monitor_off and one monitor_on arm")
    off = by_arm["monitor_off"]
    on = by_arm["monitor_on"]
    shared_fields = (
        "experiment_id",
        "paired_run_id",
        "case_id",
        "subset_id",
        "seed",
        "case_config_hash",
        "controller_id",
        "r0_over_target",
        "initial_velocity_angle_deg",
        "thrust_scale",
    )
    mismatches = [field for field in shared_fields if off.get(field) != on.get(field)]
    if mismatches:
        raise RunnerContractError(f"pair has mismatched shared fields: {mismatches}")

    off_invalid = _require_boolean(off, "invalid_simulation")
    on_invalid = _require_boolean(on, "invalid_simulation")
    off_overspeed = _require_boolean(off, "overspeed")
    on_overspeed = _require_boolean(on, "overspeed")
    off_recoverable = _require_boolean(off, "recoverable_crossing")
    on_recoverable = _require_boolean(on, "recoverable_crossing")
    off_success = _require_boolean(off, "final_simulator_success")
    on_success = _require_boolean(on, "final_simulator_success")
    pair_valid = not off_invalid and not on_invalid
    avoided_failure = pair_valid and off_overspeed and not on_overspeed
    blocked_success = pair_valid and (
        (off_recoverable and not on_recoverable) or (off_success and not on_success)
    )
    on_veto_count = int(on.get("veto_count", 0))
    unnecessary_veto = pair_valid and on_veto_count > 0 and not off_overspeed
    is_formal = bool(off.get("is_formal_experiment")) and bool(on.get("is_formal_experiment"))
    claim_eligible = pair_valid and is_formal
    reasons: list[str] = []
    if not pair_valid:
        reasons.append("invalid arm result")
    if not is_formal:
        reasons.append("nonformal run")
    off_terminal_label = str(off.get("terminal_label", "unknown"))
    on_terminal_label = str(on.get("terminal_label", "unknown"))
    off_terminal_outcome = infer_terminal_outcome(off)
    on_terminal_outcome = infer_terminal_outcome(on)
    step_delta = int(on.get("steps", 0)) - int(off.get("steps", 0))
    task_outcome_preserved = (
        off_recoverable == on_recoverable and off_success == on_success
    )
    task_recovered_after_hazard_avoidance = (
        avoided_failure and on_recoverable and on_success
    )
    terminal_outcome_transition = f"{off_terminal_outcome} -> {on_terminal_outcome}"
    return {
        "pair_schema_version": PAIR_SCHEMA_VERSION,
        "experiment_id": off["experiment_id"],
        "paired_run_id": off["paired_run_id"],
        "case_id": off["case_id"],
        "subset_id": off["subset_id"],
        "seed": off["seed"],
        "case_config_hash": off["case_config_hash"],
        "off_run_id": off["run_id"],
        "on_run_id": on["run_id"],
        "pair_complete": True,
        "pair_valid": pair_valid,
        "off_overspeed": off_overspeed,
        "on_overspeed": on_overspeed,
        "off_crossed_target_radius": _require_boolean(off, "crossed_target_radius"),
        "on_crossed_target_radius": _require_boolean(on, "crossed_target_radius"),
        "off_recoverable_crossing": off_recoverable,
        "on_recoverable_crossing": on_recoverable,
        "off_final_simulator_success": off_success,
        "on_final_simulator_success": on_success,
        "off_invalid_simulation": off_invalid,
        "on_invalid_simulation": on_invalid,
        "on_monitor_evaluation_count": int(on.get("monitor_evaluation_count", 0)),
        "on_allow_count": int(on.get("allow_count", 0)),
        "on_veto_count": on_veto_count,
        "on_fallback_count": int(on.get("fallback_count", 0)),
        "on_false_negative_count": int(on.get("false_negative_count", 0)),
        "on_fallback_failure_count": int(on.get("fallback_failure_count", 0)),
        "avoided_failure": avoided_failure,
        "blocked_success": blocked_success,
        "unnecessary_veto": unnecessary_veto,
        "performance_cost_summary": {
            "off_steps": int(off.get("steps", 0)),
            "on_steps": int(on.get("steps", 0)),
            "step_delta": step_delta,
            "veto_count": on_veto_count,
            "intervention_rate": on.get("intervention_rate"),
            "terminal_outcome_transition": terminal_outcome_transition,
        },
        "claim_eligible": claim_eligible,
        "claim_ineligibility_reason": "; ".join(reasons),
        "is_formal_experiment": is_formal,
        "terminal_label_without_monitor": off_terminal_label,
        "terminal_label_with_monitor": on_terminal_label,
        "terminal_label_changed": off_terminal_label != on_terminal_label,
        "termination_reason_without_monitor": off_terminal_outcome,
        "termination_reason_with_monitor": on_terminal_outcome,
        "terminal_outcome_transition": terminal_outcome_transition,
        "step_count_delta": step_delta,
        "monitor_induced_horizon_extension": step_delta,
        "task_outcome_preserved": task_outcome_preserved,
        "declared_hazard_avoided": avoided_failure,
        "task_recovered_after_hazard_avoidance": task_recovered_after_hazard_avoidance,
    }


def build_pair_records(arm_records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in arm_records:
        grouped.setdefault(str(record.get("paired_run_id", "")), []).append(record)
    return [build_pair_record(grouped[pair_id]) for pair_id in sorted(grouped)]


def build_diagnostic_summary(
    arm_records: Iterable[Mapping[str, object]],
    pair_records: Iterable[Mapping[str, object]],
) -> str:
    arms = list(arm_records)
    pairs = list(pair_records)
    monitor_on = [row for row in arms if row.get("arm_id") == "monitor_on"]
    evaluations = sum(int(row.get("monitor_evaluation_count", 0)) for row in monitor_on)
    vetoes = sum(int(row.get("veto_count", 0)) for row in monitor_on)
    intervention_rate = vetoes / evaluations if evaluations else 0.0
    transitions = sorted(
        {str(row.get("terminal_outcome_transition", "unknown -> unknown")) for row in pairs}
    )
    step_deltas = [int(row.get("step_count_delta", 0)) for row in pairs]
    lines = [
        "# Final Veto Paired Diagnostic Summary",
        "",
        "This summary separates declared-hazard evidence from task completion and does not make a formal-safety claim.",
        "",
        "## Declared hazard reduction",
        "",
        f"- Complete valid pairs avoiding the declared overspeed hazard: {sum(bool(row.get('declared_hazard_avoided')) for row in pairs)}.",
        "",
        "## Task outcome",
        "",
        f"- Pairs recovering the declared task after hazard avoidance: {sum(bool(row.get('task_recovered_after_hazard_avoidance')) for row in pairs)}.",
        f"- Pairs preserving the explicit recoverable-crossing and simulator-success tuple: {sum(bool(row.get('task_outcome_preserved')) for row in pairs)}.",
        "",
        "## Intervention burden",
        "",
        f"- Monitor evaluations: {evaluations}.",
        f"- Vetoes: {vetoes}.",
        f"- Aggregate intervention rate: {intervention_rate:.8f}.",
        "",
        "## Performance cost",
        "",
        f"- Monitor-on minus monitor-off step deltas: {step_deltas}.",
        "- Step deltas are reported without automatically labeling their sign as beneficial or harmful.",
        "",
        "## Terminal failure-mode transition",
        "",
    ]
    lines.extend(f"- {transition}." for transition in transitions)
    lines.extend(
        [
            "",
            "Declared hazard avoidance does not by itself mean the task recovered or completed.",
            "",
        ]
    )
    return "\n".join(lines)


def current_commit(repository_root: Path = PROJECT_ROOT) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={repository_root.as_posix()}", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or len(commit) != 40:
        raise RunnerContractError("could not determine current implementation commit")
    return commit


def _manifest_output_paths(manifest: Mapping[str, Any]) -> dict[str, Path]:
    return {
        str(item["artifact_id"]): Path(str(item["path"]))
        for item in manifest["output_contract"]["future_artifacts"]
    }


def publication_readiness_errors(
    manifest: Mapping[str, Any],
    repository_root: Path = PROJECT_ROOT,
) -> list[str]:
    # The manifest's currently_ignored_by_gitignore values describe the
    # freeze-time state. Formal preflight intentionally asks live Git instead;
    # dedicated publication exceptions added after the freeze may differ.
    errors: list[str] = []
    expected = {
        artifact_id: FORMAL_OUTPUT_DIRECTORY / filename
        for artifact_id, filename in FORMAL_ARTIFACT_NAMES.items()
    }
    actual = _manifest_output_paths(manifest)
    if actual != expected:
        errors.append("formal output paths differ from the frozen manifest contract")
    for artifact_id, relative in actual.items():
        destination = repository_root / relative
        if destination.exists():
            errors.append(f"formal output already exists: {relative.as_posix()}")
        ignored = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository_root.as_posix()}",
                "check-ignore",
                "-q",
                "--no-index",
                relative.as_posix(),
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if ignored.returncode == 0:
            errors.append(f"formal output is ignored by .gitignore: {relative.as_posix()}")
        elif ignored.returncode != 1:
            errors.append(
                "could not determine live .gitignore state for "
                f"{relative.as_posix()}: git check-ignore exited {ignored.returncode}"
            )
    return errors


def formal_preflight_errors(
    manifest: Mapping[str, Any],
    jobs: Iterable[PlannedJob],
    repository_root: Path = PROJECT_ROOT,
    *,
    run_tests: bool = True,
    decision_log_mode: str = FORMAL_DEFAULT_DECISION_LOG_MODE,
    digest_enabled: bool = True,
    full_trace_path: Path | None = None,
) -> list[str]:
    planned_jobs = list(jobs)
    errors: list[str] = []
    try:
        validate_manifest_data(dict(manifest))
        validate_planned_jobs(planned_jobs)
    except (ManifestValidationError, RunnerContractError) as exc:
        errors.append(f"contract validation failed: {exc}")
    errors.extend(
        logging_configuration_errors(
            mode=decision_log_mode,
            is_formal_experiment=True,
            full_trace_path=full_trace_path,
            repository_root=repository_root,
            digest_enabled=digest_enabled,
        )
    )
    errors.extend(
        compact_logging_preflight_errors(
            planned_jobs,
            mode=decision_log_mode,
            digest_enabled=digest_enabled,
        )
    )

    protected_guard = subprocess.run(
        [sys.executable, "scripts/check_phase_results.py"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if protected_guard.returncode != 0:
        errors.append("protected historical regression guard failed")

    tracked = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository_root.as_posix()}",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0 or tracked.stdout.strip():
        errors.append("tracked working tree is not clean")
    staged = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository_root.as_posix()}",
            "diff",
            "--cached",
            "--quiet",
        ],
        cwd=repository_root,
        check=False,
    )
    if staged.returncode != 0:
        errors.append("staged changes are present")

    try:
        validate_output_directory(
            repository_root / FORMAL_OUTPUT_DIRECTORY,
            repository_root=repository_root,
            protected_paths=manifest["protected_paths"],
        )
    except ArtifactWriteError as exc:
        errors.append(str(exc))
    errors.extend(publication_readiness_errors(manifest, repository_root))

    if run_tests:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "-q", *FORMAL_PREFLIGHT_TEST_MODULES],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            errors.append("bounded no-rollout Final Veto preflight tests have not passed")
    return errors


def _execute_jobs_to_artifact_directory(
    jobs: Iterable[PlannedJob],
    output_directory: Path,
    manifest: Mapping[str, Any],
    *,
    is_formal_experiment: bool,
    decision_log_mode: str = FORMAL_DEFAULT_DECISION_LOG_MODE,
    full_trace_path: Path | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output = validate_output_directory(
        output_directory,
        repository_root=PROJECT_ROOT,
        protected_paths=manifest["protected_paths"],
    )
    logging_errors = logging_configuration_errors(
        mode=decision_log_mode,
        is_formal_experiment=is_formal_experiment,
        full_trace_path=full_trace_path,
        repository_root=PROJECT_ROOT,
        digest_enabled=True,
    )
    if logging_errors:
        raise RunnerContractError("; ".join(logging_errors))
    commit = current_commit()
    arm_rows: list[dict[str, object]] = []
    if decision_log_mode == LOG_MODE_COMPACT:
        decision_directory = output
        decision_name = (
            "decision_log.jsonl" if is_formal_experiment else "smoke_decision_log.jsonl"
        )
        decision_repository_root: Path | None = PROJECT_ROOT
        decision_protected_paths = manifest["protected_paths"]
    else:
        if full_trace_path is None:
            raise RunnerContractError("full_trace mode requires an output path")
        decision_directory = full_trace_path.resolve().parent
        decision_name = full_trace_path.resolve().name
        decision_repository_root = None
        decision_protected_paths = ()
    with JsonlEventWriter(
        decision_directory,
        decision_name,
        repository_root=decision_repository_root,
        protected_paths=decision_protected_paths,
    ) as decision_writer:
        for job in jobs:
            decision_stream = (
                DecisionLogStream(
                    decision_writer.write_event,
                    mode=decision_log_mode,
                    is_formal_experiment=is_formal_experiment,
                )
                if job.monitor_enabled
                else None
            )
            arm_rows.append(
                execute_job(
                    job,
                    implementation_commit=commit,
                    is_formal_experiment=is_formal_experiment,
                    decision_stream=decision_stream,
                    decision_log_mode=decision_log_mode,
                )
            )
        pair_rows = build_pair_records(arm_rows)
        from scripts.check_final_veto_results import read_jsonl, validate_result_records

        structural_report = validate_result_records(
            arm_rows,
            pair_rows,
            manifest,
            decision_events=read_jsonl(decision_writer.staged_path_for_validation()),
            output_directory=output,
            repository_root=PROJECT_ROOT,
        )
        if not structural_report.structural_valid:
            raise RunnerContractError(
                "generated records failed structural validation: "
                + "; ".join(structural_report.errors)
            )
    result_name = "results.csv" if is_formal_experiment else "smoke_results.csv"
    pair_name = "paired_results.csv" if is_formal_experiment else "smoke_paired_results.csv"
    write_csv_atomic(
        output,
        result_name,
        arm_rows,
        ARM_FIELDNAMES,
        list_fields=ARM_JSON_LIST_FIELDS,
        repository_root=PROJECT_ROOT,
        protected_paths=manifest["protected_paths"],
    )
    write_csv_atomic(
        output,
        pair_name,
        pair_rows,
        PAIR_FIELDNAMES,
        repository_root=PROJECT_ROOT,
        protected_paths=manifest["protected_paths"],
    )
    if is_formal_experiment:
        write_text_atomic(
            output,
            "summary.md",
            build_diagnostic_summary(arm_rows, pair_rows),
            repository_root=PROJECT_ROOT,
            protected_paths=manifest["protected_paths"],
        )
    return arm_rows, pair_rows


def formal_package_errors(artifact_directory: Path) -> list[str]:
    directory = artifact_directory.resolve()
    errors: list[str] = []
    paths = {
        artifact_id: directory / filename
        for artifact_id, filename in FORMAL_ARTIFACT_NAMES.items()
    }
    for artifact_id, path in paths.items():
        if not path.is_file():
            errors.append(f"formal package is missing {artifact_id}: {path.name}")
        elif path.stat().st_size <= 0:
            errors.append(f"formal package artifact is empty: {path.name}")
    comparison = paths["comparison_plot"]
    if comparison.is_file() and comparison.stat().st_size > 0:
        try:
            inspect_png(comparison)
        except ComparisonRenderError as exc:
            errors.append(f"formal comparison plot is unreadable: {exc}")
    results = paths["results"]
    paired = paths["paired_results"]
    if results.is_file() and paired.is_file():
        try:
            data = load_comparison_data(results, paired)
        except ComparisonRenderError as exc:
            errors.append(f"formal CSV package is invalid: {exc}")
        else:
            if len(data.arm_rows) != 26 or len(data.pair_rows) != 13:
                errors.append("formal CSV package requires 26 arm rows and 13 pair rows")
    return errors


def require_complete_formal_package(artifact_directory: Path) -> None:
    errors = formal_package_errors(artifact_directory)
    if errors:
        raise RunnerContractError("incomplete formal artifact package: " + "; ".join(errors))


def publish_complete_formal_package(
    staged_directory: Path,
    output_directory: Path,
    manifest: Mapping[str, Any],
) -> None:
    staged = staged_directory.resolve()
    output = validate_output_directory(
        output_directory,
        repository_root=PROJECT_ROOT,
        protected_paths=manifest["protected_paths"],
    )
    expected_output = (PROJECT_ROOT / FORMAL_OUTPUT_DIRECTORY).resolve()
    if output != expected_output:
        raise RunnerContractError("formal artifacts may be published only to the frozen output directory")
    expected_paths = {
        artifact_id: FORMAL_OUTPUT_DIRECTORY / filename
        for artifact_id, filename in FORMAL_ARTIFACT_NAMES.items()
    }
    if _manifest_output_paths(manifest) != expected_paths:
        raise RunnerContractError("formal output paths differ from the frozen manifest contract")
    require_complete_formal_package(staged)
    output.mkdir(parents=True, exist_ok=True)
    publications = [
        (staged / filename, output / filename)
        for filename in FORMAL_ARTIFACT_NAMES.values()
    ]
    existing = [destination for _, destination in publications if destination.exists()]
    if existing:
        raise RunnerContractError(
            "refusing to overwrite formal artifacts: "
            + ", ".join(path.name for path in existing)
        )

    published: list[tuple[Path, Path]] = []
    try:
        for source, destination in publications:
            os.replace(source, destination)
            published.append((source, destination))
        require_complete_formal_package(output)
    except Exception:
        for source, destination in reversed(published):
            if destination.exists() and not source.exists():
                os.replace(destination, source)
        raise


def execute_jobs_to_directory(
    jobs: Iterable[PlannedJob],
    output_directory: Path,
    manifest: Mapping[str, Any],
    *,
    is_formal_experiment: bool,
    decision_log_mode: str = FORMAL_DEFAULT_DECISION_LOG_MODE,
    full_trace_path: Path | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    planned_jobs = list(jobs)
    if not is_formal_experiment:
        return _execute_jobs_to_artifact_directory(
            planned_jobs,
            output_directory,
            manifest,
            is_formal_experiment=False,
            decision_log_mode=decision_log_mode,
            full_trace_path=full_trace_path,
        )

    output = output_directory.resolve()
    expected_output = (PROJECT_ROOT / FORMAL_OUTPUT_DIRECTORY).resolve()
    if output != expected_output:
        raise RunnerContractError("formal execution requires the frozen output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".final_veto_formal_package_",
        dir=output.parent,
    ) as temporary_name:
        staged = Path(temporary_name)
        arm_rows, pair_rows = _execute_jobs_to_artifact_directory(
            planned_jobs,
            staged,
            manifest,
            is_formal_experiment=True,
            decision_log_mode=decision_log_mode,
            full_trace_path=full_trace_path,
        )
        render_comparison_plot(
            staged / FORMAL_ARTIFACT_NAMES["results"],
            staged / FORMAL_ARTIFACT_NAMES["paired_results"],
            staged / FORMAL_ARTIFACT_NAMES["comparison_plot"],
        )
        require_complete_formal_package(staged)
        publish_complete_formal_package(staged, output, manifest)
        return arm_rows, pair_rows


def _print_plan(
    jobs: Iterable[PlannedJob],
    *,
    decision_log_mode: str = FORMAL_DEFAULT_DECISION_LOG_MODE,
) -> None:
    planned = list(jobs)
    pair_count = len({job.paired_run_id for job in planned})
    preservation = sum(job.known_phase34_recoverable_case for job in planned)
    print(f"PLAN jobs={len(planned)} pairs={pair_count}")
    print(f"PLAN preservation_jobs={preservation} stress_jobs={len(planned) - preservation}")
    print("PLAN arms=monitor_off,monitor_on simulation_started=false artifacts_written=false")
    estimate = estimate_compact_logging_plan(planned, mode=decision_log_mode)
    print(
        "LOGGING "
        f"mode={decision_log_mode} digest_enabled=true "
        f"semantic_record_upper_bound={estimate.semantic_record_upper_bound_without_policy_cap} "
        f"record_limit_per_run={estimate.enforced_record_limit_per_run} "
        f"max_public_records={estimate.maximum_public_records} "
        f"max_public_bytes={estimate.maximum_public_bytes} "
        f"expected_records={estimate.expected_public_records} "
        f"expected_bytes={estimate.expected_serialized_bytes} "
        "overflow_policy=abort_atomic"
    )


def select_smoke_jobs(
    jobs: Iterable[PlannedJob],
    case_id: str | None = None,
) -> list[PlannedJob]:
    planned = list(jobs)
    if not planned:
        raise RunnerContractError("the frozen plan contains no jobs")
    selected_case_id = case_id or planned[0].case_id
    selected = [job for job in planned if job.case_id == selected_case_id]
    if not selected:
        raise RunnerContractError(f"smoke case is not in the frozen plan: {selected_case_id}")
    if (
        len(selected) != 2
        or {job.arm_id for job in selected} != set(VALID_ARM_IDS)
        or len({job.paired_run_id for job in selected}) != 1
    ):
        raise RunnerContractError(
            f"smoke case does not resolve to one complete off/on pair: {selected_case_id}"
        )
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute the frozen Final Veto paired ablation."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Validate and print the 26-job plan only.")
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifest, pairing, paths, and publication readiness without simulation.",
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run one explicitly nonformal pair in a supplied output directory.",
    )
    mode.add_argument(
        "--formal",
        action="store_true",
        help="Run all 26 jobs only after every formal preflight gate passes.",
    )
    mode.add_argument(
        "--formal-preflight",
        action="store_true",
        help="Evaluate formal gates without running any rollout.",
    )
    parser.add_argument("--output-dir", type=Path, help="Required nonformal output directory for --smoke.")
    parser.add_argument(
        "--smoke-case-id",
        help="Exact frozen case ID to run with --smoke; defaults to the first frozen pair.",
    )
    parser.add_argument(
        "--decision-log-mode",
        choices=VALID_DECISION_LOG_MODES,
        default=FORMAL_DEFAULT_DECISION_LOG_MODE,
        help="Formal defaults to bounded compact logging; full_trace is explicit nonformal only.",
    )
    parser.add_argument(
        "--full-trace-path",
        type=Path,
        help="Required external local path for explicit nonformal full_trace mode.",
    )
    parser.add_argument("--manifest", type=Path, help="Alternate manifest path for testing only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest_path = args.manifest.resolve() if args.manifest else None
        manifest = load_frozen_manifest(PROJECT_ROOT, manifest_path)
        jobs = build_planned_jobs(manifest)
    except (ManifestValidationError, RunnerContractError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1

    if args.smoke_case_id and not args.smoke:
        print("FAIL --smoke-case-id requires --smoke", file=sys.stderr)
        return 1

    if args.validate_only:
        _print_plan(jobs, decision_log_mode=args.decision_log_mode)
        blockers = publication_readiness_errors(manifest, PROJECT_ROOT)
        blockers.extend(
            logging_configuration_errors(
                mode=args.decision_log_mode,
                is_formal_experiment=True,
                full_trace_path=args.full_trace_path,
                repository_root=PROJECT_ROOT,
                digest_enabled=True,
            )
        )
        blockers.extend(
            compact_logging_preflight_errors(
                jobs,
                mode=args.decision_log_mode,
                digest_enabled=True,
            )
        )
        if blockers:
            for blocker in blockers:
                print(f"NOT_READY {blocker}")
            print("VALID structural plan passed; formal publication is not ready")
        else:
            print("VALID structural plan and formal publication paths are ready")
        return 0

    if args.formal_preflight or args.formal:
        errors = formal_preflight_errors(
            manifest,
            jobs,
            PROJECT_ROOT,
            run_tests=True,
            decision_log_mode=args.decision_log_mode,
            digest_enabled=True,
            full_trace_path=args.full_trace_path,
        )
        if errors:
            for error in errors:
                print(f"REFUSE {error}")
            print("FORMAL_EXECUTION_REFUSED no rollout started")
            return 2
        if args.formal_preflight:
            print("FORMAL_PREFLIGHT_PASS no rollout started")
            return 0
        execute_jobs_to_directory(
            jobs,
            PROJECT_ROOT / FORMAL_OUTPUT_DIRECTORY,
            manifest,
            is_formal_experiment=True,
            decision_log_mode=args.decision_log_mode,
            full_trace_path=args.full_trace_path,
        )
        return 0

    if args.smoke:
        if args.output_dir is None:
            print("FAIL --smoke requires --output-dir", file=sys.stderr)
            return 1
        output = args.output_dir.resolve()
        if output == (PROJECT_ROOT / FORMAL_OUTPUT_DIRECTORY).resolve():
            print("FAIL smoke output may not use the reserved formal directory", file=sys.stderr)
            return 1
        logging_errors = logging_configuration_errors(
            mode=args.decision_log_mode,
            is_formal_experiment=False,
            full_trace_path=args.full_trace_path,
            repository_root=PROJECT_ROOT,
            digest_enabled=True,
        )
        if logging_errors:
            for error in logging_errors:
                print(f"FAIL {error}", file=sys.stderr)
            return 1
        try:
            smoke_jobs = select_smoke_jobs(jobs, args.smoke_case_id)
        except RunnerContractError as exc:
            print(f"FAIL {exc}", file=sys.stderr)
            return 1
        execute_jobs_to_directory(
            smoke_jobs,
            output,
            manifest,
            is_formal_experiment=False,
            decision_log_mode=args.decision_log_mode,
            full_trace_path=args.full_trace_path,
        )
        print(f"SMOKE_NONFORMAL pairs=1 output={output}")
        return 0

    if args.decision_log_mode == LOG_MODE_FULL_TRACE or args.full_trace_path is not None:
        print("FAIL full_trace mode is available only with --smoke", file=sys.stderr)
        return 1
    _print_plan(jobs, decision_log_mode=args.decision_log_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
