from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
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
)


FORMAL_OUTPUT_DIRECTORY = Path("analysis/final_veto_ablation_v0")
FORMAL_ARTIFACT_NAMES = {
    "results": "results.csv",
    "paired_results": "paired_results.csv",
    "decision_log": "decision_log.jsonl",
    "summary": "summary.md",
    "comparison_plot": "comparison.png",
}
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
            if self.event_sink is not None:
                self.event_sink(self._invalid_event(context, exc))
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
        if self.event_sink is not None:
            self.event_sink(
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
            "decision_reason": "overspeed_risk" if decision.veto_applied else "unknown",
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
            "veto_status": "blocked" if decision.veto_applied else "allow",
            "veto_reason": decision.reason,
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
        "terminal_label": _terminal_label(legacy, invalid_simulation),
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
        "steps": int(legacy.get("steps", 0) or 0),
        "accepted_as_progress": False,
        "acceptance_reason": (
            "raw formal arm requires paired and aggregate validation"
            if is_formal_experiment
            else "nonformal smoke run"
        ),
        "is_formal_experiment": is_formal_experiment,
    }


def execute_job(
    job: PlannedJob,
    *,
    implementation_commit: str,
    is_formal_experiment: bool,
    event_sink: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    recorder = ArmHookRecorder(
        job,
        is_formal_experiment=is_formal_experiment,
        event_sink=event_sink,
    )
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
        return build_arm_record(
            job,
            legacy,
            recorder,
            implementation_commit=implementation_commit,
            is_formal_experiment=is_formal_experiment,
        )
    except (ArmExecutionError, ArithmeticError, ValueError) as exc:
        return build_arm_record(
            job,
            {},
            recorder,
            implementation_commit=implementation_commit,
            is_formal_experiment=is_formal_experiment,
            invalid_simulation=True,
            manual_audit_note=f"execution failed with {type(exc).__name__}; manual audit required",
        )


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
            "step_delta": int(on.get("steps", 0)) - int(off.get("steps", 0)),
            "veto_count": on_veto_count,
        },
        "claim_eligible": claim_eligible,
        "claim_ineligibility_reason": "; ".join(reasons),
        "is_formal_experiment": is_formal,
    }


def build_pair_records(arm_records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in arm_records:
        grouped.setdefault(str(record.get("paired_run_id", "")), []).append(record)
    return [build_pair_record(grouped[pair_id]) for pair_id in sorted(grouped)]


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
                relative.as_posix(),
            ],
            cwd=repository_root,
            check=False,
        )
        if ignored.returncode == 0:
            errors.append(f"formal output is ignored by .gitignore: {relative.as_posix()}")
    return errors


def formal_preflight_errors(
    manifest: Mapping[str, Any],
    jobs: Iterable[PlannedJob],
    repository_root: Path = PROJECT_ROOT,
    *,
    run_tests: bool = True,
) -> list[str]:
    errors: list[str] = []
    try:
        validate_manifest_data(dict(manifest))
        validate_planned_jobs(jobs)
    except (ManifestValidationError, RunnerContractError) as exc:
        errors.append(f"contract validation failed: {exc}")

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
        modules = (
            "Tests.test_final_veto_manifest",
            "Tests.test_final_veto_transition",
            "Tests.test_final_veto_monitor",
            "Tests.test_final_veto_runner",
            "Tests.test_final_veto_artifacts",
            "Tests.test_final_veto_result_validator",
        )
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "-q", *modules],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            errors.append("bounded Final Veto test suite has not passed")
    return errors


def execute_jobs_to_directory(
    jobs: Iterable[PlannedJob],
    output_directory: Path,
    manifest: Mapping[str, Any],
    *,
    is_formal_experiment: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output = validate_output_directory(
        output_directory,
        repository_root=PROJECT_ROOT,
        protected_paths=manifest["protected_paths"],
    )
    commit = current_commit()
    arm_rows: list[dict[str, object]] = []
    decision_name = "decision_log.jsonl" if is_formal_experiment else "smoke_decision_log.jsonl"
    with JsonlEventWriter(
        output,
        decision_name,
        repository_root=PROJECT_ROOT,
        protected_paths=manifest["protected_paths"],
    ) as decision_writer:
        for job in jobs:
            arm_rows.append(
                execute_job(
                    job,
                    implementation_commit=commit,
                    is_formal_experiment=is_formal_experiment,
                    event_sink=decision_writer.write_event,
                )
            )
    pair_rows = build_pair_records(arm_rows)
    from scripts.check_final_veto_results import validate_result_records

    structural_report = validate_result_records(
        arm_rows,
        pair_rows,
        manifest,
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
    return arm_rows, pair_rows


def _print_plan(jobs: Iterable[PlannedJob]) -> None:
    planned = list(jobs)
    pair_count = len({job.paired_run_id for job in planned})
    preservation = sum(job.known_phase34_recoverable_case for job in planned)
    print(f"PLAN jobs={len(planned)} pairs={pair_count}")
    print(f"PLAN preservation_jobs={preservation} stress_jobs={len(planned) - preservation}")
    print("PLAN arms=monitor_off,monitor_on simulation_started=false artifacts_written=false")


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

    if args.validate_only:
        _print_plan(jobs)
        blockers = publication_readiness_errors(manifest, PROJECT_ROOT)
        if blockers:
            for blocker in blockers:
                print(f"NOT_READY {blocker}")
            print("VALID structural plan passed; formal publication is not ready")
        else:
            print("VALID structural plan and formal publication paths are ready")
        return 0

    if args.formal_preflight or args.formal:
        errors = formal_preflight_errors(manifest, jobs, PROJECT_ROOT, run_tests=True)
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
        first_pair_id = jobs[0].paired_run_id
        smoke_jobs = [job for job in jobs if job.paired_run_id == first_pair_id]
        execute_jobs_to_directory(
            smoke_jobs,
            output,
            manifest,
            is_formal_experiment=False,
        )
        print(f"SMOKE_NONFORMAL pairs=1 output={output}")
        return 0

    _print_plan(jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
