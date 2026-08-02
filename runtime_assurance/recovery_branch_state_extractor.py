from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence

from runtime_assurance.final_veto_monitor import (
    MONITOR_ID,
    OVERSPEED_COMPARATOR,
    OVERSPEED_THRESHOLD,
    OneStepPrediction,
    evaluate_overspeed_veto,
)
from runtime_assurance.final_veto_runner_types import (
    ActionInterceptionResult,
    PreTransitionActionContext,
)
from runtime_assurance.recovery_branch_state_registry import (
    BRANCH_STEP,
    COMPLETED_DATE,
    CONFIG_PATH,
    CONFIG_SCHEMA_VERSION,
    LEGACY_ARTIFACT_PATH,
    LEGACY_CASE_ID,
    LEGACY_MEMBER_ID,
    MANIFEST_PATH,
    OUTPUT_PATH,
    PREFIX_TRANSITION_COUNT,
    REGISTRY_ID,
    REGISTRY_MEMBER_SCHEMA_VERSION,
    REGISTRY_SCHEMA_VERSION,
    BranchStateRegistryError,
    RegistryMember,
    attach_generated_hashes,
    canonical_json_bytes,
    canonical_json_file_bytes,
    canonical_sha256,
    file_sha256,
    load_branch_state_registry,
    load_registered_branch_state,
    manifest_scientific_payload,
    registry_aggregate_hash,
    validate_generated_branch_state_document,
)
from runtime_assurance.recovery_evaluators import (
    PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
    PHASE34_RECOVERABLE_VR_RATIO_MAX,
    PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
)
from scripts.run_final_veto_ablation import (
    PlannedJob,
    build_planned_jobs,
    load_frozen_manifest,
)
from simulator.phase34_35_transition import (
    ACTION_COMPONENT_MAX,
    ACTION_COMPONENT_MIN,
    GRAVITY_DENOMINATOR_EPSILON,
    CartesianState2D,
)


FINAL_VETO_MANIFEST_PATH = Path("analysis/final_veto_ablation_v0/manifest.json")
TRANSITION_IMPLEMENTATION_PATH = Path("simulator/phase34_35_transition.py")
SOURCE_CASE_COUNT = 13
RESULT_ARTIFACT_FILENAMES = (
    "branch_state_index.json",
    "determinism_report.json",
    "prefix_execution_report.json",
    "registry_manifest.json",
    "selection_report.json",
    "source_case_inventory.json",
    "summary.md",
)

PROTECTED_HASH_GROUPS: dict[str, tuple[str, ...]] = {
    "historical_phase34_37": (
        "analysis/phase34_post_cross_sync",
        "analysis/phase35_crossing_basin_expansion",
        "analysis/phase36b_transfer_family_benchmark",
        "analysis/phase36c_non_crossing_geometry_diagnosis",
        "analysis/phase37a_radial_commit_timing",
        "analysis/phase37b_weak_tangential_subset",
        "scripts/check_phase_results.py",
    ),
    "final_veto_v0": ("analysis/final_veto_ablation_v0",),
    "frozen_recovery_inputs": (
        "analysis/recovery_action_branching_nonformal_v0/manifest.json",
        "analysis/recovery_action_branching_nonformal_v0/branch_state.json",
    ),
    "published_recovery_results": (
        "analysis/recovery_action_branching_nonformal_v0/results.csv",
        "analysis/recovery_action_branching_nonformal_v0/decision_log.jsonl",
        "analysis/recovery_action_branching_nonformal_v0/summary.md",
        "analysis/recovery_action_branching_nonformal_v0/comparison.png",
    ),
    "recovery_mechanism_diagnosis_v0": (
        "analysis/recovery_branch_mechanism_diagnosis_v0",
    ),
    "staged_recovery_architecture_v0": (
        "runtime_assurance/staged_recovery_contract.py",
        "Tests/test_staged_recovery_contract.py",
        "docs/architecture/staged_recovery_architecture_v0.md",
        "docs/experiments/staged_recovery_minimal_experiment_plan_v0.md",
        "analysis/staged_recovery_architecture_v0",
    ),
    "staged_recovery_instrumentation_v0": (
        "runtime_assurance/staged_recovery_instrumentation.py",
        "scripts/check_staged_recovery_instrumentation.py",
        "Tests/test_staged_recovery_instrumentation.py",
        "docs/architecture/staged_recovery_instrumentation_v0.md",
        "analysis/staged_recovery_instrumentation_v0",
    ),
    "staged_recovery_runtime_logger_v0": (
        "runtime_assurance/staged_recovery_runtime_logger.py",
        "scripts/check_staged_recovery_runtime_logger.py",
        "Tests/test_staged_recovery_runtime_logger.py",
        "docs/architecture/staged_recovery_runtime_logger_v0.md",
        "analysis/staged_recovery_runtime_logger_v0",
    ),
    "staged_recovery_instrumentation_validation_v0": (
        "runtime_assurance/staged_recovery_logger_adapter.py",
        "runtime_assurance/staged_recovery_instrumentation_validation.py",
        "scripts/run_staged_recovery_instrumentation_validation_v0.py",
        "scripts/check_staged_recovery_instrumentation_validation.py",
        "Tests/test_staged_recovery_logger_adapter.py",
        "Tests/test_staged_recovery_instrumentation_validation.py",
        "docs/experiments/staged_recovery_instrumentation_validation_v0.md",
        "analysis/staged_recovery_instrumentation_validation_v0",
    ),
    "staged_recovery_guard_evidence_v0": (
        "runtime_assurance/staged_recovery_guard_evidence.py",
        "scripts/analyze_staged_recovery_guard_evidence_v0.py",
        "scripts/check_staged_recovery_guard_evidence.py",
        "Tests/test_staged_recovery_guard_evidence.py",
        "docs/architecture/staged_recovery_guard_evidence_v0.md",
        "analysis/staged_recovery_guard_evidence_v0",
    ),
    "legacy_canonical_branch_state": (
        "analysis/recovery_action_branching_nonformal_v0/branch_state.json",
    ),
}

KNOWN_PROTECTED_HASHES = {
    "historical_phase34_37": "5be1ab928ad018c433c97869a3ffb7ad796ba8a49c9eeaedd901a410245a8501",
    "final_veto_v0": "125a3f064e288eca471553b8335c757501c9c771f92aa4a711d88e6faba8fb95",
    "frozen_recovery_inputs": "241c780206bab4a1ca892159a9c310852cf538cd5f7efa254e0bf9c83d258622",
    "published_recovery_results": "1745d0c307547f0285793db6e64019ca03d1cc40f23829d68073b5e01ee49037",
    "recovery_mechanism_diagnosis_v0": "482c4f75cc85197f4057d6b77cf24617bc51aae8937f174eb6c25b65ecbecd02",
    "staged_recovery_architecture_v0": "68aa75b26ad1e9d4e9a7f1eba168d13cdf36ae48b44cf6e3885fca95320ad217",
}

COMMON_CONTROLLER_SOURCE_PATHS = (
    Path("controller/orbit_lock_controller.py"),
    Path("runtime_assurance/final_veto_monitor.py"),
    Path("runtime_assurance/final_veto_runner_types.py"),
    Path("scripts/explicit_controller_phase21_orbital_transfer_planner.py"),
    Path("scripts/explicit_controller_phase22_two_burn_transfer.py"),
)
PHASE34_CONTROLLER_SOURCE_PATH = Path(
    "scripts/explicit_controller_phase34_post_cross_sync.py"
)
PHASE35_CONTROLLER_SOURCE_PATH = Path(
    "scripts/explicit_controller_phase35_crossing_basin_expansion.py"
)


class BranchStateExtractionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SourceCaseDefinition:
    case_id: str
    subset_id: str
    seed: int
    r0_over_target: float
    initial_velocity_angle_deg: float
    thrust_scale: float
    controller_id: str
    post_cross_mode: str
    upstream_variant: str | None
    source_case_artifact: str
    source_case_hash: str
    source_configuration_hash: str
    source_commit: str
    nominal_prefix_transition_count: int
    nominal_controller_hash: str
    transition_implementation_hash: str
    eligible_for_generation: bool
    ineligibility_reason: str | None

    @property
    def registry_member_id(self) -> str:
        return f"member__{self.case_id}"

    @property
    def artifact_filename(self) -> str:
        return f"{self.case_id}.json"

    def inventory_document(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "source_artifact": self.source_case_artifact,
            "source_hash": self.source_case_hash,
            "configuration_hash": self.source_configuration_hash,
            "predicted_speed_ratio_if_available": None,
            "overspeed_class_if_available": None,
            "nominal_prefix_contract_available": self.nominal_prefix_transition_count
            == PREFIX_TRANSITION_COUNT,
            "initialization_available": self.eligible_for_generation,
            "simulator_configuration_available": self.eligible_for_generation,
            "controller_configuration_available": self.eligible_for_generation,
            "eligible_for_generation": self.eligible_for_generation,
            "ineligibility_reason": self.ineligibility_reason,
            "controller_id": self.controller_id,
            "post_cross_mode": self.post_cross_mode,
            "upstream_variant": self.upstream_variant,
            "r0_over_target": self.r0_over_target,
            "initial_velocity_angle_deg": self.initial_velocity_angle_deg,
            "thrust_scale": self.thrust_scale,
            "nominal_prefix_transition_count": self.nominal_prefix_transition_count,
            "nominal_controller_hash": self.nominal_controller_hash,
            "transition_implementation_hash": self.transition_implementation_hash,
        }


@dataclass(frozen=True, slots=True)
class PrefixExecutionResult:
    execution_id: str
    case: SourceCaseDefinition
    execution_role: str
    document_json: str
    actual_transition_count: int
    branch_step: int
    initial_state_hash: str
    prefix_action_trace_hash: str
    prefix_state_trace_hash: str
    canonical_payload_hash: str
    predicted_speed_ratio: float
    tangential_velocity_error_ratio: float

    def document(self) -> dict[str, object]:
        value = json.loads(self.document_json)
        if not isinstance(value, dict):
            raise BranchStateExtractionError("execution document is not an object")
        return value


@dataclass(frozen=True, slots=True)
class LegacyReproductionResult:
    document_json: str
    initial_state_hash: str
    prefix_action_trace_hash: str
    prefix_state_trace_hash: str
    actual_transition_count: int
    branch_step: int

    def document(self) -> dict[str, object]:
        value = json.loads(self.document_json)
        if not isinstance(value, dict):
            raise BranchStateExtractionError("legacy reproduction is not an object")
        return value


@dataclass(frozen=True, slots=True)
class RegistrySelection:
    member_a_case_id: str
    member_b_case_id: str
    member_c_case_id: str
    member_d_case_id: str

    @property
    def generated_case_ids(self) -> tuple[str, str, str]:
        return (
            self.member_b_case_id,
            self.member_c_case_id,
            self.member_d_case_id,
        )


@dataclass(frozen=True, slots=True)
class RegistryStaticValidationReport:
    valid: bool
    errors: tuple[str, ...]
    source_case_count: int
    eligible_case_count: int
    ineligible_case_count: int
    tracked_clean: bool
    staged_clean: bool
    head_commit: str


@dataclass(frozen=True, slots=True)
class RegistryPublicationResult:
    target_directory: str
    artifact_paths: tuple[str, ...]
    artifact_hashes: tuple[tuple[str, str], ...]
    registry_manifest_hash: str
    registry_aggregate_hash: str
    member_count: int
    total_execution_count: int


def _aggregate_file_hash(repository_root: Path, paths: Sequence[Path]) -> str:
    records: list[dict[str, str]] = []
    for relative in sorted(paths, key=lambda item: item.as_posix()):
        path = repository_root / relative
        if not path.is_file():
            raise BranchStateExtractionError(f"required source file is missing: {relative}")
        records.append({"path": relative.as_posix(), "sha256": file_sha256(path)})
    return canonical_sha256(records)


def _case_payload(job: PlannedJob, source_commit: str) -> dict[str, object]:
    return {
        "case_id": job.case_id,
        "subset_id": job.subset_id,
        "seed": job.seed,
        "r0_over_target": job.r0_over_target,
        "initial_velocity_angle_deg": job.initial_velocity_angle_deg,
        "thrust_scale": job.thrust_scale,
        "controller_id": job.controller_id,
        "controller_family": job.controller_family,
        "post_cross_mode": job.post_cross_mode,
        "upstream_variant": job.upstream_variant,
        "source_commit": source_commit,
    }


def load_registry_config(repository_root: Path) -> dict[str, object]:
    path = repository_root / CONFIG_PATH
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BranchStateExtractionError(f"cannot load registry configuration: {exc}") from exc
    if not isinstance(value, dict):
        raise BranchStateExtractionError("registry configuration must be an object")
    if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise BranchStateExtractionError("registry configuration schema mismatch")
    if value.get("nominal_prefix_transition_count") != PREFIX_TRANSITION_COUNT:
        raise BranchStateExtractionError("registry configuration prefix count mismatch")
    if value.get("branch_step") != BRANCH_STEP:
        raise BranchStateExtractionError("registry configuration branch step mismatch")
    if value.get("source_case_count") != SOURCE_CASE_COUNT:
        raise BranchStateExtractionError("registry configuration source-case count mismatch")
    if value.get("overspeed_threshold") != OVERSPEED_THRESHOLD:
        raise BranchStateExtractionError("registry configuration overspeed threshold mismatch")
    if value.get("overspeed_comparator") != OVERSPEED_COMPARATOR:
        raise BranchStateExtractionError("registry configuration comparator mismatch")
    return value


def build_source_case_inventory(
    repository_root: Path,
) -> tuple[SourceCaseDefinition, ...]:
    root = repository_root.resolve()
    config = load_registry_config(root)
    manifest = load_frozen_manifest(root)
    jobs = build_planned_jobs(manifest)
    monitor_off = sorted(
        (job for job in jobs if job.arm_id == "monitor_off"),
        key=lambda item: (item.case_id, item.seed, item.case_config_hash),
    )
    if len(monitor_off) != SOURCE_CASE_COUNT:
        raise BranchStateExtractionError(
            f"expected {SOURCE_CASE_COUNT} source cases, found {len(monitor_off)}"
        )
    source_commit = str(manifest.get("source_commit", ""))
    if not source_commit:
        raise BranchStateExtractionError("Final Veto source commit is missing")
    transition_hash = _aggregate_file_hash(root, (TRANSITION_IMPLEMENTATION_PATH,))
    definitions: list[SourceCaseDefinition] = []
    for job in monitor_off:
        controller_path = (
            PHASE34_CONTROLLER_SOURCE_PATH
            if job.upstream_variant is None
            else PHASE35_CONTROLLER_SOURCE_PATH
        )
        controller_hash = _aggregate_file_hash(
            root, (*COMMON_CONTROLLER_SOURCE_PATHS, controller_path)
        )
        required = [
            root / FINAL_VETO_MANIFEST_PATH,
            root / TRANSITION_IMPLEMENTATION_PATH,
            root / controller_path,
        ]
        eligible = (
            all(path.is_file() for path in required)
            and job.seed == 0
            and job.controller_id
            in {"phase34_post_cross_sync", "phase35_crossing_basin_expansion"}
            and job.post_cross_mode == "radius_priority"
            and (
                job.upstream_variant is None
                or job.upstream_variant == "radial_energy_push"
            )
        )
        reason = None if eligible else "complete frozen initialization or controller provenance unavailable"
        definitions.append(
            SourceCaseDefinition(
                case_id=job.case_id,
                subset_id=job.subset_id,
                seed=job.seed,
                r0_over_target=job.r0_over_target,
                initial_velocity_angle_deg=job.initial_velocity_angle_deg,
                thrust_scale=job.thrust_scale,
                controller_id=job.controller_id,
                post_cross_mode=job.post_cross_mode,
                upstream_variant=job.upstream_variant,
                source_case_artifact=FINAL_VETO_MANIFEST_PATH.as_posix(),
                source_case_hash=canonical_sha256(_case_payload(job, source_commit)),
                source_configuration_hash=job.case_config_hash,
                source_commit=source_commit,
                nominal_prefix_transition_count=int(
                    config["nominal_prefix_transition_count"]
                ),
                nominal_controller_hash=controller_hash,
                transition_implementation_hash=transition_hash,
                eligible_for_generation=eligible,
                ineligibility_reason=reason,
            )
        )
    case_ids = tuple(item.case_id for item in definitions)
    if len(case_ids) != len(set(case_ids)):
        raise BranchStateExtractionError("source inventory contains duplicate case IDs")
    return tuple(definitions)


def source_inventory_document(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve()
    cases = build_source_case_inventory(root)
    eligible = sum(item.eligible_for_generation for item in cases)
    document: dict[str, object] = {
        "schema_version": "recovery_branch_state_source_inventory_v0",
        "source_artifact": FINAL_VETO_MANIFEST_PATH.as_posix(),
        "source_artifact_hash": file_sha256(root / FINAL_VETO_MANIFEST_PATH),
        "source_case_count": len(cases),
        "eligible_case_count": eligible,
        "ineligible_case_count": len(cases) - eligible,
        "cases": [item.inventory_document() for item in cases],
    }
    document["canonical_payload_hash"] = canonical_sha256(document)
    return document


def _simulator_configuration(case: SourceCaseDefinition) -> dict[str, object]:
    from scripts import explicit_controller_phase21_orbital_transfer_planner as phase21

    target_radius = phase21.DEFAULT_TARGET_RADIUS * phase21.TARGET_RADIUS_SCALE
    target_speed = math.sqrt(phase21.MU / target_radius)
    constants: dict[str, object] = {
        "action_component_max": ACTION_COMPONENT_MAX,
        "action_component_min": ACTION_COMPONENT_MIN,
        "dt": phase21.DT,
        "gravity_denominator_epsilon": GRAVITY_DENOMINATOR_EPSILON,
        "integration_order": "velocity_then_position_using_updated_velocity",
        "mass": phase21.MASS,
        "max_steps": phase21.MAX_STEPS,
        "mu": phase21.MU,
        "rollout_overspeed_comparator": OVERSPEED_COMPARATOR,
        "rollout_overspeed_threshold": OVERSPEED_THRESHOLD,
        "speed_ratio_denominator_epsilon": 1.0e-12,
        "target_circular_speed": target_speed,
        "target_radius": target_radius,
        "target_radius_scale": phase21.TARGET_RADIUS_SCALE,
        "transition_function": (
            "simulator.phase34_35_transition.step_phase34_35_transition"
        ),
    }
    return {"simulator_constants": constants, "thrust_scale": case.thrust_scale}


def _state_document(state: CartesianState2D) -> dict[str, float]:
    return {
        "position_x": state.x,
        "position_y": state.y,
        "velocity_x": state.vx,
        "velocity_y": state.vy,
    }


class _PrefixCaptured(RuntimeError):
    def __init__(self, context: PreTransitionActionContext, decision: object, predicted: OneStepPrediction):
        self.context = context
        self.decision = decision
        self.predicted = predicted
        super().__init__("frozen nominal-prefix boundary captured")


class _FixedPrefixHook:
    def __init__(self, case: SourceCaseDefinition):
        self.case = case
        self.expected_step = 1
        self.states: list[dict[str, float]] = []
        self.actions: list[list[float]] = []

    def __call__(self, context: PreTransitionActionContext) -> ActionInterceptionResult:
        if context.case.case_id != self.case.case_id:
            raise BranchStateExtractionError("prefix hook received wrong source case")
        if context.step != self.expected_step:
            raise BranchStateExtractionError("prefix transition steps are not sequential")
        self.states.append(_state_document(context.current_state))
        if context.step == BRANCH_STEP:
            nominal_prediction: OneStepPrediction | None = None

            def predictor(state: CartesianState2D, action: tuple[float, float]) -> OneStepPrediction:
                nonlocal nominal_prediction
                transition = context.predict_transition(state, action)
                prediction = OneStepPrediction(
                    next_state=transition.next_state,
                    speed_ratio=context.compute_speed_ratio(transition.next_state),
                )
                if state == context.current_state and action == context.nominal_action:
                    nominal_prediction = prediction
                return prediction

            decision = evaluate_overspeed_veto(
                context.current_state,
                context.nominal_action,
                predictor,
                threshold=OVERSPEED_THRESHOLD,
            )
            if nominal_prediction is None:
                raise BranchStateExtractionError("branch prediction was not captured")
            raise _PrefixCaptured(context, decision, nominal_prediction)
        if context.step > BRANCH_STEP:
            raise BranchStateExtractionError("prefix execution passed extraction boundary")
        self.actions.append([context.nominal_action[0], context.nominal_action[1]])
        self.expected_step += 1
        return ActionInterceptionResult(
            nominal_action=context.nominal_action,
            executed_action=context.nominal_action,
            intervention_applied=False,
            decision_metadata=None,
        )


def _derived_values(
    state: CartesianState2D,
    simulator_configuration: Mapping[str, object],
) -> dict[str, object]:
    constants = simulator_configuration["simulator_constants"]
    if not isinstance(constants, dict):
        raise BranchStateExtractionError("simulator constants are unavailable")
    target_radius = float(constants["target_radius"])
    target_speed = float(constants["target_circular_speed"])
    epsilon = float(constants["speed_ratio_denominator_epsilon"])
    radius = math.hypot(state.x, state.y)
    speed = math.hypot(state.vx, state.vy)
    if radius <= 0.0:
        raise BranchStateExtractionError("position norm is nonpositive")
    er_x, er_y = state.x / radius, state.y / radius
    et_x, et_y = -er_y, er_x
    radial_velocity = state.vx * er_x + state.vy * er_y
    tangential_velocity = state.vx * et_x + state.vy * et_y
    signed_error = radius - target_radius
    radius_ratio = signed_error / target_radius
    vr_ratio = radial_velocity / (target_speed + epsilon)
    tangential_error = tangential_velocity - target_speed
    vt_ratio = tangential_error / (target_speed + epsilon)
    speed_ratio = speed / (target_speed + epsilon)
    return {
        "radius": radius,
        "speed": speed,
        "radial_velocity": radial_velocity,
        "tangential_velocity": tangential_velocity,
        "target_radius": target_radius,
        "target_circular_speed": target_speed,
        "signed_radius_error": signed_error,
        "absolute_radius_gap": abs(signed_error),
        "radius_error_ratio": radius_ratio,
        "radial_velocity_ratio": vr_ratio,
        "tangential_velocity_error": tangential_error,
        "tangential_velocity_error_ratio": vt_ratio,
        "realized_speed_ratio": speed_ratio,
        "overspeed_headroom": OVERSPEED_THRESHOLD - speed_ratio,
        "radius_component_pass": abs(radius_ratio)
        <= PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
        "radial_velocity_component_pass": abs(vr_ratio)
        <= PHASE34_RECOVERABLE_VR_RATIO_MAX,
        "tangential_velocity_component_pass": abs(vt_ratio)
        <= PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
        "phase34_compatible_recoverability_pass": (
            abs(radius_ratio) <= PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX
            and abs(vr_ratio) <= PHASE34_RECOVERABLE_VR_RATIO_MAX
            and abs(vt_ratio) <= PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX
        ),
    }


def _build_generated_document(
    case: SourceCaseDefinition,
    hook: _FixedPrefixHook,
    captured: _PrefixCaptured,
    implementation_commit: str,
) -> dict[str, object]:
    context = captured.context
    current = context.current_state
    predicted = captured.predicted.next_state
    simulator = _simulator_configuration(case)
    constants = simulator["simulator_constants"]
    assert isinstance(constants, dict)
    case_configuration = {
        "case_id": case.case_id,
        "controller_id": case.controller_id,
        "initial_velocity_angle_deg": case.initial_velocity_angle_deg,
        "post_cross_mode": case.post_cross_mode,
        "r0_over_target": case.r0_over_target,
        "seed": case.seed,
        "subset_id": case.subset_id,
        "thrust_scale": case.thrust_scale,
        "upstream_variant": case.upstream_variant,
    }
    decision = captured.decision
    document: dict[str, object] = {
        "schema_version": REGISTRY_MEMBER_SCHEMA_VERSION,
        "registry_member_id": case.registry_member_id,
        "case_id": case.case_id,
        "seed": case.seed,
        "state_origin": "deterministic_nominal_prefix_execution",
        "reconstructed_from_log": False,
        "manually_authored_state": False,
        "perturbed_from_existing_state": False,
        "source_commit": case.source_commit,
        "generation_implementation_commit": implementation_commit,
        "generated_date": COMPLETED_DATE,
        "nominal_prefix_transition_count": PREFIX_TRANSITION_COUNT,
        "actual_transition_count": len(hook.actions),
        "branch_step": context.step,
        "initial_state_hash": canonical_sha256(hook.states[0]),
        "prefix_action_count": len(hook.actions),
        "prefix_action_trace_hash": canonical_sha256(hook.actions),
        "prefix_state_trace_hash": canonical_sha256(hook.states),
        "terminal_before_branch": False,
        "terminal_reason_before_branch": None,
        "position_x": current.x,
        "position_y": current.y,
        "velocity_x": current.vx,
        "velocity_y": current.vy,
        "state": {"current_phase": context.phase, **_state_document(current)},
        "state_vector": [current.x, current.y, current.vx, current.vy],
        "phase": context.phase,
        "active_stage": context.active_stage,
        "proposed_action": [context.nominal_action[0], context.nominal_action[1]],
        "nominal_action": [context.nominal_action[0], context.nominal_action[1]],
        "predicted_position_x": predicted.x,
        "predicted_position_y": predicted.y,
        "predicted_velocity_x": predicted.vx,
        "predicted_velocity_y": predicted.vy,
        "predicted_next_state": _state_document(predicted),
        "predicted_speed": math.hypot(predicted.vx, predicted.vy),
        "predicted_speed_ratio": captured.predicted.speed_ratio,
        "monitor_decision": {
            "decision": decision.decision,
            "monitor_id": decision.monitor_id,
            "reason": decision.reason,
            "veto_applied": decision.veto_applied,
        },
        "threshold": OVERSPEED_THRESHOLD,
        "comparator": OVERSPEED_COMPARATOR,
        "case_configuration": case_configuration,
        "case_configuration_hash": canonical_sha256(case_configuration),
        "source_case_artifact": case.source_case_artifact,
        "source_case_hash": case.source_case_hash,
        "source_configuration_hash": case.source_configuration_hash,
        "simulator_configuration": simulator,
        "simulator_configuration_hash": canonical_sha256(simulator),
        "constants_hash": canonical_sha256(constants),
        "transition_implementation_hash": case.transition_implementation_hash,
        "nominal_controller_hash": case.nominal_controller_hash,
        "branch_ordering": {
            "capture_boundary": "after_27_realized_nominal_transitions_before_step_28_action_execution",
            "before_nominal_action_execution": True,
            "monitor_evaluation_completed": True,
            "nominal_action_executed": False,
            "realized_prefix_transition_count": PREFIX_TRANSITION_COUNT,
        },
    }
    document.update(_derived_values(current, simulator))
    return attach_generated_hashes(document)


def execute_nominal_prefix(
    repository_root: Path,
    case: SourceCaseDefinition,
    *,
    execution_role: str,
    execution_id: str,
    implementation_commit: str,
) -> PrefixExecutionResult:
    if execution_role not in {"candidate_discovery", "selected_reproduction"}:
        raise BranchStateExtractionError("unsupported generated prefix execution role")
    if not case.eligible_for_generation:
        raise BranchStateExtractionError("ineligible source case cannot execute")
    hook = _FixedPrefixHook(case)
    try:
        if case.upstream_variant is None:
            from scripts import explicit_controller_phase34_post_cross_sync as phase34

            mode = next(item for item in phase34.MODES if item.name == case.post_cross_mode)
            phase34.rollout_phase34_case(
                mode,
                case.r0_over_target,
                case.initial_velocity_angle_deg,
                case.thrust_scale,
                record_trajectory=False,
                case_id=case.case_id,
                pre_transition_action_hook=hook,
            )
        else:
            from scripts import explicit_controller_phase35_crossing_basin_expansion as phase35

            variant = next(item for item in phase35.VARIANTS if item.name == case.upstream_variant)
            mode = next(item for item in phase35.PHASE34_MODES if item.name == case.post_cross_mode)
            phase35.rollout_phase35_case(
                variant,
                mode,
                case.r0_over_target,
                case.initial_velocity_angle_deg,
                case.thrust_scale,
                record_trajectory=False,
                case_id=case.case_id,
                pre_transition_action_hook=hook,
            )
    except _PrefixCaptured as captured:
        document = _build_generated_document(
            case, hook, captured, implementation_commit
        )
        validate_generated_branch_state_document(document)
        return PrefixExecutionResult(
            execution_id=execution_id,
            case=case,
            execution_role=execution_role,
            document_json=canonical_json_bytes(document).decode("utf-8"),
            actual_transition_count=len(hook.actions),
            branch_step=int(document["branch_step"]),
            initial_state_hash=str(document["initial_state_hash"]),
            prefix_action_trace_hash=str(document["prefix_action_trace_hash"]),
            prefix_state_trace_hash=str(document["prefix_state_trace_hash"]),
            canonical_payload_hash=str(document["canonical_payload_hash"]),
            predicted_speed_ratio=float(document["predicted_speed_ratio"]),
            tangential_velocity_error_ratio=float(
                document["tangential_velocity_error_ratio"]
            ),
        )
    raise BranchStateExtractionError(
        f"source case terminated before fixed branch step {BRANCH_STEP}: {case.case_id}"
    )


def reproduce_legacy_canonical(repository_root: Path) -> LegacyReproductionResult:
    from scripts import explicit_controller_phase35_crossing_basin_expansion as phase35
    from scripts.extract_recovery_branch_state import (
        _BranchPointCaptured,
        _FirstVetoBranchPointHook,
        _require_frozen_source_trajectory,
    )
    from simulator.phase34_35_transition import (
        ACTION_COMPONENT_MAX as LEGACY_ACTION_COMPONENT_MAX,
        ACTION_COMPONENT_MIN as LEGACY_ACTION_COMPONENT_MIN,
        GRAVITY_DENOMINATOR_EPSILON as LEGACY_GRAVITY_EPSILON,
    )

    root = repository_root.resolve()
    _require_frozen_source_trajectory(root)
    variant = next(item for item in phase35.VARIANTS if item.name == "radial_energy_push")
    mode = phase35.PHASE34_TERMINAL_MODE
    target_radius = phase35.DEFAULT_TARGET_RADIUS * phase35.TARGET_RADIUS_SCALE
    target_speed = math.sqrt(phase35.MU / target_radius)
    simulator_constants: dict[str, object] = {
        "action_component_max": LEGACY_ACTION_COMPONENT_MAX,
        "action_component_min": LEGACY_ACTION_COMPONENT_MIN,
        "dt": phase35.DT,
        "gravity_denominator_epsilon": LEGACY_GRAVITY_EPSILON,
        "integration_order": "velocity_then_position_using_updated_velocity",
        "mass": phase35.MASS,
        "max_steps": phase35.MAX_STEPS,
        "mu": phase35.MU,
        "rollout_overspeed_comparator": OVERSPEED_COMPARATOR,
        "rollout_overspeed_threshold": OVERSPEED_THRESHOLD,
        "speed_ratio_denominator_epsilon": 1.0e-12,
        "target_circular_speed": target_speed,
        "target_radius": target_radius,
        "target_radius_scale": phase35.TARGET_RADIUS_SCALE,
        "transition_function": (
            "simulator.phase34_35_transition.step_phase34_35_transition"
        ),
    }
    simulator_configuration = {
        "simulator_constants": simulator_constants,
        "thrust_scale": 8000.0,
    }
    legacy_hook = _FirstVetoBranchPointHook(simulator_configuration)
    states: list[dict[str, float]] = []
    actions: list[list[float]] = []

    def hook(context: PreTransitionActionContext) -> ActionInterceptionResult:
        states.append(_state_document(context.current_state))
        if context.step < BRANCH_STEP:
            actions.append([context.nominal_action[0], context.nominal_action[1]])
        return legacy_hook(context)
    try:
        phase35.rollout_phase35_case(
            variant,
            mode,
            0.98,
            150.0,
            8000.0,
            phase35.TARGET_RADIUS_SCALE,
            record_trajectory=False,
            case_id=LEGACY_CASE_ID,
            pre_transition_action_hook=hook,
        )
    except _BranchPointCaptured as captured:
        if captured.document.get("case_id") != LEGACY_CASE_ID:
            raise BranchStateExtractionError("canonical reproduction returned wrong case")
        if len(actions) != PREFIX_TRANSITION_COUNT or len(states) != BRANCH_STEP:
            raise BranchStateExtractionError("canonical reproduction prefix trace is incomplete")
        return LegacyReproductionResult(
            document_json=canonical_json_bytes(captured.document).decode("utf-8"),
            initial_state_hash=canonical_sha256(states[0]),
            prefix_action_trace_hash=canonical_sha256(actions),
            prefix_state_trace_hash=canonical_sha256(states),
            actual_transition_count=len(actions),
            branch_step=int(captured.document["branch_step"]),
        )
    raise BranchStateExtractionError("canonical source did not reproduce its first-veto boundary")


def select_registry_cases(
    results: Sequence[PrefixExecutionResult],
) -> RegistrySelection:
    by_case = {result.case.case_id: result for result in results}
    if len(by_case) != len(results):
        raise BranchStateExtractionError("candidate discovery contains duplicate cases")
    noncanonical = [item for item in results if item.case.case_id != LEGACY_CASE_ID]
    below = [item for item in noncanonical if item.predicted_speed_ratio <= OVERSPEED_THRESHOLD]
    above = [item for item in noncanonical if item.predicted_speed_ratio > OVERSPEED_THRESHOLD]
    if not below:
        raise BranchStateExtractionError("no eligible distinct closest-below candidate")
    if not above:
        raise BranchStateExtractionError("no eligible distinct closest-above candidate")
    member_b = sorted(
        below,
        key=lambda item: (
            -item.predicted_speed_ratio,
            item.case.case_id,
            item.case.seed,
            item.case.source_configuration_hash,
        ),
    )[0]
    member_c = sorted(
        above,
        key=lambda item: (
            item.predicted_speed_ratio,
            item.case.case_id,
            item.case.seed,
            item.case.source_configuration_hash,
        ),
    )[0]
    excluded = {LEGACY_CASE_ID, member_b.case.case_id, member_c.case.case_id}
    tangential = [item for item in noncanonical if item.case.case_id not in excluded]
    if not tangential:
        raise BranchStateExtractionError("no distinct tangential-challenge candidate")
    member_d = sorted(
        tangential,
        key=lambda item: (
            -abs(item.tangential_velocity_error_ratio),
            item.case.case_id,
            item.case.seed,
            item.canonical_payload_hash,
        ),
    )[0]
    selected = RegistrySelection(
        member_a_case_id=LEGACY_CASE_ID,
        member_b_case_id=member_b.case.case_id,
        member_c_case_id=member_c.case.case_id,
        member_d_case_id=member_d.case.case_id,
    )
    if len(set((selected.member_a_case_id, *selected.generated_case_ids))) != 4:
        raise BranchStateExtractionError("selection did not produce four distinct cases")
    return selected


def compare_prefix_results(
    discovery: PrefixExecutionResult,
    reproduction: PrefixExecutionResult,
) -> dict[str, object]:
    if discovery.case.case_id != reproduction.case.case_id:
        raise BranchStateExtractionError("determinism comparison case mismatch")
    left = discovery.document()
    right = reproduction.document()
    ignored = {"generation_implementation_commit"}
    canonical_equal = discovery.canonical_payload_hash == reproduction.canonical_payload_hash
    cartesian_fields = ("position_x", "position_y", "velocity_x", "velocity_y")
    derived_fields = (
        "radius",
        "speed",
        "radial_velocity",
        "tangential_velocity",
        "radius_error_ratio",
        "radial_velocity_ratio",
        "tangential_velocity_error_ratio",
    )
    predicted_fields = (
        "predicted_position_x",
        "predicted_position_y",
        "predicted_velocity_x",
        "predicted_velocity_y",
        "predicted_speed_ratio",
    )
    report = {
        "case_id": discovery.case.case_id,
        "discovery_or_legacy_hash": discovery.canonical_payload_hash,
        "reproduction_hash": reproduction.canonical_payload_hash,
        "canonical_payload_equal": canonical_equal,
        "Cartesian_state_equal": all(left[field] == right[field] for field in cartesian_fields),
        "derived_state_equal": all(left[field] == right[field] for field in derived_fields),
        "predicted_state_equal": all(left[field] == right[field] for field in predicted_fields),
        "prefix_action_trace_equal": discovery.prefix_action_trace_hash
        == reproduction.prefix_action_trace_hash,
        "prefix_state_trace_equal": discovery.prefix_state_trace_hash
        == reproduction.prefix_state_trace_hash,
        "transition_count_equal": discovery.actual_transition_count
        == reproduction.actual_transition_count,
        "branch_step_equal": discovery.branch_step == reproduction.branch_step,
    }
    del ignored
    report["determinism_status"] = (
        "passed"
        if all(value is True for key, value in report.items() if key.endswith("_equal"))
        else "failed"
    )
    return report


def _git_result(repository_root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repository_root.as_posix()}", *arguments],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )


def repository_state(repository_root: Path) -> tuple[str, bool, bool]:
    root = repository_root.resolve()
    head = _git_result(root, ("rev-parse", "HEAD"))
    tracked = _git_result(root, ("diff", "--quiet"))
    staged = _git_result(root, ("diff", "--cached", "--quiet"))
    if head.returncode != 0:
        raise BranchStateExtractionError(head.stderr.strip() or "cannot resolve repository HEAD")
    return head.stdout.strip(), tracked.returncode == 0, staged.returncode == 0


def _aggregate_paths_hash(repository_root: Path, relative_paths: Sequence[str]) -> str:
    root = repository_root.resolve()
    files: list[Path] = []
    for relative in relative_paths:
        path = (root / PurePosixPath(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise BranchStateExtractionError(f"protected path escapes repository: {relative}") from exc
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
        else:
            raise BranchStateExtractionError(f"protected path is missing: {relative}")
    lines = [
        f"{path.relative_to(root).as_posix()}|{file_sha256(path)}"
        for path in sorted(files, key=lambda item: item.as_posix())
    ]
    return hashlib.sha256(os.linesep.join(lines).encode("utf-8")).hexdigest()


def protected_evidence_hashes(repository_root: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, _aggregate_paths_hash(repository_root, paths))
        for name, paths in sorted(PROTECTED_HASH_GROUPS.items())
    )


def validate_static_contract(
    repository_root: Path,
    *,
    require_output_absent: bool,
) -> RegistryStaticValidationReport:
    root = repository_root.resolve()
    errors: list[str] = []
    head = ""
    tracked_clean = False
    staged_clean = False
    try:
        head, tracked_clean, staged_clean = repository_state(root)
    except BranchStateExtractionError as exc:
        errors.append(str(exc))
    try:
        config = load_registry_config(root)
        inventory = build_source_case_inventory(root)
        if len(inventory) != SOURCE_CASE_COUNT:
            errors.append("source_case_count_mismatch")
        if sum(item.eligible_for_generation for item in inventory) < 4:
            errors.append("fewer_than_four_eligible_source_cases")
        if not any(item.case_id == LEGACY_CASE_ID for item in inventory):
            errors.append("legacy_case_missing_from_source_inventory")
        if config.get("output_path") != OUTPUT_PATH.as_posix():
            errors.append("configured_output_path_mismatch")
    except (BranchStateExtractionError, BranchStateRegistryError) as exc:
        inventory = ()
        errors.append(str(exc))
    try:
        legacy_path = root / LEGACY_ARTIFACT_PATH
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        if not isinstance(legacy, dict):
            raise BranchStateExtractionError("legacy branch state is not an object")
        from runtime_assurance.recovery_branch_executor import validate_branch_state_integrity

        validate_branch_state_integrity(legacy)
        ordering = legacy.get("branch_ordering")
        if not isinstance(ordering, dict):
            errors.append("legacy_branch_ordering_missing")
        elif ordering.get("realized_prefix_transition_count") != PREFIX_TRANSITION_COUNT:
            errors.append("legacy_prefix_count_mismatch")
        if legacy.get("branch_step") != BRANCH_STEP:
            errors.append("legacy_branch_step_mismatch")
    except (OSError, json.JSONDecodeError, BranchStateExtractionError, ValueError) as exc:
        errors.append(f"legacy_validation:{exc}")
    try:
        current_hashes = dict(protected_evidence_hashes(root))
        for name, expected in KNOWN_PROTECTED_HASHES.items():
            if current_hashes.get(name) != expected:
                errors.append(f"protected_evidence_hash_mismatch:{name}")
    except BranchStateExtractionError as exc:
        errors.append(str(exc))
    if require_output_absent and (root / OUTPUT_PATH).exists():
        errors.append("registry_output_already_exists")
    required_files = (
        "runtime_assurance/recovery_branch_state_registry.py",
        "runtime_assurance/recovery_branch_state_extractor.py",
        "scripts/generate_recovery_branch_state_registry_v0.py",
        "scripts/check_recovery_branch_state_registry.py",
        "Tests/test_recovery_branch_state_registry.py",
        "Tests/test_recovery_branch_state_extractor.py",
        "docs/architecture/recovery_branch_state_registry_v0.md",
        "docs/experiments/recovery_branch_state_registry_v0.md",
        CONFIG_PATH.as_posix(),
    )
    for relative in required_files:
        if not (root / relative).is_file():
            errors.append(f"implementation_file_missing:{relative}")
    eligible_count = sum(item.eligible_for_generation for item in inventory)
    return RegistryStaticValidationReport(
        valid=not errors,
        errors=tuple(sorted(set(errors))),
        source_case_count=len(inventory),
        eligible_case_count=eligible_count,
        ineligible_case_count=len(inventory) - eligible_count,
        tracked_clean=tracked_clean,
        staged_clean=staged_clean,
        head_commit=head,
    )


def _attach_document_hash(document: Mapping[str, object]) -> dict[str, object]:
    result = copy.deepcopy(dict(document))
    result.pop("canonical_payload_hash", None)
    result["canonical_payload_hash"] = canonical_sha256(result)
    return result


def _legacy_member(
    repository_root: Path,
    legacy_document: Mapping[str, object],
    source_case: SourceCaseDefinition,
) -> RegistryMember:
    simulator = legacy_document.get("simulator_configuration")
    if not isinstance(simulator, dict):
        raise BranchStateExtractionError("legacy simulator configuration is missing")
    constants = simulator.get("simulator_constants")
    if not isinstance(constants, dict):
        raise BranchStateExtractionError("legacy simulator constants are missing")
    return RegistryMember(
        registry_member_id=LEGACY_MEMBER_ID,
        case_id=LEGACY_CASE_ID,
        seed=int(legacy_document["seed"]),
        artifact_path=LEGACY_ARTIFACT_PATH.as_posix(),
        artifact_scope="legacy_external_artifact",
        state_origin="deterministic_nominal_prefix_execution",
        source_case_artifact=source_case.source_case_artifact,
        source_case_hash=source_case.source_case_hash,
        source_configuration_hash=source_case.source_configuration_hash,
        simulator_configuration_hash=str(legacy_document["simulator_configuration_hash"]),
        constants_hash=str(legacy_document["simulator_constants_hash"]),
        transition_implementation_hash=source_case.transition_implementation_hash,
        nominal_controller_hash=source_case.nominal_controller_hash,
        source_commit=str(legacy_document["source_commit"]),
        nominal_prefix_transition_count=PREFIX_TRANSITION_COUNT,
        branch_step=BRANCH_STEP,
        canonical_branch_state_hash=str(legacy_document["canonical_branch_state_hash"]),
        raw_artifact_hash=file_sha256(repository_root / LEGACY_ARTIFACT_PATH),
        legacy_member=True,
        generation_status="legacy_validated",
        determinism_status="passed",
        executable_status="validated",
    )


def _generated_member(
    result: PrefixExecutionResult,
    artifact_path: str,
    artifact_bytes: bytes,
) -> RegistryMember:
    document = result.document()
    return RegistryMember(
        registry_member_id=result.case.registry_member_id,
        case_id=result.case.case_id,
        seed=result.case.seed,
        artifact_path=artifact_path,
        artifact_scope="registry_local_artifact",
        state_origin="deterministic_nominal_prefix_execution",
        source_case_artifact=result.case.source_case_artifact,
        source_case_hash=result.case.source_case_hash,
        source_configuration_hash=result.case.source_configuration_hash,
        simulator_configuration_hash=str(document["simulator_configuration_hash"]),
        constants_hash=str(document["constants_hash"]),
        transition_implementation_hash=result.case.transition_implementation_hash,
        nominal_controller_hash=result.case.nominal_controller_hash,
        source_commit=result.case.source_commit,
        nominal_prefix_transition_count=result.actual_transition_count,
        branch_step=result.branch_step,
        canonical_branch_state_hash=result.canonical_payload_hash,
        raw_artifact_hash=hashlib.sha256(artifact_bytes).hexdigest(),
        legacy_member=False,
        generation_status="generated_and_validated",
        determinism_status="passed",
        executable_status="validated",
    )


def _selection_report(
    results: Sequence[PrefixExecutionResult],
    selection: RegistrySelection,
    config_hash: str,
    *,
    legacy_predicted_speed_ratio: float,
    legacy_tangential_velocity_error_ratio: float,
    legacy_state_hash: str,
) -> dict[str, object]:
    selected_roles = {
        selection.member_a_case_id: "member_a_legacy_canonical",
        selection.member_b_case_id: "member_b_closest_below",
        selection.member_c_case_id: "member_c_closest_above",
        selection.member_d_case_id: "member_d_strong_tangential_challenge",
    }
    candidates = [
        {
            "case_id": LEGACY_CASE_ID,
            "seed": 0,
            "source_configuration_hash": None,
            "canonical_generated_state_hash": legacy_state_hash,
            "predicted_speed_ratio": legacy_predicted_speed_ratio,
            "overspeed_class": (
                "above" if legacy_predicted_speed_ratio > OVERSPEED_THRESHOLD else "below_or_equal"
            ),
            "absolute_tangential_velocity_error_ratio": abs(legacy_tangential_velocity_error_ratio),
            "below_selection_rank": None,
            "above_selection_rank": None,
            "tangential_selection_rank": None,
            "selected_role": selected_roles[LEGACY_CASE_ID],
            "selection_outcome": "selected as member_a_legacy_canonical by the frozen selection contract",
        }
    ]
    below_order = sorted(
        (item for item in results if item.predicted_speed_ratio <= OVERSPEED_THRESHOLD),
        key=lambda item: (-item.predicted_speed_ratio, item.case.case_id, item.case.seed, item.case.source_configuration_hash),
    )
    above_order = sorted(
        (item for item in results if item.predicted_speed_ratio > OVERSPEED_THRESHOLD),
        key=lambda item: (item.predicted_speed_ratio, item.case.case_id, item.case.seed, item.case.source_configuration_hash),
    )
    tangential_order = sorted(
        results,
        key=lambda item: (-abs(item.tangential_velocity_error_ratio), item.case.case_id, item.case.seed, item.canonical_payload_hash),
    )
    for item in sorted(results, key=lambda value: value.case.case_id):
        role = selected_roles.get(item.case.case_id)
        if role:
            outcome = f"selected as {role} by the frozen deterministic ordering"
        else:
            outcome = "not selected because another eligible case ranked first under each remaining frozen role"
        candidates.append(
            {
                "case_id": item.case.case_id,
                "seed": item.case.seed,
                "source_configuration_hash": item.case.source_configuration_hash,
                "canonical_generated_state_hash": item.canonical_payload_hash,
                "predicted_speed_ratio": item.predicted_speed_ratio,
                "overspeed_class": "above" if item.predicted_speed_ratio > OVERSPEED_THRESHOLD else "below_or_equal",
                "absolute_tangential_velocity_error_ratio": abs(item.tangential_velocity_error_ratio),
                "below_selection_rank": (below_order.index(item) + 1) if item in below_order else None,
                "above_selection_rank": (above_order.index(item) + 1) if item in above_order else None,
                "tangential_selection_rank": tangential_order.index(item) + 1,
                "selected_role": role,
                "selection_outcome": outcome,
            }
        )
    candidates.sort(key=lambda item: str(item["case_id"]))
    return _attach_document_hash(
        {
            "schema_version": "recovery_branch_state_selection_report_v0",
            "selection_rules_frozen_before_execution": True,
            "configuration_hash": config_hash,
            "overspeed_boundary_metric_source": "generated step-28 one-step Final Veto prediction",
            "tangential_challenge_metric_source": "generated branch-state tangential_velocity_error_ratio",
            "tie_break": ["case_id lexical order", "seed", "source configuration or generated-state hash"],
            "eligible_candidates": candidates,
            "ineligible_cases": [],
            "selected_member_a": selection.member_a_case_id,
            "selected_member_b": selection.member_b_case_id,
            "selected_member_c": selection.member_c_case_id,
            "selected_member_d": selection.member_d_case_id,
        }
    )


def _execution_record(
    execution_id: str,
    case_id: str,
    role: str,
    initial_state_hash: str,
    action_hash: str,
    state_hash: str,
    final_hash: str,
    published: bool,
) -> dict[str, object]:
    return {
        "execution_id": execution_id,
        "case_id": case_id,
        "execution_role": role,
        "fresh_initialization": True,
        "seed": 0,
        "nominal_prefix_transition_count": PREFIX_TRANSITION_COUNT,
        "actual_transition_count": PREFIX_TRANSITION_COUNT,
        "branch_step": BRANCH_STEP,
        "terminated_early": False,
        "terminal_reason": None,
        "initial_state_hash": initial_state_hash,
        "prefix_action_trace_hash": action_hash,
        "prefix_state_trace_hash": state_hash,
        "final_branch_state_hash": final_hash,
        "published": published,
        "failure": None,
    }


def build_frozen_registry_payloads(
    repository_root: Path,
    *,
    implementation_commit: str,
) -> tuple[dict[str, bytes], dict[str, object]]:
    root = repository_root.resolve()
    config = load_registry_config(root)
    config_hash = file_sha256(root / CONFIG_PATH)
    inventory = build_source_case_inventory(root)
    eligible = tuple(item for item in inventory if item.eligible_for_generation)
    if len(eligible) != SOURCE_CASE_COUNT:
        raise BranchStateExtractionError("frozen generation requires all 13 provenance-complete cases")
    case_map = {item.case_id: item for item in eligible}
    legacy_path = root / LEGACY_ARTIFACT_PATH
    legacy_document = json.loads(legacy_path.read_text(encoding="utf-8"))
    if not isinstance(legacy_document, dict):
        raise BranchStateExtractionError("legacy artifact is not an object")

    legacy_reproduction = reproduce_legacy_canonical(root)
    reproduced_legacy = legacy_reproduction.document()
    legacy_hash = str(legacy_document.get("canonical_branch_state_hash"))
    if reproduced_legacy.get("canonical_branch_state_hash") != legacy_hash:
        raise BranchStateExtractionError("fresh canonical reproduction hash mismatch")
    if reproduced_legacy != legacy_document:
        raise BranchStateExtractionError("fresh canonical reproduction payload mismatch")

    discoveries: list[PrefixExecutionResult] = []
    for index, case in enumerate(
        (item for item in eligible if item.case_id != LEGACY_CASE_ID), start=1
    ):
        discoveries.append(
            execute_nominal_prefix(
                root,
                case,
                execution_role="candidate_discovery",
                execution_id=f"candidate_discovery_{index:02d}",
                implementation_commit=implementation_commit,
            )
        )
    selection = select_registry_cases(discoveries)
    discovery_by_case = {item.case.case_id: item for item in discoveries}
    reproductions: list[PrefixExecutionResult] = []
    for index, case_id in enumerate(selection.generated_case_ids, start=1):
        reproductions.append(
            execute_nominal_prefix(
                root,
                case_map[case_id],
                execution_role="selected_reproduction",
                execution_id=f"selected_reproduction_{index:02d}",
                implementation_commit=implementation_commit,
            )
        )
    determinism_records = [
        {
            "case_id": LEGACY_CASE_ID,
            "discovery_or_legacy_hash": legacy_hash,
            "reproduction_hash": reproduced_legacy["canonical_branch_state_hash"],
            "canonical_payload_equal": True,
            "Cartesian_state_equal": legacy_document["state"] == reproduced_legacy["state"],
            "derived_state_equal": True,
            "predicted_state_equal": legacy_document["predicted_next_state"] == reproduced_legacy["predicted_next_state"],
            "prefix_action_trace_equal": None,
            "prefix_state_trace_equal": None,
            "prefix_trace_limitation": "legacy artifact did not store trace hashes; fresh reproduction established them without a second canonical execution",
            "transition_count_equal": legacy_reproduction.actual_transition_count == PREFIX_TRANSITION_COUNT,
            "branch_step_equal": legacy_reproduction.branch_step == BRANCH_STEP,
            "determinism_status": "passed",
        }
    ]
    reproduction_by_case = {item.case.case_id: item for item in reproductions}
    for case_id in selection.generated_case_ids:
        comparison = compare_prefix_results(
            discovery_by_case[case_id], reproduction_by_case[case_id]
        )
        if comparison["determinism_status"] != "passed":
            raise BranchStateExtractionError(f"generated member determinism failed: {case_id}")
        determinism_records.append(comparison)

    payloads: dict[str, bytes] = {}
    generated_members: list[RegistryMember] = []
    for case_id in selection.generated_case_ids:
        result = discovery_by_case[case_id]
        document = result.document()
        artifact_relative = (
            OUTPUT_PATH / "branch_states" / result.case.artifact_filename
        ).as_posix()
        artifact_bytes = canonical_json_file_bytes(document)
        payload_key = f"branch_states/{result.case.artifact_filename}"
        payloads[payload_key] = artifact_bytes
        generated_members.append(
            _generated_member(result, artifact_relative, artifact_bytes)
        )
    members = tuple(
        sorted(
            (
                _legacy_member(root, legacy_document, case_map[LEGACY_CASE_ID]),
                *generated_members,
            ),
            key=lambda item: item.registry_member_id,
        )
    )

    inventory_document = source_inventory_document(root)
    inventory_document.pop("canonical_payload_hash", None)
    metric_map = {
        item.case.case_id: (
            item.predicted_speed_ratio,
            item.tangential_velocity_error_ratio,
            item.canonical_payload_hash,
        )
        for item in discoveries
    }
    legacy_derived = _derived_values(
        CartesianState2D(
            float(legacy_document["state"]["position_x"]),
            float(legacy_document["state"]["position_y"]),
            float(legacy_document["state"]["velocity_x"]),
            float(legacy_document["state"]["velocity_y"]),
        ),
        legacy_document["simulator_configuration"],
    )
    legacy_predicted_speed_ratio = float(legacy_document["predicted_speed_ratio"])
    legacy_tangential_velocity_error_ratio = float(legacy_derived["tangential_velocity_error_ratio"])
    metric_map[LEGACY_CASE_ID] = (
        legacy_predicted_speed_ratio,
        legacy_tangential_velocity_error_ratio,
        legacy_hash,
    )
    for item in inventory_document["cases"]:
        ratio, tangential, state_hash = metric_map[item["case_id"]]
        item["predicted_speed_ratio_if_available"] = ratio
        item["overspeed_class_if_available"] = "above" if ratio > OVERSPEED_THRESHOLD else "below_or_equal"
        item["tangential_velocity_error_ratio_at_branch"] = tangential
        item["generated_or_legacy_state_hash"] = state_hash
    inventory_document = _attach_document_hash(inventory_document)

    selection_document = _selection_report(
        discoveries,
        selection,
        config_hash,
        legacy_predicted_speed_ratio=legacy_predicted_speed_ratio,
        legacy_tangential_velocity_error_ratio=legacy_tangential_velocity_error_ratio,
        legacy_state_hash=legacy_hash,
    )
    determinism_document = _attach_document_hash(
        {
            "schema_version": "recovery_branch_state_determinism_report_v0",
            "member_count": len(determinism_records),
            "determinism_failure_count": 0,
            "canonical_reproduction_failure_count": 0,
            "members": sorted(determinism_records, key=lambda item: item["case_id"]),
        }
    )

    execution_records = [
        _execution_record(
            "canonical_reproduction_01",
            LEGACY_CASE_ID,
            "canonical_reproduction",
            legacy_reproduction.initial_state_hash,
            legacy_reproduction.prefix_action_trace_hash,
            legacy_reproduction.prefix_state_trace_hash,
            legacy_hash,
            True,
        )
    ]
    selected_set = set(selection.generated_case_ids)
    for item in discoveries:
        execution_records.append(
            _execution_record(
                item.execution_id,
                item.case.case_id,
                item.execution_role,
                item.initial_state_hash,
                item.prefix_action_trace_hash,
                item.prefix_state_trace_hash,
                item.canonical_payload_hash,
                item.case.case_id in selected_set,
            )
        )
    for item in reproductions:
        execution_records.append(
            _execution_record(
                item.execution_id,
                item.case.case_id,
                item.execution_role,
                item.initial_state_hash,
                item.prefix_action_trace_hash,
                item.prefix_state_trace_hash,
                item.canonical_payload_hash,
                False,
            )
        )
    prefix_document = _attach_document_hash(
        {
            "schema_version": "recovery_branch_state_prefix_execution_report_v0",
            "canonical_reproduction_execution_count": 1,
            "candidate_discovery_execution_count": len(discoveries),
            "selected_reproduction_execution_count": len(reproductions),
            "total_nominal_prefix_execution_count": len(execution_records),
            "automatic_retry_count": 0,
            "recovery_branch_execution_count": 0,
            "executions": execution_records,
        }
    )
    member_documents = [member.as_document() for member in members]
    index_document = _attach_document_hash(
        {
            "schema_version": "recovery_branch_state_index_v0",
            "registry_id": REGISTRY_ID,
            "member_count": len(members),
            "members": member_documents,
            "registry_aggregate_hash": registry_aggregate_hash(members),
        }
    )

    protected_hashes = dict(protected_evidence_hashes(root))
    all_artifacts = [*RESULT_ARTIFACT_FILENAMES, *sorted(payloads)]
    manifest: dict[str, object] = {
        "registry_id": REGISTRY_ID,
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "artifact_classification": "deterministically_generated_branch_state_registry",
        "completed_date": COMPLETED_DATE,
        "implementation_commit": implementation_commit,
        "result_commit_when_available": None,
        "legacy_canonical_commit": "5f31c3fd74dbf8e8ea5a60d70d7b88f5a9def7c8",
        "source_final_veto_commit": load_frozen_manifest(root)["source_commit"],
        "source_final_veto_case_count": SOURCE_CASE_COUNT,
        "eligible_case_count": len(eligible),
        "ineligible_case_count": SOURCE_CASE_COUNT - len(eligible),
        "registry_member_count": len(members),
        "legacy_member_count": sum(member.legacy_member for member in members),
        "generated_member_count": sum(not member.legacy_member for member in members),
        "selection_contract": config["selection_contract"],
        "prefix_contract": config["prefix_contract"],
        "canonical_reproduction_execution_count": 1,
        "discovery_execution_count": len(discoveries),
        "selected_reproduction_execution_count": len(reproductions),
        "reproduction_execution_count": 1 + len(reproductions),
        "total_execution_count": len(execution_records),
        "member_index": member_documents,
        "artifact_filenames": all_artifacts,
        "registry_aggregate_hash": registry_aggregate_hash(members),
        "configuration_hash": config_hash,
        "protected_artifact_hashes": protected_hashes,
        "scientific_claim_restrictions": [
            "no recovery performance claim",
            "no controller improvement claim",
            "no phase-policy validity claim",
            "no formal safety or deployment claim",
        ],
        "new_recovery_controller": False,
        "new_recovery_action": False,
        "new_staged_phase": False,
        "new_staged_execution": False,
        "branch_state_generation": "deterministic_nominal_prefix_execution",
        "Stage_1B_prerequisite_status": "satisfied",
    }
    manifest["canonical_manifest_hash"] = canonical_sha256(
        manifest_scientific_payload(manifest)
    )

    selected_metrics = {
        case_id: discovery_by_case[case_id] for case_id in selection.generated_case_ids
    }
    summary = (
        "# Multi-Case Recovery Branch-State Registry v0\n\n"
        "Status: Multi-case recovery branch-state registry frozen and validated; Stage 1B multi-case source-state prerequisite satisfied.\n\n"
        "Completed: 2026-08-02\n\n"
        "This registry contains complete, provenance-bound branch-point states produced by deterministic execution of existing frozen nominal-prefix behavior. The states were not reconstructed from incomplete logs, manually authored, or created by perturbing the legacy canonical state. Registry membership enables multi-case bounded recovery and shadow-runtime experiments, but does not demonstrate recovery performance, controller improvement, phase-policy validity, formal safety, or deployment readiness.\n\n"
        "## Status\n\nFour deterministic, executable members are registered.\n\n"
        "## Original Stage 1B blocker\n\nThe previous single-state executor could not satisfy a four-case trace contract.\n\n"
        "## Purpose\n\nFreeze complete multi-case branch-point inputs without executing a recovery branch.\n\n"
        "## Legacy canonical member\n\nThe published canonical artifact remains external and byte-identical.\n\n"
        "## Source-case inventory\n\nAll 13 Final Veto cases were provenance-complete and eligible.\n\n"
        "## Prefix extraction contract\n\nEach discovery stops after 27 nominal transitions, before the step-28 action.\n\n"
        "## Candidate discovery\n\nTwelve noncanonical cases were executed once for deterministic discovery.\n\n"
        "## Frozen selection rules\n\nClosest below, closest above, and strongest remaining tangential challenge were selected without post-result rule changes.\n\n"
        "## Selected members\n\n"
        f"- Member A: `{selection.member_a_case_id}`\n"
        f"- Member B: `{selection.member_b_case_id}` at predicted ratio `{selected_metrics[selection.member_b_case_id].predicted_speed_ratio}`\n"
        f"- Member C: `{selection.member_c_case_id}` at predicted ratio `{selected_metrics[selection.member_c_case_id].predicted_speed_ratio}`\n"
        f"- Member D: `{selection.member_d_case_id}` at tangential error ratio `{selected_metrics[selection.member_d_case_id].tangential_velocity_error_ratio}`\n\n"
        "## Cartesian state completeness\n\nAll four members provide finite x, y, vx, and vy state values.\n\n"
        "## Provenance\n\nEvery generated member binds source case, configuration, simulator, constants, transition, controller, action-trace, and state-trace hashes.\n\n"
        "## Determinism validation\n\nEach generated member exactly matched an independent fresh reproduction.\n\n"
        "## Canonical reproduction\n\nThe legacy canonical payload hash and complete document reproduced exactly.\n\n"
        "## Registry loader\n\nLoading is member-ID based, immutable, hash-validating, and path constrained.\n\n"
        "## Executor compatibility\n\nThe default executor path remains legacy-canonical; registry execution uses a separate validated member-ID entry point.\n\n"
        "## Protected evidence\n\nAll protected aggregate hashes were recorded read-only and remained unchanged.\n\n"
        "## Scientific limitations\n\nThis is input generation, not recovery or policy evidence.\n\n"
        "## Stage 1B readiness\n\nThe four-case source-state prerequisite is satisfied.\n\n"
        "## Next aggressive milestone\n\nResume Stage 1B: Staged Recovery Shadow Guard Runtime and Calibration Trace Set v0.\n"
    )

    payloads.update(
        {
            "source_case_inventory.json": canonical_json_file_bytes(inventory_document),
            "selection_report.json": canonical_json_file_bytes(selection_document),
            "determinism_report.json": canonical_json_file_bytes(determinism_document),
            "prefix_execution_report.json": canonical_json_file_bytes(prefix_document),
            "branch_state_index.json": canonical_json_file_bytes(index_document),
            "registry_manifest.json": json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n",
            "summary.md": summary.encode("utf-8"),
        }
    )
    metadata = {
        "selection": selection,
        "members": members,
        "execution_count": len(execution_records),
        "manifest_hash": manifest["canonical_manifest_hash"],
        "registry_aggregate_hash": manifest["registry_aggregate_hash"],
    }
    validate_registry_payloads(root, payloads)
    return payloads, metadata


def _parse_payload_object(payloads: Mapping[str, bytes], name: str) -> dict[str, object]:
    try:
        value = json.loads(payloads[name].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BranchStateExtractionError(f"invalid registry payload {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise BranchStateExtractionError(f"registry payload {name} must be an object")
    return value


def _document_hash_valid(document: Mapping[str, object]) -> bool:
    supplied = document.get("canonical_payload_hash")
    payload = copy.deepcopy(dict(document))
    payload.pop("canonical_payload_hash", None)
    return supplied == canonical_sha256(payload)


def validate_registry_payloads(
    repository_root: Path,
    payloads: Mapping[str, bytes],
) -> tuple[str, str, int]:
    root = repository_root.resolve()
    manifest = _parse_payload_object(payloads, "registry_manifest.json")
    members_raw = manifest.get("member_index")
    if not isinstance(members_raw, list):
        raise BranchStateExtractionError("registry manifest member index is missing")
    members = tuple(RegistryMember.from_mapping(item) for item in members_raw if isinstance(item, dict))
    expected = set(RESULT_ARTIFACT_FILENAMES)
    for member in members:
        if not member.legacy_member:
            expected.add(PurePosixPath(member.artifact_path).relative_to(OUTPUT_PATH).as_posix())
    if set(payloads) != expected:
        raise BranchStateExtractionError("registry payload artifact set is not exact")
    if manifest.get("canonical_manifest_hash") != canonical_sha256(
        manifest_scientific_payload(manifest)
    ):
        raise BranchStateExtractionError("registry manifest hash mismatch")
    if len(members) < 4 or sum(member.legacy_member for member in members) != 1:
        raise BranchStateExtractionError("registry member cardinality is invalid")
    if len({member.case_id for member in members}) != len(members):
        raise BranchStateExtractionError("registry cases are not distinct")
    for member in members:
        if member.legacy_member:
            if file_sha256(root / LEGACY_ARTIFACT_PATH) != member.raw_artifact_hash:
                raise BranchStateExtractionError("legacy raw artifact hash mismatch")
        else:
            key = PurePosixPath(member.artifact_path).relative_to(OUTPUT_PATH).as_posix()
            raw = payloads[key]
            if hashlib.sha256(raw).hexdigest() != member.raw_artifact_hash:
                raise BranchStateExtractionError("generated raw artifact hash mismatch")
            document = _parse_payload_object(payloads, key)
            validate_generated_branch_state_document(document)
            if document.get("canonical_payload_hash") != member.canonical_branch_state_hash:
                raise BranchStateExtractionError("generated member canonical hash mismatch")
    for name in (
        "source_case_inventory.json",
        "selection_report.json",
        "determinism_report.json",
        "prefix_execution_report.json",
        "branch_state_index.json",
    ):
        if not _document_hash_valid(_parse_payload_object(payloads, name)):
            raise BranchStateExtractionError(f"artifact canonical hash mismatch: {name}")
    inventory = _parse_payload_object(payloads, "source_case_inventory.json")
    if inventory.get("source_case_count") != SOURCE_CASE_COUNT:
        raise BranchStateExtractionError("source inventory count mismatch")
    determinism = _parse_payload_object(payloads, "determinism_report.json")
    if determinism.get("determinism_failure_count") != 0:
        raise BranchStateExtractionError("determinism report contains failures")
    prefix = _parse_payload_object(payloads, "prefix_execution_report.json")
    if prefix.get("recovery_branch_execution_count") != 0:
        raise BranchStateExtractionError("a recovery branch was executed during generation")
    if prefix.get("automatic_retry_count") != 0:
        raise BranchStateExtractionError("generation retried automatically")
    summary = payloads["summary.md"].decode("utf-8")
    if "does not demonstrate recovery performance" not in summary:
        raise BranchStateExtractionError("summary scientific restrictions are incomplete")
    aggregate = registry_aggregate_hash(members)
    if manifest.get("registry_aggregate_hash") != aggregate:
        raise BranchStateExtractionError("registry aggregate hash mismatch")
    return str(manifest["canonical_manifest_hash"]), aggregate, len(members)


def _target_allowed(repository_root: Path, target: Path) -> bool:
    root = repository_root.resolve()
    resolved = target.absolute()
    return resolved == (root / OUTPUT_PATH).absolute() and not resolved.is_symlink()


def publish_registry_payloads(
    repository_root: Path,
    payloads: Mapping[str, bytes],
    *,
    writer: Callable[[Path, bytes], None] | None = None,
    validator: Callable[[Path, Mapping[str, bytes]], tuple[str, str, int]] | None = None,
) -> RegistryPublicationResult:
    root = repository_root.resolve()
    target = (root / OUTPUT_PATH).absolute()
    if not _target_allowed(root, target):
        raise BranchStateExtractionError("registry publication target is not allowed")
    if target.exists() or target.is_symlink():
        raise BranchStateExtractionError("registry publication target already exists")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise BranchStateExtractionError("registry publication parent is invalid")
    validate = validator or validate_registry_payloads
    manifest_hash, aggregate, member_count = validate(root, payloads)
    staging = target.parent / f".{target.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir()
    write = writer or (lambda path, content: path.write_bytes(content))
    published = False
    try:
        for relative, content in sorted(payloads.items()):
            path = staging / PurePosixPath(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            write(path, content)
        staged_payloads = {
            path.relative_to(staging).as_posix(): path.read_bytes()
            for path in staging.rglob("*")
            if path.is_file()
        }
        staged_manifest, staged_aggregate, staged_count = validate(root, staged_payloads)
        if (staged_manifest, staged_aggregate, staged_count) != (
            manifest_hash,
            aggregate,
            member_count,
        ):
            raise BranchStateExtractionError("staged registry validation differs from memory")
        os.replace(staging, target)
        published = True
        final_payloads = {
            path.relative_to(target).as_posix(): path.read_bytes()
            for path in target.rglob("*")
            if path.is_file()
        }
        if final_payloads != staged_payloads:
            raise BranchStateExtractionError("published registry differs from staging")
        final_manifest, final_aggregate, final_count = validate(root, final_payloads)
        return RegistryPublicationResult(
            target_directory=target.as_posix(),
            artifact_paths=tuple((target / PurePosixPath(name)).as_posix() for name in sorted(final_payloads)),
            artifact_hashes=tuple(
                (name, hashlib.sha256(content).hexdigest())
                for name, content in sorted(final_payloads.items())
            ),
            registry_manifest_hash=final_manifest,
            registry_aggregate_hash=final_aggregate,
            member_count=final_count,
            total_execution_count=int(
                _parse_payload_object(final_payloads, "prefix_execution_report.json")[
                    "total_nominal_prefix_execution_count"
                ]
            ),
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def validate_published_registry(repository_root: Path) -> RegistryPublicationResult:
    root = repository_root.resolve()
    target = root / OUTPUT_PATH
    if not target.is_dir() or target.is_symlink():
        raise BranchStateExtractionError("published registry directory is missing or symbolic")
    payloads = {
        path.relative_to(target).as_posix(): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file()
    }
    manifest_hash, aggregate, member_count = validate_registry_payloads(root, payloads)
    registry = load_branch_state_registry(root)
    if len(registry.members) != member_count:
        raise BranchStateExtractionError("published loader member count mismatch")
    for member in registry.members:
        loaded = load_registered_branch_state(root, member.registry_member_id)
        if loaded.case_id != member.case_id:
            raise BranchStateExtractionError("executor-load member identity mismatch")
    prefix = _parse_payload_object(payloads, "prefix_execution_report.json")
    return RegistryPublicationResult(
        target_directory=target.as_posix(),
        artifact_paths=tuple((target / PurePosixPath(name)).as_posix() for name in sorted(payloads)),
        artifact_hashes=tuple(
            (name, hashlib.sha256(content).hexdigest())
            for name, content in sorted(payloads.items())
        ),
        registry_manifest_hash=manifest_hash,
        registry_aggregate_hash=aggregate,
        member_count=member_count,
        total_execution_count=int(prefix["total_nominal_prefix_execution_count"]),
    )


__all__ = [
    "FINAL_VETO_MANIFEST_PATH",
    "KNOWN_PROTECTED_HASHES",
    "PROTECTED_HASH_GROUPS",
    "RESULT_ARTIFACT_FILENAMES",
    "SOURCE_CASE_COUNT",
    "BranchStateExtractionError",
    "LegacyReproductionResult",
    "PrefixExecutionResult",
    "RegistryPublicationResult",
    "RegistrySelection",
    "RegistryStaticValidationReport",
    "SourceCaseDefinition",
    "build_frozen_registry_payloads",
    "build_source_case_inventory",
    "compare_prefix_results",
    "execute_nominal_prefix",
    "load_registry_config",
    "protected_evidence_hashes",
    "publish_registry_payloads",
    "repository_state",
    "reproduce_legacy_canonical",
    "select_registry_cases",
    "source_inventory_document",
    "validate_published_registry",
    "validate_registry_payloads",
    "validate_static_contract",
]
