from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

from runtime_assurance.final_veto_monitor import (
    OVERSPEED_COMPARATOR,
    OVERSPEED_THRESHOLD,
    OneStepPrediction,
    evaluate_overspeed_veto,
)
from runtime_assurance.final_veto_runner_types import (
    ActionInterceptionResult,
    PreTransitionActionContext,
)
from runtime_assurance.recovery_branch_state_extractor import (
    build_source_case_inventory,
)
from runtime_assurance.recovery_branch_state_registry import (
    LEGACY_CASE_ID,
    LEGACY_MEMBER_ID,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
)
from runtime_assurance.stage2a_prediction_boundary_discovery import (
    NormalActionEvaluation,
    evaluate_normal_action,
    execute_allowed_normal_transition,
    is_prediction_boundary_candidate,
    load_published_payloads as load_d1_published_payloads,
    prepare_legacy_discovery_source,
    protected_evidence_hashes as d1_protected_evidence_hashes,
    state_document,
    state_from_branch_document,
)
from runtime_assurance.staged_recovery_logger_adapter import runtime_state_hash
from simulator.phase34_35_transition import CartesianState2D, Phase3435TransitionResult


D2_ID = "stage2a_prediction_boundary_discovery_d2_v0"
D2_SCHEMA_VERSION = "stage2a_prediction_boundary_discovery_d2_v0"
COMPLETED_DATE = "2026-08-27"
PLAN_PATH = Path("configs/stage2a_prediction_boundary_discovery_d2_v0.json")
OUTPUT_PATH = Path("analysis/stage2a_prediction_boundary_discovery_d2_v0")
D1_OUTPUT_PATH = Path("analysis/stage2a_prediction_boundary_discovery_v0")
D1_RESULT_COMMIT = "1de43b588c18ba80157bc89b82901983fdf2644e"
D1_MANIFEST_HASH = "a288271e615b465e9dbda5c1234df7b6963d8badd83148bb68c9b32367998860"
D1_PLAN_HASH = "fd6634648f6d3f690c15466bd95cbbe6dde35e162f247ea39fff49727e0080eb"
D1_ANCHOR_PREDICTED_SPEED_RATIO = 1.8906024003603095
SOURCE_FAMILY = "phase35_radial_energy_push_overspeed_stress_v0"
UPSTREAM_VARIANT = "radial_energy_push"
POST_CROSS_MODE = "radius_priority"
CONTROLLER_ID = "phase35_crossing_basin_expansion"
R0_OVER_TARGET = 0.98
THRUST_SCALE = 8000.0
SEED = 0
ANGLE_GRID = (150.0, 155.0, 160.0, 162.5, 165.0, 167.5, 170.0, 172.5, 175.0)
RECOVERY_BRANCH_ID = "zero_action_reference_v0"
MAXIMUM_RECOVERY_TRANSITIONS = 8
RESULT_ARTIFACTS = (
    "candidate_boundaries.json",
    "coverage_summary.json",
    "discovery_manifest.json",
    "discovery_plan.json",
    "near_boundary_diagnostics.json",
    "protected_evidence_report.json",
    "source_boundary_index.json",
    "source_case_index.json",
    "summary.md",
)
D1_IMPLEMENTATION_PATHS = (
    "runtime_assurance/stage2a_prediction_boundary_discovery.py",
    "scripts/run_stage2a_prediction_boundary_discovery_v0.py",
    "scripts/check_stage2a_prediction_boundary_discovery_v0.py",
    "Tests/test_stage2a_prediction_boundary_discovery.py",
    "configs/stage2a_prediction_boundary_discovery_v0.json",
    "docs/architecture/stage2a_prediction_boundary_discovery_v0.md",
)


class D2DiscoveryError(RuntimeError):
    pass


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise D2DiscoveryError(f"{name} must be an object")
    return dict(value)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise D2DiscoveryError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise D2DiscoveryError(f"{name} must be finite")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise D2DiscoveryError(f"{name} must be an integer >= {minimum}")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _angle_token(angle: float) -> str:
    if angle.is_integer():
        return str(int(angle))
    return str(angle).replace(".", "p")


def source_case_id(angle: float) -> str:
    if angle == 150.0:
        return LEGACY_CASE_ID
    return (
        f"{SOURCE_FAMILY}__r0_0p98__angle_{_angle_token(angle)}"
        "__thrust_8000"
    )


@dataclass(frozen=True, slots=True)
class D2SourceCase:
    case_id: str
    angle: float
    r0_over_target: float
    thrust_scale: float
    seed: int
    upstream_variant: str
    controller_id: str
    post_cross_mode: str
    source_configuration_hash: str
    anchor: bool

    def configuration(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "controller_id": self.controller_id,
            "initial_velocity_angle_deg": self.angle,
            "post_cross_mode": self.post_cross_mode,
            "r0_over_target": self.r0_over_target,
            "seed": self.seed,
            "subset_id": SOURCE_FAMILY,
            "thrust_scale": self.thrust_scale,
            "upstream_variant": self.upstream_variant,
        }


def build_source_cases() -> tuple[D2SourceCase, ...]:
    cases: list[D2SourceCase] = []
    for angle in ANGLE_GRID:
        case_id = source_case_id(angle)
        configuration = {
            "case_id": case_id,
            "controller_id": CONTROLLER_ID,
            "initial_velocity_angle_deg": angle,
            "post_cross_mode": POST_CROSS_MODE,
            "r0_over_target": R0_OVER_TARGET,
            "seed": SEED,
            "subset_id": SOURCE_FAMILY,
            "thrust_scale": THRUST_SCALE,
            "upstream_variant": UPSTREAM_VARIANT,
        }
        cases.append(
            D2SourceCase(
                case_id=case_id,
                angle=angle,
                r0_over_target=R0_OVER_TARGET,
                thrust_scale=THRUST_SCALE,
                seed=SEED,
                upstream_variant=UPSTREAM_VARIANT,
                controller_id=CONTROLLER_ID,
                post_cross_mode=POST_CROSS_MODE,
                source_configuration_hash=canonical_sha256(configuration),
                anchor=angle == 150.0,
            )
        )
    return tuple(cases)


@dataclass(frozen=True, slots=True)
class D2SourceBoundary:
    case: D2SourceCase
    status: str
    unavailability_reason: str | None
    document_json: str | None
    source_execution_hash: str
    source_prefix_transition_count: int
    branch_step: int | None
    boundary_state_hash: str | None
    realized_speed_ratio: float | None
    nominal_action: tuple[float, float] | None
    predicted_state_hash: str | None
    predicted_speed_ratio: float | None
    final_veto_decision: str | None
    prefix_action_trace_hash: str
    prefix_state_trace_hash: str
    source_final_veto_rejection_count: int
    source_vetoed_proposal_transition_count: int
    fallback_execution_count: int
    anchor_equivalence_checks: tuple[tuple[str, bool], ...]

    def document(self) -> dict[str, object]:
        if self.document_json is None:
            raise D2DiscoveryError("unavailable source boundary has no state document")
        return _mapping(json.loads(self.document_json), "source boundary document")


class _SourceBoundaryCaptured(RuntimeError):
    def __init__(self, boundary: D2SourceBoundary):
        self.boundary = boundary
        super().__init__("first natural nominal Final Veto boundary captured")


def _source_execution_identity(
    case: D2SourceCase,
    *,
    status: str,
    prefix_count: int,
    branch_step: int | None,
    boundary_state_hash: str | None,
    predicted_speed_ratio: float | None,
    action_trace_hash: str,
    state_trace_hash: str,
) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "initial_velocity_angle_deg": case.angle,
        "source_configuration_hash": case.source_configuration_hash,
        "source_boundary_status": status,
        "source_prefix_transition_count": prefix_count,
        "branch_step": branch_step,
        "boundary_state_hash": boundary_state_hash,
        "nominal_predicted_speed_ratio": predicted_speed_ratio,
        "prefix_action_trace_hash": action_trace_hash,
        "prefix_state_trace_hash": state_trace_hash,
    }


class _NaturalFirstVetoHook:
    def __init__(
        self,
        case: D2SourceCase,
        simulator_configuration: Mapping[str, object],
        implementation_commit: str,
    ):
        self.case = case
        self.simulator_configuration = copy.deepcopy(dict(simulator_configuration))
        self.implementation_commit = implementation_commit
        self.states: list[dict[str, float]] = []
        self.actions: list[list[float]] = []
        self.valid_evaluation_count = 0

    def __call__(self, context: PreTransitionActionContext) -> ActionInterceptionResult:
        if (
            context.case.case_id != self.case.case_id
            or context.case.initial_velocity_angle_deg != self.case.angle
            or context.case.r0_over_target != self.case.r0_over_target
            or context.case.thrust_scale != self.case.thrust_scale
        ):
            raise D2DiscoveryError("source hook received mismatched frozen case identity")
        self.states.append(state_document(context.current_state))
        nominal_prediction: OneStepPrediction[CartesianState2D] | None = None

        def predictor(
            state: CartesianState2D,
            action: tuple[float, float],
        ) -> OneStepPrediction[CartesianState2D]:
            nonlocal nominal_prediction
            transition = context.predict_transition(state, action)
            prediction = OneStepPrediction(
                next_state=transition.next_state,
                speed_ratio=context.compute_speed_ratio(transition.next_state),
            )
            if (
                nominal_prediction is None
                and state == context.current_state
                and action == context.nominal_action
            ):
                nominal_prediction = prediction
            return prediction

        decision = evaluate_overspeed_veto(
            context.current_state,
            context.nominal_action,
            predictor,
            threshold=OVERSPEED_THRESHOLD,
        )
        self.valid_evaluation_count += 1
        if self.valid_evaluation_count != context.step:
            raise D2DiscoveryError("source monitor sequence differs from physical steps")
        if decision.decision == "allow":
            self.actions.append([context.nominal_action[0], context.nominal_action[1]])
            return ActionInterceptionResult(
                nominal_action=context.nominal_action,
                executed_action=context.nominal_action,
                intervention_applied=False,
                decision_metadata=decision,
            )
        if decision.decision != "veto" or nominal_prediction is None:
            raise D2DiscoveryError("source Final Veto evidence is invalid")
        realized_ratio = context.compute_speed_ratio(context.current_state)
        if realized_ratio > OVERSPEED_THRESHOLD:
            raise D2DiscoveryError("source nominal veto occurred after realized overspeed")
        if nominal_prediction.speed_ratio <= OVERSPEED_THRESHOLD:
            raise D2DiscoveryError("source nominal veto lacks strict predicted overspeed")
        prefix_count = len(self.actions)
        if prefix_count != context.step - 1 or len(self.states) != context.step:
            raise D2DiscoveryError("source first-veto ordering is inconsistent")
        state_hash = runtime_state_hash(context.current_state)
        predicted_hash = runtime_state_hash(nominal_prediction.next_state)
        action_hash = canonical_sha256(self.actions)
        state_trace_hash = canonical_sha256(self.states)
        identity = _source_execution_identity(
            self.case,
            status="available",
            prefix_count=prefix_count,
            branch_step=context.step,
            boundary_state_hash=state_hash,
            predicted_speed_ratio=nominal_prediction.speed_ratio,
            action_trace_hash=action_hash,
            state_trace_hash=state_trace_hash,
        )
        source_execution_hash = canonical_sha256(identity)
        constants = _mapping(
            self.simulator_configuration.get("simulator_constants"),
            "source simulator constants",
        )
        document: dict[str, object] = {
            "schema_version": "stage2a_d2_natural_source_boundary_v0",
            "case_id": self.case.case_id,
            "case_configuration": self.case.configuration(),
            "source_configuration_hash": self.case.source_configuration_hash,
            "state_origin": "natural_phase35_nominal_first_veto_execution",
            "manually_authored_state": False,
            "perturbed_from_existing_state": False,
            "reconstructed_from_log": False,
            "implementation_commit": self.implementation_commit,
            "source_prefix_transition_count": prefix_count,
            "branch_step": context.step,
            "source_boundary_status": "available",
            "position_x": context.current_state.x,
            "position_y": context.current_state.y,
            "velocity_x": context.current_state.vx,
            "velocity_y": context.current_state.vy,
            "state": state_document(context.current_state),
            "boundary_state_hash": state_hash,
            "realized_speed_ratio": realized_ratio,
            "nominal_action": list(context.nominal_action),
            "predicted_next_state": state_document(nominal_prediction.next_state),
            "predicted_state_hash": predicted_hash,
            "predicted_speed_ratio": nominal_prediction.speed_ratio,
            "monitor_decision": {
                "decision": decision.decision,
                "monitor_id": decision.monitor_id,
                "reason": decision.reason,
                "veto_applied": decision.veto_applied,
            },
            "threshold": OVERSPEED_THRESHOLD,
            "comparator": OVERSPEED_COMPARATOR,
            "simulator_configuration": self.simulator_configuration,
            "simulator_configuration_hash": canonical_sha256(
                self.simulator_configuration
            ),
            "constants_hash": canonical_sha256(constants),
            "prefix_action_trace_hash": action_hash,
            "prefix_state_trace_hash": state_trace_hash,
            "source_execution_hash": source_execution_hash,
            "source_nominal_proposal_physically_executed": False,
            "source_fallback_physically_executed": False,
            "active_authority_granted": False,
            "hazard_arrest_interventions": 0,
            "branch_ordering": {
                "before_nominal_action_execution": True,
                "before_final_veto_fallback_execution": True,
                "monitor_evaluation_completed": True,
                "realized_prefix_transition_count": prefix_count,
            },
        }
        document["canonical_source_boundary_hash"] = canonical_sha256(document)
        boundary = D2SourceBoundary(
            case=self.case,
            status="available",
            unavailability_reason=None,
            document_json=canonical_json_bytes(document).decode("utf-8"),
            source_execution_hash=source_execution_hash,
            source_prefix_transition_count=prefix_count,
            branch_step=context.step,
            boundary_state_hash=state_hash,
            realized_speed_ratio=realized_ratio,
            nominal_action=context.nominal_action,
            predicted_state_hash=predicted_hash,
            predicted_speed_ratio=nominal_prediction.speed_ratio,
            final_veto_decision="veto",
            prefix_action_trace_hash=action_hash,
            prefix_state_trace_hash=state_trace_hash,
            source_final_veto_rejection_count=1,
            source_vetoed_proposal_transition_count=0,
            fallback_execution_count=0,
            anchor_equivalence_checks=(),
        )
        raise _SourceBoundaryCaptured(boundary)


def _simulator_configuration() -> dict[str, object]:
    from scripts import explicit_controller_phase35_crossing_basin_expansion as phase35
    from simulator.phase34_35_transition import (
        ACTION_COMPONENT_MAX,
        ACTION_COMPONENT_MIN,
        GRAVITY_DENOMINATOR_EPSILON,
    )

    target_radius = phase35.DEFAULT_TARGET_RADIUS * phase35.TARGET_RADIUS_SCALE
    target_speed = math.sqrt(phase35.MU / target_radius)
    constants: dict[str, object] = {
        "action_component_max": ACTION_COMPONENT_MAX,
        "action_component_min": ACTION_COMPONENT_MIN,
        "dt": phase35.DT,
        "gravity_denominator_epsilon": GRAVITY_DENOMINATOR_EPSILON,
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
        "transition_function": "simulator.phase34_35_transition.step_phase34_35_transition",
    }
    return {"simulator_constants": constants, "thrust_scale": THRUST_SCALE}


def execute_natural_source_case(
    repository_root: Path,
    case: D2SourceCase,
    *,
    implementation_commit: str,
) -> D2SourceBoundary:
    if case.anchor:
        raise D2DiscoveryError("anchor must use the frozen legacy reproduction path")
    from scripts import explicit_controller_phase35_crossing_basin_expansion as phase35
    from scripts.extract_recovery_branch_state import _require_frozen_source_trajectory

    _require_frozen_source_trajectory(repository_root.resolve())
    variant = next(item for item in phase35.VARIANTS if item.name == UPSTREAM_VARIANT)
    mode = phase35.PHASE34_TERMINAL_MODE
    simulator = _simulator_configuration()
    hook = _NaturalFirstVetoHook(case, simulator, implementation_commit)
    try:
        terminal = phase35.rollout_phase35_case(
            variant,
            mode,
            case.r0_over_target,
            case.angle,
            case.thrust_scale,
            phase35.TARGET_RADIUS_SCALE,
            record_trajectory=False,
            case_id=case.case_id,
            pre_transition_action_hook=hook,
        )
    except _SourceBoundaryCaptured as captured:
        return captured.boundary
    action_hash = canonical_sha256(hook.actions)
    state_hash = canonical_sha256(hook.states)
    identity = _source_execution_identity(
        case,
        status="unavailable",
        prefix_count=len(hook.actions),
        branch_step=None,
        boundary_state_hash=None,
        predicted_speed_ratio=None,
        action_trace_hash=action_hash,
        state_trace_hash=state_hash,
    )
    return D2SourceBoundary(
        case=case,
        status="unavailable",
        unavailability_reason=(
            "source_trajectory_terminated_without_first_valid_nominal_veto:"
            f"{terminal.get('termination_reason')}"
        ),
        document_json=None,
        source_execution_hash=canonical_sha256(identity),
        source_prefix_transition_count=len(hook.actions),
        branch_step=None,
        boundary_state_hash=None,
        realized_speed_ratio=None,
        nominal_action=None,
        predicted_state_hash=None,
        predicted_speed_ratio=None,
        final_veto_decision=None,
        prefix_action_trace_hash=action_hash,
        prefix_state_trace_hash=state_hash,
        source_final_veto_rejection_count=0,
        source_vetoed_proposal_transition_count=0,
        fallback_execution_count=0,
        anchor_equivalence_checks=(),
    )


def _load_d1_anchor_evidence(repository_root: Path) -> dict[str, object]:
    payloads = load_d1_published_payloads(repository_root)
    manifest = _mapping(json.loads(payloads["discovery_manifest.json"]), "D1 manifest")
    if (
        manifest.get("canonical_manifest_hash") != D1_MANIFEST_HASH
        or manifest.get("discovery_plan_hash") != D1_PLAN_HASH
        or manifest.get("candidate_boundary_count") != 0
    ):
        raise D2DiscoveryError("frozen D1 identity mismatch")
    index = _mapping(json.loads(payloads["trajectory_index.json"]), "D1 trajectory index")
    entries = cast(list[object], index.get("trajectories"))
    matches = [
        _mapping(item, "D1 anchor trajectory")
        for item in entries
        if isinstance(item, Mapping)
        and item.get("case_id") == LEGACY_CASE_ID
        and item.get("branch_id") == RECOVERY_BRANCH_ID
    ]
    if len(matches) != 1:
        raise D2DiscoveryError("D1 anchor trajectory is not unique")
    records = cast(list[object], matches[0].get("records"))
    if not records:
        raise D2DiscoveryError("D1 anchor event is missing")
    event = _mapping(records[0], "D1 anchor event")
    return {
        "source_boundary_state_hash": matches[0]["source_boundary_state_hash"],
        "event_current_state_hash": event["current_state_hash"],
        "predicted_speed_ratio": event["predicted_speed_ratio"],
        "predicted_state_hash": event["predicted_state_hash"],
        "action": event["normal_action"],
        "event_index": event["event_index"],
    }


def reproduce_anchor_source(
    repository_root: Path,
    case: D2SourceCase,
) -> D2SourceBoundary:
    if not case.anchor or case.case_id != LEGACY_CASE_ID:
        raise D2DiscoveryError("anchor reproduction requires the frozen legacy case")
    legacy_case = next(
        item
        for item in build_source_case_inventory(repository_root)
        if item.case_id == LEGACY_CASE_ID
    )
    source = prepare_legacy_discovery_source(repository_root, legacy_case)
    document = source.document()
    state = state_from_branch_document(document)
    zero_evaluation = evaluate_normal_action(document, state, RECOVERY_BRANCH_ID)
    d1 = _load_d1_anchor_evidence(repository_root)
    checks = {
        "case_id": document.get("case_id") == case.case_id,
        "angle": _mapping(document.get("case_configuration"), "legacy configuration").get(
            "initial_velocity_angle_deg"
        )
        == case.angle,
        "thrust": _mapping(document.get("case_configuration"), "legacy configuration").get(
            "thrust_scale"
        )
        == case.thrust_scale,
        "r0": _mapping(document.get("case_configuration"), "legacy configuration").get(
            "r0_over_target"
        )
        == case.r0_over_target,
        "seed": document.get("seed") == case.seed,
        "boundary_state_hash": (
            runtime_state_hash(state)
            == d1["source_boundary_state_hash"]
            == d1["event_current_state_hash"]
        ),
        "zero_action": list(zero_evaluation.action) == d1["action"] == [0.0, 0.0],
        "zero_action_predicted_speed_ratio": (
            zero_evaluation.predicted_speed_ratio
            == d1["predicted_speed_ratio"]
            == D1_ANCHOR_PREDICTED_SPEED_RATIO
        ),
        "zero_action_predicted_state_hash": (
            zero_evaluation.predicted_state_hash == d1["predicted_state_hash"]
        ),
        "D1_event_index": d1["event_index"] == 0,
        "legacy_registry_member": source.registry_member_id == LEGACY_MEMBER_ID,
        "legacy_boundary_type": source.boundary_type == "legacy_fixed_prefix",
        "legacy_reproduction_checks": all(
            passed for _, passed in source.source_equivalence_checks
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise D2DiscoveryError("D2 anchor reproduction mismatch: " + ", ".join(failed))
    predicted = _mapping(document.get("predicted_next_state"), "legacy predicted state")
    predicted_state = CartesianState2D(
        _finite(predicted.get("position_x"), "predicted.position_x"),
        _finite(predicted.get("position_y"), "predicted.position_y"),
        _finite(predicted.get("velocity_x"), "predicted.velocity_x"),
        _finite(predicted.get("velocity_y"), "predicted.velocity_y"),
    )
    realized_ratio = zero_evaluation.realized_speed_ratio
    identity = _source_execution_identity(
        case,
        status="available",
        prefix_count=source.actual_transition_count,
        branch_step=source.branch_step,
        boundary_state_hash=runtime_state_hash(state),
        predicted_speed_ratio=float(document["predicted_speed_ratio"]),
        action_trace_hash=source.prefix_action_trace_hash,
        state_trace_hash=source.prefix_state_trace_hash,
    )
    return D2SourceBoundary(
        case=case,
        status="available",
        unavailability_reason=None,
        document_json=canonical_json_bytes(document).decode("utf-8"),
        source_execution_hash=canonical_sha256(identity),
        source_prefix_transition_count=source.actual_transition_count,
        branch_step=source.branch_step,
        boundary_state_hash=runtime_state_hash(state),
        realized_speed_ratio=realized_ratio,
        nominal_action=tuple(cast(list[float], document["nominal_action"])),
        predicted_state_hash=runtime_state_hash(predicted_state),
        predicted_speed_ratio=float(document["predicted_speed_ratio"]),
        final_veto_decision=str(
            _mapping(document.get("monitor_decision"), "legacy monitor")["decision"]
        ),
        prefix_action_trace_hash=source.prefix_action_trace_hash,
        prefix_state_trace_hash=source.prefix_state_trace_hash,
        source_final_veto_rejection_count=1,
        source_vetoed_proposal_transition_count=0,
        fallback_execution_count=0,
        anchor_equivalence_checks=tuple(sorted(checks.items())),
    )


@dataclass(frozen=True, slots=True)
class D2RecoveryResult:
    case_id: str
    angle: float
    source_execution_hash: str
    source_boundary_state_hash: str
    records: tuple[dict[str, object], ...]
    physical_transition_count: int
    states_evaluated: int
    candidate_count: int
    final_veto_rejection_count: int
    fallback_execution_count: int
    terminal_reason: str
    trajectory_hash: str


EvaluationFunction = Callable[
    [Mapping[str, object], CartesianState2D, str], NormalActionEvaluation
]


def _recovery_record(
    evaluation: NormalActionEvaluation,
    *,
    event_index: int,
    physical_transition_count_before: int,
    transition_executed: bool,
    realized_next_state: CartesianState2D | None,
    stop_reason: str | None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "event_index": event_index,
        "physical_transition_count_before": physical_transition_count_before,
        "current_state": state_document(evaluation.current_state),
        "current_state_hash": evaluation.current_state_hash,
        "realized_speed_ratio": evaluation.realized_speed_ratio,
        "zero_action": list(evaluation.action),
        "normal_action_hash": evaluation.action_hash,
        "predicted_state": state_document(evaluation.predicted_transition.next_state),
        "predicted_state_hash": evaluation.predicted_state_hash,
        "predicted_speed_ratio": evaluation.predicted_speed_ratio,
        "predicted_headroom": evaluation.predicted_headroom,
        "final_veto_decision": evaluation.final_veto_decision.decision,
        "candidate_boundary": evaluation.candidate_boundary,
        "transition_executed": transition_executed,
        "realized_next_state": (
            None if realized_next_state is None else state_document(realized_next_state)
        ),
        "realized_next_state_hash": (
            None if realized_next_state is None else runtime_state_hash(realized_next_state)
        ),
        "fallback_prediction_count": evaluation.fallback_prediction_count,
        "fallback_execution_count": 0,
        "active_authority_granted": False,
        "hazard_arrest_interventions": 0,
        "stop_reason": stop_reason,
    }
    record["canonical_observation_hash"] = canonical_sha256(record)
    return record


def run_zero_action_recovery(
    boundary: D2SourceBoundary,
    *,
    evaluator: EvaluationFunction = evaluate_normal_action,
    transition_executor: Callable[
        [Mapping[str, object], NormalActionEvaluation], Phase3435TransitionResult
    ] = execute_allowed_normal_transition,
) -> D2RecoveryResult:
    if boundary.status != "available":
        raise D2DiscoveryError("unavailable boundary cannot start recovery")
    document = boundary.document()
    state = state_from_branch_document(document)
    records: list[dict[str, object]] = []
    physical_count = 0
    candidate_count = 0
    veto_count = 0
    terminal_reason = "recovery_transition_horizon_complete"
    while physical_count < MAXIMUM_RECOVERY_TRANSITIONS:
        evaluation = evaluator(document, state, RECOVERY_BRANCH_ID)
        if evaluation.action != (0.0, 0.0):
            raise D2DiscoveryError("D2 recovery action is not zero action")
        event_index = len(records)
        if evaluation.candidate_boundary:
            if evaluation.final_veto_decision.decision != "veto":
                raise D2DiscoveryError("D2 candidate was not rejected by Final Veto")
            candidate_count += 1
            veto_count += 1
            terminal_reason = "candidate_boundary_detected"
            records.append(
                _recovery_record(
                    evaluation,
                    event_index=event_index,
                    physical_transition_count_before=physical_count,
                    transition_executed=False,
                    realized_next_state=None,
                    stop_reason=terminal_reason,
                )
            )
            break
        if evaluation.realized_speed_ratio > OVERSPEED_THRESHOLD:
            terminal_reason = "realized_overspeed_adverse_terminal"
            if evaluation.final_veto_decision.decision == "veto":
                veto_count += 1
            records.append(
                _recovery_record(
                    evaluation,
                    event_index=event_index,
                    physical_transition_count_before=physical_count,
                    transition_executed=False,
                    realized_next_state=None,
                    stop_reason=terminal_reason,
                )
            )
            break
        if evaluation.final_veto_decision.decision == "veto":
            veto_count += 1
            terminal_reason = "zero_action_vetoed_without_candidate"
            records.append(
                _recovery_record(
                    evaluation,
                    event_index=event_index,
                    physical_transition_count_before=physical_count,
                    transition_executed=False,
                    realized_next_state=None,
                    stop_reason=terminal_reason,
                )
            )
            break
        realized = transition_executor(document, evaluation)
        next_state = realized.next_state
        records.append(
            _recovery_record(
                evaluation,
                event_index=event_index,
                physical_transition_count_before=physical_count,
                transition_executed=True,
                realized_next_state=next_state,
                stop_reason=(
                    terminal_reason
                    if physical_count + 1 == MAXIMUM_RECOVERY_TRANSITIONS
                    else None
                ),
            )
        )
        state = next_state
        physical_count += 1
    identity = {
        "case_id": boundary.case.case_id,
        "initial_velocity_angle_deg": boundary.case.angle,
        "source_execution_hash": boundary.source_execution_hash,
        "source_boundary_state_hash": boundary.boundary_state_hash,
        "records": records,
        "physical_transition_count": physical_count,
        "terminal_reason": terminal_reason,
    }
    return D2RecoveryResult(
        case_id=boundary.case.case_id,
        angle=boundary.case.angle,
        source_execution_hash=boundary.source_execution_hash,
        source_boundary_state_hash=str(boundary.boundary_state_hash),
        records=tuple(records),
        physical_transition_count=physical_count,
        states_evaluated=len(records),
        candidate_count=candidate_count,
        final_veto_rejection_count=veto_count,
        fallback_execution_count=0,
        terminal_reason=terminal_reason,
        trajectory_hash=canonical_sha256(identity),
    )


def execute_frozen_discovery(
    repository_root: Path,
    *,
    implementation_commit: str,
    anchor_executor: Callable[[Path, D2SourceCase], D2SourceBoundary] = (
        reproduce_anchor_source
    ),
    source_executor: Callable[..., D2SourceBoundary] = execute_natural_source_case,
) -> tuple[tuple[D2SourceBoundary, ...], tuple[D2RecoveryResult, ...]]:
    boundaries: list[D2SourceBoundary] = []
    for case in build_source_cases():
        boundary = (
            anchor_executor(repository_root, case)
            if case.anchor
            else source_executor(
                repository_root,
                case,
                implementation_commit=implementation_commit,
            )
        )
        boundaries.append(boundary)
    recoveries = tuple(
        run_zero_action_recovery(boundary)
        for boundary in boundaries
        if boundary.status == "available"
    )
    return tuple(boundaries), recoveries


def plan_scientific_payload(document: Mapping[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(dict(document))
    payload.pop("canonical_plan_hash", None)
    return payload


def load_d2_plan(repository_root: Path) -> dict[str, object]:
    path = repository_root.resolve() / PLAN_PATH
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D2DiscoveryError("D2 discovery plan is unavailable") from exc
    result = _mapping(plan, "D2 discovery plan")
    validate_d2_plan(repository_root, result)
    return result


def validate_d2_plan(repository_root: Path, plan: Mapping[str, object]) -> None:
    if plan.get("discovery_id") != D2_ID or plan.get("schema_version") != D2_SCHEMA_VERSION:
        raise D2DiscoveryError("D2 plan identity mismatch")
    if plan.get("canonical_plan_hash") != canonical_sha256(plan_scientific_payload(plan)):
        raise D2DiscoveryError("D2 plan canonical hash mismatch")
    expected_ids = [case.case_id for case in build_source_cases()]
    fixed = {
        "angle_grid": list(ANGLE_GRID),
        "r0_over_target": R0_OVER_TARGET,
        "thrust_scale": THRUST_SCALE,
        "seed": SEED,
        "upstream_variant": UPSTREAM_VARIANT,
        "recovery_branch": RECOVERY_BRANCH_ID,
        "maximum_recovery_physical_transitions": MAXIMUM_RECOVERY_TRANSITIONS,
        "source_case_ids": expected_ids,
        "overspeed_threshold": OVERSPEED_THRESHOLD,
        "overspeed_comparator": OVERSPEED_COMPARATOR,
        "active_authority_granted": False,
        "hazard_arrest_interventions": 0,
        "result_publication_path": OUTPUT_PATH.as_posix(),
    }
    if any(plan.get(key) != value for key, value in fixed.items()):
        raise D2DiscoveryError("D2 frozen dimension changed")
    d1 = _mapping(plan.get("D1_dependency"), "D1 dependency")
    if (
        d1.get("result_commit") != D1_RESULT_COMMIT
        or d1.get("manifest_hash") != D1_MANIFEST_HASH
        or d1.get("scientific_plan_hash") != D1_PLAN_HASH
        or d1.get("anchor_predicted_speed_ratio") != D1_ANCHOR_PREDICTED_SPEED_RATIO
    ):
        raise D2DiscoveryError("D2 D1 dependency changed")
    candidate = _mapping(plan.get("candidate_definition"), "candidate definition")
    if candidate != {
        "current_realized_speed_ratio": "<=1.90",
        "ordering": "before_zero_action_execution",
        "zero_action_predicted_speed_ratio": ">1.90",
    }:
        raise D2DiscoveryError("D2 candidate definition changed")
    if plan.get("protected_evidence_hashes") != protected_evidence_hashes(repository_root):
        raise D2DiscoveryError("D2 plan protected hashes differ from repository")


def _aggregate_paths(repository_root: Path, relative_paths: Sequence[str]) -> str:
    root = repository_root.resolve()
    files: list[Path] = []
    for relative in relative_paths:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise D2DiscoveryError(f"protected path escapes repository: {relative}") from exc
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
        else:
            raise D2DiscoveryError(f"protected path is missing: {relative}")
    rows = [
        f"{path.relative_to(root).as_posix()}|{file_sha256(path)}"
        for path in sorted(files, key=lambda item: item.as_posix())
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def protected_evidence_hashes(repository_root: Path) -> dict[str, str]:
    values = d1_protected_evidence_hashes(repository_root)
    values.update(
        {
            "stage2a_prediction_boundary_discovery_d1_implementation_v0": _aggregate_paths(
                repository_root, D1_IMPLEMENTATION_PATHS
            ),
            "stage2a_prediction_boundary_discovery_d1_result_v0": _aggregate_paths(
                repository_root, (D1_OUTPUT_PATH.as_posix(),)
            ),
        }
    )
    return dict(sorted(values.items()))


def _source_index_entry(boundary: D2SourceBoundary) -> dict[str, object]:
    return {
        "case_id": boundary.case.case_id,
        "initial_velocity_angle_deg": boundary.case.angle,
        "r0_over_target": boundary.case.r0_over_target,
        "thrust_scale": boundary.case.thrust_scale,
        "seed": boundary.case.seed,
        "upstream_variant": boundary.case.upstream_variant,
        "source_configuration_hash": boundary.case.source_configuration_hash,
        "source_execution_hash": boundary.source_execution_hash,
        "source_boundary_status": boundary.status,
        "unavailability_reason": boundary.unavailability_reason,
        "source_prefix_transition_count": boundary.source_prefix_transition_count,
        "branch_step": boundary.branch_step,
        "boundary_state_hash": boundary.boundary_state_hash,
        "realized_speed_ratio": boundary.realized_speed_ratio,
        "nominal_controller_action": (
            None if boundary.nominal_action is None else list(boundary.nominal_action)
        ),
        "nominal_controller_predicted_state_hash": boundary.predicted_state_hash,
        "nominal_controller_predicted_speed_ratio": boundary.predicted_speed_ratio,
        "nominal_Final_Veto_result": boundary.final_veto_decision,
        "prefix_action_trace_hash": boundary.prefix_action_trace_hash,
        "prefix_state_trace_hash": boundary.prefix_state_trace_hash,
        "source_final_veto_rejection_count": boundary.source_final_veto_rejection_count,
        "source_vetoed_proposal_transition_count": (
            boundary.source_vetoed_proposal_transition_count
        ),
        "fallback_execution_count": boundary.fallback_execution_count,
        "anchor": boundary.case.anchor,
        "anchor_equivalence_checks": [
            list(item) for item in boundary.anchor_equivalence_checks
        ],
        "state_origin": (
            None if boundary.status == "unavailable" else boundary.document().get("state_origin")
        ),
        "manually_authored_state": (
            False
            if boundary.case.anchor
            else (
                None
                if boundary.status == "unavailable"
                else boundary.document().get("manually_authored_state")
            )
        ),
        "perturbed_from_existing_state": (
            False
            if boundary.case.anchor
            else (
                None
                if boundary.status == "unavailable"
                else boundary.document().get("perturbed_from_existing_state")
            )
        ),
    }


def _recovery_index_entry(result: D2RecoveryResult) -> dict[str, object]:
    valid_current = [
        item
        for item in result.records
        if float(item["realized_speed_ratio"]) <= OVERSPEED_THRESHOLD
    ]
    closest = (
        None
        if not valid_current
        else max(valid_current, key=lambda item: float(item["predicted_speed_ratio"]))
    )
    return {
        "case_id": result.case_id,
        "initial_velocity_angle_deg": result.angle,
        "source_execution_hash": result.source_execution_hash,
        "source_boundary_state_hash": result.source_boundary_state_hash,
        "recovery_branch": RECOVERY_BRANCH_ID,
        "zero_action_event_count": result.states_evaluated,
        "zero_action_physical_transition_count": result.physical_transition_count,
        "maximum_realized_speed_ratio_while_clear": (
            None
            if not valid_current
            else max(float(item["realized_speed_ratio"]) for item in valid_current)
        ),
        "maximum_zero_action_predicted_speed_ratio": (
            None
            if not result.records
            else max(float(item["predicted_speed_ratio"]) for item in result.records)
        ),
        "closest_predicted_headroom": (
            None if closest is None else float(closest["predicted_headroom"])
        ),
        "closest_event_index": None if closest is None else closest["event_index"],
        "closest_state_hash": None if closest is None else closest["current_state_hash"],
        "closest_predicted_state_hash": (
            None if closest is None else closest["predicted_state_hash"]
        ),
        "candidate_count": result.candidate_count,
        "Final_Veto_rejection_count": result.final_veto_rejection_count,
        "fallback_execution_count": result.fallback_execution_count,
        "terminal_reason": result.terminal_reason,
        "trajectory_hash": result.trajectory_hash,
        "records": list(result.records),
    }


def _candidate_documents(
    recoveries: Sequence[D2RecoveryResult],
    prefix_by_case: Mapping[str, int],
    plan_hash: str,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for result in recoveries:
        for record in result.records:
            if record.get("candidate_boundary") is not True:
                continue
            candidate: dict[str, object] = {
                "candidate_id": (
                    f"d2_candidate__angle_{_angle_token(result.angle)}"
                    f"__event_{record['event_index']}"
                ),
                "initial_velocity_angle_deg": result.angle,
                "case_id": result.case_id,
                "source_execution_hash": result.source_execution_hash,
                "boundary_state": record["current_state"],
                "boundary_state_hash": record["current_state_hash"],
                "zero_action": record["zero_action"],
                "predicted_state": record["predicted_state"],
                "predicted_state_hash": record["predicted_state_hash"],
                "realized_speed_ratio": record["realized_speed_ratio"],
                "predicted_speed_ratio": record["predicted_speed_ratio"],
                "Final_Veto_decision": record["final_veto_decision"],
                "recovery_event_index": record["event_index"],
                "source_prefix_transition_count": prefix_by_case[result.case_id],
                "trajectory_hash": result.trajectory_hash,
                "discovery_plan_hash": plan_hash,
                "candidate_zero_action_physically_executed": False,
                "fallback_execution_count": 0,
                "active_authority_granted": False,
                "hazard_arrest_interventions": 0,
            }
            candidate["canonical_candidate_hash"] = canonical_sha256(candidate)
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            float(item["initial_velocity_angle_deg"]),
            int(item["recovery_event_index"]),
            str(item["candidate_id"]),
        )
    )
    return candidates


def build_d2_payloads(
    repository_root: Path,
    boundaries: Sequence[D2SourceBoundary],
    recoveries: Sequence[D2RecoveryResult],
    *,
    implementation_commit: str,
    protected_before: Mapping[str, str],
    protected_after: Mapping[str, str],
) -> dict[str, bytes]:
    if dict(protected_before) != dict(protected_after):
        raise D2DiscoveryError("protected evidence changed during D2")
    plan = load_d2_plan(repository_root)
    plan_hash = str(plan["canonical_plan_hash"])
    expected_cases = build_source_cases()
    if [item.case.case_id for item in boundaries] != [item.case_id for item in expected_cases]:
        raise D2DiscoveryError("D2 source ordering changed")
    available = [item for item in boundaries if item.status == "available"]
    if [item.case_id for item in recoveries] != [item.case.case_id for item in available]:
        raise D2DiscoveryError("D2 recovery ordering changed")
    source_entries = [_source_index_entry(item) for item in boundaries]
    recovery_entries = [_recovery_index_entry(item) for item in recoveries]
    prefix_transitions = sum(item.source_prefix_transition_count for item in boundaries)
    recovery_transitions = sum(item.physical_transition_count for item in recoveries)
    states_evaluated = sum(item.states_evaluated for item in recoveries)
    candidate_count = sum(item.candidate_count for item in recoveries)
    source_vetoes = sum(item.source_final_veto_rejection_count for item in boundaries)
    recovery_vetoes = sum(item.final_veto_rejection_count for item in recoveries)
    fallback_count = sum(item.fallback_execution_count for item in boundaries) + sum(
        item.fallback_execution_count for item in recoveries
    )
    prefix_by_case = {
        item.case.case_id: item.source_prefix_transition_count for item in boundaries
    }
    candidates = _candidate_documents(recoveries, prefix_by_case, plan_hash)
    if candidate_count != len(candidates):
        raise D2DiscoveryError("D2 candidate count mismatch")
    angle_trend = [
        {
            "initial_velocity_angle_deg": item["initial_velocity_angle_deg"],
            "source_boundary_status": next(
                source["source_boundary_status"]
                for source in source_entries
                if source["case_id"] == item["case_id"]
            ),
            "maximum_zero_action_predicted_speed_ratio": item[
                "maximum_zero_action_predicted_speed_ratio"
            ],
            "closest_predicted_headroom": item["closest_predicted_headroom"],
            "closest_event_index": item["closest_event_index"],
            "source_state_realized_speed_ratio": next(
                source["realized_speed_ratio"]
                for source in source_entries
                if source["case_id"] == item["case_id"]
            ),
        }
        for item in recovery_entries
    ]
    valid_maxima = [
        item for item in angle_trend if item["maximum_zero_action_predicted_speed_ratio"] is not None
    ]
    closest = (
        None
        if not valid_maxima
        else max(valid_maxima, key=lambda item: float(item["maximum_zero_action_predicted_speed_ratio"]))
    )
    source_case_index = {
        "schema_version": D2_SCHEMA_VERSION,
        "source_case_ordering": "initial_velocity_angle_deg_ascending",
        "source_case_count": len(source_entries),
        "source_cases": source_entries,
    }
    source_boundary_index = {
        "schema_version": D2_SCHEMA_VERSION,
        "valid_source_boundary_count": len(available),
        "unavailable_source_boundary_count": len(boundaries) - len(available),
        "recovery_trajectory_count": len(recovery_entries),
        "recovery_trajectories": recovery_entries,
    }
    candidate_document = {
        "schema_version": D2_SCHEMA_VERSION,
        "candidate_ordering": "angle_then_recovery_event_index_then_candidate_id",
        "candidate_boundary_count": len(candidates),
        "candidate_boundaries": candidates,
    }
    diagnostics = {
        "schema_version": D2_SCHEMA_VERSION,
        "angle_trend": angle_trend,
        "maximum_zero_action_predicted_speed_ratio": (
            None if closest is None else closest["maximum_zero_action_predicted_speed_ratio"]
        ),
        "closest_headroom": None if closest is None else closest["closest_predicted_headroom"],
        "closest_angle": None if closest is None else closest["initial_velocity_angle_deg"],
        "closest_event_index": None if closest is None else closest["closest_event_index"],
        "source_state_realized_speed_ratio_at_closest": (
            None if closest is None else closest["source_state_realized_speed_ratio"]
        ),
        "scientific_limitation": (
            "This bounded angle trend is source-geometry discovery, not threshold tuning or active authority evidence."
        ),
    }
    coverage = {
        "schema_version": D2_SCHEMA_VERSION,
        "upstream_source_execution_count": len(boundaries),
        "upstream_prefix_physical_transition_count": prefix_transitions,
        "source_boundary_Final_Veto_rejection_count": source_vetoes,
        "valid_source_boundary_count": len(available),
        "unavailable_source_boundary_count": len(boundaries) - len(available),
        "recovery_trajectory_started_count": len(recoveries),
        "recovery_trajectory_completed_count": len(recoveries),
        "recovery_physical_transition_count": recovery_transitions,
        "states_evaluated": states_evaluated,
        "candidate_boundary_count": candidate_count,
        "candidate_Final_Veto_rejection_count": recovery_vetoes,
        "fallback_execution_count": fallback_count,
        "hazard_arrest_interventions": 0,
        "active_authority_granted": False,
        "total_physical_transition_count": prefix_transitions + recovery_transitions,
        "automatic_retry_count": 0,
    }
    protected_report = {
        "before": dict(protected_before),
        "after": dict(protected_after),
        "all_protected_evidence_unchanged": True,
    }
    status = (
        "one or more strict prediction boundaries found"
        if candidate_count
        else "no strict prediction boundary found"
    )
    summary = (
        "# Stage 2A-D2 Boundary-Targeted Source-State Discovery v0\n\n"
        f"Completed: {COMPLETED_DATE}\n\n"
        "## Status\n\n"
        f"Frozen targeted discovery completed: {status}.\n\n"
        "## Search\n\n"
        "Nine predeclared Phase35 thrust-8000 source cases were generated by existing nominal trajectories at their first valid nominal Final Veto boundary. The sole recovery branch was zero_action_reference_v0 for at most eight physical transitions.\n\n"
        "## Authority\n\n"
        "Active authority granted: false. Hazard-arrest interventions: 0. No source or candidate fallback executed physically.\n\n"
        "## Claim Restrictions\n\n"
        "This result does not demonstrate hazard-arrest effectiveness, recovery improvement, safety, stability, optimality, threshold validation, handoff readiness, hardware validity, or deployment readiness.\n"
    ).encode("utf-8")
    payloads: dict[str, bytes] = {
        "discovery_plan.json": (repository_root / PLAN_PATH).read_bytes(),
        "source_case_index.json": _json_bytes(source_case_index),
        "source_boundary_index.json": _json_bytes(source_boundary_index),
        "candidate_boundaries.json": _json_bytes(candidate_document),
        "near_boundary_diagnostics.json": _json_bytes(diagnostics),
        "coverage_summary.json": _json_bytes(coverage),
        "protected_evidence_report.json": _json_bytes(protected_report),
        "summary.md": summary,
    }
    artifact_hashes = {name: sha256_bytes(data) for name, data in sorted(payloads.items())}
    manifest: dict[str, object] = {
        "discovery_id": D2_ID,
        "schema_version": D2_SCHEMA_VERSION,
        "completed_date": COMPLETED_DATE,
        "implementation_commit": implementation_commit,
        "discovery_plan_hash": plan_hash,
        "D1_result_commit": D1_RESULT_COMMIT,
        "D1_manifest_hash": D1_MANIFEST_HASH,
        "upstream_source_execution_count": len(boundaries),
        "upstream_prefix_physical_transition_count": prefix_transitions,
        "source_boundary_Final_Veto_rejection_count": source_vetoes,
        "valid_source_boundary_count": len(available),
        "unavailable_source_boundary_count": len(boundaries) - len(available),
        "recovery_trajectory_count": len(recoveries),
        "recovery_physical_transition_count": recovery_transitions,
        "states_evaluated": states_evaluated,
        "candidate_boundary_count": candidate_count,
        "candidate_Final_Veto_rejection_count": recovery_vetoes,
        "fallback_execution_count": fallback_count,
        "active_authority_granted": False,
        "hazard_arrest_interventions": 0,
        "automatic_retry_count": 0,
        "total_physical_transition_count": prefix_transitions + recovery_transitions,
        "recovery_branch": RECOVERY_BRANCH_ID,
        "maximum_recovery_physical_transitions": MAXIMUM_RECOVERY_TRANSITIONS,
        "overspeed_threshold": OVERSPEED_THRESHOLD,
        "overspeed_comparator": OVERSPEED_COMPARATOR,
        "artifact_filenames": list(RESULT_ARTIFACTS),
        "artifact_hashes": artifact_hashes,
        "discovery_aggregate_hash": canonical_sha256(artifact_hashes),
        "scientific_claim": "bounded targeted natural source-state prediction-boundary discovery",
    }
    manifest["canonical_manifest_hash"] = canonical_sha256(manifest)
    payloads["discovery_manifest.json"] = _json_bytes(manifest)
    validate_d2_payloads(payloads, source_plan=plan)
    return payloads


def _validate_observation(record: Mapping[str, object]) -> None:
    copied = dict(record)
    supplied = copied.pop("canonical_observation_hash", None)
    if supplied != canonical_sha256(copied):
        raise D2DiscoveryError("D2 observation hash mismatch")
    realized = _finite(record.get("realized_speed_ratio"), "realized ratio")
    predicted = _finite(record.get("predicted_speed_ratio"), "predicted ratio")
    candidate = is_prediction_boundary_candidate(realized, predicted)
    if record.get("candidate_boundary") is not candidate:
        raise D2DiscoveryError("D2 candidate comparison mismatch")
    if record.get("zero_action") != [0.0, 0.0]:
        raise D2DiscoveryError("D2 observation contains a nonzero action")
    if candidate and (
        record.get("final_veto_decision") != "veto"
        or record.get("transition_executed") is not False
        or record.get("realized_next_state") is not None
    ):
        raise D2DiscoveryError("D2 candidate action executed")
    if (
        record.get("fallback_execution_count") != 0
        or record.get("active_authority_granted") is not False
        or record.get("hazard_arrest_interventions") != 0
    ):
        raise D2DiscoveryError("D2 authority or fallback contract changed")
    if record.get("transition_executed") is True and (
        record.get("predicted_state") != record.get("realized_next_state")
        or record.get("predicted_state_hash") != record.get("realized_next_state_hash")
    ):
        raise D2DiscoveryError("D2 prediction differs from realization")


def validate_d2_payloads(
    payloads: Mapping[str, bytes],
    *,
    source_plan: Mapping[str, object] | None = None,
) -> None:
    if set(payloads) != set(RESULT_ARTIFACTS):
        raise D2DiscoveryError("D2 result artifact set is incomplete")
    plan = _mapping(json.loads(payloads["discovery_plan.json"]), "published D2 plan")
    if plan.get("canonical_plan_hash") != canonical_sha256(plan_scientific_payload(plan)):
        raise D2DiscoveryError("published D2 plan hash mismatch")
    if source_plan is not None and plan != dict(source_plan):
        raise D2DiscoveryError("published D2 plan differs from source plan")
    manifest = _mapping(json.loads(payloads["discovery_manifest.json"]), "D2 manifest")
    manifest_copy = dict(manifest)
    supplied_manifest_hash = manifest_copy.pop("canonical_manifest_hash", None)
    if supplied_manifest_hash != canonical_sha256(manifest_copy):
        raise D2DiscoveryError("D2 manifest hash mismatch")
    expected_hashes = {
        name: sha256_bytes(payloads[name])
        for name in sorted(payloads)
        if name != "discovery_manifest.json"
    }
    if manifest.get("artifact_hashes") != expected_hashes:
        raise D2DiscoveryError("D2 artifact hash mismatch")
    if manifest.get("discovery_aggregate_hash") != canonical_sha256(expected_hashes):
        raise D2DiscoveryError("D2 aggregate hash mismatch")
    if (
        manifest.get("upstream_source_execution_count") != len(ANGLE_GRID)
        or manifest.get("recovery_branch") != RECOVERY_BRANCH_ID
        or int(manifest.get("maximum_recovery_physical_transitions", -1))
        != MAXIMUM_RECOVERY_TRANSITIONS
        or manifest.get("overspeed_threshold") != OVERSPEED_THRESHOLD
        or manifest.get("overspeed_comparator") != OVERSPEED_COMPARATOR
        or manifest.get("fallback_execution_count") != 0
        or manifest.get("active_authority_granted") is not False
        or manifest.get("hazard_arrest_interventions") != 0
        or manifest.get("automatic_retry_count") != 0
    ):
        raise D2DiscoveryError("D2 manifest contract changed")
    source_doc = _mapping(json.loads(payloads["source_case_index.json"]), "source index")
    source_entries_value = source_doc.get("source_cases")
    if not isinstance(source_entries_value, list):
        raise D2DiscoveryError("D2 source entries must be a list")
    source_entries = source_entries_value
    if len(source_entries) != len(ANGLE_GRID):
        raise D2DiscoveryError("D2 source count mismatch")
    if [float(_mapping(item, "source")["initial_velocity_angle_deg"]) for item in source_entries] != list(ANGLE_GRID):
        raise D2DiscoveryError("D2 source angle ordering mismatch")
    expected_cases = build_source_cases()
    for expected_case, item_value in zip(expected_cases, source_entries, strict=True):
        item = _mapping(item_value, "D2 source")
        if (
            item.get("case_id") != expected_case.case_id
            or item.get("source_configuration_hash")
            != expected_case.source_configuration_hash
            or item.get("upstream_variant") != UPSTREAM_VARIANT
            or item.get("r0_over_target") != R0_OVER_TARGET
            or item.get("thrust_scale") != THRUST_SCALE
            or item.get("seed") != SEED
            or item.get("source_vetoed_proposal_transition_count") != 0
            or item.get("fallback_execution_count") != 0
        ):
            raise D2DiscoveryError("D2 source contract changed")
        if item.get("source_boundary_status") == "available":
            if (
                item.get("nominal_Final_Veto_result") != "veto"
                or _finite(item.get("realized_speed_ratio"), "source realized ratio")
                > OVERSPEED_THRESHOLD
                or _finite(
                    item.get("nominal_controller_predicted_speed_ratio"),
                    "source predicted ratio",
                )
                <= OVERSPEED_THRESHOLD
            ):
                raise D2DiscoveryError("D2 natural source boundary is invalid")
            if item.get("anchor") is not True and (
                item.get("state_origin") != "natural_phase35_nominal_first_veto_execution"
                or item.get("manually_authored_state") is not False
                or item.get("perturbed_from_existing_state") is not False
            ):
                raise D2DiscoveryError("D2 source state origin is invalid")
        elif item.get("source_boundary_status") != "unavailable":
            raise D2DiscoveryError("D2 source availability status is invalid")
        source_identity = {
            "case_id": item["case_id"],
            "initial_velocity_angle_deg": item["initial_velocity_angle_deg"],
            "source_configuration_hash": item["source_configuration_hash"],
            "source_boundary_status": item["source_boundary_status"],
            "source_prefix_transition_count": item["source_prefix_transition_count"],
            "branch_step": item["branch_step"],
            "boundary_state_hash": item["boundary_state_hash"],
            "nominal_predicted_speed_ratio": item[
                "nominal_controller_predicted_speed_ratio"
            ],
            "prefix_action_trace_hash": item["prefix_action_trace_hash"],
            "prefix_state_trace_hash": item["prefix_state_trace_hash"],
        }
        if item.get("source_execution_hash") != canonical_sha256(source_identity):
            raise D2DiscoveryError("D2 source execution hash mismatch")
    anchor = _mapping(source_entries[0], "D2 anchor")
    checks = anchor.get("anchor_equivalence_checks")
    if not isinstance(checks, list) or not checks or any(
        not isinstance(check, list) or len(check) != 2 or check[1] is not True
        for check in checks
    ):
        raise D2DiscoveryError("D2 anchor equivalence failed")
    check_names = {str(check[0]) for check in checks}
    required_anchor_checks = {
        "D1_event_index",
        "angle",
        "boundary_state_hash",
        "case_id",
        "legacy_boundary_type",
        "legacy_registry_member",
        "legacy_reproduction_checks",
        "r0",
        "seed",
        "thrust",
        "zero_action",
        "zero_action_predicted_speed_ratio",
        "zero_action_predicted_state_hash",
    }
    if check_names != required_anchor_checks:
        raise D2DiscoveryError("D2 anchor equivalence check set changed")
    boundary_doc = _mapping(
        json.loads(payloads["source_boundary_index.json"]), "boundary index"
    )
    recoveries_value = boundary_doc.get("recovery_trajectories")
    if not isinstance(recoveries_value, list):
        raise D2DiscoveryError("D2 recovery entries must be a list")
    recoveries = recoveries_value
    available_sources = {
        str(item["case_id"]): item
        for value in source_entries
        for item in (_mapping(value, "D2 source"),)
        if item.get("source_boundary_status") == "available"
    }
    if (
        boundary_doc.get("valid_source_boundary_count") != len(available_sources)
        or boundary_doc.get("unavailable_source_boundary_count")
        != len(source_entries) - len(available_sources)
        or boundary_doc.get("recovery_trajectory_count") != len(recoveries)
        or len(recoveries) != len(available_sources)
    ):
        raise D2DiscoveryError("D2 source/recovery coverage mismatch")
    expected_recovery_cases = [
        str(_mapping(item, "D2 source")["case_id"])
        for item in source_entries
        if _mapping(item, "D2 source").get("source_boundary_status") == "available"
    ]
    if [str(_mapping(item, "D2 recovery").get("case_id")) for item in recoveries] != expected_recovery_cases:
        raise D2DiscoveryError("D2 recovery ordering mismatch")
    recovery_records_by_key: dict[tuple[str, int], dict[str, object]] = {}
    recovery_hash_by_case: dict[str, str] = {}
    for recovery_value in recoveries:
        recovery = _mapping(recovery_value, "D2 recovery")
        source = available_sources.get(str(recovery.get("case_id")))
        if (
            source is None
            or recovery.get("source_execution_hash") != source["source_execution_hash"]
            or recovery.get("source_boundary_state_hash") != source["boundary_state_hash"]
        ):
            raise D2DiscoveryError("D2 recovery source provenance mismatch")
        if int(recovery["zero_action_physical_transition_count"]) > MAXIMUM_RECOVERY_TRANSITIONS:
            raise D2DiscoveryError("D2 recovery exceeded horizon")
        records = cast(list[object], recovery.get("records"))
        if len(records) != recovery.get("zero_action_event_count"):
            raise D2DiscoveryError("D2 recovery record count mismatch")
        for index, record_value in enumerate(records):
            record = _mapping(record_value, "D2 observation")
            if record.get("event_index") != index:
                raise D2DiscoveryError("D2 event ordering mismatch")
            _validate_observation(record)
            key = (str(recovery["case_id"]), index)
            if key in recovery_records_by_key:
                raise D2DiscoveryError("D2 observation identity is duplicated")
            recovery_records_by_key[key] = record
        identity = {
            "case_id": recovery["case_id"],
            "initial_velocity_angle_deg": recovery["initial_velocity_angle_deg"],
            "source_execution_hash": recovery["source_execution_hash"],
            "source_boundary_state_hash": recovery["source_boundary_state_hash"],
            "records": records,
            "physical_transition_count": recovery["zero_action_physical_transition_count"],
            "terminal_reason": recovery["terminal_reason"],
        }
        if recovery.get("trajectory_hash") != canonical_sha256(identity):
            raise D2DiscoveryError("D2 trajectory hash mismatch")
        recovery_hash_by_case[str(recovery["case_id"])] = str(
            recovery["trajectory_hash"]
        )
    candidate_doc = _mapping(
        json.loads(payloads["candidate_boundaries.json"]), "D2 candidates"
    )
    candidates = cast(list[object], candidate_doc.get("candidate_boundaries"))
    if len(candidates) != candidate_doc.get("candidate_boundary_count"):
        raise D2DiscoveryError("D2 candidate count mismatch")
    ordering: list[tuple[float, int, str]] = []
    for candidate_value in candidates:
        candidate = _mapping(candidate_value, "D2 candidate")
        copied = dict(candidate)
        supplied = copied.pop("canonical_candidate_hash", None)
        if supplied != canonical_sha256(copied):
            raise D2DiscoveryError("D2 candidate hash mismatch")
        if (
            _finite(candidate.get("realized_speed_ratio"), "candidate realized")
            > OVERSPEED_THRESHOLD
            or _finite(candidate.get("predicted_speed_ratio"), "candidate predicted")
            <= OVERSPEED_THRESHOLD
            or candidate.get("Final_Veto_decision") != "veto"
            or candidate.get("candidate_zero_action_physically_executed") is not False
        ):
            raise D2DiscoveryError("D2 candidate contract violation")
        source = available_sources.get(str(candidate.get("case_id")))
        record = recovery_records_by_key.get(
            (
                str(candidate.get("case_id")),
                _integer(candidate.get("recovery_event_index"), "candidate event index"),
            )
        )
        if (
            source is None
            or record is None
            or candidate.get("source_execution_hash") != source["source_execution_hash"]
            or candidate.get("source_prefix_transition_count")
            != source["source_prefix_transition_count"]
            or candidate.get("discovery_plan_hash") != plan["canonical_plan_hash"]
            or candidate.get("fallback_execution_count") != 0
            or candidate.get("active_authority_granted") is not False
            or candidate.get("hazard_arrest_interventions") != 0
            or candidate.get("trajectory_hash")
            != recovery_hash_by_case[str(candidate["case_id"])]
            or candidate.get("boundary_state_hash") != record["current_state_hash"]
            or candidate.get("predicted_state_hash") != record["predicted_state_hash"]
            or candidate.get("realized_speed_ratio") != record["realized_speed_ratio"]
            or candidate.get("predicted_speed_ratio") != record["predicted_speed_ratio"]
        ):
            raise D2DiscoveryError("D2 candidate provenance mismatch")
        ordering.append(
            (
                float(candidate["initial_velocity_angle_deg"]),
                int(candidate["recovery_event_index"]),
                str(candidate["candidate_id"]),
            )
        )
    if ordering != sorted(ordering):
        raise D2DiscoveryError("D2 candidate ordering mismatch")
    coverage = _mapping(json.loads(payloads["coverage_summary.json"]), "D2 coverage")
    if (
        coverage.get("candidate_boundary_count") != len(candidates)
        or coverage.get("fallback_execution_count") != 0
        or coverage.get("hazard_arrest_interventions") != 0
        or coverage.get("total_physical_transition_count")
        != coverage.get("upstream_prefix_physical_transition_count")
        + coverage.get("recovery_physical_transition_count")
        or manifest.get("candidate_boundary_count") != len(candidates)
        or manifest.get("valid_source_boundary_count") != len(available_sources)
        or manifest.get("unavailable_source_boundary_count")
        != len(source_entries) - len(available_sources)
        or manifest.get("recovery_trajectory_count") != len(recoveries)
        or manifest.get("states_evaluated") != coverage.get("states_evaluated")
        or manifest.get("recovery_physical_transition_count")
        != coverage.get("recovery_physical_transition_count")
    ):
        raise D2DiscoveryError("D2 coverage contract mismatch")
    protected = _mapping(
        json.loads(payloads["protected_evidence_report.json"]), "D2 protected report"
    )
    if (
        protected.get("before") != protected.get("after")
        or protected.get("all_protected_evidence_unchanged") is not True
    ):
        raise D2DiscoveryError("D2 protected evidence changed")
    summary = payloads["summary.md"].decode("utf-8")
    for phrase in (
        "Active authority granted: false",
        "Hazard-arrest interventions: 0",
        "does not demonstrate hazard-arrest effectiveness",
    ):
        if phrase not in summary:
            raise D2DiscoveryError("D2 summary claim boundary is incomplete")


def load_published_payloads(repository_root: Path) -> dict[str, bytes]:
    source = repository_root.resolve() / OUTPUT_PATH
    if not source.is_dir():
        raise D2DiscoveryError("published D2 result is missing")
    payloads = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    plan = load_d2_plan(repository_root)
    validate_d2_payloads(payloads, source_plan=plan)
    protected = json.loads(payloads["protected_evidence_report.json"])
    if protected.get("after") != protected_evidence_hashes(repository_root):
        raise D2DiscoveryError("current protected evidence hash differs from D2 result")
    return payloads


def validate_static_sources(
    repository_root: Path,
    *,
    require_output_absent: bool = False,
) -> dict[str, object]:
    root = repository_root.resolve()
    plan = load_d2_plan(root)
    d1 = load_d1_published_payloads(root)
    d1_manifest = json.loads(d1["discovery_manifest.json"])
    if d1_manifest.get("canonical_manifest_hash") != D1_MANIFEST_HASH:
        raise D2DiscoveryError("D1 manifest changed")
    if require_output_absent and (root / OUTPUT_PATH).exists():
        raise D2DiscoveryError("D2 output already exists")
    if (root / OUTPUT_PATH).resolve() == (root / D1_OUTPUT_PATH).resolve():
        raise D2DiscoveryError("D2 output overlaps D1")
    cases = build_source_cases()
    if len(cases) != 9 or tuple(item.angle for item in cases) != ANGLE_GRID:
        raise D2DiscoveryError("D2 source grid changed")
    source = (
        root / "runtime_assurance/stage2a_prediction_boundary_discovery_d2.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    if "runtime_assurance.stage2a_hazard_arrest_authority" in imported_modules:
        raise D2DiscoveryError("D2 imports active authority")
    return {
        "valid": True,
        "source_case_count": len(cases),
        "planned_recovery_trajectory_count": len(cases),
        "maximum_recovery_physical_transitions": MAXIMUM_RECOVERY_TRANSITIONS,
        "plan_hash": plan["canonical_plan_hash"],
        "simulation_executed": False,
        "write_performed": False,
        "active_authority_granted": False,
        "hazard_arrest_interventions": 0,
    }


__all__ = [
    "ANGLE_GRID",
    "COMPLETED_DATE",
    "D1_ANCHOR_PREDICTED_SPEED_RATIO",
    "D1_MANIFEST_HASH",
    "D1_PLAN_HASH",
    "D2DiscoveryError",
    "D2RecoveryResult",
    "D2SourceBoundary",
    "D2SourceCase",
    "D2_ID",
    "D2_SCHEMA_VERSION",
    "MAXIMUM_RECOVERY_TRANSITIONS",
    "OUTPUT_PATH",
    "PLAN_PATH",
    "RECOVERY_BRANCH_ID",
    "RESULT_ARTIFACTS",
    "R0_OVER_TARGET",
    "SEED",
    "THRUST_SCALE",
    "build_d2_payloads",
    "build_source_cases",
    "execute_frozen_discovery",
    "execute_natural_source_case",
    "load_d2_plan",
    "load_published_payloads",
    "plan_scientific_payload",
    "protected_evidence_hashes",
    "reproduce_anchor_source",
    "run_zero_action_recovery",
    "source_case_id",
    "validate_d2_payloads",
    "validate_d2_plan",
    "validate_static_sources",
]
