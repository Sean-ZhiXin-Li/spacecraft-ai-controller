from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_assurance.final_veto_monitor import (  # noqa: E402
    MONITOR_ID,
    OVERSPEED_COMPARATOR,
    OVERSPEED_THRESHOLD,
    MonitorEvaluationError,
    OneStepPrediction,
    evaluate_overspeed_veto,
)
from runtime_assurance.final_veto_runner_types import (  # noqa: E402
    ActionInterceptionResult,
    PreTransitionActionContext,
)
from scripts.check_recovery_action_branching_manifest import (  # noqa: E402
    validate_manifest,
)
from scripts.check_recovery_branch_state import (  # noqa: E402
    ARTIFACT_RELATIVE_PATH,
    CANONICALIZATION_ID,
    HAZARD_COMPARATOR,
    HAZARD_THRESHOLD,
    SCHEMA_VERSION,
    SOURCE_ANGLE_DEG,
    SOURCE_CASE_ID,
    SOURCE_COMMIT,
    SOURCE_R0_OVER_TARGET,
    SOURCE_SEED,
    SOURCE_SUBSET_ID,
    SOURCE_THRUST_SCALE,
    attach_canonical_branch_state_hash,
    canonical_sha256,
    validate_branch_state_data,
    write_canonical_branch_state,
)


MANIFEST_RELATIVE_PATH = Path(
    "analysis/recovery_action_branching_nonformal_v0/manifest.json"
)
EXTRACTION_TIMESTAMP = "2026-07-19T00:00:00+08:00"
EXTRACTION_TIMESTAMP_POLICY = (
    "frozen_milestone_timestamp_for_reproducible_hashing"
)
SOURCE_TRAJECTORY_PATHS = (
    "controller/orbit_lock_controller.py",
    "runtime_assurance/final_veto_monitor.py",
    "runtime_assurance/final_veto_runner_types.py",
    "scripts/explicit_controller_phase21_orbital_transfer_planner.py",
    "scripts/explicit_controller_phase22_two_burn_transfer.py",
    "scripts/explicit_controller_phase34_post_cross_sync.py",
    "scripts/explicit_controller_phase35_crossing_basin_expansion.py",
    "simulator/phase34_35_transition.py",
)


class BranchStateExtractionError(RuntimeError):
    pass


class _BranchPointCaptured(RuntimeError):
    def __init__(self, document: dict[str, Any]):
        self.document = document
        super().__init__("first valid Final Veto branch point captured")


def _require_frozen_source_trajectory(repository_root: Path) -> None:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository_root.as_posix()}",
            "diff",
            "--quiet",
            SOURCE_COMMIT,
            "--",
            *SOURCE_TRAJECTORY_PATHS,
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return
    if completed.returncode == 1:
        raise BranchStateExtractionError(
            "source trajectory differs from the frozen source commit"
        )
    detail = completed.stderr.strip() or "Git source comparison failed"
    raise BranchStateExtractionError(detail)


def _finite_values(values: tuple[float, ...], label: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise BranchStateExtractionError(f"{label} contains a non-finite value")


def _build_case_configuration() -> dict[str, object]:
    return {
        "case_id": SOURCE_CASE_ID,
        "controller_id": "phase35_crossing_basin_expansion",
        "initial_velocity_angle_deg": SOURCE_ANGLE_DEG,
        "post_cross_mode": "radius_priority",
        "r0_over_target": SOURCE_R0_OVER_TARGET,
        "seed": SOURCE_SEED,
        "subset_id": SOURCE_SUBSET_ID,
        "thrust_scale": SOURCE_THRUST_SCALE,
        "upstream_variant": "radial_energy_push",
    }


def _build_document(
    context: PreTransitionActionContext,
    *,
    predicted_next_state: object,
    predicted_speed_ratio: float,
    monitor_decision: object,
    simulator_configuration: dict[str, object],
) -> dict[str, Any]:
    current = context.current_state
    predicted = predicted_next_state
    state_values = (current.x, current.y, current.vx, current.vy)
    predicted_values = (
        predicted.x,
        predicted.y,
        predicted.vx,
        predicted.vy,
    )
    action_values = (context.nominal_action[0], context.nominal_action[1])
    _finite_values(state_values, "current state")
    _finite_values(predicted_values, "predicted next state")
    _finite_values(action_values, "nominal action")
    _finite_values((predicted_speed_ratio,), "predicted speed ratio")

    case_configuration = _build_case_configuration()
    simulator_constants = simulator_configuration["simulator_constants"]
    document: dict[str, Any] = {
        "active_stage": context.active_stage,
        "branch_ordering": {
            "before_final_veto_fallback_execution": True,
            "before_nominal_action_execution": True,
            "capture_boundary": (
                "after_valid_monitor_evaluation_before_nominal_or_fallback_execution"
            ),
            "final_veto_fallback_executed": False,
            "monitor_evaluation_completed": True,
            "nominal_action_executed": False,
            "prior_monitor_decisions": "all_allow",
            "prior_valid_monitor_evaluation_count": context.step - 1,
            "prior_veto_count": 0,
            "realized_prefix_transition_count": context.step - 1,
        },
        "branch_step": context.step,
        "canonicalization": {
            "allow_nan": False,
            "canonicalization_id": CANONICALIZATION_ID,
            "encoding": "utf-8",
            "hash_algorithm": "sha256",
            "hash_field_excluded_from_input": "canonical_branch_state_hash",
            "json_separators": [",", ":"],
            "json_sort_keys": True,
        },
        "case_configuration": case_configuration,
        "case_configuration_hash": canonical_sha256(case_configuration),
        "case_id": SOURCE_CASE_ID,
        "comparator": HAZARD_COMPARATOR,
        "extraction_timestamp": EXTRACTION_TIMESTAMP,
        "extraction_timestamp_policy": EXTRACTION_TIMESTAMP_POLICY,
        "hazard_comparator": HAZARD_COMPARATOR,
        "hazard_threshold": HAZARD_THRESHOLD,
        "implementation_commit": SOURCE_COMMIT,
        "monitor_decision": {
            "decision": monitor_decision.decision,
            "monitor_id": monitor_decision.monitor_id,
            "reason": monitor_decision.reason,
            "veto_applied": monitor_decision.veto_applied,
        },
        "nominal_action": list(action_values),
        "nominal_proposed_action": list(action_values),
        "phase": context.phase,
        "position": [current.x, current.y],
        "predicted_next_state": {
            "position_x": predicted.x,
            "position_y": predicted.y,
            "velocity_x": predicted.vx,
            "velocity_y": predicted.vy,
        },
        "predicted_nominal_next_state": list(predicted_values),
        "predicted_nominal_speed_ratio": predicted_speed_ratio,
        "predicted_speed_ratio": predicted_speed_ratio,
        "schema_version": SCHEMA_VERSION,
        "seed": SOURCE_SEED,
        "simulator_configuration": simulator_configuration,
        "simulator_configuration_hash": canonical_sha256(simulator_configuration),
        "simulator_constants_hash": canonical_sha256(simulator_constants),
        "source_commit": SOURCE_COMMIT,
        "state": {
            "current_phase": context.phase,
            "position_x": current.x,
            "position_y": current.y,
            "velocity_x": current.vx,
            "velocity_y": current.vy,
        },
        "state_vector": list(state_values),
        "step": context.step,
        "subset_id": SOURCE_SUBSET_ID,
        "threshold": HAZARD_THRESHOLD,
        "velocity": [current.vx, current.vy],
    }
    return attach_canonical_branch_state_hash(document)


class _FirstVetoBranchPointHook:
    def __init__(self, simulator_configuration: dict[str, object]):
        self.simulator_configuration = simulator_configuration
        self.valid_evaluation_count = 0

    def __call__(
        self, context: PreTransitionActionContext
    ) -> ActionInterceptionResult:
        if context.case.case_id != SOURCE_CASE_ID:
            raise BranchStateExtractionError("rollout hook received the wrong source case")
        nominal_prediction: OneStepPrediction | None = None

        def predictor(state, action):
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

        try:
            decision = evaluate_overspeed_veto(
                context.current_state,
                context.nominal_action,
                predictor,
                threshold=HAZARD_THRESHOLD,
            )
        except MonitorEvaluationError as exc:
            raise BranchStateExtractionError(
                f"monitor evaluation was invalid at step {context.step}"
            ) from exc

        self.valid_evaluation_count += 1
        if self.valid_evaluation_count != context.step:
            raise BranchStateExtractionError(
                "monitor evaluation sequence no longer matches realized prefix steps"
            )
        if decision.decision == "allow":
            return ActionInterceptionResult(
                nominal_action=context.nominal_action,
                executed_action=context.nominal_action,
                intervention_applied=False,
                decision_metadata=decision,
            )
        if decision.decision != "veto":
            raise BranchStateExtractionError(
                f"unexpected monitor decision {decision.decision!r}"
            )
        if nominal_prediction is None:
            raise BranchStateExtractionError("nominal prediction was not captured")
        if not nominal_prediction.speed_ratio > HAZARD_THRESHOLD:
            raise BranchStateExtractionError(
                "veto branch point does not satisfy strict speed_ratio > 1.90"
            )

        document = _build_document(
            context,
            predicted_next_state=nominal_prediction.next_state,
            predicted_speed_ratio=nominal_prediction.speed_ratio,
            monitor_decision=decision,
            simulator_configuration=self.simulator_configuration,
        )
        validate_branch_state_data(document)
        raise _BranchPointCaptured(document)


def extract_branch_state(repository_root: Path) -> dict[str, Any]:
    manifest_path = repository_root / MANIFEST_RELATIVE_PATH
    validate_manifest(
        manifest_path,
        repository_root=repository_root,
        require_future_outputs_absent=True,
    )
    _require_frozen_source_trajectory(repository_root)
    if (
        MONITOR_ID != "one_step_overspeed_veto_v0"
        or OVERSPEED_THRESHOLD != HAZARD_THRESHOLD
        or OVERSPEED_COMPARATOR != HAZARD_COMPARATOR
    ):
        raise BranchStateExtractionError(
            "current monitor constants do not match the frozen branch-state contract"
        )

    # The source rollout is intentionally imported only when extraction executes.
    from scripts import explicit_controller_phase35_crossing_basin_expansion as phase35
    from simulator.phase34_35_transition import (
        ACTION_COMPONENT_MAX,
        ACTION_COMPONENT_MIN,
        GRAVITY_DENOMINATOR_EPSILON,
    )

    variant = next(
        item for item in phase35.VARIANTS if item.name == "radial_energy_push"
    )
    mode = phase35.PHASE34_TERMINAL_MODE
    target_radius = phase35.DEFAULT_TARGET_RADIUS * phase35.TARGET_RADIUS_SCALE
    target_circular_speed = math.sqrt(phase35.MU / target_radius)
    simulator_constants: dict[str, object] = {
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
        "target_circular_speed": target_circular_speed,
        "target_radius": target_radius,
        "target_radius_scale": phase35.TARGET_RADIUS_SCALE,
        "transition_function": (
            "simulator.phase34_35_transition.step_phase34_35_transition"
        ),
    }
    simulator_configuration: dict[str, object] = {
        "simulator_constants": simulator_constants,
        "thrust_scale": SOURCE_THRUST_SCALE,
    }
    hook = _FirstVetoBranchPointHook(simulator_configuration)
    try:
        phase35.rollout_phase35_case(
            variant,
            mode,
            SOURCE_R0_OVER_TARGET,
            SOURCE_ANGLE_DEG,
            SOURCE_THRUST_SCALE,
            phase35.TARGET_RADIUS_SCALE,
            record_trajectory=False,
            case_id=SOURCE_CASE_ID,
            pre_transition_action_hook=hook,
        )
    except _BranchPointCaptured as captured:
        return captured.document
    raise BranchStateExtractionError(
        "source trajectory terminated without reaching the frozen branch point"
    )


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Extract the first pre-transition Final Veto branch point from the "
            "frozen Phase35 source case."
        )
    )


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    output_path = PROJECT_ROOT / ARTIFACT_RELATIVE_PATH
    if output_path.exists():
        print(f"FAIL refusing to overwrite existing branch-state artifact: {output_path}")
        return 1
    try:
        document = extract_branch_state(PROJECT_ROOT)
        write_canonical_branch_state(output_path, document)
    except Exception as exc:
        print(f"FAIL branch-state extraction failed: {exc}")
        return 1
    print(f"PASS branch state written: {output_path}")
    print(f"PASS branch step: {document['branch_step']}")
    print(f"PASS canonical hash: {document['canonical_branch_state_hash']}")
    print("PASS no nominal, fallback, or recovery action was executed at the branch point")
    return 0


if __name__ == "__main__":
    sys.exit(main())
