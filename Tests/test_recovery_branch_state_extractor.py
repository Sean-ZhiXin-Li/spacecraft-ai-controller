from __future__ import annotations

import copy
import inspect
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime_assurance.final_veto_runner_types import (
    RolloutCaseContext,
    PreTransitionActionContext,
)
from runtime_assurance.recovery_branch_state_extractor import (
    BranchStateExtractionError,
    PrefixExecutionResult,
    SourceCaseDefinition,
    _FixedPrefixHook,
    build_source_case_inventory,
    compare_prefix_results,
    execute_nominal_prefix,
    load_registry_config,
    select_registry_cases,
    source_inventory_document,
    validate_static_contract,
)
from runtime_assurance.recovery_branch_state_registry import (
    BRANCH_STEP,
    CONFIG_PATH,
    LEGACY_ARTIFACT_PATH,
    LEGACY_CASE_ID,
    PREFIX_TRANSITION_COUNT,
    canonical_json_bytes,
)
from scripts.generate_recovery_branch_state_registry_v0 import build_parser
from simulator.phase34_35_transition import (
    CartesianState2D,
    NormalizedAction2D,
    Phase3435DynamicsContext,
    step_phase34_35_transition,
)


ROOT = Path(__file__).resolve().parents[1]


def source_case(case_id: str, *, upstream: str | None = None) -> SourceCaseDefinition:
    return SourceCaseDefinition(
        case_id=case_id,
        subset_id="synthetic_subset",
        seed=0,
        r0_over_target=1.0,
        initial_velocity_angle_deg=150.0,
        thrust_scale=8000.0,
        controller_id=(
            "phase34_post_cross_sync"
            if upstream is None
            else "phase35_crossing_basin_expansion"
        ),
        post_cross_mode="radius_priority",
        upstream_variant=upstream,
        source_case_artifact="analysis/final_veto_ablation_v0/manifest.json",
        source_case_hash="1" * 64,
        source_configuration_hash="2" * 64,
        source_commit="3" * 40,
        nominal_prefix_transition_count=PREFIX_TRANSITION_COUNT,
        nominal_controller_hash="4" * 64,
        transition_implementation_hash="5" * 64,
        eligible_for_generation=True,
        ineligibility_reason=None,
    )


def result(
    case_id: str,
    predicted: float,
    tangential: float,
    *,
    payload: dict[str, object] | None = None,
) -> PrefixExecutionResult:
    document = payload or {
        "position_x": 1.0,
        "position_y": 2.0,
        "velocity_x": 3.0,
        "velocity_y": 4.0,
        "radius": 5.0,
        "speed": 5.0,
        "radial_velocity": 1.0,
        "tangential_velocity": 2.0,
        "radius_error_ratio": 0.1,
        "radial_velocity_ratio": 0.2,
        "tangential_velocity_error_ratio": tangential,
        "predicted_position_x": 1.1,
        "predicted_position_y": 2.1,
        "predicted_velocity_x": 3.1,
        "predicted_velocity_y": 4.1,
        "predicted_speed_ratio": predicted,
    }
    return PrefixExecutionResult(
        execution_id=f"discovery_{case_id}",
        case=source_case(case_id),
        execution_role="candidate_discovery",
        document_json=canonical_json_bytes(document).decode("utf-8"),
        actual_transition_count=PREFIX_TRANSITION_COUNT,
        branch_step=BRANCH_STEP,
        initial_state_hash="6" * 64,
        prefix_action_trace_hash="7" * 64,
        prefix_state_trace_hash="8" * 64,
        canonical_payload_hash=(case_id.encode().hex() + "0" * 64)[:64],
        predicted_speed_ratio=predicted,
        tangential_velocity_error_ratio=tangential,
    )


class RecoveryBranchStateExtractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.legacy_bytes = (ROOT / LEGACY_ARTIFACT_PATH).read_bytes()

    def test_config_freezes_prefix_and_has_no_state_override(self) -> None:
        config = load_registry_config(ROOT)
        self.assertEqual(config["nominal_prefix_transition_count"], 27)
        self.assertEqual(config["branch_step"], 28)
        serialized = (ROOT / CONFIG_PATH).read_text(encoding="utf-8")
        self.assertNotIn('"position"', serialized)
        self.assertNotIn('"velocity"', serialized)

    def test_source_inventory_enumerates_all_final_veto_cases(self) -> None:
        cases = build_source_case_inventory(ROOT)
        self.assertEqual(len(cases), 13)
        self.assertEqual(len({item.case_id for item in cases}), 13)
        self.assertTrue(all(item.source_case_hash for item in cases))
        self.assertTrue(all(item.source_configuration_hash for item in cases))
        self.assertTrue(all(item.eligible_for_generation for item in cases))
        self.assertIn(LEGACY_CASE_ID, {item.case_id for item in cases})

    def test_source_inventory_is_deterministic_and_case_id_alone_is_insufficient(self) -> None:
        first = source_inventory_document(ROOT)
        second = source_inventory_document(ROOT)
        self.assertEqual(first, second)
        for item in first["cases"]:
            self.assertTrue(item["initialization_available"])
            self.assertTrue(item["simulator_configuration_available"])
            self.assertTrue(item["controller_configuration_available"])
            self.assertTrue(item["configuration_hash"])

    def test_static_validation_executes_no_prefix(self) -> None:
        with mock.patch(
            "runtime_assurance.recovery_branch_state_extractor.execute_nominal_prefix"
        ) as execute:
            report = validate_static_contract(ROOT, require_output_absent=True)
        execute.assert_not_called()
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.source_case_count, 13)

    def test_fixed_prefix_hook_preserves_actions_until_step_28(self) -> None:
        case = source_case("synthetic_case")
        hook = _FixedPrefixHook(case)
        target = 7.5e12
        target_speed = math.sqrt(1.3275182699999999e20 / target)
        dynamics = Phase3435DynamicsContext(
            mu=1.3275182699999999e20,
            dt=100.0,
            mass=722.0,
            thrust_scale=8000.0,
        )
        state = CartesianState2D(0.0, target, -target_speed, 0.0)
        case_context = RolloutCaseContext(
            case_id=case.case_id,
            controller_id=case.controller_id,
            controller_family="explicit_controller",
            r0_over_target=1.0,
            initial_velocity_angle_deg=150.0,
            thrust_scale=8000.0,
            target_radius=target,
            target_circular_speed=target_speed,
            post_cross_mode="radius_priority",
        )
        for step in range(1, BRANCH_STEP):
            context = PreTransitionActionContext(
                step=step,
                phase="DESCENT",
                active_stage="burn_a",
                current_state=state,
                nominal_action=(0.0, 0.0),
                predict_transition=lambda current, action: step_phase34_35_transition(
                    current, NormalizedAction2D(*action), dynamics
                ),
                compute_speed_ratio=lambda current: math.hypot(current.vx, current.vy)
                / (target_speed + 1.0e-12),
                case=case_context,
            )
            interception = hook(context)
            self.assertEqual(interception.executed_action, context.nominal_action)
            state = context.predict_transition(state, context.nominal_action).next_state
        self.assertEqual(len(hook.actions), PREFIX_TRANSITION_COUNT)
        self.assertEqual(len(hook.states), PREFIX_TRANSITION_COUNT)

    def test_prefix_hook_captures_before_step_28_action(self) -> None:
        case = source_case("synthetic_case")
        hook = _FixedPrefixHook(case)
        hook.expected_step = BRANCH_STEP
        hook.states = [{"position_x": 1.0, "position_y": 0.0, "velocity_x": 0.0, "velocity_y": 1.0}] * PREFIX_TRANSITION_COUNT
        hook.actions = [[0.0, 0.0]] * PREFIX_TRANSITION_COUNT
        dynamics = Phase3435DynamicsContext(mu=1.0, dt=1.0, mass=1.0, thrust_scale=1.0)
        state = CartesianState2D(1.0, 0.0, 0.0, 1.0)
        context = PreTransitionActionContext(
            step=BRANCH_STEP,
            phase="DESCENT",
            active_stage="burn_a",
            current_state=state,
            nominal_action=(0.0, 0.0),
            predict_transition=lambda current, action: step_phase34_35_transition(
                current, NormalizedAction2D(*action), dynamics
            ),
            compute_speed_ratio=lambda current: 1.0,
            case=RolloutCaseContext(
                case_id=case.case_id,
                controller_id=case.controller_id,
                controller_family="explicit_controller",
                r0_over_target=1.0,
                initial_velocity_angle_deg=150.0,
                thrust_scale=1.0,
                target_radius=1.0,
                target_circular_speed=1.0,
                post_cross_mode="radius_priority",
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "prefix boundary"):
            hook(context)
        self.assertEqual(len(hook.actions), PREFIX_TRANSITION_COUNT)

    def test_execute_nominal_prefix_uses_existing_rollout_and_extracts_complete_state(self) -> None:
        case = source_case("synthetic_phase34")
        target = 7.5e12
        target_speed = math.sqrt(1.3275182699999999e20 / target)

        def fake_rollout(_mode: object, _r0: float, _angle: float, _thrust: float, **kwargs: object) -> None:
            hook = kwargs["pre_transition_action_hook"]
            state = CartesianState2D(0.0, target, -target_speed, 0.0)
            dynamics = Phase3435DynamicsContext(
                mu=1.3275182699999999e20,
                dt=100.0,
                mass=722.0,
                thrust_scale=8000.0,
            )
            case_context = RolloutCaseContext(
                case_id=case.case_id,
                controller_id=case.controller_id,
                controller_family="explicit_controller",
                r0_over_target=1.0,
                initial_velocity_angle_deg=150.0,
                thrust_scale=8000.0,
                target_radius=target,
                target_circular_speed=target_speed,
                post_cross_mode="radius_priority",
            )
            for step in range(1, BRANCH_STEP + 1):
                context = PreTransitionActionContext(
                    step=step,
                    phase="DESCENT",
                    active_stage="burn_a",
                    current_state=state,
                    nominal_action=(0.0, 0.0),
                    predict_transition=lambda current, action: step_phase34_35_transition(
                        current, NormalizedAction2D(*action), dynamics
                    ),
                    compute_speed_ratio=lambda current: math.hypot(current.vx, current.vy)
                    / (target_speed + 1.0e-12),
                    case=case_context,
                )
                interception = hook(context)
                state = context.predict_transition(state, interception.executed_action).next_state

        with mock.patch(
            "scripts.explicit_controller_phase34_post_cross_sync.rollout_phase34_case",
            side_effect=fake_rollout,
        ):
            extracted = execute_nominal_prefix(
                ROOT,
                case,
                execution_role="candidate_discovery",
                execution_id="synthetic_discovery",
                implementation_commit="a" * 40,
            )
        document = extracted.document()
        self.assertEqual(extracted.actual_transition_count, 27)
        self.assertEqual(extracted.branch_step, 28)
        self.assertEqual(document["state_origin"], "deterministic_nominal_prefix_execution")
        self.assertEqual(
            tuple(document[field] for field in ("position_x", "position_y", "velocity_x", "velocity_y")),
            tuple(document["state"][field] for field in ("position_x", "position_y", "velocity_x", "velocity_y")),
        )
        self.assertFalse(document["reconstructed_from_log"])
        self.assertFalse(document["manually_authored_state"])
        self.assertFalse(document["perturbed_from_existing_state"])

    def test_early_terminal_is_rejected(self) -> None:
        case = source_case("synthetic_phase34")
        with mock.patch(
            "scripts.explicit_controller_phase34_post_cross_sync.rollout_phase34_case",
            return_value={},
        ):
            with self.assertRaises(BranchStateExtractionError):
                execute_nominal_prefix(
                    ROOT,
                    case,
                    execution_role="candidate_discovery",
                    execution_id="early",
                    implementation_commit="a" * 40,
                )

    def test_case_selection_rules_and_exact_boundary(self) -> None:
        candidates = [
            result("below_far", 1.80, 0.1),
            result("below_exact", 1.90, 0.2),
            result("above_close", 1.9001, 0.3),
            result("above_far", 2.0, 0.4),
            result("tangential", 1.7, -2.0),
        ]
        selection = select_registry_cases(candidates)
        self.assertEqual(selection.member_a_case_id, LEGACY_CASE_ID)
        self.assertEqual(selection.member_b_case_id, "below_exact")
        self.assertEqual(selection.member_c_case_id, "above_close")
        self.assertEqual(selection.member_d_case_id, "tangential")
        self.assertEqual(len(set((selection.member_a_case_id, *selection.generated_case_ids))), 4)

    def test_case_selection_tie_break_is_deterministic(self) -> None:
        candidates = [
            result("b_case", 1.89, 0.1),
            result("a_case", 1.89, 0.2),
            result("above", 1.91, 0.3),
            result("tangent", 1.5, 2.0),
        ]
        self.assertEqual(select_registry_cases(candidates).member_b_case_id, "a_case")

    def test_missing_selection_category_aborts(self) -> None:
        with self.assertRaises(BranchStateExtractionError):
            select_registry_cases([result("below", 1.8, 0.1), result("other", 1.7, 0.2)])
        with self.assertRaises(BranchStateExtractionError):
            select_registry_cases([result("above", 2.0, 0.1), result("other", 2.1, 0.2)])

    def test_determinism_comparison_detects_every_mismatch_class(self) -> None:
        first = result("same", 1.8, 0.2)
        second = copy.deepcopy(first)
        second = PrefixExecutionResult(
            **{**{field: getattr(second, field) for field in second.__dataclass_fields__}, "execution_role": "selected_reproduction"}
        )
        self.assertEqual(compare_prefix_results(first, second)["determinism_status"], "passed")
        changed_document = second.document()
        changed_document["velocity_x"] = 99.0
        changed = PrefixExecutionResult(
            **{
                **{field: getattr(second, field) for field in second.__dataclass_fields__},
                "document_json": canonical_json_bytes(changed_document).decode("utf-8"),
                "canonical_payload_hash": "f" * 64,
            }
        )
        comparison = compare_prefix_results(first, changed)
        self.assertFalse(comparison["Cartesian_state_equal"])
        self.assertEqual(comparison["determinism_status"], "failed")

    def test_no_manual_state_or_prefix_override_api_exists(self) -> None:
        parameters = inspect.signature(execute_nominal_prefix).parameters
        for forbidden in ("state", "position", "velocity", "prefix", "branch_step"):
            self.assertNotIn(forbidden, parameters)
        options = {option for action in build_parser()._actions for option in action.option_strings}
        for forbidden in (
            "--case",
            "--seed",
            "--state",
            "--position",
            "--velocity",
            "--prefix",
            "--branch-step",
            "--threshold",
            "--output",
            "--retry",
            "--resume",
        ):
            self.assertNotIn(forbidden, options)

    def test_default_cli_executes_no_transition_and_writes_nothing(self) -> None:
        output_existed = (ROOT / "analysis/recovery_branch_state_registry_v0").exists()
        completed = subprocess.run(
            [sys.executable, "scripts/generate_recovery_branch_state_registry_v0.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--execute-frozen-registry-generation", completed.stdout)
        self.assertEqual((ROOT / "analysis/recovery_branch_state_registry_v0").exists(), output_existed)

    def test_legacy_and_protected_source_bytes_remain_unchanged(self) -> None:
        self.assertEqual((ROOT / LEGACY_ARTIFACT_PATH).read_bytes(), self.legacy_bytes)


if __name__ == "__main__":
    unittest.main()
